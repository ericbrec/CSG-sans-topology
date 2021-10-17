import numpy as np
import scipy.integrate as integrate
import manifold as mf

# TODO: Update ContainsPoint to use integral instead of ray cast to compute winding number.

class Boundary:

    def __init__(self, manifold, domain = None):
        if domain:
            assert manifold.GetDimension() - 1 == domain.dimension
        else:
            assert manifold.GetDimension() == 1

        self.manifold = manifold
        self.domain = domain
    
    @staticmethod
    def SortKey(boundary):
        if boundary.domain:
            return 0.0
        else:
            return (boundary.manifold.Point(0.0), -boundary.manifold.Normal(0.0))

class Solid:

    # If two points are within 0.01 of each eachother, they are coincident
    minSeparation = 0.01

    @staticmethod
    def CreateSolidFromPoints(dimension, points, isVoid = False):
        # CreateSolidFromPoints only works for dimension 2 so far.
        assert dimension == 2
        assert len(points) > 2
        assert len(points[0]) == dimension

        solid = Solid(dimension, isVoid)

        previousPoint = np.array(points[len(points)-1])
        for point in points:
            point = np.array(point)
            vector = point - previousPoint
            normal = np.array([-vector[1], vector[0]])
            normal = normal / np.linalg.norm(normal)
            hyperplane = mf.Hyperplane.CreateFromNormal(normal,np.dot(normal,point))
            domain = Solid(dimension-1)
            previousPointDomain = hyperplane.DomainFromPoint(previousPoint)
            pointDomain = hyperplane.DomainFromPoint(point)
            if previousPointDomain < pointDomain:
                domain.boundaries.append(Boundary(mf.Hyperplane.CreateFromNormal(-1.0, -previousPointDomain)))
                domain.boundaries.append(Boundary(mf.Hyperplane.CreateFromNormal(1.0, pointDomain)))
            else:
                domain.boundaries.append(Boundary(mf.Hyperplane.CreateFromNormal(-1.0, -pointDomain)))
                domain.boundaries.append(Boundary(mf.Hyperplane.CreateFromNormal(1.0, previousPointDomain)))
            solid.boundaries.append(Boundary(hyperplane, domain))
            previousPoint = point

        return solid

    def __init__(self, dimension, isVoid = False):
        assert dimension > 0
        self.dimension = dimension
        self.isVoid = isVoid
        self.boundaries = []

    def Not(self):
        solid = Solid(self.dimension, not self.isVoid)
        for boundary in self.boundaries:
            solid.boundaries.append(Boundary(boundary.manifold.Flip(),boundary.domain))
        return solid
    
    def Translate(self, delta):
        assert len(delta) == self.dimension

        for boundary in self.boundaries:
            boundary.manifold.Translate(delta)
    
    def Points(self):
        for boundary in self.boundaries:
            if boundary.domain:
                for domainPoint in boundary.domain.Points():
                    point = boundary.manifold.Point(domainPoint)
                    yield point
            else:
                yield boundary.manifold.Point(0.0)
    
    def Edges(self):
        if self.dimension > 1:
            for boundary in self.boundaries:
                for domainEdge in boundary.domain.Edges():
                    yield [boundary.manifold.Point(domainEdge[0]), boundary.manifold.Point(domainEdge[1]), boundary.manifold.Normal(domainEdge[0])]
        else:
            self.boundaries.sort(key=Boundary.SortKey)
            leftB = 0
            rightB = 0
            while leftB < len(self.boundaries):
                if self.boundaries[leftB].manifold.Normal(0.0) < 0.0:
                    leftPoint = self.boundaries[leftB].manifold.Point(0.0)
                    while rightB < len(self.boundaries):
                        rightPoint = self.boundaries[rightB].manifold.Point(0.0)
                        if leftPoint - Solid.minSeparation < rightPoint and self.boundaries[rightB].manifold.Normal(0.0) > 0.0:
                            yield [leftPoint, rightPoint]
                            rightB += 1
                            break
                        rightB += 1
                leftB += 1
 
    def VolumeIntegral(self, f, args=(), epsabs=None, epsrel=None, *quadArgs):
        if not isinstance(args, tuple):
            args = (args,)
        if epsabs is None:
            epsabs = Solid.minSeparation
        if epsrel is None:
            epsrel = Solid.minSeparation

        # Initialize the return value for the integral
        sum = 0.0

        if len(self.boundaries) > 0:
            # Select the first coordinate of an arbitrary point within the volume boundary (the domain of f)
            x0 = next(self.Points())[0]

            for boundary in self.boundaries:
                # domainF is the integrand you get by applying the divergence theorem: VolumeIntegral(divergence(F)) = SurfaceIntegral(dot(F, n)).
                # Let F = [Integral(f) from x0 to x holding other coordinates fixed, 0...0]. divergence(F) = f by construction, and dot(F, n) = Integral(f) * n[0].
                # Note that the choice of x0 is arbitrary as long as it's in the domain of f and doesn't change across all surface integral boundaries.
                # Thus, we have VolumeIntegral(f) = SurfaceIntegral(Integral(f) * n[0]).
                # The surface normal, n, is the cross product of the boundary manifold's tangent space divided by its length.
                # The surface differential, dS, is the length of cross product of the boundary manifold's tangent space times the differentials of the manifold's domain variables.
                # The length of the cross product appears in the numerator and denominator of the SurfaceIntegral and cancels.
                # What's left multiplying Integral(f) is the first coordinate of the cross product plus the domain differentials (volume integral).
                # The first coordinate of the cross product of the boundary manifold's tangent space is the first cofactor of the tangent space.
                # And so, SurfaceIntegral(dot(F, n)) = VolumeIntegral(Integral(f) * first cofactor) over the boundary manifold's domain.
                def domainF(domainPoint):
                    if np.isscalar(domainPoint):
                        point = boundary.manifold.Point(np.array([domainPoint]))
                    else:
                        point = boundary.manifold.Point(domainPoint)

                    # fHat passes the scalar given by integrate.quad into the first coordinate of the vector for f. 
                    def fHat(x):
                        evalPoint = np.array(point)
                        evalPoint[0] = x
                        return f(evalPoint, *args)

                    # Calculate Integral(f) * first cofactor. Note that quad returns a tuple: (integral, error bound).
                    returnValue = 0.0
                    firstCofactor = boundary.manifold.FirstCofactor(domainPoint)
                    if abs(x0 - point[0]) > epsabs and abs(firstCofactor) > epsabs:
                        returnValue = integrate.quad(fHat, x0, point[0], epsabs=epsabs, epsrel=epsrel, *quadArgs)[0] * firstCofactor
                    return returnValue

                if boundary.domain:
                    # Add the contribution to the Volume integral from this boundary.
                    sum += boundary.domain.VolumeIntegral(domainF)
                else:
                    # This is a 1-D boundary (line interval, no domain), so just add the integrand. 
                    sum += domainF(0.0)

        return sum

    def SurfaceIntegral(self, f, args=(), epsabs=None, epsrel=None, *quadArgs):
        if not isinstance(args, tuple):
            args = (args,)
        if epsabs is None:
            epsabs = Solid.minSeparation
        if epsrel is None:
            epsrel = Solid.minSeparation

        # Initialize the return value for the integral
        sum = 0.0

        if len(self.boundaries) > 0:
            # Select an arbitrary point within the volume boundary (the domain of f) to determine the dimension of f.
            point = next(self.Points())
            fValue = f(point, *args)
            fIsVector = False
            if not np.isscalar(fValue):
                assert len(fValue) == self.dimension
                fIsVector = True
            
            def integrand(domainPoint):
                if np.isscalar(domainPoint):
                    point = boundary.manifold.Point(np.array([domainPoint]))
                else:
                    point = boundary.manifold.Point(domainPoint)
                fValue = f(point, *args)
                returnValue = boundary.manifold.Determinant(domainPoint)
                if fIsVector:
                    returnValue = np.dot(fValue, boundary.manifold.CofactorNormal(domainPoint))
                else:
                    returnValue = fValue * np.dot(boundary.manifold.Normal(domainPoint), boundary.manifold.CofactorNormal(domainPoint))
                return returnValue

            for boundary in self.boundaries:
                if boundary.domain:
                    # Add the contribution to the Volume integral from this boundary.
                    sum += boundary.domain.VolumeIntegral(integrand)
                else:
                    # This is a 1-D boundary (line interval, no domain), so just add the integrand. 
                    sum += integrand(0.0)

        return sum

    def ContainsPoint(self, point):
        # Return value is -1 for interior, 0 for on boundary, and 1 for exterior
        containment = -1
        onBoundary = False

        # The winding number is 0 if the point is outside the solid, 1 if it's inside.
        # Other values indicate issues: 
        #  * Incomplete boundaries lead to fractional values;
        #  * Interior-pointing normals lead to negative values;
        #  * Nested shells lead to absolute values of 2 or greater.
        windingNumber = 0
        if self.isVoid:
            # If the solid is a void, then the winding number starts as 1 to account for the boundary at infinity.
            windingNumber = 1

        # Intersect a ray from point along the x-axis through self's boundaries.
        # All dot products with the ray are just the first component, since the ray is along the x-axis.
        for boundary in self.boundaries:
            intersections = boundary.manifold.IntersectXRay(point)
            for intersection in intersections:
                # Each intersection is of the form [distance to intersection, domain point of intersection].
                # First, check the distance is positive.
                if intersection[0] > -Solid.minSeparation:
                    considerBoundary = True
                    if boundary.domain:
                        # Only include the boundary if the ray intersection is inside its domain.
                        considerBoundary = boundary.domain.ContainsPoint(intersection[1]) < 1
                    # If we've got a valid boundary intersection, accumulate winding number based on sign of dot(ray,normal) == normal[0].
                    if considerBoundary:
                        if intersection[0] < Solid.minSeparation:
                            onBoundary = True
                        windingNumber += np.sign(boundary.manifold.Normal(intersection[1])[0])
        
        if onBoundary:
            containment = 0
        elif windingNumber == 0:
            containment = 1
        return containment 

    def ContainsBoundary(self, boundary):
        containment = False
        if boundary.domain:
            # If boundary has a domain, loop through the boundary points of its domain until one is clearly inside or outside.
            for domainPoint in boundary.domain.Points():
                constainsPoint = self.ContainsPoint(boundary.manifold.Point(domainPoint))
                if constainsPoint != 0:
                    containment = constainsPoint < 1
                    break
        else:
            # Otherwise, the boundary is a single point, so just determine if it's inside or outside.
            containment = self.ContainsPoint(boundary.manifold.Point(0.0)) < 1

        return containment

    def Slice(self, manifold, cache = None):
        assert manifold.GetDimension() == self.dimension

        manifoldDomain = None

        # Only manifolds of dimension > 1 have a domain.
        if self.dimension > 1:
            manifoldDomain = Solid(self.dimension-1, self.isVoid)

            # Intersect each of this solid's boundaries with the manifold.
            for boundary in self.boundaries:
                # Start by intersecting the boundary's manifold with the given manifold.
                # IntersectManifold returns a collection of manifold pairs:
                #   * intersection[0] is in the boundary's domain;
                #   * intersection[1] is in the given manifold's domain.
                # Both intersections correspond to the same range (the intersection between the manifolds).
                intersections = boundary.manifold.IntersectManifold(manifold, cache)

                # For each intersection, slice the boundary domain with the intersection manifold.
                # We slice the boundary domain using intersection[0], but we add intersection[1] to the manifold domain. 
                for intersection in intersections:
                    if boundary.domain.dimension > 1:
                        intersectionSlice = boundary.domain.Slice(intersection[0], cache)
                        if intersectionSlice:
                            manifoldDomain.boundaries.append(Boundary(intersection[1],intersectionSlice))
                    else:
                        # This domain is dimension 1, a real number line, and the intersection is a point on that line.
                        # Determine if the intersection point is within domain's interior.
                        if boundary.domain.ContainsPoint(intersection[0].Point(0.0)) < 1:
                            manifoldDomain.boundaries.append(Boundary(intersection[1]))
            
            # Don't return a manifold domain if it's empty
            if len(manifoldDomain.boundaries) == 0:
                manifoldDomain = None
        
        return manifoldDomain

    def Intersection(self, solid, cache = None):
        assert self.dimension == solid.dimension

        # Manifold intersections are expensive and come in symmetric pairs (m1 intersect m2, m2 intersect m1).
        # So, we create a manifold intersections cache (dictionary) to store and reuse intersection pairs.
        if cache == None:
            cache = {}

        combinedSolid = Solid(self.dimension, self.isVoid and solid.isVoid)

        for boundary in self.boundaries:
            # Slice self boundary manifold by solid. If it intersects, intersect the domains.
            slice = solid.Slice(boundary.manifold, cache)
            if slice:
                newDomain = boundary.domain.Intersection(slice, cache)
                if len(newDomain.boundaries) > 0:
                    combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif solid.ContainsBoundary(boundary):
                combinedSolid.boundaries.append(boundary)

        for boundary in solid.boundaries:
            # Slice solid boundary manifold by self. If it intersects, intersect the domains.
            slice = self.Slice(boundary.manifold, cache)
            if slice:
                newDomain = boundary.domain.Intersection(slice, cache)
                if len(newDomain.boundaries) > 0:
                    combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif self.ContainsBoundary(boundary):
                combinedSolid.boundaries.append(boundary)

        return combinedSolid

    def Union(self, solid):
        return self.Not().Intersection(solid.Not()).Not()

    def Difference(self, solid):
        return self.Intersection(solid.Not())