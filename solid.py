import numpy as np
import scipy.integrate as integrate
import manifold as mf

class Boundary:
    
    @staticmethod
    def SortKey(boundary):
        if boundary.domain:
            return 0.0
        else:
            return (boundary.manifold.Point(0.0), -boundary.manifold.Normal(0.0))

    def __init__(self, manifold, domain = None):
        if domain:
            assert manifold.GetDomainDimension() == domain.dimension
        else:
            assert manifold.GetDomainDimension() == 0

        self.manifold = manifold
        self.domain = domain

class Solid:

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
    
    def AnyPointAndNormal(self):
        point = None
        normal = None
        if len(self.boundaries) > 0:
            boundary = self.boundaries[0]
            if boundary.domain:
                domainPoint, domainNormal = boundary.domain.AnyPointAndNormal()
            else:
                domainPoint = 0.0
            point = boundary.manifold.Point(domainPoint)
            normal = boundary.manifold.Normal(domainPoint)

        return point, normal

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
                        if leftPoint - mf.Manifold.minSeparation < rightPoint and self.boundaries[rightB].manifold.Normal(0.0) > 0.0:
                            yield [leftPoint, rightPoint]
                            rightB += 1
                            break
                        rightB += 1
                leftB += 1
 
    def VolumeIntegral(self, f, args=(), epsabs=None, epsrel=None, *quadArgs):
        if not isinstance(args, tuple):
            args = (args,)
        if epsabs is None:
            epsabs = mf.Manifold.minSeparation
        if epsrel is None:
            epsrel = mf.Manifold.minSeparation

        # Initialize the return value for the integral
        sum = 0.0

        if len(self.boundaries) > 0:
            # Select the first coordinate of an arbitrary point within the volume boundary (the domain of f)
            x0 = (self.AnyPointAndNormal())[0][0]

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
            epsabs = mf.Manifold.minSeparation
        if epsrel is None:
            epsrel = mf.Manifold.minSeparation

        # Initialize the return value for the integral
        sum = 0.0

        for boundary in self.boundaries:
            def integrand(domainPoint):
                if np.isscalar(domainPoint):
                    evalPoint = np.array([domainPoint])
                else:
                    evalPoint = domainPoint
                point = boundary.manifold.Point(evalPoint)
                normal = boundary.manifold.Normal(evalPoint)
                fValue = f(point, normal, *args)
                return np.dot(fValue, boundary.manifold.CofactorNormal(domainPoint))

            if boundary.domain:
                # Add the contribution to the Volume integral from this boundary.
                sum += boundary.domain.VolumeIntegral(integrand)
            else:
                # This is a 1-D boundary (line interval, no domain), so just add the integrand. 
                sum += integrand(0.0)

        return sum

    def WindingNumber(self, point):
        # Return value is a tuple: winding number, onBoundaryNormal
        # The winding number is 0 if the point is outside the solid, 1 if it's inside.
        #   Other values indicate issues: 
        #   * Incomplete boundaries lead to fractional values;
        #   * Interior-pointing normals lead to negative values;
        #   * Nested shells lead to absolute values of 2 or greater.
        # OnBoundaryNormal is None or the boundary normal if the point lies on a boundary. 
        windingNumber = 0.0
        onBoundaryNormal = None
        if self.isVoid:
            # If the solid is a void, then the winding number starts as 1 to account for the boundary at infinity.
            windingNumber = 1.0

        if True:
            # Intersect a ray from point along the x-axis through self's boundaries.
            # All dot products with the ray are just the first component, since the ray is along the x-axis.
            for boundary in self.boundaries:
                intersections = boundary.manifold.IntersectXRay(point)
                for intersection in intersections:
                    # Each intersection is of the form [distance to intersection, domain point of intersection].
                    # First, check the distance is positive.
                    if intersection[0] > -mf.Manifold.minSeparation:
                        considerBoundary = True
                        if boundary.domain:
                            # Only include the boundary if the ray intersection is inside its domain.
                            considerBoundary = boundary.domain.ContainsPoint(intersection[1])
                        # If we've got a valid boundary intersection, accumulate winding number based on sign of dot(ray,normal) == normal[0].
                        if considerBoundary:
                            if intersection[0] < mf.Manifold.minSeparation:
                                onBoundaryNormal = boundary.manifold.Normal(intersection[1])
                            windingNumber += np.sign(boundary.manifold.Normal(intersection[1])[0])
        else:
            nSphereArea = 2.0
            if self.dimension % 2 == 0:
                nSphereArea *= np.pi
            dimension = self.dimension
            while dimension > 2:
                nSphereArea *= 2.0 * np.pi / (dimension - 2.0)
                dimension -= 2
                
            def windingIntegrand(boundaryPoint, boundaryNormal, onBoundaryNormalList):
                vector = boundaryPoint - point
                vectorLength = np.linalg.norm(vector)
                if np.abs(vectorLength) < mf.Manifold.minSeparation:
                    onBoundaryNormalList[0] = boundaryNormal
                    vectorLength = 1.0
                return vector / (vectorLength**self.dimension)

            onBoundaryNormalList = [onBoundaryNormal]
            windingNumber += self.SurfaceIntegral(windingIntegrand, onBoundaryNormalList) / nSphereArea
            onBoundaryNormal = onBoundaryNormalList[0]

        return windingNumber, onBoundaryNormal 

    def ContainsPoint(self, point):
        windingNumber, onBoundaryNormal = self.WindingNumber(point)
        containment = True
        if onBoundaryNormal is None:
            containment = windingNumber > 0.5
        return containment 

    def ContainsBoundary(self, boundary):
        containment = False
        if boundary.domain:
            domainPoint, domainNormal = boundary.domain.AnyPointAndNormal()
        else:
            domainPoint = 0.0
        windingNumber, onBoundaryNormal = self.WindingNumber(boundary.manifold.Point(domainPoint))
        if onBoundaryNormal is None:
            containment = windingNumber > 0.5
        else:
            # Boundaries are coincident at the point, so containment is based on alignment of normals.
            containment = np.dot(onBoundaryNormal, boundary.manifold.Normal(domainPoint)) > 0.0
        return containment

    def Slice(self, manifold, cache = None):
        assert manifold.GetRangeDimension() == self.dimension

        manifoldDomain = None

        # Only manifolds of dimension > 1 have a domain.
        if self.dimension > 1:
            manifoldDomain = Solid(self.dimension-1, self.isVoid)

            # Intersect each of this solid's boundaries with the manifold.
            for boundary in self.boundaries:
                # Start by intersecting the boundary's manifold with the given manifold.
                # IntersectManifold returns a collection of manifold pairs (or an alignment value if coincident):
                #   * intersection[0] is in the boundary's domain;
                #   * intersection[1] is in the given manifold's domain.
                # Both intersections correspond to the same range (the intersection between the manifolds).
                intersections = boundary.manifold.IntersectManifold(manifold, cache)

                # For each intersection, slice the boundary domain with the intersection manifold.
                # We slice the boundary domain using intersection[0], but we add intersection[1] to the manifold domain. 
                for intersection in intersections:
                    if isinstance(intersection, list):
                        if boundary.domain.dimension > 1:
                            intersectionSlice = boundary.domain.Slice(intersection[0], cache)
                            if intersectionSlice:
                                manifoldDomain.boundaries.append(Boundary(intersection[1],intersectionSlice))
                        else:
                            # This domain is dimension 1, a real number line.
                            # The intersection is a boundary point on that line (a point and normal).
                            # If the boundary point is within the domain, add its twin to the manifoldDomain.
                            if boundary.domain.ContainsBoundary(Boundary(intersection[0])):
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
            newDomain = None
            if slice:
                newDomain = boundary.domain.Intersection(slice, cache)
                if len(newDomain.boundaries) == 0:
                    newDomain = None
            if newDomain:
                combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif solid.ContainsBoundary(boundary):
                combinedSolid.boundaries.append(boundary)

        for boundary in solid.boundaries:
            # Slice solid boundary manifold by self. If it intersects, intersect the domains.
            slice = self.Slice(boundary.manifold, cache)
            newDomain = None
            if slice:
                newDomain = boundary.domain.Intersection(slice, cache)
                if len(newDomain.boundaries) == 0:
                    newDomain = None
            if newDomain:
                combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif self.ContainsBoundary(boundary):
                combinedSolid.boundaries.append(boundary)

        return combinedSolid

    def Union(self, solid):
        return self.Not().Intersection(solid.Not()).Not()

    def Difference(self, solid):
        return self.Intersection(solid.Not())