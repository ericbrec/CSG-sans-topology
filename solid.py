import numpy as np
import scipy.integrate as integrate
import manifold as mf

class Boundary:
    
    @staticmethod
    def SortKey(boundary):
        if boundary.domain is None:
            return (boundary.manifold.Point(0.0), -boundary.manifold.Normal(0.0))
        else:
            return 0.0

    def __init__(self, manifold, domain = None):
        if domain is None:
            assert manifold.GetDomainDimension() == 0
        else:
            assert manifold.GetDomainDimension() == domain.dimension

        self.manifold = manifold
        self.domain = domain

class Solid:

    def __init__(self, dimension, containsInfinity = False):
        assert dimension > 0
        self.dimension = dimension
        self.containsInfinity = containsInfinity
        self.boundaries = []
    
    def __bool__(self):
        return self.containsInfinity or len(self.boundaries) > 0

    def IsEmpty(self):
        return not self

    def Not(self):
        solid = Solid(self.dimension, not self.containsInfinity)
        for boundary in self.boundaries:
            manifold = boundary.manifold.copy()
            manifold.FlipNormal()
            solid.boundaries.append(Boundary(manifold,boundary.domain))
        return solid

    def __neg__(self):
        return self.Not()

    def Transform(self, transform):
        assert np.shape(transform) == (self.dimension, self.dimension)

        for boundary in self.boundaries:
            boundary.manifold.Transform(transform)
    
    def Translate(self, delta):
        assert len(delta) == self.dimension

        for boundary in self.boundaries:
            boundary.manifold.Translate(delta)
    
    def AnyPoint(self):
        point = None
        if len(self.boundaries) > 0:
            boundary = self.boundaries[0]
            if boundary.domain is None:
                domainPoint = 0.0
            else:
                domainPoint = boundary.domain.AnyPoint()
            point = boundary.manifold.Point(domainPoint)
        elif self.containsInfinity:
            point = np.full((self.dimension), 0.0)

        return point

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
                            leftB = rightB
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
            x0 = self.AnyPoint()[0]

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
                    evalPoint = np.atleast_1d(domainPoint)
                    point = boundary.manifold.Point(evalPoint)

                    # fHat passes the scalar given by integrate.quad into the first coordinate of the vector for f. 
                    def fHat(x):
                        evalPoint = np.array(point)
                        evalPoint[0] = x
                        return f(evalPoint, *args)

                    # Calculate Integral(f) * first cofactor. Note that quad returns a tuple: (integral, error bound).
                    returnValue = 0.0
                    firstCofactor = boundary.manifold.FirstCofactor(evalPoint)
                    if abs(x0 - point[0]) > epsabs and abs(firstCofactor) > epsabs:
                        returnValue = integrate.quad(fHat, x0, point[0], epsabs=epsabs, epsrel=epsrel, *quadArgs)[0] * firstCofactor
                    return returnValue

                if boundary.domain is None:
                    # This is a 1-D boundary (line interval, no domain), so just add the integrand. 
                    sum += domainF(0.0)
                else:
                    # Add the contribution to the Volume integral from this boundary.
                    sum += boundary.domain.VolumeIntegral(domainF)

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
                evalPoint = np.atleast_1d(domainPoint)
                point = boundary.manifold.Point(evalPoint)
                normal = boundary.manifold.Normal(evalPoint)
                fValue = f(point, normal, *args)
                return np.dot(fValue, boundary.manifold.CofactorNormal(domainPoint))

            if boundary.domain is None:
                # This is a 1-D boundary (line interval, no domain), so just add the integrand. 
                sum += integrand(0.0)
            else:
                # Add the contribution to the Volume integral from this boundary.
                sum += boundary.domain.VolumeIntegral(integrand)

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
        if self.containsInfinity:
            # If the solid contains infinity, then the winding number starts as 1 to account for the boundary at infinity.
            windingNumber = 1.0

        if self.dimension == 1:
            # Fast winding number calculation for a number line specialized to catch boundary edges.
            for boundary in self.boundaries:
                normal = boundary.manifold.Normal(0.0)
                separation = np.dot(normal, boundary.manifold.Point(0.0) - point)
                if -2.0 * mf.Manifold.minSeparation < separation < mf.Manifold.minSeparation:
                    onBoundaryNormal = normal
                    break
                else:
                    windingNumber += np.sign(separation) / 2.0
        elif True:
            # Intersect a ray from point along the x-axis through self's boundaries.
            # All dot products with the ray are just the first component, since the ray is along the x-axis.
            for boundary in self.boundaries:
                intersections = boundary.manifold.IntersectXRay(point)
                for intersection in intersections:
                    # Each intersection is of the form [distance to intersection, domain point of intersection].
                    # First, check the distance is positive.
                    if intersection[0] > -mf.Manifold.minSeparation:
                        considerBoundary = True
                        if boundary.domain is not None:
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
                if vectorLength < mf.Manifold.minSeparation:
                    onBoundaryNormalList[0] = boundaryNormal
                    vectorLength = 1.0
                return vector / (vectorLength**self.dimension)

            onBoundaryNormalList = [onBoundaryNormal]
            windingNumber += self.SurfaceIntegral(windingIntegrand, onBoundaryNormalList) / nSphereArea
            onBoundaryNormal = onBoundaryNormalList[0]

        return windingNumber, onBoundaryNormal 

    def ContainsPoint(self, point):
        windingNumber, onBoundaryNormal = self.WindingNumber(point)
        # The default is to include points on the boundary (onBoundaryNormal is not None).
        containment = True
        if onBoundaryNormal is None:
            containment = windingNumber > 0.5
        return containment 

    def ContainsBoundary(self, boundary):
        containment = False
        if boundary.domain is None:
            domainPoint = 0.0
        else:
            domainPoint = boundary.domain.AnyPoint()
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
        coincidences = []

        # Only manifolds of dimension > 1 have a domain.
        if self.dimension > 1:
            # Start with empty slice.
            manifoldDomain = Solid(self.dimension-1, self.containsInfinity)

            # Intersect each of this solid's boundaries with the manifold.
            for boundary in self.boundaries:
                
                # Check manifold intersections cache for twin.
                intersections = None
                twin = False
                if cache != None:
                    intersections = cache.get((boundary.manifold, manifold))
                if intersections is None:
                    # Start by intersecting the boundary's manifold with the given manifold.
                    intersections = boundary.manifold.IntersectManifold(manifold)
                    # Store intersections in cache
                    if cache != None:
                        cache[(manifold,boundary.manifold)] = intersections
                    b = 0 # Index of intersection in the boundary's domain
                    m = 1 # Index of intersection in the manifold's domain   
                else:
                    twin = True
                    b = 1 # Index of intersection in the boundary's domain
                    m = 0 # Index of intersection in the manifold's domain   

                # Each intersection is either a crossing (a domain manifold) or a coincident area (a solid with the domain).
                for intersection in intersections:
                    if isinstance(intersection[b], mf.Manifold):
                        # IntersectManifold found a crossing, returned as a manifold pair:
                        #   * intersection[b] is the intersection manifold in the boundary's domain;
                        #   * intersection[m] is the intersection manifold in the given manifold's domain.
                        #   * Both intersection manifolds have the same range (the crossing between the boundary and the given manifold).
                        if boundary.domain.dimension > 1:
                            # We slice the boundary domain using intersection[b], but we add intersection[m] to the manifold domain.
                            # Both intersection manifolds have the same domain by construction in IntersectManifold.
                            intersectionSlice = boundary.domain.Slice(intersection[b], cache)
                            if intersectionSlice:
                                manifoldDomain.boundaries.append(Boundary(intersection[m],intersectionSlice))
                        else:
                            # This domain is dimension 1, a real number line.
                            # The intersection is a boundary point on that line (a point and normal).
                            # If the boundary point is within the domain, add its twin to the manifoldDomain.
                            if boundary.domain.ContainsBoundary(Boundary(intersection[b])):
                                manifoldDomain.boundaries.append(Boundary(intersection[m]))
                    
                    elif isinstance(intersection[b], Solid):
                        # IntersectManifold found a coincident area, returned as:
                        #   * intersection[b] is solid within the boundary's domain inside of which the boundary and given manifold are coincident.
                        #   * intersection[m] is solid within the manifold's domain inside of which the boundary and given manifold are coincident.
                        #   * intersection[2] is the normal alignment between the boundary and given manifold (same or opposite directions).
                        #   * intersection[3] is transform from the boundary's domain to the given manifold's domain.
                        #   * intersection[4] is the translation from the boundary's domain to the given manifold's domain.
                        #   * Together intersection[3] and intersection[4] form the mapping from the boundary's domain to the given manifold's domain.

                        # First, intersect domain coincidence with the domain boundary.
                        domainCoincidence = intersection[b].Intersection(boundary.domain)
                        # Next, transform the domain coincidence from the boundary to the given manifold.
                        # Create copies of the manifolds and boundaries, since we are changing them.
                        for i in range(len(domainCoincidence.boundaries)):
                            domainManifold = domainCoincidence.boundaries[i].manifold.copy()
                            domainManifold.Transform(intersection[3])
                            domainManifold.Translate(intersection[4])
                            domainCoincidence.boundaries[i] = Boundary(domainManifold, domainCoincidence.boundaries[i].domain)
                        # Finally, add the domain coincidence to the list of coincidences.
                        coincidences.append(domainCoincidence)
            
            # Toss out a slice without any intersections.
            if manifoldDomain.IsEmpty():
                manifoldDomain = None
        
        return manifoldDomain, coincidences

    def Intersection(self, solid, cache = None):
        assert self.dimension == solid.dimension

        # Manifold intersections are expensive and come in symmetric pairs (m1 intersect m2, m2 intersect m1).
        # So, we create a manifold intersections cache (dictionary) to store and reuse intersection pairs.
        if cache == None:
            cache = {}

        # Start with empty solid.
        combinedSolid = Solid(self.dimension, self.containsInfinity and solid.containsInfinity)

        for boundary in self.boundaries:
            # TODO: Track intersections.
            newDomain = None
            # Slice self boundary manifold by solid. If it intersects, intersect the domains.
            slice, coincidences = solid.Slice(boundary.manifold, cache)
            if slice:
                newDomain = boundary.domain.Intersection(slice, cache)
            elif coincidences:
                newDomain = boundary.domain
            # Subtract domain coincidences from the newDomain so they only appear once. Below we will intersect them.
            if newDomain:
                for domainCoincidence in coincidences:
                    # TODO: Test for intersection. If you get one, then do the difference. 
                    newDomain = newDomain.Difference(domainCoincidence)
            # TODO: Should add test for intersection.
            if newDomain:
                # Self boundary intersects solid, so create a new boundary with the intersected domain.
                combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif solid.ContainsBoundary(boundary):
                # Self boundary is separate from solid, so its containment is based on being wholly contained within solid. 
                combinedSolid.boundaries.append(boundary)

        for boundary in solid.boundaries:
            newDomain = None
            # Slice solid boundary manifold by self. If it intersects, intersect the domains.
            slice, coincidences = self.Slice(boundary.manifold, cache)
            if slice:
                newDomain = boundary.domain.Intersection(slice, cache)
            elif coincidences:
                newDomain = boundary.domain
            # Intersect domain coincidences with the newDomain. Above, we subtracted them so they only appear once.
            if newDomain:
                for domainCoincidence in coincidences:
                    newDomain = newDomain.Intersection(domainCoincidence)
            if newDomain:
                # Solid boundary intersects self, so create a new boundary with the intersected domain.
                combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif self.ContainsBoundary(boundary):
                # Solid boundary is separate from self, so its containment is based on being wholly contained within self. 
                combinedSolid.boundaries.append(boundary)

        return combinedSolid

    def __mul__(self, other):
        return self.Intersection(other)

    def Union(self, solid):
        return self.Not().Intersection(solid.Not()).Not()
    
    def __add__(self, other):
        return self.Union(other)

    def Difference(self, solid):
        return self.Intersection(solid.Not())

    def __sub__(self, other):
        return self.Difference(other)