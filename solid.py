import numpy as np
import scipy.integrate as integrate
import manifold as mf

class Boundary:
    
    @staticmethod
    def SortKey(boundary):
        if boundary.domain.dimension > 0:
            return 0.0
        else:
            return (boundary.manifold.Point(0.0), -boundary.manifold.Normal(0.0))

    def __init__(self, manifold, domain):
        assert manifold.GetDomainDimension() == domain.dimension

        self.manifold = manifold
        self.domain = domain
    
    def __str__(self):
        return "{0}, {1}".format(self.manifold, "Contains infinity" if self.domain.containsInfinity else "Excludes infinity")
    
    def __repr__(self):
        return "Boundary({0}, {1})".format(self.manifold.__repr__(), self.domain.__repr__())
    
    def AnyPoint(self):
        return self.manifold.Point(self.domain.AnyPoint())

class Solid:

    def __init__(self, dimension, containsInfinity):
        assert dimension >= 0
        self.dimension = dimension
        self.containsInfinity = containsInfinity
        self.boundaries = []
    
    def __repr__(self):
        return "Solid({0}, {1})".format(self.dimension, self.containsInfinity)
    
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
        if self.boundaries:
            point = self.boundaries[0].AnyPoint()
        elif self.containsInfinity:
            if self.dimension > 0:
                point = np.full((self.dimension), 0.0)
            else:
                point = 0.0

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

            if boundary.domain.dimension > 0:
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
                evalPoint = np.atleast_1d(domainPoint)
                point = boundary.manifold.Point(evalPoint)
                normal = boundary.manifold.Normal(evalPoint)
                fValue = f(point, normal, *args)
                return np.dot(fValue, boundary.manifold.CofactorNormal(domainPoint))

            if boundary.domain.dimension > 0:
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
                        # Only include the boundary if the ray intersection is inside its domain.
                        if boundary.domain.ContainsPoint(intersection[1]):
                            # Check if intersection lies on the boundary.
                            if intersection[0] < mf.Manifold.minSeparation:
                                onBoundaryNormal = boundary.manifold.Normal(intersection[1])
                            # Accumulate winding number based on sign of dot(ray,normal) == normal[0].
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
            containment = bool(windingNumber > 0.5)
        return containment 

    def ContainsBoundary(self, boundary):
        containment = False
        if boundary.domain is None:
            domainPoint = 0.0
        else:
            domainPoint = boundary.domain.AnyPoint()
        windingNumber, onBoundaryNormal = self.WindingNumber(boundary.manifold.Point(domainPoint))
        if onBoundaryNormal is None:
            containment = bool(windingNumber > 0.5)
        else:
            # Boundaries are coincident at the point, so containment is based on alignment of normals.
            containment = bool(np.dot(onBoundaryNormal, boundary.manifold.Normal(domainPoint)) > 0.0)
        return containment

    def Slice(self, manifold, cache = None, trimTwin = False):
        assert manifold.GetRangeDimension() == self.dimension

        # Start with an empty slice and no domain coincidences.
        slice = Solid(self.dimension-1, self.containsInfinity)
        coincidences = []

        # Intersect each of this solid's boundaries with the manifold.
        for boundary in self.boundaries:
            intersections = None
            isTwin = False
            b = 0 # Index of intersection in the boundary's domain
            m = 1 # Index of intersection in the manifold's domain   

            # Check for previously computed manifold intersections stored in cache.
            if cache != None:
                # First, check for the twin (opposite order of arguments).
                intersections = cache.get((manifold, boundary.manifold))
                if intersections != None:
                    isTwin = True
                    b = 1 # Index of intersection in the boundary's domain
                    m = 0 # Index of intersection in the manifold's domain   
                else:
                    # Next, check for the original order (not twin).
                    intersections = cache.get((boundary.manifold, manifold))

            # If intersections not previously computed, compute them now.
            if intersections is None:
                intersections = boundary.manifold.IntersectManifold(manifold)
                # Store intersections in cache.
                if cache != None:
                    cache[(boundary.manifold, manifold)] = intersections

            # Each intersection is either a crossing (domain manifold) or a coincidence (solid within the domain).
            for intersection in intersections:
                if isinstance(intersection[b], mf.Manifold):
                    # IntersectManifold found a crossing, returned as a manifold pair:
                    #   * intersection[b] is the intersection manifold in the boundary's domain;
                    #   * intersection[m] is the intersection manifold in the given manifold's domain.
                    #   * Both intersection manifolds have the same range (the crossing between the boundary and the given manifold).

                    # We slice the boundary domain using intersection[b], but we add intersection[m] to the manifold domain.
                    # Both intersection manifolds share the same domain by construction in IntersectManifold.
                    intersectionSlice = boundary.domain.Slice(intersection[b], cache)
                    if intersectionSlice:
                        slice.boundaries.append(Boundary(intersection[m],intersectionSlice))
                
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
                    # Next, invert the domain coincidence if the manifold domain contains infinity (flip containsInfinity and later the normals).
                    # But do the reverse (which removes the domain coincidence), if this is the twin or if the normals point in opposite directions.
                    invertDomainCoincidence = (trimTwin and isTwin) or intersection[2] < 0.0
                    if invertDomainCoincidence:
                        domainCoincidence.containsInfinity = not domainCoincidence.containsInfinity
                    # Next, transform the domain coincidence from the boundary to the given manifold.
                    # Create copies of the manifolds and boundaries, since we are changing them.
                    for i in range(len(domainCoincidence.boundaries)):
                        domainManifold = domainCoincidence.boundaries[i].manifold.copy()
                        if invertDomainCoincidence:
                            domainManifold.FlipNormal()
                        if isTwin:
                            domainManifold.Translate(-intersection[4])
                            domainManifold.Transform(np.linalg.inv(intersection[3]))
                        else:
                            domainManifold.Transform(intersection[3])
                            domainManifold.Translate(intersection[4])
                        domainCoincidence.boundaries[i] = Boundary(domainManifold, domainCoincidence.boundaries[i].domain)
                    # Finally, add the domain coincidence to the list of coincidences.
                    coincidences.append((invertDomainCoincidence, domainCoincidence))
        
        # We've gone through all boundaries. Now that we have a complete manifold domain, join it with each domain coincidence.
        for domainCoincidence in coincidences:
            if domainCoincidence[0]:
                slice = slice.Intersection(domainCoincidence[1], cache)
            else:
                slice = slice.Union(domainCoincidence[1])
            
        # Handle slices without coincidences or intersections (keep slices that are empty due to coincidence).
        if len(coincidences) == 0 and slice.IsEmpty():
            if slice.dimension > 0:
                # Return None for dimensional manifolds that completely miss self.
                slice = None
            else:
                # Return containment for point manifolds.
                slice.containsInfinity = self.ContainsPoint(manifold.Point(0.0))
        
        return slice

    def Intersection(self, solid, cache = None):
        assert self.dimension == solid.dimension

        # Manifold intersections are expensive and come in symmetric pairs (m1 intersect m2, m2 intersect m1).
        # So, we create a manifold intersections cache (dictionary) to store and reuse intersection pairs.
        # The cache is also used to avoid overlapping coincident domains.
        if cache is None:
            cache = {}

        # Start with empty solid.
        combinedSolid = Solid(self.dimension, self.containsInfinity and solid.containsInfinity)

        for boundary in self.boundaries:
            # Slice self boundary manifold by solid. Trim away duplicate twin.
            slice = solid.Slice(boundary.manifold, cache, True)
            if slice is not None:
                # Intersect slice with the boundary's domain.
                newDomain = boundary.domain.Intersection(slice, cache)
                if newDomain:
                    # Self boundary intersects solid, so create a new boundary with the intersected domain.
                    combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif solid.ContainsPoint(boundary.AnyPoint()):
                # Self boundary is separate from solid, so its containment is based on being wholly contained within solid. 
                combinedSolid.boundaries.append(boundary)

        for boundary in solid.boundaries:
            # Slice solid boundary manifold by self.  Trim away duplicate twin.
            slice = self.Slice(boundary.manifold, cache, True)
            if slice is not None:
                # Intersect slice with the boundary's domain.
                newDomain = boundary.domain.Intersection(slice, cache)
                if newDomain:
                    # Solid boundary intersects self, so create a new boundary with the intersected domain.
                    combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif self.ContainsPoint(boundary.AnyPoint()):
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