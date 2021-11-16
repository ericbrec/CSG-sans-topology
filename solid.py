import numpy as np
import scipy.integrate as integrate
import manifold as mf

class Boundary:
    """
    A portion of the boundary of a solid.

    Parameters
    ----------
    manifold : `manifold.Manifold`
        The differentiable function whose range is one dimension higher than its domain that defines the range of the boundary.
    
    domain : `Solid`
        The region of the domain of the manifold that's within the boundary.
    
    tangentSpace : array-like
        A array of tangents that are linearly independent and orthoganal to the normal.
    
    See also
    --------
    `Solid` : A region that separates space into an inside and outside, defined by a collection of boundaries.
    """
    @staticmethod
    def SortKey(boundary):
        if boundary.domain.dimension > 0:
            return 0.0
        else:
            return (boundary.manifold.Point(0.0), -boundary.manifold.Normal(0.0))

    def __init__(self, manifold, domain):
        assert manifold.DomainDimension() == domain.dimension

        self.manifold = manifold
        self.domain = domain

    def __str__(self):
        return "{0}, {1}".format(self.manifold, "Contains infinity" if self.domain.containsInfinity else "Excludes infinity")

    def __repr__(self):
        return "Boundary({0}, {1})".format(self.manifold.__repr__(), self.domain.__repr__())

    def AnyPoint(self):
        """
        Return an arbitrary point on the boundary.

        Returns
        -------
        point : `numpy.array`
            A point on the boundary.

        See Also
        --------
        `Solid.AnyPoint` : Return an arbitrary point on the solid.

        Notes
        -----
        The point is computed by evaluating the boundary manifold by an arbitrary point in the domain of the boundary.
        """
        return self.manifold.Point(self.domain.AnyPoint())

class Solid:
    """
    A region that separates space into an inside and outside, defined by a collection of boundaries.

    Parameters
    ----------
    dimension : `int`
        The dimension of the solid (non-negative).
    
    containsInfinity : `bool`
        Indicates whether or not the solid contains infinity.
    
    See also
    --------
    `Boundary` : A portion of the boundary of a solid.

    Notes
    -----
    Solids also contain a `list` of `boundaries`. That list may be empty.

    Solids can be of zero dimension, typically acting as the domain of boundary endpoints. Zero-dimension solids have no boundaries, they only contain infinity or not.
    """
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
        """
        Test if the solid is empty.

        Returns
        -------
        isEmpty : `bool`
            `True` if the solid is empty, `False` otherwise.

        Notes
        -----
        Casting the solid to `bool` returns not `IsEmpty`.
        """
        return not self

    def Not(self):
        """
        Return the compliment of the solid: whatever was inside is outside and vice-versa.

        Returns
        -------
        solid : `Solid`
            The compliment of the solid.

        See Also
        --------
        `Intersect` : Intersect two solids.
        `Union` : Union two solids.
        `Difference` : Subtract one solid from another. 
        """
        solid = Solid(self.dimension, not self.containsInfinity)
        for boundary in self.boundaries:
            manifold = boundary.manifold.copy()
            manifold.FlipNormal()
            solid.boundaries.append(Boundary(manifold,boundary.domain))
        return solid

    def __neg__(self):
        return self.Not()

    def Transform(self, transform):
        """
        Transform the range of the solid.

        Parameters
        ----------
        transform : `numpy.array`
            A square 2D array transformation.

        Notes
        -----
        Transforms the solid in place, so create a copy as needed.
        """
        assert np.shape(transform) == (self.dimension, self.dimension)

        for boundary in self.boundaries:
            boundary.manifold.Transform(transform)

    def Translate(self, delta):
        """
        Translate the range of the solid.

        Parameters
        ----------
        delta : `numpy.array`
            A 1D array translation.

        Notes
        -----
        Translates the solid in place, so create a copy as needed.
        """
        assert len(delta) == self.dimension

        for boundary in self.boundaries:
            boundary.manifold.Translate(delta)

    def AnyPoint(self):
        """
        Return an arbitrary point on the solid.

        Returns
        -------
        point : `numpy.array`
            A point on the solid.

        See Also
        --------
        `Boundary.AnyPoint` : Return an arbitrary point on the boundary.

        Notes
        -----
        The point is computed by calling `Boundary.AnyPoint` on the solid's first boundary.
        If the solid has no boundaries but contains infinity, `AnyPoint` returns the origin.
        If the solid has no boundaries and doesn't contain infinity, `AnyPoint` returns `None`.
        """
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
        """
        A generator for edges of the solid.

        Yields
        -------
        (point1, point2, normal) : `tuple(numpy.array, numpy.array, numpy.array)`
            Starting point, ending point, and normal for an edge of the solid.


        Notes
        -----
        The edges are not guaranteed to be connected or in any particular order, and typically aren't.

        If the solid is a number line (dimension 1), the generator yields a tuple with two scalar values (start, end).
        """
        if self.dimension > 1:
            for boundary in self.boundaries:
                for domainEdge in boundary.domain.Edges():
                    yield (boundary.manifold.Point(domainEdge[0]), boundary.manifold.Point(domainEdge[1]), boundary.manifold.Normal(domainEdge[0]))
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
                            yield (leftPoint, rightPoint)
                            leftB = rightB
                            rightB += 1
                            break
                        rightB += 1
                leftB += 1

    def VolumeIntegral(self, f, args=(), epsabs=None, epsrel=None, *quadArgs):
        """
        Compute the volume integral of a function within the solid.

        Parameters
        ----------
        f : python function `f(point: numpy.array, args : user-defined) -> scalar value`
            The function to be integrated within the solid.
            It's passed a point within the solid, as well as any optional user-defined arguments.
        
        args : tuple, optional
            Extra arguments to pass to `f`.
        
        *quadArgs : Quadrature arguments passed to `scipy.integrate.quad`.

        Returns
        -------
        sum : scalar value
            The value of the volume integral.

        See Also
        --------
        `SurfaceIntegral` : Compute the surface integral of a vector field on the boundary of the solid.
        `scipy.integrate.quad` : Integrate func from a to b (possibly infinite interval) using a technique from the Fortran library QUADPACK.

        Notes
        -----
        The volume integral is computed by recursive application of the divergence theorem: `VolumeIntegral(divergence(F)) = SurfaceIntegral(dot(F, n))`, 
        where `F` is a vector field and `n` is the outward boundary unit normal.
        
        Let `F = [Integral(f) from x0 to x holding other coordinates fixed, 0..0]`. `divergence(F) = f` by construction, and `dot(F, n) = Integral(f) * n[0]`.
        Note that the choice of `x0` is arbitrary as long as it's in the domain of f and doesn't change across all surface integral boundaries.

        Thus, we have `VolumeIntegral(f) = SurfaceIntegral(Integral(f) * n[0])`.
        The outward boundary unit normal, `n`, is the cross product of the boundary manifold's tangent space divided by its length.
        The surface differential, `dS`, is the length of cross product of the boundary manifold's tangent space times the differentials of the manifold's domain variables.
        The length of the cross product appears in the numerator and denominator of the surface integral and cancels.
        What's left multiplying `Integral(f)` is the first coordinate of the cross product plus the domain differentials (volume integral).
        The first coordinate of the cross product of the boundary manifold's tangent space is the first cofactor of the tangent space.
        And so, `SurfaceIntegral(Integral(f) * n[0]) = VolumeIntegral(Integral(f) * first cofactor)` over each boundary manifold's domain.

        So, we have `VolumeIntegral(f) = VolumeIntegral(Integral(f) * first cofactor)` over each boundary manifold's domain.
        To compute the volume integral we sum `VolumeIntegral` over the domain of the solid's boundaries, using the integrand:
        `scipy.integrate.quad(f, x0, x [other coordinates fixed]) * first cofactor`.
        This recursion continues until the boundaries are only points, where we can just sum the integrand.
        """
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
        """
        Compute the surface integral of a vector field on the boundary of the solid.

        Parameters
        ----------
        f : python function `f(point: numpy.array, normal: numpy.array, args : user-defined) -> numpy.array`
            The vector field to be integrated on the boundary of the solid.
            It's passed a point on the boundary and its corresponding outward-pointing unit normal, as well as any optional user-defined arguments.
        
        args : tuple, optional
            Extra arguments to pass to `f`.
        
        *quadArgs : Quadrature arguments passed to `scipy.integrate.quad`.

        Returns
        -------
        sum : scalar value
            The value of the surface integral.

        See Also
        --------
        `VolumeIntegral` : Compute the volume integral of a function within the solid.
        `scipy.integrate.quad` : Integrate func from a to b (possibly infinite interval) using a technique from the Fortran library QUADPACK.

        Notes
        -----
        To compute the surface integral of a scalar function on the boundary, have `f` return the product of the `normal` times the scalar function for the `point`.

        `SurfaceIntegral` sums the `VolumeIntegral` over the domain of the solid's boundaries, using the integrand: `numpy.dot(f(point, normal), Normal)`, 
        where `Normal` is the cross-product of the boundary tangents (the normal before normalization).
        """
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
        """
        Compute the winding number for a point relative to the solid.

        Parameters
        ----------
        point : array-like
            A point that may lie within the solid.

        Returns
        -------
        windingNumber : scalar value
            The `windingNumber` is 0 if the point is outside the solid, 1 if it's inside.
            Other values indicate issues:
            * A point on the boundary leads to an undefined (random) winding number;
            * Boundaries with gaps or overlaps lead to fractional winding numbers;
            * Interior-pointing normals lead to negative winding numbers;
            * Nested shells lead to winding numbers with absolute value 2 or greater.
        
        onBoundaryNormal : `numpy.array`
            The boundary normal if the point lies on a boundary, `None` otherwise.

        See Also
        --------
        `ContainsPoint` : Test if a point lies within the solid.
        `manifold.Manifold.IntersectXRay` : Intersect a ray along the x-axis with the manifold.

        Notes
        -----
        If `onBoundaryNormal` is not `None`, `windingNumber` is undefined and should be ignored.

        `WindingNumber` uses three different implementations:
        * A simple fast implementation if the solid is a number line (dimension <= 1). This is the default for dimension <= 1.
        * Dan Sunday's fast ray-casting algorithm: Sunday, Dan (2001), "Inclusion of a Point in a Polygon." This is the default for dimension > 1.
        * A surface integral with integrand: `(x - point) / norm(x - point)**dimension`. This is just for fun, though it's more robust for boundaries with gaps. 
        """
        windingNumber = 0.0
        onBoundaryNormal = None
        if self.containsInfinity:
            # If the solid contains infinity, then the winding number starts as 1 to account for the boundary at infinity.
            windingNumber = 1.0

        if self.dimension <= 1:
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
                    # Each intersection is of the form (distance to intersection, domain point of intersection).
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
            # Compute the winding number via a surface integral, normalizing by the hypersphere surface area. 
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
        """
        Test if a point lies within the solid.

        Parameters
        ----------
        point : array-like
            A point that may lie within the solid.

        Returns
        -------
        containment : `bool`
            `True` if `point` lies within the solid. `False` otherwise.

        See Also
        --------
        `WindingNumber` : Compute the winding number for a point relative to the solid.

        Notes
        -----
        A point is considered contained if it's on the boundary of the solid or it's winding number is greater than 0.5.
        """
        windingNumber, onBoundaryNormal = self.WindingNumber(point)
        # The default is to include points on the boundary (onBoundaryNormal is not None).
        containment = True
        if onBoundaryNormal is None:
            # windingNumber > 0.5 returns a np.bool_, not a bool, so we need to cast it.
            containment = bool(windingNumber > 0.5)
        return containment

    def Slice(self, manifold, cache = None, trimTwin = False):
        """
        Slice the solid by a manifold.

        Parameters
        ----------
        manifold : `manifold.Manifold`
            The `Manifold` used to slice the solid.
        
        cache : `dict`, optional
            A dictionary to cache `Manifold` intersections, speeding computation.
        
        trimTwin : `bool`, default: False
            Trim coincident boundary twins on subsequent calls to slice (avoids duplication of overlapping regions).
            Trimming twins is typically only used in conjunction with `Intersection`.

        Returns
        -------
        slice : `Solid`
            A region in the domain of `manifold` that intersects with the solid. The region may contain infinity.

        See Also
        --------
        `Intersection` : Intersect two solids.
        `manifold.Manifold.IntersectManifold` : Intersect two manifolds.

        Notes
        -----
        The dimension of the slice is always one less than the dimension of the solid.
        """
        assert manifold.RangeDimension() == self.dimension

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
                    #   * intersection[b] is the solid within the boundary's domain inside of which the boundary and given manifold are coincident.
                    #   * intersection[m] is the solid within the manifold's domain inside of which the boundary and given manifold are coincident.
                    #   * intersection[2] is the normal alignment between the boundary and given manifold (same or opposite directions).
                    #   * intersection[3] is transform from the boundary's domain to the given manifold's domain.
                    #   * intersection[4] is the translation from the boundary's domain to the given manifold's domain.
                    #   * Together intersection[3] and intersection[4] form the mapping from the boundary's domain to the given manifold's domain.

                    # First, intersect domain coincidence with the domain boundary.
                    domainCoincidence = intersection[b].Intersection(boundary.domain)
                    # Next, invert the domain coincidence (which will remove it) if this is a twin or if the normals point in opposite directions.
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
                # If the domain coincidence is inverted (domainCoincidence[0]), intersect it with the slice, thus removing it.
                slice = slice.Intersection(domainCoincidence[1], cache)
            else:
                # Otherwise, union the domain coincidence with the slice, thus adding it.
                slice = slice.Union(domainCoincidence[1])

        # For a point without coincidences, slice is based on point containment.
        if slice.dimension < 1 and len(coincidences) == 0:
            slice.containsInfinity =  self.ContainsPoint(manifold.Point(0.0))

        return slice

    def Intersection(self, solid, cache = None):
        """
        Intersect two solids.

        Parameters
        ----------
        solid : `Solid`
            The passed `Solid` intersecting the solid.
        
        cache : `dict`, optional
            A dictionary to cache `Manifold` intersections, speeding computation. If no dictionary is passed, one is created.

        Returns
        -------
        combinedSolid : `Solid`
            A `Solid` that represents the intersection between the passed `solid` and the solid.

        See Also
        --------
        `Slice` : Slice a solid by a manifold.
        `Union` : Union two solids.
        `Difference` : Subtract one solid from another.

        Notes:
        ------
        To intersect two solids, we slice each solid with the boundaries of the other solid. The slices are the region
        of the domain that intersect the solid. We then intersect the domain of each boundary with its slice of the other solid. Thus,
        the intersection of two solids becomes a set of interections within the domains of their boundaries. This recursion continues
        until we are intersecting points whose domains have no boundaries.

        The only subtlety is when two boundaries are coincident. To avoid overlapping the coincident region, we keep that region
        for one slice and trim it away for the other. We use a manifold intersection cache to keep track of these pairs, as well as reduce computation. 
        """
        assert self.dimension == solid.dimension

        # Manifold intersections are expensive and come in symmetric pairs (m1 intersect m2, m2 intersect m1).
        # So, we create a manifold intersections cache (dictionary) to store and reuse intersection pairs.
        # The cache is also used to avoid overlapping coincident regions by identifying twins that could be trimmed.
        if cache is None:
            cache = {}

        # Start with a solid without boundaries.
        combinedSolid = Solid(self.dimension, self.containsInfinity and solid.containsInfinity)

        for boundary in self.boundaries:
            # Slice self boundary manifold by solid. Trim away duplicate twin.
            slice = solid.Slice(boundary.manifold, cache, True)
            # Intersect slice with the boundary's domain.
            newDomain = boundary.domain.Intersection(slice, cache)
            if newDomain:
                # Self boundary intersects solid, so create a new boundary with the intersected domain.
                combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))

        for boundary in solid.boundaries:
            # Slice solid boundary manifold by self.  Trim away duplicate twin.
            slice = self.Slice(boundary.manifold, cache, True)
            # Intersect slice with the boundary's domain.
            newDomain = boundary.domain.Intersection(slice, cache)
            if newDomain:
                # Solid boundary intersects self, so create a new boundary with the intersected domain.
                combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))

        return combinedSolid

    def __mul__(self, other):
        return self.Intersection(other)

    def Union(self, solid):
        """
        Union two solids.

        Parameters
        ----------
        solid : `Solid`
            The passed `Solid` unioning the solid.

        Returns
        -------
        combinedSolid : `Solid`
            A `Solid` that represents the union between the passed `solid` and the solid.

        See Also
        --------
        `Intersect` : Intersect two solids.
        `Difference` : Subtract one solid from another. 
        """
        return self.Not().Intersection(solid.Not()).Not()

    def __add__(self, other):
        return self.Union(other)

    def Difference(self, solid):
        """
        Subtract one solid from another.

        Parameters
        ----------
        solid : `Solid`
            The passed `Solid` is subtracted from the solid.

        Returns
        -------
        combinedSolid : `Solid`
            A `Solid` that represents the subtraction of the passed `solid` from the solid.

        See Also
        --------
        `Intersect` : Intersect two solids.
        `Union` : Union two solids.
        """
        return self.Intersection(solid.Not())

    def __sub__(self, other):
        return self.Difference(other)