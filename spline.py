import numpy as np
from solid import Solid, Boundary
from manifold import Manifold
from hyperplane import Hyperplane
from bspy import Spline as BspySpline

class Spline(Manifold):
    """
    A spline is a `Manifold` defined by b-spline basis and set of coefficients, whose dependent 
    variables outnumber its independent variables by one.

    Parameters
    ----------
    spline : `bspy.Spline`
        The bspy Spline object that represents the spline for this manifold, where spline.nDep must be spline.nInd + 1.
    """
    def __init__(self, spline):
        if spline.nDep != spline.nInd + 1: raise ValueError("spline.nDep must be spline.nInd + 1")
        if spline.nInd < 1 or spline.nInd > 2: raise ValueError("spline must be a curve or surface (spline.nInd = 1 or 2)")
        self.spline = spline
        self.normalDirection = 1.0

    def __str__(self):
        return self.spline.__str__()

    def __repr__(self):
        return self.spline.__repr__()

    def copy(self):
        """
        Copy the spline.

        Returns
        -------
        spline : `Spline`
        """
        spline = Spline(self.spline.copy())
        spline.normalDirection = self.normalDirection
        if hasattr(self, "material"):
            spline.material = self.material
        return spline

    def range_dimension(self):
        """
        Return the range dimension.

        Returns
        -------
        dimension : `int`
        """
        return self.spline.nDep

    def normal(self, domainPoint):
        """
        Return the normal.

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the normal.

        Returns
        -------
        normal : `numpy.array`
        """
        return self.normalDirection * self.spline.normal(domainPoint)

    def point(self, domainPoint):
        """
        Return the point.

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the point.

        Returns
        -------
        point : `numpy.array`
        """
        return self.spline(domainPoint)

    def any_point(self):
        """
        Return an arbitrary point on the spline.

        Returns
        -------
        point : `numpy.array`
            A point on the spline.

        See Also
        --------
        `Solid.any_point` : Return an arbitrary point on the solid.
        `Boundary.any_point` : Return an arbitrary point on the boundary.

        Notes
        -----
        The any_point method for solids and boundaries do not call this method, because the point returned 
        may not be within the solid or boundary.
        """
        domainPoint = []
        for knots, nCoef in zip(self.spline.knots, self.spline.nCoef):
            domainPoint.append(knots[nCoef])
        return self.spline(domainPoint)

    def tangent_space(self, domainPoint):
        """
        Return the tangent space.

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the tangent space.

        Returns
        -------
        tangentSpace : `numpy.array`
        """
        tangentSpace = np.empty((self.spline.nDep, self.spline.nInd), self.spline.coefs.dtype)
        wrt = [0] * self.spline.nInd
        for i in range(self.spline.nInd):
            wrt[i] = 1
            tangentSpace[:, i] = self.spline.derivative(wrt, domainPoint)
            wrt[i] = 0
        return tangentSpace

    def cofactor_normal(self, domainPoint):
        """
        Return the cofactor normal.

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the cofactor normal.

        Returns
        -------
        cofactorNormal : `numpy.array`

        Notes
        -----
        The cofactor normal is the normal formed by the cross-product of the tangent space vectors (the tangents).

        See Also
        --------
        `solid.Solid.volume_integral` : Compute the volume integral of a function within the solid.
        `solid.Solid.surface_integral` : Compute the surface integral of a vector field on the boundary of the solid.
        """
        return self.normalDirection * self.spline.normal(domainPoint, False)

    def first_cofactor(self, domainPoint):
        """
        Return the first coordinate of the cofactor normal.

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the first cofactor.

        Returns
        -------
        firstCofactor : scalar

        Notes
        -----
        The cofactor normal is the normal formed by the cross-product of the tangent space vectors (the tangents).

        See Also
        --------
        `solid.Solid.volume_integral` : Compute the volume integral of a function within the solid.
        `solid.Solid.surface_integral` : Compute the surface integral of a vector field on the boundary of the solid.
        """
        return self.normalDirection * self.spline.normal(domainPoint, False, (0,))

    def transform(self, matrix, matrixInverseTranspose = None):
        """
        Transform the range of the spline.

        Parameters
        ----------
        matrix : `numpy.array`
            A square matrix transformation.

        matrixInverseTranspose : `numpy.array`, optional
            The inverse transpose of matrix (computed if not provided).

        Notes
        -----
        Transforms the spline in place, so create a copy as needed.

        See Also
        --------
        `solid.Solid.transform` : Transform the range of the solid.
        """
        assert np.shape(matrix) == (self.range_dimension(), self.range_dimension())
        self.spline = self.spline.transform(matrix)
        
    def translate(self, delta):
        """
        translate the range of the hyperplane.

        Parameters
        ----------
        delta : `numpy.array`
            A 1D array translation.

        Notes
        -----
        Translates the hyperplane in place, so create a copy as needed.

        See Also
        --------
        `solid.Solid.translate` : translate the range of the solid.
        """
        assert len(delta) == self.range_dimension()
        self.spline = self.spline.translate(delta)

    def flip_normal(self):
        """
        Flip the direction of the normal.

        Notes
        -----
        Negates the normal in place, so create a copy as needed.

        See Also
        --------
        `solid.Solid.complement` : Return the complement of the solid: whatever was inside is outside and vice-versa.
        """
        self.normalDirection *= -1.0
    
    @staticmethod
    def _process_zero(zero):
        # Process zero returned by bspy.Spline.zeros().
        if isinstance(zero, tuple):
            zero = 0.5 * (zero[0] + zero[1]) # Punt on overlapping areas for now
        return np.atleast_1d(zero)

    def intersect_manifold(self, other):
        """
        Intersect a spline or hyperplane.

        Parameters
        ----------
        other : `Spline` or `Hyperplane`
            The `Manifold` intersecting the `Spline`.

        Returns
        -------
        intersections : `list` (Or `NotImplemented` if other is not a `Hyperplane` nor a `Spline`)
            A list of intersections between the two manifolds. 
            Each intersection records either a crossing or a coincident region.
            Coincident regions are currently not implemented for splines.

            For a crossing, intersection is a Manifold.Crossing: (left, right)
            * left : `Manifold` in the manifold's domain where the manifold and the other cross.
            * right : `Manifold` in the other's domain where the manifold and the other cross.
            * Both intersection manifolds have the same domain and range (the crossing between the manifold and the other).

            For a coincident region, intersection is Manifold.Coincidence: (left, right, alignment, transform, inverse, translation)
            * left : `Solid` in the manifold's domain within which the manifold and the other are coincident.
            * right : `Solid` in the other's domain within which the manifold and the other are coincident.
            * alignment : scalar value holding the normal alignment between the manifold and the other (the dot product of their unit normals).
            * transform : `numpy.array` holding the transform matrix from the manifold's domain to the other's domain.
            * inverse : `numpy.array` holding the inverse transform matrix from the other's domain to the boundary's domain.
            * translation : `numpy.array` holding the translation vector from the manifold's domain to the other's domain.
            * Together transform, inverse, and translation form the mapping from the manifold's domain to the other's domain and vice-versa.

        See Also
        --------
        `solid.Solid.slice` : slice the solid by a manifold.
        `numpy.linalg.svd` : Compute the singular value decomposition of a 2D array.
        `bspy.Spline.zeros` : Find the roots of a spline (nInd must match nDep).
        `bspy.Spline.contours` : Find all the contour curves of a spline whose nInd is one larger than its nDep.

        Notes
        -----
        This method basically wraps the `bspy.Spline.zeros` and `bspy.Spline.contours` calls. We construct a spline that represents the 
        intersection and then call zeros or contours, depending on the dimension. The only subtly is getting the normals of intersections to always 
        point outward. We do that by picking an intersection point, checking the normal direction, and flipping it as needed.
        """
        assert self.range_dimension() == other.range_dimension()
        intersections = []
        nDep = self.spline.nInd # The dimension of the intersection's range

        # Spline-Hyperplane intersection.
        if isinstance(other, Hyperplane):
            # Compute the projection onto the hyperplane to map Spline-Hyperplane intersection points to the domain of the Hyperplane.
            projection = np.linalg.inv(other._tangentSpace.T @ other._tangentSpace) @ other._tangentSpace.T
            # Construct a new spline that represents the intersection.
            spline = self.spline.dot(other._normal) - np.atleast_1d(np.dot(other._normal, other._point))

            # Curve-Line intersection.
            if nDep == 1:
                # Find the intersection points and intervals.
                zeros = spline.zeros()
                # Convert each intersection point into a Manifold.Crossing and each intersection interval into a Manifold.Coincidence.
                for zero in zeros:
                    if isinstance(zero, tuple):
                        # Intersection is an interval, so create a Manifold.Coincidence.
                        planeBounds = (projection @ (self.spline((zero[0],)) - other._point), projection @ (self.spline((zero[1],)) - other._point))

                        # First, check for crossings at the boundaries of the coincidence, since splines can have discontinuous tangents.
                        # We do this first because later we may change the order of the plane bounds.
                        (bounds,) = self.spline.domain()
                        epsilon = 0.1 * Manifold.minSeparation
                        if zero[0] - epsilon > bounds[0]:
                            intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[0] - epsilon, 0.0), Hyperplane(1.0, planeBounds[0], 0.0)))
                        if zero[1] + epsilon < bounds[1]:
                            intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[1] + epsilon, 0.0), Hyperplane(1.0, planeBounds[1], 0.0)))

                        # Now, create the coincidence.
                        left = Solid(nDep, False)
                        left.boundaries.append(Boundary(Hyperplane(-1.0, zero[0], 0.0), Solid(0, True)))
                        left.boundaries.append(Boundary(Hyperplane(1.0, zero[1], 0.0), Solid(0, True)))
                        right = Solid(nDep, False)
                        if planeBounds[0] > planeBounds[1]:
                            planeBounds = (planeBounds[1], planeBounds[0])
                        right.boundaries.append(Boundary(Hyperplane(-1.0, planeBounds[0], 0.0), Solid(0, True)))
                        right.boundaries.append(Boundary(Hyperplane(1.0, planeBounds[1], 0.0), Solid(0, True)))
                        alignment = np.dot(self.normal((zero[0],)), other._normal) # Use the first zero, since B-splines are closed on the left
                        width = zero[1] - zero[0]
                        transform = (planeBounds[1] - planeBounds[0]) / width
                        translation = (planeBounds[0] * zero[1] - planeBounds[1] * zero[0]) / width
                        intersections.append(Manifold.Coincidence(left, right, alignment, np.atleast_2d(transform), np.atleast_2d(1.0 / transform), np.atleast_1d(translation)))
                    else:
                        # Intersection is a point, so create a Manifold.Crossing.
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero, 0.0), Hyperplane(1.0, projection @ (self.spline((zero,)) - other._point), 0.0)))

            # Surface-Plane intersection.
            elif nDep == 2:
                # Find the intersection contours, which are returned as splines.
                contours = spline.contours()
                # Convert each contour into a Manifold.Crossing.
                for contour in contours:
                    # The left portion is the contour returned for the spline-plane intersection. 
                    left = contour
                    # The right portion is the contour projected onto the plane's domain, which we compute with samples and a least squares fit.
                    tValues = np.linspace(0.0, 1.0, contour.nCoef[0] + 5) # Over-sample a bit to reduce the condition number and avoid singular matrix
                    points = []
                    for t in tValues:
                        zero = contour((t,))
                        points.append(projection @ (self.spline(zero) - other._point))
                    right = BspySpline.least_squares(tValues, np.array(points).T, contour.order, contour.knots)
                    intersections.append(Manifold.Crossing(Spline(left), Spline(right)))
            else:
                return NotImplemented
        
        # Spline-Spline intersection.
        elif isinstance(other, Spline):
            # Construct a new spline that represents the intersection.
            spline = self.spline.subtract(other.spline)

            # Curve-Curve intersection.
            if nDep == 1:
                # Find the intersection points and intervals.
                zeros = spline.zeros()
                # Convert each intersection point into a Manifold.Crossing and each intersection interval into a Manifold.Coincidence.
                for zero in zeros:
                    if isinstance(zero, tuple):
                        # Intersection is an interval, so create a Manifold.Coincidence.

                        # First, check for crossings at the boundaries of the coincidence, since splines can have discontinuous tangents.
                        # We do this first to match the approach for Curve-Line intersections.
                        (boundsSelf,) = self.spline.domain()
                        (boundsOther,) = other.spline.domain()
                        epsilon = 0.1 * Manifold.minSeparation
                        if zero[0][0] - epsilon > boundsSelf[0]:
                            intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[0][0] - epsilon, 0.0), Hyperplane(1.0, zero[0][1], 0.0)))
                        elif zero[0][1] - epsilon > boundsOther[0]:
                            intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[0][0], 0.0), Hyperplane(1.0, zero[0][1] - epsilon, 0.0)))
                        if zero[1][0] + epsilon < boundsSelf[1]:
                            intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[1][0] + epsilon, 0.0), Hyperplane(1.0, zero[1][1], 0.0)))
                        elif zero[1][1] + epsilon < boundsOther[1]:
                            intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[1][0], 0.0), Hyperplane(1.0, zero[1][1] + epsilon, 0.0)))

                        # Now, create the coincidence.
                        left = Solid(nDep, False)
                        left.boundaries.append(Boundary(Hyperplane(-1.0, zero[0][0], 0.0), Solid(0, True)))
                        left.boundaries.append(Boundary(Hyperplane(1.0, zero[1][0], 0.0), Solid(0, True)))
                        right = Solid(nDep, False)
                        right.boundaries.append(Boundary(Hyperplane(-1.0, zero[0][1], 0.0), Solid(0, True)))
                        right.boundaries.append(Boundary(Hyperplane(1.0, zero[1][1], 0.0), Solid(0, True)))
                        alignment = np.dot(self.normal(zero[0][0]), other.normal(zero[0][1])) # Use the first zeros, since B-splines are closed on the left
                        width = zero[1][0] - zero[0][0]
                        transform = (zero[1][1] - zero[0][1]) / width
                        translation = (zero[0][1] * zero[1][0] - zero[1][1] * zero[0][0]) / width
                        intersections.append(Manifold.Coincidence(left, right, alignment, np.atleast_2d(transform), np.atleast_2d(1.0 / transform), np.atleast_1d(translation)))
                    else:
                        # Intersection is a point, so create a Manifold.Crossing.
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[:nDep], 0.0), Hyperplane(1.0, zero[nDep:], 0.0)))
            
            # Surface-Surface intersection.
            elif nDep == 2:
                # Find the intersection contours, which are returned as splines.
                contours = spline.contours()
                # Convert each contour into a Manifold.Crossing.
                for contour in contours:
                    left = BspySpline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[:nDep], contour.metadata)
                    right = BspySpline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[nDep:], contour.metadata)
                    intersections.append(Manifold.Crossing(Spline(left), Spline(right)))
            else:
                return NotImplemented
        else:
            return NotImplemented

        # Ensure the normals point outwards for both Manifolds in each crossing intersection.
        # Note that evaluating left and right at 0.0 is always valid because either they are points or curves with [0.0, 1.0] domains.
        domainPoint = np.atleast_1d(0.0)
        for intersection in intersections:
            if isinstance(intersection, Manifold.Crossing):
                if np.dot(self.tangent_space(intersection.left.point(domainPoint)) @ intersection.left.normal(domainPoint), other.normal(intersection.right.point(domainPoint))) < 0.0:
                    intersection.left.flip_normal()
                if np.dot(other.tangent_space(intersection.right.point(domainPoint)) @ intersection.right.normal(domainPoint), self.normal(intersection.left.point(domainPoint))) < 0.0:
                    intersection.right.flip_normal()

        return intersections
    
    @staticmethod
    def establish_domain_bounds(domain, bounds):
        """
        Establish the outer bounds of a spline's domain (creates a hypercube based on the spline's bounds).

        Parameters
        ----------
        domain : `solid.Solid`
            The domain of the spline into which boundaries should be added based on the spline's bounds.

        bounds : array-like
            nInd x 2 array of the lower and upper bounds on each of the independent variables.

        See Also
        --------
        `solid.Solid.slice` : slice the solid by a manifold.
        `complete_slice` : Add any missing inherent (implicit) boundaries of this manifold's domain to the given slice.

        Notes
        -----
        A spline's inherent domain is determined by its knot array for each dimension. 
        If you pass in an empty domain, it will remain empty.
        """
        dimension = len(bounds)
        assert len(bounds[0]) == 2
        assert domain.dimension == dimension
        domain.containsInfinity = False
        for i in range(dimension):
            if dimension > 1:
                domainDomain1 = Solid(dimension - 1, False)
                Spline.establish_domain_bounds(domainDomain1, np.delete(bounds, i, axis=0))
                domainDomain2 = Solid(dimension - 1, False)
                Spline.establish_domain_bounds(domainDomain2, np.delete(bounds, i, axis=0))
            else:
                domainDomain1 = Solid(0, True)
                domainDomain2 = Solid(0, True)
            diagonal = np.identity(dimension)
            unitVector = diagonal[i]
            if dimension > 1:
                tangentSpace = np.delete(diagonal, i, axis=1)
            else:
                tangentSpace = np.array([0.0])
            hyperplane = Hyperplane(-unitVector, bounds[i][0] * unitVector, tangentSpace)
            domain.boundaries.append(Boundary(hyperplane, domainDomain1))
            hyperplane = Hyperplane(unitVector, bounds[i][1] * unitVector, tangentSpace)
            domain.boundaries.append(Boundary(hyperplane, domainDomain2))

    def complete_slice(self, slice, solid):
        """
        Add any missing inherent (implicit) boundaries of this manifold's domain to the given slice of the 
        given solid that are needed to make the slice valid and complete.

        Parameters
        ----------
        slice : `solid.Solid`
            The slice of the given solid formed by the manifold. The slice may be incomplete, missing some of the 
            manifold's inherent domain boundaries. Its dimension must match `self.domain_dimension()`.

        solid : `solid.Solid`
            The solid being sliced by the manifold. Its dimension must match `self.range_dimension()`.

        See Also
        --------
        `establish_domain_bounds` : Establish the outer bounds of a spline's domain.
        `solid.Solid.slice` : slice the solid by a manifold.

        Notes
        -----
        A spline's inherent domain is determined by its knot array for each dimension. 
        """
        assert self.domain_dimension() == slice.dimension
        assert self.range_dimension() == solid.dimension
        assert slice.dimension == 1 or slice.dimension == 2

        # Spline manifold domains have finite bounds.
        slice.containsInfinity = False
        bounds = self.spline.domain()

        # If manifold (self) has no intersections with solid, just check containment.
        if not slice.boundaries:
            if solid.contains_point(self.any_point()):
                self.establish_domain_bounds(slice, bounds)
            return

        # For curves, add domain bounds as needed.
        if slice.dimension == 1:
            slice.boundaries.sort(key=lambda b: (b.manifold.point(0.0), b.manifold.normal(0.0)))
            if abs(slice.boundaries[0].manifold._point - bounds[0][0]) >= Manifold.minSeparation and \
                slice.boundaries[0].manifold._normal > 0.0:
                slice.boundaries.insert(0, Boundary(Hyperplane(-slice.boundaries[0].manifold._normal, bounds[0][0], 0.0), Solid(0, True)))
            if abs(slice.boundaries[-1].manifold._point - bounds[0][1]) >= Manifold.minSeparation and \
                slice.boundaries[-1].manifold._normal < 0.0:
                slice.boundaries.append(Boundary(Hyperplane(-slice.boundaries[-1].manifold._normal, bounds[0][1], 0.0), Solid(0, True)))

        # For surfaces, add bounding box for domain and intersect it with existing slice boundaries.
        if slice.dimension == 2:
            boundaryCount = len(slice.boundaries) # Keep track of existing slice boundaries
            self.establish_domain_bounds(slice, bounds) # Add bounding box boundaries to slice boundaries
            for boundary in slice.boundaries[boundaryCount:]: # Mark bounding box boundaries as untouched
                boundary.touched = False

            # Nested function for adding slice points to new bounding box boundaries.
            def process_domain_point(boundary, domainPoint):
                point = boundary.manifold.point(domainPoint)
                # See if and where point touches bounding box of slice.
                for newBoundary in slice.boundaries[boundaryCount:]:
                    vector = point - newBoundary.manifold._point
                    if abs(np.dot(newBoundary.manifold._normal, vector)) < Manifold.minSeparation:
                        # Add the point onto the new boundary.
                        normal = np.sign(newBoundary.manifold._tangentSpace.T @ boundary.manifold.normal(domainPoint))
                        newBoundary.domain.boundaries.append(Boundary(Hyperplane(normal, newBoundary.manifold._tangentSpace.T @ vector, 0.0), Solid(0, True)))
                        newBoundary.touched = True
                        break

            # Go through existing boundaries and check if either of their endpoints lies on the spline's bounds.
            for boundary in slice.boundaries[:boundaryCount]:
                domainBoundaries = boundary.domain.boundaries
                domainBoundaries.sort(key=lambda boundary: (boundary.manifold.point(0.0), boundary.manifold.normal(0.0)))
                process_domain_point(boundary, domainBoundaries[0].manifold._point)
                if len(domainBoundaries) > 1:
                    process_domain_point(boundary, domainBoundaries[-1].manifold._point)
            
            # For touched boundaries, remove domain bounds that aren't needed.
            for newBoundary in slice.boundaries[boundaryCount:]:
                if newBoundary.touched:
                    domainBoundaries = newBoundary.domain.boundaries
                    assert len(domainBoundaries) > 2
                    domainBoundaries.sort(key=lambda boundary: (boundary.manifold.point(0.0), boundary.manifold.normal(0.0)))
                    # Ensure domain endpoints don't overlap and their normals are consistent.
                    if abs(domainBoundaries[0].manifold._point - domainBoundaries[1].manifold._point) < Manifold.minSeparation or \
                        domainBoundaries[1].manifold._normal < 0.0:
                        del domainBoundaries[0]
                    if abs(domainBoundaries[-1].manifold._point - domainBoundaries[-2].manifold._point) < Manifold.minSeparation or \
                        domainBoundaries[-2].manifold._normal > 0.0:
                        del domainBoundaries[-1]
            
            # Touch untouched boundaries that are connected to touched boundary endpoints.
            boundaryMap = ((2, 3, 0), (2, 3, -1), (0, 1, 0), (0, 1, -1)) # Map of which bounding box boundaries touch each other
            while True:
                noTouches = True
                for map, newBoundary, bound in zip(boundaryMap, slice.boundaries[boundaryCount:], bounds.flatten()):
                    if not newBoundary.touched:
                        leftBoundary = slice.boundaries[boundaryCount + map[0]]
                        rightBoundary = slice.boundaries[boundaryCount + map[1]]
                        if leftBoundary.touched and abs(leftBoundary.domain.boundaries[map[2]].manifold._point - bound) < Manifold.minSeparation:
                            newBoundary.touched = True
                            noTouches = False
                        elif rightBoundary.touched and abs(rightBoundary.domain.boundaries[map[2]].manifold._point - bound) < Manifold.minSeparation:
                            newBoundary.touched = True
                            noTouches = False
                if noTouches:
                    break
            
            # Remove untouched boundaries.
            i = boundaryCount
            while i < len(slice.boundaries):
                if not slice.boundaries[i].touched:
                    del slice.boundaries[i]
                else:
                    i += 1