import logging
import numpy as np
from solid import Solid, Boundary
from manifold import Manifold
from hyperplane import Hyperplane
from bspy import Spline

class BSpline(Manifold):
    """
    A BSpline is a `Manifold` defined by b-spline basis and set of coefficients, whose dependent 
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

    def __repr__(self):
        return self.spline.__repr__()

    def copy(self):
        """
        Copy the spline.

        Returns
        -------
        spline : `Spline`
        """
        spline = BSpline(self.spline.copy())
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

    def normal(self, domainPoint, normalize=True, indices=None):
        """
        Return the normal.

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the normal.
        
        normalize : `boolean`, optional
            If True the returned normal will have unit length (the default). Otherwise, the normal's length will
            be the area of the tangent space (for two independent variables, its the length of the cross product of tangent vectors).
        
        indices : `iterable`, optional
            An iterable of normal indices to calculate. For example, `indices=(0, 3)` will return a vector of length 2
            with the first and fourth values of the normal. If `None`, all normal values are returned (the default).

        Returns
        -------
        normal : `numpy.array`
        """
        return self.normalDirection * self.spline.normal(domainPoint, normalize, indices)

    def evaluate(self, domainPoint):
        """
        Return the value of the manifold (a point on the manifold).

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the point.

        Returns
        -------
        point : `numpy.array`
        """
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

    def transform(self, matrix, matrixInverseTranspose = None):
        """
        Transform the range of the spline.

        Parameters
        ----------
        matrix : `numpy.array`
            A square matrix transformation.

        matrixInverseTranspose : `numpy.array`, optional
            The inverse transpose of matrix (computed if not provided).

        Returns
        -------
        spline : `BSpline`
            The transformed spline.

        See Also
        --------
        `solid.Solid.transform` : Transform the range of the solid.
        """
        assert np.shape(matrix) == (self.range_dimension(), self.range_dimension())
        spline = BSpline(self.spline.transform(matrix))
        spline.normalDirection = self.normalDirection
        if hasattr(self, "material"):
            spline.material = self.material
        return spline
        
    def translate(self, delta):
        """
        translate the range of the spline.

        Parameters
        ----------
        delta : `numpy.array`
            A 1D array translation.

        Returns
        -------
        spline : `BSpline`
            The translated spline.

        See Also
        --------
        `solid.Solid.translate` : translate the range of the solid.
        """
        assert len(delta) == self.range_dimension()
        spline = BSpline(self.spline.translate(delta))
        spline.normalDirection = self.normalDirection
        if hasattr(self, "material"):
            spline.material = self.material
        return spline

    def negate_normal(self):
        """
        Negate the direction of the normal.

        Returns
        -------
        spline : `BSpline`
            The spline with negated normal. The spline retains the same tangent space.

        See Also
        --------
        `solid.Solid.complement` : Return the complement of the solid: whatever was inside is outside and vice-versa.
        """
        spline = BSpline(self.spline)
        spline.normalDirection = -1.0 * self.normalDirection
        if hasattr(self, "material"):
            spline.material = self.material
        return spline

    def intersect(self, other):
        """
        Intersect a spline or hyperplane.

        Parameters
        ----------
        other : `Spline` or `Hyperplane`
            The `Manifold` intersecting the `Spline`.

        Returns
        -------
        intersections : `list` (or `NotImplemented` if other is not a `Hyperplane` nor a `Spline`)
            A list of intersections between the two manifolds. 
            Each intersection records either a crossing or a coincident region.
            Coincident regions are currently not implemented for splines.

            For a crossing, intersection is a `Manifold.Crossing`: (firstPart, secondPart)
            * firstPart : `Manifold` in the manifold's domain where the manifold and the other cross.
            * secondPart : `Manifold` in the other's domain where the manifold and the other cross.
            * Both intersection manifolds have the same domain and range (the crossing between the manifold and the other).

            For a coincident region, intersection is a `Manifold.Coincidence`: (firstPart, secondPart, alignment, transform, inverse, translation)
            * firstPart : `Solid` in the manifold's domain within which the manifold and the other are coincident.
            * secondPart : `Solid` in the other's domain within which the manifold and the other are coincident.
            * alignment : scalar value holding the normal alignment between the manifold and the other (the dot product of their unit normals).
            * transform : `numpy.array` holding the transform matrix from the manifold's domain to the other's domain.
            * inverse : `numpy.array` holding the inverse transform matrix from the other's domain to the boundary's domain.
            * translation : `numpy.array` holding the translation vector from the manifold's domain to the other's domain.
            * Together transform, inverse, and translation form the mapping from the manifold's domain to the other's domain and vice-versa.

        See Also
        --------
        `solid.Solid.compute_cutout` : Compute the cutout portion of a manifold within a solid.
        `bspy.Spline.zeros` : Find the roots of a spline (nInd must match nDep).
        `bspy.Spline.contours` : Find all the contour curves of a spline whose nInd is one larger than its nDep.

        Notes
        -----
        This method basically wraps the `bspy.Spline.zeros` and `bspy.Spline.contours` calls. We construct a spline that represents the 
        intersection and then call zeros or contours, depending on the dimension. The only subtly is getting the normals of intersections to always 
        point outward. We do that by picking an intersection point, checking the normal direction, and negating it as needed.
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
                        firstPart = Solid(nDep, False)
                        firstPart.boundaries.append(Boundary(Hyperplane(-1.0, zero[0], 0.0), Solid(0, True)))
                        firstPart.boundaries.append(Boundary(Hyperplane(1.0, zero[1], 0.0), Solid(0, True)))
                        secondPart = Solid(nDep, False)
                        if planeBounds[0] > planeBounds[1]:
                            planeBounds = (planeBounds[1], planeBounds[0])
                        secondPart.boundaries.append(Boundary(Hyperplane(-1.0, planeBounds[0], 0.0), Solid(0, True)))
                        secondPart.boundaries.append(Boundary(Hyperplane(1.0, planeBounds[1], 0.0), Solid(0, True)))
                        alignment = np.dot(self.normal((zero[0],)), other._normal) # Use the first zero, since B-splines are closed on the left
                        width = zero[1] - zero[0]
                        transform = (planeBounds[1] - planeBounds[0]) / width
                        translation = (planeBounds[0] * zero[1] - planeBounds[1] * zero[0]) / width
                        intersections.append(Manifold.Coincidence(firstPart, secondPart, alignment, np.atleast_2d(transform), np.atleast_2d(1.0 / transform), np.atleast_1d(translation)))
                    else:
                        # Intersection is a point, so create a Manifold.Crossing.
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero, 0.0), Hyperplane(1.0, projection @ (self.spline((zero,)) - other._point), 0.0)))

            # Surface-Plane intersection.
            elif nDep == 2:
                # Find the intersection contours, which are returned as splines.
                contours = spline.contours()
                # Convert each contour into a Manifold.Crossing.
                for contour in contours:
                    # The firstPart portion is the contour returned for the spline-plane intersection. 
                    firstPart = contour
                    # The secondPart portion is the contour projected onto the plane's domain, which we compute with samples and a least squares fit.
                    tValues = np.linspace(0.0, 1.0, contour.nCoef[0] + 5) # Over-sample a bit to reduce the condition number and avoid singular matrix
                    points = []
                    for t in tValues:
                        zero = contour((t,))
                        points.append(projection @ (self.spline(zero) - other._point))
                    secondPart = Spline.least_squares(tValues, np.array(points).T, contour.order, contour.knots)
                    intersections.append(Manifold.Crossing(BSpline(firstPart), BSpline(secondPart)))
            else:
                return NotImplemented
        
        # Spline-Spline intersection.
        elif isinstance(other, BSpline):
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
                        firstPart = Solid(nDep, False)
                        firstPart.boundaries.append(Boundary(Hyperplane(-1.0, zero[0][0], 0.0), Solid(0, True)))
                        firstPart.boundaries.append(Boundary(Hyperplane(1.0, zero[1][0], 0.0), Solid(0, True)))
                        secondPart = Solid(nDep, False)
                        secondPart.boundaries.append(Boundary(Hyperplane(-1.0, zero[0][1], 0.0), Solid(0, True)))
                        secondPart.boundaries.append(Boundary(Hyperplane(1.0, zero[1][1], 0.0), Solid(0, True)))
                        alignment = np.dot(self.normal(zero[0][0]), other.normal(zero[0][1])) # Use the first zeros, since B-splines are closed on the firstPart
                        width = zero[1][0] - zero[0][0]
                        transform = (zero[1][1] - zero[0][1]) / width
                        translation = (zero[0][1] * zero[1][0] - zero[1][1] * zero[0][0]) / width
                        intersections.append(Manifold.Coincidence(firstPart, secondPart, alignment, np.atleast_2d(transform), np.atleast_2d(1.0 / transform), np.atleast_1d(translation)))
                    else:
                        # Intersection is a point, so create a Manifold.Crossing.
                        intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[:nDep], 0.0), Hyperplane(1.0, zero[nDep:], 0.0)))
            
            # Surface-Surface intersection.
            elif nDep == 2:
                logging.info(f"intersect({self.spline.metadata['Name']}, {other.spline.metadata['Name']})")
                # Find the intersection contours, which are returned as splines.
                swap = False
                try:
                    # First try the intersection as is.
                    contours = spline.contours()
                except ValueError:
                    # If that fails, swap the manifolds. Worth a shot since intersections are touchy.
                    swap = True

                # Convert each contour into a Manifold.Crossing.
                if swap:
                    spline = other.spline.subtract(self.spline)
                    logging.info(f"intersect({other.spline.metadata['Name']}, {self.spline.metadata['Name']})")
                    contours = spline.contours()
                    for contour in contours:
                        # Swap firstPart and secondPart, compared to not swapped.
                        firstPart = Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[nDep:], contour.metadata)
                        secondPart = Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[:nDep], contour.metadata)
                        intersections.append(Manifold.Crossing(BSpline(firstPart), BSpline(secondPart)))
                else:
                    for contour in contours:
                        firstPart = Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[:nDep], contour.metadata)
                        secondPart = Spline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[nDep:], contour.metadata)
                        intersections.append(Manifold.Crossing(BSpline(firstPart), BSpline(secondPart)))
            else:
                return NotImplemented
        else:
            return NotImplemented

        # Ensure the normals point outwards for both Manifolds in each crossing intersection.
        # Note that evaluating firstPart and secondPart at 0.5 is always valid because either they are points or curves with [0.0, 1.0] domains.
        domainPoint = np.atleast_1d(0.5)
        for i, intersection in enumerate(intersections):
            if isinstance(intersection, Manifold.Crossing):
                firstPart = intersection.firstPart
                secondPart = intersection.secondPart
                if np.dot(self.tangent_space(firstPart.evaluate(domainPoint)) @ firstPart.normal(domainPoint), other.normal(secondPart.evaluate(domainPoint))) < 0.0:
                    firstPart = firstPart.negate_normal()
                if np.dot(other.tangent_space(secondPart.evaluate(domainPoint)) @ secondPart.normal(domainPoint), self.normal(firstPart.evaluate(domainPoint))) < 0.0:
                    secondPart = secondPart.negate_normal()
                intersections[i] = Manifold.Crossing(firstPart, secondPart)

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
        `solid.Solid.compute_cutout` : Compute the cutout portion of a manifold within a solid.
        `complete_cutout` : Add any missing inherent (implicit) boundaries of this manifold's domain to the given cutout.

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
                BSpline.establish_domain_bounds(domainDomain1, np.delete(bounds, i, axis=0))
                domainDomain2 = Solid(dimension - 1, False)
                BSpline.establish_domain_bounds(domainDomain2, np.delete(bounds, i, axis=0))
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

    def complete_cutout(self, cutout, solid):
        """
        Add any missing inherent (implicit) boundaries of this manifold's domain 
        that are needed to make the cutout valid and complete.

        Parameters
        ----------
        cutout : `solid.Solid`
            The cutout portion of this manifold's domain that lies within the given solid. 
            The cutout may be incomplete, missing some of the manifold's inherent domain boundaries. 
            Its dimension must match `self.domain_dimension()`.

        solid : `solid.Solid`
            The solid determining the cutout of the manifold. Its dimension must match `self.range_dimension()`.

        See Also
        --------
        `establish_domain_bounds` : Establish the outer bounds of a spline's domain.
        `solid.Solid.compute_cutout` : Compute the cutout portion of a manifold within a solid.

        Notes
        -----
        A spline's inherent domain is determined by its knot array for each dimension. 
        """
        assert self.domain_dimension() == cutout.dimension
        assert self.range_dimension() == solid.dimension
        assert cutout.dimension == 1 or cutout.dimension == 2

        # Spline manifold domains have finite bounds.
        cutout.containsInfinity = False
        bounds = self.spline.domain()

        # If manifold (self) has no intersections with solid, just check containment.
        if not cutout.boundaries:
            if cutout.dimension == 2:
                logging.info(f"check containment: {self.spline.metadata['Name']}")
            domain = self.spline.domain().T
            if solid.contains_point(self.spline(0.5 * (domain[0] + domain[1]))):
                self.establish_domain_bounds(cutout, bounds)
            return

        # For curves, add domain bounds as needed.
        if cutout.dimension == 1:
            cutout.boundaries.sort(key=lambda b: (b.manifold.evaluate(0.0), b.manifold.normal(0.0)))
            if abs(cutout.boundaries[0].manifold._point - bounds[0][0]) >= Manifold.minSeparation and \
                cutout.boundaries[0].manifold._normal > 0.0:
                cutout.boundaries.insert(0, Boundary(Hyperplane(-cutout.boundaries[0].manifold._normal, bounds[0][0], 0.0), Solid(0, True)))
            if abs(cutout.boundaries[-1].manifold._point - bounds[0][1]) >= Manifold.minSeparation and \
                cutout.boundaries[-1].manifold._normal < 0.0:
                cutout.boundaries.append(Boundary(Hyperplane(-cutout.boundaries[-1].manifold._normal, bounds[0][1], 0.0), Solid(0, True)))

        # For surfaces, add bounding box for domain and intersect it with existing cutout boundaries.
        if cutout.dimension == 2:
            boundaryCount = len(cutout.boundaries) # Keep track of existing cutout boundaries
            self.establish_domain_bounds(cutout, bounds) # Add bounding box boundaries to cutout boundaries
            for boundary in cutout.boundaries[boundaryCount:]: # Mark bounding box boundaries as untouched
                boundary.touched = False

            # Define function for adding cutout points to new bounding box boundaries.
            def process_domain_point(boundary, domainPoint):
                point = boundary.manifold.evaluate(domainPoint)
                # See if and where point touches bounding box of cutout.
                for newBoundary in cutout.boundaries[boundaryCount:]:
                    vector = point - newBoundary.manifold._point
                    if abs(np.dot(newBoundary.manifold._normal, vector)) < Manifold.minSeparation:
                        # Add the point onto the new boundary.
                        normal = np.sign(newBoundary.manifold._tangentSpace.T @ boundary.manifold.normal(domainPoint))
                        newBoundary.domain.boundaries.append(Boundary(Hyperplane(normal, newBoundary.manifold._tangentSpace.T @ vector, 0.0), Solid(0, True)))
                        newBoundary.touched = True
                        break

            # Go through existing boundaries and check if either of their endpoints lies on the spline's bounds.
            for boundary in cutout.boundaries[:boundaryCount]:
                domainBoundaries = boundary.domain.boundaries
                domainBoundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), boundary.manifold.normal(0.0)))
                process_domain_point(boundary, domainBoundaries[0].manifold._point)
                if len(domainBoundaries) > 1:
                    process_domain_point(boundary, domainBoundaries[-1].manifold._point)
            
            # For touched boundaries, remove domain bounds that aren't needed.
            boundaryWasTouched = False
            for newBoundary in cutout.boundaries[boundaryCount:]:
                if newBoundary.touched:
                    boundaryWasTouched = True
                    domainBoundaries = newBoundary.domain.boundaries
                    assert len(domainBoundaries) > 2
                    domainBoundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), boundary.manifold.normal(0.0)))
                    # Ensure domain endpoints don't overlap and their normals are consistent.
                    if abs(domainBoundaries[0].manifold._point - domainBoundaries[1].manifold._point) < Manifold.minSeparation or \
                        domainBoundaries[1].manifold._normal < 0.0:
                        del domainBoundaries[0]
                    if abs(domainBoundaries[-1].manifold._point - domainBoundaries[-2].manifold._point) < Manifold.minSeparation or \
                        domainBoundaries[-2].manifold._normal > 0.0:
                        del domainBoundaries[-1]
            
            if boundaryWasTouched:
                # Touch untouched boundaries that are connected to touched boundary endpoints.
                boundaryMap = ((2, 3, 0), (2, 3, -1), (0, 1, 0), (0, 1, -1)) # Map of which bounding box boundaries touch each other
                while True:
                    noTouches = True
                    for map, newBoundary, bound in zip(boundaryMap, cutout.boundaries[boundaryCount:], bounds.flatten()):
                        if not newBoundary.touched:
                            firstPartBoundary = cutout.boundaries[boundaryCount + map[0]]
                            secondPartBoundary = cutout.boundaries[boundaryCount + map[1]]
                            if firstPartBoundary.touched and abs(firstPartBoundary.domain.boundaries[map[2]].manifold._point - bound) < Manifold.minSeparation:
                                newBoundary.touched = True
                                noTouches = False
                            elif secondPartBoundary.touched and abs(secondPartBoundary.domain.boundaries[map[2]].manifold._point - bound) < Manifold.minSeparation:
                                newBoundary.touched = True
                                noTouches = False
                    if noTouches:
                        break
                
                # Remove untouched boundaries.
                i = boundaryCount
                while i < len(cutout.boundaries):
                    if not cutout.boundaries[i].touched:
                        del cutout.boundaries[i]
                    else:
                        i += 1
            else:
                # No cutout boundaries touched the bounding box, so remove bounding box if it's not contained in the solid.
                if not solid.contains_point(self.evaluate(bounds[:,0])):
                    cutout.boundaries = cutout.boundaries[:boundaryCount]