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
        spline = Spline(type(self.spline)(self.spline.nInd, self.spline.nDep, self.spline.order, self.spline.nCoef, self.spline.knots, self.spline.coefs, self.spline.accuracy))
        spline.normalDirection = self.normalDirection
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
            A square 2D array transformation.

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
        `solid.Solid.compliment` : Return the compliment of the solid: whatever was inside is outside and vice-versa.
        """
        self.normalDirection *= -1.0

    def intersect_x_ray(self, point):
        """
        Intersect a ray along the x-axis with the spline.

        Parameters
        ----------
        point : array-like
            The starting point of the ray.

        Returns
        -------
        intersections : `list`
            A list of intersections between the ray and the spline. 
            Each intersection is a Manifold.RayCrossing: (distance to intersection, domain point of intersection).

        See Also
        --------
        `solid.Solid.winding_number` : Compute the winding number for a point relative to the solid.
        """
        assert len(point) == self.range_dimension()
        # Construct a lower range spline whose zeros are the ray intersection points.
        coefs = np.delete(self.spline.coefs, 0, axis=0).T
        coefs -= point[1:]
        spline = BspySpline(self.spline.nInd, self.spline.nDep - 1, self.spline.order, self.spline.nCoef, self.spline.knots, coefs.T)
        zeros = spline.zeros()

        # Generate list of intersections.
        intersections = []
        for zero in zeros:
            zero = np.atleast_1d(zero)
            intersections.append(Manifold.RayCrossing(self.spline(zero)[0] - point[0], zero))

        return intersections

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
            * transform : `numpy.array` holding the 2D transform from the boundary's domain to the other's domain.
            * inverse : `numpy.array` holding the 2D inverse transform from the other's domain to the boundary's domain.
            * translation : `numpy.array` holding the 1D translation from the manifold's domain to the other's domain.
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
        nDep = self.spline.nInd
        if isinstance(other, Hyperplane):
            # Compute the inverse of the tangent space to map Spline-Hyperplane intersection points to the domain of the Hyperplane.
            inverseTangentSpace = np.linalg.inv(other._tangentSpace.T @ other._tangentSpace)
            # Construct a new spline that represents the intersection.
            spline = self.spline.dot(other._normal) - np.atleast_1d(np.dot(other._normal, other._point))
            if nDep == 1:
                # Find the intersection points.
                zeros = spline.zeros()
                # Convert each point into a Manifold.Crossing.
                for zero in zeros:
                    zero = np.atleast_1d(zero)
                    intersections.append(Manifold.Crossing(Hyperplane(1.0, zero, 0.0), Hyperplane(1.0, inverseTangentSpace @ other._tangentSpace.T @ (self.spline(zero) - other._point), 0.0)))
            elif nDep == 2:
                # Find the intersection contours, which are returned as splines.
                contours = spline.contours()
                # Convert each contour into a Manifold.Crossing.
                for contour in contours:
                    left = contour
                    points = []
                    for t in np.linspace(0.0, 1.0, contour.nCoef[0]):
                        zero = contour((t,))
                        points.append((t, *(inverseTangentSpace @ other._tangentSpace.T @ (self.spline(zero) - other._point))))
                    right = BspySpline.least_squares(contour.nInd, nDep, contour.order, points, contour.knots, 0, contour.metadata)
                    intersections.append(Manifold.Crossing(Spline(left), Spline(right)))
            else:
                return NotImplemented
        elif isinstance(other, Spline):
            # Construct a new spline that represents the intersection.
            spline = self.spline - other.spline
            if nDep == 1:
                # Find the intersection points.
                zeros = spline.zeros()
                # Convert each point into a Manifold.Crossing.
                for zero in zeros:
                    intersections.append(Manifold.Crossing(Hyperplane(1.0, zero[:nDep], 0.0), Hyperplane(1.0, zero[nDep:], 0.0)))
            elif nDep == 2:
                # Find the intersection contours, which are returned as splines.
                contours = spline.contours()
                # Convert each contour into a Manifold.Crossing.
                for contour in contours:
                    left = BspySpline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[:nDep], contour.accuracy, contour.metadata)
                    right = BspySpline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[nDep:], contour.accuracy, contour.metadata)
                    intersections.append(Manifold.Crossing(Spline(left), Spline(right)))
            else:
                return NotImplemented
        else:
            return NotImplemented

        # Ensure the normals point outwards for both Manifolds in each intersection.
        # Note that evaluating left and right at 0.0 is always valid because either they are points or curves with [0.0, 1.0] domains.
        domainPoint = np.atleast_1d(0.0)
        for intersection in intersections:
            if np.dot(self.tangent_space(intersection.left.point(domainPoint)) @ intersection.left.normal(domainPoint), other.normal(intersection.right.point(domainPoint))) < 0.0:
                intersection.left.flip_normal()
            if np.dot(other.tangent_space(intersection.right.point(domainPoint)) @ intersection.right.normal(domainPoint), self.normal(intersection.left.point(domainPoint))) < 0.0:
                intersection.right.flip_normal()

        return intersections

    def complete_domain(self, domain):
        """
        Add any missing inherent (implicit) boundaries of this manifold to the given domain that are needed to make the domain valid and complete.

        Parameters
        ----------
        domain : `solid.Solid`
            A domain for this manifold that may be incomplete, missing some of the manifold's inherent domain boundaries. 
            Its dimension must match `self.domain_dimension()`.

        See Also
        --------
        `solid.Solid.slice` : slice the solid by a manifold.

        Notes
        -----
        A spline's inherent domain is determined by its knot array for each dimension. 
        If you pass in an empty domain, it will remain empty.
        """
        dimension = self.domain_dimension()
        assert domain.dimension == dimension
        assert domain.dimension == 1 or domain.dimension == 2
        bounds = self.spline.domain()

        if len(domain.boundaries) == 0:
            return domain

        # Direction of new boundaries
        boundaryDirection = -1.0 if domain.containsInfinity else 1.0
        
        # For curves, add endpoints of domain as needed.
        if domain.dimension == 1:
            domain.boundaries.sort(key=lambda b: (b.manifold.point(0.0), b.manifold.normal(0.0)))
            domainDomain = Solid(0, True) # Domain for 1D points.
            if boundaryDirection * domain.boundaries[0].manifold._normal > 0.0:
                domain.boundaries.insert(0, Boundary(Hyperplane(-boundaryDirection, bounds[0][0], 0.0), domainDomain))
            if boundaryDirection * domain.boundaries[-1].manifold._normal < 0.0:
                domain.boundaries.append(Boundary(Hyperplane(boundaryDirection, bounds[0][1], 0.0), domainDomain))
        
        # For surfaces, add bounding box of domain based on where partial domain touches it.
        elif domain.dimension == 2:
            newBoundaries = [None] * bounds.size # Cache for new domain boundaries
            pointDomain = Solid(0, True) # Domain for 1D points.

            # Function for adding domain points to new boundaries.
            def process_domain_point(boundary, domainPoint):
                point = boundary.manifold.point(domainPoint)
                # See if and where point touches bounding box of domain.
                for i in range(domain.dimension):
                    for j in range(2):
                        if abs(point[i] - bounds[i][j]) < Manifold.minSeparation:
                            index = i * 2 + j
                            newBoundary = newBoundaries[index] # Lookup new boundary in cache
                            if newBoundary is None:
                                # Boundary doesn't exist, so create it and add it to the cache and the domain
                                direction = boundaryDirection * (-1.0 if j == 0 else 1.0)
                                unitVector = np.array((0.0, 0.0))
                                unitVector[i] = 1.0
                                tangent = np.array((0.0, 0.0))
                                tangent[1-i] = 1.0
                                newBoundary = Boundary(Hyperplane(direction * unitVector, bounds[i][j] * unitVector, tangent), Solid(1, False))
                                newBoundaries[index] = newBoundary
                                domain.boundaries.append(newBoundary)
                            # Now add the point onto the new boundary, with a direction based on the point's normal and the domain containing infinity
                            normal = boundary.manifold.normal(domainPoint)
                            newBoundary.domain.boundaries.append(Boundary(Hyperplane(boundaryDirection * np.sign(normal[1-i]), point[1-i], 0.0), pointDomain))

            # Go through each boundary and check if either of its endpoints lies on the spline's bounds.
            # We use a while loop, because endpoints may add new boundaries.
            b = 0
            lenBoundariesOriginally = len(domain.boundaries)
            while b < len(domain.boundaries):
                boundary = domain.boundaries[b]
                domainDomain = boundary.domain
                if len(domainDomain.boundaries) > 0:
                    domainDomain.boundaries.sort(key=lambda boundary: (boundary.manifold.point(0.0), boundary.manifold.normal(0.0)))
                    if b >= lenBoundariesOriginally:
                        # New boundary, so let's complete its own domain first (basically the same code as for curves).
                        for i in range(domain.dimension):
                            if abs(boundary.manifold._normal[i]) > Manifold.minSeparation:
                                break
                        if domainDomain.boundaries[0].manifold._normal > 0.0:
                            domainDomain.boundaries.insert(0, Boundary(Hyperplane(-1.0, bounds[1-i][0], 0.0), pointDomain))
                        if domainDomain.boundaries[-1].manifold._normal < 0.0:
                            domainDomain.boundaries.append(Boundary(Hyperplane(1.0, bounds[1-i][1], 0.0), pointDomain))
                    # Process the boundary's first point.
                    process_domain_point(boundary, domainDomain.boundaries[0]._point)
                    # Process the boundary's last point.
                    if len(domainDomain.boundaries) > 1:
                        process_domain_point(boundary, domainDomain.boundaries[-1]._point)
                b += 1

        return domain