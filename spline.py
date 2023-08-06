import numpy as np
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
        return Spline(type(self.spline)(self.spline.nInd, self.spline.nDep, self.spline.order, self.spline.nCoef, self.spline.knots, self.spline.coefs, self.spline.accuracy))

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

    def transform(self, transform, transformInverseTranspose = None):
        """
        transform the range of the hyperplane.

        Parameters
        ----------
        transform : `numpy.array`
            A square 2D array transformation.

        transformInverseTranspose : `numpy.array`, optional
            The inverse transpose of transform (computed if not provided).

        Notes
        -----
        Transforms the hyperplane in place, so create a copy as needed.

        See Also
        --------
        `solid.Solid.transform` : transform the range of the solid.
        """
        assert np.shape(transform) == (self.range_dimension(), self.range_dimension())
        self.spline = self.spline.transform(transform)
        
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
        coefs = np.delete(self.spline.coefs, 0, axis=0)
        coefs[:] -= point[1:]
        spline = BspySpline(self.spline.nInd, self.spline.nDep - 1, self.spline.order, self.spline.nCoef, self.spline.knots, coefs)
        zeros = spline.zeros()

        # Generate list of intersections.
        intersections = []
        for zero in zeros:
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