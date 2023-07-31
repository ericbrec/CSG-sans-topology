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

    def RangeDimension(self):
        """
        Return the range dimension.

        Returns
        -------
        dimension : `int`
        """
        return self.spline.nDep

    def Normal(self, domainPoint):
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

    def Point(self, domainPoint):
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

    def TangentSpace(self, domainPoint):
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
        return self.tangentSpace

    def CofactorNormal(self, domainPoint):
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
        `solid.Solid.VolumeIntegral` : Compute the volume integral of a function within the solid.
        `solid.Solid.SurfaceIntegral` : Compute the surface integral of a vector field on the boundary of the solid.
        """
        return self.normalDirection * self.spline.normal(domainPoint, False)

    def FirstCofactor(self, domainPoint):
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
        `solid.Solid.VolumeIntegral` : Compute the volume integral of a function within the solid.
        `solid.Solid.SurfaceIntegral` : Compute the surface integral of a vector field on the boundary of the solid.
        """
        return self.normalDirection * self.spline.normal(domainPoint, False, (0,))

    def Transform(self, transform, transformInverseTranspose = None):
        """
        Transform the range of the hyperplane.

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
        `solid.Solid.Transform` : Transform the range of the solid.
        """
        assert np.shape(transform) == (self.RangeDimension(), self.RangeDimension())
        self.spline = self.spline.transform(transform)
        
    def Translate(self, delta):
        """
        Translate the range of the hyperplane.

        Parameters
        ----------
        delta : `numpy.array`
            A 1D array translation.

        Notes
        -----
        Translates the hyperplane in place, so create a copy as needed.

        See Also
        --------
        `solid.Solid.Translate` : Translate the range of the solid.
        """
        assert len(delta) == self.RangeDimension()
        self.spline = self.spline.translate(delta)

    def FlipNormal(self):
        """
        Flip the direction of the normal.

        Notes
        -----
        Negates the normal in place, so create a copy as needed.

        See Also
        --------
        `solid.Solid.Not` : Return the compliment of the solid: whatever was inside is outside and vice-versa.
        """
        self.normalDirection *= -1.0

    def IntersectXRay(self, point):
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
        `solid.Solid.WindingNumber` : Compute the winding number for a point relative to the solid.
        """
        assert len(point) == self.RangeDimension()
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

    def IntersectManifold(self, other):
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
        `solid.Solid.Slice` : Slice the solid by a manifold.
        `numpy.linalg.svd` : Compute the singular value decomposition of a 2D array.
        `bspy.Spline.contours` : Find all the contour curves of a spline whose nInd is one larger than its nDep.

        Notes
        -----
        This method basically wraps the `bspy.Spline.contours` call. We construct a spline that represents the 
        intersection and then call contours. The only subtly is getting the normal of the contours to always 
        points outward. We do that by picking a point on the contour, checking the normal direction, and 
        flipping it as needed.
        """
        assert self.RangeDimension() == other.RangeDimension()

        if isinstance(other, Hyperplane):
            # Construct a new spline that represents the intersection.
            spline = self.spline.dot(other.normal) - np.dot(other.normal, other.point)
            # Find the intersection contours, which are returned as splines.
            contours = spline.contours()
            # Convert each contour into a Manifold.Crossing.
            intersections = []
            nDep = self.spline.nInd
            for contour in contours:
                left = contour
                # TODO: Construct the right spline.
                right = BspySpline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[nDep:], contour.accuracy, contour.metadata)
                intersections.append(Manifold.Crossing(Spline(left), Spline(right)))
        elif isinstance(other, Spline):
            # Construct a new spline that represents the intersection.
            spline = self.spline - other.spline
            # Find the intersection contours, which are returned as splines.
            contours = spline.contours()
            # Convert each contour into a Manifold.Crossing.
            intersections = []
            nDep = self.spline.nInd
            for contour in contours:
                left = BspySpline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[:nDep], contour.accuracy, contour.metadata)
                right = BspySpline(contour.nInd, nDep, contour.order, contour.nCoef, contour.knots, contour.coefs[nDep:], contour.accuracy, contour.metadata)
                intersections.append(Manifold.Crossing(Spline(left), Spline(right)))
        else:
            return NotImplemented

        # Ensure the normal of each Spline points outward.

        return intersections