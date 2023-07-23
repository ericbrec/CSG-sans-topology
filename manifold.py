import numpy as np
from collections import namedtuple

class Manifold:
    """
    A manifold is an abstract base class for differentiable functions with
    normals and tangent spaces whose range is one dimension higher than their domain.
    """

    # If two points are within 0.01 of each each other, they are coincident
    minSeparation = 0.01

    # If a shift of 1 in the normal direction of one manifold yields a shift of 10 in the tangent plane intersection, the manifolds are parallel
    maxAlignment = 0.99 # 1 - 1/10^2

    # Return type for IntersectXRay
    RayCrossing = namedtuple('RayCrossing', ('distance','domainPoint'))

    # Return type for IntersectManifold
    Crossing = namedtuple('Crossing', ('left','right'))
    Coincidence = namedtuple('Coincidence', ('left', 'right', 'alignment', 'transform', 'inverse', 'translation'))

    def __init__(self):
        pass

    def copy(self):
        """
        Copy the manifold.

        Returns
        -------
        manifold : `Manifold`
        """
        return None

    def DomainDimension(self):
        """
        Return the domain dimension.

        Returns
        -------
        dimension : `int`
        """
        return self.RangeDimension() - 1

    def RangeDimension(self):
        """
        Return the range dimension.

        Returns
        -------
        dimension : `int`
        """
        return 0

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
        return None

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
        return None

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
        return None

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
        return None

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
        return 0.0

    def Determinant(self, domainPoint):
        """
        Returns the determinant, which is the length of the cofactor normal (also the normal dotted with the cofactor normal).

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the first cofactor.

        Returns
        -------
        determinant : scalar
        """
        return np.dot(self.Normal(domainPoint), self.CofactorNormal(domainPoint))

    def Transform(self, transform, transformInverseTranspose = None):
        """
        Transform the range of the manifold.

        Parameters
        ----------
        transform : `numpy.array`
            A square 2D array transformation.

        transformInverseTranspose : `numpy.array`, optional
            The inverse transpose of transform (computed if not provided).

        Notes
        -----
        Transforms the manifold in place, so create a copy as needed.

        See Also
        --------
        `solid.Solid.Transform` : Transform the range of the solid.
        """
        assert np.shape(transform) == (self.RangeDimension(), self.RangeDimension())

    def Translate(self, delta):
        """
        Translate the range of the manifold.

        Parameters
        ----------
        delta : `numpy.array`
            A 1D array translation.

        Notes
        -----
        Translates the manifold in place, so create a copy as needed.

        See Also
        --------
        `solid.Solid.Translate` : Translate the range of the solid.
        """
        assert len(delta) == self.RangeDimension()

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
        pass

    def IntersectXRay(self, point):
        """
        Intersect a ray along the x-axis with the manifold.

        Parameters
        ----------
        point : array-like
            The starting point of the ray.

        Returns
        -------
        intersections : `list`
            A list of intersections between the ray and the manifold. 
            Each intersection is a Manifold.RayCrossing: (distance to intersection, domain point of intersection).

        See Also
        --------
        `solid.Solid.WindingNumber` : Compute the winding number for a point relative to the solid.
        """
        assert len(point) == self.RangeDimension()
        return []

    def IntersectManifold(self, other):
        """
        Intersect two manifolds.

        Parameters
        ----------
        other : `Manifold`
            The `Manifold` intersecting the manifold.

        Returns
        -------
        intersections : `list` (Or `NotImplemented` if other is an unknown type of Manifold)
            A list of intersections between the two manifolds. 
            Each intersection records either a crossing or a coincident region.

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

        Notes
        -----
        To invert the mapping to go from the other's domain to the manifold's domain, you first subtract the translation and then multiply by the inverse of the transform.
        """
        return NotImplemented