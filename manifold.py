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

    # Return type for intersect_x_ray
    RayCrossing = namedtuple('RayCrossing', ('distance','domainPoint'))

    # Return type for intersect_manifold
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

    def domain_dimension(self):
        """
        Return the domain dimension.

        Returns
        -------
        dimension : `int`
        """
        return self.range_dimension() - 1

    def range_dimension(self):
        """
        Return the range dimension.

        Returns
        -------
        dimension : `int`
        """
        return 0

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
        return None

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
        return None

    def any_point(self):
        """
        Return an arbitrary point on the manifold.

        Returns
        -------
        point : `numpy.array`
            A point on the manifold.

        See Also
        --------
        `Solid.any_point` : Return an arbitrary point on the solid.
        `Boundary.any_point` : Return an arbitrary point on the boundary.

        Notes
        -----
        The any_point method for solids and boundaries do not call this method, because the point returned 
        may not be within the solid or boundary.
        """
        return None

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
        return None

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
        return None

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
        return 0.0

    def determinant(self, domainPoint):
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
        return np.dot(self.normal(domainPoint), self.cofactor_normal(domainPoint))

    def transform(self, matrix, matrixInverseTranspose = None):
        """
        Transform the range of the manifold.

        Parameters
        ----------
        matrix : `numpy.array`
            A square 2D array transformation.

        matrixInverseTranspose : `numpy.array`, optional
            The inverse transpose of matrix (computed if not provided).

        Notes
        -----
        Transforms the manifold in place, so create a copy as needed.

        See Also
        --------
        `solid.Solid.transform` : transform the range of the solid.
        """
        assert np.shape(matrix) == (self.range_dimension(), self.range_dimension())

    def translate(self, delta):
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
        `solid.Solid.translate` : translate the range of the solid.
        """
        assert len(delta) == self.range_dimension()

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
        pass

    def intersect_x_ray(self, point):
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
        `solid.Solid.winding_number` : Compute the winding number for a point relative to the solid.
        """
        assert len(point) == self.range_dimension()
        return []

    def intersect_manifold(self, other):
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
        `cached_intersect_manifold` : Intersect two manifolds, caching the result for twins (same intersection but swapping self and other).
        `solid.Solid.slice` : slice the solid by a manifold.

        Notes
        -----
        To invert the mapping to go from the other's domain to the manifold's domain, you first subtract the translation and then multiply by the inverse of the transform.
        """
        return NotImplemented

    def cached_intersect_manifold(self, other, cache = None):
        """
        Intersect two manifolds, caching the result for twins (same intersection but swapping self and other).

        Parameters
        ----------
        other : `Manifold`
            The `Manifold` intersecting the manifold.
        
        cache : `dict`, optional
            A dictionary to cache `Manifold` intersections, speeding computation. The default is `None`.

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

        isTwin : `bool`
            True if this intersection is the twin from the cache (the intersection with self and other swapped).

        See Also
        --------
        `intersect_manifold` : Intersect two manifolds.
        `solid.Solid.slice` : slice the solid by a manifold.

        Notes
        -----
        To invert the mapping to go from the other's domain to the manifold's domain, you first subtract the translation and then multiply by the inverse of the transform.
        """
        intersections = None
        isTwin = False
        # Check cache for previously computed manifold intersections.
        if cache is not None:
            # First, check for the twin (opposite order of arguments).
            intersections = cache.get((other, self))
            if intersections is not None:
                isTwin = True
            else:
                # Next, check for the original order (not twin).
                intersections = cache.get((self, other))

        # If intersections not previously computed, compute them now.
        if intersections is None:
            intersections = self.intersect_manifold(other)
            if intersections is NotImplemented:
                # Try the other way around in case other knows how to intersect self.
                intersections = other.intersect_manifold(self)
                isTwin = True
            # Store intersections in cache.
            if cache is not None:
                if isTwin:
                    cache[(other, self)] = intersections
                else:
                    cache[(self, other)] = intersections
        
        return intersections, isTwin

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
        For manifolds without inherent domain boundaries (like hyperplanes), the operation does nothing.
        """
        assert self.domain_dimension() == domain.dimension
