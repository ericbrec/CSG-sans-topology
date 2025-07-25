import numpy as np
from collections import namedtuple

class Manifold:
    """
    A manifold is an abstract base class for differentiable functions with
    normals and tangent spaces whose range is one dimension higher than their domain.
    """

    # If two points are within 0.01 of each each other, they are coincident
    minSeparation = 0.01

    # Return type for intersect
    Crossing = namedtuple('Crossing', ('firstPart','secondPart'))
    Coincidence = namedtuple('Coincidence', ('firstPart', 'secondPart', 'alignment', 'transform', 'inverse', 'translation'))

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
        return None

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

    def determinant(self, domainPoint):
        """
        Returns the determinant, which is the length of the cofactor normal (also the normal dotted with the cofactor normal).

        Parameters
        ----------
        domainPoint : `numpy.array`
            The 1D array at which to evaluate the determinant.

        Returns
        -------
        determinant : scalar
        """
        return np.dot(self.normal(domainPoint), self.normal(domainPoint, False))

    def transform(self, matrix, matrixInverseTranspose = None):
        """
        Transform the range of the manifold.

        Parameters
        ----------
        matrix : `numpy.array`
            A square matrix transformation.

        matrixInverseTranspose : `numpy.array`, optional
            The inverse transpose of matrix (computed if not provided).

        Returns
        -------
        manifold : `Manifold`
            The transformed manifold.

        See Also
        --------
        `solid.Solid.transform` : transform the range of the solid.
        """
        assert np.shape(matrix) == (self.range_dimension(), self.range_dimension())
        return None

    def translate(self, delta):
        """
        Translate the range of the manifold.

        Parameters
        ----------
        delta : `numpy.array`
            A 1D array translation.

        Returns
        -------
        manifold : `Manifold`
            The translated manifold.

        See Also
        --------
        `solid.Solid.translate` : translate the range of the solid.
        """
        assert len(delta) == self.range_dimension()
        return None

    def negate_normal(self):
        """
        Negate the direction of the normal.

        Returns
        -------
        manifold : `Manifold`
            The manifold with negated normal. The manifold retains the same tangent space.

        See Also
        --------
        `solid.Solid.complement` : Return the complement of the solid: whatever was inside is outside and vice-versa.
        """
        return None

    def intersect(self, other):
        """
        Intersect two manifolds (self and other).

        Parameters
        ----------
        other : `Manifold`
            The `Manifold` intersecting self.

        Returns
        -------
        intersections : `list` (or `NotImplemented` if other is an unknown type of Manifold)
            A list of intersections between the two manifolds. 
            Each intersection records either a crossing or a coincident region.

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
        `cached_intersect` : Intersect two manifolds, caching the result for twins (same intersection but swapping self and other).
        `solid.Solid.compute_cutout` : Compute the cutout portion of a manifold within a solid.

        Notes
        -----
        To invert the mapping to go from the other's domain to the manifold's domain, you first subtract the translation and then multiply by the inverse of the transform.
        """
        return NotImplemented

    def cached_intersect(self, other, cache = None):
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
        intersections : `list` (or `NotImplemented` if other is an unknown type of Manifold)
            A list of intersections between the two manifolds. 
            Each intersection records either a crossing or a coincident region.

            For a crossing, intersection is a Manifold.Crossing: (firstPart, secondPart)
            * firstPart : `Manifold` in the manifold's domain where the manifold and the other cross.
            * secondPart : `Manifold` in the other's domain where the manifold and the other cross.
            * Both intersection manifolds have the same domain and range (the crossing between the manifold and the other).

            For a coincident region, intersection is Manifold.Coincidence: (firstPart, secondPart, alignment, transform, inverse, translation)
            * firstPart : `Solid` in the manifold's domain within which the manifold and the other are coincident.
            * secondPart : `Solid` in the other's domain within which the manifold and the other are coincident.
            * alignment : scalar value holding the normal alignment between the manifold and the other (the dot product of their unit normals).
            * transform : `numpy.array` holding the matrix transform from the boundary's domain to the other's domain.
            * inverse : `numpy.array` holding the matrix inverse transform from the other's domain to the boundary's domain.
            * translation : `numpy.array` holding the 1D translation from the manifold's domain to the other's domain.
            * Together transform, inverse, and translation form the mapping from the manifold's domain to the other's domain and vice-versa.

        isTwin : `bool`
            True if this intersection is the twin from the cache (the intersection with self and other swapped).

        See Also
        --------
        `intersect` : Intersect two manifolds.
        `solid.Solid.compute_cutout` : Compute the cutout portion of a manifold within a solid.

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
            intersections = self.intersect(other)
            if intersections is NotImplemented:
                # Try the other way around in case other knows how to intersect self.
                intersections = other.intersect(self)
                isTwin = True
            # Store intersections in cache.
            if cache is not None:
                if isTwin:
                    cache[(other, self)] = intersections
                else:
                    cache[(self, other)] = intersections
        
        return intersections, isTwin

    def complete_cutout(self, cutout, solid):
        """
        Add any missing inherent (implicit) boundaries of this manifold's domain to the given cutout of the 
        given solid that are needed to make the cutout valid and complete.

        Parameters
        ----------
        cutout : `solid.Solid`
            The cutout of the given solid formed by the manifold. The cutout may be incomplete, missing some of the 
            manifold's inherent domain boundaries. Its dimension must match `self.domain_dimension()`.

        solid : `solid.Solid`
            The solid determining the cutout of the manifold. Its dimension must match `self.range_dimension()`.

        See Also
        --------
        `solid.Solid.compute_cutout` : Compute the cutout portion of a manifold within a solid.

        Notes
        -----
        For manifolds without inherent domain boundaries (like hyperplanes), the operation does nothing.
        """
        assert self.domain_dimension() == cutout.dimension
        assert self.range_dimension() == solid.dimension