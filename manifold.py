import numpy as np
import solid as sld

class Manifold:
    """
    A manifold is an abstract base class for differentiable functions with
    normals and tangent spaces whose range is one dimension higher than their domain.
    """

    # If two points are within 0.01 of each eachother, they are coincident
    minSeparation = 0.01

    # If a shift of 1 in the normal direction of one manifold yields a shift of 10 in the tangent plane intersection, the manifolds are parallel
    maxAlignment = 0.99 # 1 - 1/10^2

    def __init__(self):
        pass

    def copy(self):
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
            Each intersection is of the form (distance to intersection, domain point of intersection).

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
        intersections : `list`
            A list of intersections between the two manifolds. 
            Each intersection records either a crossing or a coincident region.

            For a crossing, intersection is a tuple holding:
            * intersection[0] : `Manifold` in the manifold's domain where the manifold and the other cross.
            * intersection[1] : `Manifold` in the other's domain where the manifold and the other cross.
            * Both intersection manifolds have the same domain and range (the crossing between the manifold and the other).

            For a coincident region, intersection is a tuple holding:
            * intersection[0] : `Solid` in the manifold's domain within which the manifold and the other are coincident.
            * intersection[1] : `Solid` in the other's domain within which the manifold and the other are coincident.
            * intersection[2] : scalar value holding the normal alignment between the manifold and the other (the dot product of their unit normals).
            * intersection[3] : `numpy.array` holding the 2D transform from the boundary's domain to the other's domain.
            * intersection[4] : `numpy.array` holding the 1D translation from the manifold's domain to the other's domain.
            * Together intersection[3] and intersection[4] form the mapping from the manifold's domain to the other's domain.

        See Also
        --------
        `solid.Solid.Slice` : Slice the solid by a manifold.
        `numpy.linalg.svd` : Compute the singular value decomposition of a 2D array.

        Notes
        -----
        To invert the mapping to go from the other's domain to the manifold's domain, you first subtract the translation and then multiply by the inverse of the transform.
        """
        return NotImplemented

class Hyperplane(Manifold):
    """
    A hyperplane is a `Manifold` defined by a unit normal, a point on the hyperplane, and a tangent space orthoganal to the normal.

    Parameters
    ----------
    normal : array-like
        The unit normal.
    
    point : array-like
        A point on the hyperplane.
    
    tangentSpace : array-like
        A array of tangents that are linearly independent and orthoganal to the normal.
    
    Notes
    -----
    The number of coordinates in the normal defines the dimension of the range of the hyperplane. The point must have the same dimension. The tangent space must be shaped: (dimension, dimension-1). 
    Thus the dimension of the domain is one less than that of the range.
    """
    def __init__(self, normal, point, tangentSpace):
        self.normal = np.atleast_1d(np.array(normal))
        self.point = np.atleast_1d(np.array(point))
        self.tangentSpace = np.atleast_1d(np.array(tangentSpace))

    def __str__(self):
        return "Normal: {0}, Point: {1}".format(self.normal, self.point)

    def __repr__(self):
        return "Hyperplane({0}, {1}, {2})".format(self.normal, self.point, self.tangentSpace)

    def copy(self):
        return Hyperplane(self.normal, self.point, self.tangentSpace)

    def RangeDimension(self):
        """
        Return the range dimension.

        Returns
        -------
        dimension : `int`
        """
        return len(self.normal)

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
        return self.normal

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
        return np.dot(self.tangentSpace, domainPoint) + self.point

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
        # Compute and cache cofactor normal on demand.
        if not hasattr(self, 'cofactorNormal'):
            dimension = self.RangeDimension()
            if dimension > 1:
                minor = np.zeros((dimension-1, dimension-1))
                self.cofactorNormal = np.array(self.normal) # We change it, so make a copy.
                sign = 1.0
                for i in range(dimension):
                    if i > 0:
                        minor[0:i, :] = self.tangentSpace[0:i, :]
                    if i < dimension - 1:
                        minor[i:, :] = self.tangentSpace[i+1:, :]
                    self.cofactorNormal[i] = sign * np.linalg.det(minor)
                    sign *= -1.0
        
                # Ensure cofactorNormal points in the same direction as normal.
                if np.dot(self.cofactorNormal, self.normal) < 0.0:
                    self.cofactorNormal = -self.cofactorNormal
            else:
                self.cofactorNormal = self.normal
        
        return self.cofactorNormal

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
        return self.CofactorNormal(domainPoint)[0]

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

        if self.RangeDimension() > 1:
            if transformInverseTranspose is None:
                transformInverseTranspose = np.transpose(np.linalg.inv(transform))

            self.normal = transformInverseTranspose @ self.normal
            if hasattr(self, 'cofactorNormal'):
                self.cofactorNormal = transformInverseTranspose @ self.cofactorNormal

            self.tangentSpace = transform @ self.tangentSpace

        self.point = transform @ self.point

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

        self.point += delta

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
        self.normal = -self.normal
        if hasattr(self, 'cofactorNormal'):
            self.cofactorNormal = -self.cofactorNormal

    def IntersectXRay(self, point):
        """
        Intersect a ray along the x-axis with the hyperplane.

        Parameters
        ----------
        point : array-like
            The starting point of the ray.

        Returns
        -------
        intersections : `list`
            A list of intersections between the ray and the hyperplane. 
            (Hyperplanes will have at most one intersection, but other types of manifolds can have several.)
            Each intersection is of the form (distance to intersection, domain point of intersection).

        See Also
        --------
        `solid.Solid.WindingNumber` : Compute the winding number for a point relative to the solid.
        """
        assert len(point) == self.RangeDimension()

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []

        # Ensure hyperplane intersects x-axis
        if self.normal[0]*self.normal[0] > 1.0 - Manifold.maxAlignment:
            # Getting the xDistance to the manifold is simple geometry.
            vectorFromManifold = point - self.point
            xDistanceToManifold = -np.dot(self.normal, vectorFromManifold) / self.normal[0]
            # Getting the domain point is a bit trickier. Turns out you throw out the x-components and invert the tangent space.
            domainPoint = np.linalg.inv(self.tangentSpace[1:,:]) @ vectorFromManifold[1:]
            # Each intersection is of the form [distance to intersection, domain point of intersection].
            intersections.append((xDistanceToManifold, domainPoint))

        return intersections

    def IntersectManifold(self, other):
        """
        Intersect two hyperplanes.

        Parameters
        ----------
        other : `Hyperplane`
            The `Hyperplane` intersecting the hyperplane.

        Returns
        -------
        intersections : `list`
            A list of intersections between the two hyperplanes. 
            (Hyperplanes will have at most one intersection, but other types of manifolds can have several.)
            Each intersection records either a crossing or a coincident region.

            For a crossing, intersection is a tuple holding:
            * intersection[0] : `Manifold` in the hyperplane's domain where the hyperplane and the other cross.
            * intersection[1] : `Manifold` in the other's domain where the hyperplane and the other cross.
            * Both intersection manifolds have the same domain and range (the crossing between the hyperplane and the other).

            For a coincident region, intersection is a tuple holding:
            * intersection[0] : `Solid` in the hyperplane's domain within which the hyperplane and the other are coincident.
            * intersection[1] : `Solid` in the other's domain within which the hyperplane and the other are coincident.
            * intersection[2] : scalar value holding the normal alignment between the hyperplane and the other (the dot product of their unit normals).
            * intersection[3] : `numpy.array` holding the 2D transform from the boundary's domain to the other's domain.
            * intersection[4] : `numpy.array` holding the 2D inverse transform from the other's domain to the boundary's domain.
            * intersection[5] : `numpy.array` holding the 1D translation from the hyperplane's domain to the other's domain.
            * Together intersection[3:6] form the mapping from the hyperplane's domain to the other's domain and vice-versa.

        See Also
        --------
        `solid.Solid.Slice` : Slice the solid by a manifold.
        `numpy.linalg.svd` : Compute the singular value decomposition of a 2D array.

        Notes
        -----
        Hyperplanes are parallel when their unit normals are aligned (dot product is nearly 1 or -1). Otherwise, they cross each other.

        To solve the crossing, we find the intersection by solving the underdetermined system of equations formed by assigning points 
        in one hyperplane (`self`) to points in the other (`other`). That is: 
        `self.tangentSpace * selfDomainPoint + self.point = other.tangentSpace * otherDomainPoint + other.point`. This system is `dimension` equations
        with `2*(dimension-1)` unknowns (the two domain points).
        
        There are more unknowns than equations, so it's underdetermined. The number of free variables is `2*(dimension-1) - dimension = dimension-2`.
        To solve the system, we rephrase it as `Ax = b`, where `A = (self.tangentSpace -other.tangentSpace)`, `x = (selfDomainPoint otherDomainPoint)`, 
        and `b = other.point - self.point`. Then we take the singular value decomposition of `A = U * Sigma * VTranspose`, using `numpy.linalg.svd`.
        The particular solution for x is given by `x = V * SigmaInverse * UTranspose * b`,
        where we only consider the first `dimension` number of vectors in `V` (the rest are zeroed out, i.e. the null space of `A`).
        The null space of `A` (the last `dimension-2` vectors in `V`) spans the free variable space, so those vectors form the tangent space of the intersection.
        Remember, we're solving for `x = (selfDomainPoint otherDomainPoint)`. So, the selfDomainPoint is the first `dimension-1` coordinates of `x`,
        and the otherDomainPoint is the last `dimension-1` coordinates of `x`. Likewise for the two tangent spaces.

        For coincident regions, we need the domains, normal alignment, and mapping from the hyperplane's domain to the other's domain. (The mapping is irrelevant and excluded for dimensions less than 2.)
        We can tell if the two hyperplanes are coincident if their normal alignment (dot product of their unit normals) is nearly 1 
        in absolute value (`alignment**2 < Manifold.maxAlignment`) and their points are barely separated:
        `-2 * Manifold.minSeparation < dot(hyperplane.normal, hyperplane.point - other.point) < Manifold.minSeparation`. (We give more room 
        to the outside than the inside to avoid compouding issues from minute gaps.)

        Since hyperplanes are flat, the domains of their coincident regions are the entire domain: `Solid(domain dimension, True)`.
        The normal alignment is the dot product of the unit normals. The mapping from the hyperplane's domain to the other's domain is derived
        from setting the hyperplanes to each other: 
        `hyperplane.tangentSpace * selfDomainPoint + hyperplane.point = other.tangentSpace * otherDomainPoint + other.point`. Then solve for
        `otherDomainPoint = inverse(transpose(other.tangentSpace) * other.tangentSpace)) * transpose(other.tangentSpace) * (hyperplane.tangentSpace * selfDomainPoint + hyperplane.point - other.point)`.
        You get the transform is `inverse(transpose(other.tangentSpace) * other.tangentSpace)) * transpose(other.tangentSpace) * hyperplane.tangentSpace`,
        and the translation is `inverse(transpose(other.tangentSpace) * other.tangentSpace)) * transpose(other.tangentSpace) * (hyperplane.point - other.point)`.

        Note that to invert the mapping to go from the other's domain to the hyperplane's domain, you first subtract the translation and then multiply by the inverse of the transform.
        """
        assert isinstance(other, Hyperplane)
        assert self.RangeDimension() == other.RangeDimension()

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []
        dimension = self.RangeDimension()

        # Check if manifolds intersect (are not parallel)
        alignment = np.dot(self.normal, other.normal)
        if alignment * alignment < Manifold.maxAlignment:
            # We're solving the system Ax = b using singular value decomposition, 
            #   where A = (self.tangentSpace -other.tangentSpace), x = (selfDomainPoint otherDomainPoint), and b = other.point - self.point.
            # Construct A.
            A = np.concatenate((self.tangentSpace, -other.tangentSpace),axis=1)
            # Compute the singular value decomposition of A.
            U, sigma, VTranspose = np.linalg.svd(A)
            # Compute the inverse of Sigma and transpose of V.
            SigmaInverse = np.diag(np.reciprocal(sigma))
            V = np.transpose(VTranspose)
            # Compute x = V * SigmaInverse * UTranspose * (other.point - self.point)
            x = V[:, 0:dimension] @ SigmaInverse @ np.transpose(U) @ (other.point - self.point)

            # The self intersection normal is just the dot product of other normal with the self tangent space.
            selfDomainNormal = np.dot(other.normal, self.tangentSpace)
            selfDomainNormal = selfDomainNormal / np.linalg.norm(selfDomainNormal)
            # The other intersection normal is just the dot product of self normal with the other tangent space.
            otherDomainNormal = np.dot(self.normal, other.tangentSpace)
            otherDomainNormal = otherDomainNormal / np.linalg.norm(otherDomainNormal)
            # The self intersection point is the first dimension-1 coordinates of x.
            selfDomainPoint = x[0:dimension-1]
            # The other intersection point is the last dimension-1 coordinates of x.
            otherDomainPoint = x[dimension-1:]
            if dimension > 2:
                # The self intersection tangent space is the first dimension-1 coordinates of the null space (the last dimension-2 vectors in V).
                selfDomainTangentSpace = V[0:dimension-1, dimension:]
                # The other intersection tangent space is the last dimension-1 coordinates of the null space (the last dimension-2 vectors in V).
                otherDomainTangentSpace = V[dimension-1:, dimension:]
            else:
                # There is no null space (dimension-2 <= 0)
                selfDomainTangentSpace = np.array([0.0])
                otherDomainTangentSpace = np.array([0.0])
            intersections.append((Hyperplane(selfDomainNormal, selfDomainPoint, selfDomainTangentSpace), Hyperplane(otherDomainNormal, otherDomainPoint, otherDomainTangentSpace)))

        # Otherwise, manifolds are parallel. Now, check if they are coincident.
        else:
            insideSeparation = np.dot(self.normal, self.point - other.point)
            # Allow for extra outside separation to avoid issues with minute gaps.
            if -2.0 * Manifold.minSeparation < insideSeparation < Manifold.minSeparation:
                # These hyperplanes are coincident. Return the domains in which they coincide (entire domain for hyperplanes) and the normal alignment.
                domainCoincidence = sld.Solid(dimension-1, True)
                if dimension > 1:
                    # For higher dimensions, also return the mapping from the self domain to the other domain.
                    tangentSpaceTranspose = np.transpose(other.tangentSpace)
                    map = np.linalg.inv(tangentSpaceTranspose @ other.tangentSpace) @ tangentSpaceTranspose
                    transform =  map @ self.tangentSpace
                    inverseTransform = np.linalg.inv(transform)
                    translation = map @ (self.point - other.point)
                    intersections.append((domainCoincidence, domainCoincidence, alignment, transform, inverseTransform, translation))
                else:
                    intersections.append((domainCoincidence, domainCoincidence, alignment))

        return intersections