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
        to the outside than the inside to avoid compounding issues from minute gaps.)

        Since hyperplanes are flat, the domains of their coincident regions are the entire domain: `Solid(domain dimension, True)`.
        The normal alignment is the dot product of the unit normals. The mapping from the hyperplane's domain to the other's domain is derived
        from setting the hyperplanes to each other: 
        `hyperplane.tangentSpace * selfDomainPoint + hyperplane.point = other.tangentSpace * otherDomainPoint + other.point`. Then solve for
        `otherDomainPoint = inverse(transpose(other.tangentSpace) * other.tangentSpace)) * transpose(other.tangentSpace) * (hyperplane.tangentSpace * selfDomainPoint + hyperplane.point - other.point)`.
        You get the transform is `inverse(transpose(other.tangentSpace) * other.tangentSpace)) * transpose(other.tangentSpace) * hyperplane.tangentSpace`,
        and the translation is `inverse(transpose(other.tangentSpace) * other.tangentSpace)) * transpose(other.tangentSpace) * (hyperplane.point - other.point)`.

        Note that to invert the mapping to go from the other's domain to the hyperplane's domain, you first subtract the translation and then multiply by the inverse of the transform.
        """
        assert self.RangeDimension() == other.RangeDimension()

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []
        dimension = self.RangeDimension()

        if isinstance(other, Hyperplane):
            pass
        elif isinstance(other, Spline):
            pass
        else:
            return NotImplemented

        # Check if manifolds intersect (are not parallel)
        alignment = np.dot(self.normal, other.normal)
        if alignment * alignment < Spline.maxAlignment:
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
            intersections.append(Manifold.Crossing(Spline(selfDomainNormal, selfDomainPoint, selfDomainTangentSpace), Spline(otherDomainNormal, otherDomainPoint, otherDomainTangentSpace)))

        return intersections