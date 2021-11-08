import numpy as np
import solid as sld

class Manifold:

    # If two points are within 0.01 of each eachother, they are coincident
    minSeparation = 0.01

    # If a shift of 1 in the normal direction of one manifold yields a shift of 10 in the tangent plane intersection, the manifolds are parallel
    maxAlignment = 0.99 # 1 - 1/10^2

    def __init__(self):
        pass

    def copy(self):
        return None

    def GetDomainDimension(self):
        return self.GetRangeDimension() - 1

    def GetRangeDimension(self):
        return 0

    def Normal(self, domainPoint):
        return None

    def Point(self, domainPoint):
        return None
    
    def TangentSpace(self, domainPoint):
        return None
    
    def CofactorNormal(self, domainPoint):
        # The cofactor normal is the normal formed by the cross-product of the tangent space vectors (the tangents).
        return None
    
    def FirstCofactor(self, domainPoint):
        return 0.0
    
    def Determinant(self, domainPoint):
        # The determinant is the length of the cofactor normal, which corresponds to the normal dotted with the cofactor normal.
        return np.dot(self.Normal(domainPoint), self.CofactorNormal(domainPoint))

    def Transform(self, transform):
        assert np.shape(transform) == (self.GetDomainDimension(), self.GetDomainDimension())

    def Translate(self, delta):
        assert len(delta) == self.GetRangeDimension()

    def FlipNormal(self):
        pass

    def IntersectXRay(self, point):
        assert len(point) == self.GetRangeDimension()

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []
        return intersections

    def IntersectManifold(self, other, cache = None):

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []
        return intersections

class Hyperplane(Manifold):

    def __init__(self):
        self.normal = None
        self.point = None
        self.tangentSpace = None

    def copy(self):
        hyperplane = Hyperplane()
        hyperplane.normal = self.normal
        hyperplane.point = self.point
        hyperplane.tangentSpace = self.tangentSpace
        return hyperplane

    def GetRangeDimension(self):
        return len(self.normal)

    def Normal(self, domainPoint):
        return self.normal

    def Point(self, domainPoint):
        return np.dot(self.tangentSpace, domainPoint) + self.point
    
    def TangentSpace(self, domainPoint):
        return self.tangentSpace
    
    def CofactorNormal(self, domainPoint):
        # The cofactor normal is the normal formed by the cross-product of the tangent space vectors (the tangents).
        # The matrix constructed by TangentSpaceFromNormal is orthonormal, so the cofactor normal is simply the normal.
        return self.normal
    
    def FirstCofactor(self, domainPoint):
        return self.normal[0]

    def Transform(self, transform):
        assert np.shape(transform) == (self.GetRangeDimension(), self.GetRangeDimension())

        self.tangentSpace = transform @ self.tangentSpace
        self.point = transform @ self.point

    def Translate(self, delta):
        assert len(delta) == self.GetRangeDimension()

        self.point += delta 

    def FlipNormal(self):
        self.normal = -self.normal

    def IntersectXRay(self, point):
        assert len(point) == self.GetRangeDimension()

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

    def IntersectManifold(self, other, cache = None):
        assert isinstance(other, Hyperplane)
        assert self.GetRangeDimension() == other.GetRangeDimension()

        # Check manifold intersections cache for previously computed intersections.
        if cache != None:
            if (self, other) in cache:
                return cache[(self, other)]
            else:
                # Prepare to cache results for IntersectManifold(other, self).
                intersectionsFlipped = []

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []

        # Check if manifolds intersect (are not parallel)
        alignment = np.dot(self.normal, other.normal)
        if alignment * alignment < Manifold.maxAlignment:
            dimension = self.GetRangeDimension()

            # We're finding the intersection by solving the underdetermined system of equations formed by assigning points in self to points in other.
            # That is: self.tangentSpace * selfDomainPoint + self.point = other.tangentSpace * otherDomainPoint + other.point
            # This system is dimension equations with 2*(dimension-1) unknowns (the two domain points).
            # There are more unknowns than equations, so it's underdetermined. The number of free variables is 2*(dimension-1) - dimension = dimension-2.
            # To solve the system, we rephrase it as Ax = b,
            #   where A = (self.tangentSpace -other.tangentSpace), x = (selfDomainPoint otherDomainPoint), and b = other.point - self.point.
            # Then we take the singular value decomposition of A = U * Sigma * VTranspose.
            # The particular solution for x is given by x = V * SigmaInverse * UTranspose * b,
            #   where we only consider the first dimension number of vectors in V (the rest are zeroed out, i.e. the null space of A).
            # The null space of A (the last dimension-2 vectors in V) spans the free variable space, so those vectors form the tangent space of the intersection.
            # Remember, we're solving for x = (selfDomainPoint otherDomainPoint). So, the selfIntersection.point is the first dimension-1 coordinates of x,
            #   and the otherIntersection.point is the last dimension-1 coordinates of x. Likewise for the two tangent spaces.

            # Okay, first construct A.
            A = np.concatenate((self.tangentSpace, -other.tangentSpace),axis=1)
            # Compute the singular value decomposition of A.
            U, sigma, VTranspose = np.linalg.svd(A)
            # Compute the inverse of Sigma and transpose of V.
            SigmaInverse = np.diag(np.reciprocal(sigma))
            V = np.transpose(VTranspose)
            # Compute x = V * SigmaInverse * UTranspose * (other.point - self.point)
            x = V[:, 0:dimension] @ SigmaInverse @ np.transpose(U) @ (other.point - self.point)
            
            selfIntersection = Hyperplane()
            otherIntersection = Hyperplane()
            # The selfIntersection normal is just the dot product of other normal with the self tangent space.
            selfIntersection.normal = np.dot(other.normal, self.tangentSpace)
            selfIntersection.normal = selfIntersection.normal / np.linalg.norm(selfIntersection.normal)
            # The otherIntersection normal is just the dot product of self normal with the other tangent space.
            otherIntersection.normal = np.dot(self.normal, other.tangentSpace)
            otherIntersection.normal = otherIntersection.normal / np.linalg.norm(otherIntersection.normal)
            # The selfIntersection point is the first dimension-1 coordinates of x.
            selfIntersection.point = x[0:dimension-1]
            # The otherIntersection point is the last dimension-1 coordinates of x.
            otherIntersection.point = x[dimension-1:]
            if dimension > 2:
                # The selfIntersection tangent space is the first dimension-1 coordinates of the null space (the last dimension-2 vectors in V).
                selfIntersection.tangentSpace = V[0:dimension-1, dimension:]
                # The otherIntersection tangent space is the last dimension-1 coordinates of the null space (the last dimension-2 vectors in V).
                otherIntersection.tangentSpace = V[dimension-1:, dimension:]
            else:
                # There is no null space (dimension-2 <= 0)
                selfIntersection.tangentSpace = np.array([0.0])
                otherIntersection.tangentSpace = np.array([0.0])

            intersections.append((selfIntersection, otherIntersection))
            if cache != None:
                intersectionsFlipped.append((otherIntersection, selfIntersection))

        # Otherwise, manifolds are parallel. Now, check if they are coincident.
        elif -2.0 * Manifold.minSeparation < np.dot(self.normal, self.point - other.point) < Manifold.minSeparation:
            # These hyperplanes are coincident.
            # Return the domain in which they coincide (entire domain for hyperplanes) and the mapping from the self domain to the other domain.
            domainCoincidence = sld.Solid(self.GetDomainDimension(), True)
            tangentSpaceTranspose = np.transpose(other.tangentSpace)
            inverseMap = np.linalg.inv(tangentSpaceTranspose @ other.tangentSpace) @ tangentSpaceTranspose
            transform =  inverseMap @ self.tangentSpace
            translation = inverseMap @ (self.point - other.point)
            intersections.append((domainCoincidence, transform, translation))

            if cache != None:
                # Do the same for the mapping from the other domain to the self domain.
                tangentSpaceTranspose = np.transpose(self.tangentSpace)
                inverseMap = np.linalg.inv(tangentSpaceTranspose @ self.tangentSpace) @ tangentSpaceTranspose
                transform =  inverseMap @ other.tangentSpace
                translation = inverseMap @ (other.point - self.point)
                intersectionsFlipped.append((domainCoincidence, transform, translation))

        # Store intersections in cache
        if cache != None:
            cache[(other,self)] = intersectionsFlipped

        return intersections