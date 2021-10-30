import numpy as np

class Manifold:

    # If two points are within 0.01 of each eachother, they are coincident
    minSeparation = 0.01

    # If a shift of 1 in the normal direction of one manifold yields a shift of 10 in the tangent plane intersection, the manifolds are parallel
    maxAlignment = 0.99 # 1 - 1/10^2

    def __init__(self):
        pass

    def Flip(self):
        return self

    def GetDomainDimension(self):
        return self.GetRangeDimension() - 1

    def GetRangeDimension(self):
        return 0

    def Normal(self, domainPoint):
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

    def Point(self, domainPoint):
        return None

    def Translate(self, delta):
        assert len(delta) == self.GetRangeDimension()

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

    @staticmethod
    def TangentSpaceFromNormal(normal):
        # Construct the Householder reflection transform using the normal
        reflector = np.add(np.identity(len(normal)), np.outer(-2*normal, normal))
        # Compute the eigenvalues and eigenvectors for the symmetric transform (eigenvalues returned in ascending order).
        eigen = np.linalg.eigh(reflector)
        # Assert the first eigenvalue is negative (the reflection whose eigenvector is the normal)
        assert(eigen[0][0] < 0.0)
        # Return the tangent space by removing the first eigenvector column (the negated normal)
        return np.delete(eigen[1], 0, 1)
    
    @staticmethod
    def CreateFromNormal(normal, offset):
        hyperplane = Hyperplane()

        # Ensure the normal is always an array
        hyperplane.normal = np.atleast_1d(normal)
        hyperplane.normal = hyperplane.normal / np.linalg.norm(hyperplane.normal)
        hyperplane.point = offset * hyperplane.normal
        if hyperplane.GetRangeDimension() > 1:
            hyperplane.tangentSpace = Hyperplane.TangentSpaceFromNormal(hyperplane.normal)
        else:
            hyperplane.tangentSpace = np.array([0.0])
        return hyperplane

    def __init__(self):
        self.normal = None
        self.point = None
        self.tangentSpace = None

    def Flip(self):
        hyperplane = Hyperplane()
        hyperplane.normal = -self.normal
        hyperplane.point = self.point
        hyperplane.tangentSpace = self.tangentSpace
        return hyperplane

    def GetRangeDimension(self):
        return len(self.normal)

    def Normal(self, domainPoint):
        return self.normal
    
    def TangentSpace(self, domainPoint):
        return self.tangentSpace
    
    def CofactorNormal(self, domainPoint):
        # The cofactor normal is the normal formed by the cross-product of the tangent space vectors (the tangents).
        # The matrix constructed by TangentSpaceFromNormal is orthonormal, so the cofactor normal is simply the normal.
        return self.normal
    
    def FirstCofactor(self, domainPoint):
        return self.normal[0]

    def Point(self, domainPoint):
        return np.dot(self.tangentSpace, domainPoint) + self.point

    def DomainFromPoint(self, point):
        return np.dot(point - self.point, self.tangentSpace)

    def Translate(self, delta):
        assert len(delta) == self.GetRangeDimension()

        self.point += delta 

    def IntersectXRay(self, point):
        assert len(point) == self.GetRangeDimension()

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []

        # Ensure hyperplane intersects x-axis
        if self.normal[0]*self.normal[0] > 1.0 - Manifold.maxAlignment:
            vectorToManifold = self.point - point
            xDistanceToManifold = np.dot(self.normal, vectorToManifold) / self.normal[0]
            intersectionPoint = np.array(point)
            intersectionPoint[0] += xDistanceToManifold
            # Each intersection is of the form [distance to intersection, domain point of intersection].
            intersections.append([xDistanceToManifold, self.DomainFromPoint(intersectionPoint)])
        
        return intersections

    def IntersectManifold(self, other, cache = None):
        assert isinstance(other, Hyperplane)

        # Check manifold intersections cache for previously computed intersections.
        if cache != None:
            if (self, other) in cache:
                return cache[(self, other)]
            elif (other, self) in cache:
                return cache[(other, self)]

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []
        intersectionsFlipped = []

        # Initialize coincident marker
        coincident = False

        # Ensure manifolds are not parallel
        alignment = np.dot(self.normal, other.normal)
        if alignment * alignment < Manifold.maxAlignment:
            dimension = len(self.normal)

            # We're finding the intersection by solving the underdetermined system of equations formed by assigning points in self to points in other.
            # That is: self.tangentSpace * selfDomainPoint + self.point = other.tangentSpace * otherDomainPoint + other.point
            # This system is dimension equations with 2*(dimension-1) unknowns (the two domain points).
            # There are more unknowns than equations, so it's underdetermined. The number of free variables is 2*(dimension-1) - dimension = dimension-1.
            # To solve the system, we rephrase it as Ax = b,
            #   where A = (self.tangentSpace -other.tangentSpace), x = (selfDomainPoint otherDomainPoint), and b = other.point - self.point.
            # Then we take the singular value decomposition of A = U * Sigma * VTranspose.
            # The particular solution for x is given by x = V * SigmaInverse * UTranspose * b,
            #   where we only consider the first dimension number of vectors in V (the rest are zeroed out, i.e. the null space of A).
            # The null space of A (the last dimension-1 vectors in V) spans the free variable values, so those vectors form the tangent space of the intersection.
            # Remember, we're solving for x = (selfDomainPoint otherDomainPoint). So, the selfIntersection.point is the first dimension-1 values of x,
            #   and the otherIntersection.point is the last dimension-1 values of x. Likewise for the two tangent spaces.

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
            # The selfIntersection normal is just the dot product of other normal with the self tangent space.
            selfIntersection.normal = np.dot(other.normal, self.tangentSpace)
            selfIntersection.normal = selfIntersection.normal / np.linalg.norm(selfIntersection.normal)
            # The selfIntersection point is the first dimension-1 values of b.
            selfIntersection.point = x[0:dimension-1]
            # The selfIntersection tangent space is the first dimension-1 values of the null space (the last dimension-1 vectors in V).
            selfIntersection.tangentSpace = V[0:dimension-1, dimension:]

            otherIntersection = Hyperplane()
            # The otherIntersection normal is just the dot product of self normal with the other tangent space.
            otherIntersection.normal = np.dot(self.normal, other.tangentSpace)
            otherIntersection.normal = otherIntersection.normal / np.linalg.norm(otherIntersection.normal)
            # The otherIntersection point is the last dimension-1 values of b.
            otherIntersection.point = x[dimension-1:]
            # The otherIntersection tangent space is the last dimension-1 values of the null space (the last dimension-1 vectors in V).
            otherIntersection.tangentSpace = V[dimension-1:, dimension:]

            intersections.append([selfIntersection, otherIntersection])
            intersectionsFlipped.append([otherIntersection, selfIntersection])
        elif -2.0 * Manifold.minSeparation < np.dot(self.normal, self.point - other.point) < Manifold.minSeparation:
            # These two hyperplanes are coincident.
            coincident = True

        # Manifolds without intersections need to report coincidence. 
        if len(intersections) == 0 and coincident:
            intersections.append(alignment)
            intersectionsFlipped.append(alignment)

        # Store intersections in cache
        if cache != None:
            cache[(self,other)] = intersections
            cache[(other,self)] = intersectionsFlipped

        return intersections