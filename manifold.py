import numpy as np

class Manifold:

    # If a shift of 1 in the normal direction of one manifold yields a shift of 10 in the tangent plane intersection, the manifolds are parallel
    maxAlignment = 0.99 # 1 - 1/10^2

    # If two points are within 0.01 of each eachother, they are coincident
    minSeparation = 0.01

    def __init__(self):
        pass

    def Flip(self):
        return self

    def GetDimension(self):
        return 0

    def NormalFromDomain(self, domainPoint):
        return None

    def PointFromDomain(self, domainPoint):
        return None

    def DomainFromPoint(self, point):
        return None

    def Translate(self, delta):
        assert len(delta) == self.GetDimension()

    def IntersectXRay(self, point):
        assert len(point) == self.GetDimension()

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
        manifold = Hyperplane()

        # Ensure the normal is always an array
        if np.isscalar(normal):
            manifold.normal = np.array([normal])
        else:
            manifold.normal = np.array(normal)
        manifold.normal = manifold.normal / np.linalg.norm(manifold.normal)
        manifold.point = offset * manifold.normal
        if manifold.GetDimension() > 1:
            manifold.tangentSpace = Hyperplane.TangentSpaceFromNormal(manifold.normal)
        else:
            manifold.tangentSpace = 1.0
        return manifold

    def __init__(self):
        self.normal = None
        self.tangentSpace = None
        self.point = None

    def Flip(self):
        manifold = Hyperplane()
        manifold.normal = -self.normal
        manifold.tangentSpace = self.tangentSpace
        manifold.point = self.point
        return manifold

    def GetDimension(self):
        return len(self.normal)

    def NormalFromDomain(self, domainPoint):
        return self.normal

    def PointFromDomain(self, domainPoint):
        return np.dot(self.tangentSpace, domainPoint) + self.point

    def DomainFromPoint(self, point):
        return np.dot(point - self.point, self.tangentSpace)

    def Translate(self, delta):
        assert len(delta) == self.GetDimension()

        self.point += delta 

    def IntersectXRay(self, point):
        assert len(point) == self.GetDimension()

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []

        # Ensure manifold intersects x-axis
        if self.normal[0]*self.normal[0] > 1.0 - Manifold.maxAlignment:
            vectorToManifold = self.point - point
            xDistanceToManifold = np.dot(self.normal, vectorToManifold) / self.normal[0]
            intersectionPoint = np.array(point)
            intersectionPoint[0] += xDistanceToManifold
            # Each intersection is of the form [distance to intersection, domain point of intersection].
            intersections.append([xDistanceToManifold, self.DomainFromPoint(intersectionPoint)])
        
        return intersections

    def IntersectManifold(self, other, cache = None):
        # Check manifold intersections cache for previously computed intersections.
        if cache != None:
            if (self, other) in cache:
                return cache[(self, other)]
            elif (other, self) in cache:
                return cache[(other, self)]

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have multiple.
        intersections = []
        intersectionsFlipped = []

        # Ensure manifolds are not parallel
        alignment = np.dot(self.normal, other.normal)
        if self != other and alignment * alignment < Manifold.maxAlignment:
            # Compute the intersecting self domain manifold
            normalSelf = np.dot(other.normal, self.tangentSpace)
            normalize = 1.0 / np.linalg.norm(normalSelf)
            normalSelf = normalize * normalSelf
            offsetSelf = normalize * np.dot(other.normal, np.subtract(other.point, self.point))

            # Compute the intersecting other domain manifold
            normalOther = np.dot(self.normal, other.tangentSpace)
            normalize = 1.0 / np.linalg.norm(normalOther)
            normalOther = normalize * normalOther
            offsetOther = normalize * np.dot(self.normal, np.subtract(self.point, other.point))

            intersection = [Hyperplane.CreateFromNormal(normalSelf, offsetSelf), Hyperplane.CreateFromNormal(normalOther, offsetOther)]
            intersections.append(intersection)
            intersectionsFlipped.append([intersection[1], intersection[0]])

        # Store intersections in cache
        if cache != None:
            cache[(self,other)] = intersections
            cache[(other,self)] = intersectionsFlipped

        return intersections