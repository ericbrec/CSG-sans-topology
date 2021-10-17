import numpy as np

class Manifold:

    def __init__(self):
        pass

    def Flip(self):
        return self

    def GetDimension(self):
        return 0

    def Normal(self, domainPoint):
        return None
    
    def TangentSpace(self, domainPoint):
        return None
    
    def FirstCofactor(self, domainPoint):
        return 0.0

    def Point(self, domainPoint):
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

    # If a shift of 1 in the normal direction of one manifold yields a shift of 10 in the tangent plane intersection, the manifolds are parallel
    maxAlignment = 0.99 # 1 - 1/10^2

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
        if np.isscalar(normal):
            hyperplane.normal = np.array([normal])
        else:
            hyperplane.normal = np.array(normal)
        hyperplane.normal = hyperplane.normal / np.linalg.norm(hyperplane.normal)
        hyperplane.point = offset * hyperplane.normal
        if hyperplane.GetDimension() > 1:
            hyperplane.tangentSpace = Hyperplane.TangentSpaceFromNormal(hyperplane.normal)
            # The first (0,0) cofactor of matrix formed by the normal and tangent space is the determinant of the tangent space with the first row deleted.
            # The sign of the first cofactor must match the sign of the first component of the normal.
            hyperplane.firstCofactor = np.sign(hyperplane.normal[0]) * abs(np.linalg.det(np.delete(hyperplane.tangentSpace, 0, 0)))
        else:
            hyperplane.tangentSpace = 1.0
            hyperplane.firstCofactor = hyperplane.normal[0]
        return hyperplane

    def __init__(self):
        self.normal = None
        self.tangentSpace = None
        self.firstCofactor = 0.0
        self.point = None

    def Flip(self):
        hyperplane = Hyperplane()
        hyperplane.normal = -self.normal
        hyperplane.tangentSpace = self.tangentSpace
        hyperplane.firstCofactor = self.firstCofactor
        hyperplane.point = self.point
        return hyperplane

    def GetDimension(self):
        return len(self.normal)

    def Normal(self, domainPoint):
        return self.normal
    
    def TangentSpace(self, domainPoint):
        return self.tangentSpace
    
    def FirstCofactor(self, domainPoint):
        return self.firstCofactor

    def Point(self, domainPoint):
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

        # Ensure hyperplane intersects x-axis
        if self.normal[0]*self.normal[0] > 1.0 - Hyperplane.maxAlignment:
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

        # Ensure manifolds are not parallel
        alignment = np.dot(self.normal, other.normal)
        if self != other and alignment * alignment < Hyperplane.maxAlignment:
            # Compute the intersecting self domain hyperplane
            normalSelf = np.dot(other.normal, self.tangentSpace)
            normalize = 1.0 / np.linalg.norm(normalSelf)
            normalSelf = normalize * normalSelf
            offsetSelf = normalize * np.dot(other.normal, np.subtract(other.point, self.point))

            # Compute the intersecting other domain hyperplane
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