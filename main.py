import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

# TODO: Update ContainsPoint to be manifold agnostic (it currently assumes a hyperplane).
# TODO: Update ContainsPoint to use integral instead of ray cast to compute winding number.

class Manifold:

    # If a shift of 1 in the normal direction of one manifold yields a shift of 10 in the tangent plane intersection, the manifolds are parallel
    maxAlignment = 0.99 # 1 - 1/10^2

    # If two points are within 0.01 of each eachother, they are coincident
    minSeparation = 0.01

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
        manifold = Manifold()

        # Ensure the normal is always an array
        if np.isscalar(normal):
            manifold.normal = np.array([normal])
        else:
            manifold.normal = np.array(normal)
        manifold.normal = manifold.normal / np.linalg.norm(manifold.normal)
        manifold.point = offset * manifold.normal
        if len(manifold.normal) > 1:
            manifold.tangentSpace = Manifold.TangentSpaceFromNormal(manifold.normal)
        else:
            manifold.tangentSpace = 1.0
        return manifold

    def __init__(self):
        self.normal = None
        self.tangentSpace = None
        self.point = None

    def Flip(self):
        manifold = Manifold()
        manifold.normal = -self.normal
        manifold.tangentSpace = self.tangentSpace
        manifold.point = self.point
        return manifold

    def PointFromDomain(self, domainPoint):
        return np.dot(self.tangentSpace, domainPoint) + self.point

    def DomainFromPoint(self, point):
        return np.dot(point - self.point, self.tangentSpace)

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

            intersection = [Manifold.CreateFromNormal(normalSelf, offsetSelf), Manifold.CreateFromNormal(normalOther, offsetOther)]
            intersections.append(intersection)
            intersectionsFlipped.append([intersection[1], intersection[0]])

        # Store intersections in cache
        if cache != None:
            cache[(self,other)] = intersections
            cache[(other,self)] = intersectionsFlipped

        return intersections

class Boundary:

    def __init__(self, manifold, domain = None):
        if domain:
            assert len(manifold.normal) - 1 == domain.dimension
        else:
            assert len(manifold.normal) == 1

        self.manifold = manifold
        self.domain = domain
    
    @staticmethod
    def SortKey(boundary):
        return boundary.manifold.point[0]

class Solid:

    @staticmethod
    def CreateSolidFromPoints(dimension, points, isVoid = False):
        # Implementation only works for dimension 2 so far.
        assert dimension == 2
        assert len(points) > 2
        assert len(points[0]) == dimension

        solid = Solid(dimension, isVoid)

        previousPoint = np.array(points[len(points)-1])
        for point in points:
            point = np.array(point)
            vector = point - previousPoint
            normal = np.array([-vector[1], vector[0]])
            normal = normal / np.linalg.norm(normal)
            manifold = Manifold.CreateFromNormal(normal,np.dot(normal,point))
            domain = Solid(dimension-1)
            domain.boundaries.append(Boundary(Manifold.CreateFromNormal(-1.0, -manifold.DomainFromPoint(previousPoint))))
            domain.boundaries.append(Boundary(Manifold.CreateFromNormal(1.0, manifold.DomainFromPoint(point))))
            solid.boundaries.append(Boundary(manifold, domain))
            previousPoint = point

        return solid

    def __init__(self, dimension, isVoid = False):
        assert dimension > 0
        self.dimension = dimension
        self.isVoid = isVoid
        self.boundaries = []

    def Not(self):
        solid = Solid(self.dimension, not self.isVoid)
        for boundary in self.boundaries:
            solid.boundaries.append(Boundary(boundary.manifold.Flip(),boundary.domain))
        return solid
    
    def Translate(self, delta):
        assert len(delta) == self.dimension

        for boundary in self.boundaries:
            boundary.manifold.point += delta
    
    def Points(self):
        for boundary in self.boundaries:
            if boundary.domain:
                for domainPoint in boundary.domain.Points():
                    point = boundary.manifold.PointFromDomain(domainPoint)
                    yield point
            else:
                yield boundary.manifold.point
    
    def Edges(self):
        if self.dimension > 1:
            for boundary in self.boundaries:
                for domainEdge in boundary.domain.Edges():
                    yield [boundary.manifold.PointFromDomain(domainEdge[0]), boundary.manifold.PointFromDomain(domainEdge[1])]
        else:
            self.boundaries.sort(key=Boundary.SortKey)
            b = 1
            while b < len(self.boundaries):
                yield [self.boundaries[b-1].manifold.point, self.boundaries[b].manifold.point]
                b += 2
 
    def ContainsPoint(self, point):
        # Return value is -1 for interior, 0 for on boundary, and 1 for exterior
        containment = -1
        onBoundary = False

        # The winding number is 0 if the point is outside the solid, 1 if it's inside.
        # Other values indicate issues: 
        #  * Incomplete boundaries lead to fractional values;
        #  * Interior-pointing normals lead to negative values;
        #  * Nested shells lead to absolute values of 2 or greater.
        windingNumber = 0
        if self.isVoid:
            # If the solid is a void, then the winding number starts as 1 to account for the boundary at infinity.
            windingNumber = 1

        # Intersect a ray from point along the x-axis through self's boundaries.
        # All dot products with the ray are just the first component, since the ray is along the x-axis.
        for boundary in self.boundaries:
            manifold = boundary.manifold
            # Ensure manifold intersects x-axis
            if manifold.normal[0]*manifold.normal[0] > 1 - Manifold.maxAlignment:
                # Calculate distance to manifold
                vectorToManifold = manifold.point - point
                distanceToManifold = np.dot(manifold.normal, vectorToManifold) / manifold.normal[0]

                # Check the distance is positive
                if distanceToManifold > -Manifold.minSeparation:
                    considerBoundary = True
                    # Don't consider boundary if ray intersection is outside its domain.
                    if boundary.domain:
                        vectorFromManifold = -vectorToManifold
                        vectorFromManifold[0] += distanceToManifold
                        domainPoint = np.dot(vectorFromManifold, manifold.tangentSpace)
                        considerBoundary = boundary.domain.ContainsPoint(domainPoint) < 1
                    # We intersect this boundary, so accumulate winding number based on sign of dot(ray,normal) == normal[0].
                    if considerBoundary:
                        if distanceToManifold < Manifold.minSeparation:
                            onBoundary = True
                        windingNumber += np.sign(manifold.normal[0])
        
        if onBoundary:
            containment = 0
        elif windingNumber == 0:
            containment = 1
        return containment 

    def ContainsBoundary(self, boundary):
        # Assumes there is no boundary intersection, though there may be coincidence
        containment = False

        if boundary.domain:
            for domainPoint in boundary.domain.Points():
                constainsPoint = self.ContainsPoint(boundary.manifold.PointFromDomain(domainPoint))
                if constainsPoint != 0:
                    containment = constainsPoint < 0
                    break
        else:
            containment = self.ContainsPoint(boundary.manifold.point) < 0

        return containment

    def Slice(self, manifold, cache = None):
        assert len(manifold.normal) == self.dimension

        manifoldDomain = None

        # Only manifolds of dimension > 1 have a domain.
        if self.dimension > 1:
            manifoldDomain = Solid(self.dimension-1, self.isVoid)

            # Intersect each of this solid's boundaries with the manifold.
            for boundary in self.boundaries:
                # Start by intersecting the boundary's manifold with the given manifold.
                # The Manifold.Intersect returns a collection of manifold pairs:
                #   * intersection[0] is in the boundary's domain;
                #   * intersection[1] is in the given manifold's domain.
                # Both intersections correspond to the same range (the intersection between the manifolds).
                intersections = boundary.manifold.IntersectManifold(manifold, cache)

                # For each intersection, slice the boundary domain with the intersection manifold.
                # We slice the boundary domain using intersection[0], but we add intersection[1] to the manifold domain. 
                for intersection in intersections:
                    if boundary.domain.dimension > 1:
                        intersectionDomain = boundary.domain.Slice(intersection[0], cache)
                        if intersectionDomain:
                            manifoldDomain.boundaries.append(Boundary(intersection[1],intersectionDomain))
                    else:
                        # This domain is dimension 1, a real number line, and the intersection is a point on that line.
                        # Determine if the intersection point is within domain's interior.
                        if boundary.domain.ContainsPoint(intersection[0].point) < 0:
                            manifoldDomain.boundaries.append(Boundary(intersection[1]))
            
            # Don't return a manifold domain if it's empty
            if len(manifoldDomain.boundaries) == 0:
                manifoldDomain = None
        
        return manifoldDomain

    def Intersection(self, solid, cache = None):
        assert self.dimension == solid.dimension

        # Manifold intersections are expensive and come in symmetric pairs (m1 intersect m2, m2 intersect m1).
        # So, we create a manifold intersections cache (dictionary) to store and reuse intersection pairs.
        if cache == None:
            cache = {}

        combinedSolid = Solid(self.dimension, self.isVoid and solid.isVoid)

        for boundary in self.boundaries:
            # Slice self boundary manifold by solid. If it intersects, intersect the domains.
            newDomain = solid.Slice(boundary.manifold, cache)
            if newDomain:
                newDomain = boundary.domain.Intersection(newDomain, cache)
                if len(newDomain.boundaries) > 0:
                    combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif solid.ContainsBoundary(boundary):
                combinedSolid.boundaries.append(boundary)

        for boundary in solid.boundaries:
            # Slice solid boundary manifold by self. If it intersects, intersect the domains.
            newDomain = self.Slice(boundary.manifold, cache)
            if newDomain:
                newDomain = boundary.domain.Intersection(newDomain, cache)
                if len(newDomain.boundaries) > 0:
                    combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif self.ContainsBoundary(boundary):
                combinedSolid.boundaries.append(boundary)

        return combinedSolid

    def Union(self, solid):
        return self.Not().Intersection(solid.Not()).Not()

    def Difference(self, solid):
        return self.Intersection(solid.Not())

class InteractiveCanvas:

    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, solidA, solidB):
        assert solidA.dimension == solidB.dimension

        fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title('Drag shape to update path')
        self.canvas = self.ax.figure.canvas

        self.origin = [0.0]*solidB.dimension 

        self.solidA = solidA
        self.solidB = solidB
        self.solid = solidA.Union(solidB)
        
        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        self.on_draw(None)

    def on_draw(self, event):
        """Callback for draws."""
        self.ax.cla()
        self.ax.set(xlim = (-4, 4), ylim = (-4, 4))
        
        for edge in self.solidA.Edges():
            self.ax.arrow(edge[0][0], edge[0][1], edge[1][0] - edge[0][0], edge[1][1] - edge[0][1], width=0.01, head_width=0.0, color="orange")

        for edge in self.solidB.Edges():
            self.ax.arrow(edge[0][0], edge[0][1], edge[1][0] - edge[0][0], edge[1][1] - edge[0][1], width=0.01, head_width=0.0, color="yellow")

        for edge in self.solid.Edges():
            self.ax.arrow(edge[0][0], edge[0][1], edge[1][0] - edge[0][0], edge[1][1] - edge[0][1], width=0.05, head_width=0.0, color="red")

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if event.inaxes is None or event.button != MouseButton.LEFT:
            return
        self.origin[0] = event.xdata
        self.origin[1] = event.ydata

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.button != MouseButton.LEFT:
            return
        
        delta = [0.0]*self.solidB.dimension
        delta[0] = event.xdata - self.origin[0]
        delta[1] = event.ydata - self.origin[1]
        self.solidB.Translate(delta)
        self.solid = self.solidA.Union(self.solidB)

        self.on_draw(None)
        self.canvas.draw()


    def on_key_press(self, event):
        """Callback for key presses."""
        if event.key == 'i':
            self.solid = self.solidA.Intersection(self.solidB)
        elif event.key == 'u':
            self.solid = self.solidA.Union(self.solidB)
        elif event.key == 'd':
            self.solid = self.solidA.Difference(self.solidB)

        self.on_draw(None)
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if (event.inaxes is None or event.button != MouseButton.LEFT):
            return

def CreateStar(radius, center, angle):
    vertices = []
    points = 5
    for i in range(points):
        vertices.append([radius*np.cos(angle + ((2*i)%points)*6.28/points) + center[0], radius*np.sin(angle + ((2*i)%points)*6.28/points) + center[1]])

    nt = (vertices[1][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[1][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0])
    u = ((vertices[3][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[3][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0]))/nt

    solid = Solid.CreateSolidFromPoints(2, vertices)
    for boundary in solid.boundaries:
        u0 = boundary.domain.boundaries[0].manifold.point[0]
        u1 = boundary.domain.boundaries[1].manifold.point[0]
        boundary.domain.boundaries.append(Boundary(Manifold.CreateFromNormal(1.0, u0 + (1.0 - u)*(u1 - u0))))
        boundary.domain.boundaries.append(Boundary(Manifold.CreateFromNormal(-1.0, -(u0 + u*(u1 - u0)))))

    return solid

squareA = Solid.CreateSolidFromPoints(2, [[-3,-3],[-3,1],[1,1],[1,-3]])
squareB = Solid.CreateSolidFromPoints(2, [[-1,-1],[-1,2],[2,2],[2,-1]])

starA = CreateStar(1.0, [-2.0, -2.0], 0.0)
starB = CreateStar(1.0, [2.0, 2.0], 45.0*6.28/360.0)

interactor = InteractiveCanvas(squareA, squareB)
#interactor = InteractiveCanvas(starA, starB)
plt.show()