import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

x = []
y = []
points = 5
for i in range(points+1):
    x.append(np.cos(i*6.28/points))
    y.append(np.sin(i*6.28/points))

print(x)
print(y)

nt = (x[2]-x[0])*(y[3]-y[1]) + (y[2]-y[0])*(x[1]-x[3])
print(nt)
u = ((x[1]-x[0])*(y[3]-y[1]) + (y[1]-y[0])*(x[1]-x[3]))/nt
print(u, 1 - u)

print(x[0] + (x[2]-x[0])*u)
print(x[1] + (x[3]-x[1])*(1-u))
print(y[0] + (y[2]-y[0])*u)
print(y[1] + (y[3]-y[1])*(1-u))

class Manifold:

    # If a shift of 1 in the normal direction of one manifold yeilds a shift of 10 in the tangent plane intersection, the manifolds are parallel
    maxAlignment = 0.99 # 1 - 1/10^2

    # If two points are within 0.01 of each eachother, they are coincident
    minSeparation = 0.01

    def __init__(self, normal, offset):
        self.normal = normal
        self.point = offset * normal
        if isinstance(normal, list):
            self.tangentSpace = Manifold.TangentSpaceFromNormal(normal)
        else:
            self.tangentSpace = None

    @staticmethod
    def TangentSpaceFromNormal(normal):
        # Construct the Householder reflection transform using the normal
        reflector = np.add(np.identity(3), np.outer(-2*normal, normal))
        # Compute the eigenvalues and eigenvectors for the symetric transform
        eigen = np.linalg.eigh(reflector)
        # Assert the first eigenvalue is negative (the reflection whose eigenvector is the normal)
        assert(eigen[0][0] < 0.0)
        # Return the tangent space by removing the first eigenvector column (the negated normal)
        return np.delete(eigen[1], 0, 1)

    def Intersect(self, other, booleanOperation):
        assert len(self.normal) == len(other.normal)

        # Initialize list of intersections. Planar manifolds will have at most one intersection, but curved manifolds could have mulitple.
        intersections = []

        # Ensure manifolds are not parallel
        alignment = np.dot(self.normal, other.normal)
        if self != other and alignment * alignment < Manifold.maxAlignment:
            # Compute the intersecting self domain manifold
            normalSelf = np.dot(other.normal, self.tangentSpace)
            normalize = Solid.booleanOperations[booleanOperation][0] / np.linalg.norm(normalSelf)
            normalSelf = normalize * normalSelf
            offsetSelf = normalize * np.dot(other.normal, np.subtract(other.point, self.point))

            # Compute the intersecting other domain manifold
            normalOther = np.dot(self.normal, other.tangentSpace)
            normalize = Solid.booleanOperations[booleanOperation][1] / np.linalg.norm(normalOther)
            normalOther = normalize * normalOther
            offsetOther = normalize * np.dot(self.normal, np.subtract(self.point, other.point))

            intersections.append([Manifold(normalSelf, offsetSelf), Manifold(normalOther, offsetOther)])

        return intersections

class Boundary:

    def __init__(self, solid, manifold, domain = None):
        assert solid.dimension > 0
        if solid.dimension > 1:
            assert len(manifold.normal) == solid.dimension
        else:
            assert not isinstance(manifold.normal, list)

        self.solid = solid
        self.manifold = manifold
        self.domain = domain
    
    @staticmethod
    def SortKey(boundary):
        return boundary.manifold.point

class Solid:

    booleanOperations = {
        "Intersection": [1.0, 1.0],
        "Union": [-1.0, -1.0],
        "Difference": [1.0, -1.0]
    }

    def __init__(self, dimension, boundaries = []):
        assert dimension > 0
        self.dimension = dimension
        self.boundaries = boundaries.copy()

    @staticmethod
    def Combine(solidA, solidB, booleanOperation = "Intersection"):
        assert solidA.dimension > 1
        assert solidA.dimension == solidB.dimension

        solidC = Solid(solidA.dimension)
        # We need a list of new boundary B domains, since we loop over B boundaries for each boundary A.
        boundaryBDomains = [None] * len(solidB.boundaries)
        
        for boundaryA in solidA.boundaries:
            # Create a new domain for boundary A, so we don't change the existing domain (no side effects).
            newDomainA = Solid(solidA.dimension-1)

            for b in len(solidB.boundaries):
                boundaryB = solidB.boundaries[b]
                newDomainB = boundaryBDomains[b]

                # Create a new domain for boundary B, so we don't change the existing domain (no side effects).
                if newDomainB == None:
                    newDomainB = Solid(solidB.dimension-1)
                    boundaryBDomains[b] = newDomainB

                # Intersect the boundaryA manifold with the boundaryB manifold.
                # The results will be a collection of manifold pairs:
                #   * intersection[0] from the domain of boundaryA;
                #   * intersection[1] from the domain of boundary B.
                # Each manifold pair evaluates to the same contour on boundaries A and B.
                intersections = boundaryA.manifold.Intersect(boundaryB.manifold, booleanOperation)
                for intersection in intersections:
                    intersectionDomainA = boundaryA.domain.Intersect(intersection[0])
                    if intersectionDomainA:
                       intersectionDomainB = boundaryB.domain.Intersect(intersection[1])
                    if intersectionDomainA and intersectionDomainB:
                        newDomainA.boundaries.append(Boundary(newDomainA,intersection[0],intersectionDomainA))
                        newDomainB.boundaries.append(Boundary(newDomainB,intersection[1],intersectionDomainB))

            if len(newDomainA.boundaries) > 0:
                solidC.boundaries.append(Boundary(solidC, boundaryA.manifold, newDomainA))
            elif booleanOperation == "Union" or booleanOperation == "Difference":
                solidC.boundaries.append(boundaryA)
        
        return solidC
    
    def Intersect(self, manifold, manifoldBoundaries = [], booleanOperation = "Intersection"):
        # Create lists of new boundaries and unintersected boundaries to return.
        newBoundaries = []
        oldBoundaries = []
        solidIntersected = False

        if self.dimension > 1:
            assert len(manifold.normal) == self.dimension

            manifoldDomain = Solid(self.dimension-1, manifoldBoundaries)
            for boundary in self.boundaries:
                # Create a new domain for this boundary, so we don't change the existing domain (no side effects).
                newDomain = Solid(self.dimension-1, boundary.domain.boundaries)
                # Intersect the boundary with the manifold.
                intersections = boundary.manifold.Intersect(manifold, booleanOperation)
                boundaryIntersected = False
                # For each intersection, intersect the resulting domain manifolds with the our domain and the manifold's domain.
                for intersection in intersections:
                    domainIntersected, newDomainBoundaries, oldDomainBoundaries = newDomain.Intersect(intersection[0])
                    if domainIntersected:
                        domainIntersected, newManifoldBoundaries, oldManifoldBoundaries = manifoldDomain.Intersect(intersection[1])
                    if domainIntersected:
                        # The intersection is within each domain, so adjust the boundaries.

                        # TODO: Figure out how to deal with oldDomainBoundaries. (oldManifoldBoundaries are always empty.)

                        newDomain.boundaries = newDomainBoundaries
                        manifoldDomain.boundaries = newManifoldBoundaries
                        boundaryIntersected = True

                if boundaryIntersected:
                    newBoundaries.append(Boundary(self, boundary.manifold, newDomain))
                    solidIntersected = True
                else:
                    oldBoundaries.append(boundary)
            
            if solidIntersected:
                newBoundaries.append(Boundary(self, manifold, manifoldDomain))

        else:
            # This solid is dimension 1, a real number line, and the manifold is a point on that line.
            assert not isinstance(manifold.normal, list)
            if len(self.boundaries) == 0:
                # The number line started empty, so add missing boundaries at +/- infinity as needed. 
                if manifold.normal > 0.0:
                    newBoundaries.append(Boundary(self, Manifold(-1.0, float('inf'))))
                    newBoundaries.append(Boundary(self, manifold))
                else:
                    newBoundaries.append(Boundary(self, manifold))
                    newBoundaries.append(Boundary(self, Manifold(1.0, float('inf'))))
                solidIntersected = True
            else:
                # Replace existing manifold point with the new one as appropriate.
                # Loop through existing manifold points until you find the first greater than the new manifold point.
                for i in range(len(self.boundaries)):
                    boundary = self.boundaries[i]
                    existingManifold = boundary.manifold
                    if manifold.point < existingManifold.point + Manifold.minSeparation:
                        # Check if new point is outside range or coincient
                        if existingManifold.normal < 0.0 or manifold.point > existingManifold.point - Manifold.minSeparation:
                            # Skip manifold
                            pass
                        else:
                            # Add new boundary based on manifold and remove old one it's replacing
                            newBoundaries = self.boundaries.copy()
                            if manifold.normal < 0.0:
                                newBoundaries.pop(i-1)
                                newBoundaries.insert(i-1, Boundary(self, manifold))
                            else:
                                newBoundaries.pop(i)
                                newBoundaries.insert(i, Boundary(self, manifold))
                            solidIntersected = True
                        break
            
            if not solidIntersected:
                oldBoundaries = self.boundaries
        
        return solidIntersected, newBoundaries, oldBoundaries

class InteractiveCanvas:

    showverts = True
    epsilon = 8  # max pixel distance to count as a vertex hit

    def __init__(self, x, y):

        fig, self.ax = plt.subplots()
        self.ax.set_title('drag vertices to update path')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.canvas = self.ax.figure.canvas

        self._ind = None

        self.line, = self.ax.plot(
            x, y, color='blue', marker='o', markerfacecolor='r', markersize=self.epsilon, animated=True)

        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_draw(self, event):
        """Callback for draws."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if (event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts):
            return
        # self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if (event.button != MouseButton.LEFT
                or not self.showverts):
            return
        # self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if (self._ind is None
                or event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts):
            return

        #vertices = self.pathpatch.get_path().vertices

        #vertices[self._ind] = event.xdata, event.ydata
        #self.line.set_data(zip(*vertices))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


interactor = InteractiveCanvas(x, y)

# plt.show()