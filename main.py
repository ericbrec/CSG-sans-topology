import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

a = np.array([34.0])
print(a, np.array(a), type(a), len(a))
print(a[0])
print(2*a, np.array(2*a), type(2*a), len(2*a))
b = np.array([-1, 2])
c = a - b
print(c, type(c))

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
        # Ensure the normal is always an array
        if np.isscalar(normal):
            self.normal = np.array([normal])
        else:
            self.normal = np.array(normal)
        self.point = offset * normal
        if len(self.normal) > 1:
            self.tangentSpace = Manifold.TangentSpaceFromNormal(normal)
        else:
            self.tangentSpace = 1.0

    @staticmethod
    def TangentSpaceFromNormal(normal):
        # Construct the Householder reflection transform using the normal
        reflector = np.add(np.identity(3), np.outer(-2*normal, normal))
        # Compute the eigenvalues and eigenvectors for the symetric transform (eigenvalues returned in ascending order).
        eigen = np.linalg.eigh(reflector)
        # Assert the first eigenvalue is negative (the reflection whose eigenvector is the normal)
        assert(eigen[0][0] < 0.0)
        # Return the tangent space by removing the first eigenvector column (the negated normal)
        return np.delete(eigen[1], 0, 1)

    def Intersect(self, other, booleanOperation = "Intersection"):
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

    def __init__(self, manifold, domain = None):
        assert len(manifold.normal) - 1 == domain.dimension

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

    def __init__(self, dimension):
        assert dimension > 0
        self.dimension = dimension
        self.boundaries = []
    
    def Points(self):
        for boundary in self.boundaries:
            if boundary.domain:
                for domainPoint in boundary.domain.Points():
                    point = np.dot(boundary.manifold.tangetSpace, domainPoint) + boundary.manifold.point
                    yield point
            else:
                yield boundary.manifold.point
 
    def ContainsPoint(self, point):
        # Return value is -1 for interior, 0 for on boundary, and 1 for exterior
        containment = -1
        windingNumber = 0
        onBoundary = False

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
                    if boundary.domain:
                        vectorFromManifold = -vectorToManifold
                        vectorFromManifold[0] += distanceToManifold
                        domainPoint = np.dot(manifold.tangentSpace, vectorFromManifold)
                        considerBoundary = boundary.domain.ContainsPoint(domainPoint) < 1
                    if considerBoundary:
                        if distanceToManifold < Manifold.minSeparation:
                            onBoundary = True
                        windingNumber += np.sign(manifold.normal[0])
        
        if onBoundary:
            containment = 0
        elif windingNumber == 0:
            containment = 1
        return containment 

    def ContainsSolid(self, solid):
        # Assumes there is no intersection, though there may be coincience
        containment = False

        for point in solid.Points():
            constainsPoint = self.ContainsPoint(point)
            if constainsPoint != 0:
                containment = constainsPoint < 0
                break

        return containment

    def Slice(self, manifold):
        assert self.dimension > 1
        assert len(manifold.normal) == self.dimension

        # Create an empty domain for the manifold to return from the method
        manifoldDomain = Solid(self.dimension-1)

        # Intersect each of this solid's boundaries with the manifold.
        for boundary in self.boundaries:
            # Start by intersecting the boundary's manifold with the given manifold.
            # The Manifold.Intersect returns a collection of manifold pairs:
            #   * intersection[0] is in the boundary's domain;
            #   * intersection[1] is in the given manifold's domain.
            # Both intersections correspond to the same range (the intersection between the manifolds).
            intersections = boundary.manifold.Intersect(manifold)

            # For each intersection, slice the boundry domain with the intersection manifold.
            # We slice the boundary domain using intersection[0], but we add intersection[1] to the manifold domain. 
            for intersection in intersections:
                if boundary.domain.dimension > 1:
                    intersectionDomain = boundary.domain.Slice(intersection[0])
                    if intersectionDomain:
                        manifoldDomain.boundaries.append(Boundary(intersection[1],intersectionDomain))
                else:
                    # This domain is dimension 1, a real number line, and the intersection is a point on that line.
                    # Determine if the intersection point is within domain's interior
                    interior = False
                    for domainBoundary in boundary.domain.boundaries:
                        domainBoundaryManifold = domainBoundary.manifold
                        # Find first domain boundary point to the right of the intersection
                        if intersection[0].point < domainBoundaryManifold.point + Manifold.minSeparation:
                            # The intersection is interior if it's inside the domain and not coincient
                            if domainBoundaryManifold.normal > 0.0 and intersection[0].point < domainBoundaryManifold.point - Manifold.minSeparation:
                                interior = True
                            break
                    
                    if interior:
                        # Insert intersection into the manifold domain boundaries, in sorted order.
                        i = 0
                        while i < len(manifoldDomain.boundaries):
                            if intersection[1].point < manifoldDomain.boundaries[i].manifold.point:
                                break
                            i += 1
                        manifoldDomain.boundaries.insert(i,Boundary(intersection[1]))
        
        # Don't return a manifold domain if it's empty
        if len(manifoldDomain.boundaries) == 0:
            manifoldDomain = None
        
        return manifoldDomain

    def Combine(self, solid, booleanOperation = "Intersection"):
        assert self.dimension > 1
        assert self.dimension == solid.dimension

        combinedSolid = Solid(self.dimension)

        # We need a list of new solidBoundary domains, since we loop over solidBoundaries for each selfBoundary.
        solidBoundaryDomains = [None] * len(solid.boundaries)
        
        for selfBoundary in self.boundaries:
            # Create a new domain for selfBoundary, so we don't change the existing domain (no side effects).
            newSelfDomain = Solid(self.dimension-1)

            for b in len(solid.boundaries):
                solidBoundary = solid.boundaries[b]
                newSolidDomain = solidBoundaryDomains[b]

                # Create a new domain for solidBoundary, so we don't change the existing domain (no side effects).
                if newSolidDomain == None:
                    newSolidDomain = Solid(solid.dimension-1)
                    solidBoundaryDomains[b] = newSolidDomain

                # Intersect the selfBoundary manifold with the solidBoundary manifold.
                # The results will be a collection of manifold pairs:
                #   * intersection[0] from the domain of selfBoundary;
                #   * intersection[1] from the domain of solidBoundary.
                # Each manifold pair evaluates to the same range on boundaries self and solid.
                intersections = selfBoundary.manifold.Intersect(solidBoundary.manifold, booleanOperation)
                for intersection in intersections:
                    # TODO: Introduce case for domain.dimension == 1, either moving code from Slice method here or creating shared methods for point containment and merge.
                    # TODO: Or we could have Slice somehow support domain.dimension == 1.
                    intersectionSelfDomain = selfBoundary.domain.Slice(intersection[0])
                    if intersectionSelfDomain:
                       intersectionSolidDomain = solidBoundary.domain.Slice(intersection[1])
                    if intersectionSelfDomain and intersectionSolidDomain:
                        newSelfDomain.boundaries.append(Boundary(intersection[0],intersectionSelfDomain))
                        newSolidDomain.boundaries.append(Boundary(intersection[1],intersectionSolidDomain))

            # Now, we've intersected selfBoundary with every solidBoundary, creating a new domain for selfBoundary.
            # If selfBoundary's new domain isn't empty, combine it with selfBoundary's old domain (using "Intersection").
            if len(newSelfDomain.boundaries) > 0:
                combinedSolid.boundaries.append(Boundary(selfBoundary.manifold, selfBoundary.domain.Combine(newSelfDomain)))
            elif booleanOperation == "Union" or booleanOperation == "Difference":
                combinedSolid.boundaries.append(selfBoundary)

        # Now, we've intersected every solidBoundary with every selfBoundary, creating a new domain for each solidBoundary.
        # If solidBoundary's new domain isn't empty, combine it with solidBoundary's old domain (using "Intersection").
        for b in len(solid.boundaries):
            solidBoundary = solid.boundaries[b]
            newSolidDomain = solidBoundaryDomains[b]
            if len(newSolidDomain.boundaries) > 0:
                combinedSolid.boundaries.append(Boundary(solidBoundary.manifold, solidBoundary.domain.Combine(newSolidDomain)))
            elif booleanOperation == "Union":
                combinedSolid.boundaries.append(solidBoundary)

        return combinedSolid

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