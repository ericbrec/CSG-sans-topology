import numpy as np
import manifold as mf
import solid as sld
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backend_bases import MouseButton

class InteractiveCanvas:

    def CreateSegmentsFromSolid(self, solid):
        segments = []
       
        for edge in solid.Edges():
            middle = 0.5 * (edge[0] + edge[1])
            normal = middle + 0.1 * edge[2]
            segments.append((edge[0], edge[1]))
            segments.append((middle, normal))
        
        return segments
    
    def PerformBooleanOperation(self, key):
        if key == 'i':
            solid = self.solidA.Intersection(self.solidB)
            self.key = key
        elif key == 'u':
            solid = self.solidA.Union(self.solidB)
            self.key = key
        elif key == 'd':
            solid = self.solidA.Difference(self.solidB)
            self.key = key
        else:
            solid = self.solidC

        return solid

    def __init__(self, solidA, solidB):
        assert solidA.dimension == solidB.dimension

        fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title('Drag shape to update solid')
        self.canvas = self.ax.figure.canvas

        self.origin = [0.0]*solidB.dimension 

        self.solidA = solidA
        self.solidB = solidB
        self.solidC = self.PerformBooleanOperation('u')

        self.linesA = LineCollection(self.CreateSegmentsFromSolid(self.solidA), linewidth=1, color="blue")
        self.linesB = LineCollection(self.CreateSegmentsFromSolid(self.solidB), linewidth=1, color="orange")
        self.linesC = LineCollection(self.CreateSegmentsFromSolid(self.solidC), linewidth=3, color="red")
        
        self.ax.set(xlim = (-4, 4), ylim = (-4, 4))

        self.ax.add_collection(self.linesA)
        self.ax.add_collection(self.linesB)
        self.ax.add_collection(self.linesC)

        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_draw(self, event):
        """Callback for draws."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.linesA)
        self.ax.draw_artist(self.linesB)
        self.ax.draw_artist(self.linesC)

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if event.inaxes is None or event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return
        self.origin[0] = event.xdata
        self.origin[1] = event.ydata

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return

        print(self.key)
        self.solidC = self.PerformBooleanOperation(self.key)
        self.linesC.set_segments(self.CreateSegmentsFromSolid(self.solidC))
        self.canvas.draw()

    def on_key_press(self, event):
        """Callback for key presses."""
        self.key = event.key
        self.solidC = self.PerformBooleanOperation(self.key)
        self.linesC.set_segments(self.CreateSegmentsFromSolid(self.solidC))
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if event.inaxes is None or event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return
        
        delta = [0.0]*self.solidB.dimension
        delta[0] = event.xdata - self.origin[0]
        delta[1] = event.ydata - self.origin[1]
        self.solidB.Translate(delta)
        self.origin[0] = event.xdata
        self.origin[1] = event.ydata

        self.linesB.set_segments(self.CreateSegmentsFromSolid(self.solidB))
        
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.linesA)
        self.ax.draw_artist(self.linesB)
        self.ax.draw_artist(self.linesC)
        self.canvas.blit(self.ax.bbox)

def TangentSpaceFromNormal(normal):
    # Construct the Householder reflection transform using the normal
    reflector = np.add(np.identity(len(normal)), np.outer(-2*normal, normal))
    # Compute the eigenvalues and eigenvectors for the symmetric transform (eigenvalues returned in ascending order).
    eigen = np.linalg.eigh(reflector)
    # Assert the first eigenvalue is negative (the reflection whose eigenvector is the normal)
    assert(eigen[0][0] < 0.0)
    # Return the tangent space by removing the first eigenvector column (the negated normal)
    return np.delete(eigen[1], 0, 1)

def HyperplaneFromNormal(normal, offset):
    hyperplane = mf.Hyperplane()

    # Ensure the normal is always an array
    hyperplane.normal = np.atleast_1d(normal)
    hyperplane.normal = hyperplane.normal / np.linalg.norm(hyperplane.normal)
    hyperplane.point = offset * hyperplane.normal
    if hyperplane.GetRangeDimension() > 1:
        hyperplane.tangentSpace = TangentSpaceFromNormal(hyperplane.normal)
    else:
        hyperplane.tangentSpace = np.array([0.0])
    return hyperplane

def CreateSolidFromPoints(dimension, points, isVoid = False):
    # CreateSolidFromPoints only works for dimension 2 so far.
    assert dimension == 2
    assert len(points) > 2
    assert len(points[0]) == dimension

    solid = sld.Solid(dimension, isVoid)

    previousPoint = np.array(points[len(points)-1])
    for point in points:
        point = np.array(point)
        vector = point - previousPoint
        normal = np.array([-vector[1], vector[0]])
        normal = normal / np.linalg.norm(normal)
        hyperplane = HyperplaneFromNormal(normal,np.dot(normal,point))
        domain = sld.Solid(dimension-1)
        previousPointDomain = hyperplane.DomainFromPoint(previousPoint)
        pointDomain = hyperplane.DomainFromPoint(point)
        if previousPointDomain < pointDomain:
            domain.boundaries.append(sld.Boundary(HyperplaneFromNormal(-1.0, -previousPointDomain)))
            domain.boundaries.append(sld.Boundary(HyperplaneFromNormal(1.0, pointDomain)))
        else:
            domain.boundaries.append(sld.Boundary(HyperplaneFromNormal(-1.0, -pointDomain)))
            domain.boundaries.append(sld.Boundary(HyperplaneFromNormal(1.0, previousPointDomain)))
        solid.boundaries.append(sld.Boundary(hyperplane, domain))
        previousPoint = point

    return solid

def CreateHypercube(size, position = None):
    dimension = len(size)
    solid = sld.Solid(dimension)
    normal = [0.0]*dimension
    if position is None:
        position = [0.0]*dimension
    else:
        assert len(position) == dimension

    for i in range(dimension):
        domain = None
        if dimension > 1:
            domainSize = size.copy()
            del domainSize[i]
            domainPosition = position.copy()
            del domainPosition[i]
            domain = CreateHypercube(domainSize, domainPosition)
        normal[i] = 1.0
        hyperplane = HyperplaneFromNormal(normal, size[i] + normal[i]*position[i])
        solid.boundaries.append(sld.Boundary(hyperplane,domain))
        normal[i] = -1.0
        hyperplane = HyperplaneFromNormal(normal, size[i] + normal[i]*position[i])
        solid.boundaries.append(sld.Boundary(hyperplane,domain))
        normal[i] = 0.0

    return solid

def CreateStar(radius, center, angle):
    vertices = []
    points = 5
    for i in range(points):
        vertices.append([radius*np.cos(angle - ((2*i)%points)*6.2832/points) + center[0], radius*np.sin(angle - ((2*i)%points)*6.2832/points) + center[1]])

    nt = (vertices[1][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[1][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0])
    u = ((vertices[3][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[3][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0]))/nt

    star = CreateSolidFromPoints(2, vertices)
    for boundary in star.boundaries:
        u0 = boundary.domain.boundaries[0].manifold.point[0]
        u1 = boundary.domain.boundaries[1].manifold.point[0]
        boundary.domain.boundaries.append(sld.Boundary(HyperplaneFromNormal(1.0, u0 + (1.0 - u)*(u1 - u0))))
        boundary.domain.boundaries.append(sld.Boundary(HyperplaneFromNormal(-1.0, -(u0 + u*(u1 - u0)))))

    return star

triangleA = CreateSolidFromPoints(2, [[1,0],[0,0],[0,1]])
print(triangleA.VolumeIntegral(lambda x: 1.0), 0.5)
print(triangleA.SurfaceIntegral(lambda x, n: n), 2 + np.sqrt(2.0))
print(triangleA.WindingNumber(np.array([.75,.75])))
print(triangleA.WindingNumber(np.array([.5,.5])))
print(triangleA.WindingNumber(np.array([.25,.25])))

squareA = CreateHypercube([2,2], [-1,-1])
print(squareA.VolumeIntegral(lambda x: 1.0), 4.0*4.0)
print(squareA.SurfaceIntegral(lambda x, n: n), 4.0*4.0)
print(squareA.WindingNumber(np.array([0.,0.])))
print(squareA.WindingNumber(np.array([-0.23870968,1.])))
squareB = CreateHypercube([1,1], [2,0])
print(squareB.VolumeIntegral(lambda x: 1.0), 2.0*2.0)
print(squareB.SurfaceIntegral(lambda x, n: n), 2.0*4.0)

starArea = 10.0 * np.tan(np.pi / 10.0) / (3.0 - np.tan(np.pi / 10.0)**2)
starPerimeter = 10.0 * np.cos(2.0*np.pi/5.0) * (np.tan(2.0*np.pi/5.0) - np.tan(np.pi/5.0))
starA = CreateStar(2.0, [-1.0, -1.0], 90.0*6.28/360.0)
print(starA.VolumeIntegral(lambda x: 1.0), starArea * 4.0)
print(starA.SurfaceIntegral(lambda x, n: n), starPerimeter * 2.0)
print(starA.WindingNumber(np.array([-1.,-1.])))
starB = CreateStar(1.0, [2.0, 2.0], 90.0*6.28/360.0)
print(starB.VolumeIntegral(lambda x: 1.0), starArea)
print(starB.SurfaceIntegral(lambda x, n: n), starPerimeter)
starB.Translate([-2.31895479, -2.69507693])

interactor = InteractiveCanvas(squareA, squareB)
#interactor = InteractiveCanvas(starA, starB)
#interactor = InteractiveCanvas(squareA, starB)
plt.show()