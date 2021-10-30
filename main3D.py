import numpy as np
import manifold as mf
import solid as sld
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import mpl_toolkits.mplot3d.proj3d as proj3d
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
    
    def GetPointFromEvent(self, event):
        minX, maxX, minY, maxY, minZ, maxZ = self.ax.get_w_lims()
        projCenter = proj3d.transform(0.5*(minX+maxX), 0.5*(minY+maxY), 0.5*(minZ+maxZ), self.ax.M)
        return np.array(proj3d.inv_transform(event.xdata, event.ydata, projCenter[2], self.ax.M))

    def __init__(self, solidA, solidB):
        assert solidA.dimension == solidB.dimension

        fig = plt.figure(figsize=(6, 6))
        self.ax = fig.add_subplot(projection='3d')
        self.ax.set_title('Drag shape to update solid')
        self.canvas = self.ax.figure.canvas

        self.origin = [0.0]*solidB.dimension 

        self.solidA = solidA
        self.solidB = solidB
        self.solidC = self.PerformBooleanOperation('u')
        
        self.linesA = art3d.Line3DCollection(self.CreateSegmentsFromSolid(self.solidA), linewidth=1, color="blue")
        self.linesB = art3d.Line3DCollection(self.CreateSegmentsFromSolid(self.solidB), linewidth=1, color="orange")
        self.linesC = art3d.Line3DCollection(self.CreateSegmentsFromSolid(self.solidC), linewidth=3, color="red")
        
        self.ax.set(xlabel="x", ylabel="y", zlabel="z")
        self.ax.set(xlim3d = (-4, 4), ylim3d = (-4, 4), zlim3d = (-4, 4))

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

        self.ax.disable_mouse_rotation()
        self.origin = self.GetPointFromEvent(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return

        self.ax.mouse_init()
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
        
        point = self.GetPointFromEvent(event)
        delta = point - self.origin
        self.solidB.Translate(delta)
        self.origin = point

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

cubeA = CreateHypercube([2,2,2], [1,1,0])
intersections = cubeA.boundaries[0].manifold.IntersectManifold(cubeA.boundaries[3].manifold)
print(cubeA.VolumeIntegral(lambda x: 1.0), 4.0*4.0*4.0)
print(cubeA.SurfaceIntegral(lambda x, n: n), 4.0*4.0*6.0)
print(cubeA.WindingNumber([1,1,0]))
print(cubeA.WindingNumber([4,1,0]))
cubeB = CreateHypercube([1,1,1], [1,1,2])
print(cubeB.VolumeIntegral(lambda x: 1.0), 2.0*2.0*2.0)
print(cubeB.SurfaceIntegral(lambda x, n: n), 2.0*2.0*6.0)

interactor = InteractiveCanvas(cubeA, cubeB)
plt.show()