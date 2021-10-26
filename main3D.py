import numpy as np
import manifold as mf
import solid as sld
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.backend_bases import MouseButton

class InteractiveCanvas:

    epsilon = 5  # max pixel distance to count as a vertex hit

    def CreatePathFromSolid(self, solid):
        vertices = [[0.0]*solid.dimension]
        commands = [Path.MOVETO]
       
        for edge in solid.Edges():
            middle = 0.5 * (edge[0] + edge[1])
            normal = middle + 0.1 * edge[2]
            vertices.append(edge[0])
            commands.append(Path.MOVETO)
            vertices.append(edge[1])
            commands.append(Path.LINETO)
            vertices.append(middle)
            commands.append(Path.MOVETO)
            vertices.append(normal)
            commands.append(Path.LINETO)
        
        return Path(vertices, commands)
    
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

        fig = plt.figure(figsize=(6, 6))
        self.ax = fig.add_subplot(projection='3d')
        self.ax.set_title('Drag shape to update solid')
        self.canvas = self.ax.figure.canvas

        self.origin = [0.0]*solidB.dimension 

        self.solidA = solidA
        self.solidB = solidB
        self.solidC = self.PerformBooleanOperation('u')

        self.patchA = patches.PathPatch3D(self.CreatePathFromSolid(self.solidA), linewidth=1, color="blue")
        self.patchB = patches.PathPatch3D(self.CreatePathFromSolid(self.solidB), linewidth=1, color="orange")
        self.patchC = patches.PathPatch3D(self.CreatePathFromSolid(self.solidC), linewidth=3, color="red")
        
        self.ax.set(xlim = (-4, 4), ylim = (-4, 4), zlim = (-4, 4))

        self.ax.add_patch(self.patchA)
        self.ax.add_patch(self.patchB)
        self.ax.add_patch(self.patchC)

        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_draw(self, event):
        """Callback for draws."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.patchA)
        self.ax.draw_artist(self.patchB)
        self.ax.draw_artist(self.patchC)

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
        self.patchC.set_path(self.CreatePathFromSolid(self.solidC))
        self.canvas.draw()

    def on_key_press(self, event):
        """Callback for key presses."""
        self.key = event.key
        self.solidC = self.PerformBooleanOperation(self.key)
        self.patchC.set_path(self.CreatePathFromSolid(self.solidC))
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

        self.patchB.set_path(self.CreatePathFromSolid(self.solidB))
        
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.patchA)
        self.ax.draw_artist(self.patchB)
        self.ax.draw_artist(self.patchC)
        self.canvas.blit(self.ax.bbox)

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
        hyperplane = mf.Hyperplane.CreateFromNormal(normal, size[i] + normal[i]*position[i])
        solid.boundaries.append(sld.Boundary(hyperplane,domain))
        normal[i] = -1.0
        hyperplane = mf.Hyperplane.CreateFromNormal(normal, size[i] + normal[i]*position[i])
        solid.boundaries.append(sld.Boundary(hyperplane,domain))
        normal[i] = 0.0

    return solid

squareA = CreateHypercube([2,2,2], [-1,-1,0])
print(squareA.VolumeIntegral(lambda x: 1.0), 4.0*4.0*4.0)
print(squareA.SurfaceIntegral(lambda x, n: n), 4.0*4.0*6.0)
squareB = CreateHypercube([1,1,1], [2,0,0])
print(squareB.VolumeIntegral(lambda x: 1.0), 2.0*2.0*2.0)
print(squareB.SurfaceIntegral(lambda x, n: n), 2.0*2.0*6.0)

interactor = InteractiveCanvas(squareA, squareB)
plt.show()