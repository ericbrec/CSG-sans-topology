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
        vertices = [[0.0,0.0]]
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

        fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title('Drag shape to update path')
        self.canvas = self.ax.figure.canvas

        self.origin = [0.0]*solidB.dimension 

        self.solidA = solidA
        self.solidB = solidB
        self.solidC = self.PerformBooleanOperation('u')

        self.patchA = patches.PathPatch(self.CreatePathFromSolid(self.solidA), linewidth=1, color="blue")
        self.patchB = patches.PathPatch(self.CreatePathFromSolid(self.solidB), linewidth=1, color="orange")
        self.patchC = patches.PathPatch(self.CreatePathFromSolid(self.solidC), linewidth=3, color="red")
        
        self.ax.set(xlim = (-4, 4), ylim = (-4, 4))

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

def CreateStar(radius, center, angle):
    vertices = []
    points = 5
    for i in range(points):
        vertices.append([radius*np.cos(angle - ((2*i)%points)*6.28/points) + center[0], radius*np.sin(angle - ((2*i)%points)*6.28/points) + center[1]])

    nt = (vertices[1][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[1][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0])
    u = ((vertices[3][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[3][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0]))/nt

    star = sld.Solid.CreateSolidFromPoints(2, vertices)
    for boundary in star.boundaries:
        u0 = boundary.domain.boundaries[0].manifold.point[0]
        u1 = boundary.domain.boundaries[1].manifold.point[0]
        boundary.domain.boundaries.append(sld.Boundary(mf.Hyperplane.CreateFromNormal(1.0, u0 + (1.0 - u)*(u1 - u0))))
        boundary.domain.boundaries.append(sld.Boundary(mf.Hyperplane.CreateFromNormal(-1.0, -(u0 + u*(u1 - u0)))))

    return star

triangleA = sld.Solid.CreateSolidFromPoints(2, [[1,0],[0,0],[0,1]])
print(triangleA.VolumeIntegral(lambda x: 1.0), 0.5)
print(triangleA.SurfaceIntegral(lambda x, n: n), 2 + np.sqrt(2.0))
print(triangleA.WindingNumber(np.array([.75,.75])))
print(triangleA.WindingNumber(np.array([.5,.5])))
print(triangleA.WindingNumber(np.array([.25,.25])))

squareA = sld.Solid.CreateSolidFromPoints(2, [[-3,-3],[-3,1],[1,1],[1,-3]])
print(squareA.VolumeIntegral(lambda x: 1.0), 4.0*4.0)
print(squareA.SurfaceIntegral(lambda x, n: n), 4.0*4.0)
print(squareA.WindingNumber(np.array([0.,0.])))
print(squareA.WindingNumber(np.array([-0.23870968,1.])))
squareB = sld.Solid.CreateSolidFromPoints(2, [[1,1],[3,1],[3,-1],[1,-1]])
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

#interactor = InteractiveCanvas(squareA, squareB)
interactor = InteractiveCanvas(starA, starB)
#interactor = InteractiveCanvas(squareA, starB)
plt.show()