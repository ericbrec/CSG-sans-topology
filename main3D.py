import numpy as np
import solidUtils as utils
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import mpl_toolkits.mplot3d.proj3d as proj3d
from matplotlib.backend_bases import MouseButton
import matplotlib.widgets as widgets

class InteractiveCanvas:
    
    def PerformBooleanOperation(self, op):
        print(op)
        if op == 'OR':
            solid = self.solidA.Union(self.solidB)
            self.op = op
        elif op == 'AND':
            solid = self.solidA.Intersection(self.solidB)
            self.op = op
        elif op == 'DIFF':
            solid = self.solidA.Difference(self.solidB)
            self.op = op
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

        axRadioButtons = fig.add_axes([0.85, 0.88, 0.12, 0.12])
        self.radioButtons = widgets.RadioButtons(axRadioButtons, ["OR", "AND", "DIFF"])
        self.radioButtons.on_clicked(self.on_radio_press)

        self.origin = [0.0]*solidB.dimension 

        self.solidA = solidA
        self.solidB = solidB
        self.solidC = self.PerformBooleanOperation('OR') # First radio button
        
        self.linesA = art3d.Line3DCollection(utils.CreateSegmentsFromSolid(self.solidA), linewidth=1, color="blue")
        self.linesB = art3d.Line3DCollection(utils.CreateSegmentsFromSolid(self.solidB), linewidth=1, color="orange")
        self.linesC = art3d.Line3DCollection(utils.CreateSegmentsFromSolid(self.solidC), linewidth=3, color="red")
        
        self.ax.set(xlabel="x", ylabel="y", zlabel="z")
        self.ax.set(xlim3d = (-4, 4), ylim3d = (-4, 4), zlim3d = (-4, 4))

        self.ax.add_collection(self.linesA)
        self.ax.add_collection(self.linesB)
        self.ax.add_collection(self.linesC)

        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_draw(self, event):
        """Callback for draws."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.linesC)
        self.ax.draw_artist(self.linesB)
        self.ax.draw_artist(self.linesA)

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if event.inaxes is not self.ax or event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return

        self.ax.disable_mouse_rotation()
        self.origin = self.GetPointFromEvent(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.inaxes is not self.ax or event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return

        self.ax.mouse_init()
        self.solidC = self.PerformBooleanOperation(self.op)
        self.linesC.set_segments(utils.CreateSegmentsFromSolid(self.solidC))
        self.canvas.draw()

    def on_radio_press(self, event):
        """Callback for radio button selection."""
        self.solidC = self.PerformBooleanOperation(event)
        self.linesC.set_segments(utils.CreateSegmentsFromSolid(self.solidC))
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if event.inaxes is not self.ax or event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return
        
        point = self.GetPointFromEvent(event)
        delta = point - self.origin
        self.solidB.Translate(delta)
        self.origin = point

        self.linesB.set_segments(utils.CreateSegmentsFromSolid(self.solidB))
        
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.linesC)
        self.ax.draw_artist(self.linesB)
        self.ax.draw_artist(self.linesA)
        self.canvas.blit(self.ax.bbox)

cubeA = utils.CreateHypercube([2,2,2], [0,0,0])
print(cubeA.VolumeIntegral(lambda x: 1.0), 4.0*4.0*4.0)
print(cubeA.SurfaceIntegral(lambda x, n: n), 4.0*4.0*6.0)
print(cubeA.WindingNumber([1,1,0]))
print(cubeA.WindingNumber([4,1,0]))
cubeB = utils.CreateHypercube([1,1,1], [1,1,1])
print(cubeB.VolumeIntegral(lambda x: 1.0), 2.0*2.0*2.0)
print(cubeB.SurfaceIntegral(lambda x, n: n), 2.0*2.0*6.0)

canvas = InteractiveCanvas(cubeA, cubeB)
plt.show()