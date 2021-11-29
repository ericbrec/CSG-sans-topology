import numpy as np
import manifold as mf
import solid as sld
import solidUtils as utils
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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

    def __init__(self, solidA, solidB):
        assert solidA.dimension == solidB.dimension

        fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title('Drag shape to update solid')
        self.canvas = self.ax.figure.canvas

        axRadioButtons = fig.add_axes([0.85, 0.88, 0.12, 0.12])
        self.radioButtons = widgets.RadioButtons(axRadioButtons, ["OR", "AND", "DIFF"])
        self.radioButtons.on_clicked(self.on_radio_press)

        self.origin = [0.0]*solidB.dimension 

        self.solidA = solidA
        self.solidB = solidB
        self.solidC = self.PerformBooleanOperation('OR') # First radio button

        self.linesA = LineCollection(utils.CreateSegmentsFromSolid(self.solidA), linewidth=1, color="blue")
        self.linesB = LineCollection(utils.CreateSegmentsFromSolid(self.solidB), linewidth=1, color="orange")
        self.linesC = LineCollection(utils.CreateSegmentsFromSolid(self.solidC), linewidth=3, color="red")
        
        self.ax.set(xlim = (-4, 4), ylim = (-4, 4))

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
        self.origin[0] = event.xdata
        self.origin[1] = event.ydata

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.inaxes is not self.ax or event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return

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
        
        delta = [0.0]*self.solidB.dimension
        delta[0] = event.xdata - self.origin[0]
        delta[1] = event.ydata - self.origin[1]
        self.solidB.Translate(delta)
        self.origin[0] = event.xdata
        self.origin[1] = event.ydata

        self.linesB.set_segments(utils.CreateSegmentsFromSolid(self.solidB))
        
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.linesC)
        self.ax.draw_artist(self.linesB)
        self.ax.draw_artist(self.linesA)
        self.canvas.blit(self.ax.bbox)

triangleA = utils.CreateSolidFromPoints(2, [[2,2],[2,-4],[-4,-4]])
print(triangleA.VolumeIntegral(lambda x: 1.0), 18)
print(triangleA.SurfaceIntegral(lambda x, n: n), 12 + 6*np.sqrt(2.0))
print(triangleA.WindingNumber(np.array([-2,-1])))
print(triangleA.WindingNumber(np.array([-1,-1])))
print(triangleA.WindingNumber(np.array([0,-1])))

squareA = utils.CreateHypercube([2,2], [-1,-1])
print(squareA.VolumeIntegral(lambda x: 1.0), 4.0*4.0)
print(squareA.SurfaceIntegral(lambda x, n: n), 4.0*4.0)
print(squareA.WindingNumber(np.array([0.,0.])))
print(squareA.WindingNumber(np.array([-0.23870968,1.])))
squareB = utils.CreateHypercube([1,1], [1.2,-0.005])
print(squareB.VolumeIntegral(lambda x: 1.0), 2.0*2.0)
print(squareB.SurfaceIntegral(lambda x, n: n), 2.0*4.0)

starArea = 10.0 * np.tan(np.pi / 10.0) / (3.0 - np.tan(np.pi / 10.0)**2)
starPerimeter = 10.0 * np.cos(2.0*np.pi/5.0) * (np.tan(2.0*np.pi/5.0) - np.tan(np.pi/5.0))
starA = utils.CreateStar(2.0, [-1.0, -1.0], 90.0*6.28/360.0)
print(starA.VolumeIntegral(lambda x: 1.0), starArea * 4.0)
print(starA.SurfaceIntegral(lambda x, n: n), starPerimeter * 2.0)
print(starA.WindingNumber(np.array([-1.,-1.])))
starB = utils.CreateStar(1.0, [2.0, 2.0], 90.0*6.28/360.0)
print(starB.VolumeIntegral(lambda x: 1.0), starArea)
print(starB.SurfaceIntegral(lambda x, n: n), starPerimeter)
starB.Translate([-2.31895479, -2.69507693])

#interactor = InteractiveCanvas(triangleA, squareA)
#interactor = InteractiveCanvas(squareA, squareB)
interactor = InteractiveCanvas(starA, starB)
#interactor = InteractiveCanvas(squareA, starB)
plt.show()