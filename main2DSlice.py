import numpy as np
import manifold as mf
import solid as sld
import solidUtils as utils
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backend_bases import MouseButton

class InteractiveCanvas:
    
    def PerformBooleanOperation(self, key):
        print(key)
        if key == 's':
            slice = self.solid.Slice(self.manifold)
            if not isinstance(slice, sld.Solid):
                slice = sld.Solid(self.solid.dimension)
            self.key = key
        else:
            slice = self.slice

        return slice

    def __init__(self, solid, manifold):
        assert solid.dimension == manifold.GetRangeDimension()

        fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title('Drag shape to update slice')
        self.canvas = self.ax.figure.canvas

        self.origin = [0.0]*solid.dimension 

        self.solid = solid
        self.manifold = manifold
        self.slice = self.PerformBooleanOperation('s')

        self.lines = LineCollection(utils.CreateSegmentsFromSolid(self.slice), linewidth=1, color="blue")
        
        self.ax.set(xlim = (-4, 4), ylim = (-4, 4))

        self.ax.add_collection(self.lines)

        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_draw(self, event):
        """Callback for draws."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.lines)

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if event.inaxes is None or event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return
        self.origin[0] = event.xdata

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return

        self.slice = self.PerformBooleanOperation(self.key)
        self.lines.set_segments(utils.CreateSegmentsFromSolid(self.slice))
        self.canvas.draw()

    def on_key_press(self, event):
        """Callback for key presses."""
        self.slice = self.PerformBooleanOperation(event.key)
        self.lines.set_segments(utils.CreateSegmentsFromSolid(self.slice))
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if event.inaxes is None or event.button != MouseButton.LEFT or self.ax.get_navigate_mode() is not None:
            return
        
        delta = [0.0]*self.solid.dimension
        delta[0] = event.xdata - self.origin[0]
        self.manifold.Translate(delta)
        self.origin[0] = event.xdata
        
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.lines)
        self.canvas.blit(self.ax.bbox)

cube = utils.CreateHypercube([2,2,2], [0,0,0])
intersections = cube.boundaries[0].manifold.IntersectManifold(cube.boundaries[3].manifold)
print(cube.VolumeIntegral(lambda x: 1.0), 4.0*4.0*4.0)
print(cube.SurfaceIntegral(lambda x, n: n), 4.0*4.0*6.0)
print(cube.WindingNumber([0,0,0]))
print(cube.WindingNumber([4,1,0]))

manifold = utils.HyperplaneAxisAligned(3, 0, 0.0)

interactor = InteractiveCanvas(cube, manifold)
plt.show()