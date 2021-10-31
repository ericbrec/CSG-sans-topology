import numpy as np
import manifold as mf
import solid as sld
import solidUtils as utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.backend_bases import MouseButton

class InteractiveCanvas:
    
    def PerformSlice(self, key):
        print(key)
        axis = -1
        if key == 'x':
            axis = 0
        elif key == 'y':
            axis = 1
        elif key == 'z':
            axis = 2

        if axis >= 0:
            self.key = key
            self.hyperplane = utils.HyperplaneAxisAligned(self.solid.dimension, axis, 0.0)
            slice = self.solid.Slice(self.hyperplane)
            if not isinstance(slice, sld.Solid):
                slice = sld.Solid(self.solid.dimension)
        else:
            slice = self.slice

        return slice
    
    def initializeCanvas(self):
        self.ax.set(xlim = (-4, 4), ylim = (-4, 4))
        return [self.lines]
    
    def animateSlice(self, offset):
        self.hyperplane.point = offset * self.hyperplane.normal
        self.slice = self.solid.Slice(self.hyperplane)
        if not isinstance(self.slice, sld.Solid):
            self.slice = sld.Solid(self.solid.dimension)
        self.lines.set_segments(utils.CreateSegmentsFromSolid(self.slice))
        return [self.lines]

    def on_key_press(self, event):
        """Callback for key presses."""
        self.slice = self.PerformSlice(event.key)
        self.lines.set_segments(utils.CreateSegmentsFromSolid(self.slice))
        self.canvas.draw()

    def __init__(self, solid, axis):

        fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title('Slice solid')
        self.canvas = self.ax.figure.canvas

        self.solid = solid
        self.slice = self.PerformSlice(axis)

        self.lines = LineCollection(utils.CreateSegmentsFromSolid(self.slice), linewidth=1, color="blue")
        self.ax.add_collection(self.lines)

        self.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.animation = animation.FuncAnimation(fig, self.animateSlice, np.arange(-4.0, 4.0, 0.25), init_func=self.initializeCanvas, blit=True)

cubeA = utils.CreateHypercube([2,2,2], [0,0,0])
print(cubeA.VolumeIntegral(lambda x: 1.0), 4.0*4.0*4.0)
print(cubeA.SurfaceIntegral(lambda x, n: n), 4.0*4.0*6.0)
print(cubeA.WindingNumber([1,1,0]))
print(cubeA.WindingNumber([4,1,0]))
cubeB = utils.CreateHypercube([1,1,1], [-1,1,1])
print(cubeB.VolumeIntegral(lambda x: 1.0), 2.0*2.0*2.0)
print(cubeB.SurfaceIntegral(lambda x, n: n), 2.0*2.0*6.0)
cube = cubeA.Difference(cubeB)
print(cube.VolumeIntegral(lambda x: 1.0), 4.0*4.0*4.0 - 2.0*2.0*2.0)
print(cube.SurfaceIntegral(lambda x, n: n), 4.0*4.0*6.0)
print(cube.WindingNumber([0,0,0]))
print(cube.WindingNumber([4,1,0]))

interactor = InteractiveCanvas(cube, 'x')
plt.show()