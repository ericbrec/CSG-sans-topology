import numpy as np
import manifold as mf
import solid as sld
import solidUtils as utils
import matplotlib.pyplot as plt
import player as player
from matplotlib.collections import LineCollection

class InteractiveCanvas:
    
    def ConstructHyperplane(self, key):
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
            self.ax.set_title("Slice solid by {axis}-axis".format(axis=key))
            hyperplane = utils.HyperplaneAxisAligned(self.solid.dimension, axis, self.offset)
        else:
            hyperplane = self.hyperplane

        return hyperplane
    
    def initializeCanvas(self):
        self.ax.set(xlim = (-4, 4), ylim = (-4, 4))

    def animateSlice(self, offset):
        self.offset = offset
        self.hyperplane.point = self.offset * self.hyperplane.normal
        self.slice = self.solid.Slice(self.hyperplane)
        if not isinstance(self.slice, sld.Solid):
            self.slice = sld.Solid(self.solid.dimension)
        self.lines.set_segments(utils.CreateSegmentsFromSolid(self.slice))

    def on_key_press(self, event):
        """Callback for key presses."""
        self.hyperplane = self.ConstructHyperplane(event.key)
        self.slice = self.solid.Slice(self.hyperplane)
        if not isinstance(self.slice, sld.Solid):
            self.slice = sld.Solid(self.solid.dimension)
        self.lines.set_segments(utils.CreateSegmentsFromSolid(self.slice))
        self.canvas.draw()

    def __init__(self, solid, axis):

        fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("Slice solid by {axis}-axis".format(axis=axis))
        self.canvas = self.ax.figure.canvas

        self.solid = solid
        self.offset = 0.0
        self.hyperplane = self.ConstructHyperplane(axis)

        self.lines = LineCollection([], linewidth=1, color="blue")
        self.ax.add_collection(self.lines)

        self.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.player = player.Player(fig, self.animateSlice, -4.0, 4.0, 0.25, 0.0, init_func=self.initializeCanvas)

star = utils.CreateStar(2.0, [-1.0, -1.0], 90.0*6.28/360.0)
cube = utils.ExtrudeSolid(star,[[0,0,0],[2,2,2],[1,1,3]])
# print(cubeA.VolumeIntegral(lambda x: 1.0), 4.0*4.0*4.0)
# print(cubeA.SurfaceIntegral(lambda x, n: n), 4.0*4.0*6.0)
# print(cubeA.WindingNumber([1,1,0]))
# print(cubeA.WindingNumber([4,1,0]))
# cubeB = utils.CreateHypercube([1,1,1], [-1,1,1])
# print(cubeB.VolumeIntegral(lambda x: 1.0), 2.0*2.0*2.0)
# print(cubeB.SurfaceIntegral(lambda x, n: n), 2.0*2.0*6.0)
# cube = cubeA.Difference(cubeB)
print(cube.VolumeIntegral(lambda x: 1.0), 4.0*4.0*4.0 - 2.0*2.0*2.0)
print(cube.SurfaceIntegral(lambda x, n: n), 4.0*4.0*6.0)
print(cube.WindingNumber([0,0,0]))
print(cube.WindingNumber([4,1,0]))

interactor = InteractiveCanvas(cube, 'z')
plt.show()