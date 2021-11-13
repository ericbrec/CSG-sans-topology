import numpy as np
import manifold as mf
import solid as sld
import solidUtils as utils
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import player as player

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
        elif key == "w":
            axis = 3

        if axis >= 0:
            self.key = key
            self.ax.set_title("Slice solid by {axis}-axis".format(axis=key))
            hyperplane = utils.HyperplaneAxisAligned(self.solid.dimension, axis, self.offset)
        else:
            hyperplane = self.hyperplane

        return hyperplane
    
    def initializeCanvas(self):
        self.ax.set(xlabel="x", ylabel="y", zlabel="z")
        self.ax.set(xlim3d = (-4, 4), ylim3d = (-4, 4), zlim3d = (-4, 4))

    def animateSlice(self, offset):
        self.offset = offset
        self.hyperplane.point = self.offset * self.hyperplane.normal
        self.slice = self.solid.Slice(self.hyperplane)
        if not isinstance(self.slice, sld.Solid):
            self.slice = sld.Solid(self.solid.dimension, False)
        self.lines.set_segments(utils.CreateSegmentsFromSolid(self.slice))

    def on_key_press(self, event):
        """Callback for key presses."""
        self.hyperplane = self.ConstructHyperplane(event.key)
        self.slice = self.solid.Slice(self.hyperplane)
        if not isinstance(self.slice, sld.Solid):
            self.slice = sld.Solid(self.solid.dimension, False)
        self.lines.set_segments(utils.CreateSegmentsFromSolid(self.slice))
        self.canvas.draw()

    def __init__(self, solid, axis):

        fig = plt.figure(figsize=(6, 6))
        self.ax = fig.add_subplot(projection='3d')
        self.ax.set_title("Slice solid by {axis}-axis".format(axis=axis))
        self.canvas = self.ax.figure.canvas

        self.solid = solid
        self.offset = 0.0
        self.hyperplane = self.ConstructHyperplane(axis)

        self.lines = art3d.Line3DCollection([], linewidth=1, color="blue")
        self.ax.add_collection(self.lines)

        self.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.player = player.Player(fig, self.animateSlice, -4.0, 4.0, 0.2, -4.0, init_func=self.initializeCanvas)


cube = utils.CreateHypercube([.75,1.5,.75], [0,0,0])
extrudedCube = utils.ExtrudeSolid(cube,[[-2,2,-2,-4],[2,-2,2,4]])
star = utils.CreateStar(1.0, [0.0, 0.0], 90.0*6.28/360.0)
starBlock = utils.ExtrudeSolid(star,[[0,0,-1],[0,0,1]])
extrudedStarBlock = utils.ExtrudeSolid(starBlock,[[-2,-2,-2,-4],[2,2,2,4]])
combined = extrudedStarBlock.Union(extrudedCube)

interactor = InteractiveCanvas(combined, 'w')
plt.show()