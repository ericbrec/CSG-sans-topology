import numpy as np
import manifold as mf
import solid as sld
import solidUtils as utils
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import player as player
import matplotlib.widgets as widgets

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
        elif key == "t":
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

    def on_radio_press(self, event):
        """Callback for radio button selection."""
        self.hyperplane = self.ConstructHyperplane(event)
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

        axRadioButtons = fig.add_axes([0.9, 0.88, 0.08, 0.12])
        buttons = ["x", "y", "z", 't']
        self.radioButtons = widgets.RadioButtons(axRadioButtons, buttons, active=buttons.index(axis))
        self.radioButtons.on_clicked(self.on_radio_press)

        self.solid = solid
        self.offset = 0.0
        self.hyperplane = self.ConstructHyperplane(axis)

        self.lines = art3d.Line3DCollection([], linewidth=1, color="blue")
        self.ax.add_collection(self.lines)

        self.player = player.Player(fig, self.animateSlice, -4.0, 4.0, 0.2, -4.0, init_func=self.initializeCanvas)


cube = utils.CreateHypercube([.75,1.5,.75], [0,0,0])
extrudedCube = utils.ExtrudeSolid(cube,[[-2,2,-2,-4],[2,-2,2,4]])
star = utils.CreateStar(1.0, [0.0, 0.0], 90.0*6.28/360.0)
starBlock = utils.ExtrudeSolid(star,[[0,0,-1],[0,0,1]])
extrudedStarBlock = utils.ExtrudeSolid(starBlock,[[-2,-2,-2,-4],[2,2,2,4]])
combined = extrudedStarBlock.Union(extrudedCube)

InteractiveCanvas(combined, 't')
plt.show()