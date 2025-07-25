from solid import Solid
from hyperplane import Hyperplane
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
            self.ax.set_title("cutout solid with {axis}-axis".format(axis=key))
            hyperplane = Hyperplane.create_axis_aligned(self.solid.dimension, axis, self.offset)
        else:
            hyperplane = self.hyperplane

        return hyperplane
    
    def initializeCanvas(self):
        self.ax.set(xlabel="x", ylabel="y", zlabel="z")
        self.ax.set(xlim3d = (-4, 4), ylim3d = (-4, 4), zlim3d = (-4, 4))

    def animateCutout(self, offset):
        self.offset = offset
        self.hyperplane._point = self.offset * self.hyperplane._normal
        self.cutout = self.solid.compute_cutout(self.hyperplane)
        if not isinstance(self.cutout, Solid):
            self.cutout = Solid(self.solid.dimension, False)
        self.lines.set_segments(utils.create_segments_from_solid(self.cutout))

    def on_radio_press(self, event):
        """Callback for radio button selection."""
        self.hyperplane = self.ConstructHyperplane(event)
        self.cutout = self.solid.compute_cutout(self.hyperplane)
        if not isinstance(self.cutout, Solid):
            self.cutout = Solid(self.solid.dimension, False)
        self.lines.set_segments(utils.create_segments_from_solid(self.cutout))
        self.canvas.draw()

    def __init__(self, solid, axis):

        fig = plt.figure(figsize=(6, 6))
        self.ax = fig.add_subplot(projection='3d')
        self.ax.set_title("cutout solid with {axis}-axis".format(axis=axis))
        self.ax.axis('scaled')
        self.ax.axis('off')
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

        self.player = player.Player(fig, self.animateCutout, -4.0, 4.0, 0.2, -4.0, init_func=self.initializeCanvas)

if __name__ == "__main__":
    cube = Hyperplane.create_hypercube([1.5, 3.0, 1.5], [-.75, -1.5, -.75])
    extrudedCube = utils.extrude_solid(cube,[[-2,2,-2,-4],[2,-2,2,4]])
    star = utils.create_star(1.0, [0.0, 0.0], 90.0*6.28/360.0)
    starBlock = utils.extrude_solid(star,[[0,0,-1],[0,0,1]])
    extrudedStarBlock = utils.extrude_solid(starBlock,[[-2,-2,-2,-4],[2,2,2,4]])
    combined = extrudedStarBlock.union(extrudedCube)

    canvas = InteractiveCanvas(combined, 't')
    plt.show()