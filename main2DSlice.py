import numpy as np
import manifold as mf
import solid as sld
import solidUtils as utils
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import matplotlib.widgets
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.backend_bases import MouseButton

class Player(FuncAnimation):
    def __init__(self, fig, func, min = 0.0, max = 100.0, step = 1.0, initial=None, init_func=None, fargs=None,
                 save_count=None, pos=(0.17, 0.94), **kwargs):
        self.min = min
        self.max = max
        self.step = step
        self.value = min if initial is None else initial
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), init_func=init_func, fargs=fargs, save_count=save_count, **kwargs)

    def play(self):
        while self.runs:
            self.value += self.step if self.forwards else -self.step
            if self.value > self.min and self.value < self.max:
                yield self.value
            else:
                self.value = min(max(self.value, self.min), self.max)
                self.stop()
                yield self.value

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        self.value += self.step if self.forwards else -self.step
        self.value = min(max(self.value, self.min), self.max)
        self.artists = self.func(self.value)
        self.slider.set_val(self.value)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        axPlayer = self.fig.add_axes([pos[0],pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(axPlayer)
        axB = divider.append_axes("right", size="80%", pad=0.05)
        axS = divider.append_axes("right", size="80%", pad=0.05)
        axF = divider.append_axes("right", size="80%", pad=0.05)
        axOldF = divider.append_axes("right", size="100%", pad=0.05)
        axSlider = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(axPlayer, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(axB, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(axS, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(axF, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(axOldF, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(axSlider, '', self.min, self.max, valinit=self.value)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, value):
        self.value = self.slider.val
        self.artists = self.func(self.value)

    def update(self, value):
        self.slider.set_val(value)
        return self.artists

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

        self.player = Player(fig, self.animateSlice, -4.0, 4.0, 0.25, 0.0, init_func=self.initializeCanvas)

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