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
    def __init__(self, fig, func, min = 0.0, max = 100.0, step = 1.0, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, pos=(0.125, 0.92), **kwargs):
        self.value = min
        self.min = min
        self.max = max
        self.step = step
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), 
                                           init_func=init_func, fargs=fargs,
                                           save_count=save_count, **kwargs )    

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
        self.func(self.value)
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
        self.slider = matplotlib.widgets.Slider(axSlider, '', 
                                                self.min, self.max, valinit=self.value)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, value):
        self.value = self.slider.val
        self.func(self.value)

    def update(self, value):
        self.slider.set_val(value)

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
        # TODO: Initialize hyperplace based on self.key
        self.hyperplane = utils.HyperplaneAxisAligned(self.solid.dimension, 0, 0.0)
    
    def animateSlice(self, offset):
        self.hyperplane.point = offset * self.hyperplane.normal
        self.slice = self.solid.Slice(self.hyperplane)
        if not isinstance(self.slice, sld.Solid):
            self.slice = sld.Solid(self.solid.dimension)
        self.lines.set_segments(utils.CreateSegmentsFromSolid(self.slice))

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
        self.key = axis
        self.slice = sld.Solid(self.solid.dimension)

        self.lines = LineCollection(utils.CreateSegmentsFromSolid(self.slice), linewidth=1, color="blue")
        self.ax.add_collection(self.lines)

        self.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.player = Player(fig, self.animateSlice, -4.0, 4.0, 0.25, init_func=self.initializeCanvas)

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