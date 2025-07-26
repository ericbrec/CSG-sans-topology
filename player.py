import mpl_toolkits.axes_grid1
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation

class Player(FuncAnimation):
    def __init__(self, fig, func, min = 0.0, max = 100.0, step = 1.0, initial=None, init_func=None, fargs=None, pos=(0.17, 0.94), **kwargs):
        self.min = min
        self.max = max
        self.step = step
        self.value = min if initial is None else initial
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.update, frames=self.play(), init_func=init_func, fargs=fargs, cache_frame_data=False, **kwargs)

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

    def oneForward(self, event=None):
        self.forwards = True
        self.oneStep()

    def oneBackward(self, event=None):
        self.forwards = False
        self.oneStep()

    def oneStep(self):
        self.value += self.step if self.forwards else -self.step
        self.value = min(max(self.value, self.min), self.max)
        self.artists = self.func(self.value)
        self.slider.set_val(self.value)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        axPlayer = self.fig.add_axes([pos[0],pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(axPlayer)
        axB = divider.append_axes("secondpart", size="80%", pad=0.05)
        axS = divider.append_axes("secondpart", size="80%", pad=0.05)
        axF = divider.append_axes("secondpart", size="80%", pad=0.05)
        axOldF = divider.append_axes("secondpart", size="100%", pad=0.05)
        axSlider = divider.append_axes("secondpart", size="500%", pad=0.07)
        self.button_oneBack = widgets.Button(axPlayer, label='$\u29CF$')
        self.button_back = widgets.Button(axB, label='$\u25C0$')
        self.button_stop = widgets.Button(axS, label='$\u25A0$')
        self.button_forward = widgets.Button(axF, label='$\u25B6$')
        self.button_oneForward = widgets.Button(axOldF, label='$\u29D0$')
        self.button_oneBack.on_clicked(self.oneBackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneForward.on_clicked(self.oneForward)
        self.slider = widgets.Slider(axSlider, '', self.min, self.max, valinit=self.value)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, value):
        self.value = self.slider.val
        self.artists = self.func(self.value)

    def update(self, value):
        self.slider.set_val(value)
        return self.artists