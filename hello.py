msg = "Hello World"
print(msg)

import matplotlib.pyplot as plt
import numpy as np

N = np.array([1,1,0])
n = N / np.linalg.norm(N)
print(n)

reflector = np.add(np.identity(3),np.outer(-2*n,n))
print(reflector)
eigen = np.linalg.eigh(reflector)
print(eigen[0])
print(np.transpose(eigen[1]))
print(np.transpose(eigen[1])[0])

from matplotlib.backend_bases import MouseButton

x = []
y = []
points = 5
for i in range(points+1):
    x.append(np.cos(i*6.28/points))
    y.append(np.sin(i*6.28/points))

print(x)
print(y)

class InteractiveCanvas:

    showverts = True
    epsilon = 8  # max pixel distance to count as a vertex hit

    def __init__(self, x, y):

        fig, self.ax = plt.subplots()
        self.ax.set_title('drag vertices to update path')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.canvas = self.ax.figure.canvas

        self._ind = None

        self.line, = self.ax.plot(
            x, y, color='blue', marker='o', markerfacecolor='r', markersize=self.epsilon, animated=True)

        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_draw(self, event):
        """Callback for draws."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if (event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts):
            return
        # self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if (event.button != MouseButton.LEFT
                or not self.showverts):
            return
        # self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if (self._ind is None
                or event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts):
            return

        #vertices = self.pathpatch.get_path().vertices

        #vertices[self._ind] = event.xdata, event.ydata
        #self.line.set_data(zip(*vertices))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


interactor = InteractiveCanvas(x, y)

plt.show()