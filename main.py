import matplotlib.pyplot as plt
import numpy as np

N = np.array([1,2,-3])
n = N / np.linalg.norm(N)
print(n)

reflector = np.add(np.identity(3),np.outer(-2*n,n))
eigen = np.linalg.eigh(reflector)
print(eigen[0])
print(eigen[0][0])
print(eigen[1])
print(np.delete(eigen[1],0,1))

from matplotlib.backend_bases import MouseButton

x = []
y = []
points = 5
for i in range(points+1):
    x.append(np.cos(i*6.28/points))
    y.append(np.sin(i*6.28/points))

print(x)
print(y)

nt = (x[2]-x[0])*(y[3]-y[1]) + (y[2]-y[0])*(x[1]-x[3])
print(nt)
u = ((x[1]-x[0])*(y[3]-y[1]) + (y[1]-y[0])*(x[1]-x[3]))/nt
print(u, 1 - u)

print(x[0] + (x[2]-x[0])*u)
print(x[1] + (x[3]-x[1])*(1-u))
print(y[0] + (y[2]-y[0])*u)
print(y[1] + (y[3]-y[1])*(1-u))

class Solid:

    def __init__(self, dimension, boundaries):
        assert boundaries.dimension == dimension
        self.dimension = dimension
        self.boundaries = boundaries

class Boundary:

    def __init__(self, dimension, manifold, domain):
        assert manifold.dimension == dimension
        assert domain.dimension == dimension
        self.dimension = dimension
        self.manifold = manifold
        self.domain = domain

class Manifold:

    def __init__(self, dimension, normal, tangentSpace, point):
        assert dimension > 1
        assert len(normal) == dimension
        assert len(tangentSpace) == dimension
        assert (dimension-1 == 1 and type(tangentSpace[0]) != 'list') or \
            len(tangentSpace[0]) == dimension-1
        assert len(point) == dimension
        self.dimension = dimension
        self.normal = normal
        self.tangentSpace = tangentSpace
        self.point = point

    @staticmethod
    def TangentSpaceFromNormal(normal):
        # Construct the Householder reflection transform using the normal
        reflector = np.add(np.identity(3),np.outer(-2*normal,normal))
        # Compute the eigenvalues and eigenvectors for the symetric transform
        eigen = np.linalg.eigh(reflector)
        # Assert the first eigenvalue is negative (the reflection whose eigenvector is the normal)
        assert(eigen[0][0] < 0.0)
        # Return the tangent space by removing the first eigenvector column (the negated normal)
        return np.delete(eigen[1],0,1)

    def Intersect(self, manifold):
        assert self.dimension == manifold.dimension

        # First, the new self boundary
        iNormalSelf = np.dot(manifold.normal, self.tangentSpace)
        normalize = 1.0 / np.linalg.norm(iNormalSelf)
        iNormalSelf = normalize * iNormalSelf
        iOffsetSelf = normalize * np.dot(manifold.normal, np.np.subtract(manifold.point,self.point))
        iPointSelf = iOffsetSelf * iNormalSelf

        # Second, the new other boundary
        iNormalOther = np.dot(self.normal, manifold.tangentSpace)
        normalize = 1.0 / np.linalg.norm(iNormalOther)
        iNormalOther = normalize * iNormalOther
        iOffsetOther = normalize * np.dot(self.normal, np.np.subtract(self.point,manifold.point))
        iPointOther = iOffsetOther * iNormalOther

        newDimension = self.dimension - 1
        assert len(iNormalSelf) == newDimension
        assert len(iNormalOther) == newDimension

        #if newDimension > 1:
            #return Boundary(newDimension, Manifold(newDimension,iNormalSelf, Manifold.TangentSpaceFromNormal(iNormalSelf), iPointSelf),

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

# plt.show()