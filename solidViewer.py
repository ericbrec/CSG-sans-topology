from collections import namedtuple
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from manifold import Manifold
from hyperplane import Hyperplane
from bSpline import BSpline
from bspy import Spline, Viewer, SplineOpenGLFrame

def triangulate(solid):
    assert solid.dimension == 2

    # First, collect all manifold contour endpoints, accounting for slight numerical error.
    class Endpoint:
        def __init__(self, curve, t, clockwise, isStart, otherEnd=None):
            self.curve = curve
            self.t = t
            self.xy = curve.manifold.evaluate((t,))
            self.clockwise = clockwise
            self.isStart = isStart
            self.otherEnd = otherEnd
            self.connection = None
    endpoints = []
    for curve in solid.boundaries:
        curve.domain.boundaries.sort(key=lambda boundary: (boundary.manifold.evaluate(0.0), -boundary.manifold.normal(0.0)))
        firstPartB = 0
        secondPartB = 0
        boundaryCount = len(curve.domain.boundaries)
        while firstPartB < boundaryCount:
            if curve.domain.boundaries[firstPartB].manifold.normal(0.0) < 0.0:
                firstPartPoint = curve.domain.boundaries[firstPartB].manifold.evaluate(0.0)[0]
                while secondPartB < boundaryCount:
                    secondPartPoint = curve.domain.boundaries[secondPartB].manifold.evaluate(0.0)[0]
                    if firstPartPoint - Manifold.minSeparation < secondPartPoint and curve.domain.boundaries[secondPartB].manifold.normal(0.0) > 0.0:
                        t = curve.manifold.tangent_space(firstPartPoint)[:,0]
                        n = curve.manifold.normal(firstPartPoint)
                        clockwise = t[0] * n[1] - t[1] * n[0] > 0.0
                        ep1 = Endpoint(curve, firstPartPoint, clockwise, secondPartPoint >= firstPartPoint)
                        ep2 = Endpoint(curve, secondPartPoint, clockwise, secondPartPoint < firstPartPoint, ep1)
                        ep1.otherEnd = ep2
                        endpoints.append(ep1)
                        endpoints.append(ep2)
                        firstPartB = secondPartB
                        secondPartB += 1
                        break
                    secondPartB += 1
            firstPartB += 1

    # Second, collect all valid pairings of endpoints (normal not negated between segments).
    Connection = namedtuple('Connection', ('distance', 'ep1', 'ep2'))
    connections = []
    for i, ep1 in enumerate(endpoints[:-1]):
        for ep2 in endpoints[i+1:]:
            if (ep1.clockwise == ep2.clockwise and ep1.isStart != ep2.isStart) or \
                (ep1.clockwise != ep2.clockwise and ep1.isStart == ep2.isStart):
                connections.append(Connection(np.linalg.norm(ep1.xy - ep2.xy), ep1, ep2))

    # Third, only keep closest pairings (prune the rest).
    connections.sort(key=lambda connection: -connection.distance)
    while connections:
        connection = connections.pop()
        connection.ep1.connection = connection.ep2
        connection.ep2.connection = connection.ep1
        connections = [c for c in connections if c.ep1 is not connection.ep1 and c.ep1 is not connection.ep2 and \
                c.ep2 is not connection.ep1 and c.ep2 is not connection.ep2]
        
    # Fourth, set up GLUT to tesselate the solid.
    tess = gluNewTess()
    gluTessProperty(tess, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_ODD)
    vertices = []
    def beginCallback(type=None):
        vertices = []
    def edgeFlagDataCallback(flag, polygonData):
        pass # Forces triangulation of polygons rather than triangle fans or strips
    def vertexCallback(vertex, otherData=None):
        vertices.append(vertex[:2])
    def combineCallback(vertex, neighbors, neighborWeights, outData=None):
        outData = vertex
        return outData
    def endCallback():
        pass
    gluTessCallback(tess, GLU_TESS_BEGIN, beginCallback)
    gluTessCallback(tess, GLU_TESS_EDGE_FLAG_DATA, edgeFlagDataCallback)
    gluTessCallback(tess, GLU_TESS_VERTEX, vertexCallback)
    gluTessCallback(tess, GLU_TESS_COMBINE, combineCallback)
    gluTessCallback(tess, GLU_TESS_END, endCallback)

    # Fifth, trace the contours from pairing to pairing, using GLUT to tesselate the interior.
    gluTessBeginPolygon(tess, 0)
    while endpoints:
        start = endpoints[0]
        if not start.isStart:
            start = start.otherEnd
        # Run backwards until you hit start again or hit an end.
        if start.connection is not None:
            originalStart = start
            next = start.connection
            start = None
            while next is not None and start is not originalStart:
                start = next.otherEnd
                next = start.connection
        # Run forwards submitting vertices for the contour.
        next = start
        gluTessBeginContour(tess)
        while next is not None:
            endpoints.remove(next)
            endpoints.remove(next.otherEnd)
            subdivisions = max(int(abs(next.otherEnd.t - next.t) / 0.1), 20) if isinstance(next.curve.manifold, BSpline) else 2
            for t in np.linspace(next.t, next.otherEnd.t, subdivisions):
                xy = next.curve.manifold.evaluate((t,))
                vertex = (*xy, 0.0)
                gluTessVertex(tess, vertex, vertex)
            next = next.otherEnd.connection
            if next is start:
                break
        gluTessEndContour(tess)
    gluTessEndPolygon(tess)
    gluDeleteTess(tess)
    return np.array(vertices, np.float32)

class SolidViewer(Viewer):
    def __init__(self, *args, **kw):
        Viewer.__init__(self, *args, **kw)
        self.set_background_color(1.0, 1.0, 1.0)

    def list_boundary(self, boundary, name="Boundary", fillColor=None, lineColor=None, options=None, inherit=True, draw=False):
        if boundary.manifold.range_dimension() != 3:
            return
        Material = namedtuple("Material", ("fillColor", "lineColor", "options"))
        vertices = triangulate(boundary.domain)
        if not inherit or not hasattr(boundary.manifold, "material"):
            boundary.manifold.material = Material(fillColor, lineColor, options)
        material = boundary.manifold.material
        if isinstance(boundary.manifold, Hyperplane):
            uvMin = vertices.min(axis=0)
            uvMax = vertices.max(axis=0)
            xyzMinMin = boundary.manifold.evaluate(uvMin)
            xyzMinMax = boundary.manifold.evaluate((uvMin[0], uvMax[1]))
            xyzMaxMin = boundary.manifold.evaluate((uvMax[0], uvMin[1]))
            xyzMaxMax = boundary.manifold.evaluate(uvMax)
            spline = Spline(2, 3, (2, 2), (2, 2), 
                np.array((uvMin, uvMin, uvMax, uvMax), np.float32).T,
                np.array(((xyzMinMin, xyzMaxMin), (xyzMinMax, xyzMaxMax)), np.float32).T)
        elif isinstance(boundary.manifold, BSpline):
            spline = boundary.manifold.spline
        self.frame.make_drawable(spline)
        if "Name" not in spline.metadata:
            spline.metadata["Name"] = name
        if material.fillColor is not None:
            spline.metadata["fillColor"] = material.fillColor
        if material.lineColor is not None:
            spline.metadata["lineColor"] = material.lineColor
        if material.options is not None:
            spline.metadata["options"] = material.options
        if not hasattr(spline, "cache"):
            spline.cache = {}
        spline.cache["trim"] = vertices
        if draw:
            self.draw(spline)
        else:
            self.list(spline)

    def draw_boundary(self, boundary, name="Boundary", fillColor=None, lineColor=None, options=None, inherit=True):
        self.list_boundary(boundary, name, fillColor, lineColor, options, inherit, draw=True)

    def list_solid(self, solid, name="Solid", fillColor=None, lineColor=None, options=None, inherit=True, draw=False):
        for i, surface in enumerate(solid.boundaries):
            self.list_boundary(surface, f"{name} boundary {i+1}", fillColor, lineColor, options, inherit, draw)

    def draw_solid(self, solid, name="Solid", fillColor=None, lineColor=None, options=None, inherit=True):
        self.list_solid(solid, name, fillColor, lineColor, options, inherit, draw=True)