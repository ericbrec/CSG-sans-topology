from collections import namedtuple
import numpy as np
from OpenGL.GLU import *
from manifold import Manifold
from spline import Spline

def triangulate(solid):
    assert solid.dimension == 2

    # First, collect all manifold contour endpoints, accounting for slight numerical error.
    class Endpoint:
        def __init__(self, curve, t, clockwise, isStart, otherEnd=None):
            self.curve = curve
            self.t = t
            self.xy = curve.manifold.point(t)
            self.clockwise = clockwise
            self.isStart = isStart
            self.otherEnd = otherEnd
            self.connection = None
    endpoints = []
    for curve in solid.boundaries:
        curve.domain.boundaries.sort(key=lambda boundary: (boundary.manifold.point(0.0), -boundary.manifold.normal(0.0)))
        leftB = 0
        rightB = 0
        boundaryCount = len(curve.domain.boundaries)
        while leftB < boundaryCount:
            if curve.domain.boundaries[leftB].manifold.normal(0.0) < 0.0:
                leftPoint = curve.domain.boundaries[leftB].manifold.point(0.0)
                while rightB < boundaryCount:
                    rightPoint = curve.domain.boundaries[rightB].manifold.point(0.0)
                    if leftPoint - Manifold.minSeparation < rightPoint and curve.domain.boundaries[rightB].manifold.normal(0.0) > 0.0:
                        t = curve.manifold.tangent_space(leftPoint)[:,0]
                        n = curve.manifold.normal(leftPoint)
                        clockwise = t[0] * n[1] - t[1] * n[0] > 0.0
                        ep1 = Endpoint(curve, leftPoint, clockwise, rightPoint >= leftPoint)
                        ep2 = Endpoint(curve, rightPoint, clockwise, rightPoint < leftPoint, ep1)
                        ep1.otherEnd = ep2
                        endpoints.append(ep1)
                        endpoints.append(ep2)
                        leftB = rightB
                        rightB += 1
                        break
                    rightB += 1
            leftB += 1

    # Second, collect all valid pairings of endpoints (normal not flipped between segments).
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
            subdivisions = int(abs(next.otherEnd.t - next.t) / 0.1) if isinstance(start.curve.manifold, Spline) else 2
            for t in np.linspace(next.t, next.otherEnd.t, subdivisions):
                xy = next.curve.manifold.point(t)
                vertex = (*xy, 0.0)
                gluTessVertex(tess, vertex, vertex)
            next = next.otherEnd.connection
            if next is start:
                break
        gluTessEndContour(tess)
    gluTessEndPolygon(tess)
    gluDeleteTess(tess)
    return vertices