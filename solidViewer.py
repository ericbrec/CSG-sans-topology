from collections import namedtuple
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from solid import Solid, Boundary
from manifold import Manifold
from hyperplane import Hyperplane
from bSpline import BSpline
from bspy import Spline, Viewer, SplineOpenGLFrame
import solidUtils as utils

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
        leftB = 0
        rightB = 0
        boundaryCount = len(curve.domain.boundaries)
        while leftB < boundaryCount:
            if curve.domain.boundaries[leftB].manifold.normal(0.0) < 0.0:
                leftPoint = curve.domain.boundaries[leftB].manifold.evaluate(0.0)[0]
                while rightB < boundaryCount:
                    rightPoint = curve.domain.boundaries[rightB].manifold.evaluate(0.0)[0]
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

class SolidOpenGLFrame(SplineOpenGLFrame):

    surfaceFragmentShaderCode = """
        #version 410 core
     
        flat in SplineInfo
        {
            int uOrder, vOrder;
            int uN, vN;
            int uKnot, vKnot;
            float uFirst, vFirst;
            float uSpan, vSpan;
            float u, v;
            float uInterval, vInterval;
        } inData;
        in vec3 worldPosition;
        in vec3 splineColor;
        in vec3 normal;
        in vec2 parameters;
        in vec2 pixelPer;

        uniform vec4 uFillColor;
        uniform vec4 uLineColor;
        uniform vec3 uLightDirection;
        uniform int uOptions;
        uniform sampler2D uTextureMap;

        out vec4 color;
     
        void main()
        {
        	vec2 tex = vec2((parameters.x - inData.uFirst) / inData.uSpan, (parameters.y - inData.vFirst) / inData.vSpan);
            float specular = pow(abs(dot(normal, normalize(uLightDirection + worldPosition / length(worldPosition)))), 25.0);
            bool line = (uOptions & (1 << 2)) > 0 && (pixelPer.x * (parameters.x - inData.uFirst) < 1.5 || pixelPer.x * (inData.uFirst + inData.uSpan - parameters.x) < 1.5);
            line = line || ((uOptions & (1 << 2)) > 0 && (pixelPer.y * (parameters.y - inData.vFirst) < 1.5 || pixelPer.y * (inData.vFirst + inData.vSpan - parameters.y) < 1.5));
            line = line || ((uOptions & (1 << 3)) > 0 && pixelPer.x * (parameters.x - inData.u) < 1.5);
            line = line || ((uOptions & (1 << 3)) > 0 && pixelPer.y * (parameters.y - inData.v) < 1.5);
            color = line ? uLineColor : ((uOptions & (1 << 1)) > 0 ? vec4(splineColor, uFillColor.a) : vec4(0.0, 0.0, 0.0, 0.0));
            color.rgb = (0.3 + 0.5 * abs(dot(normal, uLightDirection)) + 0.2 * specular) * color.rgb;
            if (color.a * texture(uTextureMap, tex).r == 0.0)
                discard;
        }
    """

    def CreateGLResources(self):
        self.frameBuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)

        self.textureBuffer = glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.textureBuffer)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 512, 512, 0, GL_RED, GL_UNSIGNED_BYTE, None)
        glActiveTexture(GL_TEXTURE0)

        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.textureBuffer, 0)
        glDrawBuffers(1, (GL_COLOR_ATTACHMENT0,))

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise ValueError("Framebuffer incomplete")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        SplineOpenGLFrame.CreateGLResources(self)

        glUseProgram(self.surface3Program.surfaceProgram)
        self.uSurfaceTextureMap = glGetUniformLocation(self.surface3Program.surfaceProgram, 'uTextureMap')
        glUniform1i(self.uSurfaceTextureMap, 1) # GL_TEXTURE1 is the texture map
        self.surface3Program.surfaceProgram.check_validate() # Now that textures are assigned, we can validate the program
        glUseProgram(0)

    def __DrawSurface(self, spline, drawCoefficients):
        """Draws a trimmed surface, but because it doesn't do any CPU work between drawing 
        the trim stencil and the surface, it's a little slower that _DrawSpline below."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)
        glDisable(GL_DEPTH_TEST)
        glViewport(0,0,512,512)
        if "trim" in spline.metadata:
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            bounds = spline.domain()
            glOrtho(bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1], -1.0, 1.0)
            glColor3f(0.0, 0.0, 0.0)
            vertices = spline.metadata["trim"]
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_TRIANGLES)
            for vertex in vertices:
                glVertex2fv(vertex)
            glEnd()
        else:
            glClearColor(1.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
        glFlush()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, self.width, self.height)
        glClearColor(self.backgroundColor[0], self.backgroundColor[1], self.backgroundColor[2], self.backgroundColor[3])
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(self.projection)
        glMatrixMode(GL_MODELVIEW)
        SplineOpenGLFrame._DrawSurface(self, spline, drawCoefficients)

class SolidViewer(Viewer):
    def __init__(self, *args, SplineOpenGLFrame=SolidOpenGLFrame, **kw):
        Viewer.__init__(self, *args, SplineOpenGLFrame=SplineOpenGLFrame, **kw)
        self.set_background_color(1.0, 1.0, 1.0)

    def _DrawSplines(self, frame, transform):
        for spline in self.splineDrawList:
            if spline.nInd == 2:
                glBindFramebuffer(GL_FRAMEBUFFER, frame.frameBuffer)
                glDisable(GL_DEPTH_TEST)
                glViewport(0,0,512,512)
                if "trim" in spline.metadata:
                    glClearColor(0.0, 0.0, 0.0, 1.0)
                    glClear(GL_COLOR_BUFFER_BIT)
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    bounds = spline.domain()
                    glOrtho(bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1], -1.0, 1.0)
                    vertices = spline.metadata["trim"]
                    glColor3f(1.0, 0.0, 0.0)
                    glBegin(GL_TRIANGLES)
                    for vertex in vertices:
                        glVertex2fv(vertex)
                    glEnd()
                else:
                    glClearColor(1.0, 0.0, 0.0, 1.0)
                    glClear(GL_COLOR_BUFFER_BIT)
                glFlush()

                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glEnable(GL_DEPTH_TEST)
                glViewport(0, 0, frame.width, frame.height)
                glClearColor(frame.backgroundColor[0], frame.backgroundColor[1], frame.backgroundColor[2], frame.backgroundColor[3])
                glMatrixMode(GL_PROJECTION)
                glLoadMatrixf(frame.projection)
                glMatrixMode(GL_MODELVIEW)

            frame.DrawSpline(spline, transform)

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
        spline.metadata["trim"] = vertices
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