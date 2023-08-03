import numpy as np
from solid import Solid, Boundary
from manifold import Manifold
from hyperplane import Hyperplane
from spline import Spline
from bspy import Spline as BspySpline

def SolidEdges(solid, subdivide = False):
    """
    A generator for edges of the solid.

    Yields
    -------
    (point1, point2, normal) : `tuple(numpy.array, numpy.array, numpy.array)`
        Starting point, ending point, and normal for an edge of the solid.


    Notes
    -----
    The edges are not guaranteed to be connected or in any particular order, and typically aren't.

    If the solid is a number line (dimension 1), the generator yields a tuple with two scalar values (start, end).
    """
    if solid.dimension > 1:
        for boundary in solid.boundaries:
            for domainEdge in SolidEdges(boundary.domain, not isinstance(boundary.manifold, Hyperplane)):
                yield (boundary.manifold.Point(domainEdge[0]), boundary.manifold.Point(domainEdge[1]), boundary.manifold.Normal(domainEdge[0]))
    else:
        solid.boundaries.sort(key=Boundary.SortKey)
        leftB = 0
        rightB = 0
        while leftB < len(solid.boundaries):
            if solid.boundaries[leftB].manifold.Normal(0.0) < 0.0:
                leftPoint = solid.boundaries[leftB].manifold.Point(0.0)
                while rightB < len(solid.boundaries):
                    rightPoint = solid.boundaries[rightB].manifold.Point(0.0)
                    if leftPoint - Manifold.minSeparation < rightPoint and solid.boundaries[rightB].manifold.Normal(0.0) > 0.0:
                        if subdivide:
                            dt = 0.1
                            t = leftPoint.copy()
                            while t + dt < rightPoint:
                                yield (t, t + dt)
                                t += dt
                            yield (t, rightPoint)
                        else:
                            yield (leftPoint, rightPoint)
                        leftB = rightB
                        rightB += 1
                        break
                    rightB += 1
            leftB += 1


def CreateSegmentsFromSolid(solid):
    segments = []
    
    for edge in SolidEdges(solid):
        middle = 0.5 * (edge[0] + edge[1])
        normal = middle + 0.1 * edge[2]
        segments.append((edge[0], edge[1]))
        segments.append((middle, normal))
    
    return segments

def HyperplaneAxisAligned(dimension, axis, offset, flipNormal=False):
    assert dimension > 0
    diagonal = np.identity(dimension)
    sign = -1.0 if flipNormal else 1.0
    normal = sign * diagonal[:,axis]
    point = offset * normal
    if dimension > 1:
        tangentSpace = np.delete(diagonal, axis, axis=1)
    else:
        tangentSpace = np.array([0.0])
    
    return Hyperplane(normal, point, tangentSpace)

def CreateHypercube(size, position = None):
    dimension = len(size)
    solid = Solid(dimension, False)
    if position is None:
        position = [0.0]*dimension
    else:
        assert len(position) == dimension

    for i in range(dimension):
        if dimension > 1:
            domainSize = size.copy()
            del domainSize[i]
            domainPosition = position.copy()
            del domainPosition[i]
            domain = CreateHypercube(domainSize, domainPosition)
        else:
            domain = Solid(0, True)
        hyperplane = HyperplaneAxisAligned(dimension, i, size[i] + position[i], False)
        solid.boundaries.append(Boundary(hyperplane,domain))
        hyperplane = HyperplaneAxisAligned(dimension, i, size[i] - position[i], True)
        solid.boundaries.append(Boundary(hyperplane,domain))

    return solid

def Hyperplane1D(normal, offset):
    assert np.isscalar(normal) or len(normal) == 1
    normalizedNormal = np.atleast_1d(normal)
    normalizedNormal = normalizedNormal / np.linalg.norm(normalizedNormal)
    return Hyperplane(normalizedNormal, offset * normalizedNormal, 0.0)

def Hyperplane2D(normal, offset):
    assert len(normal) == 2
    normalizedNormal = np.atleast_1d(normal)
    normalizedNormal = normalizedNormal / np.linalg.norm(normalizedNormal)
    return Hyperplane(normalizedNormal, offset * normalizedNormal, np.transpose(np.array([[normal[1], -normal[0]]])))

def HyperplaneDomainFromPoint(hyperplane, point):
    tangentSpaceTranspose = np.transpose(hyperplane.tangentSpace)
    return np.linalg.inv(tangentSpaceTranspose @ hyperplane.tangentSpace) @ tangentSpaceTranspose @ (point - hyperplane.point)

def CreateFacetedSolidFromPoints(dimension, points, containsInfinity = False):
    # CreateFacetedSolidFromPoints only works for dimension 2 so far.
    assert dimension == 2
    assert len(points) > 2
    assert len(points[0]) == dimension

    solid = Solid(dimension, containsInfinity)

    previousPoint = np.array(points[len(points)-1])
    for point in points:
        point = np.array(point)
        vector = point - previousPoint
        normal = np.array([-vector[1], vector[0]])
        normal = normal / np.linalg.norm(normal)
        hyperplane = Hyperplane2D(normal,np.dot(normal,point))
        domain = Solid(dimension-1, False)
        domainDomain = Solid(dimension-2, True) # Domain for 1D points.
        previousPointDomain = HyperplaneDomainFromPoint(hyperplane, previousPoint)
        pointDomain = HyperplaneDomainFromPoint(hyperplane, point)
        if previousPointDomain < pointDomain:
            domain.boundaries.append(Boundary(Hyperplane1D(-1.0, -previousPointDomain), domainDomain))
            domain.boundaries.append(Boundary(Hyperplane1D(1.0, pointDomain), domainDomain))
        else:
            domain.boundaries.append(Boundary(Hyperplane1D(-1.0, -pointDomain), domainDomain))
            domain.boundaries.append(Boundary(Hyperplane1D(1.0, previousPointDomain), domainDomain))
        solid.boundaries.append(Boundary(hyperplane, domain))
        previousPoint = point

    return solid

def CreateSmoothSolidFromPoints(dimension, points, containsInfinity = False):
    # CreateSmoothSolidFromPoints only works for dimension 2 so far.
    assert dimension == 2
    assert len(points) > 2
    assert len(points[0]) == dimension

    solid = Solid(dimension, containsInfinity)

    t = 0.0
    previousPoint = np.array(points[0])
    dataPoints = [(t, *previousPoint)]
    for point in points[1:]:
        point = np.array(point)
        t += np.linalg.norm(point - previousPoint)
        dataPoints.append((t, *point))
        previousPoint = point
    point = np.array(points[0])
    t += np.linalg.norm(point - previousPoint)
    dataPoints.append((t, *point))

    spline = Spline(BspySpline.least_squares(dimension - 1, dimension, (4,) * (dimension - 1), dataPoints))
    domain = Solid(dimension-1, False)
    domainDomain = Solid(dimension-2, True) # Domain for 1D points.
    domain.boundaries.append(Boundary(Hyperplane1D(-1.0, 0.0), domainDomain))
    domain.boundaries.append(Boundary(Hyperplane1D(1.0, t), domainDomain))
    solid.boundaries.append(Boundary(spline, domain))

    return solid

def CreateStar(radius, center, angle, smooth = False):
    points = 5
    vertices = []

    if smooth:
        dAngle = 6.2832 / points
        for i in range(points):
            vertices.append([radius*np.cos(angle + i*dAngle) + center[0], radius*np.sin(angle + i*dAngle) + center[1]])
            vertices.append([0.5*radius*np.cos(angle + (i + 0.5)*dAngle) + center[0], 0.5*radius*np.sin(angle + (i + 0.5)*dAngle) + center[1]])

        star = CreateSmoothSolidFromPoints(2, vertices)
    else:
        for i in range(points):
            vertices.append([radius*np.cos(angle - ((2*i)%points)*6.2832/points) + center[0], radius*np.sin(angle - ((2*i)%points)*6.2832/points) + center[1]])

        star = CreateFacetedSolidFromPoints(2, vertices)

        nt = (vertices[1][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[1][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0])
        u = ((vertices[3][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[3][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0]))/nt
        for boundary in star.boundaries:
            u0 = boundary.domain.boundaries[0].manifold.point[0]
            u1 = boundary.domain.boundaries[1].manifold.point[0]
            boundary.domain.boundaries.append(Boundary(Hyperplane1D(1.0, u0 + (1.0 - u)*(u1 - u0)), Solid(0, True)))
            boundary.domain.boundaries.append(Boundary(Hyperplane1D(-1.0, -(u0 + u*(u1 - u0))), Solid(0, True)))

    return star

def ExtrudeSolid(solid, path):
    assert len(path) > 1
    assert solid.dimension+1 == len(path[0])
    
    extrusion = Solid(solid.dimension+1, False)

    # Extrude boundaries along the path
    point = None
    for nextPoint in path:
        nextPoint = np.atleast_1d(nextPoint)
        if point is None:
            point = nextPoint
            continue
        tangent = nextPoint - point
        extent = tangent[solid.dimension]
        tangent = tangent / extent
        # Extrude each boundary
        for boundary in solid.boundaries:
            # Construct a normal orthogonal to both the boundary tangent space and the path tangent
            extruded_normal = np.full((extrusion.dimension), 0.0)
            extruded_normal[0:solid.dimension] = boundary.manifold.normal[:]
            extruded_normal[solid.dimension] = -np.dot(boundary.manifold.normal, tangent[0:solid.dimension])
            extruded_normal = extruded_normal / np.linalg.norm(extruded_normal)
            # Construct a point that adds the boundary point to the path point
            extruded_point = np.full((extrusion.dimension), 0.0)
            extruded_point[0:solid.dimension] = boundary.manifold.point[:]
            extruded_point += point
            # Combine the boundary tangent space and the path tangent
            extruded_tangentSpace = np.full((extrusion.dimension, solid.dimension), 0.0)
            if solid.dimension > 1:
                extruded_tangentSpace[0:solid.dimension, 0:solid.dimension-1] = boundary.manifold.tangentSpace[:,:]
            extruded_tangentSpace[:, solid.dimension-1] = tangent[:]
            extrudedHyperplane = Hyperplane(extruded_normal, extruded_point, extruded_tangentSpace)
            # Construct a domain for the extruded boundary
            if boundary.domain.dimension > 0:
                # Extrude the boundary's domain to include path domain
                domainPath = []
                domainPoint = np.full((solid.dimension), 0.0)
                domainPath.append(domainPoint)
                domainPoint = np.full((solid.dimension), 0.0)
                domainPoint[solid.dimension-1] = extent
                domainPath.append(domainPoint)
                extrudedDomain = ExtrudeSolid(boundary.domain, domainPath)
            else:
                extrudedDomain = Solid(solid.dimension, False)
                extrudedDomain.boundaries.append(Boundary(Hyperplane1D(-1.0, 0.0), Solid(0, True)))
                extrudedDomain.boundaries.append(Boundary(Hyperplane1D(1.0, extent), Solid(0, True)))
            # Add extruded boundary
            extrusion.boundaries.append(Boundary(extrudedHyperplane, extrudedDomain))
        
        # Move onto the next point
        point = nextPoint

    # Add end cap boundaries
    extrudedHyperplane = HyperplaneAxisAligned(extrusion.dimension, solid.dimension, 0.0, True)
    extrudedHyperplane.Translate(path[0])
    extrusion.boundaries.append(Boundary(extrudedHyperplane, solid))
    extrudedHyperplane = HyperplaneAxisAligned(extrusion.dimension, solid.dimension, 0.0, False)
    extrudedHyperplane.Translate(path[-1])
    extrusion.boundaries.append(Boundary(extrudedHyperplane, solid))

    return extrusion