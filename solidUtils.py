import numpy as np
from solid import Solid, Boundary
from hyperplane import Hyperplane

def CreateSegmentsFromSolid(solid):
    segments = []
    
    for edge in solid.Edges():
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

def CreateSolidFromPoints(dimension, points, containsInfinity = False):
    # CreateSolidFromPoints only works for dimension 2 so far.
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

def CreateStar(radius, center, angle):
    vertices = []
    points = 5
    for i in range(points):
        vertices.append([radius*np.cos(angle - ((2*i)%points)*6.2832/points) + center[0], radius*np.sin(angle - ((2*i)%points)*6.2832/points) + center[1]])

    nt = (vertices[1][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[1][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0])
    u = ((vertices[3][0]-vertices[0][0])*(vertices[4][1]-vertices[3][1]) + (vertices[3][1]-vertices[0][1])*(vertices[3][0]-vertices[4][0]))/nt

    star = CreateSolidFromPoints(2, vertices)
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