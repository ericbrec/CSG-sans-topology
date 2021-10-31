import numpy as np
import manifold as mf
import solid as sld

def CreateSegmentsFromSolid(solid):
    segments = []
    
    for edge in solid.Edges():
        middle = 0.5 * (edge[0] + edge[1])
        normal = middle + 0.1 * edge[2]
        segments.append((edge[0], edge[1]))
        segments.append((middle, normal))
    
    return segments

def HyperplaneAxisAligned(dimension, axis, offset):
    hyperplane = mf.Hyperplane()
    diagonal = np.identity(abs(dimension))
    sign = np.sign(dimension)
    hyperplane.normal = sign * diagonal[:,axis]
    hyperplane.point = offset * hyperplane.normal
    if abs(dimension) > 1:
        hyperplane.tangentSpace = np.delete(diagonal, axis, axis=1)
    else:
        hyperplane.tangentSpace = np.array([0.0])
    
    return hyperplane

def CreateHypercube(size, position = None):
    dimension = len(size)
    solid = sld.Solid(dimension)
    if position is None:
        position = [0.0]*dimension
    else:
        assert len(position) == dimension

    for i in range(dimension):
        domain = None
        if dimension > 1:
            domainSize = size.copy()
            del domainSize[i]
            domainPosition = position.copy()
            del domainPosition[i]
            domain = CreateHypercube(domainSize, domainPosition)
        hyperplane = HyperplaneAxisAligned(dimension, i, size[i] + position[i])
        solid.boundaries.append(sld.Boundary(hyperplane,domain))
        hyperplane = HyperplaneAxisAligned(-dimension, i, size[i] - position[i])
        solid.boundaries.append(sld.Boundary(hyperplane,domain))

    return solid

def Hyperplane1D(normal, offset):
    assert np.isscalar(normal) or len(normal) == 1
    hyperplane = mf.Hyperplane()
    hyperplane.normal = np.atleast_1d(normal)
    hyperplane.point = offset * hyperplane.normal
    hyperplane.tangentSpace = np.atleast_1d(0.0)
    return hyperplane

def Hyperplane2D(normal, offset):
    assert len(normal) == 2
    hyperplane = mf.Hyperplane()
    hyperplane.normal = np.atleast_1d(normal)
    hyperplane.point = offset * hyperplane.normal
    hyperplane.tangentSpace = np.transpose(np.array([[normal[1], -normal[0]]]))
    return hyperplane

def CreateSolidFromPoints(dimension, points, isVoid = False):
    # CreateSolidFromPoints only works for dimension 2 so far.
    assert dimension == 2
    assert len(points) > 2
    assert len(points[0]) == dimension

    solid = sld.Solid(dimension, isVoid)

    previousPoint = np.array(points[len(points)-1])
    for point in points:
        point = np.array(point)
        vector = point - previousPoint
        normal = np.array([-vector[1], vector[0]])
        normal = normal / np.linalg.norm(normal)
        hyperplane = Hyperplane2D(normal,np.dot(normal,point))
        domain = sld.Solid(dimension-1)
        previousPointDomain = hyperplane.DomainFromPoint(previousPoint)
        pointDomain = hyperplane.DomainFromPoint(point)
        if previousPointDomain < pointDomain:
            domain.boundaries.append(sld.Boundary(Hyperplane1D(-1.0, -previousPointDomain)))
            domain.boundaries.append(sld.Boundary(Hyperplane1D(1.0, pointDomain)))
        else:
            domain.boundaries.append(sld.Boundary(Hyperplane1D(-1.0, -pointDomain)))
            domain.boundaries.append(sld.Boundary(Hyperplane1D(1.0, previousPointDomain)))
        solid.boundaries.append(sld.Boundary(hyperplane, domain))
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
        boundary.domain.boundaries.append(sld.Boundary(Hyperplane1D(1.0, u0 + (1.0 - u)*(u1 - u0))))
        boundary.domain.boundaries.append(sld.Boundary(Hyperplane1D(-1.0, -(u0 + u*(u1 - u0)))))

    return star