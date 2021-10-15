import numpy as np
import manifold as mf

# TODO: Update ContainsPoint to use integral instead of ray cast to compute winding number.

class Boundary:

    def __init__(self, manifold, domain = None):
        if domain:
            assert manifold.GetDimension() - 1 == domain.dimension
        else:
            assert manifold.GetDimension() == 1

        self.manifold = manifold
        self.domain = domain
    
    @staticmethod
    def SortKey(boundary):
        if boundary.domain:
            return 0.0
        else:
            return boundary.manifold.PointFromDomain(0.0)

class Solid:

    @staticmethod
    def CreateSolidFromPoints(dimension, points, isVoid = False):
        # CreateSolidFromPoints only works for dimension 2 so far.
        assert dimension == 2
        assert len(points) > 2
        assert len(points[0]) == dimension

        solid = Solid(dimension, isVoid)

        previousPoint = np.array(points[len(points)-1])
        for point in points:
            point = np.array(point)
            vector = point - previousPoint
            normal = np.array([-vector[1], vector[0]])
            normal = normal / np.linalg.norm(normal)
            manifold = mf.Hyperplane.CreateFromNormal(normal,np.dot(normal,point))
            domain = Solid(dimension-1)
            domain.boundaries.append(Boundary(mf.Hyperplane.CreateFromNormal(-1.0, -manifold.DomainFromPoint(previousPoint))))
            domain.boundaries.append(Boundary(mf.Hyperplane.CreateFromNormal(1.0, manifold.DomainFromPoint(point))))
            solid.boundaries.append(Boundary(manifold, domain))
            previousPoint = point

        return solid

    def __init__(self, dimension, isVoid = False):
        assert dimension > 0
        self.dimension = dimension
        self.isVoid = isVoid
        self.boundaries = []

    def Not(self):
        solid = Solid(self.dimension, not self.isVoid)
        for boundary in self.boundaries:
            solid.boundaries.append(Boundary(boundary.manifold.Flip(),boundary.domain))
        return solid
    
    def Translate(self, delta):
        assert len(delta) == self.dimension

        for boundary in self.boundaries:
            boundary.manifold.Translate(delta)
    
    def Points(self):
        for boundary in self.boundaries:
            if boundary.domain:
                for domainPoint in boundary.domain.Points():
                    point = boundary.manifold.PointFromDomain(domainPoint)
                    yield point
            else:
                yield boundary.manifold.PointFromDomain(0.0)
    
    def Edges(self):
        if self.dimension > 1:
            for boundary in self.boundaries:
                for domainEdge in boundary.domain.Edges():
                    yield [boundary.manifold.PointFromDomain(domainEdge[0]), boundary.manifold.PointFromDomain(domainEdge[1])]
        else:
            self.boundaries.sort(key=Boundary.SortKey)
            b = 1
            while b < len(self.boundaries):
                yield [self.boundaries[b-1].manifold.PointFromDomain(0.0), self.boundaries[b].manifold.PointFromDomain(0.0)]
                b += 2
 
    def ContainsPoint(self, point):
        # Return value is -1 for interior, 0 for on boundary, and 1 for exterior
        containment = -1
        onBoundary = False

        # The winding number is 0 if the point is outside the solid, 1 if it's inside.
        # Other values indicate issues: 
        #  * Incomplete boundaries lead to fractional values;
        #  * Interior-pointing normals lead to negative values;
        #  * Nested shells lead to absolute values of 2 or greater.
        windingNumber = 0
        if self.isVoid:
            # If the solid is a void, then the winding number starts as 1 to account for the boundary at infinity.
            windingNumber = 1

        # Intersect a ray from point along the x-axis through self's boundaries.
        # All dot products with the ray are just the first component, since the ray is along the x-axis.
        for boundary in self.boundaries:
            intersections = boundary.manifold.IntersectXRay(point)
            for intersection in intersections:
                # Each intersection is of the form [distance to intersection, domain point of intersection].
                # First, check the distance is positive.
                if intersection[0] > -mf.Manifold.minSeparation:
                    considerBoundary = True
                    if boundary.domain:
                        # Only include the boundary if the ray intersection is inside its domain.
                        considerBoundary = boundary.domain.ContainsPoint(intersection[1]) < 1
                    # If we've got a valid boundary intersection, accumulate winding number based on sign of dot(ray,normal) == normal[0].
                    if considerBoundary:
                        if intersection[0] < mf.Manifold.minSeparation:
                            onBoundary = True
                        windingNumber += np.sign(boundary.manifold.NormalFromDomain(intersection[1])[0])
        
        if onBoundary:
            containment = 0
        elif windingNumber == 0:
            containment = 1
        return containment 

    def ContainsBoundary(self, boundary):
        containment = False
        if boundary.domain:
            # If boundary has a domain, loop through the boundary points of its domain until one is clearly inside or outside.
            for domainPoint in boundary.domain.Points():
                constainsPoint = self.ContainsPoint(boundary.manifold.PointFromDomain(domainPoint))
                if constainsPoint != 0:
                    containment = constainsPoint < 1
                    break
        else:
            # Otherwise, the boundary is a single point, so just determine if it's inside or outside.
            containment = self.ContainsPoint(boundary.manifold.PointFromDomain(0.0)) < 1

        return containment

    def Slice(self, manifold, cache = None):
        assert manifold.GetDimension() == self.dimension

        manifoldDomain = None

        # Only manifolds of dimension > 1 have a domain.
        if self.dimension > 1:
            manifoldDomain = Solid(self.dimension-1, self.isVoid)

            # Intersect each of this solid's boundaries with the manifold.
            for boundary in self.boundaries:
                # Start by intersecting the boundary's manifold with the given manifold.
                # IntersectManifold returns a collection of manifold pairs:
                #   * intersection[0] is in the boundary's domain;
                #   * intersection[1] is in the given manifold's domain.
                # Both intersections correspond to the same range (the intersection between the manifolds).
                intersections = boundary.manifold.IntersectManifold(manifold, cache)

                # For each intersection, slice the boundary domain with the intersection manifold.
                # We slice the boundary domain using intersection[0], but we add intersection[1] to the manifold domain. 
                for intersection in intersections:
                    if boundary.domain.dimension > 1:
                        intersectionSlice = boundary.domain.Slice(intersection[0], cache)
                        if intersectionSlice:
                            manifoldDomain.boundaries.append(Boundary(intersection[1],intersectionSlice))
                    else:
                        # This domain is dimension 1, a real number line, and the intersection is a point on that line.
                        # Determine if the intersection point is within domain's interior.
                        if boundary.domain.ContainsPoint(intersection[0].PointFromDomain(0.0)) < 1:
                            manifoldDomain.boundaries.append(Boundary(intersection[1]))
            
            # Don't return a manifold domain if it's empty
            if len(manifoldDomain.boundaries) == 0:
                manifoldDomain = None
        
        return manifoldDomain

    def Intersection(self, solid, cache = None):
        assert self.dimension == solid.dimension

        # Manifold intersections are expensive and come in symmetric pairs (m1 intersect m2, m2 intersect m1).
        # So, we create a manifold intersections cache (dictionary) to store and reuse intersection pairs.
        if cache == None:
            cache = {}

        combinedSolid = Solid(self.dimension, self.isVoid and solid.isVoid)

        for boundary in self.boundaries:
            # Slice self boundary manifold by solid. If it intersects, intersect the domains.
            slice = solid.Slice(boundary.manifold, cache)
            if slice:
                newDomain = boundary.domain.Intersection(slice, cache)
                if len(newDomain.boundaries) > 0:
                    combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif solid.ContainsBoundary(boundary):
                combinedSolid.boundaries.append(boundary)

        for boundary in solid.boundaries:
            # Slice solid boundary manifold by self. If it intersects, intersect the domains.
            slice = self.Slice(boundary.manifold, cache)
            if slice:
                newDomain = boundary.domain.Intersection(slice, cache)
                if len(newDomain.boundaries) > 0:
                    combinedSolid.boundaries.append(Boundary(boundary.manifold, newDomain))
            elif self.ContainsBoundary(boundary):
                combinedSolid.boundaries.append(boundary)

        return combinedSolid

    def Union(self, solid):
        return self.Not().Intersection(solid.Not()).Not()

    def Difference(self, solid):
        return self.Intersection(solid.Not())