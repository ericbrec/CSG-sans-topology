import numpy as np
import solidUtils as utils
from solid import Solid, Boundary
from spline import Spline
from bspy import Spline as BspySpline
from solidApp import solidApp

if __name__ == "__main__":
    cubeA = utils.create_hypercube([1.5,1.5,1.5], [-1,-1,-1])
    print(cubeA.volume_integral(lambda x: 1.0), 3.0*3.0*3.0)
    print(cubeA.surface_integral(lambda x, n: n), 3.0*3.0*6.0)
    print(cubeA.winding_number([-1,-1,0]))
    print(cubeA.winding_number([4,1,0]))
    cubeB = utils.create_hypercube([1,1,1], [0.5,0.5,0.5])
    print(cubeB.volume_integral(lambda x: 1.0), 2.0*2.0*2.0)
    print(cubeB.surface_integral(lambda x, n: n), 2.0*2.0*6.0)
    
    app = solidApp()

    if True:
        square = utils.create_hypercube([1,1], [0,0])
        star = utils.create_star(2.0, [0.0, 0.0], 90.0*6.28/360.0)
        extrudedSquare = utils.extrude_solid(square,[[-2,2,-4],[2,-2,4]])
        extrudedStar = utils.extrude_solid(star,[[-2,-2,-4],[2,2,4]])
        combined = extrudedStar.union(extrudedSquare)
        app.draw_solid(combined)
    if False:
        app.frame.SetBackgroundColor(1.0, 1.0, 1.0)
        sphere = utils.create_hypercube([1.0,1.0,1.0], [0, 0,0])
        #sphere = Solid(3, False)
        #sphere.boundaries.append(Boundary(Spline(BspySpline.sphere(1.0, 0.001)), utils.create_hypercube([0.5, 0.5], [0.5, 0.5])))
        #sphere.boundaries.append(Boundary(Spline(BspySpline.cone(2.0, 0.01, 3.0, 0.001) + (0.0, 0.0, -1.5)), utils.create_hypercube([0.5, 0.5], [0.5, 0.5])))
        app.draw_solid(sphere, "sphere", np.array((.4, .6, 1, 1),np.float32))
        endCurve = [[1, 0], [0, 0], [0, 1]] @ BspySpline(1, 1, (3,), (5,), (np.array((-3.0, -3.0, -3.0, -0.6, 0.6, 3.0, 3.0, 3.0)),), np.array((0, 3.0/8.0, 0, -4.0/8.0, 0))).graph()
        spline = Spline(BspySpline.ruled_surface(endCurve + (0.0, -2.0, 0.0), endCurve + (0.0, 2.0, 0.0)))
        halfSpace = Solid(3, False)
        halfSpace.boundaries.append(Boundary(spline, utils.create_hypercube([3.0, 0.5], [0.0, 0.5])))
        app.draw_solid(halfSpace, "halfSpace", np.array((0, 1, 0, 1),np.float32))
        difference = sphere - halfSpace
        app.draw_solid(difference, "difference")
        app.mainloop()
    if False:
        order = 3
        knots = [0.0] * order + [1.0] * order
        nCoef = len(knots) - order
        spline = Spline(BspySpline(2, 3, (order, order), (nCoef, nCoef), (knots, knots), \
            (((-1.0, -1.0, -1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), \
            ((1.0, 0.0, 1.0), (0.0, -5.0, 0.0), (1.0, 0.0, 1.0)), \
            ((-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0)))))
        cap = utils.hyperplane_axis_aligned(3, 1, 0.7, False)
        paraboloid = Solid(3, False)
        paraboloid.boundaries.append(Boundary(spline, utils.create_hypercube([0.5, 0.5], [0.5, 0.5])))
        paraboloid.boundaries.append(Boundary(cap, utils.create_hypercube([1.0, 1.0], [0.0, 0.0])))
        app.list_solid(paraboloid, "paraboloid", np.array((.4, .6, 1, 1),np.float32))

        spline = Spline(spline.spline.copy())
        cap = utils.hyperplane_axis_aligned(3, 1, 0.7, False)
        paraboloid2 = Solid(3,False)
        paraboloid2.boundaries.append(Boundary(spline, utils.create_hypercube([0.5, 0.5], [0.5, 0.5])))
        paraboloid2.boundaries.append(Boundary(cap, utils.create_hypercube([1.0, 1.0], [0.0, 0.0])))
        paraboloid2.translate(np.array((0.0, 0.5, 0.55)))
        app.list_solid(paraboloid2, "paraboloid2", np.array((0, 1, 0, 1),np.float32))

        paraboloid3 = paraboloid + paraboloid2
        app.draw_solid(paraboloid3, "p + p2")
    
    app.mainloop()