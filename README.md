# CSG sans topology
This example code implements our new algorithm for constructive solid geometry in any dimension is written in Python and designed for clarity, not performance. Nonetheless, the code performs interactively on an Inteli5 2.40GHz processor for 2D, 3D, and 4D polygonal solids, with short delays for 2D spline solids, and with minute-long delays for 3D spline solids (4D splines are not supported, and we haven’t figured out a nice way to display 5D+ polygonal solids).

The example code also deliberately uses rough tolerances (surfaces separated by less than 0.01 are considered touching). We wanted the code to demonstrate the algorithm’s resilience to local errors and ambiguous topologies that challenge prior algorithms.

The code has three base classes: Solid and Boundary found in solid.py, and Manifold found in manifold.py. Manifold is an abstract class that defines basic methods for manifolds. Boundary is a simple container for a Manifold and a domain Solid. Solid is the class that holds the substance of the algorithm.

In addition, there are two subclasses of Manifold: Hyperplane found in hyperplane.py and Spline found in spline.py. Hyperplane supports hyperplanes of any dimension, including their intersection with other hyperplanes. Spline supports non-uniform B-Spline curves and surfaces, using the bspy library for spline evaluation and intersection with other splines and with hyperplanes.

Finally, there are a variety of example programs that perform operations on solids, using the matplotlib to render the results: solidUtils.py that has some reused utility functions, main2D.py for operations on 2D solids, main2DSlice.py for an animated 2D slice through a 3D constructed solid, main3D.py for operations on 3D solids, and main3DSlice for an animated 3D slice through a 4D constructed solid.

All the code is available on GitHub under the MIT license using Python 3.x.