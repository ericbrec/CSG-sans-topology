# CSG sans topology
This example code implements our new algorithm for constructive solid geometry in any dimension is written in Python and designed for clarity, not performance. Nonetheless, the code performs interactively on an Inteli5 2.40GHz processor for 2D, 3D, and 4D polygonal solids, with short delays for 2D spline solids, and with minute-long delays for 3D spline solids (4D splines are not supported, and we haven’t figured out a nice way to display 5D+ polygonal solids).

The example code also deliberately uses rough tolerances (surfaces separated by less than 0.01 are considered touching). We wanted the code to demonstrate the algorithm’s resilience to local errors and ambiguous topologies that challenge prior algorithms.

The code has three base classes: Solid and Boundary found in solid.py, and Manifold found in manifold.py. Manifold is an abstract class that defines basic methods for manifolds. Boundary is a simple container for a Manifold and a domain Solid. Solid is the class that holds the substance of the algorithm.

In addition, there are two subclasses of Manifold: Hyperplane found in hyperplane.py and BSpline found in bSpline.py. Hyperplane supports hyperplanes of any dimension, including their intersection with other hyperplanes. BSpline supports non-uniform B-Spline curves and surfaces, using the BSpy library for spline evaluation and intersection with other splines and with hyperplanes.

Finally, there are a variety of example programs that perform operations on solids: solidUtils.py has some reused utility functions, solidViewer.py defines a BSpy Viewer for solids, main2D.py performs 2D solid operations, main2DSlice.py animates a 2D slice through a 3D solid, main3D.py performs 3D solid operations, main3DSlice.py animates a 3D slice through a 4D solid, and teapot.py performs solid operations on two Utah teapots (original patches).

All the code is available on GitHub under the MIT license using Python 3.x.