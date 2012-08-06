"""
Testing phantom generation module.
"""

import nose
import numpy.testing
import numpy as np

from fastlib.utils.phantom import *

# tests cases
phantoms = [yu_ye_wang, shepp_logan, modified_shepp_logan]
# not everything is working if some dimensions are 1 or 2 now:
#shapes = [(1, 1, 1), (16, 16, 16), (16, 16, 1), (16, 1, 1)]
shapes = [(16, 16, 16), (16, 16, 3), (16, 3, 3), (3, 16, 3), (3, 3, 16)]
#shape16 = shapes[0]
dtypes = [np.float32, np.float64, np.int32, np.int64]

spheres = [
    {'A':1, 'a':1., 'b':1., 'c':1., 'x0':0., 'y0':0., 'z0':0., 'phi':0., 'theta':0., 'psi':0.},
    {'A':.5, 'a':1., 'b':1., 'c':1., 'x0':0., 'y0':0., 'z0':0., 'phi':0., 'theta':0., 'psi':0.}
    ]

spheres_arrays = [
    [[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]],
    [[.5, 1., 1., 1., 0., 0., 0., 0., 0., 0.]],
    ]

# tests for all predefined phantoms
for p in phantoms:
    def test_shape():
        for shape in shapes:
            yield numpy.testing.assert_equal, p(shape).shape, shape

    def test_dtype():
        for dtype in dtypes:
            for shape in shapes:
                yield numpy.testing.assert_equal, p(shape, dtype=dtype).dtype, dtype

# tests on the phantom function
def test_central_value():
    for shape in shapes:
        i, j, k = np.asarray(shape) / 2.
        for p in spheres:
            yield numpy.testing.assert_equal, phantom(shape, [p,])[i, j, k], p['A']

# tests conversion from array to dict
def test_array_to_parameters():
    from fastlib.utils.phantom import _array_to_parameters
    for a, p in zip(spheres_arrays, spheres):
        yield numpy.testing.assert_array_equal, _array_to_parameters(a), p

if __name__ == "__main__":
    nose.run(argv=['', __file__])
