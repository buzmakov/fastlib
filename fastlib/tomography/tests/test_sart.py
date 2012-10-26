__author__ = 'makov'

import os
import sys
import time

import numpy

root_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, root_dir)

from fastlib.imageprocessing.ispmd import rotate_square, project
from fastlib.tomography.sart import sart
from fastlib.utils.phantom import modified_shepp_logan



def genrate_sinogam_ref(image, angles):
    size = image.shape[0]
    sinogram = numpy.zeros((size, len(angles)), dtype='float32')
    for ia, angle in enumerate(angles):
        tmp_proj = project(rotate_square(image, angle))
        sinogram[:, ia] = tmp_proj
    return sinogram.astype('float32'), angles.astype('float32')


def test_sart():
    size = 1024
    angles = numpy.arange(0, 180, 1.0, dtype='float32')
    x = modified_shepp_logan((size, size, 3))[:, :, 1]
    x = numpy.array(x)
    sinogram, angles = genrate_sinogam_ref(x, angles)
    t = time.time()
    res=None
    for i in range(32):
        res = sart(sinogram, angles)
    print 'Tomographic reconstruction: ' + str(time.time() - t)

    import pylab
    pylab.subplot(121)
    pylab.imshow(x)
    pylab.colorbar()
    pylab.subplot(122)
    pylab.imshow(res, vmin=0, vmax=1)
    pylab.colorbar()
    pylab.show()

if __name__ == "__main__":
    test_sart()