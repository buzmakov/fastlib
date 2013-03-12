import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from fastlib.imageprocessing import opencl

__author__ = 'makov'

import pylab
import numpy
import scipy.ndimage
import fastlib.imageprocessing.reference_implementation as ref
import fastlib.imageprocessing.opencl as opencl
from fastlib.utils.phantom import modified_shepp_logan
from fastlib.utils.mssim import MSSIM


def is_equal(ref_x, y, k=0.01, eps=1.e-6):
    return (numpy.absolute(ref_x - y) <= k * numpy.absolute(ref_x) + eps).all()


def test_project():
    N = 100
    x = modified_shepp_logan((N, N, 3))[:, :, 1]
    x = numpy.array(x)
    rp = ref.project(x)
    ip = opencl.project(x)
    assert is_equal(rp, ip)

if __name__ == "__main__":
    test_project()