# coding=utf-8
import numpy
import cython
cimport numpy
#cimport cython
from fastlib.imageprocessing.ispmd import project,rotate_square,back_project

__author__ = 'makov'

@cython.boundscheck(False)
@cython.wraparound(False)
def sart(numpy.ndarray[numpy.float32_t, ndim=2] sinogram,
        numpy.ndarray[numpy.float32_t, ndim=1] angles):

    #    #normalize sinograms
    #    sinogram *= numpy.mean(sinogram.sum(axis=0)) / sinogram.sum(axis=0)
    cdef numpy.ndarray[numpy.float32_t, ndim=2] tomo_rec, tmp_backproj_rot, tmp_backproj
    cdef numpy.ndarray[numpy.float32_t, ndim=1] shuffle_iang, tmp_proj
    cdef numpy.float32_t coeff
    cdef numpy.int32_t reconst_shape, ang_count, iang

    reconst_shape = numpy.array(sinogram.shape[0],dtype='int32')
    tomo_rec = numpy.zeros((reconst_shape, reconst_shape), dtype='float32')
    ang_count=len(angles)
    for coeff in numpy.array([0.8,0.3,],dtype='float32'):
        shuffle_iang = numpy.arange(ang_count, dtype='float32')
        numpy.random.shuffle(shuffle_iang)
        for iang in shuffle_iang:
            tmp_proj = sinogram[:, iang] - project(rotate_square(tomo_rec, angles[iang]))
            tmp_backproj = back_project(tmp_proj * coeff)
            tmp_backproj_rot = rotate_square(tmp_backproj, -angles[iang])
            tomo_rec += tmp_backproj_rot
#        tomo_rec -= 0.5*tomo_rec*(tomo_rec < 0)
    return tomo_rec