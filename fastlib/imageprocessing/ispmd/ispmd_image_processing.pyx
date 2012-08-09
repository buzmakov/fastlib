# cython: profile=False
cimport numpy
import numpy
import cython
cimport cython

#cdef extern from "C/c_image_processing.h":
#    void project(float* vin, float* vout, int count)

cdef extern from "C/objs/image_processing_ispc.h":
    void rotate_mt(float* data, float* res, float angle, int im_size)
    void rotate_volume(float* data, float* res, float angle, int im_size, int slices_count)
    void backproject(float* vin, float* vout, int im_size)
    void project(float* vin, float* vout, int im_size)
    void add_mt(float* v1, float* v2, int im_size)

@cython.boundscheck(False)
@cython.wraparound(False)
def project_fast(numpy.ndarray[numpy.float32_t, ndim=2] vin):
    cdef numpy.int32_t count=vin.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=1] vout=numpy.zeros(count,dtype='float32')
    project(<float *>vin.data,<float *> vout.data,count)
    return vout

@cython.boundscheck(False)
@cython.wraparound(False)
def project_fast_ref(numpy.ndarray[numpy.float32_t, ndim=2] vin, numpy.ndarray[numpy.float32_t, ndim=1] vout):
    cdef numpy.int32_t count=vin.shape[0]
    project(<float *>vin.data,<float *> vout.data,count)

@cython.boundscheck(False)
@cython.wraparound(False)
def backproject_fast(numpy.ndarray[numpy.float32_t, ndim=1] vin):
    cdef numpy.int32_t count=vin.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=1] tmpin=vin/count
    cdef numpy.ndarray[numpy.float32_t, ndim=2] vout=numpy.zeros((count,count),dtype='float32')
    backproject(<float *>tmpin.data,<float *> vout.data,count)
    return vout

@cython.boundscheck(False)
@cython.wraparound(False)
def backproject_fast_ref(numpy.ndarray[numpy.float32_t, ndim=1] vin,numpy.ndarray[numpy.float32_t, ndim=2] vout):
    cdef numpy.int32_t count=vin.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=1] tmpin=vin/count
    backproject(<float *>tmpin.data,<float *> vout.data,count)

@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_sq_fast(numpy.ndarray[numpy.float32_t, ndim=2] data, numpy.float32_t angle):
    cdef numpy.int32_t im_size=data.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=2] res=numpy.zeros((im_size,im_size),dtype='float32')
    rotate_mt(<float *> data.data,<float *> res.data, angle, im_size)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_sq_fast_ref(numpy.ndarray[numpy.float32_t, ndim=2] data, numpy.float32_t angle,
    numpy.ndarray[numpy.float32_t, ndim=2] res):
    cdef numpy.int32_t im_size=data.shape[0]
    rotate_mt(<float *> data.data,<float *> res.data, angle, im_size)

@cython.boundscheck(False)
@cython.wraparound(False)
def add_fast_ref(numpy.ndarray[numpy.float32_t, ndim=2] v1, numpy.ndarray[numpy.float32_t, ndim=2] v2):
#    TODO: chek shape(v1)=shape(v2)
    cdef numpy.int32_t count=v1.shape[0]*v1.shape[1]
    add_mt(<float *>v1.data,<float *> v2.data,count)

@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_volume_fast(numpy.ndarray[numpy.float32_t, ndim=3] data, numpy.float32_t angle):
    cdef numpy.int32_t im_size=data.shape[1]
    cdef numpy.int32_t slice_count=data.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=3] res=numpy.zeros_like(data)
    rotate_volume(<float *> data.data, <float *> res.data, angle, im_size,slice_count)
    return res