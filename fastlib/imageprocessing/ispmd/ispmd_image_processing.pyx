# coding=utf-8
#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False

import numpy
cimport numpy
import cython


cdef extern from "C/objs/image_processing_ispc.h":
    void rotate_mt(float* data, float* res, float angle, int im_size)
    void rotate_volume(float* data, float* res, float angle, int im_size, int slices_count)
    void rotate_volume_z(float* data, float* res, float angle, int im_size, int slices_count)

    void backproject(float* vin, float* vout, int im_size)
    void backproject_volume(float* data, float* res, int im_size, int slice_count)
    void backproject_volume_z(float* data, float* res, float lambd, int im_size, int slice_count)

    void project(float* vin, float* vout, int im_size)
    void project_volume(float* data, float* res, int im_size, int slice_count)
    void project_volume_z(float* data, float* res, int im_size, int slice_count)

    void add_mt(float* v1, float* v2, int im_size)
    void volume_filter(float* data, int elements_count)

def project_fast(numpy.ndarray[numpy.float32_t, ndim=2] vin):
    cdef numpy.int32_t count=vin.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=1] vout=numpy.empty(count,dtype='float32')
    project(<float *>vin.data,<float *> vout.data,count)
    return vout

def project_fast_ref(numpy.ndarray[numpy.float32_t, ndim=2] vin, numpy.ndarray[numpy.float32_t, ndim=1] vout):
    cdef numpy.int32_t count=vin.shape[0]
    project(<float *>vin.data,<float *> vout.data,count)

def project_volume_fast(numpy.ndarray[numpy.float32_t, ndim=3] data):
    cdef numpy.int32_t im_size=data.shape[1]
    cdef numpy.int32_t slice_count=data.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=2] res=numpy.empty(shape=(slice_count,im_size),
        dtype='float32')
    project_volume(<float *> data.data, <float *> res.data, im_size,slice_count)
    return res

def project_volume_z_fast(numpy.ndarray[numpy.float32_t, ndim=3] data):
    cdef numpy.int32_t im_size=data.shape[1]
    cdef numpy.int32_t slice_count=data.shape[2]
    cdef numpy.ndarray[numpy.float32_t, ndim=2] res=numpy.empty(shape=(im_size,slice_count),
        dtype='float32')
    project_volume_z(<float *> data.data, <float *> res.data, im_size,slice_count)
    return res

def backproject_fast(numpy.ndarray[numpy.float32_t, ndim=1] vin):
    cdef numpy.int32_t count=vin.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=1] tmpin=vin/count
    cdef numpy.ndarray[numpy.float32_t, ndim=2] vout=numpy.empty((count,count),dtype='float32')
    backproject(<float *>tmpin.data,<float *> vout.data,count)
    return vout

def backproject_fast_ref(numpy.ndarray[numpy.float32_t, ndim=1] vin,numpy.ndarray[numpy.float32_t, ndim=2] vout):
    cdef numpy.int32_t count=vin.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=1] tmpin=vin/count
    backproject(<float *>tmpin.data,<float *> vout.data,count)

def backproject_volume_fast(numpy.ndarray[numpy.float32_t, ndim=2] data,
                            numpy.ndarray[numpy.float32_t, ndim=3] res):
    cdef numpy.int32_t im_size=data.shape[1]
    cdef numpy.int32_t slice_count=data.shape[0]
    backproject_volume(<float *> data.data, <float *> res.data, im_size,slice_count)

def backproject_volume_z_fast(numpy.ndarray[numpy.float32_t, ndim=2] data,
                              numpy.ndarray[numpy.float32_t, ndim=3] res, numpy.float32_t lambd):
    cdef numpy.int32_t im_size=data.shape[0]
    cdef numpy.int32_t slice_count=data.shape[1]
    backproject_volume_z(<float *> data.data, <float *> res.data, lambd, im_size,slice_count)

def rotate_sq_fast(numpy.ndarray[numpy.float32_t, ndim=2] data, numpy.float32_t angle):
    cdef numpy.int32_t im_size=data.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=2] res=numpy.empty((im_size,im_size),dtype='float32')
    rotate_mt(<float *> data.data,<float *> res.data, angle, im_size)
    return res

def rotate_sq_fast_ref(numpy.ndarray[numpy.float32_t, ndim=2] data, numpy.float32_t angle,
    numpy.ndarray[numpy.float32_t, ndim=2] res):
    cdef numpy.int32_t im_size=data.shape[0]
    rotate_mt(<float *> data.data,<float *> res.data, angle, im_size)

def rotate_volume_fast(numpy.ndarray[numpy.float32_t, ndim=3] data, numpy.float32_t angle):
    cdef numpy.int32_t im_size=data.shape[1]
    cdef numpy.int32_t slice_count=data.shape[0]
    cdef numpy.ndarray[numpy.float32_t, ndim=3] res=numpy.empty_like(data)
    rotate_volume(<float *> data.data, <float *> res.data, angle, im_size,slice_count)
    return res

def rotate_volume_z_fast(numpy.ndarray[numpy.float32_t, ndim=3] data, numpy.float32_t angle):
    cdef numpy.int32_t im_size=data.shape[0]
    cdef numpy.int32_t slices_count=data.shape[2]
    cdef numpy.ndarray[numpy.float32_t, ndim=3] res=numpy.empty_like(data)
    rotate_volume_z(<float *> data.data, <float *> res.data, angle, im_size,slices_count)
    return res

def volume_filter_fast(numpy.ndarray[numpy.float32_t, ndim=3] data):
    cdef numpy.int32_t elements_count=data.shape[0]*data.shape[1]*data.shape[2]
    volume_filter(<float *> data.data , elements_count)

def add_fast_ref(numpy.ndarray[numpy.float32_t, ndim=2] v1, numpy.ndarray[numpy.float32_t, ndim=2] v2):
#    TODO: chek shape(v1)=shape(v2)
    cdef numpy.int32_t count=v1.shape[0]*v1.shape[1]
    add_mt(<float *>v1.data,<float *> v2.data,count)


def summ_fast(numpy.ndarray[numpy.float32_t, ndim=3] v1, numpy.ndarray[numpy.float32_t, ndim=3] v2):
#    TODO: chek shape(v1)=shape(v2)
    cdef numpy.int32_t count=v1.shape[0]*v1.shape[1]*v1.shape[2]
    add_mt(<float *>v1.data,<float *> v2.data,count)




