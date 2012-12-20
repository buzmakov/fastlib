#encoding: utf-8
#cython: profile=False
import cython
cimport cython

import numpy
import logging
#import scipy

cimport numpy

ctypedef numpy.float32_t dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
def back_project(numpy.ndarray[dtype_t, ndim=1, mode="c"] projection,
    dtype_t angle, Py_ssize_t N, Py_ssize_t nx,rays,rotate_image):

    cdef numpy.ndarray[dtype_t, ndim=2, mode="c"] tmp_solution,tmp_ray
    cdef Py_ssize_t i,ii,jj
    cdef dtype_t cur_prj
    tmp_solution=numpy.zeros((N,N),dtype='float32')
    for i in range(nx):
        tmp_ray=rays[i].toarray()
        for ii in range(N):
            cur_prj=projection[ii]
            for jj in range(N):
                tmp_solution[ii,jj]+=tmp_ray[ii,jj]*cur_prj
        if not i%10:
            logging.debug(str.format('Back project cython {0} from {1}',i,nx))
    tmp_solution/=N
    tmp_solution=rotate_image(tmp_solution,-angle)
    return tmp_solution

@cython.boundscheck(False)
@cython.wraparound(False)
def direct_project(numpy.ndarray[dtype_t, ndim=2, mode="c"] image,
    dtype_t angle, Py_ssize_t N, Py_ssize_t nx, rays, rotate_image):

    cdef numpy.ndarray[dtype_t, ndim=2, mode="c"] tmp_image, tmp_ray
    cdef numpy.ndarray[dtype_t, ndim=1, mode="c"] tmp_projection
    cdef dtype_t tmp
    cdef Py_ssize_t i,ii, jj

    tmp_image=rotate_image(image,angle)
    tmp_projection=numpy.zeros(nx,dtype='float32')
    for i in range(nx):
        tmp_ray=rays[i].toarray()
        tmp=0
        for ii in range(N):
            for jj in range(N):
                tmp+=tmp_ray[ii,jj]*tmp_image[ii,jj]
        tmp_projection[i]=tmp
        if not i%10:
            logging.debug(str.format('Direct project cython {0} from {1}',i,nx))
    return tmp_projection