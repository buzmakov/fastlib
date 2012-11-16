# coding=utf-8
__author__ = 'makov'

import numpy
import scipy.ndimage


def project(src):
    """
    Project 2D array for parallel case (summing along rows)

    src=numpy.array([[ 0.,  1.,  2.],
       [ 3.,  4.,  5.],
       [ 6.,  7.,  8.]], dtype=float32)

    >>project(src)
    array([  3.,  12.,  21.], dtype=float32)

    :param src: 2D numpy array
    :return:
    """
    return numpy.sum(src, axis=-1, dtype='float32')


def back_project(src):
    """
    Backproject 1D array to 2D array.

    >>src=numpy.arange(4)
    >>back_project(src)
   array([[ 0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.25,  0.25,  0.25,  0.25],
       [ 0.5 ,  0.5 ,  0.5 ,  0.5 ],
       [ 0.75,  0.75,  0.75,  0.75]])

    :param src:
    :return:
    """
    y = numpy.zeros([len(src), ] * 2, dtype='float32')
    y[:, ] = src / float(len(src))
    res = numpy.rot90(y, 3)
    return res


def rotate_square_image(src, angle):
    """

    :param src: square numpy array
    :param angle: rotation angle in degrees CCW
    """
    res = scipy.ndimage.rotate(src, angle, order=1, reshape=False).astype('float32')
    return res