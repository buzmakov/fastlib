import cv2
import numpy

def cv_rotate_sq(x,angle):
    """
    Rotate square array using OpenCV2 around center of the array
    :param x: numpy array
    :param angle: angle in degrees
    :return: rotated array
    """
    if not x.shape[0]==x.shape[1]:
        raise TypeError, 'Image not square'
    return cv_rotate(x,angle)

def cv_rotate(x,angle):
    """
    Rotate square array using OpenCV2 around center of the array
    :param x: numpy array
    :param angle: angle in degrees
    :return: rotated array
    """
    x_center=tuple(numpy.array((x.shape[1],x.shape[0]),dtype='float32')/2.0-0.5)
    rot_mat=cv2.getRotationMatrix2D(x_center,angle,1.0)
    xro=cv2.warpAffine(x,rot_mat,(x.shape[1],x.shape[0]),flags=cv2.INTER_LINEAR)
    return xro

def cv_project(src):
    return numpy.squeeze(cv2.reduce(src,dim=1,rtype=cv2.cv.CV_REDUCE_SUM))

def cv_backproject(src):
    ntimes=len(src)
    tmp_array=src/float(ntimes)
    return cv2.repeat(tmp_array,1,ntimes)