import numpy
from fastlib.imageprocessing.ispmd import project,rotate_square_ref,back_project_ref,add_ref

__author__ = 'makov'
#@profile
def sart(sinogram,angles):
    reconst_shape = numpy.array(sinogram.shape[0],dtype='int32')
#    #normalize sinograms
#    sinogram *= numpy.mean(sinogram.sum(axis=0)) / sinogram.sum(axis=0)
    tomo_rec = numpy.zeros((reconst_shape, reconst_shape), dtype='float32')
    tmp_backproj=numpy.zeros_like(tomo_rec)
    tmp_rot=numpy.zeros_like(tomo_rec)
    for coeff in numpy.array([0.8,0.3,],dtype='float32'):
        shuffle_iang = numpy.arange(len(angles))
        numpy.random.shuffle(shuffle_iang)
        for iang in shuffle_iang:
            rotate_square_ref(tomo_rec, angles[iang],tmp_rot)
            tmp_proj = sinogram[:, iang] - project(tmp_rot)
            back_project_ref(tmp_proj * coeff,tmp_backproj)
            rotate_square_ref(tmp_backproj, -angles[iang],tmp_rot)
            add_ref(tomo_rec,tmp_rot)
        tomo_rec -= 0.5*tomo_rec*(tomo_rec < 0)
    return tomo_rec