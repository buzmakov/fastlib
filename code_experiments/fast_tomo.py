#encoding: utf-8
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import sys
import numpy
import pylab
import h5py
import time

sys.path.insert(0, '..')
import fastlib
from fastlib.utils.phantom import modified_shepp_logan
import fastlib.imageprocessing.ispmd.ispmd_image_processing as myfast


def show_center_slice(x):
    pylab.figure()
    pylab.imshow(x[:, :, int(x.shape[2] / 2)], cmap=pylab.cm.Greys)
    pylab.colorbar()
    pylab.show()


def generate_sinogramm(volume, angles):
    sinograms = numpy.empty(shape=(volume.shape[1], volume.shape[2], angles.shape[0]), dtype='float32')
    for ia, ang in enumerate(angles):
        sinograms[:, :, ia] = myfast.project_volume_z_fast(myfast.rotate_volume_z_fast(sl, ang))
    return sinograms


# @profile
def sart(sinogram, angles):
    project = myfast.project_volume_z_fast
    rotate = myfast.rotate_volume_z_fast
    back_project = myfast.backproject_volume_z_fast

    if not sinogram.shape[2] == angles.shape[0]:
        raise TypeError('Sinogramm shape mismach with angles count')
    res = numpy.zeros(shape=(sinogram.shape[0], sinogram.shape[0],
                             sinogram.shape[1]), dtype='float32')
    tmp_vol = numpy.empty_like(res)
    #tmp_proj=numpy.empty(shape=(sinogram.shape[0],sinogram.shape[1]),dtype='float32')
    for l in numpy.array([0.8, 0.3, 0.1, ], dtype='float32'):
        ang_numbers = range(len(angles))
        numpy.random.shuffle(ang_numbers)
        for ia in ang_numbers:
            ang = angles[ia]
            print "{} {}/{}".format(l, ia, len(angles))
            tmp_vol = rotate(res, ang)
            tmp_proj = project(tmp_vol)
            tmp_proj = sinogram[:, :, ia] - tmp_proj
            back_project(tmp_proj, tmp_vol, l)
            tmp_vol = rotate(tmp_vol, -ang)
            myfast.summ_fast(res, tmp_vol)
            myfast.volume_filter_fast(res)
    return res

if __name__ == "__main__":
    sl = None
    with h5py.File('ph_new.h5', 'r') as ph:
        sl = ph['ph'].value

    # Увелииваем количество солёв в n раз
    # sl = numpy.repeat(sl, 2, axis=-1)
    # sl = numpy.repeat(sl, 2, axis=0)
    # sl = numpy.repeat(sl, 2, axis=1)
    center_slice_numb = int(sl.shape[2] / 2)
    sl = sl[:, :, center_slice_numb - 8:center_slice_numb + 8]
    sl = sl.copy()
    print sl.shape

    angles = numpy.arange(0, 180, 1)
    sinogr = generate_sinogramm(sl, angles)
    t = time.time()
    res = sart(sinogr, angles)
    print 'Total time: ', time.time() - t
    show_center_slice(res - sl)
