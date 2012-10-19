# coding=utf-8
import os
import h5py
import pylab
import logging

__author__ = 'makov'


def make_images(h5name):
    dir_name = os.path.dirname(os.path.abspath(h5name))
    images_dir = os.path.join(dir_name, 'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)

    slice_numbers = 15
    with h5py.File(h5name, 'r') as data_file:
        data = data_file["Results"]
        pylab.figure()
        for label in ['x', 'y', 'z']:
            for i in range(slice_numbers):
                pylab.clf()

                if label == 'x':
                    sh = data.shape[0]
                elif label == 'y':
                    sh = data.shape[1]
                elif label == 'z':
                    sh = data.shape[2]
                else:
                    sh = max(data.shape)
                slice_numb = int((i + 1) * sh / (slice_numbers + 2))

                if label == 'x':
                    tmp_slice = data[slice_numb, :, :]
                elif label == 'y':
                    tmp_slice = data[:, slice_numb, :]
                elif label == 'z':
                    tmp_slice = data[:, :, slice_numb]

                pylab.imshow(tmp_slice, vmin=min(0, tmp_slice.max()))
                pylab.colorbar()
                image_name = str.format('{0}_{1:03}.png', label, slice_numb)
                pylab.savefig(os.path.join(images_dir, image_name))
                logging.info('Saving image ' + image_name)

if __name__ == '__main__':
    make_images(r'/home/makov/tmp/tomo_root/Raw/bones/ikran/2012_02_17/M3_F1_cu/reconstruction/result.hdf5')
