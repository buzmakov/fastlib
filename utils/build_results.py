__author__ = 'makov'

import glob
import os
import h5py
import pylab
import numpy
import scipy.ndimage

if __name__ == "__main__":
    data_root = r'/media/WD_ext4/bones/done'
    res_name = 'result.hdf5'

#    data_dirs=glob.glob(os.path.join(data_root,'M3_F4_2_s'))
    data_dirs = glob.glob(os.path.join(data_root, 'M2_F4_s'))
    out_dir = os.path.join(data_root, 'report')
    try:
        os.mkdir(out_dir)
    except OSError:
        pass

    for data_dir in data_dirs:
        prefix = os.path.split(data_dir)[1]
        print prefix
        res_file = os.path.join(data_dir, res_name)
        slice_numb = 20
        pylab.figure()
        with h5py.File(res_file, 'r') as data_file:
            data = data_file["Results"]
            s = data.shape[0]
            for i in range(slice_numb):
                tmp_data = data[int(s / (slice_numb + 2)) * i]
                if prefix[-1] == 's':
                    tmp_data /= 10
                tmp_data *= 1000
                tmp_data *= (tmp_data > 0)
#                tmp_data=tmp_data[250:-100,250:-100]
                tmp_data = tmp_data[100:-150, 100:-200]
#                tmp_data=scipy.ndimage.median_filter(tmp_data,[3,3])
                pylab.imshow(tmp_data, cmap=pylab.cm.Greys_r, vmax=1)
                pylab.colorbar()
                pylab.savefig(os.path.join(out_dir, str.format('{0}_{1:02}', prefix, i)), format='png')
                pylab.clf()