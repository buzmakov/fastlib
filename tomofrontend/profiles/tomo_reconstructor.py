#encoding: utf-8
import sys
import os
root_dir=os.path.join(os.path.dirname(__file__),'..','..')
sys.path.insert(0,root_dir)

from fastlib.imageprocessing.ocv import rotate
from fastlib.tomography.sart import sart
from fastlib.utils.save_amira import save_amira
from utils.natsort import natsorted
from utils.mylogger import set_logger, add_logging_file
from tomofrontend.configutils.yamlutils import read_yaml


import itertools
import multiprocessing
import optparse
import cv2
import warnings
import numpy
import time
import logging
import pylab
import h5py
import scipy.ndimage
import scipy.ndimage.measurements
warnings.filterwarnings("ignore", category=Warning)


__author__ = 'makov'

def get_frame_reader(image_format):
    """
    Return frame-reader function by config string

    Now supported only **fits** and **raw** format.

    :param image_format: form config file
    :return: function returned array from file
    """
    if image_format=='fits':
        import fastlib.datarader.fitsreader as framereader
        return framereader.get_frame_from_file
    elif image_format == 'raw':
        import fastlib.datarader.binreader as framereader
        return framereader.get_frame_from_file
    elif image_format == 'tiff':
        import fastlib.datarader.tiffreader as framereader
        return framereader.get_frame_from_file
    elif image_format == 'number':
        return numpy.array
    else:
        return None

def load_preprocess_frame(frame, image_process_config=None, dark_frame=None):
    """
    Load frame from file and make simple preprocess actions
    :param frame:
    :param image_process_config:
    :return: {'data': ..., 'angle',...}
    """
    reader=get_frame_reader(frame['image_format'])
    data=reader(frame['file_name'])
    multiplyer=1.0
    if 'exposure_time' in frame:
        multiplyer*=frame['exposure_time']
    if 'beam_current' in frame:
        multiplyer*=frame['beam_current']
    if 'rotation_angle' in frame and not data.ndim==0:
        data=numpy.rot90(data,int(frame['rotation_angle']/90))
    if not image_process_config is None:
        if 'ROI' in image_process_config and not data.ndim==0:
            cr=image_process_config['ROI']
            data=data[cr['vertical'][0]:cr['vertical'][1],
                    cr['horizontal'][0]:cr['horizontal'][1]]
            logging.info(str.format('Crop frame to {0} {1}',cr['horizontal'],cr['vertical']))

    if not dark_frame is None:
        data-=dark_frame

    data/=multiplyer
    return {'data':data,'angle':frame['angle']}


def get_mean_prepoces_frame(frames,image_process_config=None, dark_frame=None):
    """
    Calculate mean empty frame from list of frames

    :param frames: list of frames
    :param image_process_config: dictionary of rules to preprocessing
    :return: array - single frame
    """
    mean_frame=0
    for f in frames:
        mean_frame+=load_preprocess_frame(f,image_process_config,dark_frame)['data']
    mean_frame/=len(frames)
    return mean_frame

def normalize_data_frames(data_frames,empty_frame, dark_frame,image_process_config=None):
    """
    Build single frame from list of frames. Frame normalized to empty frame

    :param data_frames:
    :param empty_frame:
    :param dark_frame:
    :return: list of frames ()each frame - dict{angle, data}
    """
    ed_frame=empty_frame
    ed_frame=ed_frame*(ed_frame>=1)+1.0*(ed_frame<1)
    mean_data=get_mean_prepoces_frame(data_frames,image_process_config,dark_frame=dark_frame)
    dd_frame=mean_data
    dd_frame=dd_frame*(dd_frame>=1)+1.0*(dd_frame<1)
    tmp_data=dd_frame/ed_frame
    tmp_data=tmp_data*(0<tmp_data)*(tmp_data<=1)+1.0*(tmp_data>1)
    tmp_data=-cv2.log(tmp_data)
#    tmp_data=tmp_data*(0<tmp_data)*(tmp_data<=5)+5.0*(tmp_data>5)
    tmp_angle=float(data_frames[0]['angle'])
    return {'angle':tmp_angle,'data':tmp_data}

def build_grouped_by(config,key):
    """
    Group files by key
    """
    return  dict((k,list(v)) for k,v in itertools.groupby(config, lambda x: x[key]))

def save_frames_hdf5(data_norm, h5_file):
    """
    Save list of frames to HDF5 file
    :param data_norm:
    :param h5_file:
    """
    logging.info(str.format('Start saving HDF5 file'))
    with  h5py.File(h5_file, 'w') as h5_file:
        data_size = [0, 0, len(data_norm)]
        image_shape = data_norm[0]['data'].shape
        data_size[0] = image_shape[0]
        data_size[1] = image_shape[1]
        dset = h5_file.create_dataset('Data', data_size, 'f',chunks=True)
        aset = h5_file.create_dataset('Angles', (len(data_norm),), 'f',chunks=True)
        for di in range(len(data_norm)):
            dset[:, :, di] = data_norm[di]['data']
            aset[di] = data_norm[di]['angle']
    logging.info(str.format('Stop saving HDF5 file'))

def find_axis_hor_shift(im0, im1, xmin, xmax):
    """
    Find axis shift position by 2 images in range of horizontal shifting [xmin,xmax]

    :param im0: image0
    :param im1: image1
    :param xmin:
    :param xmax:
    :return: (correlation function, optimal shift value)
    """
    if xmin==xmax:
        return 1.0, xmin
    mean_im0=numpy.mean(im0)
    mean_im1=numpy.mean(im1)
    coeff_corr=((im0-mean_im0)**2).sum()*((im1-mean_im1)**2).sum()
    corr = []
    for i in range(xmin, xmax):
        tmp_im0=numpy.roll(im0, shift=i, axis=1)
        tmp_im1=numpy.roll(im1, shift=-i, axis=1)
        tmp_corr = (tmp_im0 - mean_im0)*(tmp_im1 - mean_im1)
        corr.append(numpy.sqrt(tmp_corr * tmp_corr).sum()/numpy.sqrt(coeff_corr))
    max_ind = numpy.array(corr).argmax()
    shift_val = range(xmin, xmax)[max_ind]
    shift_val = int(shift_val)
    logging.info(str.format('Found horizontal shift {0} with corr_factor={1:.3}',shift_val,corr[max_ind]))
    return corr,shift_val

def find_axis_rotate(im0, im1, angmin, angmax, da):
    """
    Find axis rotation position by 2 images in range of rotation [angmin, angmax, da]

    :param im0:
    :param im1:
    :param angmin:
    :param angmax:
    :param da:
    :return:
    """
    if angmin==angmax:
        return 1.0, angmin

    mean_im0=numpy.mean(im0)
    mean_im1=numpy.mean(im1)
    coeff_corr=((im0-mean_im0)**2).sum()*((im1-mean_im1)**2).sum()
    corr_rot = []
    for ang in numpy.arange(angmin, angmax, da):
        tmp_im0=rotate(im0, ang)
        tmp_im1=rotate(im1, -ang)
        tmp_corr = (tmp_im0 - mean_im0)*(tmp_im1 - mean_im1)
        corr_rot.append(numpy.sqrt(tmp_corr * tmp_corr).sum()/numpy.sqrt(coeff_corr))
    max_ind = numpy.array(corr_rot).argmax()
    rot_val = numpy.arange(angmin, angmax, da)[max_ind]
    logging.info(str.format('Found rotation angle {0:.5} with corr_factor={1:.3}',rot_val,corr_rot[max_ind]))
    return corr_rot,rot_val

def get_index_opposite_files(data_norm):
    """
    get indexes of opposite files (ang0-ang1==180)

    :param data_norm: list of frames {'angle'..., ...}
    """
    angles = numpy.array([d['angle'] for d in data_norm])
    ang_diff = (angles - angles.reshape([len(angles), 1]))
    ind = scipy.ndimage.measurements.minimum_position(abs(ang_diff-180))
    logging.info(str.format('Found oposit files at angles {0} and {1}',
        data_norm[ind[0]]['angle'],data_norm[ind[1]]['angle']))
    return ind

def my_filter(xf):
    """
    Filter for remove noise for searching axis position

    :param xf: array to filter
    :return:
    """
    xf=scipy.ndimage.median_filter(xf,[3,3])
    xf-=0.1*xf.max()
    xf*=(xf>0)
    return xf

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    :param l: list of smthns
    :type l: list
    :param n: size of chunk
    :type n: int
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def save_shifted_image(corr, corr_rot, data0, data1, fname):
    """
    Save diagnistics message.

    :param corr:
    :param corr_rot:
    :param data0:
    :param data1:
    :param fname:
    """
    pylab.figure()
    pylab.subplot(2, 3, 1)
    pylab.imshow(data0)
    pylab.colorbar()
    pylab.subplot(2, 3, 2)
    pylab.imshow(data1)
    pylab.colorbar()
    pylab.subplot(2, 3, 3)
    pylab.imshow(data1 - data0)
    pylab.colorbar()
    pylab.subplot(2, 3, 4)
    pylab.plot(corr_rot)
    pylab.subplot(2, 3, 5)
    pylab.plot(corr)
    pylab.savefig(fname)
    pylab.close(pylab.gcf())


def find_shift_and_rotation(data0, data1, image_process_config, res_folder=None):
    """
    Find axis rotation and shift

    :param data0:
    :param data1:
    :param image_process_config:
    :param res_folder: folder to save diagnostic image
    :return:
    """
    ipc = image_process_config['axis_serch']
    x0 = ipc['horizontal'][0]
    x1 = ipc['horizontal'][1]
    a0 = ipc['rotation'][0]
    a1 = ipc['rotation'][1]
    x0_start=x0
    x1_start=x1
    a0_start=a0
    a1_start=a1

    delta_x = int(x1 - x0) / 2
    xc = int(x0 + x1) / 2
    delta_a = float(a1 - a0) / 2.0
    ac = (a0 + a1) / 2
    angle_step = 0.01
    tmp_data0 = data0
    tmp_data1 = data1

    corr=None
    corr_rot=None

    for i in range(10):
        corr, shift_val = find_axis_hor_shift(tmp_data0, tmp_data1, xc - delta_x, xc + delta_x)

        if shift_val<x0_start:
            shift_val=x0_start
        elif shift_val>x1_start:
            shift_val=x1_start

        tmp_data0 = numpy.roll(data0, shift=shift_val, axis=1)
        tmp_data1 = numpy.roll(data1, shift=-shift_val, axis=1)

        corr_rot, rot_ang = find_axis_rotate(tmp_data0, tmp_data1, ac - delta_a, ac + delta_a, angle_step)

        if rot_ang<a0_start:
            rot_ang=a0_start
        elif rot_ang>a1_start:
            rot_ang=a1_start

        if not delta_a==0:
            delta_a = max(0.1, 0.8 * delta_a)
            angle_step = max(0.001, angle_step * 0.8)

        if not delta_x==0:
            delta_x = max(10, int(delta_x * 0.5))

        tmp_data0 = rotate(data0, rot_ang)
        tmp_data1 = rotate(data1, -rot_ang)

        if  xc == shift_val and ac == rot_ang:
            break

        xc = shift_val
        ac = rot_ang

        logging.info(str.format('x={0} delta_x={1} ac={2:.3} delta_a={3:.3} angle_step={4:.3}',
            xc, delta_x, ac, delta_a, angle_step))
    data0 = numpy.roll(data0, shift=xc, axis=1)
    data1 = numpy.roll(data1, shift=-xc, axis=1)
    data0 = rotate(data0, ac)
    data1 = rotate(data1, -ac)
    if not res_folder is None:
        save_shifted_image(corr, corr_rot, data0, data1,os.path.join(res_folder,'shifting_pic.png'))
    return {'angle':ac, 'shift':xc}

def sart_wrapper(job):
    """
    Wrapper for multiprocessing.

    :param job:
    :return:
    """
    angles, sinogram =job
    return sart(sinogram,angles)


def get_normalized_frames(frames_groups, image_process_config):
    """
    Build normalized frames from groups of frames.

    :param frames_groups:
    :param image_process_config:
    :return:
    """
    normalized_data = []
    for group_name in natsorted(frames_groups.keys()):
        cur_group = frames_groups[group_name]
        frames_by_type = build_grouped_by(cur_group, 'frame_type')
        data_frames_by_angle = build_grouped_by(frames_by_type['data'], 'angle')
        dark_frames_by_angle = build_grouped_by(frames_by_type['dark'], 'angle')
        empty_frames_by_angle = build_grouped_by(frames_by_type['empty'], 'angle')
        mean_dark_frame = get_mean_prepoces_frame(dark_frames_by_angle[dark_frames_by_angle.keys()[0]],
            image_process_config)

        mean_empty_frame = get_mean_prepoces_frame(empty_frames_by_angle[empty_frames_by_angle.keys()[0]],
            image_process_config,dark_frame=mean_dark_frame)


        for df_angle in data_frames_by_angle:
            normalized_data.append(normalize_data_frames(data_frames_by_angle[df_angle],
                mean_empty_frame, mean_dark_frame, image_process_config))
        #            pylab.figure()
        #            pylab.imshow(normalized_data[-1]['data'])
        #            pylab.colorbar()
        #            pylab.show()
    return normalized_data


def fix_axis_position(angle_val, shift_val, normalized_data, pixel_size):
    """
    Inplace fix axis position for list of frames.

    :param angle_val:
    :param shift_val:
    :param normalized_data:
    :param pixel_size:
    """
    for i in range(len(normalized_data)):
        normalized_data[i]['data'] = numpy.roll(normalized_data[i]['data'], shift=shift_val, axis=1)
        normalized_data[i]['data'] = rotate(normalized_data[i]['data'], angle_val)
        normalized_data[i]['data'] = normalized_data[i]['data'] / pixel_size*1000 #absorption in mm^-1


def make_tomo_reconstruction(h5_postprocess_file, h5_result_file):
    """
    Make multiprocessing tomo reconstruction.

    :param h5_postprocess_file:
    :param h5_result_file:
    """
    imap_chank=2
    chunksize = imap_chank*multiprocessing.cpu_count()
    pool = multiprocessing.Pool()
    with  h5py.File(h5_postprocess_file, 'r') as data_file:
        with h5py.File(h5_result_file, 'w') as res_file:
            t_total = time.time()
            numb_of_sinograms = data_file['Data'].shape[0]
            image_shape = data_file['Data'][0].shape
            slice_h5 = res_file.create_dataset('Results',
                (numb_of_sinograms, image_shape[0], image_shape[0]),'f',chunks=True)
#            data_for_imap = []
            for isinos in list(chunks(range(numb_of_sinograms), chunksize)):
                sinogramms = []
                angles = numpy.array(data_file['Angles'], dtype='float32')
                for isino in isinos:
                    sinogramms.append(numpy.array(data_file['Data'][isino], dtype='float32'))

                time_slice = time.time()
                job_list = ((angles, sinogramm) for sinogramm in sinogramms)

                result = pool.imap(sart_wrapper, job_list, imap_chank)
                #       for Profiling python -m cProfile -s cumulative main.py > profile.txt
                #                result=itertools.imap(cv_sart,job_list)

                for ni, tomo_rec in enumerate(result):
                    isino = isinos[ni]
                    try:
                        logging.info(str.format('Slice {0} / {1} done in {2:.3} sec. Estimated time {3:.3} min',
                            isino, numb_of_sinograms, time.time() - time_slice,
                            (time.time() - t_total) / isino * (numb_of_sinograms - isino) / 60)
                        )
                    except ZeroDivisionError:
                        logging.info(str.format('Slice {0} / {1} done in {2:.3} sec.',
                            isino, numb_of_sinograms, (time.time() - time_slice)/(imap_chank*chunksize)))

                    slice_h5[isino, :, :] = tomo_rec
#                res_file.flush()
    pool.close()
    logging.info(str.format('Total reconstruction time {0:.3} min', (time.time() - t_total) / 60))

def get_profile_name(data_root):
    """
    Get profile name from **exp_description.yaml** file

    :param data_root:
    :return:
    """
    config_yaml_filename=os.path.join(data_root,'exp_description.yaml')
    config=read_yaml(config_yaml_filename)[0]
    tomo_profile=config['profile']
    return tomo_profile

def make_images(h5name):
    """
    Make images from hdf5 file with 'Results' node
    :param h5name:
    """
    dir_name=os.path.dirname(os.path.abspath(h5name))
    images_dir=os.path.join(dir_name,'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)

    slice_numbers=50
    with  h5py.File(h5name,'r') as data_file:
        data=data_file["Results"]
        pylab.figure()
        for label in ['x','y','z']:
            for i in range(slice_numbers):
                pylab.clf()

                if label=='x':
                    sh=data.shape[0]
                elif label=='y':
                    sh=data.shape[1]
                elif label=='z':
                    sh=data.shape[2]
                else:
                    sh=max(data.shape)
                slice_numb=int((i+1)*sh/(slice_numbers+2))

                if label=='x':
                    tmp_slice=data[slice_numb,:,:]
                elif label=='y':
                    tmp_slice=data[:,slice_numb,:]
                elif label=='z':
                    tmp_slice=data[:,:,slice_numb]

                pylab.imshow(tmp_slice,vmin=min(0,tmp_slice.max()))
                pylab.colorbar()
                image_name=str.format('{0}_{1:03}.png',label,slice_numb)
                pylab.savefig(os.path.join(images_dir,image_name))
                logging.info('Saving image '+image_name)


def preprocess_and_save_all_data(data_root, res_folder, tomo_profile=None):
    """
    Preprocess all data (build universal config, normalize, shift, rotate, save to hdf5)
    :param data_root:
    :param res_folder:
    :param tomo_profile:
    :return:
    :raise:
    """
    if tomo_profile is None:
        tomo_profile = get_profile_name(data_root)
    if  tomo_profile == 'amur':
        from tomofrontend.profiles.amur_config import build_universal_config
    elif tomo_profile == 'senin':
        from tomofrontend.profiles.senin_config import build_universal_config
    elif tomo_profile == 'mediana':
        from tomofrontend.profiles.mediana_config import build_universal_config
    else:
        raise NameError, str.format('Unknown profile type {0}', tomo_profile)

    build_universal_config(data_root)

    config = read_yaml(os.path.join(data_root, 'uni_config.yaml'))[0]

    image_process_config = config['preprocess_config']

    frames_groups = config['frames']

    normalized_data = get_normalized_frames(frames_groups, image_process_config)

    op_ind = get_index_opposite_files(normalized_data)
    data0 = normalized_data[op_ind[0]]['data']
    data1 = numpy.fliplr(normalized_data[op_ind[1]]['data'])
    data0 = my_filter(data0)
    data1 = my_filter(data1)

    sh_ang = find_shift_and_rotation(data0, data1, image_process_config, res_folder)
    shift_val = sh_ang['shift']
    angle_val = sh_ang['angle']

    pixel_size = image_process_config['pixel_size']

    fix_axis_position(angle_val, shift_val, normalized_data, pixel_size)

    h5_postprocess_file = os.path.join(res_folder, 'postproc_data.h5')

    save_frames_hdf5(normalized_data, h5_postprocess_file)

    return h5_postprocess_file


def do_tomo_reconstruction(data_root, tomo_profile=None, just_preprocess=False):
    """
    Do full reconstruction chain.

    :param data_root: path to directory where 'original' folder
    :param tomo_profile:
    :param just_preprocess: if True, then only preprocessed images rendered, in other case tomo reconstruction preferred
    :type just_preprocess: bool
    :raise:
    """
    #check if data folder exists
    data_root=os.path.join(data_root,'original')
    if not os.path.isdir(data_root):
        raise IOError, str.format('Data folder {0} not found.', data_root)

    #creating result folder if not exists
    res_folder = os.path.join(data_root, '..', 'reconstruction')
    if not os.path.isdir(res_folder):
        try:
            os.mkdir(res_folder)
        except OSError:
            raise IOError, str.format('Can not create results folder {0}.', res_folder)

    log_file = os.path.join(res_folder, 'tomo.log')

    logger_handler=add_logging_file(log_file)

    h5_postprocess_file = preprocess_and_save_all_data(data_root, res_folder, tomo_profile)

    if not just_preprocess:
        h5_result_file = os.path.join(res_folder, 'result.hdf5')
        make_tomo_reconstruction(h5_postprocess_file, h5_result_file)
        save_amira(h5_result_file)
        make_images(h5_result_file)

    logging.getLogger('').removeHandler(logger_handler)

if __name__=="__main__":
    set_logger()
    parser = optparse.OptionParser("Usage: %prog [options]")
    parser.add_option('-d','--data-directory', dest="data_directory", action="store", default=os.path.curdir,
        help='Directory with tomographic data')
    parser.add_option('-p','--profile', dest="profile", action="store", default=None,
        help='Reconstruction profile: amur, ...')

    (options, args) = parser.parse_args()
    data_root=options.data_directory
    tomo_profile=options.profile

    do_tomo_reconstruction(data_root, tomo_profile)
