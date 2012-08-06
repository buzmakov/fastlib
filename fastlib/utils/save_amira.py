# coding=utf-8
# coding=utf-8
#encodnig: utf-8
__author__ = 'makov'
import logging
import os
import h5py
import numpy

def save_amira(result_file):
    """
    Функция сохраняет реконструированные слои в формате Amira raw file

    Inputs:
        data_path - путь к директории, где находиться файл res_tomo.hdf5 в формате HDF5
            в этом файде должен быть раздел (node) /Results в котором в виде 3D массива
            записаны реконструированный объём
    Outputs:
        Файлы amira.raw и tomo.hx. Если файлы уже существуют, то они перезаписываются.
        Тип данных: float32 little endian
    """
    logging.info('Saving Amira files')
    data_path=os.path.dirname(result_file)
    with open(os.path.join(data_path,'amira.raw'),'wb') as amira_file:
        with h5py.File(result_file,'r') as h5f:
            x=h5f['Results']
            for i in range(x.shape[0]):
                numpy.array(x[i,:,:]).tofile(amira_file)

            file_shape=x.shape

            with open(os.path.join(data_path,'tomo.hx'),'w') as af:
                af.write('# Amira Script\n')
                af.write('remove -all\n')
                af.write(r'[ load -raw ${SCRIPTDIR}/amira.raw little xfastest float 1 '+
                         str(file_shape[1])+' '+str(file_shape[2])+' '+str(file_shape[0])+
                         ' 0 '+str(file_shape[1]-1)+' 0 '+str(file_shape[2]-1)+' 0 '+str(file_shape[0]-1)+
                         ' ] setLabel tomo.raw\n')