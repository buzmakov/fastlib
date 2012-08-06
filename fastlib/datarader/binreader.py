__author__ = 'Alexey Buzmakov'
import numpy
import os
import logging

def get_frame_from_file(file_path):
    """
    Get array from bin file in R.Senin's format

    :param file_path: Path to fits file
    :return: numpy array of float32
    :raise: IOError if input file not exists.
    """
    #TODO: now it is only Senin's format support (1024x1024)
    if os.path.exists(file_path):
        logging.info(str.format('Reading RAW file {0}',file_path))
        a = numpy.fromfile(file_path,dtype='>u2').reshape((1024,1024)).astype('float32')
        return a
    else:
        raise IOError, str.format('File {0} not found.', file_path)