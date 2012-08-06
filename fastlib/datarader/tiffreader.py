"""
Module for reading frame from TIFF file. (Podurets format)
"""
import os
import numpy
import Image
import logging

#TODO: Add tests
def get_frame_from_file(file_path):
    """
    Get array from tiff file

    :param file_path: Path to fits file
    :return: numpy array of float32
    :rtype: numpy.array
    :raise: IOError if input file not exists.
    """
    if os.path.exists(file_path):
        logging.info(str.format('Reading TIFF file {0}',file_path))
        image = Image.open(file_path)
        a = numpy.array(image.getdata(), dtype='uint16').reshape(image.size).astype('float32')
        return a
    else:
        raise IOError, str.format('File {0} not found.', file_path)