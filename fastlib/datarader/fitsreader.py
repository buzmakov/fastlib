"""
Module for reading frame from FITS file. (FIAN format)
"""

import os
import pyfits
import numpy as np
import logging

#TODO: Add tests
def get_frame_from_file(file_path):
    """
    Get array from fits file

    :param file_path: Path to fits file
    :return: numpy array of float32
    :raise: IOError if input file not exists.
    """
    if os.path.exists(file_path):
        fi=pyfits.open(file_path,ignore_missing_end=True)
        a=np.array(fi[0].data,dtype='uint16').astype('float32')
        fi.close()
        logging.info(str.format('Reading FITS file {0}',file_path))
        return a
    else:
        raise IOError, str.format('File {0} not found.',file_path)
    