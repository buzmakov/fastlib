# coding=utf-8
"""
This file should provide function build_universal_config(config_dir) to build  config from dir.
"""
import glob
import os
from tomofrontend.configutils.universalconfig import UniversalConfigBuilder
from tomofrontend.profiles.amur_config import get_groups, get_files_in_group
from utils.natsort import natsorted
import tomofrontend.configutils.yamlutils as yamlutils
from utils.dictutils import get_value, try_get_value
#from utils.mypprint import pprint
import logging

__author__ = 'makov'


def build_universal_config(config_dir):
    """
    Build uni_config.yaml in the config_dir

    :param config_dir: directory where  exp_description.yaml
    :raise:
    """
    config_yaml_filename = os.path.join(config_dir, 'exp_description.yaml')
    res = yamlutils.read_yaml(config_yaml_filename)[0]
    #    pprint(res)

    if not 'frames' in res:
        logging.error(str.format('Section FRAMES not found in config file {0}', config_yaml_filename))
        raise AttributeError
    uc = UniversalConfigBuilder()
    #TODO: fix this hack
    uc.add_section({'preprocess_config': res['preprocess_config']})
    uc.add_section({'description': res['description']})
    data_dir = os.path.join(config_dir, 'Data')
    files = glob.glob(os.path.join(data_dir, '*.tif'))
    for f_path in sorted(files):
        fname = os.path.split(f_path)[-1]
        f_prefix = fname.split('.')[0]
        if f_prefix == 'eb':
            uc.add_frame(group_name='0', frame_type='empty', file_name=f_path, image_format='tiff', angle=0)
        elif f_prefix == 'dc':
            uc.add_frame(group_name='0', frame_type='dark', file_name=f_path, image_format='tiff', angle=0)
        else:
            angle = float(f_prefix.replace(',', '.'))
            uc.add_frame(group_name='0', frame_type='data', file_name=f_path, image_format='tiff', angle=angle)
    uc.save2yaml(os.path.join(config_dir, 'uni_config.yaml'))

if __name__ == "__main__":
    from utils.mylogger import set_logger
    set_logger()
    config_dir = r'/home/makov/tmp/tomo_root/Raw/podurec/core_si02/original'
    build_universal_config(config_dir)