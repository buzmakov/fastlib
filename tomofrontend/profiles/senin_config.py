# coding=utf-8
"""
This file should provide function build_universal_config(config_dir) to build  config from dir.
"""
import os
from tomofrontend.configutils.universalconfig import UniversalConfigBuilder
import tomofrontend.configutils.yamlutils as yamlutils
import logging

__author__ = 'makov'

#TODO: add dark frames by value


def get_frame(config_dir, f, frames_folder, data_type):
    tmp_frame = {}
    tmp_frame['type'] = data_type
    tmp_frame['image_format'] = 'raw'
    tmp_frame['name'] = os.path.join(os.path.realpath(config_dir), frames_folder, f['file_name'])
    tmp_frame['angle'] = float(f['angle']) / 480.0
    tmp_frame['exposure_time'] = f['exposure']
    tmp_frame['beam_current'] = f['current']
    return tmp_frame


def get_files(config_dir):
    tomo_log = yamlutils.read_yaml(os.path.join(config_dir, 'log.txt'))[0]
    files = []

    data_files = tomo_log['---Data']
    frames_folder = 'Data'
    for f in data_files:
        tmp_frame = get_frame(config_dir, f, frames_folder, 'data')
        files.append(tmp_frame)

    empty_files = tomo_log['---Empty']
    frames_folder = 'Background'
    for f in empty_files:
        tmp_frame = get_frame(config_dir, f, frames_folder, 'empty')
        files.append(tmp_frame)

    return files


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
    files = get_files(config_dir)
    for f in files:
        uc.add_frame(group_name='0', frame_type=f['type'], file_name=f['name'],
                     image_format=f['image_format'], angle=f['angle'], exposure_time=f['exposure_time'],
                     rotation_angle=0)
    #add dark type by value
    if 'dark_value' in res['frames']:
        uc.add_frame(group_name='0', frame_type='dark', file_name=res['frames']['dark_value'],
                     image_format='number', angle=0, exposure_time=1, rotation_angle=0)
    uc.save2yaml(os.path.join(config_dir, 'uni_config.yaml'))

if __name__ == "__main__":
    from utils.mylogger import set_logger
    set_logger()
    config_dir = r'/home/makov/tmp/tomo_root/Raw/bones/senin/M2_F4/original'
    build_universal_config(config_dir)