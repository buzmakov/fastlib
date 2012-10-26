# coding=utf-8
"""
This file should provide function build_universal_config(config_dir) to build  config from dir.
"""
import os
from tomofrontend.configutils.universalconfig import UniversalConfigBuilder
import tomofrontend.configutils.yamlutils as yamlutils
import logging

__author__ = 'makov'


def get_frame(config_dir, log_entry, frames_folder, data_type):
    tmp_frame = {}
    tmp_frame['type'] = data_type
    tmp_frame['image_format'] = 'raw'
    tmp_frame['name'] = os.path.join(os.path.realpath(config_dir), frames_folder, log_entry['file_name'])
    tmp_frame['angle'] = float(log_entry['angle']) / 480.0
    tmp_frame['exposure_time'] = log_entry['exposure']
    tmp_frame['beam_current'] = log_entry['current']
    return tmp_frame


def get_all_frames(config_dir):
    tomo_log = yamlutils.read_yaml(os.path.join(config_dir, 'log.txt'))[0]
    frames = []

    data_log_entries = tomo_log['---Data']
    frames_folder = 'Data'
    for data_log_entry in data_log_entries:
        tmp_frame = get_frame(config_dir, data_log_entry, frames_folder, 'data')
        frames.append(tmp_frame)

    empty_log_entries = tomo_log['---Empty']
    frames_folder = 'Background'
    for empty_log_entry in empty_log_entries:
        tmp_frame = get_frame(config_dir, empty_log_entry, frames_folder, 'empty')
        frames.append(tmp_frame)

    return frames


def build_universal_config(config_dir):
    """
    Build uni_config.yaml in the config_dir

    :param config_dir: directory where  exp_description.yaml
    :raise:
    """
    config_yaml_filename = os.path.join(config_dir, 'exp_description.yaml')
    tomo_config = yamlutils.read_yaml(config_yaml_filename)[0]

    if not 'frames' in tomo_config:
        logging.error(str.format('Section FRAMES not found in config file {0}', config_yaml_filename))
        raise AttributeError
    
    uc = UniversalConfigBuilder()
    #TODO: fix this hack
    uc.add_section({'preprocess_config': tomo_config['preprocess_config']})
    uc.add_section({'description': tomo_config['description']})
    
    files = get_all_frames(config_dir)
    for frame in files:
        uc.add_frame(group_name='0', frame_type=frame['type'], file_name=frame['name'],
                     image_format=frame['image_format'], angle=frame['angle'], exposure_time=frame['exposure_time'],
                     rotation_angle=0)
    #add dark type by value
    if 'dark_value' in tomo_config['frames']:
        uc.add_frame(group_name='0', frame_type='dark', file_name=tomo_config['frames']['dark_value'],
                     image_format='number', angle=0, exposure_time=1, rotation_angle=0)
    uc.save2yaml(os.path.join(config_dir, 'uni_config.yaml'))

if __name__ == "__main__":
    from utils.mylogger import set_logger
    set_logger()
    config_dir = r'/home/makov/tmp/tomo_root/Raw/bones/senin/M2_F4/original'
    build_universal_config(config_dir)