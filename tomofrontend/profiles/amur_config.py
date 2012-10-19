# coding=utf-8
"""
This file should provide function build_universal_config(config_dir) to build  config from dir.
"""
import glob
import os
from tomofrontend.configutils.universalconfig import UniversalConfigBuilder
from utils.natsort import natsorted
import tomofrontend.configutils.yamlutils as yamlutils
from utils.dictutils import get_value, try_get_value
#from utils.mypprint import pprint
import logging

__author__ = 'makov'


def get_groups(config):
    """
    Return dictionary of files groups.

    :param config:
    :return: Dictionary of files groups.
    :raise:
    """
    known_keys = ('exposure_time', 'folder', 'angles_step', 'image_format', 'rotation_angle', 'frames_per_angle')
    if not 'groups' in config:
        logging.error('Section GROUPS not found in config file')
        raise AttributeError
    res = {}
    for group_name in config['groups']:
        tmp_grp = dict(config['groups'][group_name])
        try:
            for key in config:
                if not key in tmp_grp:
                    if key in known_keys:
                        tmp_grp[key] = get_value(config, ['groups', group_name], key)
        except KeyError:
            pass
        res[group_name] = tmp_grp
    return res


def get_files_in_group(group, root='.', start_data_angle=0):
    """
    Get files in each group.
    :param group:
    :param root:
    :param start_data_angle:
    :return:
    """
    files = []
    stop_data_angle = start_data_angle
    for type_key in ['dark', 'data', 'empty']:
        if type_key in group:
            if 'filesnames' in group[type_key]:
                if 'regexp' in group[type_key]['filesnames']:
                    cur_root = root
                    fname_pattern = group[type_key]['filesnames']['regexp']
                    if 'folder' in group:
                        cur_root = os.path.join(cur_root, group['folder'])

                    delta_angle = get_value(group, [type_key, ], 'angles_step')
                    exp_time = get_value(group, [type_key, ], 'exposure_time')
                    rotation_angle = get_value(group, [type_key, ], 'rotation_angle')
                    image_format = try_get_value(group, [type_key, ], 'image_format', 'fits')
                    frames_per_angle = try_get_value(group, [type_key, ], 'frames_per_angle', 1)

                    tmp_files_names_list = glob.glob(os.path.join(cur_root, fname_pattern))
                    tmp_files_names_list = natsorted(tmp_files_names_list)
                    tmp_files_list = []

                    for ef, f in enumerate(tmp_files_names_list):
                        if type_key == 'data':
                            angle = start_data_angle + delta_angle * ((ef - ef % frames_per_angle) / frames_per_angle)
                            stop_data_angle = start_data_angle + delta_angle * (((ef + 1) - (ef + 1) % frames_per_angle) / frames_per_angle)
                        else:
                            angle = 0
                        td = {'name': f, 'angle': angle, 'exposure_time': exp_time, 'type': type_key,
                              'rotation_angle': rotation_angle, 'image_format': image_format}
                        tmp_files_list.append(td)
                    files.extend(tmp_files_list)
    return files, stop_data_angle


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
    groups = get_groups(res['frames'])
    start_data_angle = 0
    for group_name in natsorted(groups.keys()):
        group = groups[group_name]
        files, start_data_angle = get_files_in_group(group, root=config_dir, start_data_angle=start_data_angle)
        for f in files:
            uc.add_frame(group_name=group_name, frame_type=f['type'], file_name=f['name'],
                         image_format=f['image_format'], angle=f['angle'], exposure_time=f['exposure_time'],
                         rotation_angle=f['rotation_angle'])
    uc.save2yaml(os.path.join(config_dir, 'uni_config.yaml'))

if __name__ == "__main__":
    from utils.mylogger import set_logger
    set_logger()
    config_dir = r'/media/WD_ext4/bones/ikran/2012_02_02/F1-M2/'
    build_universal_config(config_dir)
