import os
import pprint

__author__ = 'makov'
import logging
import yamlutils


class UniversalConfigBuilder:
    def __init__(self):
        self.frames_info = {}
        self.custom_fields = {}

    def add_frame(self, group_name, frame_type, file_name, image_format, angle, **kwargs):
        """

        :param group_name:
        :param frame_type: must be in ('data','dark','empty')
        :param file_name:
        :param image_format: ('fit','tiff' or 'raw')
        :param angle: in degrees
        :param kwargs: additional parameter ('beam_current')
        """
        if not group_name in self.frames_info:
            self.frames_info[group_name] = []
        frame = {'frame_type': frame_type, 'angle': angle,
                 'file_name': file_name, 'image_format': image_format}

        for key in kwargs:
            frame[key] = kwargs[key]
        try:
            if not os.path.exists(frame['file_name']):
                logging.warning(str.format('File {0} not found during building universal config', frame['file_name']))
        except TypeError:
            logging.warning(str.format('File {0} - possible number by value', frame['file_name']))

        if not frame['frame_type'] in ('data', 'dark', 'empty'):
            logging.warning(str.format('frame type unknown: {0}', frame['frame_type']))
        self.frames_info[group_name].append(frame)

#        pprint.pprint(self.frames_info)

    def add_section(self, custom_dict):
        if type(custom_dict) == dict:
            self.custom_fields.update(custom_dict)
        else:
            raise TypeError

    def save2yaml(self, filename):
        docs = {'frames': self.frames_info}
        docs.update(self.custom_fields)
        yamlutils.save_yaml(docs, filename)
