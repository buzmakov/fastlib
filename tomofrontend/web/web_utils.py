import UserDict
import sys
import os
import multiprocessing
import Queue
import datetime
import uuid
import glob

root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, root_dir)
from utils.mypprint import pformat
from utils.natsort import natsorted
from tomofrontend.profiles.tomo_reconstructor import do_tomo_reconstruction
from tomofrontend.configutils.yamlutils import read_yaml


__author__ = 'makov'


class TomoContainer():
    def __init__(self):
        self.tomo_objects = {}
        self.to_reconstruction_queue = multiprocessing.Queue()
        self.from_reconstruction_queue = multiprocessing.Queue()
        self.reconstruction_process = multiprocessing.Process(target=TomoContainer.reconstruction_loop,
                                                              args=(self.to_reconstruction_queue, self.from_reconstruction_queue))
        self.reconstruction_process.start()

    def append(self, tomo_object):
        self.tomo_objects[str(tomo_object['id'])] = tomo_object

    def load_tomo_objects(self, tomo_root):
        for dirpath, dirnames, filenames in os.walk(tomo_root):
            if TomoObject.is_tomo_dir(dirpath, dirnames, filenames):
                tmp_dir = dirpath.replace(tomo_root, '', 1)
                if tmp_dir[0] == r'/':
                    tmp_dir = tmp_dir[1:]
                try:
                    to = TomoObject(base_dir=tomo_root, data_dir=tmp_dir)
                except IOError:  # if error in reading config
                    break
                if not to['id'] in self.tomo_objects:
                    to.update_status('Object added to container')
                    self.append(to)
                else:
                    #if tomo_object exist - reload config
                    self.tomo_objects[to['id']].load_tomo_config()

    def add_to_reconstruction_queue(self, id, just_preprocess=False):
        self.update_objects_status()
        current_status = self.tomo_objects[id]['status'][-1]
        if not current_status['message'] in ['Tomography reconstruction started', 'Pushed in reconstruction queue']:
            self.to_reconstruction_queue.put({'id': id, 'path': self.tomo_objects[id].get_full_path(),
                                              'just_preprocess': just_preprocess})
            self.tomo_objects[id].update_status(message='Pushed in reconstruction queue')
        else:
            self.tomo_objects[id].update_status(
                'Rejected to put in reconstruction queue. Already in queue or reconstructing.')

    def update_objects_status(self):
        try:
            while True:
                x = self.from_reconstruction_queue.get_nowait()
                self.tomo_objects[x['id']].update_status(message=x['message'], date=x['datetime'])
        except Queue.Empty:
            pass

    @staticmethod
    def reconstruction_loop(in_queue, out_queue):
        while True:
            item = in_queue.get()
            if item['just_preprocess']:
                start_message = 'Tomography initial images build  started'
            else:
                start_message = 'Tomography reconstruction started'

            if item['just_preprocess']:
                stop_message = 'Tomography initial images build done'
            else:
                stop_message = 'Tomography reconstruction done'

            out_queue.put({'id': item['id'], 'message': start_message,
                           'datetime': str(datetime.datetime.now())})
            do_tomo_reconstruction(item['path'], just_preprocess=item['just_preprocess'])
            out_queue.put({'id': item['id'], 'message': stop_message,
                           'datetime': str(datetime.datetime.now())})


class TomoObject(UserDict.UserDict):
    def __init__(self, base_dir, data_dir, **kwargs):
        UserDict.UserDict.__init__(self, **kwargs)
        self['base_dir'] = base_dir
        self['data_dir'] = data_dir
        self['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, data_dir))
        self.load_tomo_config()
        self['reconstruction_process'] = None
        self['status'] = []

    def get_full_path(self):
        return os.path.join(self['base_dir'], self['data_dir'])

    def load_tomo_config(self):
        tomo_dir = self.get_full_path()
        tomo_config = os.path.join(tomo_dir, 'original', 'exp_description.yaml')
        self['config'] = read_yaml(tomo_config)[0]

    def is_reconstructed(self):
        tomo_rec_dir = os.path.join(self.get_full_path(), 'reconstruction')
        if not os.path.isdir(tomo_rec_dir):
            return False
        if os.path.isfile(os.path.join(tomo_rec_dir, 'result.hdf5')):
            return True

    def get_formated_config(self):
        return pformat(self['config']).decode('utf-8')

    def get_formated_status(self):
        s = '\n'.join([l['message'] + ' at ' + l['datetime'] for l in self['status']])
        return s

    def get_log(self):
        log_file = os.path.join(self.get_full_path(), 'reconstruction', 'tomo.log')
        if not os.path.isfile(log_file):
            return 'Log file not found'
        log_lines = ''
        with open(log_file, 'r') as lf:
            log_lines = ''.join(lf.readlines())
        return log_lines

    def is_reconstructing_now(self):
        p = self['reconstruction_process']
        if p is None:
            return False
        if type(p) == multiprocessing.process.Process:
            if p.is_alive():
                return True
            return False

    def update_status(self, message, date=None):
        if date is None:
            tmp_date = datetime.datetime.now()
        else:
            tmp_date = date
        self['status'].append({'message': message, 'datetime': str(tmp_date)})

    def is_path_in_datapath(self, path):
        real_path = os.path.realpath(os.path.join(self.get_full_path(), path))
        if not real_path.find(self.get_full_path()):
            return True
        else:
            return False

    def get_files_list(self, file_pattern):
        file_list = glob.glob(os.path.join(self.get_full_path(), file_pattern))
        full_path = self.get_full_path()
        file_list = [x[len(full_path) + 1:] for x in file_list if x.find(full_path) == 0 if os.path.isfile(x)]
        return natsorted(file_list)

    @staticmethod
    def is_tomo_dir(dirpath, dirnames, filenames):
        if 'original' in dirnames:
            return True

if __name__ == "__main__":
    pass