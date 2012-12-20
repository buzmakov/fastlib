#encoding: utf-8
import time
import logging
import os
import yaml
import scipy
import scipy.io
import scipy.ndimage
import scipy.sparse
import numpy
import pylab
import ocv_image_processing as myocv

import pyximport
pyximport.install()
import rotate_image
import sphereraytrace

from numba import autojit

def set_logger():
    """
    Start logging in file "tomo.log" and to stdout.
    """
    logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        filename='tomo.log',
                        filemode='w')
    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_logger.setFormatter(formatter)
    logging.getLogger('').addHandler(console_logger)


class Reconstructor(object):
    """
    Reconstructor of tomography data on different surfaces.
    """
    def __init__(self, config_file_dir):
        """
        :param config_file_dir: directory, where  should be config file metadata.yaml
        """
        super(Reconstructor, self).__init__()
        self.config = {"config_dir": config_file_dir}
        self.read_config()
        self.load_data()

    def read_config(self):
        """
        Reading config form metadata.yaml, which located in config_dir

        :raise: IOError if config file not exist
        """
        config_file_name = os.path.join(self.config['config_dir'], "metadata.yaml")
        if not os.path.exists(config_file_name):
            logging.error(str.format("Config file {0} NOT found", config_file_name))
            raise IOError
        with open(config_file_name, 'rb') as config_file:
            yconfig = yaml.load(config_file)
            self.config.update(yconfig)

    def load_data(self):
        """
        Loading data from first matlab file from config (self.config).
        Process sinogram and saving value to self.sinogram.
        Result saved in self.sinogram.
        """
        #TODO: Add multiply files support
        config_file_record = self.config["files"][0]
        data_file_path = os.path.join(self.config["config_dir"], config_file_record["name"])
        logging.debug(str.format("Loading data from file: {0}", data_file_path))
        data_file = scipy.io.loadmat(data_file_path, struct_as_record=True)
        self.raw_sinogram = data_file[config_file_record["var_name"]].astype('float32')
        self.sinogram = self.process_sinogramm()
#        pylab.figure()
#        pylab.imshow(self.sinogram)
#        pylab.axis('auto')
#        pylab.colorbar()
#        pylab.show()

    def process_sinogramm(self):
        """
        Normalize sinograms (self.raw_sinogram), taking log and cast to float 32

        :return: Minus log from sinogrames
        """
        tmp_sinograms = self.raw_sinogram
        if numpy.max(tmp_sinograms) > 1:
            tmp_sinograms -= numpy.min(tmp_sinograms)
            tmp_sinograms /= numpy.max(tmp_sinograms)
        tmp_sinograms = tmp_sinograms * (tmp_sinograms > 0) + 1.0 * (tmp_sinograms <= 0)
        tmp_sinograms = -numpy.log(tmp_sinograms)
        tmp_sinograms *= (tmp_sinograms > 0).astype('float32')
        logging.info("Sinograms processed")
        return tmp_sinograms

    def reconstruct(self, mode='parallel'):
        """
        Reconstruct slice from sinograms

        :param mode: mode for projection.
        """
        rec_config = {'config_dir': self.config['config_dir']}
        rec_config.update(self.config['files'][0])

        if mode == 'parallel':
            rec = ParallelSARTSolver(rec_config, self.sinogram)
        elif mode == 'sphere':
            rec = SphereSARTSolver(rec_config, self.sinogram)
        else:
            logging.error(str.format('Unknown reconstruction mode {0}', mode))
            return

        rec.tomo_it()
        self.solution = rec.solution
        numpy.save(os.path.join(self.config['config_dir'], 'res_' + rec.out_files_prefix + '.npy'), self.solution)


class TomoSolver(object):
    """
    Base class for iTomo solvers.
    """
    def __init__(self, config, sinogram):
        """
        :param config: Dictionary with parameters
        :param sinogram: numpy.array with sinograms
        """
        super(TomoSolver, self).__init__()
        self.config = config
        self.sinogram = sinogram

        self.D = self.config['surface_info']['D']  # Mirror diameter in cm
        self.px_size = self.config['pixel_size']  # pixel sinze in mkm
        self.N = scipy.around(self.D * 1e4 / self.px_size).astype('int32')  # Size of reconstruction arena in px
        self.nx = self.config['length']  # number of channels on sinogram
        self.nang = self.config['angles']  # number of angles of rotations
        self.shift = self.config['shift']  # shift if axis of rotation
        self.projection_start = scipy.around(0.5 * (self.N - self.nx) + self.shift).astype('int32')  # start position of sinogram on reconstruction field
        self.projection_end = self.projection_start + self.nx  # end position of sinogram on reconstruction field
        self.solution = numpy.zeros((self.N, self.N), dtype='float32')  # space for solution


class ParallelSARTSolver(TomoSolver):
    """
    Class for reconstruct iTomo with SART method in case of parallel beam.
    """
    def __init__(self, config, sinogram):
        super(ParallelSARTSolver, self).__init__(config, sinogram)
        self.rotate_image = self.get_rotate()
        self.out_files_prefix = 'parallel'

    def get_rotate(self):
#        import pyximport;pyximport.install()
#        import rotate_image
        # logging.info('Cython rotation installed')
        # return rotate_image.rotate_square_image_cython
        return myocv.cv_rotate

    def direct_project(self, image, angle):
        tmp_image = self.rotate_image(image, angle)
        tmp_image = tmp_image[:, self.projection_start:self.projection_end]
        tmp_projection = scipy.sum(tmp_image, axis=0)
        return tmp_projection

    def back_project(self, projection, angle):
        tmp_solution = numpy.zeros_like(self.solution)
        tmp_solution[:, self.projection_start:self.projection_end] = projection / self.N
        tmp_solution = self.rotate_image(tmp_solution, -angle)
        return tmp_solution

    def save_tmp_solution(self, tmp_image_name):
        tmp_image_dir = os.path.join(self.config['config_dir'], 'tmp_images')
        image_max_size = 1200.0
        if not os.path.exists(tmp_image_dir):
            os.mkdir(tmp_image_dir)
        if self.N > image_max_size:
            pylab.imshow(scipy.ndimage.zoom(self.solution, image_max_size / self.N, order=1))
        else:
            pylab.imshow(self.solution)
        pylab.colorbar()
        pylab.savefig(os.path.join(tmp_image_dir, tmp_image_name))
        pylab.clf()

    def SART(self):
        angles = numpy.arange(0, 180.0 * (self.nang + 1) / self.nang, 180.0 / (self.nang - 1))
        start_time = time.time()
        ls = (0.8, 0.5, 0.2,)
#        ls=(0.8,)
        for il, l in enumerate(ls):
            ang_numbs = numpy.arange(len(angles))
            scipy.random.shuffle(ang_numbs)
            for ia, ang_numb in enumerate(ang_numbs):
                ang = angles[ang_numb]
                logging.info(str.format("Shift= {0}\tLambda={1}\tAngle number={2}\tAngle={3}\ttime={4}",
                                        self.shift, l, ia, ang, time.time() - start_time))

                tmp_projection = self.direct_project(self.solution, ang)
                tmp_projection = self.sinogram[:, ang_numb] - tmp_projection
                self.solution += self.back_project(l * tmp_projection, ang)
                self.solution *= self.solution >= 0
                self.save_tmp_solution(str.format("{0}_debug_{1}_{2}.png", self.out_files_prefix, il, ia))

    def tomo_it(self):
        logging.info('Starting tomographic reconstruction with ParallelSARTSolver')
        self.SART()


class SphereSARTSolver(ParallelSARTSolver):
    """
    Class for reconstruct iTomo with SART method in case of spherical surface.
    """
    def __init__(self, config, sinogram):
        super(SphereSARTSolver, self).__init__(config, sinogram)
        self.out_files_prefix = 'sphere'
        self.build_tmp_rays()

    def get_one_ray(self, px_numb):
        """
        Build trajectory of 1 ray

        :param px_numb: position in projection.
        """
#        import pyximport
#        pyximport.install()
#        import sphereraytrace

        central_px_position = scipy.around(0.5 * self.N)
        arb_px_position = self.projection_start - central_px_position + px_numb
        ray = sphereraytrace.spherraytrace(Rcr=self.config['surface_info']['Rcr'],
                                           D=self.config['surface_info']['D'],
                                           Ls=self.config['surface_info']['Ls'],
                                           Ld=self.config['surface_info']['Ld'],
                                           px=arb_px_position)

        sparse_ray = scipy.sparse.lil_matrix(ray, dtype='float32')
        return sparse_ray

    def build_tmp_rays(self):
        out_file = os.path.join(self.config['config_dir'], 'tmp_rays_' + self.out_files_prefix + '.npy')
        if os.path.exists(out_file):
            logging.debug(str.format('Loading rays from file {0}', out_file))
            self.rays = numpy.load(out_file)
            logging.debug(str.format('Loaded rays from file {0}', out_file))
        else:
            logging.debug(str.format('Saving rays to file {0}', out_file))
            tmp_rays = []
            for i in range(self.nx):
                tmp_rays.append(self.get_one_ray(i))
                logging.debug(str.format('Adding ray {0} from {1} to tmp file.', i, self.nx))
            numpy.save(out_file, tmp_rays)
            self.rays = tmp_rays
            logging.debug(str.format('Saved rays to file {0}', out_file))

    def back_project_python(self, projection, angle):
        tmp_solution = numpy.zeros_like(self.solution)
        for i in range(self.nx):
            tmp_solution += self.rays[i].toarray() * projection[i]
            if not i % 10:
                logging.debug(str.format('Back project {0} from {1}', i, self.nx))
        tmp_solution /= self.N
        tmp_solution = self.rotate_image(tmp_solution, -angle)
        return tmp_solution
#        return super(SphereSARTSolver, self).back_project(projection, angle)

    def direct_project_python(self, image, angle):
        tmp_image = self.rotate_image(image, angle)
        tmp_projection = numpy.zeros(self.nx, dtype='float32')
        for i in range(self.nx):
            tmp_projection[i] = numpy.sum(self.rays[i].toarray() * tmp_image)
            if not i % 10:
                logging.debug(str.format('Direct project python {0} from {1}', i, self.nx))
        return tmp_projection
#        return super(SphereSARTSolver, self).direct_project(image, angle)

    def direct_project_cython(self, image, angle):
        import spherical_projections
        return spherical_projections.direct_project(
            image.astype('float32'), angle, long(self.N), long(self.nx), self.rays, self.rotate_image)

    def back_project_cython(self, projection, angle):
        import spherical_projections
        return spherical_projections.back_project(
            projection.astype('float32'), angle, long(self.N), long(self.nx), self.rays, self.rotate_image)

    def back_project(self, projection, angle):
        return self.back_project_python(projection, angle)

    def direct_project(self, projection, angle):
        return self.direct_project_python(projection, angle)

if __name__ == "__main__":
    set_logger()
    pylab.figure()

    config_file_dir = ('data/mysimogram3_shift')
    tr = Reconstructor(config_file_dir)
    tr.reconstruct(mode='parallel')
    #tr.reconstruct(mode='sphere')

    pylab.close()
