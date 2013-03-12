# coding=utf-8
"""
This file used for mass benchmaking different realisations of image processing functions/
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import time
from itertools import groupby

import numpy
import pylab

import fastlib.imageprocessing.ocv as ocv
import fastlib.imageprocessing.opencl as opencl
import fastlib.imageprocessing.ispmd as ispmd
import fastlib.imageprocessing.reference_implementation as ref
import fastlib.utils.phantom as phantom


def bencmark_projection(sizes):
    """
    Benchmark 'project' function

    :param sizes:
    :return:
    """
    bencmark_res = {}
    counts = 100
    functions = {'ocv.project': ocv.project,
                 'ispmd.project': ispmd.project,
                 'ref.project': ref.project,
                 'opencl.project': opencl.project}
    for size in sizes:
        x = phantom.modified_shepp_logan((size, size, 3))[:, :, 1]
        x = numpy.array(x)
        for func_name in functions:
            func = functions[func_name]
            t = time.time()
            for c in range(counts):
                y = func(x)
            btime = (time.time() - t) / counts
            if not func_name in bencmark_res:
                bencmark_res[func_name] = []
            bencmark_res[func_name].append({'size': size, 'time': btime})
    return bencmark_res


def bencmark_backprojection(sizes):
    """
    Benchmark 'backproject' function.

    :param sizes:
    :return:
    """
    bencmark_res = {}
    counts = 10
    functions = {'ocv.back_project': ocv.back_project,
                 'ispmd.back_project': ispmd.back_project,
                 'ref.back_project': ref.back_project}
    for size in sizes:
        x = phantom.modified_shepp_logan((size, size, 3))[:, :, 1]
        x = ref.project(x)
        for func_name in functions:
            func = functions[func_name]
            t = time.time()
            for c in range(counts):
                y = func(x)
            btime = (time.time() - t) / counts
            if not func_name in bencmark_res:
                bencmark_res[func_name] = []
            bencmark_res[func_name].append({'size': size, 'time': btime})
    return bencmark_res


def bencmark_rotation(sizes, angles):
    """
    Benchmark 'rotate' function.

    :param sizes:
    :param angles:
    :return:
    """
    bencmark_res = {}
    counts = 10
    functions = {'ocv.rotate_square': ocv.rotate_square,
                 'ispmd.rotate_square': ispmd.rotate_square,
                 #               'ref.rotate_square':ref.rotate_square
                 }
    for size in sizes:
        x = phantom.modified_shepp_logan((size, size, 3))[:, :, 1]
        x = numpy.array(x)
        for func_name in functions:
            func = functions[func_name]
            for angle in angles:
                t = time.time()
                for c in range(counts):
                    y = func(x, angle)
                btime = (time.time() - t) / counts
                if not func_name in bencmark_res:
                    bencmark_res[func_name] = []
                bencmark_res[func_name].append({'size': size, 'time': btime, 'angle': angle})
    return bencmark_res


def visualize_rotation_bench(res):
    """
    Visulise rotetion benchmark.

    :param res:
    """
    visualize_1d_bench(res, 'Rotation_benchmark')

    pylab.figure()
    for func_name in res:
        data = res[func_name]
        for ks, gs in groupby(data, lambda x: x['size']):
            plot_graph_x = []
            plot_graph_y = []
            for ka, ga in groupby(gs, lambda x: x['angle']):
                plot_graph_x.append(ka)
                plot_graph_y.append(sum([x['time'] for x in ga]))

            pylab.plot(plot_graph_x, plot_graph_y, label=func_name.split('.')[0] + '_' + str(ks))
            pylab.hold(True)
    pylab.hold(False)
    pylab.xlabel('Angle, deg.')
    pylab.ylabel('Time, sec.')
    pylab.legend()


def visualize_1d_bench(res, title):
    """
    Visulize 1-d benchmarks.

    :param res:
    :param title:
    """
    pylab.figure()
    print title
    for func_name in res:
        data = res[func_name]
        plot_graph_x = []
        plot_graph_y = []
        for k, g in groupby(data, lambda x: x['size']):
            gl = list(g)
            plot_graph_x.append(k)
            plot_graph_y.append(sum([x['time'] for x in gl]) / len(gl))
        print func_name.split('.')[0],
        print[str.format('{0:.3}', x) for x in plot_graph_y]
        pylab.plot(plot_graph_x, plot_graph_y, label=func_name.split('.')[0])
        pylab.hold(True)
    pylab.hold(False)
    pylab.xlabel('Image size, px.')
    pylab.ylabel('Time, sec.')
    pylab.legend()
    pylab.title(title)


if __name__ == "__main__":
    sizes = [100, 500, 1000, ]
    res_proj = bencmark_projection(sizes)
    visualize_1d_bench(res_proj, 'Projection_benchmark')

    res_bproj = bencmark_backprojection(sizes)
    visualize_1d_bench(res_bproj, 'Backprojection_benchmark')

    res_rot = bencmark_rotation(sizes, range(0, 360, 10))
    visualize_rotation_bench(res_rot)

#    pylab.show()
