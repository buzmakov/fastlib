# coding=utf-8
__author__ = 'makov'
import numpy
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
project_prg = cl.Program(ctx, """
        __kernel void project(__global const float *src,
        __global float *res, int const stride)
        {
            int gid = get_global_id(0);
            float t=0;
            for(int i=0; i<stride; i++){
                t+= src[gid*stride+i];
            }
            res[gid]=t;
        }
        """).build()


def project(src):
    """
    Project 2d numpy array.

    :param src:
    :return:
    """
    stride = src.shape[0]
    res = numpy.zeros(stride, dtype='float32')

    mf = cl.mem_flags

    src_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src)
    res_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=res)

    project_prg.project(queue, (stride,), None, src_buf, res_buf, numpy.int32(stride))
    cl.enqueue_copy(queue, res, res_buf).wait()
    return res