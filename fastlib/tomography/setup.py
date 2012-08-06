__author__ = 'makov'
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = 'sart_cython',
    ext_modules=[
        Extension('sart_cython', ['sart_cython.pyx'],)
    ],
    cmdclass = {'build_ext': build_ext}
)
