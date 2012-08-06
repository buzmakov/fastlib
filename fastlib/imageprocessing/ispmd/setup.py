from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name = 'ispmd_image_processing',
    ext_modules=[
        Extension('ispmd_image_processing', ['ispmd_image_processing.pyx'],
            extra_objects=["./C/objs/c_image_processing.o",
                           "./C/objs/image_processing_ispc.o","./C/objs/image_processing_ispc_sse2.o",
                           "./C/objs/image_processing_ispc_sse4.o","./C/objs/image_processing_ispc_avx.o",
                           "./C/objs/tasksys.o"],
            libraries=["stdc++",'m'])
    ],
    cmdclass = {'build_ext': build_ext}
)