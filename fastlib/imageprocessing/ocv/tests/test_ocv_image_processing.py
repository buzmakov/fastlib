__author__ = 'makov'
import numpy
import pylab
import fastlib.imageprocessing.reference_implementation as ref
import fastlib.imageprocessing.ocv as ocv
from fastlib.utils.phantom import modified_shepp_logan
from fastlib.utils.mssim import MSSIM

def is_equal(ref_x,y,k=0.01,eps=1.e-6):
    return (numpy.absolute(ref_x-y)<=k*numpy.absolute(ref_x)+eps).all()

def test_project():
    N=50
    x=modified_shepp_logan((N,N,3))[:,:,1]
    x=numpy.array(x)
    rp=ref.project(x)
    op=ocv.project(x)
    assert is_equal(rp,op)

def test_backproject():
    N=50
    x=modified_shepp_logan((N,N,3))[:,int(N/2),1]
    x=numpy.array(x)
    rbp=ref.back_project(x)
    obp=ocv.back_project(x)
    assert is_equal(rbp,obp)

def test_rotate_square():
    N=501
    x=modified_shepp_logan((N,N,3))[:,:,1]
    x=numpy.array(x)
    for angle in [0.0, 10.0,45.0,90.0,150.0,180.0,210.0]:
        r_rot=ref.rotate_square(x,angle)
        o_rot=ocv.rotate_square(x,angle)
        mssim=MSSIM(r_rot,o_rot,16)
        print angle,mssim.min()
        if not (numpy.min(mssim)>0.90).all():
            pylab.figure()
            pylab.subplot(121)
            pylab.imshow(mssim)
            pylab.colorbar()
            pylab.subplot(122)
            pylab.imshow(r_rot-o_rot)
            pylab.colorbar()
            pylab.show()
            assert False