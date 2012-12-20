#encoding: utf-8
__author__ = 'makov'
import pyximport
pyximport.install()
import sphereraytrace
import pylab

def get_one_trace():
    rpn=6.0/0.0013
    tr=sphereraytrace.spherraytrace(Rcr=25,D=6,Ls=50,Ld=1,px=1000)
    tr+=sphereraytrace.spherraytrace(Rcr=25,D=6,Ls=50,Ld=1,px=-1000)
    tr+=sphereraytrace.spherraytrace(Rcr=25,D=6,Ls=50,Ld=1,px=0)
    pylab.figure()
    pylab.imshow(tr,interpolation='nearest')
    pylab.colorbar()
    pylab.show()

if __name__=="__main__":
    get_one_trace()