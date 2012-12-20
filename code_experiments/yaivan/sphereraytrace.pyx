#encoding: utf-8
#cython: profile=False
import numpy as np
import cython
cimport numpy as np
cimport cython

ctypedef np.float32_t dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
def spherraytrace(dtype_t Rcr,dtype_t D,dtype_t Ls,dtype_t Ld,dtype_t px):
    ## parameters initialization
    # primary parameters
    # Rcr=25;
    # D=6;
    # Ls=50;
    # Ld=1;
    # Xdet=(0.5);
    # rec_pix_num=1000;

    #spherraytrace(Rcr,D,Ls,Ld,px,rec_pix_num)
    #Rcr - радиус кривизны зеркала (25 см)
    #D - диаметр зеркала (6 см)
    #Ls - расстояние от зеркала до источника (50 см)
    #Ld - расстояние от зеркала до детектора (2 см)
    #px - координата в сигнале вдоль пятна (центр сигнала соответствует 0 см)
    #rec_pix_num - разрешение реконструкции


    cdef Py_ssize_t i,pn1,pn2
    cdef np.ndarray[dtype_t, ndim=1, mode="c"] trY, Rs, Rd, Npl
    cdef dtype_t pix_size,nop, step, y,a,b,c,x,x_,ost,m1,m2
    cdef dtype_t rec_pix_num

    #все длины в сантиметрах
    sa=D/(2*Rcr) #sin(alpha)
    ca=np.sqrt(1.0-sa*sa) #cos(beta)
    pix_size=0.0013
    px*=pix_size
    rec_pix_num=D/pix_size

    cdef np.ndarray[dtype_t, ndim=2, mode="c"] rayfield=np.zeros((rec_pix_num,rec_pix_num),dtype='float32')
    Rs=np.array([0, -Ls*ca-0.5*D, -Rcr*ca+Ls*sa],dtype='float32')
    Rd=np.array([px, Ld*ca+0.5*D, -Rcr*ca+Ld*sa],dtype='float32')
    Npl=np.cross(Rs,Rd)

    #trace drawing
    nop=rec_pix_num #number of points in the arc
    step=D/(nop)
    trY=np.arange(-0.5*D,0.5*D,step,dtype='float32')

    for i in np.arange(nop,dtype='int32'):
        y=trY[i]
        #equation coefficients
        a=Npl[0]*Npl[0]+Npl[2]*Npl[2]
        b=y*Npl[0]*Npl[1]
        c=y*y*(Npl[1]*Npl[1]+Npl[2]*Npl[2])-(Rcr*Npl[2])*(Rcr*Npl[2])
        x=(-b+np.sign(px)*np.sqrt(b*b-a*c))/a
        x_=x+0.5*D
        pn1=int(np.floor(x_/step)) # важно убрать +1 из-за разности нидексирования питона и матлаба
        ost=x_/step-pn1
        m1=np.abs(0.5-ost)
        m2=1-m1
        if 1<pn1<rec_pix_num-1:
            if ost<0.5:
                pn2=pn1-1
                rayfield[pn2,i]=m1
                rayfield[pn1,i]=m2
            elif ost>0.5:
                pn2=pn1+1
                rayfield[pn1,i]=m2
                rayfield[pn2,i]=m1
            else:
                rayfield[pn1,i]=1
    return rayfield