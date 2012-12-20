# cython: profile=False
import numpy as np
import cython
cimport numpy as np
cimport cython

ctypedef np.float32_t dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_square_image_cython(np.ndarray[dtype_t, ndim=2, mode="c"] data, dtype_t angle):
    cdef Py_ssize_t x,y
    cdef dtype_t alpha, center_res, sa, ca, sx, sy, deltax, deltay,r00,r01,r10,r11
    cdef np.ndarray[dtype_t, ndim=1, mode="c"] xmiddle, xmiddle_c, xmiddle_s,ymiddle, ymiddle_c, ymiddle_s
    cdef np.ndarray[dtype_t, ndim=2, mode="c"] res
    cdef np.int32_t im_size, isx, isy

    alpha = np.pi*angle/180.0
    sa=np.sin(alpha)
    ca=np.cos(alpha)
    res=np.zeros_like(data)
    im_size=res.shape[0]
    center_res=float(im_size/2)

    xmiddle = np.arange(im_size,dtype='float32')-center_res
    xmiddle_c=xmiddle*ca + center_res
    xmiddle_s=xmiddle*sa - center_res
    ymiddle = np.arange(im_size,dtype='float32')-center_res
    ymiddle_c=ymiddle*ca
    ymiddle_s=ymiddle*sa
    with nogil:
        for x in range(im_size):
            for y in range(im_size):
                sx =  xmiddle_c[x]+ymiddle_s[y]
                sy = -xmiddle_s[x]+ymiddle_c[y]

                isx = int(sx)
                isy = int(sy)
                if(((0 < isx < im_size-2) and (0 < isy < im_size-2))):
                    deltax= sx-isx
                    deltay= sy-isy
                    if deltax>0:
                        if deltay>0:
                            r00=data[isx,isy]
                            r01=data[isx,isy+1]
                            r10=data[isx+1,isy]
                            r11=data[isx+1,isy+1]
                        else:
                            r01=data[isx,isy-1]
                            r00=data[isx,isy]
                            r11=data[isx+1,isy-1]
                            r10=data[isx+1,isy]
                    else:
                        if deltay>0:
                            r10=data[isx-1,isy]
                            r11=data[isx-1,isy+1]
                            r00=data[isx,isy]
                            r01=data[isx,isy+1]
                        else:
                            r11=data[isx-1,isy-1]
                            r10=data[isx-1,isy]
                            r01=data[isx,isy-1]
                            r00=data[isx,isy]

                    res[x,y] = deltax*(r10*(1.0-deltay)+r01*deltay)+(1.0-deltax)*(r00*(1.0-deltay)+r11*deltay)
                else:
                    res[x,y]=0.0
    return res