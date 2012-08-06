import cv2
import os
import numpy
import scipy.ndimage
import h5py
import pylab
import glob

__author__ = 'makov'

if __name__=="__main__":
    data_root=r'/media/WD_ext4/bones/done/'
#    exps={'M2_F4_s':0.0002,}
    exps={'M2_F5_s':0.00015,'M2_S4_s':0.00015,'M2_F4_s':0.0002,'M2_S5_s':0.00015,
          'M3_F4_2_s':0.0003,'M3_S4_2_s':0.0003,'M3_F5_s':0.0003,'M3_F4_s':0.0003,
          'M2_S3_cu':0.00005,
          'M2_F3_cr':2e-4,'M2_F1_cr':2e-4,'M2_S1_cr':2e-4,'M2_S4_cr':2e-4,'M2_S2_cr':2e-4,
          'M3_F2_cu':1.2e-3,'M3_S2_cu':1.2e-3,'M3_F1_cu':1.2e-3}

    for exp in exps:

        min_val=exps[exp]*1
        data_dir=os.path.join(data_root,exp)
        h5file=os.path.join(data_dir,'result.hdf5')
        exp_title = exp

        print exp_title
        if os.path.exists(exp_title+'.png'):
            continue

        if exp_title[-1]=='s':
            pixel_size=0.010
        else:
            pixel_size=0.013

        data=None

        with  h5py.File(h5file,'r') as data_file:
            data=data_file["Results"][:].astype('float32')

        data=numpy.where(numpy.isnan(data), numpy.zeros_like(data),data)
        data*=(data>min_val)
        sh=data.shape
        data_filt=cv2.medianBlur(data,3)
        if len(data_filt.shape)==2:
            data_filt=data

        del data

        if exp_title[-1]=='s':
            data_filt/=10
        vol=numpy.array(data_filt>0,dtype='float32').sum()*(pixel_size**3)
        absorp=data_filt.sum()
        print 'Voulume=',vol
        print 'absobtion_integral=',absorp
#        for k in numpy.arange(0,0.05,0.002):
#            mask=data_filt>k*th
#            mask=numpy.array(mask,dtype='float32')
#            mask=cv2.medianBlur(mask,5)
#            tmp_data_filt=data[mask>0]
##            tmp_data_filt=numpy.where(mask>0,data_filt,numpy.zeros_like(data_filt))
#            x.append(k)
#            v=mask.sum()*(pixel_size**3)
#            d=tmp_data_filt.sum()
#            volume.append(v)
#            dens.append(d)
#            dv.append(d/v)
#            print str.format('{0:4.4} {1:4.4} {2:4.4} {3:4.4}', k,v,d, d/v)

        pylab.figure()
        for i in range(4):
            tmp_slice=data_filt[int((i+1)*sh[0]/6)]
            pylab.subplot(221+i)
            pylab.imshow(tmp_slice,vmin=min(min_val,tmp_slice.max()))
            pylab.colorbar()
        pylab.savefig(exp_title)

#        pylab.figure()
#        pylab.plot(x,volume)
#        pylab.title(exp_title +' volume')
#        pylab.grid()
#        pylab.savefig(exp_title +'_volume')
#    #    pylab.show()
#        pylab.clf()
#
#    #    pylab.figure()
#        pylab.plot(x,dens)
#        pylab.grid()
#        pylab.title(exp_title +' denst')
#        pylab.savefig(exp_title +'_denst')
#        pylab.clf()
#    #    pylab.show()
#
#        pylab.figure()
#        pylab.plot(x,dv)
#        pylab.grid()
#        pylab.title(exp_title +' d/v')
#        pylab.savefig(exp_title +'_d_v')
#        pylab.clf()
#    #    pylab.show()


