/* good ispc implementation of a sum reduction */
export void sum_raw(uniform float array_in[], uniform float array_out[],uniform int count) {
    float sum = 0;
    foreach (i = 0 ... count)
        sum += array_in[i];
    array_out[0]= reduce_add(sum);
}

//task for project_mt function
task void sum_task(uniform float array_in[], uniform float array_out[],uniform int im_size) {
    uniform int delta_x=(im_size+taskCount-1)/taskCount;
    uniform int x_start= taskIndex*delta_x;
    uniform int x_stop= min(im_size,x_start+delta_x);

    foreach(x = x_start ...x_stop){
        float sum = 0;
        for(uniform int i = 0;i< im_size;i++){
            sum += array_in[x*im_size+i];
            };
        array_out[x]= sum;
    }
}

//multi_thread project function
export void project_mt(uniform float array_in[], uniform float array_out[],uniform int im_size) {
    launch[16] sum_task(array_in, array_out,im_size);
    sync;
}

task void backproject_task(uniform float array_in[], uniform float array_out[],uniform int im_size) {
    uniform int delta_x=(im_size+taskCount-1)/taskCount;
    uniform int x_start= taskIndex*delta_x;
    uniform int x_stop= min(im_size,x_start+delta_x);

    for(uniform int x = x_start;x< x_stop;x++){
        foreach(i = 0 ... im_size){
            array_out[x*im_size+i]=array_in[x];
            };
    }
}

export void backproject(uniform float array_in[], uniform float array_out[],uniform int count){
    foreach (i = 0 ... count, j= 0 ... count){
        array_out[i*count+j]=array_in[i];
    }
}

export void backproject_mt(uniform float array_in[], uniform float array_out[],uniform int count){
    launch[8] backproject_task(array_in, array_out,count);
    sync;
}

//rotate one image (slice)
void rotate_one_image(uniform float data[], uniform float res[],
            uniform float angle, uniform int im_size){
    uniform float alpha = angle/57.29577951308232;
    uniform float sa,ca;
    sincos(alpha,&sa,&ca);
    uniform float * uniform tmp = uniform new uniform float[im_size*2];
    uniform float center_res= (float)im_size/2.0-0.5;
    for(uniform int x = 0; x< im_size;x++){
           uniform float xmiddle= x-center_res;
           tmp[x]= xmiddle*ca+center_res;
           tmp[im_size+x]=xmiddle*sa;
    }
    foreach_tiled(x = 0 ... im_size){
        for(uniform int y = 0; y<im_size;y++){
            float sx=tmp[x]+tmp[y+im_size];
            float sy=-tmp[x+im_size]+tmp[y];
            int isx=floor(sx);
            int isy=floor(sy);
            cif((0 < isx) && (isx < im_size-2) && (0 < isy) && (isy < im_size-2)){
                int offset=isx*im_size+isy;
                float deltax= sx-isx;
                float deltay= sy-isy;
                float r00=data[offset]*(1.0f-deltax)*(1.0f-deltay);
                float r01=data[offset+1]*(1.0f-deltax)*deltay;
                float r10=data[offset+im_size]*deltax*(1.0f-deltay);
                float r11=data[offset+im_size+1]*deltax*deltay;;
                res[x*im_size+y]= r00+r01+r10+r11;
                // if ((r00+r01+r10+r11)>2.0){
                //     print("%\t%\t%\t%\n",r00,r01,r10,r11);
                //     break;
                // }
            }else{
                res[x*im_size+y]=0.0;
            }
        }
    }
    delete[] tmp;
}

//rotate subvlobume (range of slices)
task void rotate_subvolume(uniform float array_in[], uniform float array_out[],
                        uniform float angle, uniform int im_size,  uniform int slices_count){
    uniform int delta_slice=(slices_count+taskCount-1)/taskCount;
    uniform int slice_start= taskIndex*delta_slice;
    uniform int slice_stop= min(slices_count,slice_start+delta_slice);
    uniform int stride=im_size*im_size;

    for(uniform int slice= slice_start;slice < slice_stop;slice++){
        rotate_one_image(&array_in[stride*slice],&array_out[stride*slice],angle,im_size);
    }
}

export void rotate_volume(uniform float array_in[], uniform float array_out[],
                        uniform float angle, uniform int im_size,  uniform int slices_count) {
    launch[16] rotate_subvolume(array_in, array_out, angle, im_size, slices_count);
    sync;
}

//function for rotation part of image (for multithreading)
task void rotate_task(uniform float data[], uniform float res[], uniform float tmp[], uniform int im_size){
    uniform int delta_y=(im_size+taskCount-1)/taskCount;
    uniform int y_start= taskIndex*delta_y;
    uniform int y_stop= min(im_size,y_start+delta_y);
    foreach(x = 0 ... im_size, y= y_start ... y_stop){
        float sx=tmp[x]+tmp[y+im_size];
        float sy=-tmp[x+im_size]+tmp[y];
        int isx=floor(sx);
        int isy=floor(sy);
        cif((0 < isx) && (isx < im_size-2) && (0 < isy) && (isy < im_size-2)){
            int offset=isx*im_size+isy;
            float deltax= sx-isx;
            float deltay= sy-isy;
            float r00=data[offset]*(1.0f-deltax)*(1.0f-deltay);
            float r01=data[offset+1]*(1.0f-deltax)*deltay;
            float r10=data[offset+im_size]*deltax*(1.0f-deltay);
            float r11=data[offset+im_size+1]*deltax*deltay;
            res[x*im_size+y]= r00+r01+r10+r11;
        }else{
            res[x*im_size+y]=0.0;
        }
    }
}

//multithread version of single image rotation
export void rotate_mt(uniform float data[], uniform float res[], uniform float angle, uniform int im_size){
    uniform float alpha = angle/57.29577951308232;
    uniform float sa,ca;
    sincos(alpha,&sa,&ca);
    uniform float * uniform tmp = uniform new uniform float[im_size*2];
    uniform float center_res= (float)im_size/2.0-0.5;
    for(uniform int x = 0; x< im_size;x++){
           uniform float xmiddle= x-center_res;
           tmp[x]= xmiddle*ca+center_res;
           tmp[im_size+x]=xmiddle*sa;
    }
    launch[16] rotate_task(data,res,tmp,im_size);
    sync;
    delete[] tmp;
}

task void add_task(uniform float array_1[], uniform float array_2[],uniform int im_size) {
    uniform int delta_x=(im_size+taskCount-1)/taskCount;
    uniform int x_start= taskIndex*delta_x;
    uniform int x_stop= min(im_size,x_start+delta_x);

    foreach(x = x_start ... x_stop){
        array_1[x]+=array_2[x];
    }
}

export void add_mt(uniform float array_1[], uniform float array_2[],uniform int count){
    launch[16] add_task(array_1,array_2,count);
    sync;
}


task void rotate_subvolume_z(uniform float data[], uniform float res[],
                        uniform float angle, uniform int im_size,  uniform int slices_count) {
    uniform float alpha = angle/57.29577951308232;
    uniform float sa,ca;
    sincos(alpha,&sa,&ca);
    uniform float * uniform tmp = uniform new uniform float[im_size*2];
    uniform float center_res= (float)im_size/2.0-0.5;
    for(uniform int x = 0; x< im_size;x++){
           uniform float xmiddle= x-center_res;
           tmp[x]= xmiddle*ca+center_res;
           tmp[im_size+x]=xmiddle*sa;
    }

    uniform int delta_x=(im_size+taskCount-1)/taskCount;
    uniform int x_start= taskIndex*delta_x;
    uniform int x_stop= min(im_size,x_start+delta_x);
    for(uniform int x = x_start; x < x_stop;x++){
        for(uniform int y = 0;y < im_size;y++){   
            uniform float sx=tmp[x]+tmp[y+im_size];
            uniform float sy=-tmp[x+im_size]+tmp[y];
            uniform int isx=floor(sx);
            uniform int isy=floor(sy);
            if((0 < isx) && (isx < im_size-2) && (0 < isy) && (isy < im_size-2)){
                uniform int offset=(isx*im_size+isy)*slices_count;
                uniform float deltax= sx-isx;
                uniform float deltay= sy-isy;
                foreach(z=0 ... slices_count){
                    float r00=data[offset+z]*(1.0f-deltax)*(1.0f-deltay);
                    float r01=data[offset+slices_count+z]*(1.0f-deltax)*deltay;
                    float r10=data[offset+slices_count*im_size+z]*deltax*(1.0f-deltay);
                    float r11=data[offset+(im_size+1)*slices_count+z]*deltax*deltay;
                    res[(x*im_size+y)*slices_count+z]= r00+r01+r10+r11;
                }
            }else{
                foreach(z=0 ... slices_count){
                    res[(x*im_size+y)*slices_count+z]=0.0;
                }
            }
        }
    }
    delete[] tmp;
}

export void rotate_volume_z(uniform float data[], uniform float res[],
                        uniform float angle, uniform int im_size,  uniform int slices_count){ 
    launch[16] rotate_subvolume_z(data,res,angle,im_size,slices_count);
    sync;
}

task void project_subvolume(uniform float data[], uniform float res[],
                        uniform int im_size, uniform int slices_count) {
    uniform int delta_x=(slices_count+taskCount-1)/taskCount;
    uniform int x_start= taskIndex*delta_x;
    uniform int x_stop= min(slices_count,x_start+delta_x);
    for(uniform int x = x_start; x < x_stop;x++){
        for(uniform int y = 0;y < im_size;y++){
            uniform int offset =(x*im_size+y)*im_size;   
            float sum=0;
            foreach(z=0 ... im_size){
                sum+=data[offset+z];
                }
            res[x*im_size+y]=reduce_add(sum);
        }
    }
}

export void project_volume(uniform float data[], uniform float res[],
                        uniform int im_size, uniform int slices_count){ 
    launch[16] project_subvolume(data,res,im_size,slices_count);
    sync;
}

task void backproject_subvolume(uniform float data[], uniform float res[],
                        uniform int im_size, uniform int slices_count) {
    uniform int delta_x=(slices_count+taskCount-1)/taskCount;
    uniform int x_start= taskIndex*delta_x;
    uniform int x_stop= min(slices_count,x_start+delta_x);
    for(uniform int x = x_start; x < x_stop;x++){
        for(uniform int y = 0;y < im_size;y++){
            uniform int offset =(x*im_size+y)*im_size;   
            float sum=0;
            float r=data[x*im_size+y]/im_size;
            foreach(z=0 ... im_size){
                res[offset+z]+=r;
                }
        }
    }
}

export void backproject_volume(uniform float data[], uniform float res[],
                        uniform int im_size, uniform int slices_count){ 
    launch[16] backproject_subvolume(data,res,im_size,slices_count);
    sync;
}