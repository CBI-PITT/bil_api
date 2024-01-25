# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:38:10 2022

@author: awatson
"""

import numpy as np
import os
import sys
import glob
import time
import math
from skimage import img_as_float32, img_as_uint, img_as_ubyte, img_as_float64
from skimage.filters import gaussian
from natsort import natsorted
from io import BytesIO
from skimage import io
import dask.array as da
from dask.delayed import delayed
import zarr
from distributed import Client
import dask
import tifffile
from itertools import product

'''
45.49245145075851 hours for 2 color fmost
compression_level = 8
storage_chunks = (1,1,8,512,512)
'''

if os.name == 'nt':
    path = r'Z:\cbiPythonTools\bil_api\converters\H5_zarr_store3'
else:
    path = r'/CBI_FastStore/cbiPythonTools/bil_api/converters/H5_zarr_store3'
    
if path not in sys.path:
    sys.path.append(path)

from H5_zarr_store6 import H5Store
from tiff_manager import tiff_manager_3d
# from Z:\cbiPythonTools\bil_api\converters\H5_zarr_store3 import H5Store

if os.name == 'nt':
    jp2_location = r'H:\globus\pitt\bil\fMOST RAW'
else:
    jp2_location = r'/CBI_Hive/globus/pitt/bil/fMOST RAW'

if os.name == 'nt':
    out_location = r'Z:\testData\h5_zarr_test4/scale0'
    # out_location = r'Z:\testData\h5_zarr_test4/scale0'
else:
    out_location = r'/CBI_FastStore/testData/h5_zarr_test4/scale0'
    # out_location = r'/CBI_Hive/globus/pitt/bil/h5_zarr_test4/scale0'

jp2=False
if jp2==True:
    if os.name== 'nt':
        jp2_location = r'H:\globus\pitt\bil\jp2\download.brainimagelibrary.org\8a\d7\8ad742d9c0b886fd\Calb1_GFP_F_F5_200420\level1'
        out_location = r'H:\globus\pitt\bil\jp2\zarr_h5_test'
    else:
        jp2_location = r'/CBI_Hive/globus/pitt/bil/jp2/download.brainimagelibrary.org/8a/d7/8ad742d9c0b886fd/Calb1_GFP_F_F5_200420/level1'
        out_location = r'/CBI_Hive/globus/pitt/bil/jp2/zarr_h5_test'

# if os.name == 'nt':
#     out_location = r'H:\testData\h5_zarr_test3'
# else:
#     out_location = r'/CBI_Hive/testData/h5_zarr_test3'

def dtype_convert(data,dtype):
    
    if dtype == data.dtype:
        return data
    
    if dtype == np.dtype('uint16'):
        return img_as_uint(data)
    
    if dtype == np.dtype('ubyte'):
        return img_as_ubyte(data)
    
    if dtype == np.dtype('float32'):
        return img_as_float32(data)
    
    if dtype == np.dtype(float):
        return img_as_float64(data)
    
    raise TypeError("No Matching dtype : Conversion not possible")

def organize_by_groups(a_list,group_len):

    new = []
    working = []
    idx = 0
    for aa in a_list:
        working.append(aa)
        idx += 1
        
        if idx == group_len:
            new.append(working)
            idx = 0
            working = []
    
    if working != []:
        new.append(working)
    return new

# chunk_limit_MB = 1024
# cpu_number = 32
# storage_chunks = (1,1,4,512,512)
# chunk_depth = (test_image.shape[1]//4) - (test_image.shape[1]//4)%storage_chunks[3]
# z_plane_shape = (30967,20654)

def determine_read_depth(storage_chunks,num_workers,z_plane_shape,chunk_limit_MB=1024,cpu_number=os.cpu_count()):
    chunk_depth = storage_chunks[3]
    current_chunks = (storage_chunks[0],storage_chunks[1],storage_chunks[2],chunk_depth,z_plane_shape[1])
    current_size = math.prod(current_chunks)*2/1024/1024
    
    if current_size >= chunk_limit_MB:
        return chunk_depth
    
    while current_size <= chunk_limit_MB:
        chunk_depth += storage_chunks[3]
        current_chunks = (storage_chunks[0],storage_chunks[1],storage_chunks[2],chunk_depth,z_plane_shape[1])
        current_size = math.prod(current_chunks)*2/1024/1024
        
        if chunk_depth >= z_plane_shape[0]:
            chunk_depth = z_plane_shape[0]
            break
    return chunk_depth
        

sim_jobs = 8
compression_level = 9
storage_chunks = (1,1,4,1024,1024)
chunk_limit_MB = 2048
# read_depth = 512

def test():
    # os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    colors = natsorted(glob.glob(os.path.join(jp2_location,'*')))
    files = []
    for cc in colors:
        if jp2:
            files.append(
                natsorted(glob.glob(os.path.join(cc,'*.jp2')))
                )
        else:
            files.append(
                natsorted(glob.glob(os.path.join(cc,'*.tif')))
                )
    
    # print('Reading Test Image')
    # test_image = tiff_manager(files[0][0],desired_chunk_depth=4096)
    # return test_image
    
    print('Building Virtual Stack')
    stack = []
    for color in files:
        
        s = organize_by_groups(color,storage_chunks[2])
        test_image = tiff_manager_3d(s[0],desired_chunk_depth_y=storage_chunks[2])
        # chunk_depth = (test_image.shape[1]//4) - (test_image.shape[1]//4)%storage_chunks[3]
        chunk_depth = determine_read_depth(storage_chunks,num_workers=sim_jobs,z_plane_shape=test_image.shape[1:],chunk_limit_MB=chunk_limit_MB)
        test_image = tiff_manager_3d(s[0],desired_chunk_depth_y=chunk_depth)
        print(test_image.shape)
        print(test_image.chunks)
        
        s = [test_image.clone_manager_new_file_list(x) for x in s]
        print(len(s))
        for ii in s:
            print(ii.chunks)
            print(len(ii.fileList))
        # print(s[-3].chunks)
        print('From_array')
        print(s[0].dtype)
        s = [da.from_array(x,chunks=x.chunks,name=False,asarray=False) for x in s]
        # print(s)
        print(len(s))
        s = da.concatenate(s)
        # s = da.stack(s)
        print(s)
        stack.append(s)
    stack = da.stack(stack)
    stack = stack[None,...]
    
    # sc =  stack.chunksize
    # stack = stack.rechunk((sc[0],sc[1],storage_chunks[2],sc[3],sc[4]))
    
    print(stack)
    # return stack
    
    # time.sleep(10)
    # return stack
    from numcodecs import Blosc
    compressor=Blosc(cname='zstd', clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
    
    # store = H5Store(r'Z:\testData\test_h5_store2')
    # z = zarr.zeros((1, 2, 11500, 20000, 20000), chunks=(1,1,256,256,256), store=store, overwrite=True, compressor=compressor)
    
    store = H5Store(out_location,verbose=2)
    # z = zarr.zeros(stack.shape, chunks=(1,1,1,1024,1024), store=store, overwrite=True, compressor=compressor)
    # z = zarr.zeros(stack.shape, chunks=stack.chunksize, store=store, overwrite=True, compressor=compressor)
    
    z = zarr.zeros(stack.shape, chunks=storage_chunks, store=store, overwrite=True, compressor=compressor,dtype=stack.dtype)
    
    # Align stack chunks with zarr chunks in z (since this is how the h5 files are stored)
    # Default values are to have y*4 and x*4 chunksize to reduce the number of chunks that dask has to manage by 16 fold (4*4)
    # This improves 1) write efficiency, 2) storage efficiency, 3) avoids any io collisions
    # stack = stack.rechunk((z.chunks[0],z.chunks[1],z.chunks[2],z.chunks[3]*10,z.chunks[4]*10))
    # stack = stack.rechunk((z.chunks[0],z.chunks[1],z.chunks[2]*8,z.chunks[3]*5,z.chunks[4]*5))
    # stack = stack.rechunk((1,1,2,1024,1024))
    
    # with Client('c001.cbiserver:8786') as client:
    # return stack

    if os.name == 'nt':
        with Client(n_workers=1,threads_per_worker=1) as client:
            da.store(stack,z,lock=False)
    else:
        with dask.config.set({'temporary_directory': '/CBI_FastStore/tmp_dask'}):
            
            # with Client(n_workers=sim_jobs,threads_per_worker=os.cpu_count()//sim_jobs) as client:
            # with Client(n_workers=8,threads_per_worker=2) as client:
            with Client(n_workers=16,threads_per_worker=1) as client:
                # print(client.run(lambda: os.environ["HDF5_USE_FILE_LOCKING"]))
                da.store(stack,z,lock=False)
                # da.to_zarr(stack,store)
                # return stored_array
    
    # with Client(n_workers=sim_jobs,threads_per_worker=os.cpu_count()//sim_jobs) as client:
    #     da.store(stack, z,lock=False)

def overlap_helper(start_idx, max_shape, read_len, overlap):
    '''
    For any axis, provide:
        start_idx : start location
        max_shape : length of axis
        read_len : how long should the read be
        overlap : maximum overlap value
    '''
    #determine z_start
    overlap_neg = overlap
    while start_idx-overlap_neg < 0:
        overlap_neg -= 1
        if overlap_neg == 0:
            break
    start = start_idx-overlap_neg
    print(start)
    #determine z_stop
    overlap_pos = 0
    if start_idx+read_len >= max_shape:
        stop = max_shape-1
    else:
        while start_idx+read_len+overlap_pos < max_shape-1:
            overlap_pos += 1
            if overlap_pos == overlap:
                break
        stop = start_idx+read_len+overlap_pos
    print(stop)
    
    # read_loc = [(t,t+1),(c,c+1),(zstart,zstop)]
    # print(read_loc)
    # trim_overlap = [overlap_neg,overlap_pos]
    return ( (start,stop), (overlap_neg,overlap_pos) )


def pad_3d_2x(image,info,final_pad=2):
    
    '''
    Force an array to be padded with mirrored data to a specific size (default 2)
    an 'info' dictionary must be provided with keys: 'z','y','x' where the second
    value is a tuple describing the overlap already present on the array.  The 
    overlap values should be between 0 and the 'final_pad'    
    
    {'z':[(2364,8245),(0,1)]}
    '''
    
    for idx,axis in enumerate(['z','y','x']):
        print(idx)
        overlap_neg, overlap_pos = info[axis][1]
        print(overlap_neg, overlap_pos)
        if overlap_neg < final_pad:
            if idx == 0:
                new = image[0:final_pad-overlap_neg]
                new = new[::-1] # Mirror output
                print(idx,new.shape)
            elif idx == 1:
                new = image[:,0:final_pad-overlap_neg]
                new = new[:,::-1] # Mirror output
                print(idx,new.shape)
            elif idx == 2:
                new = image[:,:,0:final_pad-overlap_neg]
                new = new[:,:,::-1] # Mirror output
            image = np.concatenate((new,image),axis=idx)
            
        if overlap_pos < final_pad:
            if idx == 0:
                new = image[-final_pad+overlap_pos:]
                new = new[::-1] # Mirror output
                print(idx,new.shape)
            elif idx == 1:
                new = image[:,-final_pad+overlap_pos:]
                new = new[:,::-1] # Mirror output
                print(idx,new.shape)
            elif idx == 2:
                new = image[:,:,-final_pad+overlap_pos:]
                new = new[:,:,::-1] # Mirror output
                print(idx,new.shape)
            image = np.concatenate((image,new),axis=idx)
    
    return image


def fast_mean_3d_downsample(from_path,to_path,info,store=H5Store):
    if store == H5Store:
        # zstore = H5Store(from_path, verbose=2)
        zstore = H5Store(from_path)
    else:
        zstore = store(from_path)
    
    zarray = zarr.open(zstore)
    
    dtype = zarray.dtype
    
    print('Reading location {}'.format(info))
    working = zarray[
        info['t'],
        info['c'],
        info['z'][0][0]:info['z'][0][1],
        info['y'][0][0]:info['y'][0][1],
        info['x'][0][0]:info['x'][0][1]
        ]
    
    # print('Read shape {}'.format(working.shape))
    while len(working.shape) > 3:
        working = working[0]
    # print('Axes Removed shape {}'.format(working.shape))
    
    del zarray
    del zstore
    
    working = pad_3d_2x(working,info,final_pad=2)
    working = local_mean_3d_downsample_2x(working)
    
    # #Trim 1 from all sides
    # working = working[
    #     1:-1,
    #     1:-1,
    #     1:-1
    #     ]
    
    # print('Preparing to write')
    if store == H5Store:
        # zstore = H5Store(to_path, verbose=2)
        zstore = H5Store(to_path)
    else:
        zstore = store(to_path)
    
    zarray = zarr.open(zstore)
    
    print('Writing Downsampled shape {}'.format(working.shape))
    
    zarray[
        info['t'],
        info['c'],
        (info['z'][0][0]+info['z'][1][0])//2:(info['z'][0][1]-info['z'][1][1])//2, #Compensate for overlap in new array
        (info['y'][0][0]+info['y'][1][0])//2:(info['y'][0][1]-info['y'][1][1])//2, #Compensate for overlap in new array
        (info['x'][0][0]+info['x'][1][0])//2:(info['x'][0][1]-info['x'][1][1])//2  #Compensate for overlap in new array
        ] = working[1:-1,1:-1,1:-1] # Write array trimmed of 1 value along all edges
    
    return


def local_mean_3d_downsample_2x(image):
    dtype = image.dtype
    print(image.shape)
    image = img_as_float32(image)
    canvas = np.zeros(
        (image.shape[0]//2,
        image.shape[1]//2,
        image.shape[2]//2),
        dtype = np.dtype('float32')
        )
    print(canvas.shape)
    for z,y,x in product(range(2),range(2),range(2)):
        tmp = image[z::2,y::2,x::2][0:canvas.shape[0]-1,0:canvas.shape[1]-1,0:canvas.shape[2]-1]
        print(tmp.shape)
        canvas[0:tmp.shape[0],0:tmp.shape[1],0:tmp.shape[2]] += tmp
        
    canvas /= 8
    return dtype_convert(canvas,dtype)


def local_mean_3d(image):
    print(image.shape)
    image = img_as_float32(image)
    canvas = np.zeros(
        (image.shape[0]+2,image.shape[1]+2,image.shape[2]+2),
        dtype = np.dtype('float32')
        )
    print(canvas.shape)
    # canvas = np.zeros(
    #     image.shape,
    #     dtype = np.dtype('float32')
    #     )
    
    # canvas[:] = image
    for z,y,x in product(range(3),range(3),range(3)):
        # if all([x==0 for x in [z,y,x]]):
        #     pass
        print('{}:{}:{}'.format(z,y,x))
        canvas[
            z:image.shape[0]+z,
            y:image.shape[1]+y,
            x:image.shape[2]+x
            ] += image
    #trim
    canvas = canvas[1:-1,1:-1,1:-1]
    print('dividing')
    canvas /= 12
    print('converting 16bit')
    canvas = img_as_uint(canvas)
    
    return canvas

def local_mean_3d_downsample(from_path,to_path,info,store=H5Store):
    if store == H5Store:
        # zstore = H5Store(from_path, verbose=2)
        zstore = H5Store(from_path)
    else:
        zstore = store(from_path)
    
    zarray = zarr.open(zstore)
    
    dtype = zarray.dtype
    
    print('Reading location {}'.format(info))
    working = zarray[
        info['t'],
        info['c'],
        info['z'][0][0]:info['z'][0][1],
        info['y'][0][0]:info['y'][0][1],
        info['x'][0][0]:info['x'][0][1]
        ]
    
    # print('Read shape {}'.format(working.shape))
    while len(working.shape) > 3:
        working = working[0]
    # print('Axes Removed shape {}'.format(working.shape))
    
    del zarray
    del zstore
    
    working = local_mean_3d(working)
    
    #Trim
    working = working[
        info['z'][1][0]:-info['z'][1][1] if info['z'][1][1] != 0 else None,
        info['y'][1][0]:-info['y'][1][1] if info['y'][1][1] != 0 else None,
        info['x'][1][0]:-info['x'][1][1] if info['x'][1][1] != 0 else None
        ]
    # print('Trimmed to shape {}'.format(working.shape))
    
    #Downsample 2x
    working = working[1::2,1::2,1::2]
    # print('Downsamples to shape {}'.format(working.shape))
    
    # print('Preparing to write')
    if store == H5Store:
        # zstore = H5Store(to_path, verbose=2)
        zstore = H5Store(to_path)
    else:
        zstore = store(to_path)
    
    zarray = zarr.open(zstore)
    
    print('Writing Downsampled shape {}'.format(working.shape))
    
    zarray[
        info['t'],
        info['c'],
        (info['z'][0][0]+info['z'][1][0])//2:(info['z'][0][1]-info['z'][1][1])//2, #Compensate for overlap in new array
        (info['y'][0][0]+info['y'][1][0])//2:(info['y'][0][1]-info['y'][1][1])//2, #Compensate for overlap in new array
        (info['x'][0][0]+info['x'][1][0])//2:(info['x'][0][1]-info['x'][1][1])//2  #Compensate for overlap in new array
        ] = working
    
    return

def smooth_downsample(from_path,to_path,sigma,info,store=H5Store):
    '''
    5D-zarray read full z-planes from zarray at 'loc' (t,c,zstart:zstop), smooth, return
    '''
    
    
    if store == H5Store:
        zstore = H5Store(from_path, verbose=2)
    else:
        zstore = store(from_path)
    
    zarray = zarr.open(zstore)
    
    dtype = zarray.dtype
    
    print('Reading location {}'.format(info))
    working = zarray[
        info['t'],
        info['c'],
        info['z'][0][0]:info['z'][0][1],
        info['y'][0][0]:info['y'][0][1],
        info['x'][0][0]:info['x'][0][1]
        ]
    
    print('Read shape {}'.format(working.shape))
    while len(working.shape) > 3:
        working = working[0]
    print('Axes Removed shape {}'.format(working.shape))
    
    del zarray
    del zstore
    
    #Smooth
    print('Smoothing')
    working = img_as_float32(working)
    working = gaussian(working,sigma)
    working = dtype_convert(working,dtype)
    
    #Trim
    # working = working[
    #     info['z'][1][0]:-info['z'][1][1],
    #     info['y'][1][0]:-info['y'][1][1],
    #     info['x'][1][0]:-info['x'][1][1]
    #     ]
    working = working[
        info['z'][1][0]:-info['z'][1][1] if info['z'][1][1] != 0 else None,
        info['y'][1][0]:-info['y'][1][1] if info['y'][1][1] != 0 else None,
        info['x'][1][0]:-info['x'][1][1] if info['x'][1][1] != 0 else None
        ]
    print('Trimmed to shape {}'.format(working.shape))
    
    #Downsample 2x
    working = working[1::2,1::2,1::2]
    print('Downsamples to shape {}'.format(working.shape))
    
    print('Preparing to write')
    if store == H5Store:
        zstore = H5Store(to_path, verbose=2)
    else:
        zstore = store(to_path)
    
    zarray = zarr.open(zstore)
    
    print('Writing Downsampled shape {}'.format(working.shape))
    # print(
    #     info['t'],
    # info['c'],
    # (info['z'][0][0]//2,info['z'][0][1]//2-1),
    # (info['y'][0][0]//2,info['y'][0][1]//2-1),
    # (info['x'][0][0]//2,info['x'][0][1]//2-1))
    
    zarray[
        info['t'],
        info['c'],
        (info['z'][0][0]+info['z'][1][0])//2:(info['z'][0][1]-info['z'][1][1])//2, #Compensate for overlap in new array
        (info['y'][0][0]+info['y'][1][0])//2:(info['y'][0][1]-info['y'][1][1])//2, #Compensate for overlap in new array
        (info['x'][0][0]+info['x'][1][0])//2:(info['x'][0][1]-info['x'][1][1])//2  #Compensate for overlap in new array
        ] = working
    
    return

def dsample_write(zarray_path, array_to_write, loc, store=H5Store):
    
    if store == H5Store:
        store = H5Store(zarray_path, verbose=2)
    else:
        store = store(zarray_path)
    
    zarray = zarr.open(store)
    
    zarray[
        loc[0][0]:loc[0][1],
        loc[1][0]:loc[1][1],
        loc[2][0]:loc[2][1]
        ] = array_to_write
    

def down_samp(parent_location,res):
    out_location = parent_location[:-1] + '{}'.format(res)
    
    # parent_array = self.open_store(res-1)
    print('Getting Parent Zarr as Dask Array')
    # parent_array = da.from_zarr(self.get_store(res-1))
    parent_array = zarr.open(H5Store(parent_location,verbose=2))
    
    new_array_store = H5Store(out_location,verbose=2)
    
    new_shape = (1, 2, 5732, 15400, 10410)
    print(new_shape)
    # new_chunks = (1, 1, 16, 512, 4096)
    new_chunks = (1, 1, 16, 512, 512)
    print(new_chunks)
    
    from numcodecs import Blosc
    compressor=Blosc(cname='zstd', clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
    
    
    new_array = zarr.zeros(new_shape, chunks=new_chunks, store=new_array_store, overwrite=True, compressor=compressor,dtype=parent_array.dtype)
    print('new_array, {}, {}'.format(new_array.shape,new_array.chunks))
    # z = zarr.zeros(stack.shape, chunks=self.origionalChunkSize, store=store, overwrite=True, compressor=self.compressor,dtype=stack.dtype)

    to_run = []
    z_depth = new_chunks[-3] * 8
    y_depth = new_chunks[-2] * 8
    x_depth = new_chunks[-1] * 8
    print(z_depth)
    for t in range(parent_array.shape[0]):
        for c in range(parent_array.shape[1]):
            
            ## How to deal with overlap?
            
            overlap = 2
            for y in range(0,parent_array.shape[-2],y_depth):
                y_info = overlap_helper(y, parent_array.shape[-2], y_depth, overlap)
                for x in range(0,parent_array.shape[-1],x_depth):
                    x_info = overlap_helper(x, parent_array.shape[-1], x_depth, overlap)
                    for z in range(0,parent_array.shape[-3],z_depth):
                        z_info = overlap_helper(z, parent_array.shape[-3], z_depth, overlap)
                    
                        info = {
                            't':t,
                            'c':c,
                            'z':z_info,
                            'y':y_info,
                            'x':x_info
                            }
                        
                        # working = delayed(smooth_downsample)(parent_location,out_location,1,info,store=H5Store)
                        # working = delayed(local_mean_3d_downsample)(parent_location,out_location,info,store=H5Store)
                        working = delayed(fast_mean_3d_downsample)(parent_location,out_location,info,store=H5Store)
                        print('{},{},{},{},{}'.format(t,c,z,y,x))
                        to_run.append(working)
            
        # with dask.config.set(num_workers=16):
        #     dask.compute(to_run)
        
        future = client.compute(to_run)
        future = client.gather(future)
            
            
            
            # running = []
            # num = len(range(0,working_array.shape[0],working_array.chunksize[0])) #z depth
            # num *= 4
            # for idx in range(len(to_compute)):
            #     print('Computing {}'.format(idx))
            #     running.append(to_compute[idx])
            #     del to_compute[idx]
            #     if idx > 0 and (idx%num == 0 or len(to_compute) == 0):
            #         dask.compute(running)
            #         running = []
                
                
            
            
def write(array_tuple,to_array):
    print('Writing {}'.format(array_tuple[1]))
    to_write = array_tuple[0]
    # to_array[
    #     array_tuple[1][0][0]:array_tuple[1][0][1],
    #     array_tuple[1][1][0]:array_tuple[1][1][1],
    #     array_tuple[1][2][0]:array_tuple[1][2][1],
    #     array_tuple[1][3][0]:array_tuple[1][3][1],
    #     array_tuple[1][4][0]:array_tuple[1][4][1],
    #     ] = to_write[]
    to_array[
        array_tuple[1][0][0],
        array_tuple[1][1][0],
        array_tuple[1][2][0]:array_tuple[1][2][1],
        array_tuple[1][3][0]:array_tuple[1][3][1],
        array_tuple[1][4][0]:array_tuple[1][4][1],
        ] = to_write
    # to_array[loc] = image
    return

# def write(image,to_array,loc):
#     print('Writing {}'.format(loc))
#     to_array[
#         loc[0][0]:loc[0][1],
#         loc[1][0]:loc[1][1],
#         loc[2][0]:loc[2][1],
#         loc[3][0]:loc[3][1],
#         loc[4][0]:loc[4][1]
#         ] = image
#     # to_array[loc] = image
#     return

if __name__ == '__main__':
    print('Running')
    start = time.time()
    # test()
    # if os.name == 'nt':
    #     with Client(n_workers=1,threads_per_worker=1) as client:
    #         pass
    # else:
    #     with dask.config.set({'temporary_directory': '/CBI_FastStore/tmp_dask'}):
            
    #         # with Client(n_workers=sim_jobs,threads_per_worker=os.cpu_count()//sim_jobs) as client:
    #         # with Client(n_workers=8,threads_per_worker=2) as client:
    #         with Client(n_workers=16,threads_per_worker=1) as client:

    #             down_samp(out_location,1)
    # with Client(n_workers=sim_jobs,threads_per_worker=os.cpu_count()//sim_jobs) as client:
    with Client(n_workers=8,threads_per_worker=4) as client:
        with dask.config.set({'temporary_directory': '/CBI_FastStore/tmp_dask'}):
            down_samp(out_location,1)
    total = time.time() - start
    total_hours = total /60/60
    print('Zarr conversion took {} hours'.format(total_hours))
    
    
    
    
# start = time.time()
# local_mean_3d_downsample_2x(np.zeros((512,512,512),dtype=np.dtype('uint16')))
# print(time.time()-start)
    
# start = time.time()
# local_mean_3d(np.zeros((512,512,512),dtype=np.dtype('uint16')))
# print(time.time()-start)

# start = time.time()
# working = img_as_float32(np.zeros((512,512,512),dtype=np.dtype('uint16')))
# working = gaussian(working,1)
# img_as_uint(working)
# print(time.time()-start)



