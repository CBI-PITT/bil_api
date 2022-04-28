# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:46:38 2021

@author: alpha
"""

import os, glob, zarr, time
import numpy as np
import dask
from dask.delayed import delayed
import dask.array as da
from skimage import io
# from skimage.filters import gaussian
from numcodecs import Blosc
from io import BytesIO
from distributed import Client
import natsort
from itertools import product
from skimage import img_as_uint, img_as_float32
import itertools
import json

'''
Working to make resolution level 0 conversion of jp2 stack to
zarr-like dataset with zip store

Working for interpretation class
'''
### WORKS !!!

def build():
    client = Client()
    
    location = r"/CBI_Hive/globus/pitt/bil"
    # location = r"Z:\zarr"
    out_location = r"/CBI_Hive/globus/pitt/bil/fmost.z_sharded"
    os.makedirs(out_location,exist_ok=True)
    fileType = 'tif'
    imageGeometry = (50,1,1) #(z,y,x)
    origionalChunkSize = (8,256,256) #(z,y,x)
    sim_jobs = 5
    
    
    # https://github.com/Blosc/python-blosc
    # 'zstd:BITSHUFFLE:5' = 20GB
    # 'zstd:BITSHUFFLE:8' = 19.9GB
    compressor = Blosc(cname='zstd', clevel=8, shuffle=Blosc.BITSHUFFLE)
    
    filesList = []
    for ii in sorted(glob.glob(os.path.join(location,'CH1'))):
        print(ii)
        filesList.append(sorted(glob.glob(os.path.join(ii,'*.{}'.format(fileType)))))
    
    numColors = len(filesList)
    
    
    # def read_buffer(fileName):
    #     with open(fileName,'rb') as fh:
    #         print('Reading {}'.format(fileName))
    #         buf = BytesIO(fh.read())
    #         print('Decoding Image {}'.format(fileName))
    #         return io.imread(buf)
    
    def read_buffer(fileName):
        with open(fileName,'rb') as fh:
            print('Reading {}'.format(fileName))
            return BytesIO(fh.read())
            # print('Decoding Image {}'.format(fileName))
            # return io.imread(buf)
    
    def decode_image(buf):
            return io.imread(buf)
    
    
    print('Finding Image Files')
    testImage = read_buffer(filesList[0][0])
    testImage = decode_image(testImage)
    
    
    
    imageStack = []
    for ii in filesList:
        
        stack = [delayed(read_buffer)(x) for x in ii]
        stack = [delayed(decode_image)(x) for x in stack]
        stack = [da.from_delayed(x, shape=testImage.shape, dtype=testImage.dtype) for x in stack]
        stack = da.stack(stack)
        imageStack.append(stack)
    
    imageStack = da.stack(imageStack)
    
    ## Add time dimension as place holder (t,c,z,y,x):
    imageStack = imageStack[None,:]
    
    
    
    def imagePyramidNum(imageStack, origionalChunkSize):
        '''
        Map of pyramids accross a single 3D color
        '''
        
        pyramidMap = {0:[imageStack.shape,origionalChunkSize]}
        out = imageStack.shape
        minimumChunkSize = origionalChunkSize
        topPyramidLevel = 0
        print(out)
        
        
        while True:
            if any([x<=y for x,y in zip(out,minimumChunkSize)]) == False:
                out = tuple([x//2 for x in out])
                topPyramidLevel += 1
                pyramidMap[topPyramidLevel] = [out,minimumChunkSize]
                
            else:
                minimumChunkSize = (minimumChunkSize[0]*4,minimumChunkSize[1]//2,minimumChunkSize[2]//2)
                break
            print(out)
        
        while True:
            if any([x<=y for x,y in zip(out,minimumChunkSize)]) == False:
                out = tuple([x//2 for x in out])
                topPyramidLevel += 1
                pyramidMap[topPyramidLevel] = [out,minimumChunkSize]
                # out = tuple([x//2 for x in out])
            else:
                minimumChunkSize = (minimumChunkSize[0]*4,minimumChunkSize[1]//2,minimumChunkSize[2]//2)
                break
            print(out)
        
        while True:
            if any([x<=y for x,y in zip(out,minimumChunkSize)]) == False:
                out = tuple([x//2 for x in out])
                topPyramidLevel += 1
                pyramidMap[topPyramidLevel] = [out,minimumChunkSize]
                # out = tuple([x//2 for x in out])
            else:
                minimumChunkSize = (minimumChunkSize[0]*4,minimumChunkSize[1]//2,minimumChunkSize[2]//2)
                break
            print(out)
            
        return pyramidMap
    
    
    # pyramidMap = imagePyramidNum(imageStack[0,0,0], origionalChunkSize) # put in last 3 dims
    pyramidMap = imagePyramidNum(imageStack[0,0], origionalChunkSize) # put in last 3 dims
    print(pyramidMap)
    
    
    '''
    Storage structure:
        resolution INT
            timepoint INT
                color INT
                    zarr_zip_store with file name=z_shard_start INT.  Contents YX Shards
    '''
    
    ## Create zarr zipstore objects
    # each zarr store will be 1 Z_chunk deep with YZ shards
    # z_chunk stores will be named for the starting voxel location chunks of (8,256,256) (z,y,x)
    # 0,8,16...
    
    def store_location_formatter(res,t,c,z):
        return '{}/{}/{}/{}.zip'.format(res,t,c,z)
    
    ## Need to make this conform to ome-ngff
    # https://ngff.openmicroscopy.org/latest/
    def write_z_sharded_array_meta(location, pyramidMap, imageStack, resolution=(1,1,50,1,1), store='zip',axes='tczyx'):
        metadata = {}
        
        metadata['shape'] = imageStack.shape
        metadata['axes'] = axes
        
        metadata['resolution'] = {}
        metadata['resolution']['sampling'] = resolution #(t,c,z,y,x)
        metadata['resolution']['units'] = ('s','c','um','um','um')
        
        metadata['series'] = {}
        for key in pyramidMap:
            metadata['series'][key] = {}
            metadata['series'][key]['chunks'] = pyramidMap[key][1]
            metadata['series'][key]['store'] = store
            metadata['series'][key]['shape'] = pyramidMap[key][0]
            metadata['series'][key]['dtype'] = str(imageStack.dtype)
        
        with open(os.path.join(location,'.z_sharded_array'), 'w') as f:
            json.dump(metadata, f, indent=1)
        
        return metadata
    
    # Write empty zarr zip stores for all z_shards
    # zarrObjs = {} # Store all zarrObjects for easy write access
    for t in range(imageStack.shape[0]):
        for c in range(imageStack.shape[1]):
            for key in pyramidMap:
                # currentShape = pyramidMap[key][0]
                
                for z_shards in range(0,pyramidMap[key][0][0],pyramidMap[key][1][0]):
                    print(z_shards)
                    
                    # make location
                    location = os.path.join(out_location,store_location_formatter(key,t,c,z_shards))
                    os.makedirs(os.path.split(location)[0],exist_ok=True)
                    
                    z_shape = pyramidMap[key][1][0] \
                        if (z_shards + pyramidMap[key][1][0]) <= pyramidMap[key][0][0] \
                            else pyramidMap[key][0][0] % pyramidMap[key][1][0]
                    
                    with zarr.ZipStore(location) as store:
                        z = zarr.zeros((z_shape,*pyramidMap[key][0][1:]), chunks=pyramidMap[key][1], store=store, dtype=imageStack.dtype, compressor=compressor, overwrite=True)
    
    
    
    def write_to_zip_store(toWrite,location=None):
        print('In write')
        if toWrite.shape==(0,0,0):
            return True
        with zarr.ZipStore(location) as store:
            print('In with')
            print(toWrite.shape)
            array = zarr.open(store)
            print('Reading {}'.format(location))
            # toWrite = toWrite.compute()
            print('Writing {}'.format(location))
            array[0:toWrite.shape[0],0:toWrite.shape[1],0:toWrite.shape[2]] = toWrite
            print('Completed {}'.format(location))
            return True
    
    
    ## Write first resolution 0 and 1 first
    # zarrObjs = {} # Store all zarrObjects for easy write access
    to_compute = []
    for t in range(imageStack.shape[0]):
        for c in range(imageStack.shape[1]):
            current_stack = imageStack[t,c]
            for key in pyramidMap:
                if key > 0:
                    break
                currentShape = current_stack[::2**key,::2**key,::2**key].shape
                
                for z_shards in range(0,currentShape[0],pyramidMap[key][1][0]):
                    print(z_shards)
                    location = os.path.join(out_location,store_location_formatter(key,t,c,z_shards))
                    
                    toWrite = current_stack[z_shards:z_shards+pyramidMap[key][1][0]]
                    
                    future = delayed(write_to_zip_store)(toWrite,location)
                    # future = toWrite.map_blocks(write_to_zip_store, location=location, dtype=bool)
                    to_compute.append(future)
    
    
    total_jobs = len(to_compute)
    print('Submitting {} of {}'.format(1,total_jobs))
    submit = client.compute(to_compute[0], priority=1)
    to_compute = to_compute[1:]
    submitted = [submit]
    del submit
    
    idx = 2
    while True:
        time.sleep(2)
        if len(to_compute) == 0:
            break
        
        while sum( [x.status == 'pending' for x in submitted] ) >= sim_jobs:
            time.sleep(2)
            
        print('Submitting {} of {}'.format(idx,total_jobs))
        submit = client.compute(to_compute[0], priority=idx)
        to_compute = to_compute[1:]
        submitted.append(submit)
        del submit
        idx += 1
        submitted = [x for x in submitted if x.status != 'finished']
    
    submitted = client.gather(submitted)
    del submitted
    
    
    
    
    
    def build_array_res_level(location,res):
        '''
        Build a dask array representation of a specific resolution level
        Always output a 5-dim array (t,c,z,y,x)
        '''
        
        # Determine the number of TimePoints (int)
        TimePoints = len(glob.glob(os.path.join(location,str(res),'[0-9]')))
        
        # Determine the number of Channels (int)
        Channels = len(glob.glob(os.path.join(location,str(res),'0','[0-9]')))
        
        # Build a dask array from underlying zarr ZipStores
        stack = None
        single_color_stack = None
        multi_color_stack = None
        for t in range(TimePoints):
            for c in range(Channels):
                z_shard_list = natsort.natsorted(glob.glob(os.path.join(location,str(res),str(t),str(c),'*.zip')))
                
                single_color_stack = [da.from_zarr(zarr.ZipStore(file),name=file) for file in z_shard_list]
                single_color_stack = da.concatenate(single_color_stack,axis=0)
                if c == 0:
                    multi_color_stack = single_color_stack[None,None,:]
                else:
                    single_color_stack = single_color_stack[None,None,:]
                    multi_color_stack = da.concatenate([multi_color_stack,single_color_stack], axis=1)
                
            if t == 0:
                stack = multi_color_stack
            else:
                stack = da.concatenate([stack,multi_color_stack], axis=0)
        
        return stack
    
    ## Build z_sharded_zip_store
    to_compute = []
    for key in pyramidMap:
        if key == 0:
            continue
        print('Assembling dask array at resolution level {}'.format(key))
        imageStack = build_array_res_level(out_location,key-1)
        
        to_compute = []
        for t in range(imageStack.shape[0]):
            for c in range(imageStack.shape[1]):
                
                current_stack = imageStack[t,c] #Previous stack to be downsampled
                
                mean_downsampled_stack = []
                # min_shape = current_stack[1::2,1::2,1::2].shape
                min_shape = pyramidMap[key][0]
                for z,y,x in product(range(2),range(2),range(2)):
                    downsampled = current_stack[z::2,y::2,x::2][:min_shape[0],:min_shape[1],:min_shape[2]]
                    downsampled = downsampled.rechunk()
                    mean_downsampled_stack.append(downsampled)
                    del downsampled
                
                mean_downsampled_stack = da.stack(mean_downsampled_stack)
                mean_downsampled_stack = mean_downsampled_stack.map_blocks(img_as_float32, dtype=float)
                mean_downsampled_stack = mean_downsampled_stack.mean(axis=0)
                mean_downsampled_stack = mean_downsampled_stack.map_blocks(img_as_uint, dtype=np.uint16)
                # mean_downsampled_stack = mean_downsampled_stack.rechunk(pyramidMap[key][1])
                    
                
                for z_shards in range(0,pyramidMap[key][0][0],pyramidMap[key][1][0]):
                    print(z_shards)
                    location = os.path.join(out_location,store_location_formatter(key,t,c,z_shards))
                    
                    toWrite = mean_downsampled_stack[z_shards:z_shards+pyramidMap[key][1][0]]
                    
                    future = delayed(write_to_zip_store)(toWrite,location)
                    # future = toWrite.map_blocks(write_to_zip_store, location=location, dtype=bool)
                    # print('Computing Res {}, time {}, channel {}, shard {}'.format(key,t,c,z_shards))
                    # future = client.compute(future)
                    # future = client.gather(future)
                    
                    to_compute.append(future)
                
                
        total_jobs = len(to_compute)
        print('Submitting {} of {}'.format(1,total_jobs))
        submit = client.compute(to_compute[0], priority=1)
        to_compute = to_compute[1:]
        submitted = [submit]
        del submit
    
        idx = 2
        while True:
            time.sleep(2)
            if len(to_compute) == 0:
                break
            
            while sum( [x.status == 'pending' for x in submitted] ) >= sim_jobs:
                time.sleep(2)
            
            print('Submitting {} of {}'.format(idx,total_jobs))
            submit = client.compute(to_compute[0], priority=idx)
            to_compute = to_compute[1:]
            submitted.append(submit)
            del submit
            idx += 1
            submitted = [x for x in submitted if x.status != 'finished']
    
        submitted = client.gather(submitted)
        del submitted
    
    
    client.close()


# if __name__ == "__main__":
#     client = Client()
#     build()
#     client.close()
# else:
#     print('NOPE')
#     exit()



