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

import h5py
import hdf5plugin


'''
Working to make resolution level 0 conversion of jp2 stack to
HDF5 VDS-like dataset

No Interpreter yet
'''
### WORKS !!!

# def build():
# client = Client()

## SETTINGS SPECIFIC TO THE CURRENT ARRAY BEING CREATED ##

if os.name == 'nt':
    location = r"h:/globus/pitt/bil/jp2/download.brainimagelibrary.org/8a/d7/8ad742d9c0b886fd/Calb1_GFP_F_F5_200420/level1"
else:
    location = r"/CBI_Hive/globus/pitt/bil/jp2/download.brainimagelibrary.org/8a/d7/8ad742d9c0b886fd/Calb1_GFP_F_F5_200420/level1"
# location = r"Z:\zarr"
if os.name == 'nt':
    out_location = r"h:/globus/pitt/bil/jp2/h5_shard"
else:
    out_location = r"/CBI_Hive/globus/pitt/bil/jp2/h5_shard"

# if os.name == 'nt':
#     location = r"H:\globus\pitt\bil\TEST"
# else:
#     location = r"/CBI_Hive/globus/pitt/bil/jp2/download.brainimagelibrary.org/8a/d7/8ad742d9c0b886fd/Calb1_GFP_F_F5_200420/level1"
# # location = r"Z:\zarr"
# if os.name == 'nt':
#     out_location = r"H:\globus\pitt\bil\TEST\z_sharded"
# else:
#     out_location = r"/CBI_Hive/globus/pitt/bil/jp2/zarr3"
    
# https://github.com/Blosc/python-blosc
# 'zstd:BITSHUFFLE:5' = 20GB
# 'zstd:BITSHUFFLE:8' = 19.9GB

class z_sharded_builder:
    
    def __init__(
            self,location,out_location,fileType,
            geometry=(50,1,1),origionalChunkSize=(8,256,256),minChunkSize=(8,256,256),
            sim_jobs=10, compressor=hdf5plugin.Blosc(cname='zstd', clevel=8, shuffle=hdf5plugin.Blosc.BITSHUFFLE),
            build_imediately = False
            ):
        
        self.location = location
        self.out_location = out_location
        self.fileType = fileType
        self.geometry = geometry
        self.origionalChunkSize = origionalChunkSize
        self.minChunkSize = minChunkSize
        self.sim_jobs = sim_jobs
        self.compressor = compressor
        self.store_ext = 'h5'
        
        os.makedirs(self.out_location,exist_ok=True)
        
        ##  LIST ALL FILES TO BE CONVERTED  ##
        ## Assume files are laid out as "color_dir/images"
        filesList = []
        for ii in sorted(glob.glob(os.path.join(self.location,'*'))):
            filesList.append(sorted(glob.glob(os.path.join(ii,'*.{}'.format(self.fileType)))))
        
        self.filesList = filesList
        self.Channels = len(self.filesList)
        self.TimePoints = 1
        print(self.Channels)
        print(self.filesList)
        
        
        testImage = self.read_file(self.filesList[0][0])
        self.dtype = testImage.dtype
        self.ndim = testImage.ndim
        self.shape_3d = (len(self.filesList[0]),*testImage.shape)
        
        self.shape = (self.TimePoints, self.Channels, *self.shape_3d)
        
        self.pyramidMap = self.imagePyramidNum()
        
        self.record_z_shards()
        # self.write_empty_z_shards()
        # self.write_resolution_0()
    
    @staticmethod
    def read_file(fileName):
        return io.imread(fileName)
    
    def imagePyramidNum(self):
        '''
        Map of pyramids accross a single 3D color
        '''
        
        pyramidMap = {0:[self.shape_3d,self.origionalChunkSize]}
        out = self.shape_3d
        minimumChunkSize = self.minChunkSize
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
    
    def store_location_formatter(self,res,t,c,z):
        '''
        Storage structure:
            resolution INT
                timepoint INT
                    color INT
                        zarr_zip_store with file name=z_shard_start INT.  Contents YX Shards


        # Create zarr zipstore objects
        each zarr store will be 1 Z_chunk deep with YZ shards
        z_chunk stores will be named for the starting voxel location chunks of (8,256,256) (z,y,x)
        0,8,16...
        '''
        return os.path.join(self.out_location,'{}/{}/{}/{}.{}'.format(res,t,c,z,self.store_ext)).replace('\\','/')
    
    @staticmethod
    def regular_path(path):
        return path.replace('\\','/')
    
    def record_z_shards(self):
        '''
        Form a list of all z_shard files + shape
        store at: self.shards_list
        
        List of dict of dict:
            shard_file_name identifies the main key:
                'file':fileName(str),
                'shape':shape_of_shard(tuple),
                'chunks':chunks_of_shard(tuple)
        '''
        self.shards_dict = {}
        for t in range(self.TimePoints):
            for c in range(self.Channels):
                for key in self.pyramidMap:
                    # currentShape = pyramidMap[key][0]
                    
                    for z_shards in range(0,self.pyramidMap[key][0][0],self.pyramidMap[key][1][0]):
                        print(z_shards)
                        if t+c+key+z_shards == 0:
                            self.shards_list = []
                        # make location
                        location = self.store_location_formatter(key,t,c,z_shards)
                        os.makedirs(os.path.split(location)[0],exist_ok=True)
                        
                        z_shape = self.pyramidMap[key][1][0] \
                            if (z_shards + self.pyramidMap[key][1][0]) <= self.pyramidMap[key][0][0] \
                                else self.pyramidMap[key][0][0] % self.pyramidMap[key][1][0]
                        
                        self.shards_dict[location] ={
                            # 'file':location,
                            'shape':(z_shape,*self.pyramidMap[key][0][1:]),
                            'chunks':(z_shape,*self.pyramidMap[key][1][1:]),
                            'z_start':z_shards,
                            'z_stop':z_shards+z_shape,
                            't':t,
                            'c':c,
                            'res':key
                            }
    
    def write_empty_z_shards(self):
        '''
        All empty shard containers are written to disk
        Info is derived from self.shards_dict
        '''
                        
        for key in self.shards_dict:
            print('Creating {}'.format(key))
            with h5py.File(key,'a') as store:
                # zarr.zeros((z_shape,*self.pyramidMap[key][0][1:]), chunks=self.pyramidMap[key][1], store=store, dtype=self.dtype, compressor=self.compressor, overwrite=True)
                store.create_dataset("data", self.shards_dict[key]['shape'], chunks=self.shards_dict[key]['chunks'],dtype=self.dtype, **self.compressor)
    
    
    def write_resolution_0(self):
        
        z_depth = self.pyramidMap[0][1][0]
        
        to_run = []
        for t in range(self.TimePoints):
            for c in range(self.Channels):
                
                for idx in range(0,self.shape[2],z_depth):
                    stack = []
                    for z in range(idx,idx+z_depth):
                        
                        if z < self.shape[2]:
                            file = self.filesList[c][z]
                            print('Reading {}'.format(file))
                            stack.append( delayed(self.read_file)(file) )
                    stack = delayed(np.stack)(stack,axis=0)
                    # print(stack.shape)
                    # print(stack.dtype)
                    # print(stack.min())
                    # print(stack.max())
                    location = self.store_location_formatter(0,t,c,idx)
                    stack = delayed(self.dump_to_zip)(location,stack)
                    to_run.append(stack)
                    del stack
                    
        with Client() as client:
            running = []
            run = client.compute(to_run[0])
            running.append(run)
            del run
            del to_run[0]
            while len(to_run) > 0:
                
                run = client.compute(to_run[0])
                running.append(run)
                del run
                del to_run[0]
                while len(running) >= self.sim_jobs:
                    print('{} jobs running, {} remaining'.format(len(running),len(to_run)))
                    running = [x for x in running if x.status != 'finished']
                    time.sleep(2)
            
            while len(running) > 0:
                print('{} jobs running, {} remaining'.format(len(running),len(to_run)))
                running = [x for x in running if x.status !='finished']
                time.sleep(2)
            print('Jobs complete')
    
    
    def write_resolution_series(self):
        '''
        Make downsampled versions of dataset based on pyramidMap
        '''
        for ii in self.pyramidMap:
            pass
            
    
    
    def dump_to_zip(self,file,data):
        print('Writing {}'.format(file))
        with h5py.File(file,'a') as store:
            dataset = store["data"]
            dataset[:] = data
        return 'complete'
    
    def list_files(self,res,t,c):
        files = natsort.natsorted(
            glob.glob(
                os.path.join(
                    os.path.split(self.store_location_formatter(res,t,c,0))[0],'*'
                    )
                )
            )
        print(files)
        return [self.regular_path(x) for x in files]
    
    def vds(self, res):
        
        layout = h5py.VirtualLayout(shape=(self.TimePoints,self.Channels,*self.pyramidMap[res][0]), dtype=self.dtype)
        
        for t,c in product(range(self.TimePoints),range(self.Channels)):
            files = self.list_files(res,t,c)
            for ii in files:
                    
                print('Layout file {}'.format(ii))
                vsource = h5py.VirtualSource(ii, 'data', shape=self.shards_dict[ii]['shape'])
                layout[t,c,
                       self.shards_dict[ii]['z_start']:self.shards_dict[ii]['z_stop'],
                       :,:] = vsource
        
        with h5py.File(os.path.join(self.out_location,'VDS_{}.hf'.format(res)), 'w', libver='latest') as f:
            
            print('Creating VDS')
            f.create_virtual_dataset('vdata', layout,fillvalue=None)
        
        with h5py.File(os.path.join(self.out_location,'VDS_{}.hf'.format(res)), 'r') as f:
            print("Virtual dataset:")
            print(f["vdata"].shape)
            print(f["vdata"][0,1,0:10,0:10,0:10])
        
        # self.vds = h5py.File(os.path.join(self.out_location,'VDS_{}.hf'.format(res)), 'r')
        
        return
        
        
    
    
# ## Need to make this conform to ome-ngff
# # https://ngff.openmicroscopy.org/latest/
# def write_z_sharded_array_meta(location, pyramidMap, imageStack, resolution=(1,1,50,1,1), store='zip',axes='tczyx'):
#     metadata = {}
    
#     metadata['shape'] = imageStack.shape
#     metadata['axes'] = axes
    
#     metadata['resolution'] = {}
#     metadata['resolution']['sampling'] = resolution #(t,c,z,y,x)
#     metadata['resolution']['units'] = ('s','c','um','um','um')
    
#     metadata['series'] = {}
#     for key in pyramidMap:
#         metadata['series'][key] = {}
#         metadata['series'][key]['chunks'] = pyramidMap[key][1]
#         metadata['series'][key]['store'] = store
#         metadata['series'][key]['shape'] = pyramidMap[key][0]
#         metadata['series'][key]['dtype'] = str(imageStack.dtype)
    
#     with open(os.path.join(location,'.z_sharded_array'), 'w') as f:
#         json.dump(metadata, f, indent=1)
    
#     return metadata


# def write_to_zip_store(toWrite,location=None):
#     print('In write')
#     if toWrite.shape==(0,0,0):
#         return True
#     with zarr.ZipStore(location) as store:
#         print('In with')
#         print(toWrite.shape)
#         array = zarr.open(store)
#         print('Reading {}'.format(location))
#         # toWrite = toWrite.compute()
#         print('Writing {}'.format(location))
#         array[0:toWrite.shape[0],0:toWrite.shape[1],0:toWrite.shape[2]] = toWrite
#         print('Completed {}'.format(location))
#         return True


# ## Write first resolution 0 and 1 first
# # zarrObjs = {} # Store all zarrObjects for easy write access
# to_compute = []
# for t in range(imageStack.shape[0]):
#     for c in range(imageStack.shape[1]):
#         current_stack = imageStack[t,c]
#         for key in pyramidMap:
#             if key > 0:
#                 break
#             currentShape = current_stack[::2**key,::2**key,::2**key].shape
            
#             for z_shards in range(0,currentShape[0],pyramidMap[key][1][0]):
#                 print(z_shards)
#                 location = os.path.join(out_location,store_location_formatter(key,t,c,z_shards))
                
#                 toWrite = current_stack[z_shards:z_shards+pyramidMap[key][1][0]]
                
#                 future = delayed(write_to_zip_store)(toWrite,location)
#                 # future = toWrite.map_blocks(write_to_zip_store, location=location, dtype=bool)
#                 to_compute.append(future)


# total_jobs = len(to_compute)
# print('Submitting {} of {}'.format(1,total_jobs))
# submit = client.compute(to_compute[0], priority=1)
# to_compute = to_compute[1:]
# submitted = [submit]
# del submit

# idx = 2
# while True:
#     time.sleep(2)
#     if len(to_compute) == 0:
#         break
    
#     while sum( [x.status == 'pending' for x in submitted] ) >= sim_jobs:
#         time.sleep(2)
        
#     print('Submitting {} of {}'.format(idx,total_jobs))
#     submit = client.compute(to_compute[0], priority=idx)
#     to_compute = to_compute[1:]
#     submitted.append(submit)
#     del submit
#     idx += 1
#     submitted = [x for x in submitted if x.status != 'finished']

# submitted = client.gather(submitted)
# del submitted





# def build_array_res_level(location,res):
#     '''
#     Build a dask array representation of a specific resolution level
#     Always output a 5-dim array (t,c,z,y,x)
#     '''
    
#     # Determine the number of TimePoints (int)
#     TimePoints = len(glob.glob(os.path.join(location,str(res),'[0-9]')))
    
#     # Determine the number of Channels (int)
#     Channels = len(glob.glob(os.path.join(location,str(res),'0','[0-9]')))
    
#     # Build a dask array from underlying zarr ZipStores
#     stack = None
#     single_color_stack = None
#     multi_color_stack = None
#     for t in range(TimePoints):
#         for c in range(Channels):
#             z_shard_list = natsort.natsorted(glob.glob(os.path.join(location,str(res),str(t),str(c),'*.zip')))
            
#             single_color_stack = [da.from_zarr(zarr.ZipStore(file),name=file) for file in z_shard_list]
#             single_color_stack = da.concatenate(single_color_stack,axis=0)
#             if c == 0:
#                 multi_color_stack = single_color_stack[None,None,:]
#             else:
#                 single_color_stack = single_color_stack[None,None,:]
#                 multi_color_stack = da.concatenate([multi_color_stack,single_color_stack], axis=1)
            
#         if t == 0:
#             stack = multi_color_stack
#         else:
#             stack = da.concatenate([stack,multi_color_stack], axis=0)
    
#     return stack

# ## Build z_sharded_zip_store
# to_compute = []
# for key in pyramidMap:
#     if key == 0:
#         continue
#     print('Assembling dask array at resolution level {}'.format(key))
#     imageStack = build_array_res_level(out_location,key-1)
    
#     to_compute = []
#     for t in range(imageStack.shape[0]):
#         for c in range(imageStack.shape[1]):
            
#             current_stack = imageStack[t,c] #Previous stack to be downsampled
            
#             mean_downsampled_stack = []
#             # min_shape = current_stack[1::2,1::2,1::2].shape
#             min_shape = pyramidMap[key][0]
#             for z,y,x in product(range(2),range(2),range(2)):
#                 downsampled = current_stack[z::2,y::2,x::2][:min_shape[0],:min_shape[1],:min_shape[2]]
#                 downsampled = downsampled.rechunk()
#                 mean_downsampled_stack.append(downsampled)
#                 del downsampled
            
#             mean_downsampled_stack = da.stack(mean_downsampled_stack)
#             mean_downsampled_stack = mean_downsampled_stack.map_blocks(img_as_float32, dtype=float)
#             mean_downsampled_stack = mean_downsampled_stack.mean(axis=0)
#             mean_downsampled_stack = mean_downsampled_stack.map_blocks(img_as_uint, dtype=np.uint16)
#             # mean_downsampled_stack = mean_downsampled_stack.rechunk(pyramidMap[key][1])
                
            
#             for z_shards in range(0,pyramidMap[key][0][0],pyramidMap[key][1][0]):
#                 print(z_shards)
#                 location = os.path.join(out_location,store_location_formatter(key,t,c,z_shards))
                
#                 toWrite = mean_downsampled_stack[z_shards:z_shards+pyramidMap[key][1][0]]
                
#                 future = delayed(write_to_zip_store)(toWrite,location)
#                 # future = toWrite.map_blocks(write_to_zip_store, location=location, dtype=bool)
#                 # print('Computing Res {}, time {}, channel {}, shard {}'.format(key,t,c,z_shards))
#                 # future = client.compute(future)
#                 # future = client.gather(future)
                
#                 to_compute.append(future)
            
            
#     total_jobs = len(to_compute)
#     print('Submitting {} of {}'.format(1,total_jobs))
#     submit = client.compute(to_compute[0], priority=1)
#     to_compute = to_compute[1:]
#     submitted = [submit]
#     del submit

#     idx = 2
#     while True:
#         time.sleep(2)
#         if len(to_compute) == 0:
#             break
        
#         while sum( [x.status == 'pending' for x in submitted] ) >= sim_jobs:
#             time.sleep(2)
        
#         print('Submitting {} of {}'.format(idx,total_jobs))
#         submit = client.compute(to_compute[0], priority=idx)
#         to_compute = to_compute[1:]
#         submitted.append(submit)
#         del submit
#         idx += 1
#         submitted = [x for x in submitted if x.status != 'finished']

#     submitted = client.gather(submitted)
#     del submitted


# client.close()


# # if __name__ == "__main__":
# #     client = Client()
# #     build()
# #     client.close()
# # else:
# #     print('NOPE')
# #     exit()



