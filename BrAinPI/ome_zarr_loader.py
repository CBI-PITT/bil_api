# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:12:11 2021

@author: alpha
"""
import io

import zarr, os, itertools
import numpy as np

# # Import zarr stores
from zarr.storage import NestedDirectoryStore
# from zarr_stores.archived_nested_store import Archived_Nested_Store
# from zarr_stores.h5_nested_store import H5_Nested_Store

from collections.abc import MutableMapping
from zarr._storage.store import Store, BaseStore
from typing import Union
Path = Union[str, bytes, None]
StoreLike = Union[BaseStore, Store, MutableMapping]
from logger_tools import logger
# import s3fs

class ome_zarr_loader:
    """
    A loader class for handling OME-Zarr datasets with multi-resolution access and metadata extraction.
    """
    def __init__(self, location, ResolutionLevelLock=None, zarr_store_type: StoreLike=NestedDirectoryStore, verbose=None, squeeze=True, cache=None):
        """
        Initialize the ome_zarr_loader object.

        Args:
            location (str): Path to the OME-Zarr dataset.
            ResolutionLevelLock (int, optional): Lock for accessing a specific resolution level. Defaults to None.
            zarr_store_type (zarr.storage, optional): Zarr store type. Defaults to NestedDirectoryStore.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to None.
            squeeze (bool, optional): Whether to remove singleton dimensions from arrays. Defaults to True.
            cache (object, optional): Cache object for storing slices. Defaults to None.
        """
        # assert StoreLike is s3fs.S3Map or any([issubclass(zarr_store_type,x) for x in StoreLike.__args__]), 'zarr_store_type is not a zarr storage class'

        self.location = location
        self.s3 = False
        # if 's3://' in location:
        #     self.s3 = s3fs.S3FileSystem(anon=True)
        self.ResolutionLevelLock = 0 if ResolutionLevelLock is None else ResolutionLevelLock

        self.verbose = verbose
        self.squeeze = squeeze
        self.cache = cache
        self.metaData = {}

        # Open zarr store
        self.zarr_store = zarr_store_type # Only relevant for non-s3 datasets
        store = self.zarr_store_type(self.location) # opens the store based on whether data are on s3 or local
        zgroup = zarr.open(store)
        self.zattrs = zgroup.attrs
        
        if 'omero' in self.zattrs:
            self.omero = zgroup.attrs['omero']
        # assert 'omero' in self.zattrs
        # self.omero = zgroup.attrs['omero']
        assert 'multiscales' in self.zattrs
        self.multiscales = zgroup.attrs['multiscales']
        logger.info(self.multiscales)
        del zgroup
        del store
        
        try:
            self.multiscale_datasets = self.multiscales[0]['datasets']
        except:
            self.multiscale_datasets = self.multiscales['datasets']
        self.ResolutionLevels = len(self.multiscale_datasets)
        
        self.dataset_paths = []
        self.dataset_scales = []
        for res in range(self.ResolutionLevels):
            self.dataset_paths.append(self.multiscale_datasets[res]['path'])
            self.dataset_scales.append(self.multiscale_datasets[res]['coordinateTransformations'][0]['scale'])
        
        
        for r in range(self.ResolutionLevels):
            array = self.open_array(r)
            if r == 0:
                self.TimePoints = array.shape[0]
                self.Channels = array.shape[1]
                
            for t,c in itertools.product(range(self.TimePoints),range(self.Channels)):
                
                # Collect attribute info
                self.metaData[r,t,c,'shape'] = (t+1,c+1,*array.shape[2:])
                ## Need to extract resolution by some other means.  For now, default to 1,1,1 and divide by 2 for each series
                self.metaData[r,t,c,'resolution'] = self.dataset_scales[r][2:]
                         
                # Collect dataset info
                self.metaData[r,t,c,'chunks'] = array.chunks
                self.metaData[r,t,c,'dtype'] = array.dtype
                self.metaData[r,t,c,'ndim'] = array.ndim
                
                try:
                    self.metaData[r,t,c,'max'] = self.omero['channels'][c]['window']['end']
                    self.metaData[r,t,c,'min'] = self.omero['channels'][c]['window']['start']
                except:
                    pass
        
        self.change_resolution_lock(self.ResolutionLevelLock)
        
        self.arrays = {}
        for res in range(self.ResolutionLevels):
            self.arrays[res] = self.open_array(res)

    def zarr_store_type(self, path):
        """
        Return the appropriate Zarr store for the dataset.

        Args:
            path (str): Path to the dataset.

        Returns:
            zarr.storage: The Zarr store object.
        """
        if self.s3:
            pass
            # return s3fs.S3Map(path, s3=self.s3)
        else:
            return self.zarr_store(path)


    def change_resolution_lock(self,ResolutionLevelLock):
        """
        Update the resolution lock and associated metadata.

        Args:
            ResolutionLevelLock (int): The resolution level to lock.
        """
        self.ResolutionLevelLock = ResolutionLevelLock
        self.shape = self.metaData[self.ResolutionLevelLock,0,0,'shape']
        self.ndim = len(self.shape)
        self.chunks = self.metaData[self.ResolutionLevelLock,0,0,'chunks']
        self.resolution = self.metaData[self.ResolutionLevelLock,0,0,'resolution']
        self.dtype = self.metaData[self.ResolutionLevelLock,0,0,'dtype']
    


    def __getitem__(self,key):
        """
        Access a specific slice of the OME-Zarr dataset.

        Args:
            key (int, slice, or tuple): Index or slice specifying the data to access.

        Returns:
            np.ndarray: The requested data slice, optionally squeezed.
        """
        res = 0 if self.ResolutionLevelLock is None else self.ResolutionLevelLock
        logger.info(key)
        if isinstance(key,slice) == False and isinstance(key,int) == False and len(key) == 6:
            res = key[0]
            if res >= self.ResolutionLevels:
                raise ValueError('Layer is larger than the number of ResolutionLevels')
            key = tuple([x for x in key[1::]])
        logger.info(res)
        logger.info(key)
        
        if isinstance(key, int):
            key = [slice(key,key+1)]
            for _ in range(self.ndim-1):
                key.append(slice(None))
            key = tuple(key)
            
        if isinstance(key,tuple):
            key = [slice(x,x+1) if isinstance(x,int) else x for x in key]
            while len(key) < self.ndim:
                key.append(slice(None))
            key = tuple(key)
        
        logger.info(key)
        newKey = []
        for ss in key:
            if ss.start is None and isinstance(ss.stop,int):
                newKey.append(slice(ss.stop,ss.stop+1,ss.step))
            else:
                newKey.append(ss)
                
        key = tuple(newKey)
        logger.info(key)
        
        
        array = self.getSlice(
                        r=res,
                        t = key[0],
                        c = key[1],
                        z = key[2],
                        y = key[3],
                        x = key[4]
                        )
        
        if self.squeeze:
            return np.squeeze(array)
        else:
            return array
        
    
    def _get_memorize_cache(self, name=None, typed=False, expire=None, tag=None, ignore=()):
        if tag is None: tag = self.location
        return self.cache.memorize(
            name=name,
            typed=typed,
            expire=expire,
            tag=tag,
            ignore=ignore
            ) if self.cache is not None else lambda x: x
    
    def getSlice(self,r,t,c,z,y,x):
        """
        Retrieve a 3D chunk of data for the specified coordinates.

        Args:
            r (int): Resolution level.
            t (slice): Time dimension slice.
            c (slice): Channel dimension slice.
            z (slice): Z-axis slice.
            y (slice): Y-axis slice.
            x (slice): X-axis slice.

        Returns:
            np.ndarray: The requested 3D chunk of data.
        """
        incomingSlices = (r,t,c,z,y,x)
        logger.info(incomingSlices)
        if self.cache is not None:
            key = f'{self.location}_getSlice_{str(incomingSlices)}'
            # key = self.location + '_getSlice_' + str(incomingSlices)
            result = self.cache.get(key, default=None, retry=True)
            if result is not None:
                logger.info(f'Returned from cache: {incomingSlices}')
                return result
        
        result = self.arrays[r][t,c,z,y,x]

        if self.cache is not None:
            self.cache.set(key, result, expire=None, tag=self.location, retry=True)
            # test = True
            # while test:
            #     # logger.info('Caching slice')
            #     self.cache.set(key, result, expire=None, tag=self.location, retry=True)
            #     if result == self.getSlice(*incomingSlices):
            #         test = False

        
        return result
        # return self.open_array(r)[t,c,z,y,x]
    
    
    def locationGenerator(self,res):
        """
        Generate the file path for a specific resolution level.

        Args:
            res (int): The resolution level.

        Returns:
            str: The file path corresponding to the resolution level.
        """
        return os.path.join(self.location,self.dataset_paths[res])
    
    def open_array(self,res):
        """
        Open the Zarr array for the specified resolution level.

        Args:
            res (int): The resolution level.

        Returns:
            zarr.core.Array: The Zarr array for the resolution level.
        """
        store = self.zarr_store_type(self.locationGenerator(res))
        logger.info('OPENING ARRAYS')
        store = self.wrap_store_in_chunk_cache(store)
        # if self.cache is not None:
        #     logger.info('OPENING CHUNK CACHE ARRAYS')
        #     from zarr_stores.zarr_disk_cache import Disk_Cache_Store
        #     store = Disk_Cache_Store(store, unique_id=store.path, diskcache_object=self.cache, persist=False)
        # # try:
        # #     if self.cache is not None:
        # #         store = disk_cache_store(store=store, uuid=self.locationGenerator(res), diskcache_object=self.cache, persist=None, meta_data_expire_min=15)
        # # except Exception as e:
        # #     logger.info('Caught Exception')
        # #     logger.info(e)
        # #     pass
        return zarr.open(store)


    def wrap_store_in_chunk_cache(self, store):
        """
        Wrap the Zarr store with a chunk cache for efficient access.

        Args:
            store (zarr.storage): The Zarr store to wrap.

        Returns:
            zarr.storage: The wrapped Zarr store with chunk caching.
        """
        if self.cache is not None:
            logger.info('OPENING CHUNK CACHE ARRAYS')
            logger.info(store.path)
            from zarr_chunk_cache import disk_cache_store as Disk_Cache_Store
            store = Disk_Cache_Store(store, uuid=store.path, diskcache_object=self.cache, persist=True)
        return store




# class ome_zarr_loader:
#     def __init__(self, location, ResolutionLevelLock=None, zarr_store_type=H5_Shard_Store, verbose=None, squeeze=True):
        
#         self.location = location
#         self.ResolutionLevelLock = 0 if ResolutionLevelLock is None else ResolutionLevelLock
#         self.zarr_store_type = zarr_store_type
#         self.verbose = verbose
#         self.squeeze = squeeze
#         self.metaData = {}
        
#         store = self.zarr_store_type(self.location)
#         zgroup = zarr.open(store)
#         self.zattrs = zgroup.attrs
        
#         assert 'omero' in self.zattrs
#         self.omero = zgroup.attrs['omero']
#         assert 'multiscales' in self.zattrs
#         self.multiscales = zgroup.attrs['multiscales']
#         del zgroup
#         del store
        
#         self.multiscale_datasets = self.multiscales[0]['datasets']
#         self.ResolutionLevels = len(self.multiscale_datasets)
        
#         self.dataset_paths = []
#         self.dataset_scales = []
#         for res in range(self.ResolutionLevels):
#             self.dataset_paths.append(self.multiscale_datasets[res]['path'])
#             self.dataset_scales.append(self.multiscale_datasets[res]['coordinateTransformations'][0]['scale'])
        
        
#         for r in range(self.ResolutionLevels):
            
#             array = self.open_array(r)
            
#             if r == 0:
#                 self.TimePoints = array.shape[0]
#                 self.Channels = array.shape[1]
            
#             # Collect attribute info
#             self.metaData[r,'shape'] = array.shape
#             ## Need to extract resolution by some other means.  For now, default to 1,1,1 and divide by 2 for each series
#             self.metaData[r,'resolution'] = self.dataset_scales[r]
                     
#             # Collect dataset info
#             self.metaData[r,'chunks'] = array.chunks
#             self.metaData[r,'dtype'] = array.dtype
#             self.metaData[r,'ndim'] = array.ndim
        
#         self.change_resolution_lock(self.ResolutionLevelLock)
    

#     def change_resolution_lock(self,ResolutionLevelLock):
#         self.ResolutionLevelLock = ResolutionLevelLock
#         self.shape = self.metaData[self.ResolutionLevelLock,'shape']
#         self.ndim = len(self.shape)
#         self.chunks = self.metaData[self.ResolutionLevelLock,'chunks']
#         self.resolution = self.metaData[self.ResolutionLevelLock,'resolution']
#         self.dtype = self.metaData[self.ResolutionLevelLock,'dtype']
    
    
#     def __getitem__(self,key):
        
#         res = 0 if self.ResolutionLevelLock is None else self.ResolutionLevelLock
#         logger.info(key)
#         if isinstance(key,slice) == False and isinstance(key,int) == False and len(key) == 6:
#             res = key[0]
#             if res >= self.ResolutionLevels:
#                 raise ValueError('Layer is larger than the number of ResolutionLevels')
#             key = tuple([x for x in key[1::]])
#         logger.info(res)
#         logger.info(key)
        
#         if isinstance(key, int):
#             key = [slice(key,key+1)]
#             for _ in range(self.ndim-1):
#                 key.append(slice(None))
#             key = tuple(key)
            
#         if isinstance(key,tuple):
#             key = [slice(x,x+1) if isinstance(x,int) else x for x in key]
#             while len(key) < self.ndim:
#                 key.append(slice(None))
#             key = tuple(key)
        
#         logger.info(key)
#         newKey = []
#         for ss in key:
#             if ss.start is None and isinstance(ss.stop,int):
#                 newKey.append(slice(ss.stop,ss.stop+1,ss.step))
#             else:
#                 newKey.append(ss)
                
#         key = tuple(newKey)
#         logger.info(key)
        
        
#         return self.getSlice(
#                         r=res,
#                         t = key[0],
#                         c = key[1],
#                         z = key[2],
#                         y = key[3],
#                         x = key[4]
#                         )
        


#     def getSlice(self,r,t,c,z,y,x):
        
#         '''
#         IMS stores 3D datasets ONLY with Resolution, Time, and Color as 'directory'
#         structure witing HDF5.  Thus, data access can only happen accross dims XYZ
#         for a specific RTC.  
#         '''
        
#         incomingSlices = (r,t,c,z,y,x)
#         logger.info(incomingSlices)
        
#         array = self.open_array(r)[t,c,z,y,x]
#         if self.squeeze:
#             return np.squeeze(array)
#         else:
#             return array
    
    
#     def locationGenerator(self,res):
#         return os.path.join(self.location,self.dataset_paths[res])
    
#     def open_array(self,res):
#         store = self.zarr_store_type(self.locationGenerator(res))
#         return zarr.open(store)
    
    
    
    
    
######################################################################
# Attempt boto3-based s3:// store (READ-ONLY)
# Why? s3fs does not play well with async gunicorn workers
# Want to enable reading s3:// ome.zarr
######################################################################

# -*- coding: utf-8 -*-
# """
# Created on Tue Jul 19 10:29:42 2022
#
# @author: awatson
# """
#
'''
A Zarr store that uses boto3 (and not s3fs) to access zarr stores in s3://
'''

import os
import errno
import shutil
import time
import numpy as np
import uuid
import glob
import re

from zarr.errors import (
    MetadataError,
    BadCompressorError,
    ContainsArrayError,
    ContainsGroupError,
    FSPathExistNotDir,
    ReadOnlyError,
)

from numcodecs.abc import Codec
from numcodecs.compat import (
    ensure_bytes,
    ensure_text,
    ensure_contiguous_ndarray,
    ensure_contiguous_ndarray_like
)

# from numcodecs.registry import codec_registry

# from threading import Lock, RLock
# from filelock import Timeout, FileLock, SoftFileLock

from zarr.util import (buffer_size, json_loads, nolock, normalize_chunks,
                       normalize_dimension_separator,
                       normalize_dtype, normalize_fill_value, normalize_order,
                       normalize_shape, normalize_storage_path, retry_call)

from zarr._storage.absstore import ABSStore  # noqa: F401

from zarr._storage.store import Store, array_meta_key

_prog_number = re.compile(r'^\d+$')

## BOTO3 Way to do dir and files from s3
import boto3
from botocore import UNSIGNED, exceptions
from botocore.client import Config
import functools

####################################
# HELPER FUNCTIONS
# Duplicated from utils
# may integrate into store class
####################################


def s3_get_bucket_and_path_parts(path):
    path = s3_clean_path(path)
    path_split = path.split('/')
    # logger.info(path_split)
    if isinstance(path_split, str):
        path_split = [path_split]
    bucket = path_split[0]
    return bucket, path_split
def s3_clean_path(path):
    if 's3://' in path.lower():
        path = path[5:]
    elif path.startswith('/'):
        path = path[1:]
    if path.endswith('/'):
        path = path[:-1]
    return path


def list_all_contents(path):
    parent, dirs, files = get_dir_contents(path)
    dirs = [os.path.join(parent,x) for x in dirs]
    files = [os.path.join(parent, x) for x in files]
    return dirs + files

    # if 's3://' in path:
    #     return s3.glob(os.path.join(path,'*'))
    # else:
    #     return glob.glob(os.path.join(path,'*'))

def isdir(path):
    # if 's3://' in path:
    #     return s3.isdir(path)
    if 's3://' in path:
        return s3_isdir(path)
    else:
        return os.path.isdir(path)

def isfile(path):
    if 's3://' in path:
        return s3_isfile(path)
    else:
        return os.path.isfile(path)

def get_dir_contents(path,skip_s3=False):
    if 's3://' in path:
        if skip_s3:
            return path, [], []
        parent, dirs, files, _, _ = s3_get_dir_contents(path)
        return f's3://{parent}', dirs, files
    else:
        for parent, dirs, files in os.walk(path):
            return parent, dirs, files

