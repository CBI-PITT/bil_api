# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:05:34 2021

@author: alpha
"""


import numpy as np
import io,ast,os

import imaris_ims_file_reader as ims
from bil_api.dataset_info import dataset_info
from bil_api import zarrLoader
# from bil_api import config
from numcodecs import Blosc

from diskcache import FanoutCache



def compress_np(nparr):
    """
    Receives a numpy array,
    Returns a compressed bytestring, uncompressed and the compressed byte size.
    """
    
    comp = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)
    bytestream = io.BytesIO()
    np.save(bytestream, nparr)
    uncompressed = bytestream.getvalue()
    compressed = comp.encode(uncompressed)
    return compressed, len(uncompressed), len(compressed)


def uncompress_np(bytestring):
    """
    Receives a compressed bytestring,
    Returns a numpy array.
    """
    
    comp = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE)
    array = comp.decode(bytestring)
    array = io.BytesIO(array)
    
    # sequeeze = False
    # if "NAPARI_ASYNC" in os.environ or "NAPARI_OCTREE" in os.environ:
    #     sequeeze = False
    # if sequeeze == True and (os.environ["NAPARI_ASYNC"] == '1' or os.environ["NAPARI_OCTREE"] == "1"):
    #     return np.squeeze(np.load(array))
    # else:
    return np.load(array)


def convertMetaDataDict(meta):
    
    '''
    json serialized dict can not have tuple as key.
    This assumes that any key value that 'literal_eval's to tuple 
    will be converted.  Tuples are used to designate (r,t,c) information.
    
    Example: a key for the shape of an array at resolution level 2, 
    timepoint 3, channel 4 = (2,3,4,'shape')
    '''
    
    newMeta = {}
    for idx in meta:
        
        try:
            if isinstance(ast.literal_eval(idx),tuple):
                newMeta[ast.literal_eval(idx)] = meta[idx]
            else:
                newMeta[idx] = meta[idx]
        
        except ValueError:
            newMeta[idx] = meta[idx]
    
    return newMeta


class config:
    '''
    This class will be used to manage open datasets and persistant cache
    '''
    def __init__(self, 
                 cacheLocation=None, cacheSizeGB=100, 
                 evictionPolicy='least-recently-used', timeout=0.100
                 ):
        '''
        evictionPolicy Options:
            "least-recently-stored" #R only
            "least-recently-used"  #R/W (maybe a performace hit but probably best cache option)
        '''
        self.opendata = {}
        self.cacheLocation = cacheLocation
        self.cacheSizeGB = cacheSizeGB
        self.evictionPolicy = evictionPolicy
        self.timeout = timeout
        
        self.cacheSizeBytes = self.cacheSizeGB * (1024**3)
        
        if self.cacheLocation is not None:
            # Init cache
            self.cache = FanoutCache(self.cacheLocation,shards=16)
            # self.cache = FanoutCache(self.cacheLocation,shards=16,timeout=self.timeout, size_limit = self.cacheSizeBytes)
            ## Consider removing this and always leaving open to improve performance
            self.cache.close()
        else:
            self.cache = None

    
    def loadDataset(self, selection: int):
    
        dataPath = dataset_info()[selection][1]
        print(dataPath)
        
        if dataPath in self.opendata:
            return dataPath
        
        if os.path.splitext(dataPath)[-1] == '.ims':
            print('Is IMS')
            
            print('Creating ims object')
            self.opendata[dataPath] = ims.ims(dataPath)
            
            if self.opendata[dataPath].hf is None or self.opendata[dataPath].dataset is None:
                print('opening ims object')
                self.opendata[dataPath].open()
        
        
        elif os.path.splitext(dataPath)[-1] == '.zarr':
            print('Is Zarr')
            print('Creating zarrSeries object')
            self.opendata[dataPath] = zarrLoader.zarrSeries(dataPath)
            
        return dataPath
        
    
    
def prettyPrintDict(aDict):
    print('{}{}{}'.format('Number'.ljust(10),'Name'.ljust(20),'File'))
    for k,v in aDict.items():
        print('{}{}{}'.format(k.ljust(10),v[0].ljust(20),v[1]))
    
    
    
    
    
    
    
    
    
    
    