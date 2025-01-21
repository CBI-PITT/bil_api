# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:29:42 2022

@author: awatson
"""

from collections.abc import MutableMapping
from threading import Lock

from zarr._storage.absstore import ABSStore  # noqa: F401

from zarr._storage.store import Store, BaseStore
from typing import Union
Path = Union[str, bytes, None]
StoreLike = Union[BaseStore, MutableMapping]
from logger_tools import logger

class disk_cache_store(Store):
    """
    Storage class that implements a DiskCache layer over some other store. 
    Intended primarily for use with stores that may be on slow spinning 
    storage to be cached to fast flash storage.

    Parameters
    ----------
    store : Store
        The store containing the actual data to be cached.
    max_size : int
        The maximum size that the cache may grow to, in number of bytes. 
        Provide `None` if you would like the cache to have unlimited size.

    Examples
    --------
    """

    def __init__(self, store: StoreLike, uuid: str=None, diskcache_object=None, persist=None, meta_data_expire_min=15):
        """
        Designating an uuid will allow the cache to persist accross instances
        """
        self._store: BaseStore = BaseStore._ensure_store(store)
        self._diskcache_object = diskcache_object
        # self._current_size = 0
        self._meta_data_expire = meta_data_expire_min * 60
        self._mutex = Lock()
        self._metadata_keys = ['.zarray','.zgroup','.zattrs','.zmetadata']
        if uuid is None:
            self.persist = False
            import uuid
            self.uuid = str(uuid.uuid4())
        else:
            self.persist = True if persist is None else persist
            self.uuid = uuid
        
        self._cache_key_prefix = self.uuid + '_'
        
        logger.info(self._store)
        logger.info(self._diskcache_object)
        logger.info(self._meta_data_expire)
        logger.info(self._metadata_keys)
        logger.info(self.uuid)
        logger.info(self._cache_key_prefix)
        

    # def __getstate__(self):
    #     return (self._store, self._max_size, self._current_size, self._keys_cache,
    #             self._contains_cache, self._listdir_cache, self._values_cache, self.hits,
    #             self.misses)

    # def __setstate__(self, state):
    #     (self._store, self._max_size, self._current_size, self._keys_cache,
    #      self._contains_cache, self._listdir_cache, self._values_cache, self.hits,
    #      self.misses) = state
    #     self._mutex = Lock()

    def __len__(self):
        # print('__len__')
        return len(self._keys())

    def __iter__(self):
        # print('__iter__')
        return self.keys()

    def __contains__(self, key):
        # print('__contains__')
        # print(key)
        cache_key = self.cache_key('CONTAINS_'+ key)
        if cache_key not in self._diskcache_object:
            self._diskcache_object.set(cache_key, self._store.__contains__(key), expire=self._meta_data_expire, tag=self.uuid + '_CONTAINS')
        return self._diskcache_object.get(cache_key)

    # def clear(self):
    #     self._store.clear()
    #     self.invalidate()

    def keys(self):
        # print('keys')
        with self._mutex:
            return iter(self._keys())

    def _keys(self):
        # print('_keys')
        cache_key = self.cache_key('KEYS')
        if cache_key not in self._diskcache_object:
            # print('Setting Keys from Cache')
            self._diskcache_object.set(cache_key, list(self._store.keys()),tag=self.uuid)
        return self._diskcache_object.get(cache_key)

    # def listdir(self, path: Path = None):
    #     with self._mutex:
    #         try:
    #             return self._listdir_cache[path]
    #         except KeyError:
    #             listing = listdir(self._store, path)
    #             self._listdir_cache[path] = listing
    #             return listing

    def cache_key(self,key):
        # print('cache_key')
        return self._cache_key_prefix+key

    def __getitem__(self, key):
        # print('__getitem__')
        cache_key = self.cache_key(key)
        try:
            # first try to obtain the value from the cache
            value = self._diskcache_object[cache_key]
            # print('GOT FROM CACHE: {}'.format(cache_key))
            # treat the end as most recently used
            if key in self._metadata_keys:
                self._diskcache_object.touch(cache_key, expire=self._meta_data_expire)

        except KeyError:
            # cache miss, retrieve value from the store
            value = self._store[key]
            if key in self._metadata_keys:
                # Metadata expires after awhile
                self._diskcache_object.add(cache_key, value, tag=self.uuid)
                self._diskcache_object.touch(cache_key, expire=self._meta_data_expire)
            else:
                # Chunks do not expire
                self._diskcache_object.add(cache_key, value, tag=self.uuid)
            # print('SET TO CACHE: {}'.format(cache_key))

        return value

    
    def _invalidate(self):
        # print('_invalidate')
        self._invalidate_keys()
        self._invalidate_listdir()
        self._invalidate_contains()
        
    def _invalidate_keys(self):
        cache_key = self.cache_key('KEYS')
        if cache_key in self._diskcache_object:
            del self._diskcache_object[cache_key]
    
    def _invalidate_listdir(self):
        cache_key = self.cache_key('LISTDIR')
        if cache_key in self._diskcache_object:
            del self._diskcache_object[cache_key]
    
    def _invalidate_contains(self):
        self._diskcache_object.evict(self.uuid + '_CONTAINS')
        
        
    def __setitem__(self, key, value):
        # print('__setitem__')
        # print(key)
        cache_key = self.cache_key(key)
        self._store[key] = value
        if key in self._metadata_keys:
            self._diskcache_object.set(cache_key, value, tag=self.uuid, expire=self._meta_data_expire)
        else:
            self._diskcache_object.set(cache_key, value, tag=self.uuid)
        
        self._invalidate()

    def __delitem__(self, key):
        del self._store[key]
        del self._diskcache_object[self.cache_key(key)]
        self._invalidate()
    
    def __del__(self):
        if not self.persist:
            # print('Cleaning up cache for uuid: {}'.format(self.uuid))
            self._invalidate()
            self._diskcache_object.evict(self.uuid)
    
