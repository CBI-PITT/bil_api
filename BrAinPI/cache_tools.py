

from diskcache import FanoutCache
import os
import config_tools

settings = config_tools.get_config()
def get_cache():
    """
    Setup cache location based on OS type. Optional situations like machine name can be used to customize.
    Initialize and configure a disk-based cache using `FanoutCache`.The cache settings, such as location, 
    size, eviction policy, and shards, are loaded from the application's configuration file.

    Returns:
        FanoutCache: A configured disk-based cache object, or None if no cache location is specified.
    """
    if os.name == 'nt':
        cacheLocation = settings.get('disk_cache', 'location_win')
    else:
        cacheLocation = settings.get('disk_cache', 'location_unix')
        # cacheLocation = None

    if cacheLocation is not None:
        # Instantiate class that will manage all open datasets
        # This will remain in the global env and be accessed by multiple route methods
        cacheLocation=cacheLocation
        cacheSizeGB=settings.getint('disk_cache', 'cacheSizeGB')
        cacheSizeBytes = cacheSizeGB * (1024 ** 3)
        evictionPolicy=settings.get('disk_cache', 'evictionPolicy')
        shards=settings.getint('disk_cache', 'shards')
        timeout=settings.getfloat('disk_cache', 'timeout')

        # return FanoutCache(cacheLocation, shards=shards, timeout=timeout,
        #                              size_limit=cacheSizeBytes, eviction_policy=evictionPolicy)
        cache = FanoutCache(cacheLocation, shards=shards, timeout=timeout,
                           size_limit=cacheSizeBytes, eviction_policy=evictionPolicy)
        print('INIT OF CACHE IN CONFIG CLASS')
        cache.close()
        return cache

import sys
from psutil import virtual_memory
from collections import OrderedDict
import uuid
import functools
class cache_head_space:
    """
    This cache takes the free_ram_gb argument which defines the amount of ram that you want to keep
    FREE on the system that the cache is active.  Thus, with free_ram_gb=20 will ensure that the cache can
    grow until 20GB of RAM remain available on the system.  At this point, the cache will trim itself
    to ensure that 20GB of RAM remains free at all times.
    """

    def __init__(self, free_ram_gb=20):
        """
        Initialize the cache with a specified amount of free memory to maintain.

        Args:
            free_ram_gb (int): The amount of memory (in GB) to keep free. Defaults to 20.
        """
        self.head_space = free_ram_gb * (1024**3)
        self.cache = OrderedDict()
        self.update_available_space()
        self.uuid = str(uuid)

    def memoize(self, func):
        """
        Decorate a function to cache its output based on its arguments.

        Args:
            func (callable): The function to memoize.

        Returns:
            callable: A wrapped function with caching enabled.
        """
        kwd_mark = object()
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            key = args + (kwd_mark,) + tuple(sorted(kwargs.items()))
            # print(key)
            out = self.__getitem__(key)
            if out is not None:
                pass
            else:
                out = func(*args, **kwargs)
                self.__setitem__(key, out)
                print('SETITEM 66')
                print(out)
            return out
        return wrapper


    def update_available_space(self):
        """
        Update the available system memory in bytes.
        """
        self.available_space = virtual_memory().available
        # print(f'Available GB: {self.available_space / 1024**3}')

    @staticmethod
    def get_size_object(object):
        """
        Calculate the memory size of an object.

        Args:
            object: The object to measure.

        Returns:
            int: The size of the object in bytes.
        """
        return sys.getsizeof(object)

    def __getitem__(self, key):
        """
        Retrieve an item from the cache.

        Args:
            key: The key of the item to retrieve.

        Returns:
            object: The cached item, or None if the key is not found.
        """
        print('GETITEM 80')
        result = self.cache.get(key)
        if result is not None:
            self.cache.move_to_end(key)
            print('Got from RAM CACHE')
        self.trim_cache()
        print('GETITEM 86')
        return result

    def __setitem__(self, key, value):
        """
        Add an item to the cache, ensuring enough memory is available.

        Args:
            key: The key of the item.
            value: The item to cache.
        """
        while True:
            lock_value = self.cache.get('lock')
            if lock_value is None:
                self.cache['lock'] = self.uuid
            else:
                continue

            if self.cache.get('lock') == self.uuid:
                break

        try:
            new_obj_size = self.get_size_object(value) + self.get_size_object(key)
            space_available = self.trim_cache(extra_space=new_obj_size)
            if space_available:
                self.cache[key] = value
                print('SET TO RAM CACHE')
        finally:
            if self.cache.get('lock') == self.uuid:
                print('IM HERE IN FINALLY NOW')
                self.cache.pop('lock')

    def trim_cache(self, extra_space=0):
        """
        Trim cache to fit head_space
        extra_space will trim further to allow for object insertion
        Return True if trim was successful, False if cache is empty and space is not available

        Args:
            extra_space (int, optional): Additional memory required for new items. Defaults to 0.

        Returns:
            bool: True if the trim was successful, False if the cache is empty and space is unavailable.
        """
        self.update_available_space()
        while self.available_space < self.head_space + extra_space:
            try:
                self.cache.popitem(last=False)
            except KeyError: # If dict is empty thus cache can not be reduced
                return False
            self.update_available_space()
        return True

    def __del__(self):
        """
        Clear all items from the cache when the instance is deleted.
        """
        for key in self.cache.keys():
            self.cache[key] = None

def test(head_space=20):
    """
    Test the `cache_head_space` class by populating it with large numpy arrays.

    Args:
        head_space (int, optional): The amount of memory (in GB) to keep free. Defaults to 20.
    """
    import numpy as np
    a = cache_head_space(head_space)
    for ii in range(100000):
        a[ii] = np.ones((100,100,100))


cache_ram = cache_head_space(20)