# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:49:10 2022

@author: awatson
"""

"""
Attempt at creating file type hooks to produce links for opening files in other
applications like neuroglancer or napari

Can be used as a filter to determine if the file has the ability to be used by
another process
"""

import utils
from flask import url_for
import neuroGlancer
import openSeadragon
def ng_links(req_path):
    """
    req_path is a string from the 'browse_fs' endpoint to a file.
    If the file type is supported for neuroglancer
    return the ng file entrypoint else return None
    """
    
    file_types = neuroGlancer.neuroglancer_dtypes()
    
    file_type_supported = utils.is_file_type(file_types, req_path)
    
    if file_type_supported:
        new_path = req_path.replace(url_for('browse_fs'),url_for('neuro_glancer_entry'),1)
        return new_path
    
    else:
        return None

def opsd_links(req_path):
    """
    req_path is a string from the 'browse_fs' endpoint to a file.
    If the file type is supported for Openseadragon
    return the osd file entrypoint else return None
    """
    file_types = openSeadragon.openseadragon_dtypes()
    file_type_supported = utils.is_file_type(file_types, req_path)
    if file_type_supported:
        new_path = req_path.replace(url_for('browse_fs'),url_for('openseadragon_entry'),1)
        return new_path
    
    else:
        return None


def downloadable(req_path,size=None, max_sizeGB=None):
    """
    Determine if the requested file can be downlaoded
    max_size imposes a limit on the size of the file that can
    be downloaded
    
    req_path = str to url or file
    size = bytes as int/float
    max_sizeGB = int/float in GB
    """
    if max_sizeGB is not None:
        if size/1000/1000/1000 > max_sizeGB:
            return None
    return req_path



def dir_as_file(req_path):
    """
    Some directories should be treated like files.  For instance,
    it is not helpful to open a zarr directory in the browser,
    but options to view in neuroglancer should be available
    
    Temporalilly this looks at neuroglancer support only
    """
    
    file_types = neuroGlancer.neuroglancer_dtypes()
    file_type_supported = utils.is_file_type(file_types, req_path)
    
    if file_type_supported:
        return req_path
    else:
        return None
    
    
    
