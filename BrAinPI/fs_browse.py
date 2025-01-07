# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:47:11 2022

@author: awatson
"""

'''
Make a browseable filesystem that limits paths to those configured in 
settings.ini and according to authentication / groups.ini
'''

import traceback
from flask_login import (
                         current_user,
                         login_required,
                         logout_user
                         )

from flask import (
    render_template,
    request, 
    flash, 
    url_for, 
    redirect, 
    send_file,
    jsonify
    )

import glob, os
from natsort import natsorted
import datetime
from logger_tools import logger
## Project-specific imports
import utils, auth
import file_type_support as fts

dl_size_GB = 8


def time_format(time_from_os_stat):
    """
    Format a timestamp into a human-readable string.

    Args:
        time_from_os_stat (int, float, or datetime.datetime): The timestamp to format.

    Returns:
        str: A string in the format "YYYY-MM-DD HH:mm".
    """
    if isinstance(time_from_os_stat, (int,float)):
        return datetime.datetime.fromtimestamp(time_from_os_stat).strftime("%Y-%m-%d %H:%I")
    elif isinstance(time_from_os_stat, datetime.datetime):
        return time_from_os_stat.strftime("%Y-%m-%d %H:%I")


#########################################################################################################
##  END TESTING
##########################################################################################################

'''
path_info = { # Name of object that tracks path information for browser endpoint
    'paths' :dict { # A dict where each entry is a dir/file entry in the browser 
        path_name :dict { # The name of the dir/file shown in the browser: 
                # this is an assigned name in settings.ini for the root of the endpoint
                # This is the name of the on disk dir/file for all subsequent browsing
            'location':str, # Physical path of dir/file
            'parent':str, # Location of physical path directly above 'location'
            'is_dir':bool, # True of directory, False is assumed to be a file
            'file_system':str,  # 'posix','s3'
            'mod_time':time_format(), # Modification time of the this path
            'contents':{ # Internal contents: only applicable to directories
                'dirs':list, # directory contents
                'files':list, # file_contents
                'total_entries':int, # total num of dirs and files
                'total_dirs':int, # total num of dirs
                'total_files':int,# total num of files
            },
            'extension'str, # Extension associated with file OR directory
                            # For example directory .ome.zarr indicates OME-Zarr dataset
                            # For example directory .omehans indicates OME-Zarr using zarr h5_nested_store
        },
    },
}
'''

# def path_info_to_dict(file_system_path):
#
#     if not os.path.exists(file_system_path):
#         return None
#
#     path_info = {
#         'children': {},
#         'location': file_system_path,
#         'parent': os.path.split(file_system_path)[0],
#         'is_dir': os.path.isdir(file_system_path)
#     }
#     path_info['is_file'] = not path_info['is_dir']
#
#     if path_info['is_dir']:
#         for _, dirs, files in os.walk(file_system_path):
#             path_info['children'][path_name]['children'] = {}
#             path_info['children'][path_name]['children']['dirs'] = dirs
#             path_info['children'][path_name]['children']['files'] = files
#             path_info['children'][path_name]['children']['total_entries'] = len(dirs) + len(files)
#             path_info['children'][path_name]['children']['total_dirs'] = len(dirs)
#             path_info['children'][path_name]['children']['total_files'] = len(files)
#
# def collect_path_info(base,request,browser_root=True):
#     html_path_split = utils.split_html(request.path)
#     settings = utils.get_config()
#     path_map = utils.get_path_map(settings, current_user.is_authenticated)
#
#
#     # Dict to manage path information
#     path_info = {
#         'parent': {
#
#         },
#         'children': {
#
#         },
#     }
#
#     # Get paths to display in browser based on the requested path
#     if browser_root:
#         # If browsing the root gather current_path info directly from path_map
#         to_browse = [x for x in path_map]
#         to_browse = natsorted(to_browse)
#
#         for ii in to_browse:
#             path_info['children'][ii] = {'location': path_map[ii], 'parent': path_map[ii], 'is_dir':os.path.isdir(path_map[ii])}
#             path_info['children'][ii]['mod_time'] = time_format(os.stat(path_map[ii]).st_mtime)
#
#     else:
#         # Construct real paths from names in path_map dict
#         to_browse = utils.from_html_to_path(request.path, path_map)
#         # Get a list of all files/dirs in to_browse
#         for parent, dirs, files in os.walk(to_browse):
#             # Construct FULL physical paths
#             # to_browse = natsorted([os.path.join(parent,x) for x in dirs]) + natsorted([os.path.join(parent,x) for x in files])
#             break
#
#         dirs = natsorted(dirs)
#         for path_name in dirs:
#             path_info['children'][path_name] = {'location': os.path.join(parent,path_name), 'parent': parent, 'is_dir': True}
#             path_info['children'][path_name]['mod_time'] = time_format(os.stat(path_info['children'][path_name]['location']).st_mtime)
#
#         files = natsorted(files)
#         for path_name in files:
#             path_info['children'][path_name] = {'location': os.path.join(parent,path_name), 'parent': parent, 'is_dir': False}
#             path_info['children'][path_name]['mod_time'] = time_format(os.stat(path_info['children'][path_name]['location']).st_mtime)
#
#     # Add filesystem type
#     for path_name in path_info['children']:
#         if 's3://' in path_info['children'][path_name]['location'].lower():
#             path_info['children'][path_name]['file_system'] = 's3'
#         else:
#             path_info['children'][path_name]['file_system'] = 'posix'
#
#     # Collect the contents of subdirectories
#     for path_name in path_info['children']:
#         for _, dirs, files in os.walk(path_info['children'][path_name]['location']):
#             path_info['children'][path_name]['children'] = {}
#             path_info['children'][path_name]['children']['dirs'] = dirs
#             path_info['children'][path_name]['children']['files'] = files
#             path_info['children'][path_name]['children']['total_entries'] = len(dirs) + len(files)
#             path_info['children'][path_name]['children']['total_dirs'] = len(dirs)
#             path_info['children'][path_name]['children']['total_files'] = len(files)
#             break
#
#     # Finally grab info about the parent of the path being browsed
#     path_info['parent'] = {}
#     if browser_root:
#         path_info['parent']['location'] = '/'
#         path_info['parent']['parent'] = '/'
#         path_info['parent']['is_dir'] = True
#         path_info['parent']['mod_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%I")
#         path_info['parent']['dirs'] = list(path_map.keys())
#         path_info['parent']['files'] = []
#         path_info['parent']['total_entries'] = len(path_map)
#         path_info['parent']['total_dirs'] = len(path_map)
#         path_info['parent']['total_files'] = 0
#     else:
#         path_info['parent']['location'] = to_browse
#         path_info['parent']['parent'] = os.path.split(to_browse)[0]
#         path_info['parent']['is_dir'] = os.path.isdir(to_browse)
#         path_info['parent']['mod_time'] = time_format(os.stat(to_browse).st_mtime)
#         path_info['parent']['dirs'] = list(path_map.keys())
#         path_info['parent']['files'] = []
#         path_info['parent']['total_entries'] = len(path_map)
#         path_info['parent']['total_dirs'] = len(path_map)
#         path_info['parent']['total_files'] = 0
#
#     return path_info

# def collect_path_info(base,request,browser_root=True):
#     # Dict to manage path information
#     path_info = {
#         'paths': {
#         }
#     }
#
#     html_path_split = utils.split_html(request.path)
#     settings = utils.get_config()
#     path_map = utils.get_path_map(settings, current_user.is_authenticated)
#
#     # Get paths to display in browser based on the requested path
#     if browser_root:
#         # If browsing the root gather current_path info directly from path_map
#         to_browse = [x for x in path_map]
#         to_browse = natsorted(to_browse)
#
#         for ii in to_browse:
#             path_info['paths'][ii] = {'location': path_map[ii], 'parent': path_map[ii], 'is_dir':os.path.isdir(path_map[ii])}
#             path_info['paths'][ii]['mod_time'] = time_format(os.stat(path_map[ii]).st_mtime)
#
#     else:
#         # Construct real paths from names in path_map dict
#         to_browse = utils.from_html_to_path(request.path, path_map)
#         # Get a list of all files/dirs in to_browse
#         for parent, dirs, files in os.walk(to_browse):
#             # Construct FULL physical paths
#             # to_browse = natsorted([os.path.join(parent,x) for x in dirs]) + natsorted([os.path.join(parent,x) for x in files])
#             break
#
#         dirs = natsorted(dirs)
#         for path_name in dirs:
#             path_info['paths'][path_name] = {'location': os.path.join(parent,path_name), 'parent': parent, 'is_dir': True}
#             path_info['paths'][path_name]['mod_time'] = time_format(os.stat(path_info['paths'][path_name]['location']).st_mtime)
#
#         files = natsorted(files)
#         for path_name in files:
#             path_info['paths'][path_name] = {'location': os.path.join(parent,path_name), 'parent': parent, 'is_dir': False}
#             path_info['paths'][path_name]['mod_time'] = time_format(os.stat(path_info['paths'][path_name]['location']).st_mtime)
#
#     # Add filesystem type
#     for path_name in path_info['paths']:
#         if 's3://' in path_info['paths'][path_name]['location'].lower():
#             path_info['paths'][path_name]['file_system'] = 's3'
#         else:
#             path_info['paths'][path_name]['file_system'] = 'posix'
#
#     # Collect the contents of subdirectories
#     for path_name in path_info['paths']:
#         for _, dirs, files in os.walk(path_info['paths'][path_name]['location']):
#             path_info['paths'][path_name]['contents'] = {}
#             path_info['paths'][path_name]['contents']['dirs'] = dirs
#             path_info['paths'][path_name]['contents']['files'] = files
#             path_info['paths'][path_name]['contents']['total_entries'] = len(dirs) + len(files)
#             path_info['paths'][path_name]['contents']['total_dirs'] = len(dirs)
#             path_info['paths'][path_name]['contents']['total_files'] = len(files)
#             break
#
#     # Finally grab info about the parent of the path being browsed
#     path_info['parent'] = {}
#     if browser_root:
#         path_info['parent']['location'] = '/'
#         path_info['parent']['parent'] = '/'
#         path_info['parent']['is_dir'] = True
#         path_info['parent']['mod_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%I")
#         path_info['parent']['dirs'] = list(path_map.keys())
#         path_info['parent']['files'] = []
#         path_info['parent']['total_entries'] = len(path_map)
#         path_info['parent']['total_dirs'] = len(path_map)
#         path_info['parent']['total_files'] = 0
#     else:
#         path_info['parent']['location'] = to_browse
#         path_info['parent']['parent'] = os.path.split(to_browse)[0]
#         path_info['parent']['is_dir'] = os.path.isdir(to_browse)
#         path_info['parent']['mod_time'] = time_format(os.stat(to_browse).st_mtime)
#         path_info['parent']['dirs'] = list(path_map.keys())
#         path_info['parent']['files'] = []
#         path_info['parent']['total_entries'] = len(path_map)
#         path_info['parent']['total_dirs'] = len(path_map)
#         path_info['parent']['total_files'] = 0
#
#     return path_info


# def get_path_data(base, request):
#     return jsonify(collect_path_info(base, request, browser_root=False))
#     # Split the requested path to a tuple that can be reused below
#     html_path_split = utils.split_html(request.path)
#     print(html_path_split)
#
#     # Extract settings information that can be reused
#     # Doing this here allows changes to paths to be dynamic (ie changes can be made while server is live)
#     # May want to change this so that each browser access does not require access to settings file on disk.
#     ## DEFAULT ## utils.get_config(file='settings.ini',allow_no_value=True)
#     settings = utils.get_config()
#
#     page_description = {}
#     page_description['title'] = settings['browser']['title']
#     page_description['header'] = settings['browser']['header']
#     page_description['footer'] = settings['browser']['footer']
#     page_description['logo'] = settings['app']['logo']
#     # print(page_description)
#
#     # Determine what directories that users are allowed to browse
#     # based on authentication status and boot them if the path is not valid
#     path_map = utils.get_path_map(settings, current_user.is_authenticated)
#
#     # user={'is_authenticated':current_user.is_authenticated, 'id':current_user.id if current_user.is_authenticated else None}
#
#     if current_user.is_authenticated:
#
#         # Read in group information
#         groups = utils.get_config('groups.ini', allow_no_value=True)
#
#         # Build a list of allowed folders
#         allowed_list = [current_user.id.lower()]
#         for ii in groups:  # Group names
#             if ii.lower() == 'all':
#                 continue
#             for oo in groups[ii]:  # Users in each group
#                 if current_user.id.lower() == oo.lower():  # Current user matches the user in the group
#                     # print('Line 168')
#                     allowed_list.append(ii.lower())
#         print(allowed_list)
#
#     # Dict to manage path information
#     path_info = {
#         'paths':{
#     }
#     }
#
#     # If browsing the root gather current_path info in a special way
#     if len(html_path_split) == 1:
#
#         to_browse = [x for x in path_map]
#         to_browse = natsorted(to_browse)
#         # print(to_browse)
#         for ii in to_browse:
#             path_info['paths'][ii] = {'location':path_map[ii], 'root':None}
#             if 's3://' in path_info['paths'][ii]['location'].lower():
#                 path_info['paths'][ii]['file_system'] = 's3'
#             else:
#                 path_info['paths'][ii]['file_system'] = 'posix'
#         # Limit the paths that are seen by an authenticated user based on the contents of the next directory
#         # limit by username and groups
#         limited_path_info = {'paths':{}}
#         if current_user.is_authenticated:
#             real_paths = [path_map[x] for x in to_browse]  # converts to real path
#             # print(real_paths)
#             print(path_info)
#             for path in path_info['paths']:
#                 print(path)
#                 for _, dirs, files in os.walk(path_info['paths'][path]['location']):
#                     path_info['paths'][path]['contents'] = {}
#                     path_info['paths'][path]['contents']['dirs'] = dirs
#                     path_info['paths'][path]['contents']['files'] = files
#                     path_info['paths'][path]['contents']['total_entries'] = len(dirs) + len(files)
#                     path_info['paths'][path]['contents']['total_dirs'] = len(dirs)
#                     path_info['paths'][path]['contents']['total_files'] = len(files)
#                     break
#
#                 keep = False
#                 if path in settings['dir_anon']:
#                     keep = True
#                 elif current_user.id.lower() in groups['all']:
#                     keep = True
#                 elif not settings.getboolean('auth', 'restrict_paths_to_matched_username'):
#                     keep = True
#                 elif settings.getboolean('auth',
#                                          'restrict_paths_to_matched_username') and current_user.id.lower() in dirs:
#                     keep = True
#                 elif any([x.lower() in dirs for x in allowed_list]):
#                     keep = True
#
#                 if keep:
#                     # Collect mod time for remaining dirs/files
#                     path_info['paths'][path]['mod_time'] = os.stat(path_info['paths'][path]['location']).st_mtime
#                     limited_path_info['paths'][path] = path_info['paths'][path]
#
#             # print(interior_listing)
#             path_info = limited_path_info
#             return jsonify(path_info)
#
#
#         current_path = {}
#         # current_path['root'] = base[:-1]
#         current_path['files'] = []
#         current_path['html_path_split'] = html_path_split
#         current_path['current_path'] = base[:-1]
#         current_path['current_path_name'] = base[1:-1]
#         current_path['current_path_modtime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%I")
#         current_path['current_path_entries'] = (len(to_browse), 0)
#
#         current_path['parent_path'] = base[:-1]
#         current_path['parent_is_root'] = True
#         current_path['parent_folder_name'] = base[1:-1]
#
#         current_path['dirs'] = [path_map[x] for x in to_browse]  # converts to real path
#         current_path['dirs_name'] = to_browse
#         # print(current_path['dirs'])
#         current_path['dirs_stat'] = [os.stat(x) for x in current_path['dirs']]
#         # print(current_path['dirs_stat'])
#         current_path['files_stat'] = []
#         current_path['files_size'] = []
#         current_path['files_name'] = []
#         current_path['files_modtime'] = []
#
#         current_path['dirs_entries'] = [utils.num_dirs_files(x) for x in current_path['dirs']]
#         current_path['dirs_modtime'] = [time_format(x.st_mtime) for x in current_path['dirs_stat']]
#         # print(current_path['dirs_entries'])
#
#         ## Reconstruct html_paths
#         current_path['dirs'] = to_browse
#         current_path['files'] = []
#
#         ## Tuples of each derivitive path
#         current_path['all_parents'] = [
#             (html_path_split[-(idx + 1)], '/' + os.path.join(*html_path_split[0:len(html_path_split) - idx])) for idx, _
#             in enumerate(html_path_split)]
#
#         ## Special case, some directories should be treated like files (ie. .zarr, .weave, .z_sharded)
#         remove_dirs_idx = []
#         for idx, _ in enumerate(current_path['dirs']):
#             supported = fts.ng_links(current_path['dirs'][idx])
#             if supported:
#                 remove_dirs_idx.append(idx)
#                 current_path['files'].append(supported)
#                 current_path['files_stat'].append(current_path['dirs_stat'][idx])
#                 current_path['files_size'].append((0, 'B', 0))
#                 current_path['files_modtime'].append(current_path['dirs_modtime'][idx])
#                 current_path['files_name'].append(current_path['dirs_name'][idx])
#
#         # print(remove_dirs_idx)
#         if remove_dirs_idx != []:
#             current_path['dirs'] = [x for idx, x in enumerate(current_path['dirs']) if idx not in remove_dirs_idx]
#             current_path['dirs_name'] = [x for idx, x in enumerate(current_path['dirs_name']) if
#                                          idx not in remove_dirs_idx]
#             current_path['dirs_stat'] = [x for idx, x in enumerate(current_path['dirs_stat']) if
#                                          idx not in remove_dirs_idx]
#             current_path['dirs_entries'] = [x for idx, x in enumerate(current_path['dirs_entries']) if
#                                             idx not in remove_dirs_idx]
#             current_path['dirs_modtime'] = [x for idx, x in enumerate(current_path['dirs_modtime']) if
#                                             idx not in remove_dirs_idx]
#
#         ## Determine what options each files has
#         current_path['files_ng_slug'] = [fts.ng_links(x) for x in current_path['files']]
#         current_path['files_ng_info'] = [os.path.join(x, 'info') if x is not None else None for x in
#                                          current_path['files_ng_slug']]
#         current_path['files_dl'] = [fts.downloadable(x, size=current_path['files_stat'][idx].st_size,
#                                                      max_sizeGB=settings.getint('browser', 'max_dl_file_size_GB')) for
#                                     idx, x in enumerate(current_path['files'])]
#
#     else:
#
#         try:
#             print('Line 73')
#
#             if html_path_split[1] not in path_map:
#                 flash('You are not authorized to browse to path {}'.format(request.path))
#                 return redirect(url_for('login'))
#
#             ## Boot user if they are gaining access to an inappropriate folder
#             ## Retain access to anon folders
#             ##Retain access to everything if in 'all' group
#             ## Determine if folders should be restricted to user-name-matched only
#             ## Retain if folder name is in allowed_list of usernames / groups
#             if len(html_path_split) >= 3 and \
#                     not html_path_split[1].lower() in [x.lower() for x in settings['dir_anon']] and \
#                     not current_user.id.lower() in [x.lower() for x in groups['all']] and \
#                     settings.getboolean('auth', 'restrict_paths_to_matched_username') and \
#                     not html_path_split[2].lower() in allowed_list:
#                 flash('You are not authorized to browse to path {}'.format(request.path))
#                 return redirect(url_for('login'))
#
#             # Construct real paths from names in path_map dict
#             to_browse = utils.from_html_to_path(request.path, path_map)
#             print('Live 110')
#
#             # Get current directory listing by Files, Directories and stats on each
#             current_path = {}
#             current_path['html_path_split'] = html_path_split
#             current_path['current_path'] = request.path
#             current_path['current_path_name'] = os.path.split(current_path['current_path'])[1]
#             current_path['parent_path'] = '/' + os.path.join(*html_path_split[:-1]) if len(
#                 html_path_split) > 2 else base[:-1]
#             current_path['current_path_entries'] = utils.num_dirs_files(to_browse)
#
#             current_path['parent_is_root'] = False if len(html_path_split) > 2 else True
#             current_path['parent_folder_name'] = html_path_split[-2]
#             if os.path.isdir(to_browse):
#                 current_path['current_path_modtime'] = time_format(os.stat(to_browse).st_mtime)
#             if os.path.isdir(to_browse):
#                 # current_path['isfile'] = False
#                 for root, dirs, files in os.walk(to_browse):
#                     # current_path['root'] = root
#                     current_path['dirs'] = dirs
#                     current_path['files'] = files
#                     break
#                 current_path['dirs'] = [os.path.join(root, x) for x in current_path['dirs']]
#                 current_path['files'] = [os.path.join(root, x) for x in current_path['files']]
#
#                 # keep only directories that have the correct user/group names
#                 if not html_path_split[1].lower() in [x.lower() for x in settings['dir_anon']] and \
#                         current_user.is_authenticated and \
#                         not current_user.id.lower() in [x.lower() for x in groups['all']]:
#
#                     if settings.getboolean('auth', 'restrict_paths_to_matched_username'):
#                         # For predictable filter ALWAYS filter on the html path and not the file system path
#                         tmp_dirs = [x for x in current_path['dirs'] if
#                                     utils.from_path_to_html(x, path_map, request.path, base)[2].lower() in allowed_list]
#                         tmp_dirs = [utils.from_path_to_html(x, path_map, request.path, base) for x in
#                                     current_path['dirs']]
#                         tmp_dirs = [utils.split_html(x) for x in tmp_dirs]
#                         current_path['dirs'] = [x for x, y in zip(current_path['dirs'], tmp_dirs) if
#                                                 y[2].lower() in allowed_list]
#
#                     if settings.getboolean('auth', 'restrict_files_to_listed_file_types'):
#                         to_view = []
#                         for ii in settings['file_types']:
#                             # Further limit to the file type being filtered
#                             current_files = [x for x in current_path['files'] if
#                                              utils.is_file_type(settings['file_types'][ii], utils.split_html(x)[
#                                                  -1])]  # <-- add 'a' to make split consistent when files start with '.'
#                             to_view = to_view + current_files
#                         current_path['files'] = to_view
#
#                 current_path['dirs'] = natsorted(current_path['dirs'])
#                 current_path['files'] = natsorted(current_path['files'])
#
#                 current_path['dirs_name'] = [os.path.split(x)[-1] if x[-1] != '/' else os.path.split(x[:-1])[-1] for x
#                                              in current_path['dirs']]
#                 current_path['dirs_stat'] = [os.stat(x) for x in current_path['dirs']]
#                 current_path['files_stat'] = [os.stat(x) for x in current_path['files']]
#                 current_path['files_size'] = [utils.get_file_size(x.st_size) for x in current_path['files_stat']]
#                 current_path['files_modtime'] = [time_format(x.st_mtime) for x in current_path['files_stat']]
#                 current_path['files_name'] = [os.path.split(x)[-1] if x[-1] != '/' else os.path.split(x[:-1])[-1] for x
#                                               in current_path['files']]
#
#                 current_path['dirs_entries'] = [utils.num_dirs_files(x) for x in current_path['dirs']]
#                 current_path['dirs_modtime'] = [time_format(x.st_mtime) for x in current_path['dirs_stat']]
#
#                 ## Reconstruct html_paths
#                 current_path['dirs'] = [utils.from_path_to_html(x, path_map, request.path, base) for x in
#                                         current_path['dirs']]
#                 current_path['files'] = [utils.from_path_to_html(x, path_map, request.path, base) for x in
#                                          current_path['files']]
#
#                 ## Tuples of each derivitive path
#                 current_path['all_parents'] = [
#                     (html_path_split[-(idx + 1)], '/' + os.path.join(*html_path_split[0:len(html_path_split) - idx]))
#                     for idx, _ in enumerate(html_path_split)]
#
#                 ## Special case, some directories should be treated like files (ie. .zarr, .weave, .z_sharded)
#                 remove_dirs_idx = []
#                 for idx, _ in enumerate(current_path['dirs']):
#                     supported = fts.ng_links(current_path['dirs'][idx])
#                     if supported:
#                         remove_dirs_idx.append(idx)
#                         current_path['files'].append(supported)
#                         current_path['files_stat'].append(current_path['dirs_stat'][idx])
#                         current_path['files_size'].append((0, 'B', 0))
#                         current_path['files_modtime'].append(current_path['dirs_modtime'][idx])
#                         current_path['files_name'].append(current_path['dirs_name'][idx])
#
#                 # print(remove_dirs_idx)
#                 if remove_dirs_idx != []:
#                     current_path['dirs'] = [x for idx, x in enumerate(current_path['dirs']) if
#                                             idx not in remove_dirs_idx]
#                     current_path['dirs_name'] = [x for idx, x in enumerate(current_path['dirs_name']) if
#                                                  idx not in remove_dirs_idx]
#                     current_path['dirs_stat'] = [x for idx, x in enumerate(current_path['dirs_stat']) if
#                                                  idx not in remove_dirs_idx]
#                     current_path['dirs_entries'] = [x for idx, x in enumerate(current_path['dirs_entries']) if
#                                                     idx not in remove_dirs_idx]
#                     current_path['dirs_modtime'] = [x for idx, x in enumerate(current_path['dirs_modtime']) if
#                                                     idx not in remove_dirs_idx]
#
#                 ## Determine what options each files has
#                 current_path['files_ng_slug'] = [fts.ng_links(x) for x in current_path['files']]
#                 current_path['files_ng_info'] = [os.path.join(x, 'info') if x is not None else None for x in
#                                                  current_path['files_ng_slug']]
#                 current_path['files_dl'] = [fts.downloadable(x, size=current_path['files_stat'][idx].st_size,
#                                                              max_sizeGB=settings.getint('browser',
#                                                                                         'max_dl_file_size_GB')) for
#                                             idx, x in enumerate(current_path['files'])]
#
#             elif os.path.isfile(to_browse):
#                 current_path['isFile'] = True
#                 # return jsonify(current_path)
#                 return send_file(to_browse, download_name=os.path.split(to_browse)[1], as_attachment=True)
#
#             else:
#                 # If a non-file / dir is passed, move backward to the nearest file/dir
#                 return redirect(current_path['parent_path'])
#
#         except Exception:
#             flash('You must not be authorized to browse to path {}'.format(request.path))
#             print(traceback.format_exc())
#             return redirect(url_for('login'))
#
#     '''
#     Build a dict for each file with each available option
#     This is used by javascript to build the modl
#     '''
#     files_json = {}
#     for idx, file in enumerate(current_path['files']):
#         files_json[file] = {}
#         files_json[file]['files'] = file
#         files_json[file]['files_name'] = current_path['files_name'][idx]
#         files_json[file]['files_size'] = current_path['files_size'][idx]
#         files_json[file]['files_modtime'] = current_path['files_modtime'][idx]
#         files_json[file]['files_ng_slug'] = current_path['files_ng_slug'][
#             idx]  # + '/?neuroglancer' if isinstance(current_path['files_ng_slug'][idx],str) else ''  #Assist with google analytics
#         files_json[file]['files_ng_info'] = current_path['files_ng_info'][idx]
#         files_json[file]['files_dl'] = current_path['files_dl'][idx]
#
#     current_path['files_json'] = files_json
#     # print(files_json)
#
#     '''
#     Everything above this builds
#     'page_description', 'current_path'
#     This is the data passed to render template to build the browser
#     Return renders the browser
#     '''
#
#     return 'render_template', page_description, current_path
#

#########################################################################################################
##  END TESTING
##########################################################################################################
def get_path_data(base, request):
    """
    Retrieve metadata and structure for the requested path in the filesystem.

    This function builds information about the current path, directories, and files. It applies
    user permissions and group-based restrictions to determine accessible paths.

    Args:
        base (str): The base entry point for the browsing interface.
        request (Flask.Request): The Flask request object containing the current path.

    Returns:
        tuple or str: If successful, returns a tuple of `('render_template', page_description, current_path)`.
                      If unauthorized, redirects to the login page or sends a file response.
    """
    # Split the requested path to a tuple that can be reused below
    html_path_split = utils.split_html(request.path)
    logger.info(html_path_split)

    # Extract settings information that can be reused
    # Doing this here allows changes to paths to be dynamic (ie changes can be made while server is live)
    # May want to change this so that each browser access does not require access to settings file on disk.
    ## DEFAULT ## utils.get_config(file='settings.ini',allow_no_value=True)
    from config_tools import get_config
    settings = get_config()

    page_description = {}
    page_description['title'] = settings['browser']['title']
    page_description['header'] = settings['browser']['header']
    page_description['footer'] = settings['browser']['footer']
    page_description['logo'] = settings['app']['logo']
    # print(page_description)

    # Determine what directories that users are allowed to browse
    # based on authentication status and boot them if the path is not valid
    path_map =  utils.get_path_map(settings,current_user.is_authenticated)

    # user={'is_authenticated':current_user.is_authenticated, 'id':current_user.id if current_user.is_authenticated else None}

    if current_user.is_authenticated:

        # Read in group information
        groups = get_config('groups.ini',allow_no_value=True)

        # Build a list of allowed folders
        allowed_list = [current_user.id.lower()]
        for ii in groups: # Group names
            if ii.lower() == 'all':
                continue
            for oo in groups[ii]: # Users in each group
                if current_user.id.lower() == oo.lower(): # Current user matches the user in the group
                    # print('Line 168')
                    allowed_list.append(ii.lower())
        logger.info(allowed_list)


    # If browsing the root gather current_path info in a special way
    if len(html_path_split) == 1:

        to_browse = [x for x in path_map]
        to_browse = natsorted(to_browse)
        # print(to_browse)

        # Limit the paths that are seen by an authenticated user based on the contents of the next directory
        # limit by username and groups
        limited_list = []
        if current_user.is_authenticated:
            real_paths = [path_map[x] for x in to_browse]  # converts to real path
            # print(real_paths)
            for browse,path in zip(to_browse,real_paths):
                # print(browse)
                # print(path)
                interior_listing = utils.list_all_contents(path)
                # print(interior_listing)
                interior_listing = [os.path.relpath(x,path).lower() for x in interior_listing]
                # print(interior_listing)
                if browse in settings['dir_anon']:
                    limited_list.append(browse)
                elif current_user.id.lower() in groups['all']:
                    limited_list.append(browse)
                elif settings.getboolean('auth', 'restrict_paths_to_matched_username') and current_user.id.lower() in interior_listing:
                    limited_list.append(browse)
                elif not settings.getboolean('auth', 'restrict_paths_to_matched_username'):
                    limited_list.append(browse)
                elif any([x.lower() in interior_listing for x in allowed_list]):
                    limited_list.append(browse)

            # print(interior_listing)
            to_browse = limited_list
            # print(to_browse)


        current_path = {}
        # current_path['root'] = base[:-1]
        current_path['files'] = []
        current_path['html_path_split'] = html_path_split
        current_path['current_path'] = base[:-1]
        current_path['current_path_name'] = base[1:-1]
        current_path['current_path_modtime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%I")
        current_path['current_path_entries'] = (len(to_browse),0)

        current_path['parent_path'] = base[:-1]
        current_path['parent_is_root'] = True
        current_path['parent_folder_name'] = base[1:-1]


        current_path['dirs'] = [path_map[x] for x in to_browse] #converts to real path
        current_path['dirs_name'] = to_browse
        # print(current_path['dirs'])
        # current_path['dirs_stat'] = [os.stat(x) for x in current_path['dirs']]
        # print(current_path['dirs_stat'])
        current_path['files_stat'] = []
        current_path['files_size'] = []
        current_path['files_name'] = []
        current_path['files_modtime'] = []

        current_path['dirs_entries'] = [utils.num_dirs_files(x) for x in current_path['dirs']]
        # current_path['dirs_modtime'] = [time_format(x.st_mtime) for x in current_path['dirs_stat']]
        current_path['dirs_modtime'] = [time_format(utils.get_mod_time(x)) for x in current_path['dirs']]
        # print(current_path['dirs_entries'])

        ## Reconstruct html_paths
        current_path['dirs'] = to_browse
        current_path['files'] = []

        ## Tuples of each derivitive path
        current_path['all_parents'] = [ (html_path_split[-(idx+1)], '/' + os.path.join(*html_path_split[0:len(html_path_split)-idx])) for idx,_ in enumerate(html_path_split) ]

        ## Special case, some directories should be treated like files (ie. .zarr, .weave, .z_sharded)
        remove_dirs_idx = []
        for idx,_ in enumerate(current_path['dirs']):
            supported = fts.ng_links(current_path['dirs'][idx])
            if supported:
                remove_dirs_idx.append(idx)
                current_path['files'].append(supported)
                current_path['files_stat'].append(current_path['dirs_stat'][idx])
                current_path['files_size'].append((0,'B',0))
                current_path['files_modtime'].append(current_path['dirs_modtime'][idx])
                current_path['files_name'].append(current_path['dirs_name'][idx])

        # print(remove_dirs_idx)
        if remove_dirs_idx != []:
            current_path['dirs'] = [ x for idx,x in enumerate(current_path['dirs']) if idx not in remove_dirs_idx ]
            current_path['dirs_name'] = [ x for idx,x in enumerate(current_path['dirs_name']) if idx not in remove_dirs_idx ]
            current_path['dirs_stat'] = [ x for idx,x in enumerate(current_path['dirs_stat']) if idx not in remove_dirs_idx ]
            current_path['dirs_entries'] = [ x for idx,x in enumerate(current_path['dirs_entries']) if idx not in remove_dirs_idx ]
            current_path['dirs_modtime'] = [ x for idx,x in enumerate(current_path['dirs_modtime']) if idx not in remove_dirs_idx ]

        ## Determine what options each files has
        current_path['files_ng_slug'] = [fts.ng_links(x) for x in current_path['files']]
        current_path['files_ng_info'] = [os.path.join(x,'info') if x is not None else None for x in current_path['files_ng_slug']]
        current_path['files_dl'] = [fts.downloadable(x, size=current_path['files_stat'][idx].st_size, max_sizeGB=settings.getint('browser','max_dl_file_size_GB')) for idx, x in enumerate(current_path['files'])]
        current_path['files_opsd_slug'] = [fts.opsd_links(x) for x in current_path['files']]
    else:

        try:
            logger.info('Line 73')

            if html_path_split[1] not in path_map:
                flash('You are not authorized to browse to path {}'.format(request.path))
                return redirect(url_for('login'))

            ## Boot user if they are gaining access to an inappropriate folder
            ## Retain access to anon folders
            ##Retain access to everything if in 'all' group
            ## Determine if folders should be restricted to user-name-matched only
            ## Retain if folder name is in allowed_list of usernames / groups
            if len(html_path_split) >= 3 and \
                not html_path_split[1].lower() in [x.lower() for x in settings['dir_anon']] and \
                not current_user.id.lower() in [x.lower() for x in groups['all']] and \
                settings.getboolean('auth', 'restrict_paths_to_matched_username') and \
                not html_path_split[2].lower() in allowed_list:

                flash('You are not authorized to browse to path {}'.format(request.path))
                return redirect(url_for('login'))

            # Construct real paths from names in path_map dict
            to_browse = utils.from_html_to_path(request.path, path_map)
            logger.info('Live 110')

            if utils.isfile(to_browse):
                return utils.send_file(to_browse)
                # current_path['isFile'] = True
                # # return jsonify(current_path)
                # return send_file(to_browse, download_name=os.path.split(to_browse)[1], as_attachment=True)

            # Get current directory listing by Files, Directories and stats on each
            current_path = {}
            current_path['html_path_split'] = html_path_split
            current_path['current_path'] = request.path
            current_path['current_path_name'] = os.path.split(current_path['current_path'])[1]
            current_path['parent_path'] = '/' + os.path.join(*html_path_split[:-1]) if len(html_path_split) > 2 else base[:-1]
            current_path['current_path_entries'] = utils.num_dirs_files(to_browse)

            current_path['parent_is_root'] = False if len(html_path_split) > 2 else True
            current_path['parent_folder_name'] = html_path_split[-2]
            if utils.isdir(to_browse):
                current_path['current_path_modtime'] = time_format(utils.get_mod_time(to_browse))
                # current_path['isfile'] = False
                root, dirs, files = utils.get_dir_contents(to_browse)
                # current_path['root'] = root
                current_path['dirs'] = dirs
                current_path['files'] = files
                current_path['dirs'] = [os.path.join(root,x) for x in current_path['dirs']]
                current_path['files'] = [os.path.join(root,x) for x in current_path['files']]

                #keep only directories that have the correct user/group names
                if not html_path_split[1].lower() in [x.lower() for x in settings['dir_anon']] and \
                    current_user.is_authenticated and \
                    current_user.is_authenticated and \
                    not current_user.id.lower() in [x.lower() for x in groups['all']]:

                        if settings.getboolean('auth', 'restrict_paths_to_matched_username'):
                            # For predictable filter ALWAYS filter on the html path and not the file system path
                            tmp_dirs = [x for x in current_path['dirs'] if utils.from_path_to_html(x,path_map,request.path,base)[2].lower() in allowed_list]
                            tmp_dirs = [utils.from_path_to_html(x,path_map,request.path,base) for x in current_path['dirs']]
                            tmp_dirs = [utils.split_html(x) for x in tmp_dirs]
                            current_path['dirs'] = [x for x,y in zip(current_path['dirs'], tmp_dirs) if y[2].lower() in allowed_list]

                        if settings.getboolean('auth','restrict_files_to_listed_file_types'):
                            to_view = []
                            for ii in settings['file_types']:
                                # Further limit to the file type being filtered
                                current_files = [ x for x in current_path['files'] if utils.is_file_type(settings['file_types'][ii], utils.split_html(x)[-1]) ]#<-- add 'a' to make split consistent when files start with '.'
                                to_view = to_view + current_files
                            current_path['files'] = to_view

                current_path['dirs'] = natsorted(current_path['dirs'])
                current_path['files'] = natsorted(current_path['files'])

                current_path['dirs_name'] = [os.path.split(x)[-1] if x[-1] != '/' else os.path.split(x[:-1])[-1] for x in current_path['dirs']]
                # current_path['dirs_stat'] = [os.stat(x) for x in current_path['dirs']]
                # current_path['files_stat'] = [os.stat(x) for x in current_path['files']]
                current_path['files_size'] = [utils.format_file_size(utils.get_file_size(x)) for x in current_path['files']]
                current_path['files_modtime'] = [time_format(utils.get_mod_time(x)) for x in current_path['files']]
                current_path['files_name'] = [os.path.split(x)[-1] if x[-1] != '/' else os.path.split(x[:-1])[-1] for x in current_path['files']]

                # print(current_path['dirs'])
                current_path['dirs_entries'] = [utils.num_dirs_files(x) for x in current_path['dirs']]
                current_path['dirs_modtime'] = [time_format(utils.get_mod_time(x)) for x in current_path['dirs']]

                ## Reconstruct html_paths
                current_path['files_real_path'] = current_path['files']
                current_path['dirs_real_path'] = current_path['dirs']
                current_path['dirs'] = [utils.from_path_to_html(x,path_map,request.path,base) for x in current_path['dirs']]
                current_path['files'] = [utils.from_path_to_html(x,path_map,request.path,base) for x in current_path['files']]

                ## Tuples of each derivitive path
                current_path['all_parents'] = [ (html_path_split[-(idx+1)], '/' + os.path.join(*html_path_split[0:len(html_path_split)-idx])) for idx,_ in enumerate(html_path_split) ]

                ## Special case, some directories should be treated like files (ie. .zarr, .weave, .z_sharded)
                remove_dirs_idx = []
                for idx,_ in enumerate(current_path['dirs']):
                    supported = fts.ng_links(current_path['dirs'][idx])
                    if supported:
                        remove_dirs_idx.append(idx)
                        current_path['files'].append(supported)
                        current_path['files_real_path'].append(current_path['dirs_real_path'][idx])
                        # current_path['files_stat'].append(current_path['dirs_stat'][idx])
                        current_path['files_size'].append((0,'B',0))
                        current_path['files_modtime'].append(current_path['dirs_modtime'][idx])
                        current_path['files_name'].append(current_path['dirs_name'][idx])

                # print(remove_dirs_idx)
                if remove_dirs_idx != []:
                    current_path['dirs'] = [ x for idx,x in enumerate(current_path['dirs']) if idx not in remove_dirs_idx ]
                    current_path['dirs_name'] = [ x for idx,x in enumerate(current_path['dirs_name']) if idx not in remove_dirs_idx ]
                    # current_path['dirs_stat'] = [ x for idx,x in enumerate(current_path['dirs_stat']) if idx not in remove_dirs_idx ]
                    current_path['dirs_entries'] = [ x for idx,x in enumerate(current_path['dirs_entries']) if idx not in remove_dirs_idx ]
                    current_path['dirs_modtime'] = [ x for idx,x in enumerate(current_path['dirs_modtime']) if idx not in remove_dirs_idx ]

                ## Determine what options each files has
                current_path['files_ng_slug'] = [fts.ng_links(x) for x in current_path['files']]
                current_path['files_ng_info'] = [os.path.join(x,'info') if x is not None else None for x in current_path['files_ng_slug']]
                current_path['files_dl'] = [fts.downloadable(x, size=utils.get_file_size(current_path['files_real_path'][idx]), max_sizeGB=settings.getint('browser','max_dl_file_size_GB')) for idx, x in enumerate(current_path['files'])]
                current_path['files_opsd_slug'] = [fts.opsd_links(x) for x in current_path['files']]
            else:
                # If a non-file / dir is passed, move backward to the nearest file/dir
                return redirect(current_path['parent_path'])

        except Exception:
            flash('You must not be authorized to browse to path {}'.format(request.path))
            logger.info(traceback.format_exc())
            return redirect(url_for('login'))

    '''
    Build a dict for each file with each available option
    This is used by javascript to build the modl
    '''
    files_json = {}
    for idx, file in enumerate(current_path['files']):
        files_json[file] = {}
        files_json[file]['files'] = file
        files_json[file]['files_name'] = current_path['files_name'][idx]
        files_json[file]['files_size'] = current_path['files_size'][idx]
        files_json[file]['files_modtime'] = current_path['files_modtime'][idx]
        files_json[file]['files_ng_slug'] = current_path['files_ng_slug'][idx] #+ '/?neuroglancer' if isinstance(current_path['files_ng_slug'][idx],str) else ''  #Assist with google analytics
        files_json[file]['files_ng_info'] = current_path['files_ng_info'][idx]
        files_json[file]['files_dl'] = current_path['files_dl'][idx]
        files_json[file]['files_opsd_slug'] = current_path['files_opsd_slug'][idx]
    current_path['files_json'] = files_json
    # print(files_json)

    '''
    Everything above this builds
    'page_description', 'current_path'
    This is the data passed to render template to build the browser
    Return renders the browser
    '''

    return 'render_template', page_description, current_path
    


def initiate_browseable(app,config):
    """
    Initialize the browseable filesystem interface with Flask.

    This function sets up two routes:
    - `/browser/`: Serves a web interface for browsing directories and files.
    - `/browser_json/`: Serves the directory structure as JSON.

    Args:
        app (Flask): The Flask application instance.
        config (object): The configuration object containing application settings.

    Returns:
        Flask: The modified Flask application with the added routes.
    """
    from brain_api_main import login_manager
    
    # base entrypoint must always begin and end with '/' --> /my_entry/
    base = '/browser/'
    @app.route(base + '<path:req_path>')
    @app.route(base, defaults={'req_path': ''})
    def browse_fs(req_path):
        
        
        logger.info(request.path)
        # print(request.remote_addr)
        # print(request.environ.get('HTTP_X_REAL_IP', request.remote_addr))
        # print(request.environ['REMOTE_ADDR'])
        # import pprint
        # pprint.pprint(request.environ)
        # print(request.access_route[-1])
        
        out = get_path_data(base, request)
        
        if isinstance(out,tuple) and out[0] == 'render_template':
            page_description, current_path = out[1:]
            return render_template(
                'fl_browse_table_dir.html',
                current_path={**page_description, **current_path}, 
                user=auth.user_info(),
                gtag=config.settings.get('GA4', 'gtag')
                )
        else:
            return out
    
    
    # base entrypoint must always begin and end with '/' --> /my_entry/
    base_json = '/browser_json/'
    @app.route(base_json + '<path:req_path>')
    @app.route(base_json, defaults={'req_path': ''})
    def browse_fs_json(req_path):
        
        
        print(request.path)
        
        out = get_path_data(base, request)
        
        if isinstance(out,tuple) and out[0] == 'render_template':
            page_description, current_path = out[1:]
            return jsonify({**page_description, **current_path})
        else:
            return 'This did not return a JSON, maybe you have the wrong path'
    
    return app


def initiate_NOT_browseable(app, config):
    """
    Initialize a disabled browseable filesystem interface with Flask.

    This function sets up the `/browser/` route but always returns a 404 error, effectively
    disabling the browsing functionality.

    Args:
        app (Flask): The Flask application instance.
        config (object): The configuration object containing application settings.

    Returns:
        Flask: The modified Flask application with the disabled route.
    """
    from flask import abort

    # base entrypoint must always begin and end with '/' --> /my_entry/
    base = '/browser/'

    @app.route(base + '<path:req_path>')
    @app.route(base, defaults={'req_path': ''})
    def browse_fs(req_path):
        abort(404)

    return app
    
    
