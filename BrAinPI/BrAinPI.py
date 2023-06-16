# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:06:07 2021

@author: alpha
"""

import flask, json, os, ast, re, io, sys
from flask import request, Response, send_file, render_template, jsonify
from flask_cors import cross_origin, CORS
import numpy as np
from datetime import datetime, timedelta
# import dask.array as da

## Project imports
import auth
import utils


## File-type handler imports (some are project specific)
import tifffile as tf
import zarrLoader
import zarr_zip_sharded_loader
import imaris_ims_file_reader as ims
from ome_zarr_loader import ome_zarr_loader

from functools import lru_cache




os.chdir(sys.path[0])

# from weave.weave_read import weave_read

'''
https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world

To run w/ gunicorn:  gunicorn -w 1 -b 0.0.0.0:5000 --chdir /CBI_FastStore/cbiPythonTools/bil_api/bil_api wsgi:app
To run w/ gunicorn:  gunicorn -b 0.0.0.0:5000 --chdir /CBI_FastStore/cbiPythonTools/bil_api/bil_api wsgi:app -w 1 --threads 12
To run w/ gunicorn:  gunicorn -b 0.0.0.0:5000 --chdir /CBI_FastStore/cbiPythonTools/bil_api/bil_api wsgi:app -w 2 --threads 12
To run w/ gunicorn:  gunicorn -b 0.0.0.0:5000 --chdir /CBI_FastStore/cbiPythonTools/bil_api/bil_api wsgi:app -w 4 --threads 6

To run development:  python -i /CBI_FastStore/cbiPythonTools/bil_api/bil_api/flaskAPI_gunicorn.py
'''
## Grab settings information from config.ini file
settings = utils.get_config('settings.ini')

## Setup cache location based on OS type
## Optional situations like machine name can be used to customize
if os.name == 'nt':
    cacheLocation = settings.get('disk_cache','location_win')
elif 'c00' in os.uname()[1]:
    cacheLocation = settings.get('disk_cache','location_unix')
else:
    cacheLocation = settings.get('disk_cache','location_unix')
    #cacheLocation = None

# Instantiate class that will manage all open datasets
# This will remain in the global env and be accessed by multiple route methods
config = utils.config(
    cacheLocation=cacheLocation,
    cacheSizeGB=settings.getint('disk_cache','cacheSizeGB'),
    evictionPolicy=settings.get('disk_cache','evictionPolicy'),
    shards=settings.getint('disk_cache','shards'),
    timeout=settings.getfloat('disk_cache','timeout')
    )

config.settings = utils.get_config('settings.ini') #<-- need to add this to config class so each chunk access doesn't require a read of settings file

TEMPLATE_DIR = os.path.abspath(settings.get('app','templates_location'))
STATIC_DIR = os.path.abspath(settings.get('app','static_location'))

app = flask.Flask(__name__,template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["DEBUG"] = settings.getboolean('app','debug')

# cors = CORS(app, resources={r"/api/ng/*": {"origins": "*"}})
# app.config['CORS_HEADERS'] = 'Content-Type'

## Import auth routes
from auth import setup_auth
app,login_manager = setup_auth(app)

## Initiate browser functionality
from fs_browse import initiate_browseable
app = initiate_browseable(app,config)

from ome_zarr import setup_omezarr
app = setup_omezarr(app,config)

from coordination_endpoints import inititate
app = inititate(app, config)

# ## Initiate development_browser functionality
# from fs_browse_new import initiate_browseable
# app = initiate_browseable(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html',
                           user=auth.user_info(),
                           app_name=settings.get('app','name'),
                           page_name='Home',
                           gtag=settings.get('GA4','gtag'))
#     return '''<h1>Brain Image Library Archive API</h1>
# <p>A prototype API for chunked loading of large Brain Image Library datasets.</p>'''

##############################################################################

@app.route('/api/available-datasets', methods=['GET'])
def datasets():
    return json.dumps(dataset_info())

##############################################################################

# Return descriptive information about a given dataset
@app.route('/api/metadata', methods=['GET'])
def api_meta():
    print(request.args)
    if 'id' in request.args:
        # dsetNum = int(request.args['id'])
        # dsetPath = dataset_info()[dsetNum][1]
        dsetPath = request.args['id']
        print(dsetPath)
        print(os.path.split(dsetPath)[1])
        
        if os.path.splitext(dsetPath)[1] == '.ims':
            z = ims.ims(dsetPath)
        elif os.path.splitext(dsetPath)[1] == '.zarr':
            z = zarrLoader.zarrSeries(dsetPath)
        elif '.omezarr' in os.path.split(dsetPath)[-1]:
            z = ome_zarr_loader(dsetPath)
        elif os.path.splitext(dsetPath)[1] == '.z_sharded':
            z = zarr_zip_sharded_loader.zarr_zip_sharded(dsetPath)
        elif os.path.exists(os.path.join(dsetPath,'weave.json')):
            z = weave_read(dsetPath)
            z.metaData = z.meta
        else:
            print('API can currently only load Zarr, IMS and z_sharded datasets')
            return
            
        metadata = utils.metaDataExtraction(z,strKey=True)
        
        return json.dumps(metadata)
    else:
        return "No dataset id was provided"
    

###############################################################################

# Return descriptive information about a given dataset
meta_base = '/metadata/'
@app.route(meta_base + '<path:req_path>')
@app.route(meta_base, defaults={'req_path': ''})
def meta(req_path):
    print(request.args)
    try:
        # dsetNum = int(request.args['id'])
        # dsetPath = dataset_info()[dsetNum][1]
        _, datapath = utils.get_html_split_and_associated_file_path(config,request)
        dsetPath = config.loadDataset(datapath)
        # dsetPath = request.args['id']
        print(dsetPath)
        print(os.path.split(dsetPath)[1])
        
        if os.path.splitext(dsetPath)[1] == '.ims':
            z = ims.ims(dsetPath)
        elif '.omezarr' in os.path.split(dsetPath)[-1]:
            z = ome_zarr_loader(dsetPath)
        elif os.path.splitext(dsetPath)[1] == '.zarr':
            z = zarrLoader.zarrSeries(dsetPath)
        elif os.path.splitext(dsetPath)[1] == '.z_sharded':
            z = zarr_zip_sharded_loader.zarr_zip_sharded(dsetPath)
        elif os.path.exists(os.path.join(dsetPath,'weave.json')):
            z = weave_read(dsetPath)
            z.metaData = z.meta
        else:
            print('API can currently only load Zarr, IMS and z_sharded datasets')
            return
            
        metadata = utils.metaDataExtraction(z,strKey=True)
        
        return json.dumps(metadata)
    except Exception:
        return "Dataset was not found"
    

# A route to return specific dataset chunks.
@app.route('/api/arrayCompress', methods=['GET'])
def fmostCompress():
    '''
    Retrieve a slice: resolutionLevel, (t,c,z,y,x) specified with argments as int or None

    tstart,tstop,tstep
    cstart,cstop,cstep
    zstart,zstop,zstep
    ystart,ystop,ystep
    xstart,xstop,xstep

    Returns
    -------
    Bytestring of compresed numpy array

    '''

    requiredParam = (
        'dset',
        'res',
        'tstart','tstop','tstep',
        'cstart','cstop','cstep',
        'zstart','zstop','zstep',
        'ystart','ystop','ystep',
        'xstart','xstop','xstep'
        )

    print(request.args)
    if all((x in request.args for x in requiredParam)):
        pass
    else:
        return 'A required data set, resolution level or (t,c,z,y,x) start/stop/step parameter is missing'

    # for x in request.args:
    #     print(x)
    # print(jsonify(request.args))
    intArgs = {}
    for x in request.args:
        # print(x)
        if x == 'dset':
            # print(x + ' in dset')
            intArgs[x] = request.args[x]
        else:
            intArgs[x] = ast.literal_eval(request.args[x])
    # print(intArgs)



    # dataPath = dataset_info()[intArgs['dset']][1]
    datapath = config.loadDataset(intArgs['dset'])

    # if os.path.splitext(dataPath)[1] == '.ims':
    #     z = ims.ims(dataPath)

    out = grabArray(datapath,intArgs)


    # out = z[intArgs['res'],tslice,cslice,zslice,yslice,xslice]
    # print(out.max())
    # out = np.zeros((5,5,5))
    if config.cache is not None:
        out,_,_ = compress_np(out)
    else:
        out,_,_ = utils.compress_np(out)
        
    return Response(response=out, status=200,
                    mimetype="application/octet_stream")



##############################################################################

requiredParam = (
    'res',
    'tstart','tstop','tstep',
    'cstart','cstop','cstep',
    'zstart','zstop','zstep',
    'ystart','ystop','ystep',
    'xstart','xstop','xstep'
    )

# A route to return specific dataset chunks.
# array_base = '/array/'
# @app.route(array_base + '<path:req_path>')
# @app.route(array_base, defaults={'req_path': ''})
def get_compressed_array(req_path):
    '''
    Retrieve a slice: resolutionLevel, (t,c,z,y,x) specified with argments as int or None
    
    tstart,tstop,tstep
    cstart,cstop,cstep
    zstart,zstop,zstep
    ystart,ystop,ystep
    xstart,xstop,xstep
    
    The datasets to be retrieved is specified by the path passed after 'array_base'
    This will be equivilant to url_for('fs_browse') but replaced with 'array_bas')
    Returns
    -------
    Bytestring of compresed numpy array

    '''
    
    print(request.path)
    print(request.args)
    if all((x in request.args for x in requiredParam)):
        pass
    else:
        return 'A required data set, resolution level or (t,c,z,y,x) start/stop/step parameter is missing'
    
    intArgs = {}
    for x in request.args:
        intArgs[x] = ast.literal_eval(request.args[x])
    # print(intArgs)
    
    
    
    # dataPath = dataset_info()[intArgs['dset']][1]
    # datapath = config.loadDataset(intArgs['dset'])
    _, datapath = utils.get_html_split_and_associated_file_path(config,request)
    datapath = config.loadDataset(datapath)
    # print(datapath)
    
    # if os.path.splitext(dataPath)[1] == '.ims':
    #     z = ims.ims(dataPath)
    
    out = grabArray(datapath,intArgs)
    
    # out = z[intArgs['res'],tslice,cslice,zslice,yslice,xslice]
    # print(out.max())
    # out = np.zeros((5,5,5))
    if config.cache is not None:
        out,_,_ = compress_np(out)
    else:
        out,_,_ = utils.compress_np(out)
        
    return Response(response=out, status=200,
                    mimetype="application/octet_stream")



##############################################################################




##############################################################################

def makesSlices(intArgs):
    tslice = slice(intArgs['tstart'],intArgs['tstop'],intArgs['tstep'])
    cslice = slice(intArgs['cstart'],intArgs['cstop'],intArgs['cstep'])
    zslice = slice(intArgs['zstart'],intArgs['zstop'],intArgs['zstep'])
    yslice = slice(intArgs['ystart'],intArgs['ystop'],intArgs['ystep'])
    xslice = slice(intArgs['xstart'],intArgs['xstop'],intArgs['xstep'])
    
    return tslice, cslice, zslice, yslice, xslice

##############################################################################

def grabArray(datapath,intArgs):
    '''
    intArgs = eval(intArgs) was used to deal with lru_cache which did not 
    like tuples as arguments.  However, diskcache.memorize() works fine.
    '''
    t,c,z,y,x = makesSlices(intArgs)
    out = config.opendata[datapath][intArgs['res'],t,c,z,y,x]
    return out

###############################################################################
# ## INIT FlaskCache
# from flask_caching import Cache
# fcache = Cache(config={'CACHE_TYPE': 'SimpleCache','CACHE_THRESHOLD': 1000000})
# config.fcache.init_app(app)

print('Importing Neuroglancer endpoints')
import neuroGlancer
app = neuroGlancer.setup_neuroglancer(app, config)



print(config.cache)
if config.cache is not None:
    @config.cache.memoize()
    def compress_np(np_array):
        print('Reading from disk')
        return utils.compress_np(np_array)
    # utils.compress_np = config.cache.memoize()(utils.compress_np)
    # get_compressed_array = config.cache.memoize()(get_compressed_array)
    # fmostCompress = config.cache.memoize()(fmostCompress)
    print(config.cache)

array_base = '/array/'
get_compressed_array = app.route(array_base + '<path:req_path>')(get_compressed_array)
get_compressed_array = app.route(array_base, defaults={'req_path': ''})(get_compressed_array)

@app.after_request
def add_header(response):
    '''
    Add cache-control headers to all responses to reduce burden on server
    Changing seconds object will determine how long the response is valid
    '''
    seconds = 600
    then = datetime.now() + timedelta(seconds=seconds)
    response.headers.add('Expires', then.strftime("%a, %d %b %Y %H:%M:%S GMT"))
    response.headers.add('Cache-Control', 'public,max-age=%d' % seconds)
    return response

# @app.route('/test/route/**/', methods=['GET'])
# def fmostCompress():


# CORS(app)

# if __name__ == '__main__':
#     app.run(threaded=True,host='0.0.0.0')
    
if __name__ == '__main__':
    if os.name == 'nt':
        app.run(host='0.0.0.0',port=5000)
    elif 'c00' in os.uname()[1]:
        app.run(host='0.0.0.0',port=5001)
    else:
        app.run(host='0.0.0.0',port=5000)

    


    
    
    
    # Request parameters
    # path             /foo/page.html
    # full_path        /foo/page.html?x=y
    # script_root      /myapplication
    # base_url         http://www.example.com/myapplication/foo/page.html
    # url              http://www.example.com/myapplication/foo/page.html?x=y
    # url_root         http://www.example.com/myapplication/
    
    
    
    
