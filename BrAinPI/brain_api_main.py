# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:06:07 2021

@author: alpha

https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
To run w/ gunicorn:  gunicorn --worker-class gevent -b 0.0.0.0:5001 --chdir /CBI_FastStore/cbiPythonTools/BrAinPI/BrAinPI wsgi:app -w 24 --threads=2 --worker-connections=2
To run development (1 worker, 1 thread):  python -i /CBI_FastStore/cbiPythonTools/BrAinPI/BrAinPI/BrAinPI.py
"""

# Place some critical caching tool into the builtins space to enable access across modules
import builtins
from cache_tools import cache_head_space
builtins.brainpi_cache_ram = cache_head_space(10) #Make available accross all modules as simply calling 'cache_ram'

import flask, os
from flask import request, render_template
import pathlib

# Change working directory to parent of this file
cwd = pathlib.Path(__file__).parent
os.chdir(cwd)

## Project imports
import config_tools
import auth
from utils import compress_flask_response



## Grab settings information from config.ini file
settings = config_tools.get_config('settings.ini')

# ## Setup cache location based on OS type
# ## Optional situations like machine name can be used to customize
# if os.name == 'nt':
#     cacheLocation = settings.get('disk_cache','location_win')
# else:
#     cacheLocation = settings.get('disk_cache','location_unix')
#     #cacheLocation = None
#
# # Instantiate class that will manage all open datasets
# # This will remain in the global env and be accessed by multiple route methods
# config = config_tools.config(
#     cacheLocation=cacheLocation,
#     cacheSizeGB=settings.getint('disk_cache','cacheSizeGB'),
#     evictionPolicy=settings.get('disk_cache','evictionPolicy'),
#     shards=settings.getint('disk_cache','shards'),
#     timeout=settings.getfloat('disk_cache','timeout')
#     )
config = config_tools.config()
# Read settings.ini file and append to the config class
# config.settings = config_tools.get_config('settings.ini') #<-- need to add this to config class so each chunk access doesn't require a read of settings file

# Establish constants based on settings.ini
TEMPLATE_DIR = os.path.abspath(settings.get('app','templates_location'))
STATIC_DIR = os.path.abspath(settings.get('app','static_location'))
LOGO = settings.get('app','logo') #Relative to STATIC_DIR
APP_NAME = settings.get('app','name')


# Establish FLASK app
app = flask.Flask(__name__,template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["DEBUG"] = settings.getboolean('app','debug')

## Initiate endpoints in other modules and attach them to app
## New endpoints can be added here

browser_active = settings.getboolean('browser','browser_active')

# Must init auth first to get login_manager
if browser_active:
    print('Initiating auth functionality')
    from auth import setup_auth
    app,login_manager = setup_auth(app)
else:
    from auth import setup_NO_auth
    app, login_manager = setup_NO_auth(app)

if browser_active:
    print('Initiating browser functionality')
    from fs_browse import initiate_browseable
    app = initiate_browseable(app,config)
else:
    from fs_browse import initiate_NOT_browseable
    app = initiate_NOT_browseable(app, config)

print('Initiating OME Zarr endpoints')
from ome_zarr_ep import setup_omezarr
app = setup_omezarr(app,config)

print('Importing Neuroglancer endpoints')
import neuroGlancer
app = neuroGlancer.setup_neuroglancer(app, config)

print('Importing Openseadragon endpoints')
import openSeadragon
openSeadragon.setup_openseadragon(app, config)

print('Initiating Coordination endpoints')
from coordination_endpoints import inititate
app = inititate(app, config)

print('Initiating Coordination endpoints')
from volume_extractor_endpoints import setup_extractor_endpoint
app = setup_extractor_endpoint(app, config)

print('Initiating Tokenized endpoints')
from tokenized_urls import setup_tokenized_endpoint
app = setup_tokenized_endpoint(app, config)


print('Initiating Landing Zone')
@app.route('/', methods=['GET'])
def home():
    """
    Landing zone for BrainPi
    """
    return render_template('home.html',
                           browser_active=browser_active,
                           user=auth.user_info(),
                           app_name=settings.get('app','name'),
                           app_description=settings.get('app','description'),
                           app_motto=settings.get('app','motto'),
                           app_logo=LOGO,
                           page_name='Home',
                           gtag=settings.get('GA4','gtag'))

##############################################################################




@app.after_request
def add_header(response):
    '''
    Add cache-control headers to all responses to reduce burden on server
    Changing seconds object will determine how long the response is valid
    '''
    seconds = 864000 # 10 days
    # then = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    # response.headers.add('Expires', then.strftime("%a, %d %b %Y %H:%M:%S GMT"))
    content_type = response.headers.get('Content-Type')
    if 'octet_stream' in content_type:
        '''
        Cache but don't gzip
        Gzip can be handled by the specific process sending these data
        Neuroglancer data is not compressed natively so it is enabled
        However, ome-zarr is compressed, so gzip would not be helpful
        '''
        response.headers.add('Cache-Control', f'public,max-age={seconds}')

    elif 'application/json' in content_type or 'html' in content_type:
        '''
        ################################################
        ## ADD HEADERS TO DISABLE CACHE ON JSON/HTML  ##
        ## In general these docs may update often, so ##
        ## caching may break the program              ##
        ################################################
        Cache-Control: no-cache, no-store, must-revalidate
        Pragma: no-cache
        Expires: 0
        '''
        response.headers.add('Cache-Control', 'no-cache,no-store,must-revalidate,max-age=0')
        response.headers.add('Pragma', 'no-cache')
        response.headers.add('Expires', '0')

        response = compress_flask_response(response, request, 3)

    else:
        # Everything else is cached
        response.headers.add('Cache-Control', f'public,max-age={seconds}')
        response = compress_flask_response(response, request, 3)

    if APP_NAME:
        response.headers.add('X-Service', APP_NAME)
    else:
        response.headers.add('X-Service', 'BrAinPI')

    return response


# Enable system to be run in development mode with debug (auto reload) on by running this scripy directly
if __name__ == '__main__':
    if os.name == 'nt':
        app.run(host='0.0.0.0',port=5001,debug=True)
    else:
        app.run(host='0.0.0.0',port=5001,debug=True)


# Request parameters notes
# path             /foo/page.html
# full_path        /foo/page.html?x=y
# script_root      /myapplication
# base_url         http://www.example.com/myapplication/foo/page.html
# url              http://www.example.com/myapplication/foo/page.html?x=y
# url_root         http://www.example.com/myapplication/
    
    
    
    
