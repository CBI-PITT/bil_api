# -*- coding: utf-8 -*-
"""
Created on Wed Mar  22 11:12:07 2023

@author: alpha

Endpoints/functions placed here are to help coordinate applications both internally/externally to interacting
with the BrAinPI interface.

For example, the path_to_html_options function allows a dictionary of links to be returned for any file on disk.
This function may be used internally to generate links for other services; however,
the /path_to_html_options/ endpoint allows user to query a 'path' and return a json of the output from path_to_html_options
"""
import os, glob
from flask import (
    render_template,
    request,
    send_file,
    redirect,
    jsonify,
    url_for
    )
from flask_cors import cross_origin

# from urllib.parse import quote
from logger_tools import logger
from file_type_support import ng_links,opsd_links
from neuroGlancer import neuroglancer_dtypes
from openSeadragon import openseadragon_dtypes
from utils import (from_path_to_html,
                   get_path_map,
                   dict_key_value_match,
                   from_path_to_browser_html,
                   strip_leading_trailing_slash,
                   fix_special_characters_in_html,
                   strip_trailing_new_line,
                   exists
                   )

def inititate(app,config):
    """
    Initialize Flask endpoints for Neuroglancer and OpenSeadragon integration.

    Args:
        app (Flask): The Flask application instance.
        config (object): The configuration object containing application settings.

    Endpoints:
        - `/ng_supported_filetypes/`: Returns a list of supported file types for Neuroglancer.
        - `/opsd_supported_filetypes/`: Returns a list of supported file types for OpenSeadragon.
        - `/path_to_html_options/`: Converts a file system path to a set of HTML links.
        - `/curated_datasets/`: Retrieves curated dataset collections and their corresponding links.

    Returns:
        Flask: The Flask application with the added endpoints.
    """
    settings = config.settings
    @app.route('/ng_supported_filetypes/', methods=['GET'])
    @cross_origin(allow_headers=['Content-Type'])
    def neuroglancer_support():
        return jsonify(neuroglancer_dtypes())
    
    @app.route('/opsd_supported_filetypes/', methods=['GET'])
    @cross_origin(allow_headers=['Content-Type'])
    def opsd_support():
        return jsonify(openseadragon_dtypes())

    @app.route('/path_to_html_options/', methods=['GET'])
    @cross_origin(allow_headers=['Content-Type'])
    def html_options():
        logger.info(f"{request.remote_addr=}")
        assert(isinstance(request.args, dict)), 'Expects a dictionary'
        assert 'path' in request.args, 'Expects a path key'
        verify_file_exists = request.args.get('verify_file_exists')
        if verify_file_exists is None or verify_file_exists.lower() == 'true' or verify_file_exists.lower() == 't':
            verify_file_exists = True
        else:
            verify_file_exists = False
        return jsonify(path_to_html_options(request.args['path'],verify_file_exists=verify_file_exists))

    # paths = {}
    # paths['path'] = None

    # paths['neuroglancer'] = None
    # paths['neuroglancer_metadata'] = None

    # paths['omezarr'] = None
    # paths['omezarr_neuroglancer_optimized'] = None
    # paths['omezarr_8bit'] = None
    # paths['omezarr_8bit_neuroglancer_optimized'] = None
    # paths['omezarr_metadata'] = None
    # paths['omezarr_8bit_metadata'] = None
    # paths['omezarr_validator'] = None
    # paths['omezarr_8bit_validator'] = None
    # paths['omezarr_neuroglancer_optimized'] = None
    # paths['omezarr_8bit_neuroglancer_optimized_validator'] = None

    def path_to_html_options(path, verify_file_exists=True):
        """
        Takes the path (on disk) to a potential BrAinPI compatible dataset and return all possible links in a dictionary.
        Convert a file system path into a dictionary of possible HTML links.

        Args:
            path (str): The file system path to a BrAinPI-compatible dataset.
            verify_file_exists (bool, optional): Whether to check if the file exists. Defaults to True.

        Returns:
            dict: A dictionary containing HTML links for various visualization services and metadata.
        """
        path_map = get_path_map(settings,
                                user_authenticated=True)  # <-- Force user_auth=True to get all possible paths, in this way all ng links will be shareable to anyone
        key = path
        if len(key) > 1 and key[-1] == '/':
            key = key[:-1]

        if len(key.strip()) == 0 or len(key.strip()) == '/':
            key = '/'

        paths = {}
        paths['path'] = key

        paths['neuroglancer'] = None
        paths['neuroglancer_metadata'] = None
        
        paths['omezarr'] = None
        paths['omezarr_neuroglancer_optimized'] = None
        paths['omezarr_8bit'] = None
        paths['omezarr_8bit_neuroglancer_optimized'] = None
        paths['omezarr_metadata'] = None
        paths['omezarr_8bit_metadata'] = None
        paths['omezarr_validator'] = None
        paths['omezarr_8bit_validator'] = None
        paths['omezarr_neuroglancer_optimized'] = None
        paths['omezarr_8bit_neuroglancer_optimized_validator'] = None
        paths['openseadragon'] = None
        paths['openseadragon_metadata'] = None

        logger.info(f"{paths['path']=}")
        if verify_file_exists:
            if not exists(paths['path']):
                return paths
        # if not os.path.exists(paths['path']):
        #     return paths

        html_base = settings.get('app','url')
        html_base = strip_leading_trailing_slash(html_base)
        logger.info(f"{html_base=}")
        html_path = from_path_to_browser_html(paths['path'],path_map, html_base)

        if html_path is not None:
            req_path = html_path.replace(html_base, '')
            logger.info(f"{req_path=}")
            ng_link = ng_links('/' + req_path)
            opsd_link = opsd_links('/' + req_path)
            logger.info(f"{ng_link=}")
            logger.info(f"{opsd_link=}")
            if ng_link is not None:
                validator_url = 'https://ome.github.io/ome-ngff-validator'
                validator_url = validator_url + '/?source='

                ng_link = html_base + '/' + strip_leading_trailing_slash(ng_link)

                paths['neuroglancer'] = ng_link
                paths['neuroglancer_metadata'] = paths['neuroglancer'] + '/info'

                omezarr_entry = paths['neuroglancer'].replace(url_for('neuro_glancer_entry'),url_for('omezarr_entry'))

                paths['omezarr'] = omezarr_entry + '.ome.zarr'
                paths['omezarr_neuroglancer_optimized'] = omezarr_entry + '.ng.ome.zarr'
                paths['omezarr_8bit'] = omezarr_entry + '.8bit.ome.zarr'
                paths['omezarr_8bit_neuroglancer_optimized'] = omezarr_entry + '.ng.8bit.ome.zarr'
                paths['omezarr_metadata'] = omezarr_entry + '.ome.zarr/.zattrs'
                paths['omezarr_8bit_metadata'] = omezarr_entry + '.8bit.ome.zarr/.zattrs'
                paths['omezarr_validator'] = validator_url + omezarr_entry + '.ome.zarr'
                paths['omezarr_8bit_validator'] = validator_url + omezarr_entry + '.8bit.ome.zarr'
                paths['omezarr_neuroglancer_optimized_validator'] = validator_url + omezarr_entry + '.ng.ome.zarr'
                paths['omezarr_8bit_neuroglancer_optimized_validator'] = validator_url + omezarr_entry + '.ng.8bit.ome.zarr'

                # Replace values that don't translate directly to url
                # Probably others that are missing
                # for key,value in paths.items():
                #     # Replace space with %20 (' ')
                #     if value is not None and value.startswith('http'):
                #         paths[key] = fix_special_characters_in_html(value)
            if opsd_link is not None:
                opsd_link = html_base + '/' + strip_leading_trailing_slash(opsd_link)

                paths['openseadragon'] = opsd_link
                paths['openseadragon_metadata'] = paths['openseadragon'] + '/info'
            for key,value in paths.items():
                # Replace space with %20 (' ')
                if value is not None and value.startswith('http'):
                    paths[key] = fix_special_characters_in_html(value)
        return paths

    # @app.route('/curated_datasets/', methods=['GET'])
    # @cross_origin(allow_headers=['Content-Type'])
    # def curated_datasets_3():
    #
    #     html_base = settings.get('app', 'url')
    #     html_base = strip_leading_trailing_slash(html_base)
    #
    #     html_options_url = url_for("html_options")
    #
    #     # Locations are directories which contain files or files which have each line pointing to a dataset on disk
    #     locations = settings['curated_datasets']
    #     locations = dict(locations)
    #
    #     datasets = {'collections': {}}
    #     for set_name, file in locations.items():
    #         datasets['collections'][set_name] = []
    #         with open(file, 'r') as f:
    #             for line in f.readlines():
    #                 l = strip_trailing_new_line(line)
    #                 query_to_path_to_html_options = f'{html_base}{html_options_url}?path={l}'
    #                 name = os.path.split(line)[-1]
    #                 dataset = {
    #                     'name':strip_trailing_new_line(name),
    #                     'links':fix_special_characters_in_html(query_to_path_to_html_options)
    #                 }
    #                 datasets['collections'][set_name].append(dataset)
    #
    #         # from pprint import pprint as print
    #         print(datasets)
    #
    #         # Cache curated_datasets in the config object (doesn't allow for dynamic updates) but is better performance
    #         # Commenting below turns off caching in the config object so each time the files are reloaded
    #         # (allows for dynamic updates to curated datasets)
    #         # config.curated_datasets = datasets
    #
    #     return jsonify(datasets)

    @app.route('/curated_datasets/', methods=['GET'])
    @cross_origin(allow_headers=['Content-Type'])
    def curated_datasets_4():
        """
        Retrieve a collection of curated datasets and their links.

        The datasets are grouped by type and include links to their respective 
        HTML representations or metadata.

        Returns:
            Flask.Response: A JSON object containing the curated datasets.
        """

        html_base = settings.get('app', 'url')
        html_base = strip_leading_trailing_slash(html_base)

        html_options_url = url_for("html_options")

        # Locations are directories which contain files or files which have each line pointing to a dataset on disk
        locations = settings['curated_datasets']
        locations = dict(locations)

        datasets = {'collections': []}
        for set_name, file in locations.items():
            current_collection = {'type':set_name}
            with open(file, 'r') as f:
                details = []
                for line in f.readlines():
                    l = strip_trailing_new_line(line)
                    query_to_path_to_html_options = f'{html_base}{html_options_url}?path={l}&verify_file_exists=False'
                    name = os.path.split(line)[-1]
                    dataset = {
                        'name': strip_trailing_new_line(name),
                        'links': fix_special_characters_in_html(query_to_path_to_html_options)
                    }
                    details.append(dataset)
                    # datasets['collections'][set_name].append(dataset)
            current_collection['details'] = details
            logger.info(f'Sending {len(details)} entries for set {set_name}')
            datasets['collections'].append(current_collection)
            # from pprint import pprint as print
            # print(datasets)

            # Cache curated_datasets in the config object (doesn't allow for dynamic updates) but is better performance
            # Commenting below turns off caching in the config object so each time the files are reloaded
            # (allows for dynamic updates to curated datasets)
            # config.curated_datasets = datasets

        return jsonify(datasets)

    return app