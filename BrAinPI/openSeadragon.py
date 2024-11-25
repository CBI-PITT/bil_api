import utils
import os
from flask import (
    render_template,
    request,
    redirect,
    jsonify,
    Response,
)
from PIL import Image
import io
from flask_cors import cross_origin

import hashlib
from logger_tools import logger
import cv2
import re
import json

def openseadragon_dtypes():
    return [".tif", ".tiff", ".ome.tif", ".ome.tiff", ".ome-tif", ".ome-tiff", ".jp2"]


def calculate_hash(input_string):
    # Calculate the SHA-256 hash of the input string
    hash_result = hashlib.sha256(input_string.encode()).hexdigest()
    return hash_result


openSeadragonPath = "/osd/"


def setup_openseadragon(app, config):
    allowed_file_size_gb = int(
        config.settings.get("tif_loader", "pyramids_images_allowed_generation_size_gb")
    )
    allowed_file_size_byte = allowed_file_size_gb * 1024 * 1024 * 1024
    # print('allowed_file_size',allowed_file_size_byte)
    file_pattern = "[0-9]+_[0-9]+-[0-9]+_[0-9]+-[0-9]+"
    get_html_split_and_associated_file_path = (
        utils.get_html_split_and_associated_file_path
    )
    match = re.match
    Match_class = re.Match
    @logger.catch
    def openseadragon_entry(req_path):
        path_split, datapath = get_html_split_and_associated_file_path(config, request)
        # if isinstance(match(file_pattern, path_split[-1]), Match_class):
        #     datapath = os.path.split(datapath)[0]
        #     datapath = os.path.split(datapath)[0]
        logger.trace(req_path)
        # logger.info(f'{path_split},{datapath}')

        if (
            utils.split_html(datapath)[-1]
            .lower()
            .endswith(tuple(openseadragon_dtypes()))
        ):

            datapath_split = datapath.split("/")
            file_name = datapath_split[-1]
            view_path = request.path + "/osd_view"

            return render_template(
                "file_loading.html",
                gtag=config.settings.get("GA4", "gtag"),
                redirect_url=view_path,
                redirect_name="OpenSeadragon",
                description=datapath,
                file_name=file_name,
            )

        elif utils.split_html(datapath)[-1].endswith("osd_view"):
            # path_split_list = list(path_split)
            # path_split_list.remove("osd_view")
            # path_split_tuple = tuple(path_split_list)
            try:
                path_split = tuple(part for part in path_split if part != "osd_view")
                datapath = datapath.replace("/osd_view", "")
                stat = os.stat(datapath)
                file_ino = str(stat.st_ino)
                modification_time = str(stat.st_mtime)
                datapath_key = config.loadDataset(
                    file_ino + modification_time, datapath
                )
                img_obj = config.opendata[datapath_key]
                #   further check if the file has been deleted during server runing
                #   mainly used for the generated pyramid images
                if not os.path.exists(img_obj.metadata.get('datapath')):
                    logger.info("may delete")
                    del config.opendata[file_ino + modification_time]
                    datapath_key = config.loadDataset(
                        file_ino + modification_time, datapath
                    )
                    img_obj = config.opendata[datapath_key]
                # logger.info(img_obj.metadata.get('datapath'))
                return render_template(
                    "openseadragon_temp.html",
                    height=int(img_obj.metadata.get('shape')[-2]),
                    width=int(img_obj.metadata.get('shape')[-1]),
                    tileSize=int(img_obj.metadata.get('chunks')[0]),
                    # tileWidth= int(img_obj.tile_size[1]),  
                    # tileHeight= int(img_obj.tile_size[0]), 
                    host=config.settings.get("app", "url"),
                    parent_url="/".join(path_split),
                    t_point=img_obj.metadata.get('TimePoints'),
                    channel=img_obj.metadata.get('Channels'),
                    z_stack=img_obj.metadata.get('shape')[-3],
                )
            except Exception as e:
                logger.error(f'{datapath}: {e}')
                return render_template(
                    "file_exception.html",
                    gtag=config.settings.get("GA4", "gtag"),
                    exception=e,
                )

        # elif utils.split_html(datapath)[-1].endswith("png"):
        elif isinstance(match(file_pattern, path_split[-1]), Match_class):
            # return 'break point'
            datapath_split = datapath.split("/")
            # The actual path excluded the r-t-c-z-y-x parameters
            datapath = "/" + os.path.join(*datapath_split[:-4])
            stat = os.stat(datapath)
            file_ino = str(stat.st_ino)
            modification_time = str(stat.st_mtime)
            datapath_key = config.loadDataset(file_ino + modification_time, datapath)
            # print('datapath', datapath)

            img_obj = config.opendata[datapath_key]

            # key = datapath_split[-7:-1]
            key = datapath_split[-4:]
            r = int(key[0])
            t = int(key[1])
            c = int(key[2])
            z, y, x = key[3].split("_")
            z = int(z)
            y = y.split("-")
            x = x.split("-")
            y = [int(x) for x in y]
            x = [int(x) for x in x]
            # return get_slice(tif_obj,key)
            img = None
            if config.cache is not None:
                # print("cache not none")
                cache_key = f"osd_{file_ino + modification_time}-{r}-{t}-{c}-{z}-{y}-{x}"
                img = config.cache.get(cache_key, default=None, retry=True)
                if img is not None:
                    logger.info("cached chunck found")
            if img is None:
                chunk = img_obj[r,
                                slice(t,t+1),
                                slice(c,c+1),
                                slice(z,z+1),
                                slice(y[0],y[1]),
                                slice(x[0],x[1]),
                                ]
                logger.info("chunck returned from disk")
                logger.info(chunk.shape)
                if len(chunk.shape) == 3 and chunk.shape[2] == 3:  # Color image
                    chunk = cv2.cvtColor(chunk, cv2.COLOR_RGB2BGR)

                image_stream = io.BytesIO()

                # Encode the image as PNG and write it to the in-memory byte stream
                success, encoded_image = cv2.imencode(".png", chunk)
                if not success:
                    raise RuntimeError("Failed to encode image as PNG")

                image_stream.write(encoded_image.tobytes())

                # Seek to the beginning of the stream (important)
                image_stream.seek(0)
                img = image_stream
                # img = image_stream
                # img.seek(0)

                if config.cache is not None:
                    config.cache.set(
                        cache_key, img, expire=None, tag=datapath, retry=True
                    )
                    logger.info("chunk has been saved")
            return Response(img, mimetype="image/png")
        elif utils.split_html(datapath)[-1].endswith("info"):
            try:
                datapath = datapath.replace("/info", "")
                # print(datapath)

                # stat = os.stat(datapath)
                # file_ino = str(stat.st_ino)
                # modification_time = str(stat.st_mtime)
                # datapath_key = str(config.loadDataset(file_ino + modification_time, datapath))
                # tif_obj = config.opendata[datapath_key]
                stat = os.stat(datapath)
                file_ino = str(stat.st_ino)
                modification_time = str(stat.st_mtime)
                datapath_key = config.loadDataset(file_ino + modification_time, datapath)
                # print('datapath', datapath)

                img_obj = config.opendata[datapath_key]
                # file_precheck_info = tif_file_precheck(datapath)
                # meta_data_info = file_precheck_info.metaData
                # # print(asizeof.asizeof(file_precheck_info))
                # del file_precheck_info
                # gc.collect()
                json_serializable_metadata = {
                    str(key): value for key, value in img_obj.metadata.items()
                }
                return json.dumps(json_serializable_metadata, indent=4)
                # return json.dumps(img_obj.metadata)
            except Exception as e:
                logger.error(e)
                return render_template(
                    "file_exception.html",
                    gtag=config.settings.get("GA4", "gtag"),
                    exception=e,
                )
        else:
            return "No end point recognized!"

    openseadragon_entry = cross_origin(allow_headers=["Content-Type"])(
        openseadragon_entry
    )
    openseadragon_entry = app.route(openSeadragonPath + "<path:req_path>")(
        openseadragon_entry
    )
    # Not sure if it should be included or not
    openseadragon_entry = app.route(openSeadragonPath, defaults={"req_path": ""})(
        openseadragon_entry
    )
