import tifffile
import math
import zarr
import os
import time
import hashlib
from flask import render_template
from filelock import FileLock
from pathlib import Path
import shutil
import gc
from logger_tools import logger
import itertools
import numpy as np
from utils import calculate_hash, get_directory_size, delete_oldest_files
# def calculate_hash(input_string):
#     # Calculate the SHA-256 hash of the input string
#     hash_result = hashlib.sha256(input_string.encode()).hexdigest()
#     return hash_result


# def get_directory_size(directory):
#     total_size = 0
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for f in filenames:
#             fp = os.path.join(dirpath, f)
#             total_size += os.path.getsize(fp)
#     return total_size


# def delete_oldest_files(directory, size_limit):
#     items = sorted(Path(directory).glob("*"), key=os.path.getctime)
#     total_size = get_directory_size(directory)

#     # Delete oldest items until the total size is within the size limit
#     for item in items:
#         if total_size <= size_limit:
#             break
#         if item.is_file():
#             item_size = os.path.getsize(item)
#             os.remove(item)
#             total_size -= item_size
#             logger.success(f"Deleted file {item} of size {item_size} bytes")
#         elif item.is_dir():
#             dir_size = get_directory_size(item)
#             shutil.rmtree(item)
#             total_size -= dir_size
#             logger.success(f"Deleted directory {item} of size {dir_size} bytes")


class tiff_loader:
    """
    A class to load, validate, and process TIFF image files.

    This class supports pyramid image generation, metadata extraction, and multi-resolution
    slicing of TIFF images. It integrates with caching systems to improve performance and
    supports handling large datasets.
    """
    def __init__(
        self, file_location, pyramid_images_connection, cache, settings, ResolutionLevelLock = None, squeeze = True
    ) -> None:
        """
        Initialize the tiff_loader object.

        Args:
            file_location (str): Path to the TIFF file.
            pyramid_images_connection (dict): Mapping of hash values to pyramid images.
            cache (object, optional): Cache object for storing slices. Defaults to None.
            settings (configparser.ConfigParser): Configuration settings.
            ResolutionLevelLock (int, optional): Resolution level lock. Defaults to None.
            squeeze (bool, optional): Whether to squeeze singleton dimensions. Defaults to True.
        """
        # logger.info('pyramid_images_connection',pyramid_images_connection)
        self.cache = cache
        self.squeeze = squeeze
        self.settings = settings
        self.datapath = file_location
        self.location = file_location
        self.metaData = {}
        self.file_stat = os.stat(file_location)
        self.file_ino = str(self.file_stat.st_ino)
        self.modification_time = str(self.file_stat.st_mtime)
        self.file_size = self.file_stat.st_size
        self.allowed_store_size_gb = float(
            self.settings.get("tif_loader", "pyramids_images_allowed_store_size_gb")
        )
        self.allowed_store_size_byte = self.allowed_store_size_gb * 1024 * 1024 * 1024
        self.allowed_file_size_gb = float(
            self.settings.get("tif_loader", "pyramids_images_allowed_generation_size_gb")
        )
        self.allowed_file_size_byte = self.allowed_file_size_gb * 1024 * 1024 * 1024
        self.ResolutionLevelLock = 0 if ResolutionLevelLock is None else ResolutionLevelLock
        self.image = self.validate_tif_file(self.datapath)
        self.filename, self.filename_extension = self.file_extension_split()
        self.tags = self.image.pages[0].tags
        self.photometric = self.image.pages[0].photometric
        self.compression = self.image.pages[0].compression
        # Extract width and height
        self.height = self.tags["ImageLength"].value
        self.width = self.tags["ImageWidth"].value
        # logger.info(self.width,self.height)
        self.dataType = self.image.pages[0].dtype
        self.metaData["dataType"] = self.dataType
        # logger.info(self.dataType)
        self.series = len(self.image.series)
        self.is_pyramidal = self.image.series[0].is_pyramidal
        # logger.info("series", self.series)
        # logger.info("levels", len(self.image.series[0].levels))
        if self.image.pages[0].is_tiled:
            # Get the tile size
            self.tile_size = (
                self.image.pages[0].tilelength,
                self.image.pages[0].tilewidth,
            )
        else:
            logger.info("Assigning tile size (128, 128)")
            self.tile_size = (128, 128)

        self.arrays = {}

        self.type = self.image.series[0].axes
        self.axes_pos_dic = self.axes_pos_extract(self.type)
        self.axes_value_dic = self.axes_value_extract(
            self.type, self.image.series[0].shape
        )
        self.Channels = self.axes_value_dic.get("C")
        self.z = self.axes_value_dic.get("Z")
        self.TimePoints = (
            self.axes_value_dic.get("T")
            if self.axes_value_dic.get("T") != 1
            else (
                self.axes_value_dic.get("Q")
                if self.axes_value_dic.get("Q") != 1
                else (
                    self.axes_value_dic.get("I")
                    if self.axes_value_dic.get("I") != 1
                    else 1
                )
            )
        )

        self.pyramid_dic = pyramid_images_connection
        logger.info(self.type)
        logger.info(f"axes_pos_dic, {self.axes_pos_dic}")
        logger.info(f"axes_value_dic, {self.axes_value_dic}")
        for i_s, s in enumerate(self.image.series):
            logger.info(f"Series {i_s}: {s}")
            for i_l, level in enumerate(s.levels):
                logger.info(f"Level {i_l}: {level}")
                # self.metaData[f"Series:{i_s}, Level:{i_l}"] = str(level)
        # if already pyramid image --> building the arrays
        # elif no pyramid but connection exist --> replace the location, building the arrays
        # elif no pyramid and no connection --> pyramid image generation, building connection using hash func and replace location, building arrays
        self.pyramid_validators(self.image)
        self.metaData['datapath'] = self.datapath
        self.ResolutionLevels = len(self.image.series[0].levels) if self.is_pyramidal else len(self.image.series)
        layers = self.image.series[0].levels if self.is_pyramidal else self.image.series

        for r in range(self.ResolutionLevels):
            array = layers[r]
            for t, c in itertools.product(range(self.TimePoints), range(self.Channels)):
                # print(f"shape, {array.shape}")
                
                # Collect attribute info
                shape_z = array.shape[self.axes_pos_dic.get("Z")] if self.axes_pos_dic.get("Z") is not None else 1
                shape_y = array.shape[self.axes_pos_dic.get("Y")] if self.axes_pos_dic.get("Y") is not None else 1
                shape_x = array.shape[self.axes_pos_dic.get("X")] if self.axes_pos_dic.get("X") is not None else 1
                self.metaData[r, t, c, 'shape'] = (t + 1, c + 1, shape_z, shape_y, shape_x)
                
                # Collect resolution and dataset info
                xy_resolution = array.pages[0].get_resolution()
                self.metaData[r, t, c, 'resolution'] = (1/xy_resolution[0], 1/xy_resolution[1])
                self.metaData[r, t, c, 'chunks'] = self.tile_size
                self.metaData[r, t, c, 'dtype'] = array.dtype
                self.metaData[r, t, c, 'ndim'] = array.ndim

                
                self.change_resolution_lock(self.ResolutionLevelLock)
                # logger.info(self.t)
                # del self.image
                # gc.collect()

    def change_resolution_lock(self,ResolutionLevelLock):
        """
        Change the resolution lock level and update metadata accordingly.

        Args:
            ResolutionLevelLock (int): The new resolution lock level.
        """
        self.ResolutionLevelLock = ResolutionLevelLock
        self.shape = self.metaData[self.ResolutionLevelLock,0,0,'shape']
        self.ndim = len(self.shape)
        self.chunks = self.metaData[self.ResolutionLevelLock,0,0,'chunks']
        self.resolution = self.metaData[self.ResolutionLevelLock,0,0,'resolution']
        self.dtype = self.metaData[self.ResolutionLevelLock,0,0,'dtype']


    def __getitem__(self,key):
        """
        Access a slice of the TIFF image.

        Args:
            key (tuple): slice to access.

        Returns:
            np.ndarray: The extracted image slice.
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
    def getSlice(self,r,t,c,z,y,x):
        """
        Retrieve a slice of the image at a specific resolution and dimensions.

        Args:
            r (int): Resolution level.
            t (slice): Time dimension slice.
            c (slice): Channel dimension slice.
            z (slice): Z-axis slice.
            y (slice): Y-axis slice.
            x (slice): X-axis slice.

        Returns:
            np.ndarray: The requested image slice.
        """
        list_tp = None
        if self.type.endswith("S"):
            list_tp = [0] * (len(self.type) - 1)
        else:
            list_tp = [0] * len(self.type)
        r = r

        if (
            self.axes_pos_dic.get("T") != None
            or self.axes_pos_dic.get("Q") != None
            or self.axes_pos_dic.get("I") != None
        ):
            if self.axes_pos_dic.get("T") != None:
                list_tp[self.axes_pos_dic.get("T")] = t
            elif self.axes_pos_dic.get("Q") != None:
                list_tp[self.axes_pos_dic.get("Q")] = t
            elif self.axes_pos_dic.get("I") != None:
                list_tp[self.axes_pos_dic.get("I")] = t
        if self.axes_pos_dic.get("C") != None:
            list_tp[self.axes_pos_dic.get("C")] = c
        if self.axes_pos_dic.get("Z") != None:
            list_tp[self.axes_pos_dic.get("Z")] = z
        if self.axes_pos_dic.get("Y") != None:
            y = y
            list_tp[self.axes_pos_dic.get("Y")] = y
        if self.axes_pos_dic.get("X") != None:
            x = x
            list_tp[self.axes_pos_dic.get("X")] = x
        logger.info(f'{list_tp},{self.type}')
        zarr_array = None
        if self.is_pyramidal:
            zarr_array = self.image.aszarr(series=0, level=r)
        else:
            zarr_array = self.image.aszarr(series=r, level=0)
        zarr_store = zarr.open(zarr_array)
        tp = tuple(list_tp)
        result = zarr_store[tp]

        # Here for python > 3.11, to use the unpack operator *
        # result = zarr_store[
        #     *(tp),
        #     y * tile_size_heiht : (y + 1) * tile_size_heiht,
        #     x * tile_size_width : (x + 1) * tile_size_width,
        # ]

        # numpy_array = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
        # return numpy_array

        cache_key = f"{self.file_ino + self.modification_time}-{r}-{t}-{c}-{z}-{y}-{x}"
        if self.cache is not None:
            self.cache.set(
                cache_key, result, expire=None, tag=self.datapath, retry=True
            )
        return result

    # def __getitem__(self, key):
    #     list_tp = None
    #     if self.type.endswith("S"):
    #         list_tp = [0] * (len(self.type) - 1)
    #     else:
    #         list_tp = [0] * len(self.type)
    #     r = int(key[0])

    #     if (
    #         self.axes_pos_dic.get("T") != None
    #         or self.axes_pos_dic.get("Q") != None
    #         or self.axes_pos_dic.get("I") != None
    #     ):
    #         t = int(key[1])
    #         if self.axes_pos_dic.get("T") != None:
    #             list_tp[self.axes_pos_dic.get("T")] = t
    #         elif self.axes_pos_dic.get("Q") != None:
    #             list_tp[self.axes_pos_dic.get("Q")] = t
    #         elif self.axes_pos_dic.get("I") != None:
    #             list_tp[self.axes_pos_dic.get("I")] = t
    #     if self.axes_pos_dic.get("C") != None:
    #         c = int(key[2])
    #         list_tp[self.axes_pos_dic.get("C")] = c
    #     if self.axes_pos_dic.get("Z") != None:
    #         z = int(key[3])
    #         list_tp[self.axes_pos_dic.get("Z")] = z
    #     if self.axes_pos_dic.get("Y") != None:
    #         tile_size_height = int(self.tile_size[0])
    #         y = int(key[4])
    #         list_tp[self.axes_pos_dic.get("Y")] = slice(
    #             y * tile_size_height, (y + 1) * tile_size_height
    #         )
    #     if self.axes_pos_dic.get("X") != None:
    #         tile_size_width = int(self.tile_size[1])
    #         x = int(key[5])
    #         list_tp[self.axes_pos_dic.get("X")] = slice(
    #             x * tile_size_width, (x + 1) * tile_size_width
    #         )
    #     # logger.info("tp",tp)
    #     zarr_array = None
    #     if self.is_pyramidal:
    #         zarr_array = self.image.aszarr(series=0, level=r)
    #     else:
    #         zarr_array = self.image.aszarr(series=r, level=0)
    #     zarr_store = zarr.open(zarr_array)
    #     tp = tuple(list_tp)
    #     result = zarr_store[tp]

    #     # Here for python > 3.11, to use the unpack operator *
    #     # result = zarr_store[
    #     #     *(tp),
    #     #     y * tile_size_heiht : (y + 1) * tile_size_heiht,
    #     #     x * tile_size_width : (x + 1) * tile_size_width,
    #     # ]

    #     # numpy_array = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
    #     # return numpy_array

    #     cache_key = f"{self.file_ino + self.modification_time}-{key[0]}-{key[1]}-{key[2]}-{key[3]}-{key[4]}-{key[5]}"
    #     if self.cache is not None:
    #         self.cache.set(
    #             cache_key, result, expire=None, tag=self.datapath, retry=True
    #         )
    #     return result

    def validate_tif_file(self,file_path):
        """
        Validate the provided TIFF file.

        Args:
            file_path (str): Path to the TIFF file.

        Returns:
            tifffile.TiffFile: The validated TIFF file object.
        """
        img = tifffile.TiffFile(file_path)
        return img
        # try:
        #     # Attempt to load the file
        #     img = tifffile.TiffFile(file_path)
        #     return img
        # except Exception as e:
        #     logger.error(f"File '{file_path}' is not a valid tiff file. Error: {e}")
        #     raise
        # raise Exception(f"File '{file_path}' is not a valid tiff file. Error: {e}")


    def pyramid_validators(self, tif):
        """
        Validate if the TIFF file has a valid pyramid structure. If not, trigger pyramid generation.

        Args:
            tif (tifffile.TiffFile): The TIFF file object to validate.

        Raises:
            Exception: If the file size exceeds the allowed limit.
        """
        inspector_result = self.pyramid_inspectors(tif)
        # inspector_result = False
        logger.info(f"inspector_result: {inspector_result}")
        if inspector_result:
            return
        else:
            # logger.info(f'{self.file_size},{self.allowed_file_size_byte}')
            if self.file_size > self.allowed_file_size_byte:
                logger.info(f"File '{self.filename}' can not generate pyramid structure. Due to resource constrait, {self.allowed_file_size_gb}GB and below are acceptable for generation process.")
                raise Exception(f"File '{self.filename}' can not generate pyramid structure. Due to resource constrait, {self.allowed_file_size_gb}GB and below are acceptable for generation process.")
            self.pyramid_builders(tif)
            return

    def pyramid_inspectors(self, tif):
        """
        Inspect the pyramid structure of the TIFF file.

        Args:
            tif (tifffile.TiffFile): The TIFF file object.

        Returns:
            bool: True if the pyramid structure is valid, False otherwise.
        """
        if self.is_pyramidal:
            # may need more check opeation
            return True
        else:
            for r in range(len(tif.series)):
                if r != 0:
                    x_value_cur = tif.series[r].shape[self.axes_pos_dic.get("X")]
                    x_value_pre = tif.series[r - 1].shape[self.axes_pos_dic.get("X")]
                    y_value_cur = tif.series[r].shape[self.axes_pos_dic.get("Y")]
                    y_value_pre = tif.series[r - 1].shape[self.axes_pos_dic.get("Y")]
                    if (
                        x_value_cur != x_value_pre // 2
                        or y_value_cur != y_value_pre // 2
                    ) and (
                        x_value_cur != math.ceil(x_value_pre / 2)
                        or y_value_cur != math.ceil(y_value_pre / 2)
                    ):
                        return False
                if r == len(tif.series) - 1:
                    if (
                        tif.series[r].pages[0].imagelength > self.tile_size[0]
                        or tif.series[r].pages[0].imagewidth > self.tile_size[1]
                    ):
                        return False
                    else:
                        return True
            return True

    def file_extension_split(self):
        """
        Split the file name and extension of the TIFF file.

        Returns:
            list: A list containing the base file name and extension.
        """
        file = None
        extension = None
        image_name = self.image.filename
        # logger.info(f'image name',{image_name})
        if image_name.endswith(".ome.tif"):
            extension_index = image_name.rfind(".ome.tif")
            file = image_name[:extension_index]
            extension = ".ome.tif"
        elif image_name.endswith(".ome.tiff"):
            extension_index = image_name.rfind(".ome.tiff")
            file = image_name[:extension_index]
            extension = ".ome.tiff"
        elif image_name.endswith(".ome-tiff"):
            extension_index = image_name.rfind(".ome-tiff")
            file = image_name[:extension_index]
            extension = ".ome-tiff"
        elif image_name.endswith(".ome-tif"):
            extension_index = image_name.rfind(".ome-tif")
            file = image_name[:extension_index]
            extension = ".ome-tif"
        elif image_name.endswith(".tif"):
            extension_index = image_name.rfind(".tif")
            file = image_name[:extension_index]
            extension = ".tif"
        elif image_name.endswith(".tiff"):
            extension_index = image_name.rfind(".tiff")
            file = image_name[:extension_index]
            extension = ".tiff"

        return [file, extension]

    def pyramid_builders(self, tif):
        """
        Build a pyramid structure for the TIFF file if it does not already exist.

        Args:
            tif (tifffile.TiffFile): The TIFF file object.
        """
        hash_value = calculate_hash(self.file_ino + self.modification_time)
        pyramids_images_store = self.settings.get("tif_loader", "pyramids_images_store")
        pyramids_images_store_dir = (
            pyramids_images_store + hash_value[0:2] + "/" + hash_value[2:4] + "/"
        )
        suffix = self.settings.get("tif_loader", "extension_type")
        pyramid_image_location = pyramids_images_store_dir + hash_value + suffix
        if self.pyramid_dic.get(hash_value) and os.path.exists(pyramid_image_location):
            self.datapath = self.pyramid_dic.get(hash_value)
            # self.image = tifffile.TiffFile(pyramid_image_location)
            logger.info("Location replaced by generated pyramid image")
        else:
            # Avoid other gunicore workers to build pyramids images
            if os.path.exists(pyramid_image_location):
                logger.info(
                    "Pyramid image was already built by first worker and picked up now by others"
                )
                self.pyramid_dic[hash_value] = pyramid_image_location
                self.datapath = pyramid_image_location
                # self.image = tifffile.TiffFile(pyramid_image_location)
            # 1 hash exists but the pyramid images(not loaded) are deleted during server running
            # 2 no hash and no pyramid images (first time generation)
            else:
                if tif.filename.endswith("ome.tif"):
                    # write pyramids based on ome.tif
                    self.pyramid_building_process(
                        tif.series[0].levels[0],
                        2,
                        hash_value,
                        pyramids_images_store,
                        pyramids_images_store_dir,
                        pyramid_image_location,
                    )
                elif tif.filename.endswith(".tif") or tif.filename.endswith(".tiff"):
                    # write pyramids based on tif
                    self.pyramid_building_process(
                        tif.series[0],
                        2,
                        hash_value,
                        pyramids_images_store,
                        pyramids_images_store_dir,
                        pyramid_image_location,
                    )
        self.image = self.validate_tif_file(pyramid_image_location)
        self.is_pyramidal = True

    def pyramid_building_process(
        self,
        first_series,
        factor,
        hash_value,
        pyramids_images_store,
        pyramids_images_store_dir,
        pyramid_image_location,
    ):
        """
        Generate a pyramid structure for the TIFF file.

        Args:
            first_series (tifffile.TiffPageSeries): The first series of the TIFF file.
            factor (int): The downscaling factor.
            hash_value (str): Hash value of the file.
            pyramids_images_store (str): Path to the pyramid images store.
            pyramids_images_store_dir (str): Directory for storing pyramid images.
            pyramid_image_location (str): Final location of the pyramid image.
        """
        os.makedirs(pyramids_images_store_dir, exist_ok=True)
        file_temp = pyramid_image_location.replace(hash_value, "temp_" + hash_value)
        file_temp_lock = file_temp + ".lock"
        file_lock = FileLock(file_temp_lock)
        try:
            with file_lock.acquire():
                logger.info("File lock acquired.")
                if not os.path.exists(pyramid_image_location):
                    logger.success(f"==> pyramid image is building...")
                    start_time = time.time()
                    subresolutions = self.divide_time(
                        first_series.shape, factor, self.tile_size
                    )
                    start_load = time.time()
                    data = first_series.asarray()
                    end_load = time.time()
                    load_time = end_load - start_load
                    logger.success(
                        f"loading first series or level {self.datapath} time: {load_time}"
                    )
                    xy_resolution = self.image.pages[0].resolution  # micrometer
                    # prefix = 'py_'
                    # suffix = '.ome.tif'
                    # pyramids_images_store = self.settings.get('tif_loader', 'pyramids_images_store')
                    # pyramid_image_location = pyramids_images_store  + hash_value + suffix
                    with tifffile.TiffWriter(file_temp, bigtiff=True) as tif:

                        metadata = {
                            "axes": self.axe_compatibility_check(),
                            # "SignificantBits": 10,
                            # "TimeIncrement": 0.1,
                            # "TimeIncrementUnit": "s",
                            # "PhysicalSizeX": pixelsize,
                            # "PhysicalSizeXUnit": "Âµm",
                            # "PhysicalSizeY": pixelsize,
                            # "PhysicalSizeYUnit": "Âµm",
                            #          'Channel': {'Name': ['Channel 1', 'Channel 2']},
                            #          'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['Âµm'] * 16}
                        }
                        options = dict(
                            photometric=self.photometric,
                            tile=self.tile_size,
                            compression=self.compression,
                            resolutionunit=self.image.pages[0].resolutionunit,
                        )

                        tif.write(
                            data,
                            subifds=subresolutions,
                            resolution=xy_resolution,
                            metadata=metadata,
                            **options,
                        )
                        # in production use resampling to generate sub-resolution images
                        for level in range(subresolutions):
                            mag = factor ** (level + 1)
                            if self.type.endswith("S"):
                                tif.write(
                                    data[..., ::mag, ::mag, :],
                                    subfiletype=1,
                                    resolution=(
                                        xy_resolution[0] / mag,
                                        xy_resolution[1] / mag,
                                    ),
                                    **options,
                                )
                            else:
                                tif.write(
                                    data[..., ::mag, ::mag],
                                    subfiletype=1,
                                    resolution=(
                                        xy_resolution[0] / mag,
                                        xy_resolution[0] / mag,
                                    ),
                                    **options,
                                )
                    end_time = time.time()
                    execution_time = end_time - start_time
                    logger.success(
                        f"actual pyramid generation {self.datapath} time:{execution_time - load_time}"
                    )
                    os.rename(file_temp, pyramid_image_location)
                    logger.success(
                        f"{self.datapath} connected to ==> {pyramid_image_location}"
                    )
                    logger.success(
                        f"pyramid image building complete {self.datapath} total execution time: {execution_time}"
                    )
                    if (
                        get_directory_size(pyramids_images_store)
                        > self.allowed_store_size_byte
                    ):
                        delete_oldest_files(
                            pyramids_images_store, self.allowed_store_size_byte
                        )
                else:
                    logger.info("file detected!")
                    if os.path.exists(file_temp):
                        os.remove(file_temp)
            self.pyramid_dic[hash_value] = pyramid_image_location
            self.datapath = pyramid_image_location
        except Exception as e:
            logger.error(f"An error occurred during generation process: {e}")
        finally:
            # self.image = tifffile.TiffFile(pyramid_image_location)
            # Ensure any allocated memory or resources are released
            if "data" in locals():
                del data
            logger.success("Resources cleaned up.")

    def axe_compatibility_check(self):
        """
        Check and adjust axis labels for compatibility with the TIFF structure.

        Returns:
            str: Adjusted axis labels.
        """
        if "I" in self.type:
            self.type = self.type.replace("I", "T")
        if "Q" in self.type:
            self.type = self.type.replace("Q", "T")
            logger.info(self.type)
        return self.type

    def divide_time(self, shape, factor, tile_size):
        """
        Calculate the number of sub-resolutions required for a pyramid structure.

        Args:
            shape (tuple): Dimensions of the image (height, width).
            factor (int): Downscaling factor.
            tile_size (tuple): Tile dimensions.

        Returns:
            int: Number of sub-resolutions.
        """
        # max_axes = max(
        #     shape[self.axes_pos_dic.get("Y")], shape[self.axes_pos_dic.get("X")]
        # )
        shape_y = shape[self.axes_pos_dic.get("Y")]
        shape_x = shape[self.axes_pos_dic.get("X")]
        times = 0
        while shape_y > tile_size[0] or shape_x > tile_size[1]:
            shape_y = shape_y // factor
            shape_x = shape_x // factor
            times = times + 1
        return times

    def axes_pos_extract(self, axes):
        """
        Extract the positions of axes from their labels.

        Args:
            axes (str): String representing axis labels.

        Returns:
            dict: Mapping of axis labels to their positions.
        """
        dic = {
            "T": None,
            "C": None,
            "Z": None,
            "Q": None,
            "I": None,
            "Y": None,
            "X": None,
            "S": None,
        }
        characters = list(axes)
        # logger.info("Axis characters:", characters)
        for index, char in enumerate(characters):
            if char in dic:
                dic[char] = index
        return dic

    def axes_value_extract(self, axes, shape):
        """
        Extract the sizes of axes from their labels and the image shape.

        Args:
            axes (str): String representing axis labels.
            shape (tuple): Shape of the image.

        Returns:
            dict: Mapping of axis labels to their sizes.
        """
        dic = {"T": 1, "C": 1, "Z": 1, "Q": 1, "I": 1}
        characters = list(axes)
        # logger.info("Axis characters:", characters)
        for index, char in enumerate(characters):
            if char in dic:
                dic[char] = shape[index]
        return dic


# class tif_file_precheck:
#     def __init__(self, file_location) -> None:
#         self.location = file_location
#         self.image = tifffile.TiffFile(file_location)
#         self.file_stat = os.stat(file_location)
#         self.size = self.file_stat.st_size
#         self.metaData = {}
#         self.dataType = str(self.image.pages[0].dtype)
#         self.metaData["dataType"] = self.dataType
#         if self.image.pages[0].is_tiled:
#             # Get the tile size
#             self.tile_size = (
#                 self.image.pages[0].tilelength,
#                 self.image.pages[0].tilewidth,
#             )
#         else:
#             logger.info("Assigning tile size (128, 128)")
#             self.tile_size = (128, 128)
#         self.type = self.image.series[0].axes
#         self.is_pyramidal = self.image.series[0].is_pyramidal
#         self.axes_pos_dic = self.axes_char_extract(self.type)
#         self.inspectors_result = self.pyramid_inspectors(self.image)
#         self.metaData["inspectors_result"] = self.inspectors_result
#         for i_s, s in enumerate(self.image.series):
#             logger.info(f"Series {i_s}: {s}")
#             for i_l, level in enumerate(s.levels):
#                 logger.info(f"Level {i_l}: {level}")
#                 self.metaData[f"Series:{i_s}, Level:{i_l}"] = str(level)
#         del self.image
#         gc.collect()

#     def pyramid_inspectors(self, tif):
#         if self.is_pyramidal:
#             # may need more check opeation
#             return True
#         else:
#             for r in range(len(tif.series)):
#                 if r != 0:
#                     x_value_cur = tif.series[r].shape[self.axes_pos_dic.get("X")]
#                     x_value_pre = tif.series[r - 1].shape[self.axes_pos_dic.get("X")]
#                     y_value_cur = tif.series[r].shape[self.axes_pos_dic.get("Y")]
#                     y_value_pre = tif.series[r - 1].shape[self.axes_pos_dic.get("Y")]
#                     if (
#                         x_value_cur != x_value_pre // 2
#                         or y_value_cur != y_value_pre // 2
#                     ) and (
#                         x_value_cur != math.ceil(x_value_pre / 2)
#                         or y_value_cur != math.ceil(y_value_pre / 2)
#                     ):
#                         return False
#                 if r == len(tif.series) - 1:
#                     if (
#                         tif.series[r].pages[0].imagelength > self.tile_size[0]
#                         or tif.series[r].pages[0].imagewidth > self.tile_size[1]
#                     ):
#                         return False
#                     else:
#                         return True
#             return True

#     def axes_char_extract(self, axes):
#         dic = {
#             "T": None,
#             "C": None,
#             "Z": None,
#             "Q": None,
#             "I": None,
#             "Y": None,
#             "X": None,
#             "S": None,
#         }
#         characters = list(axes)
#         # logger.info("Axis characters:", characters)
#         for index, char in enumerate(characters):
#             if char in dic:
#                 dic[char] = index
#         return dic
