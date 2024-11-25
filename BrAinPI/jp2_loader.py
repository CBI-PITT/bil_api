import multiprocessing.process
from logger_tools import logger
import numpy as np
import itertools
import os
import glymur
import tifffile
import hashlib
from pathlib import Path
import shutil
import time
from filelock import FileLock
import tiff_loader
import multiprocessing


def calculate_hash(input_string):
    # Calculate the SHA-256 hash of the input string
    hash_result = hashlib.sha256(input_string.encode()).hexdigest()
    return hash_result


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def delete_oldest_files(directory, size_limit):
    items = sorted(Path(directory).glob("*"), key=os.path.getctime)
    total_size = get_directory_size(directory)

    # Delete oldest items until the total size is within the size limit
    for item in items:
        if total_size <= size_limit:
            break
        if item.is_file():
            item_size = os.path.getsize(item)
            os.remove(item)
            total_size -= item_size
            logger.success(f"Deleted file {item} of size {item_size} bytes")
        elif item.is_dir():
            dir_size = get_directory_size(item)
            shutil.rmtree(item)
            total_size -= dir_size
            logger.success(f"Deleted directory {item} of size {dir_size} bytes")


def separate_process_generation(
    jp2_img, factor, file_temp, subresolutions, datapath, tile_size
):

    start_load = time.time()
    # data = jp2_img[:]
    height, width = jp2_img.shape[:2]

    # Initialize an empty array to store the final result
    data = np.zeros(jp2_img.shape, dtype=jp2_img.dtype)
    chunk_size = 10000
    # Process the image in chunks
    logger.info("Chunks loading start...")
    for row_start in range(0, height, chunk_size):
        for col_start in range(0, width, chunk_size):
            logger.info(f"loading chunk {row_start}, {col_start}")
            # Calculate chunk boundaries
            row_end = min(row_start + chunk_size, height)
            col_end = min(col_start + chunk_size, width)
            
            # Load the chunk
            chunk = jp2_img[row_start:row_end, col_start:col_end]
            
            # Insert the chunk into the final array
            data[row_start:row_end, col_start:col_end] = chunk
    logger.info(f"Entire data loading completed, shape {data.shape}")
    end_load = time.time()
    load_time = end_load - start_load
    logger.success(f"loading first series or level {datapath} time: {load_time}")
    # hard coded
    xy_resolution = (1,1)
    resolutionunit = 2
    start_generation = time.time()
    with tifffile.TiffWriter(file_temp, bigtiff=True) as tif:

        metadata = {
            "axes": "YXS",
            # "SignificantBits": 10,
            # "TimeIncrement": 0.1,
            # "TimeIncrementUnit": "s",
            # "PhysicalSizeX": xy_resolution,
            # "PhysicalSizeXUnit": "Âµm",
            # "PhysicalSizeY": xy_resolution,
            # "PhysicalSizeYUnit": "Âµm",
            # 'Channel': {'Name': ['Channel 1', 'Channel 2']},
            # 'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['Âµm'] * 16}
        }
        options = dict(
            # photometric=self.photometric,
            tile=tile_size,
            # compression=self.compression,
            # resolutionunit="CENTIMETER",
            resolutionunit=resolutionunit
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
            tif.write(
                data[..., ::mag, ::mag, :],
                subfiletype=1,
                resolution=(
                    xy_resolution[0] / mag,
                    xy_resolution[1] / mag,
                ),
                **options,
            )
    end_generation = time.time()
    generation_time = end_generation - start_generation
    logger.success(f"actual pyramid generation {datapath} time:{generation_time}")


class jp2_loader:
    def __init__(
        self,
        location,
        pyramid_images_connection,
        settings,
        ResolutionLevelLock=None,
        verbose=None,
        squeeze=True,
        cache=None,
    ):

        # assert StoreLike is s3fs.S3Map or any([issubclass(zarr_store_type,x) for x in StoreLike.__args__]), 'zarr_store_type is not a zarr storage class'

        self.location = location
        self.datapath = location
        self.ResolutionLevelLock = (
            0 if ResolutionLevelLock is None else ResolutionLevelLock
        )
        self.file_stat = os.stat(location)
        self.filename = os.path.split(self.datapath)[1]
        self.file_ino = str(self.file_stat.st_ino)
        self.modification_time = str(self.file_stat.st_mtime)
        self.file_size = self.file_stat.st_size
        self.settings = settings
        self.allowed_store_size_gb = float(
            self.settings.get("jp2_loader", "pyramids_images_allowed_store_size_gb")
        )
        self.allowed_store_size_byte = self.allowed_store_size_gb * 1024 * 1024 * 1024
        self.allowed_file_size_gb = float(
            self.settings.get(
                "jp2_loader", "pyramids_images_allowed_generation_size_gb"
            )
        )
        self.allowed_file_size_byte = self.allowed_file_size_gb * 1024 * 1024 * 1024
        self.pyramid_dic = pyramid_images_connection
        self.verbose = verbose
        self.squeeze = squeeze
        self.cache = cache
        # self.metaData = {}
        if self.datapath.endswith(".jp2"):
            jp2_img = self.validate_jp2_file(self.datapath)
            self.tile_size = jp2_img.tilesize if jp2_img.tilesize else (128, 128)



            self.pyramid_builders(jp2_img)
            self.tif_obj = tiff_loader.tiff_loader(
                self.datapath,
                pyramid_images_connection,
                self.cache,
                self.settings,
                ResolutionLevelLock,
            )
            self.metaData = self.tif_obj.metaData
            self.ResolutionLevelLock = self.tif_obj.ResolutionLevelLock
            self.shape = self.tif_obj.shape
            self.ndim = self.tif_obj.ndim
            self.chunks = self.tif_obj.chunks
            self.resolution = self.tif_obj.resolution
            self.dtype = self.tif_obj.dtype
            self.TimePoints = self.tif_obj.TimePoints
            self.ResolutionLevels = self.tif_obj.ResolutionLevels
            self.Channels = self.tif_obj.Channels

        # self.change_resolution_lock(self.ResolutionLevelLock)

    def validate_jp2_file(self, file_path):
        if self.file_size > self.allowed_file_size_byte:
            logger.info(
                f"File '{self.filename}' can not generate pyramid structure. Due to resource constrait, {self.allowed_file_size_gb}GB and below are acceptable for generation process."
            )
            raise Exception(
                f"File '{self.filename}' can not generate pyramid structure. Due to resource constrait, {self.allowed_file_size_gb}GB and below are acceptable for generation process."
            )
        self.jp2_img = glymur.Jp2k(file_path)
        glymur.set_option('lib.num_threads', 4)
        if len(self.jp2_img.shape)!=3 and self.jp2_img.shape[-1]!=3:
            raise TypeError("Jp2 image shape not supported")
        return self.jp2_img

    def pyramid_builders(self, jp2_img):
        hash_value = calculate_hash(self.file_ino + self.modification_time)
        pyramids_images_store = self.settings.get("jp2_loader", "pyramids_images_store")
        pyramids_images_store_dir = (
            pyramids_images_store + hash_value[0:2] + "/" + hash_value[2:4] + "/"
        )
        suffix = self.settings.get("jp2_loader", "extension_type")
        pyramid_image_location = pyramids_images_store_dir + hash_value + suffix
        if self.pyramid_dic.get(hash_value) and os.path.exists(pyramid_image_location):
            self.datapath = self.pyramid_dic.get(hash_value)
            logger.info("Location replaced by generated pyramid image")
        else:
            # Avoid other gunicore workers to build pyramids images
            if os.path.exists(pyramid_image_location):
                logger.info(
                    "Pyramid image was already built by first worker and picked up now by others"
                )
                self.pyramid_dic[hash_value] = pyramid_image_location
                self.datapath = pyramid_image_location
            # 1 hash exists but the pyramid images are deleted during server running
            # 2 no hash and no pyramid images (first time generation)
            else:
                self.pyramid_building_process(
                    jp2_img,
                    2,
                    hash_value,
                    pyramids_images_store,
                    pyramids_images_store_dir,
                    pyramid_image_location,
                )
        # self.datapath = pyramid_image_location

    def pyramid_building_process(
        self,
        jp2_img,
        factor,
        hash_value,
        pyramids_images_store,
        pyramids_images_store_dir,
        pyramid_image_location,
    ):
        os.makedirs(pyramids_images_store_dir, exist_ok=True)
        file_temp = pyramid_image_location.replace(hash_value, "temp_" + hash_value)
        file_temp_lock = file_temp + ".lock"
        file_lock = FileLock(file_temp_lock)
        try:
            with file_lock.acquire():
                logger.info("File lock acquired.")
                if not os.path.exists(pyramid_image_location):
                    logger.success(f"==> pyramid image is building...")
                    subresolutions = self.divide_time(
                        jp2_img.shape, factor, self.tile_size
                    )
                    start_time = time.time()
                    process = multiprocessing.Process(
                        target=separate_process_generation,
                        args=(
                            jp2_img,
                            factor,
                            file_temp,
                            subresolutions,
                            self.datapath,
                            self.tile_size,
                        ),
                    )
                    process.start()
                    process.join()
                    logger.success("Process complete!")
                    end_time = time.time()
                    execution_time = end_time - start_time
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

    def change_resolution_lock(self, ResolutionLevelLock):
        self.ResolutionLevelLock = ResolutionLevelLock
        self.shape = self.metaData[self.ResolutionLevelLock, 0, 0, "shape"]
        self.ndim = len(self.shape)
        self.chunks = self.metaData[self.ResolutionLevelLock, 0, 0, "chunks"]
        self.resolution = self.metaData[self.ResolutionLevelLock, 0, 0, "resolution"]
        self.dtype = self.metaData[self.ResolutionLevelLock, 0, 0, "dtype"]

    def __getitem__(self, key):
        return self.tif_obj[key]

        res = 0 if self.ResolutionLevelLock is None else self.ResolutionLevelLock
        logger.info(key)
        if (
            isinstance(key, slice) == False
            and isinstance(key, int) == False
            and len(key) == 6
        ):
            res = key[0]
            if res >= self.ResolutionLevels:
                raise ValueError("Layer is larger than the number of ResolutionLevels")
            key = tuple([x for x in key[1::]])
        logger.info(res)
        logger.info(key)

        if isinstance(key, int):
            key = [slice(key, key + 1)]
            for _ in range(self.ndim - 1):
                key.append(slice(None))
            key = tuple(key)

        if isinstance(key, tuple):
            key = [slice(x, x + 1) if isinstance(x, int) else x for x in key]
            while len(key) < self.ndim:
                key.append(slice(None))
            key = tuple(key)

        logger.info(key)
        newKey = []
        for ss in key:
            if ss.start is None and isinstance(ss.stop, int):
                newKey.append(slice(ss.stop, ss.stop + 1, ss.step))
            else:
                newKey.append(ss)

        key = tuple(newKey)
        logger.info(key)

        array = self.getSlice(r=res, t=key[0], c=key[1], z=key[2], y=key[3], x=key[4])

        if self.squeeze:
            return np.squeeze(array)
        else:
            return array

    def getSlice(self, r, t, c, z, y, x):
        """
        Access the requested slice based on resolution level and
        5-dimentional (t,c,z,y,x) access to zarr array.
        """

        incomingSlices = (r, t, c, z, y, x)
        logger.info(incomingSlices)
        if self.cache is not None:
            key = f"{self.datapath}_getSlice_{str(incomingSlices)}"
            # key = self.datapath + '_getSlice_' + str(incomingSlices)
            result = self.cache.get(key, default=None, retry=True)
            if result is not None:
                logger.info(f"Returned from cache: {incomingSlices}")
                return result
        result = self.jp2_img[y.start*2**r:y.stop*2**r:2**r,x.start*2**r:x.stop*2**r:2**r]
        # result = self.arrays[r][t, c, z, y, x]

        if self.cache is not None:
            self.cache.set(key, result, expire=None, tag=self.datapath, retry=True)
            # test = True
            # while test:
            #     # logger.info('Caching slice')
            #     self.cache.set(key, result, expire=None, tag=self.datapath, retry=True)
            #     if result == self.getSlice(*incomingSlices):
            #         test = False

        return result
        # return self.open_array(r)[t,c,z,y,x]

    def locationGenerator(self, res):
        return os.path.join(self.datapath, self.dataset_paths[res])

    def divide_time(self, shape, factor, tile_size):
        shape_y = shape[0]
        shape_x = shape[1]
        times = 0
        while shape_y > tile_size[0] or shape_x > tile_size[1]:
            shape_y = shape_y // factor
            shape_x = shape_x // factor
            times = times + 1
        return times