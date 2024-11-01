import zarr, os, itertools
import numpy as np
import hashlib
import shutil
import time
import nibabel as nib
from filelock import FileLock
from zarr.storage import NestedDirectoryStore
from collections.abc import MutableMapping
from zarr._storage.store import Store, BaseStore
from typing import Union
from niizarr import nii2zarr
Path = Union[str, bytes, None]
StoreLike = Union[BaseStore, Store, MutableMapping]
from logger_tools import logger
import gc
import multiprocessing


def niigz2niizarr(inp, out, time_axe):
    nii2zarr(inp, out, no_time=time_axe)

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


class nifti_zarr_loader:
    def __init__(
        self,
        location,
        pyramid_images_connection,
        settings,
        ResolutionLevelLock=None,
        zarr_store_type: StoreLike = NestedDirectoryStore,
        verbose=None,
        squeeze=True,
        cache=None,
    ):

        self.location = location
        self.ResolutionLevelLock = (
            0 if ResolutionLevelLock is None else ResolutionLevelLock
        )
        self.file_stat = os.stat(location)
        self.filename = os.path.split(self.location)[1]
        self.file_ino = str(self.file_stat.st_ino)
        self.modification_time = str(self.file_stat.st_mtime)
        self.file_size = self.file_stat.st_size
        self.settings = settings
        self.allowed_store_size_gb = float(
            self.settings.get("nifti_loader", "pyramids_images_allowed_store_size_gb")
        )
        self.allowed_store_size_byte = self.allowed_store_size_gb * 1024 * 1024 * 1024
        self.allowed_file_size_gb = float(
            self.settings.get("nifti_loader", "pyramids_images_allowed_generation_size_gb")
        )
        self.allowed_file_size_byte = self.allowed_file_size_gb * 1024 * 1024 * 1024
        self.pyramid_dic = pyramid_images_connection

        self.verbose = verbose
        self.squeeze = squeeze
        self.cache = cache
        self.metaData = {}
        # Go through pyramid generation process for nii.gz files
        if self.location.endswith(".nii.gz") or self.location.endswith(".nii"):
            self.validate_nifti_file(self.location)
            self.pyramid_builders(self.location)
        # Open zarr store
        self.zarr_store = zarr_store_type  # Only relevant for non-s3 datasets
        store = self.zarr_store_type(
            self.location
        )
        zgroup = zarr.open(store)
        self.zattrs = zgroup.attrs

        if "omero" in self.zattrs:
            self.omero = zgroup.attrs["omero"]
        # assert 'omero' in self.zattrs
        # self.omero = zgroup.attrs['omero']
        assert "multiscales" in self.zattrs
        self.multiscales = zgroup.attrs["multiscales"]
        try:
            self.axes = self.multiscales[0]["axes"]
        except:
            self.axes = self.multiscales["axes"]
        # self.axes = self.multiscales[0]['axes']
        if len(self.axes) < 3:
            raise Exception()
        self.dim_pos_dic = {"t": None, "c": None, "z": None, "y": None, "x": None}
        self.space_unit = None
        for index, axe in enumerate(self.axes):
            self.dim_pos_dic[axe["name"]] = index
            if axe["type"] == "space":
                self.space_unit = axe["unit"]
        logger.info(self.dim_pos_dic)
        logger.info(self.multiscales)
        logger.info(self.space_unit)
        del zgroup
        del store

        try:
            self.multiscale_datasets = self.multiscales[0]["datasets"]
        except:
            self.multiscale_datasets = self.multiscales["datasets"]

        self.ResolutionLevels = len(self.multiscale_datasets)

        self.dataset_paths = []
        self.dataset_scales = []
        self.arrays = {}
        for r in range(self.ResolutionLevels):
            self.dataset_paths.append(self.multiscale_datasets[r]["path"])
            self.dataset_scales.append(
                self.multiscale_datasets[r]["coordinateTransformations"][0]["scale"]
            )
            array = self.open_array(r)
            if r == 0:
                self.TimePoints = (
                    array.shape[self.dim_pos_dic["t"]] if self.dim_pos_dic["t"] else 1
                )
                self.Channels = (
                    array.shape[self.dim_pos_dic["c"]] if self.dim_pos_dic["c"] else 1
                )
            # shape_z = array.shape[self.dim_pos_dic['z']]
            # shape_y = array.shape[self.dim_pos_dic['y']]
            # shape_x = array.shape[self.dim_pos_dic['x']]
            # if shape_z <= 64 and shape_y <= 64 and shape_x <=64:
            #     self.ResolutionLevels = r + 1
            #     break

            for t, c in itertools.product(range(self.TimePoints), range(self.Channels)):

                # Collect attribute info
                self.metaData[r, t, c, "shape"] = array.shape
                

                # change to um if mm
                if self.space_unit == "mm" or self.space_unit == "millimeter":
                    self.metaData[r, t, c, "resolution"] = [val * 1000 for val in self.dataset_scales[r][-3:]]
                else:
                    self.metaData[r, t, c, "resolution"] = self.dataset_scales[r][-3:]

                # Collect dataset info
                self.metaData[r, t, c, "chunks"] = array.chunks[-3:]
                dtype = array.dtype[0]
                if dtype == "int8":
                    dtype = "uint8"
                elif dtype == "int16":
                    dtype = "uint16"
                elif dtype == "float64" or dtype == "float16":
                    dtype = "float32"
                self.metaData[r, t, c, "dtype"] = dtype
                self.metaData[r, t, c, "ndim"] = array.ndim

                try:
                    self.metaData[r, t, c, "max"] = self.omero["channels"][c]["window"][
                        "end"
                    ]
                    self.metaData[r, t, c, "min"] = self.omero["channels"][c]["window"][
                        "start"
                    ]
                except:
                    pass
            self.arrays[r] = array
            shape_z = array.shape[self.dim_pos_dic["z"]]
            shape_y = array.shape[self.dim_pos_dic["y"]]
            shape_x = array.shape[self.dim_pos_dic["x"]]
            if shape_z <= 64 and shape_y <= 64 and shape_x <= 64:
                self.ResolutionLevels = r + 1
                break

        self.change_resolution_lock(self.ResolutionLevelLock)
        logger.info(self.metaData)

    def validate_nifti_file(self, file_path):
        if self.file_size > self.allowed_file_size_byte:
                raise Exception(f"File '{self.filename}' can not generate pyramid structure. Due to resource constrait, {self.allowed_file_size_gb}GB and below are acceptable for generation process.")
        try:
            # Attempt to load the file
            img = nib.load(file_path)
        except Exception as e:
            raise Exception(f"File '{file_path}' is not a valid nifti file. Error: {e}")

    def pyramid_builders(self, nifti_file_location):
        hash_value = calculate_hash(self.file_ino + self.modification_time)
        pyramids_images_store = self.settings.get(
            "nifti_loader", "pyramids_images_store"
        )
        pyramids_images_store_dir = (
            pyramids_images_store + hash_value[0:2] + "/" + hash_value[2:4] + "/"
        )
        suffix = self.settings.get("nifti_loader", "extension_type")
        pyramid_image_location = pyramids_images_store_dir + hash_value + suffix
        if self.pyramid_dic.get(hash_value) and os.path.exists(pyramid_image_location):
            self.location = self.pyramid_dic.get(hash_value)
            logger.info("Location replaced by generated pyramid image")
        else:
            # Avoid other gunicore workers to build pyramids images
            if os.path.exists(pyramid_image_location):
                logger.info(
                    "Pyramid image was already built by first worker and picked up now by others"
                )
                self.pyramid_dic[hash_value] = pyramid_image_location
                self.location = pyramid_image_location
            # 1 hash exists but the pyramid images are deleted during server running
            # 2 no hash and no pyramid images (first time generation)
            else:
                self.pyramid_building_process(
                    nifti_file_location,
                    False,
                    hash_value,
                    pyramids_images_store,
                    pyramids_images_store_dir,
                    pyramid_image_location,
                )
        self.location = pyramid_image_location

    def pyramid_building_process(
        self,
        nifti_file_location,
        time_axe,
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
                logger.success("File lock acquired.")
                if not os.path.exists(pyramid_image_location):
                    logger.success(f"==> Pyramid image is building...")
                    start_time = time.time()
                    # nii2zarr(nifti_file_location,file_temp, no_time=time_axe)

                    # thread = threading.Thread(target=niigz2niizarr,args=(nifti_file_location,file_temp,time_axe))
                    # thread.start()
                    # thread.join()
                    # print("thread complete!")

                    process = multiprocessing.Process(target=niigz2niizarr, args=(nifti_file_location, file_temp, time_axe))
                    process.start()
                    process.join()
                    logger.success("Process complete!")

                    end_time= time.time()
                    execution_time = end_time - start_time
                    os.rename(file_temp, pyramid_image_location)
                    logger.success(
                        f"{nifti_file_location} connected to ==> {pyramid_image_location}"
                    )
                    logger.success(
                        f"Pyramid image building complete {nifti_file_location} total execution time: {execution_time}"
                    )
                    if (
                        get_directory_size(pyramids_images_store)
                        > self.allowed_store_size_byte
                    ):
                        delete_oldest_files(
                            pyramids_images_store, self.allowed_store_size_byte
                        )
                else:
                    logger.info("File detected!")
                    if os.path.exists(file_temp):
                        logger.warning('file_temp exist!')
                        os.remove(file_temp)
            self.pyramid_dic[hash_value] = pyramid_image_location
            self.location = pyramid_image_location
        except Exception as e:
            logger.error(f"An error occurred during generation process: {e}")
        finally:
            if "data" in locals():
                del data
            gc.collect()
            logger.success("Resources cleaned up.")
            

    def zarr_store_type(self, path):
        return self.zarr_store(path)

    def change_resolution_lock(self, ResolutionLevelLock):
        self.ResolutionLevelLock = ResolutionLevelLock
        self.shape = self.metaData[self.ResolutionLevelLock, 0, 0, "shape"]
        self.ndim = len(self.shape)
        self.chunks = self.metaData[self.ResolutionLevelLock, 0, 0, "chunks"]
        self.resolution = self.metaData[self.ResolutionLevelLock, 0, 0, "resolution"]
        self.dtype = self.metaData[self.ResolutionLevelLock, 0, 0, "dtype"]

    def __getitem__(self, key):

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

    def _get_memorize_cache(
        self, name=None, typed=False, expire=None, tag=None, ignore=()
    ):
        if tag is None:
            tag = self.location
        return (
            self.cache.memorize(
                name=name, typed=typed, expire=expire, tag=tag, ignore=ignore
            )
            if self.cache is not None
            else lambda x: x
        )

    def getSlice(self, r, t, c, z, y, x):
        """
        Access the requested slice based on resolution level and
        5-dimentional (t,c,z,y,x) access to zarr array.
        """

        incomingSlices = (r, t, c, z, y, x)
        logger.info(incomingSlices)
        if self.cache is not None:
            key = f"{self.location}_getSlice_{str(incomingSlices)}"
            # key = self.location + '_getSlice_' + str(incomingSlices)
            result = self.cache.get(key, default=None, retry=True)
            if result is not None:
                logger.info(f"Returned from cache: {incomingSlices}")
                return result
        list_tp = [0] * self.ndim
        if self.dim_pos_dic.get("t") != None:
            list_tp[self.dim_pos_dic.get("t")] = t
        if self.dim_pos_dic.get("c") != None:
            list_tp[self.dim_pos_dic.get("c")] = c
        if self.dim_pos_dic.get("z") != None:
            list_tp[self.dim_pos_dic.get("z")] = z
        if self.dim_pos_dic.get("y") != None:
            list_tp[self.dim_pos_dic.get("y")] = y
        if self.dim_pos_dic.get("x") != None:
            list_tp[self.dim_pos_dic.get("x")] = x
        tp = tuple(list_tp)
        # logger.success(tp)
        result = self.arrays[r][tp]
        if len(result.shape) < 4:
            result = np.expand_dims(result, axis=0)
        result = result.astype(self.dtype)
        logger.info(result.shape)
        if self.cache is not None:
            self.cache.set(key, result, expire=None, tag=self.location, retry=True)
            # test = True
            # while test:
            #     # logger.info('Caching slice')
            #     self.cache.set(key, result, expire=None, tag=self.location, retry=True)
            #     if result == self.getSlice(*incomingSlices):
            #         test = False

        return result


    def locationGenerator(self, res):
        return os.path.join(self.location, self.dataset_paths[res])

    def open_array(self, res):
        store = self.zarr_store_type(self.locationGenerator(res))
        logger.info("OPENING ARRAYS")
        store = self.wrap_store_in_chunk_cache(store)
        # if self.cache is not None:
        #     logger.info('OPENING CHUNK CACHE ARRAYS')
        #     from zarr_stores.zarr_disk_cache import Disk_Cache_Store
        #     store = Disk_Cache_Store(store, unique_id=store.path, diskcache_object=self.cache, persist=False)
        # # try:
        # #     if self.cache is not None:
        # #         store = disk_cache_store(store=store, uuid=self.locationGenerator(res), diskcache_object=self.cache, persist=None, meta_data_expire_min=15)
        # # except Exception as e:
        # #     logger.info('Caught Exception')
        # #     logger.info(e)
        # #     pass
        return zarr.open(store)

    def wrap_store_in_chunk_cache(self, store):
        if self.cache is not None:
            logger.info("OPENING CHUNK CACHE ARRAYS")
            logger.info(store.path)
            from zarr_chunk_cache import disk_cache_store as Disk_Cache_Store

            store = Disk_Cache_Store(
                store, uuid=store.path, diskcache_object=self.cache, persist=True
            )
        return store


# uicontrol bool channel0_visable checkbox(default=true);

# uicontrol invlerp channel0_lut (range=[1.6502681970596313,180.84588623046875],window=[0,180.84588623046875]);

# uicontrol vec3 channel0_color color(default="green");

# vec3 channel0 = vec3(0);


# void main() {

# if (channel0_visable == true)
# channel0 = channel0_color *   channel0_lut();

# vec3 rgb = (channel0);

# vec3 render = min(rgb,vec3(1));

# emitRGB(render);
# }
