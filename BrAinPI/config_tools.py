
import os
import sys
import imaris_ims_file_reader as ims
# Import zarr stores
from zarr.storage import NestedDirectoryStore
from zarr_stores.archived_nested_store import Archived_Nested_Store
from zarr_stores.h5_nested_store import H5_Nested_Store
# import tiff_loader
import hashlib
def calculate_hash(input_string):
    # Calculate the SHA-256 hash of the input string
    hash_result = hashlib.sha256(input_string.encode()).hexdigest()
    return hash_result
def get_config(file='settings.ini',allow_no_value=True):
    import configparser
    # file = os.path.join(os.path.split(os.path.abspath(__file__))[0],file)
    # file_path = os.path.join(sys.path[0], file)
    # This condition is used for documentation generation through sphinx and readTheDoc, plz always have settings.ini.
    # if os.path.exists(file) is False:
    #     file_path = os.path.join(sys.path[0], 'template_' + file)
    #     print('sphinx generation',file)
    # config = configparser.ConfigParser(allow_no_value=allow_no_value)
    # config.read(file_path)
    # return config
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, file)
    if os.path.exists(file_path) is False:
        file_path = os.path.join(dir_path, 'template_' + file)
        print('sphinx generation',file_path)
    config = configparser.ConfigParser(allow_no_value=allow_no_value)
    config.read(file_path)
    return config
    
def get_pyramid_images_connection(settings):
    # os.makedirs(settings.get('tif_loader','pyramids_images_store'),exist_ok=True)
    # connection = {}
    # directory = settings.get('tif_loader', 'pyramids_images_store')
    # extension_type = settings.get('tif_loader', 'extension_type')
    # for root, dirs, files in os.walk(directory):
    #     for file in files:
    #         if file.endswith(extension_type):
    #             extension_index = file.rfind(extension_type)
    #             hash_value = file[:extension_index]
    #             file_path = os.path.join(root, file)
    #             connection[hash_value] = file_path
    connection = {}
    directory = settings.get('pyramids_images_location', 'location')
    tif_extension = settings.get('tif_loader', 'extension_type')
    nifti_extension = settings.get('nifti_loader', 'extension_type')
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.endswith(nifti_extension):
                extension_index = dir.rfind(nifti_extension)
                hash_value = dir[:extension_index]
                dir_path = os.path.join(root, dir)
                connection[hash_value] = dir_path
        for file in files:
            if file.endswith(tif_extension):
                extension_index = file.rfind(tif_extension)
                hash_value = file[:extension_index]
                file_path = os.path.join(root, file)
                connection[hash_value] = file_path
                break
    # print(connection)
    return connection
class config:
    """
    This class will be used to manage open datasets and persistant cache
    """

    def __init__(self):
        """
        evictionPolicy Options:
            "least-recently-stored" #R only
            "least-recently-used"  #R/W (maybe a performace hit but probably best cache option)
        """
        self.opendata = {}
        self.settings = get_config('settings.ini')
        self.pyramid_images_connection = get_pyramid_images_connection(self.settings)
        from cache_tools import get_cache
        self.cache = get_cache()

        def __del__(self):
            if self.cache is not None:
                self.cache.close()

    def loadDataset(self, key: str, dataPath: str ):
        """
        Given the filesystem path to a file, open that file with the appropriate
        reader and store it in the opendata attribute with the dataPath as
        the key

        If the key exists return
        Always return the name of the dataPath
        """
        # print(dataPath , file_ino , modification_time)
        from logger_tools import logger
        if key in self.opendata:
            logger.info(f'DATAPATH ENTRIES__{tuple(self.opendata.keys())}')
            return key
        if os.path.splitext(dataPath)[-1] == '.ims':

            logger.info('Creating ims object')
            self.opendata[key] = ims.ims(dataPath, squeeze_output=False)

            if self.opendata[key].hf is None or self.opendata[key].dataset is None:
                logger.info('opening ims object')
                self.opendata[key].open()
                
        elif dataPath.endswith('.ome.zarr'):
            from ome_zarr_loader import ome_zarr_loader
            self.opendata[key] = ome_zarr_loader(dataPath, squeeze=False, zarr_store_type=NestedDirectoryStore, cache=self.cache)
            # self.opendata[dataPath].isomezarr = True

        elif '.omezans' in os.path.split(dataPath)[-1]:
            from ome_zarr_loader import ome_zarr_loader
            self.opendata[key] = ome_zarr_loader(dataPath, squeeze=False, zarr_store_type=Archived_Nested_Store, cache=self.cache)

        elif '.omehans' in os.path.split(dataPath)[-1]:
            from ome_zarr_loader import ome_zarr_loader
            self.opendata[key] = ome_zarr_loader(dataPath, squeeze=False, zarr_store_type=H5_Nested_Store, cache=self.cache)

        elif 's3://' in dataPath and dataPath.endswith('.zarr'):
            # import s3fs
            # self.opendata[dataPath] = ome_zarr_loader(dataPath, squeeze=False, zarr_store_type=s3fs.S3Map,
            #                                           cache=self.cache)
            from s3_utils import s3_boto_store
            self.opendata[key] = ome_zarr_loader(dataPath, squeeze=False, zarr_store_type=s3_boto_store,
                                                    cache=self.cache)
        elif dataPath.lower().endswith('tif') or dataPath.lower().endswith('tiff'):
            import tiff_loader
            # To do for metadata attribute rebuild, currently not compatible
            self.opendata[key] = tiff_loader.tiff_loader(dataPath, self.pyramid_images_connection, self.cache,self.settings)
        elif dataPath.lower().endswith('.terafly'):
            import terafly_loader
            self.opendata[key] = terafly_loader.terafly_loader(dataPath, squeeze=False,cache=self.cache)
        elif dataPath.lower().endswith('.nii.zarr') or dataPath.lower().endswith('.nii.gz') or dataPath.lower().endswith('.nii'):
            import nifti_loader
            self.opendata[key] = nifti_loader.nifti_zarr_loader(dataPath, self.pyramid_images_connection,self.settings,squeeze=False,cache=self.cache)
        elif dataPath.lower().endswith('.jp2'):
            import jp2_loader
            self.opendata[key] = jp2_loader.jp2_loader(dataPath, self.pyramid_images_connection,self.settings,squeeze=False,cache=self.cache)
        ## Append extracted metadata as attribute to open dataset
        try:
            from utils import metaDataExtraction # Here to get around curcular import at BrAinPI init
            self.opendata[key].metadata = metaDataExtraction(self.opendata[key])
            logger.info(self.opendata[key].metadata)
        except Exception:
            pass

        return key
