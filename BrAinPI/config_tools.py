
import os
import imaris_ims_file_reader as ims
# Import zarr stores
from zarr.storage import NestedDirectoryStore
from zarr_stores.archived_nested_store import Archived_Nested_Store
from zarr_stores.h5_nested_store import H5_Nested_Store
import hashlib

# def calculate_hash(input_string):
#     """
#     Calculate the SHA-256 hash of the input string
#     """
#     hash_result = hashlib.sha256(input_string.encode()).hexdigest()
#     return hash_result

def get_config(file='settings.ini',allow_no_value=True):
    """
    Load configuration settings from the created setting.ini file.

    This function reads configuration settings from a specified settings,ini file.
    It is primarily used for loading settings. It can be used for Sphinx 
    documentation generation if the file does not exist, it will fall back to a 
    template version of the file.

    Args:
        file (str, optional): The name of the INI file to load. Defaults to 'settings.ini'.
        allow_no_value (bool, optional): Whether to allow keys without values in the INI file. 
                                         Defaults to True.

    Returns:
        configparser.ConfigParser: A ConfigParser object containing the parsed configuration.
    """
    import configparser
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, file)
    # This condition is used for documentation generation through sphinx and readTheDoc, plz always have settings.ini.
    if os.path.exists(file_path) is False:
        file_path = os.path.join(dir_path, 'template_' + file)
        print('sphinx generation',file_path)
    config = configparser.ConfigParser(allow_no_value=allow_no_value)
    config.read(file_path)
    return config
    
def get_pyramid_images_connection(settings):
    """
    Build a connection mapping for pyramid image directories and files.

    This function walks through a specified directory (from settings) to locate 
    NIfTI and TIFF files/directories based on their extensions. It builds a 
    mapping of unique hash values (derived from files) to their corresponding paths.

    Args:
        settings (configparser.ConfigParser): A configuration object containing
                                              location of generated pyramid files.

    Returns:
        dict: A dictionary where keys are hash values (derived from file or directory names
              without extensions) and values are their full paths.
    """
    connection = {}
    dict_extension = set()
    file_extension = set()
    directory = settings.get('pyramids_images_location', 'location')
    tif_extension = settings.get('tif_loader', 'extension_type')
    nifti_extension = settings.get('nifti_loader', 'extension_type')
    jp2_extension = settings.get('jp2_loader', 'extension_type')
    dict_extension.update([nifti_extension])
    file_extension.update([tif_extension,jp2_extension])
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            dir_matching_ext = next((ext for ext in dict_extension if dir.endswith(ext)), None)
            if dir_matching_ext:
                dir_extension_index = dir.rfind(dir_matching_ext)
                hash_value = dir[:dir_extension_index]
                dir_path = os.path.join(root, dir)
                connection[hash_value] = dir_path
        for file in files:
            file_matching_ext = next((ext for ext in file_extension if file.endswith(ext)), None)
            if file_matching_ext:
                file_extension_index = file.rfind(file_matching_ext)
                hash_value = file[:file_extension_index]
                file_path = os.path.join(root, file)
                connection[hash_value] = file_path
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
        Initialize the `config` object.

        This method sets up the necessary configurations, establishes connections to pyramid images,
        and initializes a persistent cache for efficient data management.

        Args:
            opendata (dict): A dictionary to store open datasets, with keys as dataset identifiers and values as dataset objects.
            settings (configparser.ConfigParser): Loaded configuration settings from `settings.ini`.
            pyramid_images_connection (dict): A mapping of hash values to pyramid image paths, built from configuration settings.
            cache (diskcache.FanoutCache): A persistent cache object for managing dataset resources efficiently.
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
        reader and store it in the opendata attribute with the self/hash of dataPath
        as the key

        If the key exists return
        Always return the self/hash of the dataPath

        Args:
            key (str): self/hash of dataPath
            dataPath (str): dataPath

        Returns:
            key (str): self/hash of dataPath
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
