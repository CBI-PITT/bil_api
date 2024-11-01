import itertools
import numpy as np
import os
import re
from v3dpy.terafly import TeraflyInterface
from logger_tools import logger


class terafly_loader:
    def __init__(
        self, location, ResolutionLevelLock=None, squeeze=True, cache=None
    ) -> None:
        file_list = self.extract_and_sort_by_multiplication(location)
        # print(file_list)
        self.location = location
        self.cache = cache
        self.squeeze = squeeze
        self.ResolutionLevels = len(file_list)
        self.ResolutionLevelLock = (
            0 if ResolutionLevelLock is None else ResolutionLevelLock
        )

        # todo later
        # ok for the current imgs we have
        # assign to 1 and seems no time point can be obtained by the functions provided
        self.TimePoints = 1

        self.array = {}
        self.metaData = {}
        for res, file in enumerate(file_list):
            # print(file_list)
            self.array[res] = self.validate_terafly_file(file)
            # get the numbers of channel and the data type
            if res == len(file_list) - 1:
                self.dtype = str(self.array[res].get_sub_volume(0, 1, 0, 1, 0, 1).dtype)
                dim_x_y_z_c = self.array[res].get_dim()
                self.Channels = dim_x_y_z_c[-1]

        for r in list(self.array.keys()):
            for t, c in itertools.product(range(self.TimePoints), range(self.Channels)):
                # reverse x_y_z to z_y_x
                reverse_tuple = self.array[r].get_dim()[0:3][::-1]
                self.metaData[r, t, c, "shape"] = (t + 1, c + 1, *reverse_tuple)
                self.metaData[r, t, c, "resolution"] = (
                    self.array[r]._volume.VXL_D,
                    self.array[r]._volume.VXL_V,
                    self.array[r]._volume.VXL_H,
                )
                self.metaData[r, t, c, "chunks"] = (128, 128, 128)
                self.metaData[r, t, c, "dtype"] = self.dtype
                self.metaData[r, t, c, "ndim"] = len(self.array[r].get_dim()) + 1
        self.change_resolution_lock(0)
        logger.info(self.metaData)

    def extract_and_sort_by_multiplication(self, directory):
        # entries = [
        #     os.path.join(directory, entry)
        #     for entry in os.listdir(directory)
        #     if os.path.isdir(os.path.join(directory, entry))
        # ]
        # results = []

        # pattern = r"^\D+(\d+)x(\d+)x(\d+)"

        # # Extract numbers and their multiplication result
        # for entry in entries:
        #     filename = os.path.basename(entry)
        #     match = re.match(pattern, filename)
        #     if match:
        #         num1 = int(match.group(1))
        #         num2 = int(match.group(2))
        #         num3 = int(match.group(3))
        #         result = num1 * num2 * num3
        #         results.append((entry, result))

        # # Sort results by multiplication result in descending order
        # results.sort(key=lambda x: x[1], reverse=True)

        # # Return sorted full paths
        # sorted_paths = [entry[0] for entry in results]
        # return sorted_paths
        entries = [
            os.path.join(directory, entry)
            for entry in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, entry))
        ]

        # Sort by length of the name first, then alphabetically for same length names
        entries.sort(key=lambda x: (len(x), x), reverse=True)
        return entries

    def change_resolution_lock(self, ResolutionLevelLock):
        self.ResolutionLevelLock = ResolutionLevelLock
        self.shape = self.metaData[self.ResolutionLevelLock, 0, 0, "shape"]
        self.ndim = len(self.shape)
        self.chunks = self.metaData[self.ResolutionLevelLock, 0, 0, "chunks"]
        self.resolution = self.metaData[self.ResolutionLevelLock, 0, 0, "resolution"]
        self.dtype = self.metaData[self.ResolutionLevelLock, 0, 0, "dtype"]

    def __getitem__(self, key):

        res = 0 if self.ResolutionLevelLock is None else self.ResolutionLevelLock
        # print(key)
        if (
            isinstance(key, slice) == False
            and isinstance(key, int) == False
            and len(key) == 6
        ):
            res = key[0]
            if res >= self.ResolutionLevels:
                raise ValueError("Layer is larger than the number of ResolutionLevels")
            key = tuple([x for x in key[1::]])
        # print(res)
        # print(key)

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

        # print(key)
        newKey = []
        for ss in key:
            if ss.start is None and isinstance(ss.stop, int):
                newKey.append(slice(ss.stop, ss.stop + 1, ss.step))
            else:
                newKey.append(ss)

        key = tuple(newKey)
        # print(key)

        array = self.getSlice(r=res, t=key[0], c=key[1], z=key[2], y=key[3], x=key[4])

        if self.squeeze:
            return np.squeeze(array)
        else:
            return array

    def getSlice(self, r, t, c, z, y, x):
        """
        Access the requested 3D chunck with all channels available based on resolution level and
        (z,y,x). t is always 0.
        """

        incomingSlices = (r, t, c, z, y, x)
        key = f"{self.location}_getSlice_{str(incomingSlices)}"
        if self.cache is not None:
            # key = self.location + '_getSlice_' + str(incomingSlices)
            result = self.cache.get(key, default=None, retry=True)
            if result is not None:
                logger.info(f"Returned from cache: {incomingSlices}")
                return result

        x_start = x.start
        x_stop = x.stop
        y_start = y.start
        y_stop = y.stop
        z_start = z.start
        z_stop = z.stop
        if all(
            coord is None
            for coord in [x_start, x_stop, y_start, y_stop, z_start, z_stop]
        ):
            dim_x_y_z = self.array[r].get_dim()[0:3]
            x_start = 0
            x_stop = dim_x_y_z[0]
            y_start = 0
            y_stop = dim_x_y_z[1]
            z_start = 0
            z_stop = dim_x_y_z[2]
        # print(x_start, x_stop, y_start, y_stop, z_start, z_stop)

        # returns the specific chuncks with all channels if available
        result = self.array[r].get_sub_volume(
            x_start, x_stop, y_start, y_stop, z_start, z_stop
        )

        # print(result.shape)
        if self.cache is not None:
            self.cache.set(key, result, expire=None, tag=self.location, retry=True)
        return result
    
    def validate_terafly_file(self, file_path):
        try:
            # Attempt to load the file
            img = TeraflyInterface(file_path)
            return img
        except Exception as e:
            raise Exception(f"File '{file_path}' is not a valid terafly file. Error: {e}")


# import struct
# import os
# import re
# import numpy as np
# import tifffile
# class terafly_loader:
#     def __init__(self, location, ResolutionLevelLock=None, squeeze=False) -> None:
#         # file_list = [
#         #     "/home/kelin/Documents/BrainPI_image_data/terafly_data/mouseID_339952-17782/mouse17782_teraconvert/RES_875x1406x368_",
#         #     "/home/kelin/Documents/BrainPI_image_data/terafly_data/mouseID_339952-17782/mouse17782_teraconvert/RES_1750x2812x736_",
#         #     "/home/kelin/Documents/BrainPI_image_data/terafly_data/mouseID_339952-17782/mouse17782_teraconvert/RES_3500x5625x1473_",
#         #     "/home/kelin/Documents/BrainPI_image_data/terafly_data/mouseID_339952-17782/mouse17782_teraconvert/RES_7000x11250x2946_",
#         # ]
#         file_list = self.extract_and_sort_by_multiplication(location)
#         print(file_list)
#         self.squeeze = squeeze
#         self.ResolutionLevels = len(file_list)
#         self.ResolutionLevelLock = 0 if ResolutionLevelLock is None else ResolutionLevelLock
#         self.array = {}
#         self.TimePoints = 1
#         self.Channels = 1
#         self.metaData = {}
#         for res, file in enumerate(file_list):
#             self.read_metaData(res, file, self.metaData, self.array)
#         print(self.metaData)
#         self.change_resolution_lock(self.ResolutionLevelLock)

#     def extract_and_sort_by_multiplication(self, directory) :
#         entries = [os.path.join(directory, entry) for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry))]
#         results = []

#         pattern = r'^\D+(\d+)x(\d+)x(\d+)'

#         # Extract numbers and their multiplication result
#         for entry in entries:
#             filename = os.path.basename(entry)
#             match = re.match(pattern, filename)
#             if match:
#                 num1 = int(match.group(1))
#                 num2 = int(match.group(2))
#                 num3 = int(match.group(3))
#                 result = num1 * num2 * num3
#                 results.append((entry, result))

#         # Sort results by multiplication result in descending order
#         results.sort(key=lambda x: x[1], reverse=True)

#         # Return sorted full paths
#         sorted_paths = [entry[0] for entry in results]
#         return sorted_paths

#     def read_metaData(self,res, filename, metaData, array, mDataDebug=False):
#         # res = filename.split('/')[-1]
#         print(res)
#         inputdir = filename

#         if not os.path.isdir(inputdir):
#             print(f"Empty folder: {inputdir}")
#             return -1

#         blockNamePrefix = inputdir + "/"
#         filename = inputdir + "/mdata.bin"

#         if not os.path.exists(filename):
#             print(f"{filename} does not exist")
#             return -1

#         with open(filename, "rb") as file:
#             mdata_version = struct.unpack("f", file.read(4))[0]
#             reference_V = struct.unpack("i", file.read(4))[0]
#             reference_H = struct.unpack("i", file.read(4))[0]
#             reference_D = struct.unpack("i", file.read(4))[0]

#             layer = {
#                 "vs_x": struct.unpack("f", file.read(4))[0],
#                 "vs_y": struct.unpack("f", file.read(4))[0],
#                 "vs_z": struct.unpack("f", file.read(4))[0],
#             }

#             # Re-reading layer.vs_x, layer.vs_y, layer.vs_z as per the original code
#             layer["vs_x"] = struct.unpack("f", file.read(4))[0]
#             layer["vs_y"] = struct.unpack("f", file.read(4))[0]
#             layer["vs_z"] = struct.unpack("f", file.read(4))[0]

#             org_V = struct.unpack("f", file.read(4))[0]
#             org_H = struct.unpack("f", file.read(4))[0]
#             org_D = struct.unpack("f", file.read(4))[0]

#             layer["dim_V"] = struct.unpack("I", file.read(4))[0]
#             layer["dim_H"] = struct.unpack("I", file.read(4))[0]
#             layer["dim_D"] = struct.unpack("I", file.read(4))[0]
#             layer["rows"] = struct.unpack("H", file.read(2))[0]
#             layer["cols"] = struct.unpack("H", file.read(2))[0]


#             sy = layer["dim_V"]
#             sx = layer["dim_H"]
#             sz = layer["dim_D"]
#             metaData[res,self.TimePoints - 1,self.Channels - 1, 'shape'] = (self.TimePoints, self.Channels, layer["dim_D"], layer['dim_V'], layer['dim_H'])
#             metaData[res,self.TimePoints - 1,self.Channels - 1, 'chunks'] = None
#             metaData[res,self.TimePoints - 1,self.Channels - 1, 'dtype'] ='uint16'
#             metaData[res,self.TimePoints - 1,self.Channels - 1, 'resolution'] = (layer['vs_z'], layer['vs_y'], layer['vs_x'])
#             metaData[res,self.TimePoints - 1,self.Channels - 1, 'ndim'] = len(metaData[res,self.TimePoints - 1,self.Channels -1 , 'shape'][2:])

#             # if mDataDebug:
#             print(f"filename {filename}")
#             print(f"meta.mdata_version {mdata_version}")
#             print(f"meta.reference_V {reference_V}")
#             print(f"meta.reference_H {reference_H}")
#             print(f"meta.reference_D {reference_D}")
#             print(f"layer.vs_x {layer['vs_x']}")
#             print(f"layer.vs_y {layer['vs_y']}")
#             print(f"layer.vs_z {layer['vs_z']}")
#             print(f"meta.org_V {org_V}")
#             print(f"meta.org_H {org_H}")
#             print(f"meta.org_D {org_D}")
#             print(f"layer.dim_V {layer['dim_V']}")
#             print(f"layer.dim_H {layer['dim_H']}")
#             print(f"layer.dim_D {layer['dim_D']}")
#             print(f"layer.rows {layer['rows']}")
#             print(f"layer.cols {layer['cols']}")

#             n = layer["rows"] * layer["cols"]
#             tree = {}
#             xoff, yoff, zoff = [], [], []
#             color = 0  # Placeholder for the color variable which is not defined in the original code
#             count = 0

#             for _ in range(n):
#                 # print(f'n = {_}')
#                 yxfolder = {
#                     "height": struct.unpack("I", file.read(4))[0],
#                     "width": struct.unpack("I", file.read(4))[0],
#                     "cubes": {},
#                 }
#                 layer["dim_D"] = struct.unpack("I", file.read(4))[0]
#                 yxfolder["ncubes"] = struct.unpack("I", file.read(4))[0]
#                 color = struct.unpack("I", file.read(4))[0]
#                 # print(struct.unpack('i', file.read(4))[0])
#                 yxfolder["offset_V"] = struct.unpack("i", file.read(4))[0]
#                 yxfolder["offset_H"] = struct.unpack("i", file.read(4))[0]
#                 yxfolder["lengthDirName"] = struct.unpack("H", file.read(2))[0]
#                 dirName = file.read(yxfolder["lengthDirName"]).decode(
#                     "utf-8", errors="ignore"
#                 )
#                 yxfolder["dirName"] = dirName
#                 if mDataDebug:
#                     print("...")
#                     print(f"HEIGHT {yxfolder['height']}")
#                     print(f"WIDTH {yxfolder['width']}")
#                     print(f"DEPTH {layer['dim_D']}")
#                     print(f"N_BLOCKS {yxfolder['ncubes']}")
#                     print(f"N_CHANS {color}")
#                     print(f"ABS_V {yxfolder['offset_V']}")
#                     print(f"ABS_H {yxfolder['offset_H']}")
#                     print(f"str_size {yxfolder['lengthDirName']}")
#                     print(f"DIR_NAME {yxfolder['dirName']}.")

#                 for _ in range(yxfolder["ncubes"]):
#                     try:
#                         # Read the length of the file name (unsigned short, 2 bytes)
#                         lengthFileName = struct.unpack("H", file.read(2))[0]

#                         # print(f'lengthFileName,{lengthFileName}')

#                         # Read the actual file name characters
#                         fileName = file.read(lengthFileName).decode(
#                             "utf-8", errors="ignore"
#                         )

#                         # print(fileName)

#                         # Read the actual file name characters
#                         # fileName = struct.unpack('c', file.read(lengthFileName))[0]
#                         depth = struct.unpack("I", file.read(4))[0]
#                         offset_D = struct.unpack("i", file.read(4))[0]
#                     except struct.error as e:
#                         print(f"Error reading cube data: {e}")
#                         return -1
#                     if metaData[res, self.TimePoints - 1, self.Channels -1,'chunks'] == None:
#                         # ===============>
#                         metaData[res,self.TimePoints -1 ,self.Channels -1, 'chunks'] = (depth,yxfolder['height'],yxfolder['width'])
#                     cube = {"fileName": fileName, "depth": depth, "offset_D": offset_D}
#                     yxfolder["cubes"][offset_D] = cube

#                     block = {
#                         "fileName": blockNamePrefix
#                         + yxfolder["dirName"]
#                         + "/"
#                         + cube["fileName"],
#                         "offset_H": yxfolder["offset_H"],
#                         "offset_V": yxfolder["offset_V"],
#                         "offset_D": cube["offset_D"],
#                         "width": yxfolder["width"],
#                         "height": yxfolder["height"],
#                         "depth": cube["depth"],
#                     }

#                     if count == 0:
#                         cubex = yxfolder["width"]
#                         cubey = yxfolder["height"]
#                         cubez = cube["depth"]
#                         count += 1

#                     if block["offset_H"] not in xoff:
#                         xoff.append(block["offset_H"])
#                     if block["offset_V"] not in yoff:
#                         yoff.append(block["offset_V"])
#                     if block["offset_D"] not in zoff:
#                         zoff.append(block["offset_D"])

#                     index = (
#                         block["offset_D"] * sx * sy
#                         + block["offset_V"] * sx
#                         + block["offset_H"]
#                     )
#                     tree[index] = block

#                     if mDataDebug:
#                         print("... ...")
#                         print(f"str_size {lengthFileName}")
#                         print(f"FILENAMES[{cube['offset_D']}] {cube['fileName']}.")
#                         print(f"BLOCK_SIZE+i {cube['depth']}")
#                         print(f"BLOCK_ABS_D+i {cube['offset_D']}")

#                 try:
#                     bytesPerVoxel = struct.unpack("I", file.read(4))[0]
#                 except struct.error as e:
#                     print(f"Error reading bytes per voxel: {e}")
#                     return -1

#                 if mDataDebug:
#                     print(f"N_BYTESxCHAN {bytesPerVoxel}")

#                 layer["yxfolders"] = {yxfolder["dirName"]: yxfolder}

#         return 0
#     def change_resolution_lock(self,ResolutionLevelLock):
#         self.ResolutionLevelLock = ResolutionLevelLock
#         self.shape = self.metaData[self.ResolutionLevelLock,0,0,'shape']
#         self.ndim = len(self.shape)
#         self.chunks = self.metaData[self.ResolutionLevelLock,0,0,'chunks']
#         self.resolution = self.metaData[self.ResolutionLevelLock,0,0,'resolution']
#         self.dtype = self.metaData[self.ResolutionLevelLock,0,0,'dtype']

#     def __getitem__(self,key):

#         res = 0 if self.ResolutionLevelLock is None else self.ResolutionLevelLock
#         print("check",key)
#         if isinstance(key,slice) == False and isinstance(key,int) == False and len(key) == 6:
#             res = key[0]
#             if res >= self.ResolutionLevels:
#                 raise ValueError('Layer is larger than the number of ResolutionLevels')
#             key = tuple([x for x in key[1::]])
#         print(res)
#         print(key)

#         if isinstance(key, int):
#             key = [slice(key,key+1)]
#             for _ in range(self.ndim-1):
#                 key.append(slice(None))
#             key = tuple(key)

#         if isinstance(key,tuple):
#             key = [slice(x,x+1) if isinstance(x,int) else x for x in key]
#             while len(key) < self.ndim:
#                 key.append(slice(None))
#             key = tuple(key)

#         print(key)
#         newKey = []
#         for ss in key:
#             if ss.start is None and isinstance(ss.stop,int):
#                 newKey.append(slice(ss.stop,ss.stop+1,ss.step))
#             else:
#                 newKey.append(ss)

#         key = tuple(newKey)
#         print(key)


#         array = self.getSlice(
#                         r=res,
#                         t = key[0],
#                         c = key[1],
#                         z = key[2],
#                         y = key[3],
#                         x = key[4]
#                         )

#         if self.squeeze:
#             return np.squeeze(array)
#         else:
#             return array

#     def getSlice(self,r,t,c,z,y,x):

#         '''
#         Access the requested slice based on resolution level and
#         5-dimentional (t,c,z,y,x) access to zarr array.
#         '''

#         incomingSlices = (r,t,c,z,y,x)
#         print(incomingSlices)
#         with tifffile.TiffFile('/home/kelin/Documents/BrainPI_image_data/terafly_data/mouseID_339952-17782/mouse17782_teraconvert.terafly/RES_875x1406x368_/000000/000000_000000/000000_000000_058880.tif', mode='r') as tif:
#             data = tif.asarray()


#             t = (self.Channels,) + data.shape
#             reshaped_array = np.reshape(data, t)
#             # print(reshaped_array)
#             print(reshaped_array.dtype)
#             return reshaped_array

#         # if self.cache is not None:
#         #     key = f'{self.location}_getSlice_{str(incomingSlices)}'
#         #     # key = self.location + '_getSlice_' + str(incomingSlices)
#         #     result = self.cache.get(key, default=None, retry=True)
#         #     if result is not None:
#         #         print(f'Returned from cache: {incomingSlices}')
#         #         return result

#         result = self.arrays[r][t,c,z,y,x]

#         # if self.cache is not None:
#         #     self.cache.set(key, result, expire=None, tag=self.location, retry=True)
#             # test = True
#             # while test:
#             #     # print('Caching slice')
#             #     self.cache.set(key, result, expire=None, tag=self.location, retry=True)
#             #     if result == self.getSlice(*incomingSlices):
#             #         test = False


#         return result
