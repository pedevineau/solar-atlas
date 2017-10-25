from nclib2.dataset import DataSet, np
from nclib2.visualization import *

dir = '/data/model_data_himawari/sat_data_procseg'
pattern = 'H08LATLON_VIS064_2000__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'

a = DataSet.read(dirs=[dir],
    file_pattern=pattern,
    variable_name="VIS064_2000", fill_value=np.nan, interpolation=None, max_processes=0,
    extent={
        # "latitude": 40,
        # "longitude": {"start": 125., "end": 140.},
        "dfb": 13759,
        "slot": {"start": 1, "end": 16},
    }, dimensions_order=("dfb", "slot", "latitude", "longitude"),
    allow_masked=True,
    )  # reading GHI_2015_5_mtsat_c59_r13.nc
array = a["data"]  # axes order in data: dfb, slot, latitude, longitude
print a['latitude']['enumeration']
# print(type(array))
# array[array.mask] =0
# array.mask= False
# from collections import namedtuple
# Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
# visualize_map_3d(array,Bbox(-180,-90,180,90))

# print(array)
# show_raw()