
read_dirs = ['/data/model_data_himawari/sat_data_procseg']

writs_dir= '/tmp/data'
wnc_name = 'H08LATLON__TMON_2017_09__SDEG05_r%2d_c%2d.nc'

variables = [
    'dfb', 'slot', 'longitude', 'latitude', 'longitude_bounds', 'latitude_bounds',
    'latlon_coordinate_reference_system', 'nv'
]

channels = ['VIS064_2000', 'VIS160_2000', 'IR390_2000', 'IR124_2000']

bounds_radiance = {
    'VIS064_2000': [0,1],
    'VIS160_2000': [0,1],
    'IR390_2000': [200,300],
    'IR124_2000': [200,300]
}

non_common_variables = []
common_variables = []

selected_channels = []

print('Do you want all the channels? (1/0) \n')
if raw_input() == '1':
    selected_channels.extend(channels)
else:
    for channel in channels:
        print('Do you want ', channel, '? (1/0) \n')
        if raw_input() == '1':
            selected_channels.append(channel)

non_common_variables.extend(selected_channels)
selected_rows = []
selected_cols = []

# print('row/col or latitude/longitude? (rc/ll)')
# coord = raw_input()
# coord = coord.lower()
coord = 'll'

if coord == 'rc':

    print('Which cols do you want (eg: 58, or 51-58)?')
    arr = raw_input().split('-')
    if len(arr) == 1:
        selected_cols = arr
    elif len(arr) > 1:
        selected_cols = list(range(int(arr[0]), int(arr[-1])))
        non_common_variables.append('longitude')
        non_common_variables.append('longitude_bounds')

    print('Which rows do you want (eg: 6, or 5-9)?')
    arr = raw_input().split('-')
    if len(arr) == 1:
        selected_rows = arr
    elif len(arr) > 1:
        selected_rows = list(range(int(arr[0]), int(arr[-1])))

elif coord == 'll':

    print('Which latitude do you want (eg: 35-55)?')
    arr = raw_input().split('-')
    if len(arr) == 1:
        selected_rows = [(85 - int(arr[0])) // 5]
    else:
        selected_rows = list(range((85 - int(arr[0])) // 5, (85 - int(arr[-1]) // 5)))
        non_common_variables.append('latitude')
        non_common_variables.append('latitude_bounds')

    print('Which longitude do you want (eg: 125-140)?')
    arr = raw_input().split('-')
    if len(arr) == 1:
        selected_cols = [(180 + int(arr[0])) // 5]
    else:
        selected_rows = list(range((180 + int(arr[0]) // 5), (180 + int(arr[-1]) // 5)))

n = len(selected_rows)
m = len(selected_cols)
for k in range(n):
    if selected_rows[k] < 10:
        selected_rows[k] = '0' + str(selected_rows[k])
    else:
        selected_rows[k] = str(selected_rows[k])
for j in range(m):
    if selected_cols[j] < 10:
        selected_cols[j] = '0' + str(selected_cols[j])
    else:
        selected_cols[j] = str(selected_cols[j])

print('choose month? (ex: 01)')
month = raw_input().split('-')[0]

satellite = 'H08LATLON'
suffix_pattern = '__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'

chan_patterns = {}
for channel in selected_channels:
    chan_patterns[channel] = satellite + '_' + channel + suffix_pattern

for variable in variables:
    if variable not in non_common_variables:
        common_variables.append(variable)


from nclib2.dataset import DataSet
from nclib2.visualization import show_raw, visualize_map_3d
from draft import *


datasets = list()
print(chan_patterns)
for channel in chan_patterns:
    dataset = DataSet.read(dirs=read_dirs,
                            extent = {'latitude': {"start": 35., "end": 40.},
                                      'longitude': {"start": 125., "end": 130.},
                                      'slot': 50
                                      },
                            file_pattern = chan_patterns[channel],
                            variable_name = channel, fill_value=np.nan, interpolation=None, max_processes=0,
                           )

    # for day in dataset['data']:
    #     show_raw(day)
    from collections import namedtuple

    Bbox = namedtuple("Bbox", ("xmin", "ymin", "xmax", "ymax"))
    [vmin, vmax] = bounds_radiance[channel]
    visualize_map_3d(dataset['data'], Bbox(125, 35, 130, 40), vmin=vmin, vmax=vmax)


#
# dimensions = {
#     'dfb': None,
#     'slot': None,
#     'latitude': None,
#     'longitude': None,
#     'nv': None
# }
#
# common_variables_object, common_variables_content = {}, {}
# non_common_variables_object, non_common_variables_content = {}, {}
#
# for var in common_variables:
#     common_variables_object[var] = None
#     common_variables_content[var] = []
# for var in non_common_variables:
#     non_common_variables_object[var] = None
#     non_common_variables_content[var] = []
#
# datasets = list()
#
# # for filepath in filepaths:
# #     dataset = Dataset(filepath,'r')
# #     datasets.append(dataset)
# #     for dim_name in dimensions.keys():
# #         if dimensions[dim_name] == None:
# #             dimensions[dim_name] = dataset.dimensions[dim_name]
# #
# # for chan_number in range(len(chans)):
# #     filename = rnc_name % chans[chan_number]
# #     dataset = Dataset(rfolder+filename,'r')
# #     datasets.append(dataset)
# #
# #     for dim_name in dimensions.keys():
# #         if dimensions[dim_name] == None:
# #             dimensions[dim_name] = dataset.dimensions[dim_name]
# #
# #     for var_name in common_variables_object.keys():
# #         if common_variables_object[var_name] == None:
# #             common_variables_object[var_name] = dataset.variables[var_name]
# #     try:
# #         channels_variables_object[chans[chan_number]] = dataset.variables[chans[chan_number]]
# #     except 'KeyError':
# #         print 'channel ', chans[chan_number], ' not in this file',
# #
# #
# #     for var_content_name in variables_content.keys():
# #         try:
# #             if variables_content[var_content_name] == []:
# #                 variables_content[var_content_name].extend(dataset.variables[var_content_name][:])
# #         except:
# #             continue
# #
# #     for chan_content_name in channels_content.keys():
# #         try:
# #             channels_content[chan_content_name].extend(dataset.variables[chan_content_name][:, :, :])
# #         except:
# #             continue
# #
# #
# #
# # if True:
# #     rootgrp = Dataset(wfolder+wnc_name, 'w', 'NETCDF4')
# #     for dim_name in dimensions.keys():
# #         dimension = dimensions[dim_name]
# #         rootgrp.createDimension(dim_name, len(dimension) if not dimension.isunlimited() else None)
# #     for var_name in common_variables_object.keys():
# #         variable = common_variables_object[var_name]
# #         rootgrp.createVariable(var_name, variable.datatype, variable.dimensions)
# #     for chan_name in channels_variables_object.keys():
# #         rootgrp.createVariable(chan_name, channels_variables_object[chan_name].datatype, channels_variables_object[chan_name].dimensions)
# #     for var_name in variables_content.keys():
# #         rootgrp.variables[var_name][:] = variables_content[var_name]
# #     for chan_name in channels_content.keys():
# #         rootgrp.variables[chan_name][:,:,:] = channels_content[chan_name]
# #     rootgrp.close()
# #
# # for data_number in range(len(datasets)):
# #     datasets[data_number].close()
