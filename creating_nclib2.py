dir_ = '/tmp/data'
pattern = '543_VIS064_2000__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'

from nclib2.dataset import DataSet, DC, np, NCError

try:
    ds = DataSet.create(file_pattern=pattern,
                        globals_definition=dict(DC.CF16_DEFAULT_GLOBALS_DEFINITION,
                                                **{"title": "haha", "source": "me", "comments": "haha"}),
                        dimension_definition={
                            "latitude": {"extent": {"start": 88., "end": +90., "step": 0.1}, "sadd": "sad"},
                            "longitude": np.arange(-180. + 0.3 / 2, -178. - 0.3 / 2, 0.3),
                            "dfb": range(1, 4),
                            "slot": range(1, 10),
                        },
                        variables_definition={
                            "VIS": {"_FillValue": -999., "units": "m", "long_name": "height map",
                                            "datatype": "f8",
                                            "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                                            "grid_mapping": "coordinate_reference_system",
                                            "dimensions": ("slot", "dfb", "latitude", "longitude")}},
                        dir=dir_,
                        overwrite_level=2)

    # list files created or skipped during file creation
    for file_ in ds.files_involved_list():
        print("file %s exits" % file_)
except NCError as e:
    print("Some problem occured", type(e), e)



ndArray = np.zeros((31,9,50, 17))
try:
    DataSet.write(dir=dir_,
        file_pattern=pattern,
        variable_name="VIS",
        data_array=ndArray,
        extent={"latitude": {"start": 88., "end": +90., "step": 0.2},
                "longitude": np.arange(-180. + 0.3 / 2, -178. - 0.3 / 2, 0.3),
                "dfb": range(1, 4),
                "slot": range(1, 10),
            # "dfb": range(13070, 13080),
            },
        dimensions_order=["dfb", "slot", "latitude", "longitude"])
except WritingError as e:
    print("writing error", type(e), e)
except NCError as e:
    print("Unspecific problem occured", type(e), e)

