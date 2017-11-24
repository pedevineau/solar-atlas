
def write(variables_definitions_, variables_data_, dfbs, slots, latitudes, longitudes):
    from nclib2.dataset import DataSet, DC, NCError, WritingError
    dir_writing = '/data/computed'
    pattern_writing = 'H08LATLON_VISIBLE_INDEXES_2000__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc'
    try:
        ds = DataSet.create(file_pattern=pattern_writing,
                            globals_definition=dict(DC.CF16_DEFAULT_GLOBALS_DEFINITION,
                                                    **{"title": "Computed visible indexes", "source": "PED", "comments": "For improved visualization"}),
                            dimension_definition={
                                "latitude": latitudes,
                                "longitude": longitudes,
                                "slot": slots,
                                "dfb": dfbs,
                            },
                            variables_definition=variables_definitions_,
                            dir=dir_writing,
                            overwrite_level=2)

        # list files created or skipped during file creation
        for file_ in ds.files_involved_list():
            print("file %s exits" % file_)
    except NCError as e:
        print("Some problem occured", type(e), e)


    try:
        for var in variables_data_:
            DataSet.write(dir=dir_writing,
                file_pattern=pattern_writing,
                variable_name=var,
                data_array=variables_data_[var],
                extent={
                    "latitude": latitudes,
                    "longitude": longitudes,
                    "slot": slots,
                    "dfb": dfbs,
                    },
                dimensions_order=["dfb", "slot", "latitude", "longitude"])
    except WritingError as e:
        print("writing error", type(e), e)
    except NCError as e:
        print("Unspecific problem occured", type(e), e)


if __name__ == '__main__':
    from get_data import get_features
    from utils import *
    from numpy import reshape

    latitude_beginning = 35.+10  # salt lake mongolia  45.
    latitude_end = 40.+15
    longitude_beginning = 125.
    longitude_end = 130.
    dfb_beginning = 13516+30
    nb_dfbs = 1
    satellite_timestep = 10
    nb_slots_per_day = get_nb_slots_per_day(satellite_timestep)
    slot_step_ = 1
    # increase slotstep to get a divisor of nb_slots_per_day
    slot_step_ = upper_divisor_slot_step(slot_step_, nb_slots_per_day)

    dfb_ending = dfb_beginning+nb_dfbs-1

    times = get_times(dfb_beginning, dfb_ending, satellite_timestep, slot_step_)

    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    features_visible = get_features(
        'visible',
        latitudes,
        longitudes,
        dfb_beginning,
        dfb_ending,
        compute_indexes=True,
        slot_step=slot_step_,
        normalize=False,
        normalization='standard',
        weights=None,
        return_m_s=False,
        return_mu=False,
        dirs=['/data/model_data_himawari/sat_data_procseg'],
        satellite='H08LATLON',
        pattern_suffix='__TMON_{YYYY}_{mm}__SDEG05_r{SDEG5_LATITUDE}_c{SDEG5_LONGITUDE}.nc',
        satellite_frequency=satellite_timestep,
        dir_ocean='/data/ocean_shape',
        pattern_ocean='landsea_mask_2arcmin.nc'
    )
    ndsi = features_visible[:, :, :, 0]
    stressed_ndsi = features_visible[:, :, :, 1]

    nb_latitudes, nb_longitudes = len(latitudes), len(longitudes)
    dfbs, slots = get_dfbs_slots(dfb_beginning, dfb_ending, satellite_timestep, slot_step_)

    ndsi = reshape(ndsi, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))
    stressed_ndsi = reshape(stressed_ndsi, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))

    from numpy import *
    print shape(ndsi
                )

    variables_ndsi = {
        "NDSI": ndsi,
        "Stressed_NDSI": stressed_ndsi
    }

    variables_definitions_ndsi = {
        "NDSI": {"_FillValue": -999., "units": "no unit", "long_name": "snow index",
                 "datatype": "f8",
                 "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                 "grid_mapping": "coordinate_reference_system",
                 "dimensions": ("dfb", "slot", "latitude", "longitude")},
        "Stressed_NDSI": {"_FillValue": -999., "units": "no unit", "long_name": "1-period stressed snow index",
                 "datatype": "f8",
                 "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                 "grid_mapping": "coordinate_reference_system",
                 "dimensions": ("dfb", "slot", "latitude", "longitude")},
    }

    # variables_cli = {
    #     "CLI": cli,
    #     "Stressed_CLI": stressed_cli
    # }
    #
    # variables_definitions_cli = {
    #     "CLI": {"_FillValue": -999., "units": "no unit", "long_name": "cloud index",
    #              "datatype": "f8",
    #              "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
    #              "grid_mapping": "coordinate_reference_system",
    #              "dimensions": ("dfb", "slot", "latitude", "longitude")},
    #     "Stressed_CLI": {"_FillValue": -999., "units": "no unit", "long_name": "stressed cloud index",
    #              "datatype": "f8",
    #              "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
    #              "grid_mapping": "coordinate_reference_system",
    #              "dimensions": ("dfb", "slot", "latitude", "longitude")},
    # }

    write(variables_definitions_ndsi, variables_ndsi, dfbs, slots, latitudes, longitudes)
