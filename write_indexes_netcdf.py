def write(type_chan, variables_definitions_, variables_data_, dfbs, slots, latitudes, longitudes):
    from nclib2.dataset import DataSet, DC, NCError, WritingError
    from read_metadata import read_indexes_dir_and_pattern
    dir, pattern = read_indexes_dir_and_pattern(type_chan)

    try:
        ds = DataSet.create(file_pattern=pattern,
                            globals_definition=dict(DC.CF16_DEFAULT_GLOBALS_DEFINITION,
                                                    **{"title": "Computed visible indexes", "source": "PED", "comments": "For improved visualization"}),
                            dimension_definition={
                                "latitude": latitudes,
                                "longitude": longitudes,
                                "slot": slots,
                                "dfb": dfbs,
                            },
                            variables_definition=variables_definitions_,
                            dir=dir,
                            overwrite_level=2)

        # list files created or skipped during file creation
        for file_ in ds.files_involved_list():
            print("file %s exits" % file_)
    except NCError as e:
        print("Some problem occured", type(e), e)


    try:
        for var in variables_data_:
            DataSet.write(dir=dir,
                file_pattern=pattern,
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

    type_output = 'visible'   # infrared, visible, classes

    latitude_beginning = 40.  # salt lake mongolia  45.
    latitude_end = 45.
    longitude_beginning = 125.
    longitude_end = 130.
    dfb_beginning = 13544
    nb_dfbs = 10
    satellite_timestep = 10
    slot_step_ = 1
    nb_slots_per_day = get_nb_slots_per_day(satellite_timestep, slot_step_)
    # increase slotstep to get a divisor of nb_slots_per_day
    slot_step_ = upper_divisor_slot_step(slot_step_, nb_slots_per_day)

    dfb_ending = dfb_beginning+nb_dfbs-1

    times = get_times(dfb_beginning, dfb_ending, satellite_timestep, slot_step_)

    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)

    nb_latitudes, nb_longitudes = len(latitudes), len(longitudes)
    dfbs, slots = get_dfbs_slots(dfb_beginning, dfb_ending, satellite_timestep, slot_step_)

    if type_output == 'infrared':
        features = get_features(
            type_output,
            latitudes,
            longitudes,
            dfb_beginning,
            dfb_ending,
            compute_indexes=True,
            slot_step=slot_step_,
            normalize=False,
            weights=None,
            return_m_s=False,
        )

        features = np.flip(features, axis=1)

        big_clouds = features[:, :, :, 0]
        var_difference = features[:, :, :, 1]
        warm = features[:, :, :, 2]
        cold = features[:, :, :, 3]

        big_clouds = np.reshape(big_clouds, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))
        var_difference = np.reshape(var_difference, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))
        warm = np.reshape(warm, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))
        cold = np.reshape(cold, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))


        variables_definitions_cli = {
            "Big_clouds": {"_FillValue": -999., "units": "no unit", "long_name": "mask with hot water clouds",
                     "datatype": "f8",
                     "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                     "grid_mapping": "coordinate_reference_system",
                     "dimensions": ("dfb", "slot", "latitude", "longitude")},
            "Var_difference": {"_FillValue": -999., "units": "no unit", "long_name": "5 days variability difference (for small hot clouds)",
                     "datatype": "f8",
                     "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                     "grid_mapping": "coordinate_reference_system",
                     "dimensions": ("dfb", "slot", "latitude", "longitude")},
            "Warm": {"_FillValue": -999., "units": "no unit", "long_name": "Warm mask",
                               "datatype": "f8",
                               "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                               "grid_mapping": "coordinate_reference_system",
                               "dimensions": ("dfb", "slot", "latitude", "longitude")},
            "Cold": {"_FillValue": -999., "units": "no unit", "long_name": "Cold mask",
                     "datatype": "f8",
                     "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                     "grid_mapping": "coordinate_reference_system",
                     "dimensions": ("dfb", "slot", "latitude", "longitude")},
        }

        variables_cli = {
            "Big_clouds": big_clouds,
            "Var_difference": var_difference,
            "Warm": warm,
            "Cold": cold
        }

        write('infrared', variables_definitions_cli, variables_cli, dfbs, slots, latitudes, longitudes)

    if type_output == 'visible':

        features = get_features(
            type_output,
            latitudes,
            longitudes,
            dfb_beginning,
            dfb_ending,
            compute_indexes=True,
            slot_step=slot_step_,
            normalize=False,
            weights=None,
            return_m_s=False,
        )

        features = np.flip(features, axis=1)

        ndsi = features[:, :, :, 0]
        var_ndsi = features[:, :, :, 1]
        cloudy_sea = features[:, :, :, 2]

        ndsi = np.reshape(ndsi, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))
        var_ndsi = np.reshape(var_ndsi, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))
        cloudy_sea = np.reshape(cloudy_sea, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))

        variables_ndsi = {
            "Snow_index": ndsi,
            "Var_snow_index": var_ndsi,
            "Cloudy_sea": cloudy_sea
        }

        variables_definitions_ndsi = {
            "Snow_index": {"_FillValue": -999., "units": "no unit", "long_name": "snow index",
                     "datatype": "f8",
                     "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                     "grid_mapping": "coordinate_reference_system",
                     "dimensions": ("dfb", "slot", "latitude", "longitude")},
            "Var_snow_index": {"_FillValue": -999., "units": "no unit", "long_name": "5 days variability of snow index",
                           "datatype": "f8",
                           "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                           "grid_mapping": "coordinate_reference_system",
                           "dimensions": ("dfb", "slot", "latitude", "longitude")},
            "Cloudy_sea": {"_FillValue": -999., "units": "no unit", "long_name": "Cloudy sea",
                              "datatype": "f8",
                              "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                              "grid_mapping": "coordinate_reference_system",
                              "dimensions": ("dfb", "slot", "latitude", "longitude")},
        }

        write('visible', variables_definitions_ndsi, variables_ndsi, dfbs, slots, latitudes, longitudes)

    if type_output == 'classes':

        from decision_tree import *

        classes = get_classes_v1_point(
            latitudes,
            longitudes,
            dfb_beginning,
            dfb_ending,
            slot_step=slot_step_
        )

        classes = np.flip(classes, axis=1)

        classes = np.reshape(classes, (nb_dfbs, nb_slots_per_day, nb_latitudes, nb_longitudes))

        variables_definitions_classes = {
            "Classes": {"_FillValue": -999., "units": "no unit", "long_name": "Decision tree classification",
                     "datatype": "f8",
                     "cell_methods": "time: mean (interval: 1 day comment: hourly sum averages) latitude: mean longitude: mean",
                     "grid_mapping": "coordinate_reference_system",
                     "dimensions": ("dfb", "slot", "latitude", "longitude")},
        }

        variables_classes = {
            "Classes": classes,
        }

        write('classes', variables_definitions_classes, variables_classes, dfbs, slots, latitudes, longitudes)


