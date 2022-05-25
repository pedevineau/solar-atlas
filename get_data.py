from scipy.stats import linregress

from infrared_predictors import infrared_outputs
from read_metadata import read_satellite_step, read_channels_names
from read_netcdf import read_channels, read_land_mask, read_temperature_forecast
from utils import *
from visible_predictors import visible_outputs


# precomputing data and indexes
def mask_all_sun_outliers(sun_zenith_angle):
    """
    Return a boolean matrix indicating if the sun position is between sunset line and the line 10 degrees before sunset

    :param sun_zenith_angle: 3-d array (time, latitude, longitude) indicating the angle between the position of
        the sun observed from a pixel and the relative zenith position
    :return: (boolean 3-d array) of shape (time, latitude, longitude) indicating if the sun is between the sunrise
        and the position 10 degree before sunset
    """
    maskup = sun_zenith_angle > 350
    maskdown = sun_zenith_angle < 0
    masknan = np.isnan(sun_zenith_angle) | np.isinf(sun_zenith_angle)
    mask = maskup | maskdown | masknan
    return mask


def mask_one_sun_outlier(angle):
    """
    Return a boolean indicating if the sun position is between sunset line and the line 10 degrees before sunset

    :param angle:
    :return: (bool) indicating if the sun is between the sunrise and the position 10 degree before sunset
    """
    return np.isnan(angle) or np.isinf(angle) or angle > 350 or angle < 0


def interpolate_the_missing_slots(
    satellite_measurements,
    missing_slots_list_1,
    missing_slots_list_2,
    missing_slots_list_3,
    interpolation,
):  # interpolation: 'keep-last', 'linear', 'none'
    """
    Return sensors measurements where data has been interpolated when its missing during at maximum 3 consecutiv slots

    :param satellite_measurements: 4-d array (time, latitude, longitude, channel) with possible missing data due to
        sensors restart
    :param missing_slots_list_1: list of slot indexes where only one consecutive slot is missing
    :param missing_slots_list_1: list of slot indexes where 2 consecutive slot are missing
    :param missing_slots_list_1: list of slot indexes where 2 consecutive slot are missing
    :param interpolation: (str) name of chosen interpolation method. Possibilities so far: `keep-last`, `linear`, `none`
    :return: 4-d array (time, latitude, longitude, channel)
    """
    assert interpolation in ["keep-last", "linear", "none"], (
        "Only acceptable options for method "
        "`interpolate_the_missing_slots` are `keep-last`, `linear`"
        " or `none`, not {}".format(option)
    )

    for slot in missing_slots_list_1:
        if interpolation == "linear":
            satellite_measurements[slot] = 0.5 * (
                satellite_measurements[slot - 1] + satellite_measurements[slot + 1]
            )
        elif interpolation == "keep-last":
            satellite_measurements[slot] = satellite_measurements[slot - 1]
    for slot in missing_slots_list_2:
        if interpolation == "linear":
            satellite_measurements[slot] = (
                satellite_measurements[slot - 1] * 2.0
                + satellite_measurements[slot + 2]
            ) / 3.0
            satellite_measurements[slot + 1] = (
                satellite_measurements[slot - 1]
                + satellite_measurements[slot + 2] * 2.0
            ) / 3.0
        elif interpolation == "keep-last":
            satellite_measurements[slot] = satellite_measurements[slot - 1]
            satellite_measurements[slot + 1] = satellite_measurements[slot - 1]
    for slot in missing_slots_list_3:
        if interpolation == "linear":
            satellite_measurements[slot] = (
                satellite_measurements[slot - 1] * 3.0
                + satellite_measurements[slot + 3]
            ) / 4.0
            satellite_measurements[slot + 1] = (
                satellite_measurements[slot - 1] * 2.0
                + satellite_measurements[slot + 3] * 2
            ) / 4.0
            satellite_measurements[slot + 2] = (
                satellite_measurements[slot - 1] * 1.0
                + satellite_measurements[slot + 3] * 3
            ) / 4.0
        elif interpolation == "keep-last":
            satellite_measurements[slot] = satellite_measurements[slot - 1]
            satellite_measurements[slot + 1] = satellite_measurements[slot - 1]
            satellite_measurements[slot + 2] = satellite_measurements[slot - 1]
    return satellite_measurements


# get list of isolated slots
def get_list_isolated_missing_slots(array, maximal_scope=3):
    mask = (
        mask_all_sun_outliers(array[:, 0, 0])
        & mask_all_sun_outliers(array[:, -1, -1])
        & mask_all_sun_outliers(array[:, -1, 0])
        & mask_all_sun_outliers(array[:, 0, -1])
    )

    # indexes list of isolated slots
    indexes_isolated_1, indexes_isolated_2, indexes_isolated_3 = [], [], []
    # indexes list of dawn slots to be removed
    # dawn_indexes = []
    if maximal_scope >= 1:
        for k in range(1, len(mask) - 1):
            if mask[k] and not mask[k - 1] and not mask[k + 1]:
                indexes_isolated_1.append(k)
    if maximal_scope >= 2:
        for k in range(1, len(mask) - 2):
            if mask[k] and mask[k + 1] and not mask[k - 1] and not mask[k + 2]:
                indexes_isolated_2.append(
                    k
                )  # keep only the index of the first of the two consecutive missing slots
    if maximal_scope >= 3:
        for k in range(1, len(mask) - 3):
            if (
                mask[k]
                and mask[k + 1]
                and mask[k + 2]
                and not mask[k - 1]
                and not mask[k + 3]
            ):
                indexes_isolated_3.append(k)
    if maximal_scope >= 4:
        raise Exception("the maximal interpolation scope allowed is 3")
    return indexes_isolated_1, indexes_isolated_2, indexes_isolated_3


def mask_channels(satellite_measurements):
    """
    Return a boolean array of pixels to mask due to invalid satellite sensors measurements

    :param satellite_measurements: 4-D array (time, latitude, longitude, channels) measured with satellite sensors
    :return: (boolean 3-D array) (time, latitude, longitude) matrix indicating the pixels where measurements are invalid
    """
    (nb_slots, nb_latitudes, nb_longitudes, nb_channels) = np.shape(
        satellite_measurements
    )

    mask = np.zeros((nb_slots, nb_latitudes, nb_longitudes), dtype=bool)
    for chan in range(nb_channels):
        (
            slots_to_interpolate_1,
            slots_to_interpolate_2,
            slots_to_interpolate_3,
        ) = get_list_isolated_missing_slots(satellite_measurements[:, :, :, chan])
        # filter isolated nan and aberrant
        satellite_measurements[:, :, :, chan] = interpolate_the_missing_slots(
            satellite_measurements[:, :, :, chan],
            slots_to_interpolate_1,
            slots_to_interpolate_2,
            slots_to_interpolate_3,
            interpolation="linear",
        )
        # get mask for non isolated nan and aberrant
        mask_current_channels = mask_all_sun_outliers(
            satellite_measurements[:, :, :, chan]
        )
        mask = mask | mask_current_channels

    satellite_measurements[mask] = -1
    return satellite_measurements, mask


def apply_smooth_threshold(x, th, order=2):
    return np.exp(-(x - th))


def compute_variability(
    cloud_index,
    mask=None,
    cos_zen=None,
    step=1,
    return_mask=False,
    abs_value=False,
    negative_variation_only=False,
    option="default",  # ['default, 'without-bias']
    normalization="none",
):
    """
    Return cloud variability index based on cloud index maps. This method can apply various normalization techniques,
    and remove measurement bias due to sun position

    :param cloud_index: array (slots, latitudes, longitudes) with cloud index (cli or unbiased difference)
    :param mask: boolean array (slots, latitudes, longitudes) indicating which measurements are valid
    :return: array (slots, latitudes, longitudes) with cloud variability index between consecutive cloud index maps
    """
    assert option in ["default", "without-bias"], (
        "Only acceptable options for method `compute_variability` are "
        "`default` and `without-bias`, not {}".format(option)
    )
    step_left = step
    if option == "without-bias":
        try:
            cloud_index = remove_cos_zen_correlation(cloud_index, cos_zen, mask)
        except:
            raise Exception("Cos-zen is compulsory to compute without-bias")
    previous = np.roll(cloud_index, step_left, axis=0)
    cloud_index = cloud_index - previous
    if mask is not None:
        mask = mask + np.roll(mask, step_left, axis=0)
        if step_left >= 0:
            mask[:step_left] = True
        else:
            mask[step_left:] = True
    if negative_variation_only:
        cloud_index[cloud_index > 0] = 0
        cloud_index[~mask] *= -1
    if abs_value:
        cloud_index = np.abs(cloud_index)
    if mask is not None:
        cloud_index[mask] = -10
    if normalization != "none":
        cloud_index = normalize(cloud_index, mask=mask, normalization=normalization)
    if return_mask:
        return cloud_index, mask
    else:
        return cloud_index


# def compute_intra_variability(*groups):
#     intra = []
#     inter = 0
#     for group in groups:


def remove_cos_zen_correlation(index, cos_zen, mask, pre_mask=None):
    """
    Return a 3-d array (time, latitude, longitude) of measurements where artefacts due to sun position has been removed

    :param index: 3-d array representing the variable to test
    :param cos_zen: 3-d array with cos of zenith angle
    :param mask: mask associated with the variable
    :param pre_mask: mask covering points to exclude in linear regression computing (eg: cloudy pre-mask)
    :return: the index minus its average cos(zenith) component
    """
    to_return = index.copy()
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(to_return)
    if pre_mask is not None:
        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slice_mask = mask[:, lat, lon]
                slice_pre_mask = pre_mask[:, lat, lon]
                slice_total_mask = slice_mask | slice_pre_mask
                if not np.all(slice_total_mask):
                    # exclude pre-masked values from slope computing, but remove the cos-zen component all the same
                    slope = linregress(
                        cos_zen[:, lat, lon][~slice_total_mask],
                        index[:, lat, lon][~slice_total_mask],
                    )[0]
                    to_return[:, lat, lon][~slice_mask] -= (
                        slope * cos_zen[:, lat, lon][~slice_mask]
                    )
    else:
        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slice_mask = mask[:, lat, lon]
                if not np.all(slice_mask):
                    slope = linregress(
                        cos_zen[:, lat, lon][~slice_mask],
                        index[:, lat, lon][~slice_mask],
                    )[0]
                    to_return[:, lat, lon][~slice_mask] -= (
                        slope * cos_zen[:, lat, lon][~slice_mask]
                    )
    return to_return


def get_variability_array_modified(
    array,
    mask,
    step=1,
    th_1=0.02,
    th_2=0.3,
    positive_variation=True,
    negative_variation=True,
):
    return compute_variability(array, mask=mask, step=step)[0]


def get_features(
    type_channels,
    latitudes,
    longitudes,
    dfb_beginning,
    dfb_ending,
    output_level,
    slot_step=1,
    gray_scale=False,
):
    """
    Wrapper returning the predictors for cloud-classification model, using the desired type of inpit light channels

    :param type_channels:
    :param latitudes:
    :param longitudes:
    :param dfb_beginning:
    :param dfb_ending:
    :param output_level:
    :param slot_step:
    :param gray_scale:
    :return:
    """
    satellite_step = read_satellite_step()

    is_land = read_land_mask(latitudes, longitudes)
    temperatures = read_temperature_forecast(
        latitudes, longitudes, dfb_beginning, dfb_ending
    )
    times = get_times_utc(dfb_beginning, dfb_ending, satellite_step, slot_step)

    channels = read_channels_names(type_channels)
    if type_channels == "visible":
        content_visible = read_channels(
            channels, latitudes, longitudes, dfb_beginning, dfb_ending, slot_step
        )
        return visible_outputs(
            times,
            latitudes,
            longitudes,
            is_land,
            content_visible,
            satellite_step,
            slot_step,
            output_level,
            gray_scale,
        )

    elif type_channels == "infrared":
        content_infrared = read_channels(
            channels, latitudes, longitudes, dfb_beginning, dfb_ending, slot_step
        )
        return infrared_outputs(
            times,
            latitudes,
            longitudes,
            temperatures,
            content_infrared,
            satellite_step,
            slot_step,
            output_level,
            gray_scale,
        )
    else:
        raise AttributeError("The type of channels should be 'visible' or 'infrared'")
