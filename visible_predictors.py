from angles_geom import get_likelihood_variable_cos_zen
from get_data import compute_variability
from get_data import mask_channels
from static_tests import dawn_day_test, sea_coasts_cloud_test
from utils import *


def visible_outputs(
    times,
    latitudes,
    longitudes,
    is_land,
    content_visible,
    satellite_step,
    slot_step,
    output_level="abstract",
    gray_scale=False,
):
    content_visible, mask_input = mask_channels(content_visible)

    if output_level == "channel":
        if not gray_scale:
            return content_visible
        else:
            nb_channels = np.shape(content_visible)[-1]
            for chan in range(nb_channels):
                content_visible[:, :, :, chan] = normalize(
                    content_visible[:, :, :, chan], mask_input, "gray-scale"
                )
            return np.asarray(content_visible, dtype=np.uint8)

    elif output_level == "ndsi":
        zen, vis, ndsi = get_zen_vis_ndsi(times, latitudes, longitudes, content_visible)
        return zen, vis, ndsi, mask_input
    else:
        zen, vis, ndsi = get_zen_vis_ndsi(times, latitudes, longitudes, content_visible)
        del content_visible
        return visible_abstract_predictors(
            zen, is_land, vis, mask_input, ndsi, satellite_step, slot_step, gray_scale
        )


def visible_abstract_predictors(
    zen, is_land, vis, mask_input, ndsi, satellite_step, slot_step, gray_scale
):
    mask_output = ~dawn_day_test(zen) | mask_input | ~is_land
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(vis)

    nb_features = (
        4  # snow index, negative variability, positive variability, cloudy sea
    )
    if not gray_scale:
        array_indexes = np.empty(
            shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features)
        )
        array_indexes[:, :, :, 3] = sea_coasts_cloud_test(zen, is_land, vis)
        array_indexes[:, :, :, 1] = get_bright_negative_variability_5d(
            ndsi, mask_output, satellite_step, slot_step
        )
        array_indexes[:, :, :, 2] = get_bright_positive_variability_5d(
            ndsi, mask_output, satellite_step, slot_step
        )
        ndsi[mask_output] = -10
        array_indexes[:, :, :, 0] = ndsi
    else:
        array_indexes = np.empty(
            shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features), dtype=np.uint8
        )
        array_indexes[:, :, :, 3] = sea_coasts_cloud_test(zen, is_land, vis)
        array_indexes[:, :, :, 2] = normalize(
            get_bright_negative_variability_5d(
                ndsi, mask_output, satellite_step, slot_step
            ),
            mask_output,
            normalization="gray-scale",
        )
        array_indexes[:, :, :, 2][mask_output] = 0
        array_indexes[:, :, :, 1] = normalize(
            get_bright_positive_variability_5d(
                ndsi, mask_output, satellite_step, slot_step
            ),
            mask_output,
            normalization="gray-scale",
        )
        array_indexes[:, :, :, 1][mask_output] = 0
        array_indexes[:, :, :, 0] = normalize(
            ndsi, mask_output, normalization="gray-scale"
        )
        array_indexes[:, :, :, 2][mask_output] = 0
    return array_indexes


def get_zen_vis_ndsi(times, latitudes, longitudes, content_visible):
    """
    all useful data for static snow test (except cli)
    :param times:
    :param latitudes:
    :param longitudes:
    :param content_visible:
    :return:
    """
    from angles_geom import get_zenith_angle

    zen = get_zenith_angle(times, latitudes, longitudes)
    ndsi = get_snow_index(
        vis=content_visible[:, :, :, 1],
        sir=content_visible[:, :, :, 0],
        zen=zen,
        threshold_denominator=0.02,
        index="ndsi-zenith",
    )
    return zen, content_visible[:, :, :, 1], ndsi


def get_snow_index(vis, sir, zen, threshold_denominator, index):
    if index == "ndsi":
        # sir *= 5
        ndsi = (vis - sir) / np.maximum(sir + vis, threshold_denominator)
    elif index == "ndsi-zenith":
        ndsi = (vis - sir) / np.maximum(
            sir + vis, threshold_denominator
        ) + 0.15 * np.square(1 - np.cos(zen))
    else:
        ndsi = vis / np.maximum(sir, threshold_denominator)
    return ndsi


def get_bright_negative_variability_5d(
    index, definition_mask, satellite_step, slot_step
):
    """
    To recognize covered snow (drop of ndsi compared to normal)
    NB: we loose information about the first slot (resp the last slot) if night is 1 slot longer during 1 of the 5 days
    :param index:
    :param definition_mask:
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :return:
    """
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    nb_days = np.shape(index)[0] / nb_slots_per_day
    to_return = np.full_like(index, -10)
    if nb_days >= 2:
        var_ndsi_1d_past = compute_variability(
            cloud_index=index,
            mask=definition_mask,
            step=nb_slots_per_day,
            negative_variation_only=True,
            abs_value=False,
        )

        var_ndsi_1d_future = compute_variability(
            cloud_index=index,
            mask=definition_mask,
            step=-nb_slots_per_day,
            negative_variation_only=True,
            abs_value=False,
        )
        if nb_days == 2:
            to_return[:nb_slots_per_day] = var_ndsi_1d_future[:nb_slots_per_day]
            to_return[nb_slots_per_day:] = var_ndsi_1d_past[nb_slots_per_day:]
        else:  # nb_days >=3
            var_ndsi_2d_past = compute_variability(
                cloud_index=index,
                mask=definition_mask,
                step=nb_slots_per_day * 2,
                negative_variation_only=True,
                abs_value=False,
            )
            var_ndsi_2d_future = compute_variability(
                cloud_index=index,
                mask=definition_mask,
                step=-2 * nb_slots_per_day,
                negative_variation_only=True,
                abs_value=False,
            )
            # first day
            to_return[:nb_slots_per_day] = np.maximum(
                var_ndsi_1d_future[:nb_slots_per_day],
                var_ndsi_2d_future[:nb_slots_per_day],
            )
            # last day
            to_return[-nb_slots_per_day:] = np.maximum(
                var_ndsi_1d_past[-nb_slots_per_day:],
                var_ndsi_2d_past[-nb_slots_per_day:],
            )

            if nb_days == 3:
                # second day
                to_return[nb_slots_per_day : 2 * nb_slots_per_day] = np.maximum(
                    var_ndsi_1d_past[nb_slots_per_day : 2 * nb_slots_per_day],
                    var_ndsi_1d_future[nb_slots_per_day : 2 * nb_slots_per_day],
                )
            else:  # nb_days >= 4
                # the day previous the last one
                to_return[-2 * nb_slots_per_day : -nb_slots_per_day] = np.maximum(
                    np.maximum(
                        var_ndsi_1d_past[-2 * nb_slots_per_day : -nb_slots_per_day],
                        var_ndsi_2d_past[-2 * nb_slots_per_day : -nb_slots_per_day],
                    ),
                    var_ndsi_1d_future[-2 * nb_slots_per_day : -nb_slots_per_day],
                )
                # second day
                to_return[nb_slots_per_day : 2 * nb_slots_per_day] = np.maximum(
                    np.maximum(
                        var_ndsi_1d_future[nb_slots_per_day : 2 * nb_slots_per_day],
                        var_ndsi_2d_future[nb_slots_per_day : 2 * nb_slots_per_day],
                    ),
                    var_ndsi_1d_past[nb_slots_per_day : 2 * nb_slots_per_day],
                )
                if nb_days >= 5:
                    to_return[
                        2 * nb_slots_per_day : -2 * nb_slots_per_day
                    ] = np.maximum(
                        np.maximum(
                            var_ndsi_1d_past[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                            var_ndsi_2d_past[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                        ),
                        np.maximum(
                            var_ndsi_1d_future[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                            var_ndsi_2d_future[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                        ),
                    )
    to_return[to_return < 0] = 0
    return to_return


def get_bright_positive_variability_5d(
    index, definition_mask, satellite_step, slot_step
):
    """
    To recognize some icy clouds (peaks of ndsi)
    NB: we loose information about the first slot (resp the last slot) if night is 1 slot longer during 1 of the 5 days
    :param index:
    :param definition_mask:
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :return:
    """
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    nb_days = np.shape(index)[0] / nb_slots_per_day
    to_return = np.full_like(index, -10)
    if nb_days >= 2:
        var_ndsi_1d_past = compute_variability(
            cloud_index=index,
            mask=definition_mask,
            step=nb_slots_per_day,
            negative_variation_only=False,
            abs_value=False,
        )

        var_ndsi_1d_future = compute_variability(
            cloud_index=index,
            mask=definition_mask,
            step=-nb_slots_per_day,
            negative_variation_only=False,
            abs_value=False,
        )
        if nb_days == 2:
            to_return[:nb_slots_per_day] = var_ndsi_1d_future[:nb_slots_per_day]
            to_return[nb_slots_per_day:] = var_ndsi_1d_past[nb_slots_per_day:]
        else:  # nb_days >=3
            var_ndsi_2d_past = compute_variability(
                cloud_index=index,
                mask=definition_mask,
                step=nb_slots_per_day * 2,
                negative_variation_only=False,
                abs_value=False,
            )
            var_ndsi_2d_future = compute_variability(
                cloud_index=index,
                mask=definition_mask,
                step=-2 * nb_slots_per_day,
                negative_variation_only=False,
                abs_value=False,
            )
            # first day
            to_return[:nb_slots_per_day] = np.minimum(
                var_ndsi_1d_future[:nb_slots_per_day],
                var_ndsi_2d_future[:nb_slots_per_day],
            )
            # last day
            to_return[-nb_slots_per_day:] = np.minimum(
                var_ndsi_1d_past[-nb_slots_per_day:],
                var_ndsi_2d_past[-nb_slots_per_day:],
            )

            if nb_days == 3:
                # second day
                to_return[nb_slots_per_day : 2 * nb_slots_per_day] = np.minimum(
                    var_ndsi_1d_past[nb_slots_per_day : 2 * nb_slots_per_day],
                    var_ndsi_1d_future[nb_slots_per_day : 2 * nb_slots_per_day],
                )
            else:  # nb_days >= 4
                # the day previous the last one
                to_return[-2 * nb_slots_per_day : -nb_slots_per_day] = np.minimum(
                    np.minimum(
                        var_ndsi_1d_past[-2 * nb_slots_per_day : -nb_slots_per_day],
                        var_ndsi_2d_past[-2 * nb_slots_per_day : -nb_slots_per_day],
                    ),
                    var_ndsi_1d_future[-2 * nb_slots_per_day : -nb_slots_per_day],
                )
                # second day
                to_return[nb_slots_per_day : 2 * nb_slots_per_day] = np.minimum(
                    np.minimum(
                        var_ndsi_1d_future[nb_slots_per_day : 2 * nb_slots_per_day],
                        var_ndsi_2d_future[nb_slots_per_day : 2 * nb_slots_per_day],
                    ),
                    var_ndsi_1d_past[nb_slots_per_day : 2 * nb_slots_per_day],
                )
                if nb_days >= 5:
                    to_return[
                        2 * nb_slots_per_day : -2 * nb_slots_per_day
                    ] = np.minimum(
                        np.minimum(
                            var_ndsi_1d_past[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                            var_ndsi_2d_past[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                        ),
                        np.minimum(
                            var_ndsi_1d_future[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                            var_ndsi_2d_future[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                        ),
                    )
    to_return[to_return < 0] = 0
    return to_return


def get_flat_sir(
    variable,
    cos_zen,
    mask,
    nb_slots_per_day,
    slices_per_day,
    tolerance,
    persistence_sigma,
    mask_not_proper_weather=None,
):
    if mask_not_proper_weather is not None:
        mask = mask | mask_not_proper_weather
    return get_likelihood_variable_cos_zen(
        variable=variable,
        cos_zen=cos_zen,
        mask=mask,
        nb_slots_per_day=nb_slots_per_day,
        nb_slices_per_day=slices_per_day,
        under_bound=-tolerance,
        upper_bound=tolerance,
        persistence_sigma=persistence_sigma,
    )


def get_tricky_transformed_ndsi(snow_index, summit, gamma=4):
    recentered = np.abs(snow_index - summit)
    # beta = full_like(snow_index, 0.5)
    # alpha = -0.5/(max(1-summit, summit)**2)
    # return beta + alpha * recentered * recentered
    return normalize(np.exp(-gamma * recentered) - np.exp(-gamma * summit))


def get_cloudy_sea(vis, is_land, thresholds):
    to_return = np.zeros_like(vis)
    for slot in range(len(to_return)):
        to_return[slot, :, :][~is_land & (vis[slot, :, :] > thresholds[slot])] = 1
    return to_return
