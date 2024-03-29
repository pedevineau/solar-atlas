from scipy.stats import pearsonr

from angles_geom import get_next_midday
from angles_geom import get_zenith_angle
from get_data import compute_variability
from get_data import mask_channels
from get_data import remove_cos_zen_correlation
from read_metadata import read_epsilon_param
from read_metadata import read_satellite_name
from static_tests import night_test, dynamic_temperature_test, dawn_day_test
from utils import *


def infrared_outputs(
    times,
    latitudes,
    longitudes,
    temperatures,
    content_infrared,
    satellite_step,
    slot_step,
    output_level="abstract",
    gray_scale=False,
):
    """
    Wrapper outputting the low-level predictors computed from infrared measurements (Short-IR, Medium-IR, Long-IR)

    :param times: a datetime matrix giving the times of all sampled slots
    :param latitudes: a float matrix giving the bottom latitudes of all pixels
    :param longitudes: a float matrix giving the bottom longitudes of all pixels
    :param temperatures: map of temperatures estimates at 2-meter above ground estimated with wheather predictive model
    :param content_infrared: matrix with all infrared channels available
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :param output_level
    :param gray_scale: (default: False) if True, output predictors are normalized in the interval [0, 1]
    :return: a matrix with all infrared predictors (shape: slots, latitudes, longitudes, predictors)
    """
    content_infrared, mask_input = mask_channels(content_infrared)
    nb_channels = np.shape(content_infrared)[-1]

    if output_level == "channel":
        if not gray_scale:
            return content_infrared
        else:
            for chan in range(nb_channels):
                content_infrared[:, :, :, chan] = normalize(
                    content_infrared[:, :, :, chan], mask_input, "gray-scale"
                )
            return np.asarray(content_infrared, dtype=np.uint8)

    angles = get_zenith_angle(times, latitudes, longitudes)

    if read_satellite_name() in ["GOES16", "H08"]:
        cli_eps = get_cloud_index(
            cos_zen=np.cos(angles),
            mir=content_infrared[:, :, :, 1],
            lir=content_infrared[:, :, :, 0],
            method="epsilon",
        )
    else:
        cli_eps = np.zeros_like(angles)
    if output_level == "cli":
        return (
            get_cloud_index(
                cos_zen=np.cos(angles),
                mir=content_infrared[:, :, :, nb_channels - 1],
                lir=content_infrared[:, :, :, nb_channels - 2],
                method="mu-normalization",
            ),
            cli_eps,
            mask_input,
        )

    else:
        difference = get_cloud_index(
            cos_zen=np.cos(angles),
            mir=content_infrared[:, :, :, nb_channels - 1],
            lir=content_infrared[:, :, :, nb_channels - 2],
            method="default",
        )
        lir = content_infrared[:, :, :, nb_channels - 2]
        del content_infrared
        return infrared_abstract_predictors(
            angles,
            lir,
            mask_input,
            temperatures,
            cli_eps,
            difference,
            satellite_step,
            slot_step,
            gray_scale,
        )


def infrared_abstract_predictors(
    zen,
    lir,
    mask_input,
    temperatures,
    cli,
    ancillary_cloud_index,
    satellite_step,
    slot_step,
    gray_scale,
):
    """
    Wrapper outputting the high-level predictors computed from infrared measurements (Short-IR, Medium-IR, Long-IR)

    :param zen: map of the solar-zenith angle for every pixel
    :param lir: map of long-infrared measurements for every pixel
    :param mask_input:
    :param temperatures: map of temperatures estimates at 2-meter above ground estimated with wheather predictive model
    :param cli: map of cloud cover index for every pixel
    :param ancillary_cloud_index:
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :param gray_scale: (default: False) if True, output predictors are normalized in the interval [0, 1]
    :return: a matrix with all infrared predictors (shape: slots, latitudes, longitudes, predictors)
    """
    mask_output = mask_input | ~night_test(zen)
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(lir)
    nb_features = 3  # cli, variations, cold
    if not gray_scale:  # if on-point mode
        array_indexes = np.empty(
            shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features)
        )
        array_indexes[:, :, :, 0] = cli
        array_indexes[:, :, :, 1] = ancillary_cloud_index
    else:  # if image
        array_indexes = np.empty(
            shape=(nb_slots, nb_latitudes, nb_longitudes, nb_features), dtype=np.uint8
        )
        array_indexes[:, :, :, 0] = normalize(
            cli, mask=mask_output, normalization="gray-scale"
        )
        array_indexes[:, :, :, 1] = normalize(
            # np.abs(get_cloud_short_variability(cloud_index=ancillary_cloud_index, definition_mask=mask_input)) * \
            get_cloud_index_positive_variability_7d(
                cloud_index=ancillary_cloud_index,
                definition_mask=(mask_input | ~dawn_day_test(zen)),
                # pre_cloud_mask= high_cli_mask | (cold == 1),
                pre_cloud_mask=None,
                satellite_step=satellite_step,
                slot_step=slot_step,
            ),
            normalization="gray_scale",
        )
    cold = dynamic_temperature_test(lir, temperatures, satellite_step, slot_step)
    # cold==1 for cold things: snow, icy clouds, or cold water clouds like altocumulus
    cold[mask_output] = 0
    array_indexes[:, :, :, 2] = cold
    return array_indexes


def get_cloud_index(
    cos_zen, mir, lir, mask=None, pre_cloud_mask=None, method="default"
):
    """
    :param cos_zen: cos of zenith angle matrix (shape: slots, latitudes, longitudes)
    :param mir: medium infra-red band (centered on 3890nm for Himawari 8)
    :param fir: far infra-red band (centered on 12380nm for Himawari 8)
    :param mask: mask for outliers and missing isolated data
    :param pre_cloud_mask:
    :param method: {'default', 'mu-normalization', 'clear-sky', 'without-bias', 'norm-diff', 'epsilon'}
    :return: a cloud index matrix (shape: slots, latitudes, longitudes)
    """

    difference = mir - lir
    if method == "mu-normalization":
        mu_threshold = 0.05
        return difference / np.maximum(cos_zen, mu_threshold)
    elif method == "norm-diff":
        return difference / (mir + lir)
    elif method == "default":
        return difference
    elif method == "epsilon":
        dlen = read_epsilon_param()
        h = 6.626 * 10 ** (-34)
        k = 1.38 * 10 ** (-23)
        c = 3.0 * 10**8
        K = h * c / (k * dlen * 10 ** (-6))
        from scipy import exp

        return 1 - exp(-K * (1.0 / lir - 1.0 / mir))

        # return 1. - K*(1./fir-1./mir)
    else:
        if method == "without-bias":
            diffstd = normalize(difference, mask, normalization="standard")
            # NB: ths use of 'normalize_array' method causes a bias term alpha*std(index)*m(cos-zen)/std(cos_zen)
            # mustd = normalize_array(cos_zen, mask, 'standard')

            mask = mask | pre_cloud_mask
            cli = remove_cos_zen_correlation(diffstd, cos_zen, mask)
            cli = normalize(cli, mask, "standard")
        elif method == "clear-sky":
            diffstd = normalize(difference, mask, normalization="standard")
            cli = diffstd - normalize(cos_zen, mask, normalization="standard")
        else:
            raise Exception("Please choose a valid method to compute cloud index")
        return cli


def get_cloud_short_variability(cloud_index, valid_measurements_mask):
    """
    Wrapper returning cloud variability index computed between cloud index maps at step t and step t+1

    :param cloud_index: array (slots, latitudes, longitudes) with cloud index (cli or unbiased difference)
    :param valid_measurements_mask: boolean array (slots, latitudes, longitudes) indicating which measurements are valid
    :return: array (slots, latitudes, longitudes) with cloud variability index between consecutive cloud index maps
    """
    return compute_variability(
        cloud_index=cloud_index, mask=valid_measurements_mask, step=1
    )


def get_cloud_index_positive_variability_7d(
    cloud_index,
    definition_mask,
    satellite_step,
    pre_cloud_mask=None,
    slot_step=1,
    only_past=False,
):
    """
    This function is supposed to help finding small clouds with low positive clouds (in partiular NOT icy clouds)

    :param only_past: MODE of the function. If True, compare one day with the previous days only.
     If False, compare one day both with the previous and the following days.
    :param cloud_index: array (slots, latitudes, longitudes) with cloud index (cli or unbiased difference)
    :param definition_mask: mask points where cloud index is not defined
    :param pre_cloud_mask: mask points where we don't want to compute 5 days variability (eg: obvious water clouds)
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :return:
    """
    if pre_cloud_mask is not None:
        mask = definition_mask | pre_cloud_mask
    else:
        mask = definition_mask
    slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    nb_days = np.shape(cloud_index)[0] / slots_per_day
    to_return = np.zeros_like(cloud_index)
    if only_past:
        if nb_days >= 2:
            var_cli_1d_past = compute_variability(
                cloud_index=cloud_index, mask=mask, step=slots_per_day
            )
            to_return[slots_per_day:] = var_cli_1d_past[slots_per_day:]

            if nb_days >= 3:
                var_cli_2d_past = compute_variability(
                    cloud_index=cloud_index, mask=mask, step=slots_per_day * 2
                )
                # second day
                to_return[slots_per_day : 2 * slots_per_day] = np.maximum(
                    var_cli_1d_past[slots_per_day : 2 * slots_per_day],
                    var_cli_2d_past[-slots_per_day : 2 * slots_per_day],
                )

                if nb_days == 3:
                    # second day
                    to_return[slots_per_day : 2 * slots_per_day] = np.maximum(
                        var_cli_1d_past[slots_per_day : 2 * slots_per_day],
                        var_cli_1d_future[slots_per_day : 2 * slots_per_day],
                    )
                else:  # nb_days >= 4
                    # the day previous the last one
                    to_return[-2 * slots_per_day : -slots_per_day] = np.maximum(
                        np.maximum(
                            var_cli_1d_past[-2 * slots_per_day : -slots_per_day],
                            var_cli_2d_past[-2 * slots_per_day : -slots_per_day],
                        ),
                        var_cli_1d_future[-2 * slots_per_day : -slots_per_day],
                    )
                    # second day
                    to_return[slots_per_day : 2 * slots_per_day] = np.maximum(
                        np.maximum(
                            var_cli_1d_future[slots_per_day : 2 * slots_per_day],
                            var_cli_2d_future[slots_per_day : 2 * slots_per_day],
                        ),
                        var_cli_1d_past[slots_per_day : 2 * slots_per_day],
                    )
                    if nb_days >= 5:
                        to_return[2 * slots_per_day : -2 * slots_per_day] = np.maximum(
                            np.maximum(
                                var_cli_1d_past[2 * slots_per_day : -2 * slots_per_day],
                                var_cli_2d_past[2 * slots_per_day : -2 * slots_per_day],
                            ),
                            np.maximum(
                                var_cli_1d_future[
                                    2 * slots_per_day : -2 * slots_per_day
                                ],
                                var_cli_2d_future[
                                    2 * slots_per_day : -2 * slots_per_day
                                ],
                            ),
                        )
                    else:  # nb_days >= 6
                        var_cli_3d_past = compute_variability(
                            cloud_index=cloud_index, mask=mask, step=slots_per_day * 3
                        )

                        var_cli_3d_future = compute_variability(
                            cloud_index=cloud_index, mask=mask, step=-slots_per_day * 3
                        )
                        # two days previous the last one
                        to_return[-3 * slots_per_day : -2 * slots_per_day] = np.maximum(
                            np.maximum(
                                np.maximum(
                                    var_cli_1d_past[
                                        -3 * slots_per_day : -2 * slots_per_day
                                    ],
                                    var_cli_2d_past[
                                        -3 * slots_per_day : -2 * slots_per_day
                                    ],
                                ),
                                var_cli_3d_past[
                                    -3 * slots_per_day : -2 * slots_per_day
                                ],
                            ),
                            np.maximum(
                                var_cli_1d_future[
                                    -3 * slots_per_day : -2 * slots_per_day
                                ],
                                var_cli_2d_future[
                                    -3 * slots_per_day : -2 * slots_per_day
                                ],
                            ),
                        )
                        # third day
                        to_return[2 * slots_per_day : 3 * slots_per_day] = np.maximum(
                            np.maximum(
                                np.maximum(
                                    var_cli_1d_future[
                                        2 * slots_per_day : 3 * slots_per_day
                                    ],
                                    var_cli_2d_future[
                                        2 * slots_per_day : 3 * slots_per_day
                                    ],
                                ),
                                var_cli_3d_future[
                                    2 * slots_per_day : 3 * slots_per_day
                                ],
                            ),
                            np.maximum(
                                var_cli_1d_past[2 * slots_per_day : 3 * slots_per_day],
                                var_cli_2d_past[2 * slots_per_day : 3 * slots_per_day],
                            ),
                        )
                        if nb_days >= 7:
                            to_return[
                                3 * slots_per_day : -3 * slots_per_day
                            ] = np.maximum(
                                np.maximum(
                                    np.maximum(
                                        var_cli_1d_past[
                                            3 * slots_per_day : -3 * slots_per_day
                                        ],
                                        var_cli_2d_past[
                                            3 * slots_per_day : -3 * slots_per_day
                                        ],
                                    ),
                                    var_cli_3d_past[
                                        3 * slots_per_day : -3 * slots_per_day
                                    ],
                                ),
                                np.maximum(
                                    np.maximum(
                                        var_cli_1d_future[
                                            3 * slots_per_day : -3 * slots_per_day
                                        ],
                                        var_cli_2d_future[
                                            3 * slots_per_day : -3 * slots_per_day
                                        ],
                                    ),
                                    var_cli_3d_future[
                                        3 * slots_per_day : -3 * slots_per_day
                                    ],
                                ),
                            )
    else:
        if nb_days >= 2:
            var_cli_1d_past = compute_variability(
                cloud_index=cloud_index, mask=mask, step=slots_per_day
            )

            var_cli_1d_future = compute_variability(
                cloud_index=cloud_index, mask=mask, step=-slots_per_day
            )
            if nb_days == 2:
                to_return[:slots_per_day] = var_cli_1d_future[:slots_per_day]
                to_return[slots_per_day:] = var_cli_1d_past[slots_per_day:]
            else:  # nb_days >=3
                var_cli_2d_past = compute_variability(
                    cloud_index=cloud_index, mask=mask, step=slots_per_day * 2
                )
                var_cli_2d_future = compute_variability(
                    cloud_index=cloud_index, mask=mask, step=-2 * slots_per_day
                )

                # first day
                to_return[:slots_per_day] = np.maximum(
                    var_cli_1d_future[:slots_per_day], var_cli_2d_future[:slots_per_day]
                )
                # last day
                to_return[-slots_per_day:] = np.maximum(
                    var_cli_1d_past[-slots_per_day:], var_cli_2d_past[-slots_per_day:]
                )

                if nb_days == 3:
                    # second day
                    to_return[slots_per_day : 2 * slots_per_day] = np.maximum(
                        var_cli_1d_past[slots_per_day : 2 * slots_per_day],
                        var_cli_1d_future[slots_per_day : 2 * slots_per_day],
                    )
                else:  # nb_days >= 4
                    # the day previous the last one
                    to_return[-2 * slots_per_day : -slots_per_day] = np.maximum(
                        np.maximum(
                            var_cli_1d_past[-2 * slots_per_day : -slots_per_day],
                            var_cli_2d_past[-2 * slots_per_day : -slots_per_day],
                        ),
                        var_cli_1d_future[-2 * slots_per_day : -slots_per_day],
                    )
                    # second day
                    to_return[slots_per_day : 2 * slots_per_day] = np.maximum(
                        np.maximum(
                            var_cli_1d_future[slots_per_day : 2 * slots_per_day],
                            var_cli_2d_future[slots_per_day : 2 * slots_per_day],
                        ),
                        var_cli_1d_past[slots_per_day : 2 * slots_per_day],
                    )
                    if nb_days >= 5:
                        to_return[2 * slots_per_day : -2 * slots_per_day] = np.maximum(
                            np.maximum(
                                var_cli_1d_past[2 * slots_per_day : -2 * slots_per_day],
                                var_cli_2d_past[2 * slots_per_day : -2 * slots_per_day],
                            ),
                            np.maximum(
                                var_cli_1d_future[
                                    2 * slots_per_day : -2 * slots_per_day
                                ],
                                var_cli_2d_future[
                                    2 * slots_per_day : -2 * slots_per_day
                                ],
                            ),
                        )
                    else:  # nb_days >= 6
                        var_cli_3d_past = compute_variability(
                            cloud_index=cloud_index, mask=mask, step=slots_per_day * 3
                        )

                        var_cli_3d_future = compute_variability(
                            cloud_index=cloud_index, mask=mask, step=-slots_per_day * 3
                        )
                        # two days previous the last one
                        to_return[-3 * slots_per_day : -2 * slots_per_day] = np.maximum(
                            np.maximum(
                                np.maximum(
                                    var_cli_1d_past[
                                        -3 * slots_per_day : -2 * slots_per_day
                                    ],
                                    var_cli_2d_past[
                                        -3 * slots_per_day : -2 * slots_per_day
                                    ],
                                ),
                                var_cli_3d_past[
                                    -3 * slots_per_day : -2 * slots_per_day
                                ],
                            ),
                            np.maximum(
                                var_cli_1d_future[
                                    -3 * slots_per_day : -2 * slots_per_day
                                ],
                                var_cli_2d_future[
                                    -3 * slots_per_day : -2 * slots_per_day
                                ],
                            ),
                        )
                        # third day
                        to_return[2 * slots_per_day : 3 * slots_per_day] = np.maximum(
                            np.maximum(
                                np.maximum(
                                    var_cli_1d_future[
                                        2 * slots_per_day : 3 * slots_per_day
                                    ],
                                    var_cli_2d_future[
                                        2 * slots_per_day : 3 * slots_per_day
                                    ],
                                ),
                                var_cli_3d_future[
                                    2 * slots_per_day : 3 * slots_per_day
                                ],
                            ),
                            np.maximum(
                                var_cli_1d_past[2 * slots_per_day : 3 * slots_per_day],
                                var_cli_2d_past[2 * slots_per_day : 3 * slots_per_day],
                            ),
                        )
                        if nb_days >= 7:
                            to_return[
                                3 * slots_per_day : -3 * slots_per_day
                            ] = np.maximum(
                                np.maximum(
                                    np.maximum(
                                        var_cli_1d_past[
                                            3 * slots_per_day : -3 * slots_per_day
                                        ],
                                        var_cli_2d_past[
                                            3 * slots_per_day : -3 * slots_per_day
                                        ],
                                    ),
                                    var_cli_3d_past[
                                        3 * slots_per_day : -3 * slots_per_day
                                    ],
                                ),
                                np.maximum(
                                    np.maximum(
                                        var_cli_1d_future[
                                            3 * slots_per_day : -3 * slots_per_day
                                        ],
                                        var_cli_2d_future[
                                            3 * slots_per_day : -3 * slots_per_day
                                        ],
                                    ),
                                    var_cli_3d_future[
                                        3 * slots_per_day : -3 * slots_per_day
                                    ],
                                ),
                            )
    # we are interested only in positive cli variations
    to_return[to_return < 0] = 0
    return to_return


def get_cloud_index_total_variability_7d(
    cloud_index, definition_mask, pre_cloud_mask, satellite_step, slot_step
):
    """

    :param cloud_index: array (slots, latitudes, longitudes) with cloud index (cli or unbiased difference)
    :param definition_mask: mask points where cloud index is not defined
    :param pre_cloud_mask: mask points where we don't want to compute 5 days variability (eg: obvious water clouds)
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :return:
    """

    if pre_cloud_mask is not None:
        mask = definition_mask | pre_cloud_mask
    else:
        mask = definition_mask
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    nb_days = np.shape(cloud_index)[0] / nb_slots_per_day
    to_return = np.full_like(cloud_index, -10)
    if nb_days >= 2:
        var_cli_1d_past = compute_variability(
            cloud_index=cloud_index, mask=mask, abs_value=True, step=nb_slots_per_day
        )

        var_cli_1d_future = compute_variability(
            cloud_index=cloud_index, mask=mask, abs_value=True, step=-nb_slots_per_day
        )
        if nb_days == 2:
            to_return[:nb_slots_per_day] = var_cli_1d_future[:nb_slots_per_day]
            to_return[nb_slots_per_day:] = var_cli_1d_past[nb_slots_per_day:]
        else:  # nb_days >=3
            var_cli_2d_past = compute_variability(
                cloud_index=cloud_index,
                mask=mask,
                abs_value=True,
                step=nb_slots_per_day * 2,
            )
            var_cli_2d_future = compute_variability(
                cloud_index=cloud_index,
                mask=mask,
                abs_value=True,
                step=-2 * nb_slots_per_day,
            )

            # first day
            to_return[:nb_slots_per_day] = np.minimum(
                var_cli_1d_future[:nb_slots_per_day],
                var_cli_2d_future[:nb_slots_per_day],
            )
            # last day
            to_return[-nb_slots_per_day:] = np.minimum(
                var_cli_1d_past[-nb_slots_per_day:], var_cli_2d_past[-nb_slots_per_day:]
            )

            if nb_days == 3:
                # second day
                to_return[nb_slots_per_day : 2 * nb_slots_per_day] = np.minimum(
                    var_cli_1d_past[nb_slots_per_day : 2 * nb_slots_per_day],
                    var_cli_1d_future[nb_slots_per_day : 2 * nb_slots_per_day],
                )
            else:  # nb_days >= 4
                # the day previous the last one
                to_return[-2 * nb_slots_per_day : -nb_slots_per_day] = np.minimum(
                    np.minimum(
                        var_cli_1d_past[-2 * nb_slots_per_day : -nb_slots_per_day],
                        var_cli_2d_past[-2 * nb_slots_per_day : -nb_slots_per_day],
                    ),
                    var_cli_1d_future[-2 * nb_slots_per_day : -nb_slots_per_day],
                )
                # second day
                to_return[nb_slots_per_day : 2 * nb_slots_per_day] = np.minimum(
                    np.minimum(
                        var_cli_1d_future[nb_slots_per_day : 2 * nb_slots_per_day],
                        var_cli_2d_future[nb_slots_per_day : 2 * nb_slots_per_day],
                    ),
                    var_cli_1d_past[nb_slots_per_day : 2 * nb_slots_per_day],
                )
                if nb_days == 5:
                    to_return[
                        2 * nb_slots_per_day : -2 * nb_slots_per_day
                    ] = np.minimum(
                        np.minimum(
                            var_cli_1d_past[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                            var_cli_2d_past[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                        ),
                        np.minimum(
                            var_cli_1d_future[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                            var_cli_2d_future[
                                2 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                        ),
                    )
                else:  # nb_days >= 6
                    var_cli_3d_past = compute_variability(
                        cloud_index=cloud_index,
                        mask=mask,
                        abs_value=True,
                        step=nb_slots_per_day * 3,
                    )

                    var_cli_3d_future = compute_variability(
                        cloud_index=cloud_index,
                        mask=mask,
                        abs_value=True,
                        step=-nb_slots_per_day * 3,
                    )
                    # two days previous the last one
                    to_return[
                        -3 * nb_slots_per_day : -2 * nb_slots_per_day
                    ] = np.minimum(
                        np.minimum(
                            np.minimum(
                                var_cli_1d_past[
                                    -3 * nb_slots_per_day : -2 * nb_slots_per_day
                                ],
                                var_cli_2d_past[
                                    -3 * nb_slots_per_day : -2 * nb_slots_per_day
                                ],
                            ),
                            var_cli_3d_past[
                                -3 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                        ),
                        np.minimum(
                            var_cli_1d_future[
                                -3 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                            var_cli_2d_future[
                                -3 * nb_slots_per_day : -2 * nb_slots_per_day
                            ],
                        ),
                    )
                    # third day
                    to_return[2 * nb_slots_per_day : 3 * nb_slots_per_day] = np.minimum(
                        np.minimum(
                            np.minimum(
                                var_cli_1d_future[
                                    2 * nb_slots_per_day : 3 * nb_slots_per_day
                                ],
                                var_cli_2d_future[
                                    2 * nb_slots_per_day : 3 * nb_slots_per_day
                                ],
                            ),
                            var_cli_3d_future[
                                2 * nb_slots_per_day : 3 * nb_slots_per_day
                            ],
                        ),
                        np.minimum(
                            var_cli_1d_past[
                                2 * nb_slots_per_day : 3 * nb_slots_per_day
                            ],
                            var_cli_2d_past[
                                2 * nb_slots_per_day : 3 * nb_slots_per_day
                            ],
                        ),
                    )
                    if nb_days >= 7:
                        to_return[
                            3 * nb_slots_per_day : -3 * nb_slots_per_day
                        ] = np.minimum(
                            np.minimum(
                                np.minimum(
                                    var_cli_1d_past[
                                        3 * nb_slots_per_day : -3 * nb_slots_per_day
                                    ],
                                    var_cli_2d_past[
                                        3 * nb_slots_per_day : -3 * nb_slots_per_day
                                    ],
                                ),
                                var_cli_3d_past[
                                    3 * nb_slots_per_day : -3 * nb_slots_per_day
                                ],
                            ),
                            np.minimum(
                                np.minimum(
                                    var_cli_1d_future[
                                        3 * nb_slots_per_day : -3 * nb_slots_per_day
                                    ],
                                    var_cli_2d_future[
                                        3 * nb_slots_per_day : -3 * nb_slots_per_day
                                    ],
                                ),
                                var_cli_3d_future[
                                    3 * nb_slots_per_day : -3 * nb_slots_per_day
                                ],
                            ),
                        )

    return to_return


def get_warm(mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median):
    """
    Not used anymore. Mask warm pixels
    :param mir: medium infra-red band (centered on 3890nm for Himawari 8)
    :param cos_zen: cos of zenith angle matrix (shape: slots, latitudes, longitudes)
    :param satellite_step: the satellite characteristic time step between two slots (10 minutes for Himawari 8)
    :param slot_step: the chosen sampling of slots. if slot_step = n, the sampled slots are s[0], s[n], s[2*n]...
    :param cloudy_mask: the mask of supposed clouds, in order not to take them into account for median of temperature
    :param threshold_median: an hyper-parameter giving the minimum median infra-red (3890nm) temperature to be considered as hot pixel
    :return: a 0-1 integer matrix (shape: slots, latitudes, longitudes)
    """
    to_return = np.zeros_like(mir, dtype=np.uint8)
    warm_mask = get_warm_ground_mask(
        mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median
    )
    to_return[warm_mask] = 1
    return to_return


def get_cold(fir, mask, threshold):
    """
    Not used anymore. Has been replaced by static tests
    recognize some high altitudes clouds
    there is no risk of temperature inversion at these latitudes (it occurs mostly over poles)
    :param fir: medium infra-red band (centered on 12380nm for Himawari 8)
    :param threshold
    :return: a 0-1 integer matrix (shape: slots, latitudes, longitudes)
    """
    # TODO: add some cos-zen correlation (correlation => it was not clouds)
    to_return = np.zeros_like(fir, dtype=np.uint8)
    to_return[fir < threshold] = 1
    to_return[mask] = -10
    return to_return


def get_warm_ground_mask(
    mir, cos_zen, satellite_step, slot_step, cloudy_mask, threshold_median
):
    # compute median temperature around noon (clouds are masked in order not to bias the median)
    # and compare it with a threshold

    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(mir)
    warm_ground_mask = np.zeros((nb_slots, nb_latitudes, nb_longitudes), dtype=bool)
    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            warm_ground_mask[:, lat, lon] = get_warm_array_on_point(
                mir[:, lat, lon],
                cos_zen[:, lat, lon],
                satellite_step,
                slot_step,
                cloudy_mask[:, lat, lon],
                threshold_median,
            )
    return warm_ground_mask


def get_warm_array_on_point(
    mir_point, mu_point, satellite_step, slot_step, cloud_mask_point, threshold_median
):
    nb_slots_per_day = get_nb_slots_per_day(satellite_step, slot_step)
    nb_slots = len(mir_point)
    width_window_in_minutes = 240
    width_window_in_slots = width_window_in_minutes / (slot_step * satellite_step)

    noon = get_next_midday(mu_point, nb_slots_per_day)
    is_warm_array = np.zeros_like(mir_point)
    beginning_slice = noon - nb_slots_per_day
    ending_slice = beginning_slice + width_window_in_slots + 1
    while ending_slice < nb_slots:
        slice_cloud_mask = cloud_mask_point[max(0, beginning_slice) : ending_slice]
        slice_mir = mir_point[max(0, beginning_slice) : ending_slice]
        median_excluding_clouds = np.median(slice_mir[~slice_cloud_mask])
        previous_midnight = max(noon - nb_slots_per_day / 2, 0)
        next_midnight = min(noon + nb_slots_per_day / 2, nb_slots)
        if median_excluding_clouds > threshold_median:
            is_warm_array[previous_midnight:next_midnight] = np.ones(
                next_midnight - previous_midnight, dtype=bool
            )
        beginning_slice += nb_slots_per_day
        ending_slice += nb_slots_per_day

    return is_warm_array


def get_lag_high_peak(difference, cos_zen, satellite_step, slot_step):
    """
    not used anymore
    function to get high temperature peak, in order to do a proper mu-normalization
    this function seems useless as it appears that clear-sky difference mir-fir has always a peak at noon
    :param difference:
    :param cos_zen:
    :param satellite_step:
    :param slot_step:
    :return:
    """

    # lag is expected between 2:30 and 4:30
    start_lag_minutes = 10
    stop_lag_minutes = 240
    testing_lags = np.arange(
        start_lag_minutes, stop_lag_minutes, step=slot_step * satellite_step, dtype=int
    )

    nb_slots, nb_lats, nb_lons = np.shape(cos_zen)[0:3]
    nb_days = nb_slots / get_nb_slots_per_day(satellite_step, slot_step)
    n = 400
    computed_shifts = np.zeros(n)
    computed_shifts_dtw = np.zeros(n)
    indexes_lag = []
    # montecarlo
    for i in range(n):
        corrs = []

        dtws = []
        lat = np.random.randint(0, nb_lats)
        lon = np.random.randint(0, nb_lons)
        day = np.random.randint(0, nb_days)
        diff_1d = difference[1 + 144 * day : 60 + 144 * day, lat, lon]
        mu_1d = cos_zen[1 + 144 * day : 60 + 144 * day, lat, lon]

        for lag in testing_lags:
            # negative shift of diff = diff is shifted to the past (to be coherent with mu because diff is late)
            shift = -lag / (satellite_step * slot_step)

            r, p = pearsonr(np.roll(diff_1d, shift=shift)[:shift], mu_1d[:shift])
            # dtw = LB_Keogh(np.roll(diff_1d, shift=shift)[:shift], mu_1d[:shift], r=10)
            # dtw = get_dtw(np.roll(diff_1d, shift=shift)[:shift], mu_1d[:shift])
            corrs.append(r)
            # dtws.append(dtw)
        index_lag = np.argmax(corrs)
        if index_lag >= 0 and np.max(corrs) > 0.9:
            if index_lag == 0:
                index_lag = 1
            # visualize_input(np.roll(diff_1d, shift=-index_lag)[:-index_lag], title=str(index_lag), display_now=False)
            # visualize_input(mu_1d[:-index_lag])
            indexes_lag.append(index_lag)
            computed_shifts[index_lag] += 1
            # computed_shifts_dtw[np.argmin(dtws)]
            minutes_lag = testing_lags[index_lag]
    return start_lag_minutes / (slot_step * satellite_step) + np.argmax(
        computed_shifts[1:]
    )


if __name__ == "__main__":
    lis = np.arange(0, 10)
    T = 1
    r1 = 1 * (np.random.random_sample(len(lis)) - 0.5)
    r1[10:30] = 0
    diff = np.sin(2 * np.pi * (lis + 5) / T) + r1
    mu = np.sin(2 * np.pi * lis / T)
    corrs = []
    for k in range(20):
        r, p = pearsonr(np.roll(diff, shift=k), mu)
        corrs.append(r)
