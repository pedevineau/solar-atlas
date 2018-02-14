from thresholds import *


def threshold_test_positive(observed, forecast, thresh):
    return (observed - forecast) > thresh


def gross_cloud_test(lir_observed, lir_forecast):
    thresh = compute_gross_cloud_threshold()
    return threshold_test_positive(lir_forecast, lir_observed, thresh)


def lir_fir_test(lir_observed, fir_observed, dlw=12.3-10.3):
    # the test can be apply over any type of surface
    # mostly for semi-transparent clouds, like cirrus. Quite blind to opaque thick clouds
    h = 6.626 * 10 ** (-34)
    k = 1.38 * 10 ** (-23)
    c = 3. * 10 ** 8
    K = h * c / (k * dlw * 10 ** (-6))
    thresh_epsilon = compute_epsilon_threshold()
    thresh_epsilon_maximal_lir = compute_epsilon_maximal_lir_threshold()
    from scipy import exp
    return (exp(-K * (1. / fir_observed - 1. / lir_observed)) < thresh_epsilon) & (lir_observed < thresh_epsilon_maximal_lir)


def perso_cli_test(mir_observed, fir_observed):
    return 1


def low_cloud_test_sun_glint(cos_zen, vis, mir, lir, mask):
    from infrared_predictors import get_cloud_index
    cli410 = get_cloud_index(cos_zen, mir, lir, mask, 'mu-normalization')
    thresh_mir_sunglint = compute_mir_sun_glint_threshold()
    thresh_vis_sunglint = compute_vis_sun_glint_threshold()
    return day_test(cos_zen) & (mir < thresh_mir_sunglint) & (vis > thresh_vis_sunglint) & (cli410 > 0) & \
           (vis > (1/0.15)*cli410)


def local_spatial_texture(is_land, mir, lir, mask):
    # Le Gleau 2006 (simplified)
    # eliminate pixels near sea
    is_land = decrease_connectivity_2(decrease_connectivity_2(is_land))
    from filter import local_std
    thresh_coherence_lir_land, thresh_coherence_mir_lir_land = compute_lir_texture_land(), compute_mir_lir_texture_land()
    sd_lir = local_std(lir , mask, scope=3)
    sd_diff410 = local_std(mir - lir, mask, scope=3)
    return is_land & (sd_lir > thresh_coherence_lir_land) & (sd_diff410 > thresh_coherence_mir_lir_land)


def gross_snow_test(lir_observed, lir_forecast):
    '''
    Hocking (2011) - Duerr (2006)
    snow (or ice) is supposed to be colder than snow-free pixels under clear-sky conditions. However, this cold snow
    on the ground is still supposed to appear warmer than a potential cold icy cloud
    :param lir_observed:
    :param lir_forecast:
    :return:
    '''
    thresh_gross = compute_gross_cloud_threshold()  # coldness which would be due to clouds if there were clouds
    thresh_snow = compute_gross_snow_threshold()  # coldness only due to the presence of snow on the ground
    return threshold_test_positive(lir_observed, lir_forecast, - thresh_gross - thresh_snow)


def static_temperature_test(lir_observed):
    '''
    snow (or ice) is supposed to be colder than snow-free pixels under clear-sky conditions
    TODO: the threshold should depends on local factors
    :param lir_observed: channel fir
    :return:
    '''
    thresh_temperature = compute_thresh_temperature()
    return lir_observed < thresh_temperature


def dynamic_temperature_test(lir_observed, temperature_mask, satellite_step, slot_step):
    from numpy import shape, empty_like
    s = shape(lir_observed)
    to_return = empty_like(lir_observed, dtype=bool)
    from thresholds import expected_brightness_temperature_only_emissivity
    from read_metadata import read_satellite_name
    lw_nm = {
        'GOES16': 10.3*10**(-6),
        'H08': 12.4*10**(-6),
    }[read_satellite_name()]
    for slot in range(s[0]):
        try:
            nearest_temp_meas = int(0.5+satellite_step*slot_step*slot/60)
            to_return[slot] = (expected_brightness_temperature_only_emissivity(
                temperature_mask[nearest_temp_meas]+273.15, lw_nm=lw_nm, eps=0.85) - lir_observed[slot]) > 5+3
        except IndexError:
            nearest_temp_meas = int(satellite_step*slot_step*slot/60)
            to_return[slot] = (expected_brightness_temperature_only_emissivity(
                temperature_mask[nearest_temp_meas]+273.5, lw_nm=lw_nm, eps=0.85) - lir_observed[slot]) > 5+3
    return to_return


def ndsi_test(ndsi, cos_scat=None):
    from numpy import square
    static_thresh = compute_snow_ndsi_threshold()
    if cos_scat is None:
        return ndsi > static_thresh
    else:
        return ndsi > static_thresh + 0.15 * square(cos_scat - 1)


def broad_cirrus_snow_test(cli_epsilon):
    # to elimimate cirrus
    thresh = compute_broad_cirrus_threshold()
    return cli_epsilon < thresh


def cli_snow_test(cli):
    thresh = compute_cli_snow_threshold()
    return cli < thresh


def cli_water_cloud_test(cli):
    thresh = compute_cli_cloud_threshold()
    return cli > thresh


def cli_stability(cloud_var):
    # stability test (PED 2018)
    thresh = compute_cloud_stability_threshold()
    return cloud_var > thresh


def cli_thin_water_cloud_test(cloud_epsilon):
    # epsilon test (PED 2018)
    thresh = compute_cloud_epsilon_threshold()
    return cloud_epsilon > thresh


def dawn_day_test(angles):
    '''
    to avoid twilight and night
    :param angles:
    :return: boolean array. True when test is completed
    '''
    from numpy import pi
    thresh_inf_radians = 0./180. * pi
    thresh_sup_radians = 85./180. * pi
    return (angles > thresh_inf_radians) & (angles < thresh_sup_radians)


def day_test(angles):
    '''
    to avoid negative or low zenith angles
    :param angles:
    :return: boolean array. True when test is completed
    '''
    from numpy import pi
    thresh_inf_radians = 15./180. * pi
    thresh_sup_radians = 85./180. * pi
    return (angles > thresh_inf_radians) & (angles < thresh_sup_radians)


def night_test(angles):
    '''
    to avoid night
    :param angles:
    :return: boolean array. True when test is completed
    '''
    from numpy import pi
    thresh_inf_radians = 0./180. * pi
    thresh_sup_radians = 90./180. * pi
    return (angles > thresh_inf_radians) & (angles < thresh_sup_radians)


def twilight_test(angles):
    '''
    to select twilight (80 <= theta < 90)
    :param angles:
    :return: boolean array. True when test is completed
    '''
    from numpy import pi
    thresh_inf_radians = 80./180. * pi
    thresh_sup_radians = 90./180. * pi
    return (angles > thresh_inf_radians) & (angles < thresh_sup_radians)


def glint_angle_temporal_test(is_land, specular_angles, glint_angles):
    from numpy import pi
    thresh1_glint_radians = 25./180. * pi
    thresh2_glint_radians = 40./180. * pi
    thresh_specular_radians = 50./180. * pi
    return (glint_angles > thresh1_glint_radians) | ((glint_angles > thresh2_glint_radians) & ~is_land &
                                                     (specular_angles > thresh_specular_radians))


def satellite_angle_temporal_test(specular_angles, satellite_angles):
    from numpy import pi
    thresh1_specular_radians = 25./180. * pi
    thresh2_specular_radians = 50./180. * pi
    thresh_satellite_radians = 70./180. * pi
    return (satellite_angles < thresh_satellite_radians) & ((specular_angles < thresh1_specular_radians) |
                                                            (specular_angles > thresh2_specular_radians))


def solar_angle_temporal_test(angles):
    '''
    to select twilight (80 <= theta < 90)
    :param angles:
    :return: boolean array. True when test is completed
    '''
    from numpy import pi
    thresh_inf_radians = 75./180. * pi
    thresh_sup_radians = 89./180. * pi
    return (angles > thresh_inf_radians) & (angles < thresh_sup_radians)


def specular_satellite_test(specular_angles):
    '''
    to avoid severe glint (low specular angles) and edge of the specular disk (high specular angles)
    :param specular_angles:
    :return: boolean array. True when test is completed
    '''
    from numpy import pi
    angles_in_degree = 180./pi * specular_angles
    return (angles_in_degree > 15) & (angles_in_degree < 70)


def visible_snow_test(vis):
    '''
    to avoid shadows being mistaken for snow and ice
    :param vis: available visible channel (6*10^2 or 8*10^2 nm)
    :return: boolean array. True when test is completed
    '''
    thresh = compute_vis_snow_threshold()
    return vis > thresh


def expand_connectivity_2(to_expand):
    from scipy.ndimage import binary_dilation, generate_binary_structure
    struct = generate_binary_structure(2, 2)
    return binary_dilation(to_expand, struct)


def decrease_connectivity_2(to_decrease):
    from scipy.ndimage import binary_dilation, generate_binary_structure
    struct = generate_binary_structure(2, 2)
    return binary_dilation(to_decrease, struct)


def get_borders_connectivity_2(to_expand):
    return ~to_expand & expand_connectivity_2(to_expand)


def angular_factor(angles):
    from numpy import power, cos
    po = power(cos(angles), 0.3)
    po[po < 0.04] = 0.04
    return 1./po


def land_visible_test(is_land, vis, clear_sky_vis):
    thresh = compute_land_visible_threshold()
    return is_land & threshold_test_positive(vis, clear_sky_vis, thresh)


def sea_cloud_test(angles, is_land, vis_observed):
    thresholds = compute_vis_sea_cloud_all_thresholds(angles)
    # expand is_land because of high visibility of coast pixels
    is_land_expanded = expand_connectivity_2(is_land)
    return ~is_land_expanded & (vis_observed > thresholds)


def thin_cirrus_test(is_land, lir_observed, fir_observed, lir_forecast, fir_forecast):
    thresh = compute_thin_cirrus_threshold()
    supposed_cirrus = threshold_test_positive(lir_observed - fir_observed, lir_forecast - fir_forecast, thresh)
    # additional test to avoid warm lands
    return supposed_cirrus & ((lir_observed < 305) | ~is_land)


def stability_test(channel_observed, past_channel_observed, thresh):
    from numpy import abs
    return abs(channel_observed - past_channel_observed) < thresh


def thermal_stability_test(is_land, lir_observed, fir_observed, past_lir_observed, past_fir_observed):
    thresh_lir_fir = compute_evolution_lir_fir_threshold()
    land_thresh_lir = compute_land_evolution_lir()
    sea_thresh_lir = compute_sea_evolution_lir()
    return (is_land | stability_test(lir_observed, past_lir_observed, land_thresh_lir)) | \
           (~is_land | stability_test(lir_observed, past_lir_observed, sea_thresh_lir)) | \
           stability_test(lir_observed - fir_observed, past_lir_observed - past_fir_observed, thresh_lir_fir)


def flagged_cloud_and_thermally_stable(is_land, dawn_day_clouds, lir_observed, fir_observed,
                                       past_lir_observed, past_fir_observed):
    from numpy import roll
    from read_metadata import read_satellite_step
    number_slots_45_min = 45./read_satellite_step()
    number_slots_1_hour = 60./read_satellite_step()
    return roll(dawn_day_clouds, number_slots_45_min) & roll(dawn_day_clouds, number_slots_1_hour) & \
           thermal_stability_test(is_land, lir_observed, fir_observed, past_lir_observed, past_fir_observed)


def twilight_temporal_low_cloud_test(is_land, dawn_day_clouds, cirrus_clouds, angles, specular_angles, satellite_angles,
                                     glint_angles, vis_observed,  lir_observed, fir_observed,
                                     past_lir_observed, past_fir_observed):
    '''
    Derrien & Le Gleau (2007) as quoted by Hocking, Francis & Saunders (2011)
    :param is_land:
    :param dawn_day_clouds:
    :param cirrus_clouds:
    :param angles:
    :param specular_angles:
    :param satellite_angles:
    :param glint_angles:
    :param vis_observed:
    :param lir_observed:
    :param fir_observed:
    :param past_lir_observed:
    :param past_fir_observed:
    :return:
    '''
    # exclude pixels identified as:
    # - current snow (to avoid false positive)
    # - former cirrus (not low cloud)
    # - former one-pixel clouds (identified only by spatial coherence tests)
    from numpy import any, shape, zeros_like
    twilight = twilight_test(angles)
    found_clouds = zeros_like(angles, dtype=bool)
    for slot in range(shape(angles)[0]):
        if any(twilight[slot]):
            seeds = flagged_cloud_and_thermally_stable(is_land, dawn_day_clouds, lir_observed,
                                                       fir_observed, past_lir_observed,
                                                       past_fir_observed)[slot]
            seeds = remove_cirrus(seeds, cirrus_clouds[slot])
            found_clouds[slot] = grow_seeds(is_land, seeds, angles[slot], specular_angles[slot],
                                            satellite_angles[slot], glint_angles[slot], vis_observed[slot],
                                            lir_observed[slot])
    return found_clouds


def remove_cirrus(seeds, cirrus_clouds):
    return seeds & ~cirrus_clouds


def get_local_avg(seeds, channel):
    # this mean is computed locally, on every group of seeds
    from scipy.ndimage import label, mean, generate_binary_structure, maximum_filter
    from numpy import unique, zeros_like
    struct = generate_binary_structure(2, 2)
    labels = label(seeds, structure=struct)[0]
    index = unique(labels)
    means = mean(channel, labels=labels, index=index)
    channel_avg = zeros_like(channel)
    for k in range(1, means.size):
        channel_avg[labels == k] = means[k]
    channel_avg = maximum_filter(channel_avg, size=3)
    return channel_avg


def seeds_vis_test(vis, vis_avg):
    from numpy import minimum
    return vis > 1.05*vis_avg


def seeds_lir_test(lir, lir_avg):
    return ((lir_avg - 5.0) < lir) & (lir < (lir_avg + 0.5))


def keep_only_3_3_square(seeds):
    from scipy.ndimage import minimum_filter
    return minimum_filter(seeds, size=3)


def grow_seeds(seeds, is_land, angles, specular_angles, satellite_angles, glint_angles, vis, lir):
    from numpy import array
    try_growing = True
    seeds = keep_only_3_3_square(seeds)
    while try_growing:
        borders = get_borders_connectivity_2(seeds)
        vis_avg = get_local_avg(seeds, vis)
        lir_avg = get_local_avg(seeds, lir)
        new_found_seeds = borders & seeds_lir_test(lir, lir_avg) & seeds_vis_test(vis, vis_avg) & \
                          seeds_angular_tests(is_land, angles, specular_angles, satellite_angles, glint_angles)
        if array(new_found_seeds).sum() == 0:
            try_growing = False
        seeds = seeds | new_found_seeds
    return seeds


def seeds_angular_tests(is_land, angles, specular_angles, satellite_angles, glint_angles):
    return specular_satellite_test(specular_angles) & solar_angle_temporal_test(angles) & \
           glint_angle_temporal_test(is_land, specular_angles, glint_angles) & \
           satellite_angle_temporal_test(specular_angles, satellite_angles)


def exhaustive_dawn_day_cloud_test(angles, is_land, cli_mu_observed, cloud_var_observed, cli_epsilon_observed,
                                   vis_observed, lir_observed, fir_observed, lir_forecast, fir_forecast):
    return dawn_day_test(angles) & ((cli_water_cloud_test(cli_mu_observed) & cli_stability(cloud_var_observed)) |
                                    cli_thin_water_cloud_test(cli_epsilon_observed) |
                                    sea_cloud_test(angles, is_land, vis_observed) |
                                    gross_cloud_test(lir_observed, lir_forecast) |
                                    thin_cirrus_test(is_land, lir_observed, fir_observed, lir_forecast, fir_forecast))


def partial_dawn_day_cloud_test(angles, is_land, cli_observed, cloud_var_observed, vis_observed, lir_observed,
                                lir_forecast):
    return dawn_day_test(angles) & ((cli_water_cloud_test(cli_observed) & cli_stability(cloud_var_observed)) |
                                    sea_cloud_test(angles, is_land, vis_observed)
                                    | gross_cloud_test(lir_observed, lir_forecast))


def exhaustive_dawn_day_snow_test(angles, is_land, ndsi_observed, cli_mu_observed, cli_epsilon_observed, vis_observed,
                                  lir_observed, lir_forecast):
    # from quick_visualization import visualize_map_time
    # visualize_map_time(broad_cirrus_snow_test(cli_epsilon_observed), typical_bbox())
    snow_dawn_day = (dawn_day_test(angles) & is_land & ndsi_test(ndsi_observed) & cli_snow_test(cli_mu_observed) &
                     gross_snow_test(lir_observed, lir_forecast) & broad_cirrus_snow_test(cli_epsilon_observed) &
                     visible_snow_test(vis_observed))
    return snow_dawn_day


def partial_dawn_day_snow_test(angles, is_land, ndsi_observed, cli_observed, vis_observed):
    snow_dawn_day = (dawn_day_test(angles) & is_land & ndsi_test(ndsi_observed) & cli_snow_test(cli_observed) &
                     visible_snow_test(vis_observed))
    return snow_dawn_day


def is_lir_available():
    return True


def suspect_snow_classified_pixels(snow, ndsi, mask_input):
    from visible_predictors import get_bright_negative_variability_5d, get_bright_positive_variability_5d
    from get_data import compute_short_variability
    return ndsi_test(ndsi) & ((get_bright_positive_variability_5d(ndsi, mask_input, typical_time_step(), 1) > 0.2) |
                              (get_bright_negative_variability_5d(ndsi, mask_input, typical_time_step(), 1) > 0.2) |
                              (compute_short_variability(ndsi, mask=mask_input, abs_value=True) > 0.05)) & snow


# def score(not_flagged_yet, vis):
#     thresh = compute_land_visible_threshold()
#     potential_clouds =

def maybe_cloud_after_all(is_land, is_supposed_free, vis):
    # apply only for a few consecutive days
    is_supposed_free_for_long = is_supposed_free & np.roll(is_supposed_free, -1) & np.roll(is_supposed_free, 2) &\
                       is_supposed_free & np.roll(is_supposed_free, -2) & np.roll(is_supposed_free, 2)
    from read_metadata import read_satellite_step
    (slots, lats, lons) = np.shape(vis)
    slot_per_day = get_nb_slots_per_day(read_satellite_step(), 1)
    entire_days = slots / slot_per_day
    vis_copy = vis.copy()
    vis_copy[~is_supposed_free_for_long] = 100
    supposed_clear_sky = np.min(vis_clear_sky_rolling_mean_on_time(vis_copy, 5).reshape((entire_days, slot_per_day, lats, lons)), axis=0)
    # visualize_map_time(supposed_clear_sky, typical_bbox())
    del vis_copy
    vis = vis.reshape((entire_days, slot_per_day, lats, lons))
    # visualize_map_time(is_supposed_free & land_visible_test(is_land, vis, supposed_clear_sky).reshape((slots, lats, lons)), typical_bbox())
    return is_supposed_free & land_visible_test(is_land, vis, supposed_clear_sky).reshape((slots, lats, lons))


def typical_static_classifier():
    from infrared_predictors import get_cloud_index, get_cloud_index_positive_variability_5d
    from utils import typical_outputs
    zen, vis, ndsi, mask_input = typical_outputs('visible', 'ndsi')
    infrared = typical_outputs('infrared', 'channel')
    lands = typical_land_mask()
    is_exhaustive = (np.shape(infrared)[-1] == 3)
    if is_exhaustive:
        cli_epsilon = typical_outputs('infrared', 'cli')[0]
        cli_default = get_cloud_index(np.cos(zen), mir=infrared[:, :, :, 2], lir=infrared[:, :, :, 1],
                                 method='default')
        snow = exhaustive_dawn_day_snow_test(zen, lands, ndsi, cli_default, cli_epsilon, vis, infrared[:, :, :, 1],
                                             expected_brightness_temperature_only_emissivity(
                                                 typical_temperatures_forecast(), lw_nm=10.3, eps=0.95))
    else:
        cli_default = get_cloud_index(np.cos(zen), mir=infrared[:, :, :, 1], lir=infrared[:, :, :, 0], method='default')
        snow = partial_dawn_day_snow_test(zen, lands, ndsi, cli_default, vis)

    from quick_visualization import visualize_map_time

    cli_var = get_cloud_index_positive_variability_5d(cli_default, definition_mask=mask_input, pre_cloud_mask=None,
                                                      satellite_step=typical_time_step(), slot_step=1)
    del cli_default
    cli_mu = get_cloud_index(np.cos(zen), mir=infrared[:, :, :, 2], lir=infrared[:, :, :, 1],
                             method='mu-normalization')

    if is_exhaustive:
        clouds = exhaustive_dawn_day_cloud_test(zen, lands, cli_mu, cli_var, cli_epsilon, vis, infrared[:, :, :, 1],
                                                infrared[:, :, :, 0], expected_brightness_temperature_only_emissivity(
                typical_temperatures_forecast(), lw_nm=10.3, eps=0.95), expected_brightness_temperature_only_emissivity(
                typical_temperatures_forecast(), lw_nm=12.3, eps=0.95),
                                                )
        del cli_epsilon, cli_mu, cli_var
    else:
        clouds = partial_dawn_day_cloud_test(zen, lands, cli_mu, cli_var, vis, infrared[:, :, :, 0],
                                             expected_brightness_temperature_only_emissivity(
                                                 typical_temperatures_forecast(), lw_nm=12.4, eps=0.95))
        del cli_mu, cli_var

    # visualize_map_time(snow, typical_bbox())
    # visualize_map_time(clouds, typical_bbox())
    output = np.asarray(snow, dtype=int)
    output[~dawn_day_test(zen) | mask_input] = -1
    output[clouds] = 2
    output[suspect_snow_classified_pixels(snow, ndsi, mask_input)] = 3
    output[maybe_cloud_after_all(lands, (output == 0), vis)] = 4
    visualize_map_time(output, typical_bbox(), vmin=-1, vmax=4)
    return output


if __name__ == '__main__':
    from utils import *
    from quick_visualization import visualize_map_time
    typical_static_classifier()

