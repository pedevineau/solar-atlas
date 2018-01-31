from thresholds import *


def threshold_test_positive(observed, forecast, thresh):
    return (observed - forecast) > thresh


def gross_cloud_test(lir_observed, lir_forecast):
    thresh = compute_gross_cloud_threshold()
    return threshold_test_positive(- lir_observed, - lir_forecast, thresh)


def gross_snow_test(lir_observed, lir_forecast):
    '''
    snow (or ice) is supposed to be colder than snow-free pixels under clear-sky conditions
    :param lir_observed:
    :param lir_forecast:
    :return:
    '''
    thresh_gross = compute_gross_cloud_threshold()  # coldness which would be due to clouds if there were clouds
    thresh_snow = compute_gross_snow_threshold()  # coldness due to snow
    thresh = - thresh_gross - thresh_snow
    return threshold_test_positive(lir_observed, lir_forecast, thresh)


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
    for slot in range(s[0]):
        try:
            nearest_temp_meas = int(0.5+satellite_step*slot_step*slot/60)
            to_return[slot] = (273.15+temperature_mask[nearest_temp_meas] - lir_observed[slot]) > 20
        except IndexError:
            nearest_temp_meas = int(satellite_step*slot_step*slot/60)
            to_return[slot] = (273.15+temperature_mask[nearest_temp_meas] - lir_observed[slot]) > 20
    return to_return


def ndsi_test(ndsi, cos_scat=None):
    from numpy import square
    static_thresh = compute_snow_ndsi_threshold()
    if cos_scat is None:
        return ndsi > static_thresh
    else:
        return ndsi > static_thresh + 0.15 * square(cos_scat - 1)


def broad_cirrus_snow_test(lir_observed, fir_observed):
    thresh = compute_broad_cirrus_threshold()
    return (lir_observed - fir_observed) < thresh


def cli_snow_test(cli):
    thresh = compute_cli_snow_threshold()
    return cli < thresh


def cli_water_cloud_test(cli):
    thresh = compute_cli_cloud_threshold()
    return cli > thresh


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


def exhaustive_dawn_day_cloud_test(angles, is_land, cli_observed, vis_observed, lir_observed, fir_observed, lir_forecast, fir_forecast):
    return dawn_day_test(angles) & (cli_water_cloud_test(cli_observed) |
                                    sea_cloud_test(angles, is_land, vis_observed) |
                                    gross_cloud_test(lir_observed, lir_forecast) |
                                    thin_cirrus_test(is_land, lir_observed, fir_observed, lir_forecast, fir_forecast))


def partial_dawn_day_cloud_test(angles, is_land, cli_observed, vis_observed):
    return dawn_day_test(angles) & (cli_water_cloud_test(cli_observed) | sea_cloud_test(angles, is_land, vis_observed))


def exhaustive_dawn_day_snow_test(angles, is_land, ndsi_observed, cli_observed, vis_observed, lir_observed, fir_observed, lir_forecast):
    snow_dawn_day = (dawn_day_test(angles) & is_land & ndsi_test(ndsi_observed) & cli_snow_test(cli_observed) &
                     gross_snow_test(lir_observed, lir_forecast) & broad_cirrus_snow_test(lir_observed, fir_observed) &
                     visible_snow_test(vis_observed))
    return snow_dawn_day


def partial_dawn_day_snow_test(angles, is_land, ndsi_observed, cli_observed, vis_observed):
    snow_dawn_day = (dawn_day_test(angles) & is_land & ndsi_test(ndsi_observed) & cli_snow_test(cli_observed) &
                     visible_snow_test(vis_observed))
    return snow_dawn_day


def is_lir_available():
    return True


if __name__ == '__main__':
    print 'running static_tests.py'
    from numpy import array
    seeds = array([ [False, False, False, False],[True, False, False, False], [False, False, True, False]])
    visible = array([[1.0,0,0,0],[0.8, 0.9, 0.2,0], [0.3, 0.1, 0.6,0]])
    lll = [[0,0,0,0],[0, 0, 0,0], [0, 0, 0,0]]
