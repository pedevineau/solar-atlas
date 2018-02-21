def compute_gross_cloud_threshold():
    return 5


def compute_gross_snow_threshold():
    # this threshold is a little random here...
    return 5


def compute_cli_snow_threshold():
    return 15


def compute_cli_cloud_threshold():
    return 10


def compute_cloud_stability_threshold():
    return 5


def compute_cloud_epsilon_threshold():
    return 0.25


def compute_thresh_temperature():
    return 240


def compute_vis_snow_threshold():
    return 0.25


def compute_broad_vis_snow_threshold():
    return 0.2


def compute_broad_ndsi_snow_threshold():
    return 0.25


def compute_epsilon_maximal_lir_threshold():
    return 303.15


def compute_epsilon_threshold():
    return 0.2


def compute_mir_sun_glint_threshold():
    # Le Gleau 2016 - to make sure 3.8 is not saturated
    return 320


def compute_vis_sun_glint_threshold():
    # Le Gleau 2016
    return 0.6


def compute_lir_texture_land():
    return 2


def compute_mir_lir_texture_land():
    return 2


def expected_brightness_temperature_only_emissivity(forecast_temperature, lw_nm, eps):
    # if there is also infrared reflectance, the observed brightness temperature will be higher
    # this function is designed for clouds recognition (their characteristics are low emissivity & very low reflectance in long infrared)
    from numpy import log, exp
    c = 3.0 * 10 ** 8
    h = 6.626 * 10 ** (-34)
    k = 1.38 * 10 ** (-23)
    K = h / k
    nu = c / lw_nm
    return 1. / (1 / (K * nu) * log(1 + (exp(K * nu / forecast_temperature) - 1) / eps))


def compute_vis_sea_coasts_cloud_factors(zen):
    from numpy import cos, power
    cos_zen = cos(zen)
    cos_zen[cos_zen < 0.03] = 0.03
    cos_zen = power(cos_zen, 0.3)
    return 1/cos_zen


def compute_vis_coasts_cloud_all_coefficient():
    return 0.3


def compute_vis_sea_cloud_all_coefficient():
    return 0.15


def compute_snow_ndsi_threshold():
    return 0.3


def compute_broad_cirrus_threshold():
    return 0.3


def compute_thin_cirrus_threshold():
    return 10


def compute_evolution_lir_fir_threshold():
    return 10


def compute_land_evolution_lir():
    return 5


def compute_sea_evolution_lir():
    return 3


def compute_land_visible_threshold():
    return 0.1
