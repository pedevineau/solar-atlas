def compute_gross_cloud_threshold():
    return 10


def compute_gross_snow_threshold():
    return 10


def compute_cli_snow_threshold():
    return 10


def compute_cli_cloud_threshold():
    return 30


def compute_thresh_temperature():
    return 240


def compute_vis_snow_threshold():
    return 0.35


def compute_vis_sea_cloud_all_thresholds(zen):
    from numpy import cos, power
    cos_zen = cos(zen)
    cos_zen[cos_zen < 0.03] = 0.03
    cos_zen = power(cos_zen, 0.3)
    return 0.2/cos_zen


def compute_snow_ndsi_threshold():
    return 0.3


def compute_broad_cirrus_threshold():
    return 10


def compute_thin_cirrus_threshold():
    return 10


def compute_evolution_lir_fir_threshold():
    return 10


def compute_land_evolution_lir():
    return 5


def compute_sea_evolution_lir():
    return 3


def compute_land_visible_threshold():
    return 2
