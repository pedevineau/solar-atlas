
def prepare_temperature_mask(lats, lons, beginning, ending, slot_step=1):
    '''
    Create a temperature mask which has the same temporal sampling than spectral channels
    :param lats: latitudes array
    :param lons: longitudes array
    :param beginning: dfb beginning sampling
    :param ending: dfb ending sampling
    :param slot_step: slot sampling chosen by the user (probably 1)
    :return:
    '''
    from read_netcdf import read_temperature_forecast
    from read_metadata import read_satellite_step
    from utils import get_nb_slots_per_day
    from numpy import empty
    satellite_step = read_satellite_step()
    nb_slots = get_nb_slots_per_day(satellite_step, slot_step)*(ending-beginning+1)
    temperatures = read_temperature_forecast(lats, lons, beginning, ending)
    to_return = empty((nb_slots, len(lats), len(lons)))
    for slot in range(nb_slots):
        try:
            nearest_temp_meas = int(0.5+satellite_step*slot_step*slot/60)
            to_return[slot] = temperatures[nearest_temp_meas] + 273.15
        except IndexError:
            nearest_temp_meas = int(satellite_step*slot_step*slot/60)
            to_return[slot] = temperatures[nearest_temp_meas] + 273.15
    return to_return


def expected_brightness_temperature_only_emissivity(forecast_temperature, lw_nm, eps):
    '''
    Compute the brightness temperature we can reasonably expect from the temperature forecast, the wavelength lw and
        the emissivity parameter ems
    NB: if there is also infrared reflectance, the observed brightness temperature will be superior to this "expected" brightness
    NB: this function is designed for clouds recognition (their characteristics are low emissivity & very low reflectance in long infrared)
    :param forecast_temperature:
    :param lw_nm: the wavelength (in nanometers) of the brightness temperature we want compute
    :param eps: the emissivity parameter (in [0, 1]). EG typical emissivity of snow is up to 0.95
    :return: the expected brightness
    '''

    from numpy import log, exp
    c = 3.0 * 10 ** 8
    h = 6.626 * 10 ** (-34)
    k = 1.38 * 10 ** (-23)
    K = h / k
    nu = c / lw_nm
    return 1. / (1 / (K * nu) * log(1 + (exp(K * nu / forecast_temperature) - 1) / eps))