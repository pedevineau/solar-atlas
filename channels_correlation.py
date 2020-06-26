from time import time

from scipy.stats import pearsonr

from get_data import get_features
from utils import *
from visualize import get_bbox, visualize_map

if __name__ == '__main__':

    beginning = 13517
    nb_days = 10
    ending = beginning + nb_days - 1
    latitude_beginning = 35. + 15
    latitude_end = 40. + 15
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    date_begin, date_end = print_date_from_dfb(beginning, ending)
    print beginning, ending

    type_channels = 0  # 0: infrared, 1: visible

    if type_channels == 0:
        infra = get_features(
            'infrared',
            latitudes,
            longitudes,
            beginning,
            ending,
            output_level=True,
            slot_step=1,
            gray_scale=False,
        )

        cli = infra[:, :, :, 0]
        unbiased = infra[:, :, :, 1]

        infrared_features = get_features(
            'infrared',
            latitudes,
            longitudes,
            beginning,
            ending,
            output_level=False,
            slot_step=1,
            gray_scale=False
        )

    elif type_channels == 1:
        visible_features = get_features(
            'visible',
            latitudes,
            longitudes,
            beginning,
            ending,
            output_level=False,
            slot_step=1,
            gray_scale=False
        )
        features = get_features(
            'visible',
            latitudes,
            longitudes,
            beginning,
            ending,
            output_level=True,
            slot_step=1,
            gray_scale=False,
        )

        ndsi = features[:, :, :, 0]

    nb_latitudes = len(latitudes)
    nb_longitudes = len(longitudes)

    if type_channels == 0:
        infrared_means = np.empty((nb_latitudes, nb_longitudes))
        infrared_correlations = np.empty((nb_latitudes, nb_longitudes))
        cli_means = np.empty((nb_latitudes, nb_longitudes))
        var_means = np.empty((nb_latitudes, nb_longitudes))
    elif type_channels == 1:
        visible_correlations = np.empty((nb_latitudes, nb_longitudes))
        visible_means = np.empty((nb_latitudes, nb_longitudes))

    t_corrs = time()
    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            if type_channels == 0:
                mask = (infra[:, lat, lon, 0] == -10)
                infrared_means[lat, lon] = np.mean(infrared_features[:, lat, lon, 1][~mask] -
                                                   infrared_features[:, lat, lon, 0][~mask])
                infrared_correlations[lat, lon] = pearsonr(infrared_features[:, lat, lon, 0][~mask],
                                                           infrared_features[:, lat, lon, 1][~mask])[0]
                cli_means[lat, lon] = np.mean(cli[:, lat, lon][~mask])
                var_means[lat, lon] = np.mean(unbiased[:, lat, lon][~mask])

            elif type_channels == 1:
                mask_mu = ((mu[:, lat, lon] < 0.05) | (visible_features[:, lat, lon, 0] == -1)
                           | (visible_features[:, lat, lon, 1] == -1))
                visible_means[lat, lon] = np.mean(ndsi[:, lat, lon][~mask_mu])
                visible_correlations[lat, lon] = pearsonr(visible_features[:, lat, lon, 0][~mask_mu],
                                                          visible_features[:, lat, lon, 1][~mask_mu])[0]

    print 'total time corr:', time() - t_corrs, '; nb pixels:', nb_latitudes * nb_longitudes, '; nb slots:', 144 * nb_days

    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)

    if type_channels == 0:
        visualize_map(infrared_correlations)
        visualize_map(infrared_means)
        visualize_map(cli_means)
        visualize_map(var_means)
        bias = normalize(infrared_means, normalization='standard') - \
               normalize(var_means, normalization='standard')
        visualize_map(bias)
    elif type_channels == 1:
        visualize_map(visible_correlations)
        visualize_map(visible_means)
