from utils import *
from get_data import get_features
from quick_visualization import visualize_input, get_bbox


def get_auto_corr_array(x):
    result = np.correlate(x, x, mode='full')
    # print result
    return result[1 + result.size / 2:]


def get_autocorr_predictor_array_2d(x):
    auto_x = get_auto_corr_array(x)
    return np.max(auto_x[142:146]) / (np.max(auto_x))


def get_random_database(normalize):
    latitude_beginning = 45.0   # salt lake mongolia  45.
    latitude_end = 46.0
    longitude_beginning = 125.0
    longitude_end = 126.0
    dfb_beginning = 13512+60
    lat, lon = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    bbox=get_bbox(latitude_beginning,latitude_end,longitude_beginning,longitude_end)
    features_pre = get_features(lat, lon, dfb_beginning, dfb_beginning+10, True, normalize=normalize)
    latitude_beginning = 36.0   # salt lake mongolia  45.
    latitude_end = 37.0
    longitude_beginning = 128.0
    longitude_end = 129.0
    dfb_beginning = 13527+60
    lat, lon = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    features_bis = get_features(lat, lon, dfb_beginning, dfb_beginning+10, True, normalize=normalize)
    latitude_beginning = 35.0   # salt lake mongolia  45.
    latitude_end = 36.0
    longitude_beginning = 128.0
    longitude_end = 129.0
    dfb_beginning = 13542+60
    lat, lon = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    features_ter = get_features(lat, lon, dfb_beginning, dfb_beginning+10, True, normalize=normalize)
    latitude_beginning = 38.0   # salt lake mongolia  45.
    latitude_end = 39.0
    longitude_beginning = 125.0
    longitude_end = 126.0
    dfb_beginning = 13562+60
    lat, lon = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    features_qua = get_features(lat, lon, dfb_beginning, dfb_beginning+10, True, normalize=normalize)
    return np.concatenate((features_pre, features_bis, features_ter, features_qua))


def get_automatic_threshold(normalize, nb_days=3):
    features = get_random_database(normalize)
    number_attempts = 200
    scores = []
    coordinates = []
    feature_number = 1
    perc = 95 # should be an integer between 0 et 100
    for att in range(number_attempts):
        latind = np.random.randint(0, 30)
        lonind = np.random.randint(0, 30)
        day_beg = np.random.randint(0, 40-nb_days)
        slot_b = 144 * day_beg
        slot_e = slot_b + 144 * nb_days
        data = features[slot_b:slot_e, latind, lonind, feature_number]
        scores.append(get_autocorr_predictor_array_2d(data))
        coordinates.append([slot_b, slot_e, latind, lonind])
    alpha = np.percentile(scores, perc)
    print 'alpha', alpha
    print 'scores', scores
    for k in range(number_attempts):
        if scores[k] >= alpha:
            [slot_b, slot_e, latind, lonind] = coordinates[k]
            data_main = features[slot_b:slot_e, latind, lonind, feature_number]
            print 'current score', scores[k]
            visualize_input(data_main, display_now=False, title='main')
            if feature_number in [0, 1]:
                data_secondary = features[slot_b:slot_e, latind, lonind, 1-feature_number]
                visualize_input(data_secondary, display_now=True, title='other')


def get_threshold_manually(data_array_3d, bbox=None):
    number_attempts = 20
    th = []
    rjt = []
    nb_days = 2
    for k in range(number_attempts):
        latind = np.random.randint(0, 30)
        lonind = np.random.randint(0, 30)
        day_beg = np.random.randint(0, 20)
        slot_b = 144 * day_beg
        slot_e = slot_b + 144 * nb_days
        data_1d_cli = data_array_3d[slot_b:slot_e, latind, lonind, 1]
        print (slot_b, latind, lonind)
        # visualize_map(data_array_3d[slot_b+30,:,:,1])
        visualize_input(data_1d_cli, display_now=True, title='Input')
        print 'Seems regular? (1/0)'
        auto = get_autocorr_predictor_array_2d(data_1d_cli)
        if raw_input() == '1':
            data_1d_ndsi = data_array_3d[slot_b:slot_e, latind, lonind, 0]
            print np.max(np.correlate(data_1d_cli,data_1d_cli))
            print np.max(np.correlate(data_1d_ndsi, data_1d_ndsi))
            print np.max(np.correlate(data_1d_ndsi,data_1d_cli))
            visualize_input(data_1d_ndsi, display_now=True, title='Input')
            # visualize_map_time(data_array_3d[:, :, :, 1:], bbox)
            th.append(auto)
        else:
            rjt.append(auto)
        # del data_1d_cli, data_1d_nsdi
    print 'th', th
    print 'rjt', rjt
    return np.min(th)


if __name__ == '__main__':
    # random bbox
    # bbox=get_bbox(0,5,0,5)
    print get_automatic_threshold(normalize=True, nb_days=3)