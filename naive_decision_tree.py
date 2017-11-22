from get_data import *
from scipy.stats import pearsonr


# WARNING: the following function is utterly empirical. This correlation should be studied from physical point of view
def looks_like_bare_land(cli_ref, mu_ref, tolerance=0.05):
    from quick_visualization import visualize_input
    # visualize_input(cli_ref, display_now=True)
    # visualize_input(mu_ref, display_now=True)

    # tolerance is the key parameter of the algorithm
    p, r = pearsonr(cli_ref, mu_ref)
    print p, 1-tolerance
    return p > 1 - tolerance


if __name__ == '__main__':


    #### WARNING!!!!!!    CURRENT ALGORITHM CAN NOT WORK LIKE THAT, BECAUSE IT SUPPOSE IMPLICITLY THAT NO CLOUDS APPEAR OR DISAPPEAR IN THE TILE, WHAT IS NONSENSE !!! ####

    step_percen = 10

    slot_step = 1  # dont change it
    training_rate = 0.05 # critical     # mathematical training rate is training_rate / slot_step
    randomization = False  # to select training data among input data
    beginning = 13527+6
    nb_days = 1
    ending = beginning + nb_days - 1
    compute_indexes = True
    normalize = False
    normalization = 'standard'

    latitude_beginning = 35.0
    latitude_end = 40.
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    infrared_features, mu = get_features(
        'infrared',
        latitudes,
        longitudes,
        beginning,
        ending,
        compute_indexes,
        slot_step,
        normalize,
        normalization,
        return_mu=True
    )

    from quick_visualization import visualize_map_time, get_bbox, visualize_input
    bbox = get_bbox(latitudes[0], latitudes[-1], longitudes[0], longitudes[-1])

    # visualize_map_time(infrared_features[:,:,:,0], bbox=bbox, vmax=1, vmin=-1)

    from numpy import percentile, empty_like, shape, empty, zeros, full

    from numpy.random import randint

    (a,b,c,d) = shape(infrared_features)
    class_clouds = full((a,b,c), -10)

    num = 5
    mu_ref = mu[:, len(latitudes) / 2, len(longitudes) / 2]  # use middle pixel

    nb_slots = len(infrared_features)
    last_percens_bare_land = zeros(nb_slots)

    # lat0, lon0 = 38.35, 126.55
    # print 'lat, lon', lat0, lon0
    # lat_pix, lon_pix = int((lat0 - latitude_beginning) * 60 / 2.), int((lon0 - longitude_beginning) * 60 / 2.)
    # print 'pixs', lat_pix, lon_pix
    # visualize_input(infrared_features[:, lat_pix,lon_pix,0])

    while num <= 100:
        print 'num', num
        percens = zeros(nb_slots)
        is_day = zeros(nb_slots, dtype=bool)
        for slot in range(nb_slots):
            current_cli = infrared_features[slot, :,:,0]
            data = current_cli[current_cli > - 10]
            if data.size > 0:
                p = percentile(data, num)          # cli = - 10 is the mask
                percens[slot] = p
                is_day[slot] = True
        slot = 0
        while slot < nb_slots:
            if slot >= 4 and not is_day[slot] and is_day[slot-1]:  # first slot of night => it is just after twilight. remove 4 previous slots
                is_day[slot-1] = False
                is_day[slot-2] = False
                is_day[slot-3] = False
                is_day[slot-4] = False
            if slot < nb_slots - 4 and not is_day[slot] and is_day[slot+1]:  # last slot of night => it is just before dawn. remove 3 following slots
                is_day[slot+1] = False
                is_day[slot+2] = False
                is_day[slot+3] = False
                is_day[slot + 4] = False
                slot += 4
            slot += 1

        # print percens
        # print percens[is_day]
        # print mu_ref[is_day]
        if looks_like_bare_land(cli_ref=percens[is_day], mu_ref=mu_ref[is_day]):    # sure, this correlation is not the same in winter and summer. Winter: more zeros => more correlation
            last_percens_bare_land = percens.copy()  # in case of mutable array (to check)
            num += step_percen
        else:
            print 'break'
            break  # leave while(num) loop

    for slot in range(nb_slots):
        ### threshold will be p
        if is_day[slot]:
            # print 'threshold', last_percens_bare_land[slot]
            class_clouds[slot, (infrared_features[slot, :,:,0] > -10) & (infrared_features[slot, :,:,0] <= last_percens_bare_land[slot])] = 0  # points which are nor masked nor clouds
            class_clouds[slot, infrared_features[slot, :,:,0] > last_percens_bare_land[slot]] = 1
            print 'ratio_1', len(class_clouds[class_clouds == 1])/(1.*len(class_clouds[class_clouds == 0])+len(class_clouds[class_clouds == 1]))
            class_clouds[slot, infrared_features[slot, :,:,2]==2] = 1

    visualize_map_time(array_map=class_clouds, bbox=bbox)


