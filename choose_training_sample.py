from numpy.random import randint, shuffle
from utils import *


def get_temporally_stratified_samples(array, training_rate, number_of_layers, me, std):
    nb_slots, nb_lats, nb_lons, nb_features = np.shape(array)
    subdivisions = np.linspace(0, nb_slots, number_of_layers+1, endpoint=True, dtype=int)  # supposed to be integers
    pis = np.empty(number_of_layers)
    montecarlo_number = 36
    ### beginning of Montecarlo ###
    for k in range(len(subdivisions)-1):
        ck = 0
        for test in range(montecarlo_number):
            lat = randint(0, nb_lats)
            lon = randint(0, nb_lons)
            slot = randint(subdivisions[k], subdivisions[k+1])
            if not looks_like_night(array[slot, lat, lon], me, std):
                ck += 1
        pis[k]=ck
    s = 1.*sum(pis)
    # normalization of empirical factors
    pis = pis / s
    nb_points_picked = 0
    total_training_len = int(training_rate * nb_slots)
    array_to_return = np.empty((total_training_len, nb_lats, nb_lons, nb_features))
    for k in range(len(subdivisions)-1):
        nb_points_to_pick = int(total_training_len * pis[k])
        slice_for_picking = array[subdivisions[k]:subdivisions[k + 1]]
        shuffle(slice_for_picking)
        array_to_return[nb_points_picked:nb_points_picked + nb_points_to_pick] = slice_for_picking[0:nb_points_to_pick]
        nb_points_picked += nb_points_to_pick
        if subdivisions[k + 1] == nb_slots and nb_points_picked < total_training_len:  # possible because of rounded errors
            try:
                array_to_return[nb_points_picked:total_training_len] \
                    = slice_for_picking[nb_points_to_pick:nb_points_to_pick+total_training_len-nb_points_picked]
            except:
                continue
    return array_to_return


def get_spatially_stratified_samples(array, training_rate, number_of_layers, me, std):
    nb_slots, nb_lats, nb_lons, nb_features = np.shape(array)
    subdivisions = np.linspace(0, nb_slots, number_of_layers+1, endpoint=True, dtype=int)  # supposed to be integers
    pis = np.empty(number_of_layers)
    montecarlo_number = 36
    ### beginning of Montecarlo ###
    for k in range(len(subdivisions)-1):
        ck = 0
        for test in range(montecarlo_number):
            lat = randint(0, nb_lats)
            lon = randint(0, nb_lons)
            slot = randint(subdivisions[k], subdivisions[k+1])
            if not looks_like_night(array[slot, lat, lon], me, std):
                ck += 1
        pis[k]=ck
    s = 1.*sum(pis)
    # normalization of empirical factors
    pis = pis / s
    nb_points_picked = 0
    total_training_len = int(training_rate * nb_slots)
    array_to_return = np.empty((total_training_len, nb_lats, nb_lons, nb_features))
    for k in range(len(subdivisions)-1):
        nb_points_to_pick = int(total_training_len * pis[k])
        slice_for_picking = array[subdivisions[k]:subdivisions[k + 1]]
        shuffle(slice_for_picking)
        array_to_return[nb_points_picked:nb_points_picked + nb_points_to_pick] = slice_for_picking[0:nb_points_to_pick]
        nb_points_picked += nb_points_to_pick
        if subdivisions[k + 1] == nb_slots and nb_points_picked < total_training_len:  # possible because of rounded errors
            try:
                array_to_return[nb_points_picked:total_training_len] \
                    = slice_for_picking[nb_points_to_pick:nb_points_to_pick+total_training_len-nb_points_picked]
            except:
                continue
    return array_to_return


def evaluate_randomization(array, me, std):
    nb_slots, nb_lats, nb_lons, nb_features = np.shape(array)
    count = 0
    N = 500
    for test in range(N):
        lat = randint(0, nb_lats)
        lon = randint(0, nb_lons)
        slot = randint(0, nb_slots)
        if not looks_like_night(array[slot, lat, lon], me, std):
            count += 1
    print 'COUNT IS'
    print count


if __name__ == '__name__':
    print