import numpy as np


def merge_3d_time(array_1, array_2):
    return array_1 + array_2


def merge_3d_lat(array_1, array_2):
    return np.array([
        array_1[k] + array_2[k] for k in range(len(array_1))
    ])


def merge_3d_lon(array_1, array_2):
    return np.array([
        array_1[:][k] + array_2[:][k] for k in range(len(array_1[0]))
    ])


training = [
    [
        [1,2,np.nan],[4,5,6]
    ],
    [
        [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]
    ]
]

def filter_aberrant_training_data(training_data):
    training_set = []
    for given_slot in training_data:
        valide_slot = []
        for given_slot_latitude in given_slot:
            valide_latitude = []
            for given_coordinates in given_slot_latitude:
                if not np.isnan(given_coordinates):
                    valide_latitude.append(given_coordinates)
            if valide_latitude:
                valide_slot.append(valide_latitude)
        training_set.append(valide_slot)
    return np.asarray(training_set)

L = np.full((5,31,31), -1)
print L[0,1]