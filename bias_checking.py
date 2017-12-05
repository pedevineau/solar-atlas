from utils import *


def statistics_classes(classes, display_now=True):
    nb_classes = int(np.max(classes))
    nb_slots = np.shape(classes)[0]
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    patches = []
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'burlywood', 'chartreuse', '#cccccc', '#444444', '#333333']
    for class_number in range(nb_classes):
        time_occurrence = np.empty(nb_slots)
        for slot in range(nb_slots):
            time_occurrence[slot] = (classes[slot, :, :] == class_number).sum() / float(classes.size)
        m = np.max(time_occurrence)
        if m > 0 and np.log10(m) > -5: # if it is sometimes more than 1/10e-5
            plt.plot(time_occurrence, color=colors[class_number])
            patches.append(mpatches.Patch(color=colors[class_number], label='class '+str(class_number)))
    plt.legend(handles=patches)
    if display_now:
        plt.show()


if __name__ == '__main__':
    from read_netcdf import read_classes
    slot_step = 1
    beginning = 13516+17
    nb_days = 3
    ending = beginning + nb_days - 1

    latitude_beginning = 45.
    latitude_end = 50.
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    classes = read_classes(latitudes, longitudes, beginning, ending, slot_step)
    statistics_classes(classes)