from utils import *


def statistics_classes(classes, display_now=True):
    nb_classes = int(np.max(classes))
    nb_slots = np.shape(classes)[0]
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    patches = []
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'burlywood', 'chartreuse', '#cccccc', '#444444', '#333333',
              '#666666', '#777777', '#888888', '#999999', '#aaaaaa', '#bbbbbb', '#222222']
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


def medians_index(index, mask, display_now=True):
    import matplotlib.pyplot as plt
    medians = []
    for slot in range(len(index)):
        medians.append(np.median(index[slot][~mask[slot]]))
    plt.plot(medians)
    plt.title('Medians of the index:')
    if display_now:
        plt.show()


def comparision_visible(vis, classes):
    threshold_visible = 0.4
    comparision = np.empty_like(vis)
    comparision[(vis > threshold_visible) & (classes == 0)] = -2
    comparision[(vis > threshold_visible) & ((classes == 2) | (classes == 4))] = -1
    comparision[(vis < threshold_visible) & ((classes == 2) | (classes == 4))] = 1
    comparision[(vis < threshold_visible) & (classes != 0) & (classes != 2) & (classes != 4) & (classes != 1-2)] = 2
    return comparision


if __name__ == '__main__':
    slot_step = 1
    beginning = 13516+17
    nb_days = 5
    ending = beginning + nb_days - 1

    latitude_beginning = 45.
    latitude_end = 50.
    longitude_beginning = 125.
    longitude_end = 130.
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end,
                                                     longitude_beginning, longitude_end)

    from get_data import get_features
    from decision_tree import get_classes_decision_tree

    vis = get_features('visible', latitudes, longitudes, beginning, ending, False)[:, :, :, 1]
    classes = get_classes_decision_tree(latitudes, longitudes, beginning, ending, slot_step=1, normalize=False)

    comparision = comparision_visible(
        vis,
        classes
    )

    from quick_visualization import get_bbox, visualize_map_time
    bbox = get_bbox(latitude_beginning,
                    latitude_end,
                    longitude_beginning,
                    longitude_end)

    visualize_map_time(classes,
                       bbox,
                       vmin=0,
                       vmax=12,
                       )

    visualize_map_time(vis,
                       bbox
                       )

    visualize_map_time(comparision,
                       bbox,
                       vmin=-2,
                       vmax=2
                       )

    statistics_classes(classes)

    from read_netcdf import read_classes
    # classes = read_classes(latitudes, longitudes, beginning, ending, slot_step)
    # statistics_classes(classes)
    #