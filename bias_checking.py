import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from decision_tree import get_classes_v2_image
from get_data import get_features
from utils import *
from visualize import get_bbox, visualize_map_time


def statistics_classes(classes, display_now=True):
    nb_classes = int(np.max(classes)) + 1
    nb_slots = np.shape(classes)[0]
    plt.clf()
    patches = []
    colors = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "k",
        "burlywood",
        "chartreuse",
        "#cccccc",
        "#444444",
        "#333333",
        "#666666",
        "#777777",
        "#888888",
        "#999999",
        "#aaaaaa",
        "#bbbbbb",
        "#222222",
    ]
    for class_number in range(nb_classes):
        time_occurrence = np.empty(nb_slots)
        for slot in range(nb_slots):
            time_occurrence[slot] = (classes[slot, :, :] == class_number).sum() / float(
                classes.size
            )
        m = np.max(time_occurrence)
        if m > 0 and np.log10(m) > -5:  # if it is sometimes more than 1/10e-5
            plt.plot(time_occurrence, color=colors[class_number])
            patches.append(
                mpatches.Patch(
                    color=colors[class_number], label="class " + str(class_number)
                )
            )
    plt.legend(handles=patches)
    if display_now:
        plt.show()


def medians_index(index, mask, display_now=True):
    medians = []
    for slot in range(len(index)):
        medians.append(np.median(index[slot][~mask[slot]]))
    plt.plot(medians)
    plt.title("Medians of the index:")
    if display_now:
        plt.show()


def comparision_visible(vis, classes):
    threshold_visible = 0.35
    comparision = np.empty_like(vis)
    comparision[(vis > threshold_visible) & (classes == 0)] = -1
    comparision[
        (vis < threshold_visible)
        & (classes != 0)
        & (classes != 2)
        & (classes != 4)
        & (classes != 7)
        & (classes != 12)
    ] = 1
    return comparision


def comparision_algorithms(reduced_classes_1, reduced_classes_2):
    comparision = np.zeros_like(reduced_classes_1)
    cloudy_1 = reduced_classes_1 == 1
    cloudy_2 = reduced_classes_2 == 1
    comparision[cloudy_1 & ~cloudy_2] = 1
    comparision[~cloudy_1 & cloudy_2] = -1
    return comparision


if __name__ == "__main__":
    slot_step = 1
    beginning = 13525
    nb_days = 3
    ending = beginning + nb_days - 1

    latitude_beginning = 40.0
    latitude_end = 45.0
    longitude_beginning = 125.0
    longitude_end = 130.0

    latitudes, longitudes = get_latitudes_longitudes(
        latitude_beginning, latitude_end, longitude_beginning, longitude_end
    )

    print_date_from_dfb(beginning, ending)

    vis = get_features("visible", latitudes, longitudes, beginning, ending, False)[
        :, :, :, 1
    ]
    # classes = get_classes_v1_point(latitudes, longitudes, beginning, ending, slot_step=1)
    classes = get_classes_v2_image(
        latitudes, longitudes, beginning, ending, slot_step=1, method="watershed-3d"
    )

    comparision = comparision_visible(vis, classes)

    bbox = get_bbox(
        latitude_beginning, latitude_end, longitude_beginning, longitude_end
    )

    visualize_map_time(
        classes,
        bbox,
        vmin=0,
        vmax=12,
    )

    visualize_map_time(comparision, bbox, vmin=-2, vmax=2)

    statistics_classes(classes)
