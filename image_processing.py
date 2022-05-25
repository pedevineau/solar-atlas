import cv2
import numpy as np
from cv2 import equalizeHist
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.filters import rank
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import disk
from skimage.morphology import opening, octahedron

# from skimage.feature import peak_local_max
from skimage.morphology import watershed

from get_data import get_features
from utils import *


def segmentation(method, feature, chan=None, thresh_method="otsu", static=None):
    if len(np.shape(feature)) == 4:
        assert (
            chan is not None and chan < np.shape(feature)[-1]
        ), "please give a valid channel number"
        feature = feature[:, :, :, chan]
    assert method in [
        "otsu-2d",
        "otsu-3d",
        "watershed-2d",
        "watershed-3d",
    ], "segmentation method unknown"
    if method == "otsu-2d":
        return segmentation_otsu_2d(feature)
    elif method == "otsu-3d":
        return segmentation_otsu_3d(feature)
    elif method == "watershed-2d":
        return segmentation_watershed_2d(feature, thresh_method, static=static)
    elif method == "watershed-3d":
        return segmentation_watershed_3d(feature, thresh_method, static=static)


def segmentation_otsu_2d(feature):
    to_return = np.zeros_like(feature, dtype=bool)
    nb_slots = np.shape(to_return)[0]
    for slot in range(nb_slots):
        try:
            to_return[slot] = feature[slot] > threshold_otsu(feature[slot])
        except ValueError:
            pass
    return to_return


def segmentation_otsu_3d(feature):
    # skimage library
    try:
        return feature > threshold_otsu(feature)
    except ValueError:
        # if the feature is monochromatic
        return np.zeros_like(feature, dtype=bool)


def segmentation_watershed_2d(
    feature, thresh_method="otsu", coherence=0.2, static=None
):
    (slots, lats, lons) = np.shape(feature)
    to_return = np.empty((slots, lats, lons), dtype=bool)
    for slot in range(slots):
        # opencv library
        to_return[slot] = watershed_2d(feature[slot], thresh_method, coherence, static)
    return to_return


def segmentation_watershed_3d(
    feature, thresh_method="otsu", coherence=0.3, static=None
):
    return watershed_3d(feature, coherence, thresh_method, static)


def watershed_3d(feature, coherence, thresh_method, static):
    # kernel = cube(3)  # try other forms
    kernel = octahedron(1)
    assert thresh_method in [
        "otsu",
        "static",
        "static-inverted",
        "binary",
    ], "threshing method unknown"
    if thresh_method == "otsu":
        try:
            thresh = feature < threshold_otsu(feature)  # mask=True for background
        except ValueError:
            # if the feature is monochromatic
            return np.zeros_like(feature, dtype=bool)
    elif thresh_method == "static-inverted":
        thresh = feature > static
    elif thresh_method == "static":
        thresh = feature < static
    else:  # if the image is already binary
        thresh = feature == 0
    opened = opening(thresh, kernel)
    # opened = opening(opened, kernel)
    # opened = opening(opened, kernel)

    # sure background area
    # sure_bg = dilation(opened, kernel)
    # sure_bg = dilation(sure_bg, kernel)
    # sure_bg = dilation(sure_bg, kernel)
    # Finding sure foreground area
    dist_transform = distance_transform_edt(opened)
    sure_fg = dist_transform > coherence * dist_transform.max()

    # Finding unknown region
    unknown = opened ^ sure_fg > 0

    # Marker labelling
    markers = label(sure_fg, background=0)
    # Add one to all labels so that sure background is not 0, but 1
    markers += 1
    # Now, mark the region of unknown with zero
    markers[unknown] = 0
    try:
        water = watershed(-dist_transform, markers, mask=thresh)
    except NameError:
        water = watershed(-dist_transform, markers)
    return water == 1


def watershed_2d(image, thresh_method, spatial_coherence=0.2, static=None):
    # Image Segmentation with Watershed Algorithm
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
    # noise removal
    assert thresh_method in ["otsu", "static", "binary"], "threshing method unknown"
    if thresh_method == "otsu":
        thresh = apply_inverted_otsu(image)[0]
    elif thresh_method == "static":
        thresh = image > static
    else:
        thresh = image

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    # opencv3: cv2.DIST_L2, opencv2: cv2.cv.CV_DIST_L2
    try:
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    except:
        dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
    ret, sure_fg = cv2.threshold(
        dist_transform, spatial_coherence * dist_transform.max(), 255, 0
    )

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    markers = label(sure_fg, background=0)
    # Add one to all labels so that sure background is not 0, but 1
    markers += 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    D = ndimage.distance_transform_edt(thresh)

    water = watershed(-D, markers, mask=thresh)
    # loop over the unique labels returned by the Watershed
    # algorithm
    # fig, ax = plt.subplots()
    # contours = find_contours(labels, 1)
    # for n, contour in enumerate(contours):
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # ax.imshow(image, cmap=plt.cm.gray)
    # plt.show()
    return water == 1


def apply_otsu(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th, ret


def apply_inverted_otsu(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th, ret


def get_local_otsu(img):
    radius = 5
    selem = disk(radius)
    local_otsu = rank.otsu(img, selem)
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    # ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img > local_otsu


def equalize_histogram_2d(img):
    return equalizeHist(img)


def equalize_histograms_all_features(features):
    nb_dims = len(features.shape)
    assert nb_dims in [3, 4], "invalid shape for features"
    for slot in range(len(features)):
        if nb_dims == 3:
            features[slot] = equalize_histogram_2d(features[slot])
        elif nb_dims == 4:
            for feat in range(features.shape[-1]):
                features[slot, :, :, feat] = equalize_histogram_2d(
                    features[slot, :, :, feat]
                )
    return features


if __name__ == "__main__":
    types_channel = ["infrared", "visible"]
    compute_indexes = True
    chan = 0
    channel_number = 1
    type_channels = types_channel[channel_number]
    (
        beginning,
        ending,
        latitude_beginning,
        latitude_end,
        longitude_beginning,
        longitude_end,
    ) = typical_input(seed=0)
    date_begin, date_end = print_date_from_dfb(beginning, ending)
    lat, lon = get_latitudes_longitudes(
        latitude_beginning, latitude_end, longitude_beginning, longitude_end
    )
    # from quick_visualization import get_bbox
    # bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    features = get_features(
        type_channels,
        lat,
        lon,
        beginning,
        ending,
        compute_indexes,
        slot_step=1,
        gray_scale=True,
    )

    segmentation_watershed_2d(features[16, :, :, chan])

    segmentation_otsu_2d(features, chan)
