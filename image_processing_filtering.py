from utils import *


def segmentation(features, chan):
    (slots, lats, lons) = np.shape(features)[0:3]
    from skimage import measure
    # from skimage.exposure import rescale_intensity
    # features = rescale_intensity(features))
    to_return = np.empty((slots, lats, lons), dtype=bool)
    for slot in range(slots):
        # Find contours at a constant value of 0.8
        import matplotlib.pyplot as plt
        img, ret = apply_otsu(features[slot, :, :, chan])
        to_return[slot] = (img == 255)
        # contours = measure.find_contours(img, 0.8)
        # array_contours.append(contours)
        # fig, ax = plt.subplots()
        # ax.imshow(features[slot,:,:, chan], interpolation='nearest', cmap=plt.cm.gray)
        # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        # for n, contour in enumerate(contours):
        #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        # plt.show()
    return to_return


def segmentation_watershed(image):
    # Image Segmentation with Watershed Algorithm
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
    from scipy import ndimage
    # from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    from skimage.measure import find_contours, label
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # noise removal
    thresh = apply_otsu(image)[0]

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 1.0 * dist_transform.max(), 255, 0)

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
    # localMax = peak_local_max(D, indices=False, min_distance=20,
    #                           labels=thresh)

    labels = watershed(-D, markers, mask=thresh)

    # loop over the unique labels returned by the Watershed
    # algorithm
    fig, ax = plt.subplots()
    contours = find_contours(labels, 1)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()


def apply_otsu(img):
    import cv2
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th, ret


def get_local_otsu(img):
    from skimage.filters import rank
    from skimage.morphology import disk
    radius = 5
    selem = disk(radius)
    local_otsu = rank.otsu(img, selem)
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    # ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img > local_otsu

if __name__ == '__main__':
    from get_data import get_features
    types_channel = ['infrared', 'visible']
    compute_indexes = True
    chan = 1
    channel_number = 0
    type_channels = types_channel[channel_number]
    dfb_beginning = 13517
    nb_days = 3
    dfb_ending = dfb_beginning + nb_days - 1
    latitude_beginning = 40.
    latitude_end = 45.
    longitude_beginning = 125.
    longitude_end = 130.
    date_begin, date_end = print_date_from_dfb(dfb_beginning, dfb_ending)
    lat, lon = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    # from quick_visualization import get_bbox
    # bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    features = get_features(type_channels, lat, lon, dfb_beginning, dfb_ending, compute_indexes, slot_step=1,
                            normalize=True)

    segmentation_watershed(features[16, :, :, chan])

    segmentation(features, chan)
