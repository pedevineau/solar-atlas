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
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    import numpy as np
    import argparse
    import cv2

    # construct the argument parse and parse the arguments
    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    # loop over the unique labels returned by the Watershed
    # algorithm
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

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
