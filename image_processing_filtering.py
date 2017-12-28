from utils import *


def segmentation(features, chan):
    (slots, lats, lons) = np.shape(features)[0:3]
    from skimage import measure
    # from skimage.exposure import rescale_intensity
    # features = rescale_intensity(features))
    array_contours = []
    for slot in range(slots):
        # Find contours at a constant value of 0.8
        import matplotlib.pyplot as plt
        img = get_otsu(features[slot, :, :, chan])
        contours = measure.find_contours(img, 0.8)
        array_contours.append(contours)
        fig, ax = plt.subplots()
        ax.imshow(features[slot,:,:, chan], interpolation='nearest', cmap=plt.cm.gray)
        # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()
    return array_contours


def get_otsu(img):
    import cv2
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
    segmentation(features, chan)
