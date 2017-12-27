from utils import *


def get_contours(features):
    # from scipy.ndimage import sobel, prewitt
    (slots, lats, lons) = np.shape(features)[0:3]
    from numpy.random import randint
    from skimage import measure
    from skimage.exposure import rescale_intensity
    # features = rescale_intensity(features))
    array_contours = []
    for slot in range(slots):
        # Find contours at a constant value of 0.8
        import matplotlib.pyplot as plt
        img = get_otsu(features[slot, :, :, 0])
        contours = measure.find_contours(img, 0.8)
        array_contours.append(contours)
        fig, ax = plt.subplots()
        ax.imshow(features[slot,:,:,0], interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()
    return array_contours


def get_otsu(img):
    import cv2
    print np.shape(img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


if __name__ == '__main__':
    from get_data import get_features
    types_channel = ['infrared', 'visible']
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
    features = get_features(type_channels, lat, lon, dfb_beginning, dfb_ending, True, slot_step=1,
                            normalize=True)
    get_contours(features)
