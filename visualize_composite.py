from utils import *


def get_snow_composite(latitudes, longitudes, dfb_beginning, dfb_ending, slot_step):
    from get_data import get_features
    visible_features = get_features('visible', latitudes, longitudes, dfb_beginning, dfb_ending, False, slot_step)
    vis = visible_features[:, :, :, 1]
    nir = visible_features[:, :, :, 0]
    mir = get_features('infrared', latitudes, longitudes, dfb_beginning, dfb_ending, False, slot_step)[:, :, :, 1]
    mir-=250
    mir/=100
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(vis)
    vis[vis < 0] = 0
    nir[nir < 0] = 0
    mir[mir < 0] = 0
    composite = np.empty((nb_slots, nb_latitudes, nb_longitudes, 3))
    composite[:, :, :, 0] = vis
    composite[:, :, :, 1] = nir
    composite[:, :, :, 2] = mir
    return 255*composite


if __name__ == '__main__':
    dfb_beginning = 13548-15
    nb_days = 8
    dfb_ending = dfb_beginning + nb_days - 1
    latitude_beginning = 40.+5
    latitude_end = 45.+5
    longitude_beginning = 125.
    longitude_end = 130.
    slot_step = 5
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    date_begin, date_end = print_date_from_dfb(dfb_beginning, dfb_ending)
    composite = get_snow_composite(latitudes, longitudes, dfb_beginning, dfb_ending, slot_step)

    from quick_visualization import get_bbox
    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    from matplotlib.pyplot import imshow, show
    for slot in range(np.shape(composite)[0]):
        if not np.all(composite[slot, :, :, 0] == 0):
            print slot_step*slot
            imshow(composite[slot])
            show()
