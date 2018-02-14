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


def infrared_low_cloud_composite(latitudes, longitudes, dfb_beginning, dfb_ending, slot_step):
    from get_data import get_features
    from numpy import round
    infrared_features = get_features('infrared', latitudes, longitudes, dfb_beginning, dfb_ending, 'channel', slot_step)
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(infrared_features)[0:3]
    composite = np.zeros((nb_slots, nb_latitudes, nb_longitudes, 3))
    try:
        fir = infrared_features[:, :, :, 2]
        fir = round(2 * (fir - 200))
        composite[:, :, :, 0] = fir
    except IndexError:
        print 'Computing '
        pass
    lir = infrared_features[:, :, :, 1]
    mir = infrared_features[:, :, :, 0]
    mir = round(2*(mir - 200))
    lir = round(2*(lir - 200))
    composite[:, :, :, 1] = lir
    composite[:, :, :, 2] = mir
    composite[(lir<=0) | (lir>250)]=0
    return 255*composite


if __name__ == '__main__':
    from utils import typical_input
    slot_step = 1
    beginning, ending, latitude_beginning, latitude_end, longitude_beginning, longitude_end = typical_input()
    latitudes, longitudes = get_latitudes_longitudes(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    date_begin, date_end = print_date_from_dfb(beginning, ending)
    composite = infrared_low_cloud_composite(latitudes, longitudes, beginning, ending, slot_step)

    from quick_visualization import get_bbox
    bbox = get_bbox(latitude_beginning, latitude_end, longitude_beginning, longitude_end)
    from matplotlib.pyplot import imshow, show
    start = 70
    for slot in range(start, np.shape(composite)[0]):
        if not np.all(composite[slot, :, :, 0] == 0):
            print slot_step*slot
            imshow(composite[slot])
            show()
