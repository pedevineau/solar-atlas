from utils import *


def recognize_edge():
    '''
    Could be relevant over snow snow
    :return:
    '''
    return


def recognize_cloud_shade(vis, real_cloud_mask, cos_zen, th=0.2):
    from read_metadata import read_satellite_step
    nb_slots_per_day = get_nb_slots_per_day(read_satellite_step(), slot_step=1)
    (nb_slots, nb_lats, nb_lons) = np.shape(vis)[0:3]
    nb_days = nb_slots / nb_slots_per_day
    detected_cloud_shade = np.zeros_like(vis, dtype=bool)
    for slot in range(nb_slots_per_day):
        for lat in range(nb_lats):
            for lon in range(nb_lons):
                l = []
                for day in range(nb_days):
                    s = day*nb_slots_per_day+slot
                    if not real_cloud_mask[s, lat, lon]:
                        l.append(vis[s, lat, lon])
                if len(l) > 0:
                    supposed_albedo = np.median(l)
                    for day in range(nb_days):
                        s = day*nb_slots_per_day+slot
                        if not real_cloud_mask[s, lat, lon] and (supposed_albedo - vis[s, lat, lon]) \
                                > th*cos_zen[s, lat, lon]: #not real_cloud_mask[s, lat, lon]:
                            detected_cloud_shade[s, lat, lon] = True
    return detected_cloud_shade
