from utils import *
from scipy.ndimage.filters import gaussian_filter1d


def recognize_pattern_ndsi(ndsi, mu, mask, nb_slots_per_day, slices_per_day=1, tolerance=0.0, persistence_sigma=0.):
    print 'begin recognize pattern'
    from time import time
    t_begin_reco = time()
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
    from scipy.stats import pearsonr, linregress

    # computing of correlation need enough temporal information. If we have data on a too small window, ignore it
    minimal_nb_unmasked_slots = 12


    # we'll return a "doped" ndsi. cloudy looks like -1, snowy looks like 1, and other situations are not changed
    # classes: 1 for snow, -1 for clouds, 0 otherwise
    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(ndsi)
    nb_slots_per_step = int(nb_slots_per_day / slices_per_day)

    nb_steps = int(np.ceil(nb_slots/nb_slots_per_step)) + 1 # +1 because first slot is not the darkest slot for every point

    map_first_darkest_points = get_map_next_midnight_slots(mu, nb_slots_per_day, current_slot=0)

    stressed_ndsi = np.zeros_like(ndsi)

    persistence = persistence_sigma > 0

    if persistence:
        persistence_array = np.zeros((nb_steps, nb_latitudes, nb_longitudes), dtype=float)
        # complete persistence array
        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slot_beginning_slice = 0
                slot_ending_slice = map_first_darkest_points[lat, lon] % nb_slots_per_step
                med = np.median(ndsi[:, lat, lon][~mask[:, lat, lon]])
                step = 0
                persistence_mask_1d = np.ones(nb_steps, dtype=bool)
                while slot_beginning_slice < nb_slots:
                    slice_ndsi = ndsi[slot_beginning_slice:slot_ending_slice, lat, lon]
                    slice_mu = mu[slot_beginning_slice:slot_ending_slice, lat, lon]
                    slice_mask = mask[slot_beginning_slice:slot_ending_slice, lat, lon]

                    if slice_ndsi[~slice_mask].size > minimal_nb_unmasked_slots:
                        persistence_mask_1d[step] = False
                        if True:
                            slope, intercept, r_value, p_value, std_err = linregress(
                                slice_mu[~slice_mask],
                                slice_ndsi[~slice_mask],
                            )
                            if r_value > 1 - tolerance:
                                persistence_array[step, lat, lon] = 1 # maximum(med+0.4, 1.) # med of 0.6 is considered as snow-like with p=1
                    step += 1
                    slot_beginning_slice = slot_ending_slice
                    slot_ending_slice += nb_slots_per_step
                persistence_array_1d = persistence_array[:, lat, lon]
                persistence_array[:, lat, lon][~persistence_mask_1d] = \
                    apply_gaussian_persistence(persistence_array_1d, persistence_mask_1d,
                                               persistence_sigma, persistence_scope=slices_per_day)

        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slot_beginning_slice = 0
                slot_ending_slice = map_first_darkest_points[lat, lon] % nb_slots_per_step
                step = 0
                while slot_beginning_slice < nb_slots:
                    slice_ndsi = ndsi[slot_beginning_slice:slot_ending_slice, lat, lon]
                    slice_mask = mask[slot_beginning_slice:slot_ending_slice, lat, lon]
                    if slice_ndsi[~slice_mask].size > minimal_nb_unmasked_slots:
                        stressed_ndsi[slot_beginning_slice: slot_ending_slice, lat, lon][~slice_mask] =\
                            persistence_array[step, lat, lon]
                    step += 1
                    slot_beginning_slice = slot_ending_slice
                    slot_ending_slice += nb_slots_per_step

    else:
        for lat in range(nb_latitudes):
            for lon in range(nb_longitudes):
                slot_beginning_slice = 0
                slot_ending_slice = map_first_darkest_points[lat, lon] % nb_slots_per_step
                # last_slope, last_intercept = 0, 0
                med = np.median(ndsi[:, lat, lon][~mask[:, lat, lon]])
                while slot_beginning_slice < nb_slots:
                    slice_ndsi = ndsi[slot_beginning_slice:slot_ending_slice, lat, lon]
                    slice_mu = mu[slot_beginning_slice:slot_ending_slice, lat, lon]
                    slice_mask = mask[slot_beginning_slice:slot_ending_slice, lat, lon]

                    # print 'ndsi and then mu'
                    # slice_ndsi, m_ndsi, s_ndsi = normalize_array(slice_ndsi, slice_mask, 'center')
                    # slice_mu, m_mu, s_mu = normalize_array(slice_mu, slice_mask, 'center')

                    if slice_ndsi[~slice_mask].size > minimal_nb_unmasked_slots:
                        slope, intercept, r_value, p_value, std_err = linregress(
                            slice_mu[~slice_mask],
                            slice_ndsi[~slice_mask],
                        )
                        if r_value > 1 - tolerance:
                            # print med
                            ### NB maths: there is not optimal offset beta so that the
                            # stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon][~slice_mask] = \
                            #     m_ndsi * (slice_ndsi[~slice_mask]+0) / (slice_mu[~slice_mask]+0)

                            if True:
                                # # TODO: the following information should be used (later)
                                # stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon][~slice_mask] = \
                                #   med + (slice_ndsi[~slice_mask] - slope * slice_mu[~slice_mask] - intercept)
                                stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon][~slice_mask] = 1
                                # stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon][~slice_mask] = \
                                # maximum(med+0.4,1)
                                    # 1+slice_ndsi[~slice_mask]-(slope*slice_mu[~slice_mask]+intercept)


                        # elif -r_value > 1 - 4*tolerance:
                        #     # print med
                        #     # print 'anticorrelated'
                        #     # if med < 0.3:
                        #     stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon][~slice_mask] = -1
                        # #     stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon] = slice_ndsi

                    slot_beginning_slice = slot_ending_slice
                    slot_ending_slice += nb_slots_per_step

    print 'time recognition', time() - t_begin_reco
    return stressed_ndsi[:,:,:]


def apply_gaussian_persistence(persistence_array_1d, persistence_mask_1d, persistence_sigma, persistence_scope):
    persistence_sigma = float(persistence_sigma)
    trunc = persistence_scope/persistence_sigma
    return normalize_array(gaussian_filter1d(persistence_array_1d[~persistence_mask_1d],
                             sigma=persistence_sigma, axis=0, truncate=trunc),
                           normalization='maximum', return_m_s=False)


def recognize_pattern_vis(ndsi, vis, nir, mu, mask, time_step_satellite, slot_step, slices_by_day=1, tolerance=0.15):
    print 'begin recognize pattern'
    from time import time
    t_begin_reco = time()
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
    from scipy.stats import pearsonr

    # computing of correlation need enough temporal information. If we have data on a too small window, ignore it
    minimal_nb_unmasked_slots = 12

    (nb_slots, nb_latitudes, nb_longitudes) = np.shape(ndsi)
    nb_slots_per_day = get_nb_slots_per_day(time_step_satellite, slot_step)
    nb_slots_per_step = int(nb_slots_per_day / slices_by_day)

    map_first_darkest_points = get_map_next_midnight_slots(mu, nb_slots_per_day, current_slot=0)

    stressed_ndsi = ndsi

    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            slot_beginning_slice = 0
            slot_ending_slice = map_first_darkest_points[lat, lon] % nb_slots_per_step

            while slot_beginning_slice < nb_slots:
                slice_vis = vis[slot_beginning_slice:slot_ending_slice, lat, lon]
                slice_nir = nir[slot_beginning_slice:slot_ending_slice, lat, lon]
                slice_ndsi = ndsi[slot_beginning_slice:slot_ending_slice, lat, lon]
                slice_mu = mu[slot_beginning_slice:slot_ending_slice, lat, lon]
                slice_mask = mask[slot_beginning_slice:slot_ending_slice, lat, lon]
                if slice_ndsi[~slice_mask].size > minimal_nb_unmasked_slots:
                    p_vis, r_vis = pearsonr(  # is expected to be correlated
                        slice_vis[~slice_mask],
                        slice_mu[~slice_mask]
                    )
                    p_nir, r_nir = pearsonr(   # is expected to be anti-correlated
                        slice_nir[~slice_mask],
                        slice_mu[~slice_mask]
                    )
                    if p_nir < 4*tolerance: # not correlated with mu
                        print p_vis, p_nir
                        # visualize_input(slice_vis[~slice_mask], display_now=False)
                        # visualize_input(slice_nir[~slice_mask])
                        if p_vis > 1 - tolerance:
                            print 'suspect snow ?', lat, lon
                            stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon] = \
                                slice_ndsi / slice_mu
                slot_beginning_slice = slot_ending_slice
                slot_ending_slice += nb_slots_per_step

    print 'time recognition ir', time() - t_begin_reco
    return stressed_ndsi[:, :, :]
