from utils import *
from quick_visualization import visualize_input

# def estimate_gaussian_from_samples(inp):
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import LogNorm
#     from sklearn import mixture
#     X_train = inp.reshape(-1, 1)
#     z = np.linspace(-1,1, np.shape(X_train)[0])
#
#     clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
#     clf.fit(X_train)
#     print clf.means_
#     print clf.covariances_
#
#     # display predicted scores by the model as a contour plot
#     x = np.linspace(-1., 1.,300)
#     y_pred=clf.predict(x.reshape((-1,1)))
#     # y = np.linspace(-20., 40.)
#     # X, Y = np.meshgrid(x, y)
#     # XX = np.array([X.ravel(), Y.ravel()]).T
#     # Z = -clf.score_samples(XX)
#     # Z = Z.reshape(X.shape)
#     #
#     # CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
#     #                  levels=np.logspace(0, 3, 10))
#     # CB = plt.colorbar(CS, shrink=0.8, extend='both')
#     plt.plot(z, inp, 'b*')
#     plt.plot(x, y_pred, 'g^')
#
#     plt.title('Negative log-likelihood predicted by a GMM')
#     plt.axis('tight')
#     plt.show()


def recognize_pattern_ndsi(ndsi, mu, mask, time_step_satellite, slices_per_day=1, tolerance=0.1):
    print 'begin recognize pattern'
    from get_data import normalize_array
    from time import time
    t_begin_reco = time()
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
    from scipy.stats import pearsonr
    from numpy import percentile, shape

    # computing of correlation need enough temporal information. If we have data on a too small window, ignore it
    minimal_nb_unmasked_slots = 12


    # we'll return a "doped" ndsi. cloudy looks like -1, snowy looks like 1, and other situations are not changed
    # classes: 1 for snow, -1 for clouds, 0 otherwise
    (nb_slots, nb_latitudes, nb_longitudes) = shape(ndsi)
    nb_slots_per_day = get_nb_slots_per_day(time_step_satellite)
    nb_slots_per_step = int(nb_slots_per_day / slices_per_day)

    map_first_darkest_points = get_map_next_darkest_slot(mu, nb_slots_per_day, current_slot=0)

    ndsi, m_ndsi, s_ndsi = normalize_array(ndsi, mask, 'standard')
    mu, m_mu, s_mu = normalize_array(mu, mask, 'standard')

    stressed_ndsi = ndsi

    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            slot_beginning_slice = 0
            slot_ending_slice = map_first_darkest_points[lat, lon] % nb_slots_per_step
            while slot_beginning_slice < nb_slots:
                slice_ndsi = ndsi[slot_beginning_slice:slot_ending_slice, lat, lon]
                slice_mu = mu[slot_beginning_slice:slot_ending_slice, lat, lon]
                slice_mask = mask[slot_beginning_slice:slot_ending_slice, lat, lon]
                if slice_ndsi[~slice_mask].size > minimal_nb_unmasked_slots:
                    p_snow, r_snow = pearsonr(
                        slice_ndsi[~slice_mask],
                        slice_mu[~slice_mask]
                    )
                    if -p_snow > 1 - tolerance:  # if it is anti-correlated with cos zen
                        # percentile computing enables to find the summit of the convex function
                        low_summit_clouds = percentile(slice_ndsi[~slice_mask], 5)
                        # stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon, 1] = -1  # -1 is cloud
                        stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon][~slice_mask] = \
                            - (slice_ndsi[~slice_mask]) / (1+low_summit_clouds - slice_mu[~slice_mask])
                        # the previous +2 and +3 (=+2+1) serves numerical stability
                    elif p_snow > 1 - tolerance:
                        # print r_snow
                        # stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon, 1] = 1  # 1 is snow
                        stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon][~slice_mask] = \
                            slice_ndsi[~slice_mask] / slice_mu[~slice_mask]
                    # already computed before the loop
                    # else:
                    #     stressed_ndsi[slot_beginning_slice:slot_ending_slice, lat, lon] = slice_ndsi

                slot_beginning_slice = slot_ending_slice
                slot_ending_slice += nb_slots_per_step

    print 'time recognition', time() - t_begin_reco
    return stressed_ndsi[:,:,:]


def recognize_pattern_vis(ndsi, vis, nir, mu, mask, time_step_satellite, slices_by_day=1, tolerance=0.15):
    print 'begin recognize pattern'
    from time import time
    t_begin_reco = time()
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
    from scipy.stats import pearsonr
    from numpy import percentile, shape

    # computing of correlation need enough temporal information. If we have data on a too small window, ignore it
    minimal_nb_unmasked_slots = 12

    (nb_slots, nb_latitudes, nb_longitudes) = shape(ndsi)
    nb_slots_per_day = get_nb_slots_per_day(time_step_satellite)
    nb_slots_per_step = int(nb_slots_per_day / slices_by_day)

    map_first_darkest_points = get_map_next_darkest_slot(mu, nb_slots_per_day, current_slot=0)

    stressed_ndsi = ndsi

    for lat in range(nb_latitudes):
        for lon in range(nb_longitudes):
            slot_beginning_slice = 0
            slot_enging_slice = map_first_darkest_points[lat, lon] % nb_slots_per_step

            while slot_beginning_slice < nb_slots:
                slice_vis = vis[slot_beginning_slice:slot_enging_slice, lat, lon]
                slice_nir = nir[slot_beginning_slice:slot_enging_slice, lat, lon]
                slice_ndsi = ndsi[slot_beginning_slice:slot_enging_slice, lat, lon]
                slice_mu = mu[slot_beginning_slice:slot_enging_slice, lat, lon]
                slice_mask = mask[slot_beginning_slice:slot_enging_slice, lat, lon]
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
                            stressed_ndsi[slot_beginning_slice:slot_enging_slice, lat, lon] = \
                                slice_ndsi / slice_mu
                slot_beginning_slice = slot_enging_slice
                slot_enging_slice += nb_slots_per_step

    print 'time recognition ir', time() - t_begin_reco
    return stressed_ndsi[:, :, :]
