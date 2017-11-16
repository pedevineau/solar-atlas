from utils import *

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


def recognize_pattern(ndsi, mu, mask, time_step_satellite, slices_by_day=1, tolerance=0.1):
    print 'begin recognize pattern'
    from time import time
    t_begin_reco = time()
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
    from scipy.stats import pearsonr
    from numpy import percentile, shape

    # computing of correlation need enough temporal information. If we have data on a too small window, ignore it
    minimal_nb_unmasked_slots = 12


    # we'll return a "doped" ndsi. cloudy looks like -1, snowy looks like 1, and other situations are not changed
    # classes: 1 for snow, -1 for clouds, 0 otherwise
    (a, b, c) = shape(ndsi)
    to_return = ndsi
    nb_slots_by_day = get_nb_slots_by_day(time_step_satellite)
    nb_days = a/nb_slots_by_day
    nb_steps = slices_by_day * nb_days
    nb_slots_by_step = nb_slots_by_day / slices_by_day
    for k in range(nb_steps):
        for latitude in range(b):
            for longitude in range(c):
                slice_ndsi = ndsi[k*nb_slots_by_step:(k+1)*nb_slots_by_step, latitude, longitude]
                slice_mu = mu[k*nb_slots_by_step:(k+1)*nb_slots_by_step, latitude, longitude]
                slice_mask = mask[k*nb_slots_by_step:(k+1)*nb_slots_by_step, latitude, longitude]
                if slice_ndsi[~slice_mask].size > minimal_nb_unmasked_slots:
                    p_snow, r_snow = pearsonr(
                        slice_ndsi[~slice_mask],
                        slice_mu[~slice_mask]
                    )
                    if -p_snow > 1 - tolerance:  # if it is anti-correlated with cos zen
                        # percentile computing enables to find the summit of the convex function
                        low_summit_clouds = percentile(slice_ndsi[~slice_mask], 5)
                        # to_return[k*nb_slots_by_step:(k+1)*nb_slots_by_step, latitude, longitude, 1] = -1  # -1 is cloud
                        to_return[k * nb_slots_by_step:(k + 1) * nb_slots_by_step, latitude, longitude] = \
                            - (2+slice_ndsi) / (3+low_summit_clouds - slice_mu)
                        # the previous +2 and +3 (=+2+1) serves numerical stability
                    elif p_snow > 1 - 2*tolerance:
                        # print 'snow'
                        # print r_snow
                        # to_return[k*nb_slots_by_step:(k+1)*nb_slots_by_step, latitude, longitude, 1] = 1  # 1 is snow
                        to_return[k * nb_slots_by_step:(k + 1) * nb_slots_by_step, latitude, longitude] = \
                            slice_ndsi / slice_mu
                    # already computed before the loop
                    # else:
                    #     to_return[k * nb_slots_by_step:(k + 1) * nb_slots_by_step, latitude, longitude] = slice_ndsi
    print 'time recognition', time() - t_begin_reco
    return to_return[:,:,:]



