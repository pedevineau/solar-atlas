def median_filter_3d(array, scope=5):
    from scipy import ndimage
    if scope == 1:
        return array
    for k in range(len(array)):
        array[k] = ndimage.median_filter(array[k], scope)
    return array

# low pass spatial filter case use: NOT ndsi, perhaps CLI or stressed NDSI
def low_pass_filter_3d(array, cutoff, omega=0):
    from numpy import shape, empty, abs, sin, pi, gradient, square
    from scipy import fftpack
    (a, b, c) = shape(array)
    filt = empty((b, c))
    filtered_spatial = empty((a, b, c))
    # grs = empty((a, b, c, 4))
    # grsq = empty((a,b,c,2))
    for k in range(len(array)):
        fft2 = fftpack.fft2(array[k, :, :])
        filt[abs(fft2) < cutoff - omega] = 1
        filt[abs(fft2) > cutoff + omega] = 0
        mask = (cutoff - omega < abs(fft2)) & (abs(fft2) < cutoff + omega)
        filt[mask] = 0.5 * (1 - sin(pi * (abs(fft2[mask]) - cutoff) / (2 * omega)))
        fft2 = fft2 * filt
        filtered_spatial[k] = fftpack.ifft2(fft2)
    return filtered_spatial
        # g = gradient(filtered_spatial[k])
        # grs[k, :, :, 0] = g[0]  # gradient
        # grs[k, :, :, 1] = g[1]  # gradient
        # grsq[k, :, :, 0] = square(grs[k, :, :, 0]) + square(grs[k, :, :, 1])


def digital_low_cut_filtering_time(array, mask, satellite_step):
    from utils import get_nb_slots_per_day
    # the slot step does not matter here
    fs = 0.5*get_nb_slots_per_day(satellite_step, 1)
    cutoff = 20./(fs*1)
    from scipy import signal
    b, a = signal.butter(8, cutoff, 'high', analog=False, output='ba')
    X1 = signal.lfilter(b, a, array, axis=0)
    X1[mask] = 0
    return X1
    #
    #
    # # cutoff : day-time
    # from scipy import fftpack
    # fs = 0.5*get_nb_slots_per_day(satellite_timestep)
    # cutoff = 7./fs
    # freq = fftpack.rfftfreq(array.shape[0])
    # # print cutoff
    # print freq
    # X = fftpack.rfft(array, axis=0)
    # X[abs(freq) < cutoff] = 0
    # y = fftpack.irfft(X)
    # y[mask] = 0
    # return y
    #



# unused. relevant??
def time_smoothing(array_3D_to_smoothen):
    if smoothing:
        import time
        from numpy import shape, empty, mean
        time_start_smoothing = time.time()
        shape = shape(array_3D_to_smoothen)
        array = empty(shape)
        for k in range(nb_neighbours_smoothing, shape[0]-nb_neighbours_smoothing):
            array[k] = mean(array_3D_to_smoothen[k-nb_neighbours_smoothing:k+nb_neighbours_smoothing+1])
        time_stop_smoothing = time.time()
        print 'time smoothing', str(time_stop_smoothing-time_start_smoothing)
        return array / (1 + 2 * nb_neighbours_smoothing)
    else:
        return array_3D_to_smoothen


if __name__ == '__main__':
    frequency_low_cut = 0.03
    frequency_high_cut = 0.2
    nb_neighbours_smoothing = 0  # number of neighbours used in right and left to smoothe
    smoothing = nb_neighbours_smoothing > 0