from utils import np, get_nb_slots_per_day


def median_filter_3d(array, scope=5):
    from scipy import ndimage
    if scope <= 1:
        return array
    for k in range(len(array)):
        array[k] = ndimage.median_filter(array[k], scope)
    return array


def contrast_filter(array):
    from scipy import signal
    s = np.shape(array)
    contrast = 0.125 * np.array([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]])
    if len(s) == 2:
        return signal.convolve2d(array, contrast, boundary='symm', mode='same')
    if len(s) == 3:
        for k in range(s[0]):
            array[k] = signal.convolve2d(array[k], contrast, boundary='symm', mode='same')
        return array


def local_std(array, mask=None, scope=3):
    from scipy import ndimage
    if scope <= 1:
        return array
    from numpy import empty_like
    arr = empty_like(array)
    for k in range(len(array)):
        arr[k] = np.sqrt(ndimage.generic_filter(array[k], np.var, size=scope))
        if mask is not None:
            mask[k] = ndimage.morphology.binary_dilation(mask[k])
            arr[k][mask[k]] = -10
    return arr


def local_max(array, mask=None, scope=3):
    from scipy import ndimage
    if scope <= 1:
        return array
    from numpy import empty_like
    arr = empty_like(array)
    for k in range(len(array)):
        arr[k] = ndimage.maximum_filter(array[k], size=scope)
        if mask is not None:
            arr[k][ndimage.morphology.binary_dilation(mask[k])] = -10
    return arr


# low pass spatial filter case use: NOT ndsi, perhaps CLI or stressed NDSI
def low_pass_filter_3d(array, cutoff, omega=0):
    from scipy import fftpack
    (a, b, c) = np.shape(array)
    filt = np.empty((b, c))
    filtered_spatial = np.empty((a, b, c))
    # grs = empty((a, b, c, 4))
    # grsq = empty((a,b,c,2))
    for k in range(len(array)):
        fft2 = fftpack.fft2(array[k, :, :])
        filt[abs(fft2) < cutoff - omega] = 1
        filt[abs(fft2) > cutoff + omega] = 0
        mask = (cutoff - omega < abs(fft2)) & (abs(fft2) < cutoff + omega)
        filt[mask] = 0.5 * (1 - np.sin(np.pi * (abs(fft2[mask]) - cutoff) / (2 * omega)))
        fft2 = fft2 * filt
        filtered_spatial[k] = fftpack.ifft2(fft2)
    return filtered_spatial
        # g = gradient(filtered_spatial[k])
        # grs[k, :, :, 0] = g[0]  # gradient
        # grs[k, :, :, 1] = g[1]  # gradient
        # grsq[k, :, :, 0] = square(grs[k, :, :, 0]) + square(grs[k, :, :, 1])


def digital_low_cut_filtering_time(array, mask, satellite_step):
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
def time_smoothing(array_3D_to_smoothen, nb_neighbours_smoothing=5):
    smoothing = nb_neighbours_smoothing > 0
    if smoothing:
        import time
        time_start_smoothing = time.time()
        shape = np.shape(array_3D_to_smoothen)
        array = np.empty(shape)
        for k in range(nb_neighbours_smoothing, shape[0]-nb_neighbours_smoothing):
            array[k] = np.mean(array_3D_to_smoothen[k-nb_neighbours_smoothing:k+nb_neighbours_smoothing+1])
        time_stop_smoothing = time.time()
        print 'time smoothing', str(time_stop_smoothing-time_start_smoothing)
        return array / (1 + 2 * nb_neighbours_smoothing)
    else:
        return array_3D_to_smoothen


if __name__ == '__main__':
    arr = np.ones((2,7,7))
    arr[:, 3,3] = 3
    print local_std(arr)