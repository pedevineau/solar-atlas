
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