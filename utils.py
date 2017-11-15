def get_nb_slots_by_day(time_step):
    return 24*60/time_step


def looks_like_night(point, me=None, std=None):
    ## wait for array such as [0,17,-25,3]
    if me is None and std is None:
        for k in range(len(point)-1):
            if abs(point[k]) > 0.001:
                return False
        return True
    elif me is not None and std is not None:
        for k in range(len(point)-1):
            m = me[k]
            s = std[k]
            if abs(point[k]*s + m) > 0.001:
                return False
        return True
    elif me is None:
        raise AttributeError('standard deviation is known but not mean')
    elif std is None:
        raise AttributeError('mean is known but not standard deviation')