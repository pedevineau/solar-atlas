# inspiration from: http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
from numpy import sqrt

def get_dtw(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])


def get_dtw_windowed(s1, s2, w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
    return sqrt(DTW[len(s1)-1, len(s2)-1])


def get_dtw_windowed_ir(diff_ir, mu, satellite_step, slot_step):
    # assuming that (positive) lag between (fir-mir) pikes and mu pikes (=noon) is usually >2h30 and <4h30
    # PHYSICS WARNING: however, highs don't always happen during the day, nor lows at night
    w_inf = int(0 / satellite_step*slot_step)
    w_sup = int(240 / satellite_step*slot_step)

    DTW={}

    # w = max(w, abs(len(s1)-len(s2)))
    for i in range(-1,len(diff_ir)):
        for j in range(-1,len(mu)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(diff_ir)):
        for w in range(w_inf, min(len(mu)-i, w_sup)):
            dist = (diff_ir[i]-mu[w+i])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])
    return sqrt(DTW[len(diff_ir)-1, len(mu)-1])


def LB_Keogh(s1, s2, r):
    LB_sum = 0
    for ind,i in enumerate(s1):
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    return sqrt(LB_sum)


def LB_Keogh_ir(diff_ir, mu, satellite_timestep):
    r = int(240/satellite_timestep)
    return LB_Keogh(s1=diff_ir, s2=mu, r=r)


if __name__ == '__main__':
    print 'dtw'
