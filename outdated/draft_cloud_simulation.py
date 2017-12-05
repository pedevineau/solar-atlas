import numpy as np
import math

from scipy.stats import pearsonr, linregress
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html

#
import matplotlib.pyplot as plt
# from scipy import fftpack
# from scipy import signal
#

from filter import digital_low_cut_filtering_time
from get_data import normalize_array
from scipy.misc import derivative
T = 144
ignore_sun_low_angle = 7
lis = np.arange(ignore_sun_low_angle, T-ignore_sun_low_angle)




mu = np.sin(np.pi * (lis) / T)
muprim = mu - np.roll(mu, shift=-1)
x1 = mu
mustd = normalize_array(mu, normalization='standard', return_m_s=False)





r1 = 0.08*(np.random.random_sample(len(lis))-0.5)
corrs_rec = []
corrs_remaining = []
corrs_classical = []

rates = np.arange(0,2,0.1)

# percentages = np.linspace(0,1,10, endpoint=True)
percentages = [0.9]


def generate_cloud_noise(percentage_clouds, rate, offset, leng):
    noise = np.zeros(leng)
    if np.random.rand() < percentage_clouds:
        noise[0] = rate * np.random.rand()

    # inertia (arbitrary choice for now. Should be relative to percentage of clouds and to spatial coherence
    p_cc = (1.*percentage_clouds+3)/(3+1)
    p_cc = np.maximum(p_cc, 1-(1-percentage_clouds)/percentage_clouds)  # necessary condition so that p_cg < 1
    p_cg = percentage_clouds / (1-percentage_clouds) * (1-p_cc)

    # print p_cc, p_cg
    # proba of changes or not
    # if percentage_clouds == 0.5:
    #     p_cc, p_cg = 0.5, 0.5
    # else:
    #     matr = np.array([
    #         [percentage_clouds, 1-percentage_clouds],
    #         [1 - percentage_clouds, percentage_clouds]
    #     ])
    #     (p_cc, p_cg) = np.dot(np.linalg.inv(matr), np.array([percentage_clouds,percentage_clouds]))
    #     print 'coef inv',  p_cc, p_cg

    for i in range(1, leng):
        if noise[i-1] == 0:  # cloudless
            if np.random.rand() < p_cg:
               noise[i] = rate * np.random.rand()
        else:  # cloudy
            if np.random.rand() < p_cc:
                r = np.random.rand()
                noise[i] = (rate * r+offset)

    # noise[1:] = 0.5*(noise[1:]+noise[:-1])
    return noise


for montecarlo in range(5):
    off = 0.3
    r2 = generate_cloud_noise(percentage_clouds=0.8, offset=off, rate=0.2, leng=len(x1))
    for k in range(len(r2)):
        if r2[k] != 0.:
            r2[k] -= (r2[k] - off)* mu[k]
    v1 = np.var(x1)
    v2 = np.var(r1)
    v3 = np.var(r2)

    # print 'rate 1', v1/(v1+v2+v3)
    # print 'rate 2', v2/(v1+v2+v3)
    # print 'rate 3', v3/(v1+v2+v3)

    diff = x1+r1+r2

    diffstd = normalize_array(diff, normalization='standard', return_m_s=False)


    # y1 = diffstd-np. roll(diffstd-mustd, shift=-1)
    vardiffstd = diffstd-mustd-np.roll(diffstd-mustd, shift=-1)
    # x2 = np.sin(2*np.pi * (lis+10) / T)+r2
    # y2 = x2-np.roll(x2, shift=-1)
    # X = np.empty((len(lis),2))
    # Y = np.empty((len(lis),2))

    corr = pearsonr(diffstd, r2)
    # print 'corr reconstructed noise', corr

    # print 'corr remaining with mu', pearsonr(diffstd, mustd)   # huge !!!

    # print 'correlation with classical cli', pearsonr(diff/mu, r2)

    corrs_rec.append(corr)
    corrs_remaining.append(pearsonr(diffstd, mustd))
    corrs_classical.append(pearsonr(diff/mu, r2))

    covs = np.cov(diffstd, mustd)
    unbasied_coeff = covs[0,1]/np.sqrt(covs[1,1]*covs[0,0])

    slope, intercept = linregress(mustd, diffstd)[0:2]
    print slope, intercept

    plt.plot(diffstd,'g')
    plt.plot(r2,'r')
    plt.plot(normalize_array(diff/mu, normalization='standard', return_m_s=False), 'c')
    plt.plot(diffstd-unbasied_coeff*mustd, 'b')
    # plt.plot(diffstd-slope*mustd-intercept,'y')
    plt.show()


# plt.plot(percentages, corrs_rec, 'b--')


print 'pearson chaos median', np.median(corrs_rec)
print 'pearson remaining median', np.median(corrs_remaining)
print 'pearson classical median', np.median(corrs_classical)


# plt.plot(FX[:,0],'b')

# plt.plot(FY[:,0],'c')
print 'stop here', stop
plt.show()



mask=np.zeros((len(lis),2), dtype=bool)
mask[144/2-5:,0]=True
mask[144/2-10:,1]=True

X[:,0]=x1
X[:,1]=x2
X[mask]=0

Y[:,0]=vardiffstd
Y[:,1]=y2
Y[mask]=0

FX = digital_low_cut_filtering_time(array=X, mask=mask, satellite_step=10)

FY = digital_low_cut_filtering_time(array=Y, mask=mask, satellite_step=10)


plt.plot(x2,'g')
plt.plot(r2,'r')
plt.plot(y2-muprim,'y')
plt.plot(FX[:, 1], 'b')
# plt.plot(FY[:, 1], 'c')
plt.show()

# x1[x1<0.01]=0
# y1=fftpack.fft(x1)
# freq = np.fft.fftfreq(x1.shape[-1])
# cut = 2/T
# b, a = signal.butter(10, cut, 'high', analog=False, output='ba')
# print a,b
# w, h = signal.freqs(b, a)
# X1 = signal.lfilter(b, a, x1)
# y1[abs(freq)<cut]=0
# # plt.plot(freq, abs(y1),'r')
# # plt.show()
#
# x2 = fftpack.ifft(y1)
# x2[x1<0.01]=0
# r[x1<0.01]=0
#




# plt.plot(x2,'g')
# plt.plot(X1,'r')
# plt.plot(r,'b')
#
# plt.show()

from scipy.ndimage.filters import gaussian_filter1d

# import dtw_computing
#

# for j in range(100):
#     x2 = np.sin(np.roll(lis, 10) * np.pi / T) + 1. * np.random.random_sample(len(lis)) - 0.1
#     dtws = []
#     k=0
#     for supposed_lag in range(k,20):
#         dtws.append(dtw_computing.LB_Keogh(x1, np.roll(x2,-supposed_lag),r=5))
#
#     m = k+np.argmin(dtws)
#     # print dtws
#     print m
#
# # plt.plot(x1,'r')
# plt.plot(x2,'g')
# plt.plot(x2[m:],'b')
# plt.show()