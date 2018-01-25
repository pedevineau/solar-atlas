'''
Created on Feb 21, 2016

@author: tomas
'''

import numpy
from general_utils import daytimeconv

#sun - sat angle correction for backscattering (for VIS)
def backscater_cor_for_npix(angle):
    res=numpy.ones_like(angle)
    A=16.2
    B=8.5
    C=numpy.radians(40.)
    wh=angle<C
    D = 1. + (numpy.cos(C)**A)/B
    res[wh]= D - ((numpy.cos(angle[wh])**A)/B)
    
    # force correction for very small angles
    C = numpy.radians(15.)
    wh=angle<C
    res[wh] = res[wh] - 0.30*((C-angle[wh]))
    
    # force correction for very small angles
    C = numpy.radians(5.)
    wh=angle<C
    res[wh] = res[wh] - 0.45*((C-angle[wh]))

    C = numpy.radians(3.5)
    wh=angle<C
    res[wh] = res[wh] - 0.45*((C-angle[wh]))
    
    res2=numpy.ones_like(angle)
    A=3.
    B=10.
    C=numpy.radians(60.)
    wh=angle<C
    D = 1. + (numpy.cos(C)**A)/B
    res2[wh]= D - ((numpy.cos(angle[wh])**A)/B)

    return(res*res2)


def forescater_cor_for_npix(angle):
    res=numpy.ones_like(angle)
    A=1.6 #v4
    B=1.1
    C=numpy.radians(90.)
    
    wh=angle>C
    res[wh] = 1. + numpy.power(numpy.abs(numpy.cos(C)),A)/B
    
    res[wh] -= numpy.power(numpy.abs(numpy.cos(angle[wh])),A)/B
    
    return(res)

def backscatter_forescatter_cor_for_npix(angle):
    back_cor = backscater_cor_for_npix(angle)
    fore_cor = forescater_cor_for_npix(angle)

    #backscatter for HIMAWARI8
#    bcor_offset = -((1-back_cor) * 0.10)
#    bcor_rescale = 1 + ((1-back_cor) * 0.15)
    #himawari data are corrected, only negligible correction is needed,if any
    bcor_offset = -((1-back_cor) * 0.05)
    bcor_rescale = 1 + ((1-back_cor) * 0.075)
    

    #forescatter for HIMAWARI8
    fcor_offset = fore_cor * 0
    fcor_rescale = 1+((fore_cor-1)*0.55)
    
    
    cor_offset=fcor_offset  + bcor_offset
    cor_rescale=fcor_rescale * bcor_rescale

    return cor_offset, cor_rescale


    # correction for mirrorong effect - sun ground satellite - in one line hsat and hun equal
    #index close to 1 stands for high mirroring potential
def mirror_cor_for_npix(mirror_index):
    threshold=0.82
    rescale=0.08
    aux=(mirror_index-threshold)/(1.-threshold)
    aux[aux<0]=0.
    
    mcor_rescale = 1 + ((aux*rescale))
    mcor_offset = -((aux*rescale)*0.25)
    
    return mcor_offset, mcor_rescale


def h0_cor_for_npix(angle):
    cor=numpy.ones_like(angle)
    A=0.1
    
    wh=(angle>A) | (angle < 0.0)
    cor[wh] = 0. 
    wh=(angle<=A) & (angle >= 0.0)
    aux = (A-angle[wh])*10
    aux[aux>1.]=1.0
    aux[aux<0.]=0.0
    
    cor[wh] = aux

    h0cor_rescale = (cor*0.4) + 1.
    h0cor_offset = cor * 0.075 

#    import pylab
#    pylab.plot(angle.flatten(),cor.flatten(),'ro', ms=1, markeredgewidth=0)
#    pylab.show()
    
    return h0cor_offset, h0cor_rescale

def plot_backforescatter_scatterplot(solarGeom_TimesDict, CalibratedPostLaunchSatDataDict):
    from pylab import plot, show, title, subplot, xlabel, ylabel #@UnresolvedImport
    xh = solarGeom_TimesDict['h0_ref']
    xs= solarGeom_TimesDict['sun_sat_angle']
    xm= solarGeom_TimesDict['sun_sat_mirror']

    OutN = (CalibratedPostLaunchSatDataDict['NORPIXraw'])
    OutNc = (CalibratedPostLaunchSatDataDict['NORPIX'])
    OutNcbf = (CalibratedPostLaunchSatDataDict['NORPIXcor_bf'])
    OutNcm = (CalibratedPostLaunchSatDataDict['NORPIXcor_m'])
    OutNch = (CalibratedPostLaunchSatDataDict['NORPIXcor_h0'])
    
    subplot(2,3,1)
    plot(xh.flatten(), OutNcbf.flatten(),  'bo', ms=1, markeredgewidth=0,label='norpix_bf_cor')
    plot(xh.flatten(), OutN.flatten(),  'ro', ms=1, markeredgewidth=0,label='norpix')
    xlabel('h0')
    ylabel('norpix_bf_cor,norpix')
    title("backascatter and forescatter correction")

    subplot(2,3,2)
    plot(xs.flatten(), OutNcbf.flatten(),  'bo', ms=1, markeredgewidth=0,label='norpix_bf_cor')
    plot(xs.flatten(), OutN.flatten(),  'ro', ms=1, markeredgewidth=0,label='norpix')
    xlabel('sun sat angle')
    ylabel('norpix_bf_cor,norpix')
    title("backascatter and forescatter correction")

    subplot(2,3,3)
    plot(xm.flatten(), OutNcm.flatten(),  'bo', ms=1, markeredgewidth=0,label='norpix_m_cor')
    plot(xm.flatten(), OutN.flatten(),  'ro', ms=1, markeredgewidth=0,label='norpix')
    xlabel('sun_sat_mirror')
    ylabel('norpix_m_cor,norpix')
    title("mirror correction")

    subplot(2,3,4)
    plot(xh.flatten(), OutNch.flatten(),  'bo', ms=1, markeredgewidth=0,label='norpix_h_cor')
    plot(xh.flatten(), OutN.flatten(),  'ro', ms=1, markeredgewidth=0,label='norpix')
    xlabel('h0')
    ylabel('norpix_h_cor,norpix')
    title("h0 correction")

    
    subplot(2,3,5)
    plot(xs.flatten(), OutNc.flatten(),  'bo', ms=1, markeredgewidth=0,label='norpix_all_cor')
    plot(xs.flatten(), OutN.flatten(),  'ro', ms=1, markeredgewidth=0,label='norpix')
    xlabel('sun sat angle')
    ylabel('norpix_all_cor,norpix')
    title("all corrections")


    show()

#cloud index (Derrien, Gleau 2005 in Durr, Zelenka, in press) - for sinh0 input
def himawari_CLI2(BT_ir_039, BT_ir_108, sinh0):
    sinh0_2=sinh0.copy()
    sinh0_2[sinh0_2 < 0.087155] = 0.087155
    cli=(BT_ir_039-BT_ir_108)/sinh0_2
    cli[cli<-10]=-10
    cli[cli>200]=200
    return (cli)


#normalized difference snow index (Durr, Zelenka)
def himawari_NDSI(vis006, ir_016):
    ndsi = numpy.empty(vis006.shape,dtype=numpy.float64)
    ndsi[:] = numpy.nan
    aux = vis006 + ir_016
    wh = (numpy.absolute(aux) > 0.01) & (vis006==vis006) & (ir_016==ir_016)
    
    ndsi[wh]=(vis006[wh]-ir_016[wh])/(aux[wh])
    ndsi[ndsi>1.] = 1.0
    ndsi[ndsi<-1.] = -1.0
    
    return (ndsi)



def alt_snow_probab(doy, z, lat=45.):
    #process list
    day_angle=(2.*numpy.pi*doy/366) - (numpy.pi*0.14)
    if lat<0:
        day_angle+=numpy.pi
    lat0=47.
    latcor_range=0.6
    latcor1=(abs(lat)-lat0)/90.*latcor_range
    latcor2=(1.+latcor1)
#    a,b,c,d,e,f = 1.5,5000.,1.9,1.3,3000.,1.7 
    a,b,c,d,e,f = 1.6,4000.,1.95,1.4,2500.,1.7 
    res=latcor1+latcor2*(a+(numpy.cos(day_angle)*numpy.exp(-z/b)/c)-d*numpy.exp(-z/e))/f
    res=max(0.01,min(0.97,res))
    return res


def alt_snow_probab_for_doys(alt, latit):
    s_probab=numpy.empty((367), dtype=numpy.float32)
    for doy in range(0,366+1):
        s_prob=alt_snow_probab(doy, alt, latit)
        s_probab[doy]=(s_prob)
    return (s_probab)


def alt_snow_probab_for_dfbs(alt, lat,dfbStart, dfbEnd):
    s_probab = numpy.zeros((dfbEnd-dfbStart+1))
    for dfb in range(dfbStart,dfbEnd+1):
        s_probab[dfb-dfbStart]=alt_snow_probab(daytimeconv.dfb2doy(dfb), alt, lat) 
    return s_probab
