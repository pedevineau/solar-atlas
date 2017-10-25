'''
Created on Sep 13, 2016

@author: petero
'''

import dateutil.parser
import numpy
import numpy as np
import solar_geom_v5
# import shading_utils
from math import pi, sin, cos, asin, acos, tan, radians, degrees, exp, pow,log


def asVect(a):
    return numpy.asarray([float(a)]);


def parseDateTime(dtIsoString):
    dt = dateutil.parser.parse(dtIsoString)
    y = dt.year    
    hh = dt.hour
    mm = dt.minute
    ss = dt.second
    
    doy = dt.timetuple().tm_yday
        
    _time = hh + (mm + (ss / 60.0)) / 60.0
    
    return [y, doy, _time];


# Sun position in relation to the inclined surface
# Version for fixed plane and 2 and 1 axis tracker with vertical rotating axis (Strategy A)  
# scalar prodact version
# Input
# sinGammaN,cosGammaN - sin and cos of the surface slope (inclination) angle
# sinAN,cosAN - sin and cos of the surface azimuth angle
# sina0,cosa0 - sin and cos of the sun azimuth angle
# sinh0,cosh0 - sin and cos of the sun height angle 
# OUtput
# cos_incidence_angle - sin (or cos) of the angle between the surface (or normal to the surface) and the direction of the sun 
def Rel_sunpos_V1(sinGammaN,cosGammaN,sinAN,cosAN,sina0,cosa0,sinh0,cosh0):
    
    x=cosAN*sinGammaN
    y=sinAN*sinGammaN
    z=cosGammaN
    
    x0=cosa0*cosh0
    y0=sina0*cosh0
    z0=sinh0
 
    cosINC=x*x0+y*y0+z*z0
    cosINC[cosINC<0]=0
    return cosINC


def mounting_geom_angles( mounting, sina0, cosa0, a0, sinh0,cosh0, h0, GN, AN, latitude, rotation_limit, rotation_limit2, backtrack, relative_spacing_rows, relative_spacing_columns):

    '''
    calculate cosine of incidence angle and module inclination angle for different mounting methods - core calculation function
    inputs pre-calculated sin's and cos's
    inputs:
        mounting - type of mounting geometry:
            1 - fixed
            5 - t1xV 1-axis tracker, vertical axis, inclined module 
            6 - t1xI 1-axis tracker NS, inclined axis
            7 - t1xH 1-axis tracker NS, horizontal axis
            8 - t1xEW 1-axis tracker, horizontal EW axis
            9 - t2x 2-axis tracker
        sinh0, cosh0 - sin, cos of sun height (radians, matrix)
        sina0, cosa0 - sin, cos of sun azimuth (radians, matrix)
        GN - module (axis) inclination(0- horizontal) (radians) 
        sinGN, cosGN - sin, cos of module (axis) inclination(0- horizontal) (radians) 
        sinAN, cosANV - like sinAN, cosAN but in vector form
        sinAN, cosAN - sin, cos of module azimuth (radians)
        lat - latitude to know if we are on Northern or Southern hemisphere
        h0 - sun height in radians
        RotRange - single value in radians, the limiting angle of horizontal or inclined tracker axis rotation
        backtrack - bool True/False condition if back tracking algorithm should be applied 
        relative_spacing_rows - the relative spacing calculated as (distance between axles)/(tracker wing width)
        relative_spacing_columns - the relative spacing calculated as (distance between axles)/(tracker wing width)
    outputs:
        GammaNV - module inclination (considering tracking) (matrix, radians)
        rot - tracker axis rotation angle in degrees
    tracker axis rotation angle in degrees
        incidence_angle - angle between module normal and sun (radians, matrix)
        cos_incidence_angle = cos_incidence_angle - cosine of incidence angle (matrix)
        sinGammaNV, cosGammaNV
    '''


    sinAN, cosAN, sinGN, cosGN = None, None, None, None
    if mounting in [1,2,3,4]:
        sinAN = np.sin(AN)
        cosAN = np.cos(AN)
    if mounting in [1,2,3,4,5,6]:
        sinGN = np.sin(GN)
        cosGN = np.cos(GN)

    if mounting in [1,2,3,4]: #Fixed 1 position system - FixedOneAngle
        GammaNV,sinGammaNV,cosGammaNV = GN*np.ones_like(a0), sinGN*np.ones_like(a0), cosGN*np.ones_like(a0)
        ANV,    sinANV,    cosANV     = AN*np.ones_like(a0), sinAN*np.ones_like(a0), cosAN*np.ones_like(a0)
        rot,    sinrot,    cosrot     = GammaNV,             sinGammaNV,             cosGammaNV
        cos_incidence_angle=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,sina0,cosa0,sinh0,cosh0)

    else:
        raise 'Unsupported module type'

    rot=np.arccos(cosrot)
    cos_incidence_angle[cos_incidence_angle>1]=1
    cos_incidence_angle[cos_incidence_angle<-1]=-1
    incidence_angle = np.arccos(cos_incidence_angle)



    return incidence_angle, cos_incidence_angle, GammaNV ,sinGammaNV, cosGammaNV, ANV, sinANV, cosANV, rot, sinrot, cosrot


def diffuse_isotropic_shading_meshgrids(skydomeAzimsNum=48, skydomeHeightsNum=60):
    AzimuthStep=(np.pi*2.)/skydomeAzimsNum
    HeightStep=(np.pi/2.)/skydomeHeightsNum
    h0s_r=np.arange((HeightStep/2.),np.pi/2.,HeightStep)
    a0s_r=np.arange(-np.pi+(AzimuthStep/2.),np.pi,AzimuthStep)
    
    sinh0s=np.sin(h0s_r)
    cosh0s=np.cos(h0s_r)
    
    sina0s=np.sin(a0s_r)
    cosa0s=np.cos(a0s_r)
    
    a0,h0=np.meshgrid(a0s_r,h0s_r)
    sina0,sinh0=np.meshgrid(sina0s,sinh0s)
    cosa0,cosh0=np.meshgrid(cosa0s,cosh0s)
    
    return a0,h0,sina0,cosa0,sinh0,cosh0


def prepare_shading(HorAspect, HorHeight, h0_sat, a0_sat, longit, latit):

    # interpolate horizon 
    # returns 
    # shading_direct - shading for direct component
    #     array of the length of all data (e.g. a0_sat) 
    #     0/1 values indicate if for given time there will be direct shading 1 - shading, 0 - no shading

    from scipy.interpolate import interp1d   
    HorizonInterpolator = interp1d(HorAspect, HorHeight,kind=1)

    #shading for direct component and circumsolar diffuse
    horiz_sat=HorizonInterpolator(a0_sat)
    shading_direct=np.zeros_like(h0_sat)
    shading_direct=shading_direct.astype(np.int16)
    wh=h0_sat<horiz_sat
    shading_direct[wh]=1

    #shading for isotropic diffuse
    # define model of skydome and calculate which part is shaded. this is later used to calculate  
    skydomeAspectsNum=48
    skydomeHeightsNum=60
#     skydomeAspectsNum=2
#     skydomeHeightsNum=8
    skydome_a0_arr,skydome_h0_arr,skydome_sina0_arr,skydome_cosa0_arr,skydome_sinh0_arr,skydome_cosh0_arr = diffuse_isotropic_shading_meshgrids(skydomeAzimsNum=skydomeAspectsNum, skydomeHeightsNum=skydomeHeightsNum)
    
    skydomeAspectStep = 360./skydomeAspectsNum
    AspectHor=np.arange(-180.+skydomeAspectStep/2.,180.,skydomeAspectStep)
    horiz_isotrop_vect = HorizonInterpolator(AspectHor)
    horiz_isotrop_vect = np.radians(horiz_isotrop_vect)
    horiz_isotrop_arr = np.reshape(np.tile(horiz_isotrop_vect,skydomeHeightsNum),(skydomeHeightsNum,skydomeAspectsNum))
    
    skydome_shading_isotrop_arr = np.zeros((skydomeHeightsNum,skydomeAspectsNum),dtype=np.int16)
    
    wh = skydome_h0_arr < horiz_isotrop_arr
    skydome_shading_isotrop_arr[wh] = 1
    
    # shading must be weighted by incidence angle from skydome
    # cosh0_arr - used for pixel weight - with increasing h0 the pixel size is shorter in lon direction   
    # sina0_arr,cosa0_arr,sinh0_arr,cosh0_arr - used for calculation of the incidence angle from every segment of sky dome
    # calculation is done
    #    calculate skydome_incidence_angle_arr
    #    aux = skydome_incidence_angle_arr*cosh0_arr
    #    isotropic_shading_factor = ((1-skydome_shading_isotrop_arr)*aux).sum()/aux.sum()
    
    shading_data_dict = {}
    shading_data_dict['skydome_sina0_arr'] = skydome_sina0_arr
    shading_data_dict['skydome_cosa0_arr'] = skydome_cosa0_arr
    shading_data_dict['skydome_sinh0_arr'] = skydome_sinh0_arr
    shading_data_dict['skydome_cosh0_arr'] = skydome_cosh0_arr
    shading_data_dict['skydome_shading_isotrop_arr'] = skydome_shading_isotrop_arr
    shading_data_dict['skydome_a0_arr'] = skydome_a0_arr
    shading_data_dict['skydome_h0_arr'] = skydome_h0_arr
    shading_data_dict['shading_direct_vect'] = shading_direct
    return shading_data_dict 


def skydome_isotropic_shading_factor(shading_data_dict, sinGammaNV, cosGammaNV, sinANV, cosANV):

    if not shading_data_dict.has_key('skydome_cosh0_arr'):
        return 1.0
    
    skydome_sina0_arr = shading_data_dict['skydome_sina0_arr']
    skydome_cosa0_arr = shading_data_dict['skydome_cosa0_arr']
    skydome_sinh0_arr = shading_data_dict['skydome_sinh0_arr']
    skydome_cosh0_arr = shading_data_dict['skydome_cosh0_arr']
    skydome_shading_isotrop_arr = shading_data_dict['skydome_shading_isotrop_arr']

    skydome_incidence_angle_arr = Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,skydome_sina0_arr,skydome_cosa0_arr,skydome_sinh0_arr,skydome_cosh0_arr)  # calculate incidence angle for all skydome pointsi
    aux = skydome_incidence_angle_arr * skydome_cosh0_arr
    isotropic_shading_factor = ((1.0-skydome_shading_isotrop_arr)*aux).sum()/aux.sum()

    print 'isotropic_shading_factor = %f'%(isotropic_shading_factor)
    return isotropic_shading_factor


def diffinc_PerezFast(DiffHorInput, B0c, BHc, h0, sinGammaN, cosGammaN, Albedo, sinDeltaexpInput, shadow=0.0, shadow_an=None, sinh0=None, isotropic_shading_factor=1.0):
    '''    DiffHorInput - diffuse horizontal (VECTOR)
    B0c - direct normal DNI (VECTOR)
    BHc - direct horizontal (VECTOR)
    h0 - sun height (elevation) over horizon in radians (VECTOR)
    sinGammaN - sin of GammaN (tilt of module plain) (VECTOR)
    cosGammaN - cos of GammaN (tilt of module plain) (VECTOR)
    Albedo - surface albedo ca 0.11  (SCALAR)
    sinDeltaexpInput - inc_angle_cos - cosine of incidence angle (1. - perpendicular to plane)  (VECTOR) 
    shadow - shading of direct component (1/0)  (VECTOR)
    isotropic_shading_factor - (SCALAR (for fixed) or VECTOR) factor representing integration of reduction if isotropic diffuse component by shading
    optional:
    sinh0 - if not specified - calculated from h0 or Z  (VECTOR)
    shadow_an  - an angle (in radians) of near horizon used to reduce isotropic diffuse by part of shading from previous row  (VECTOR) 
    '''

    if False:
        print '!!!!overriding the diffinc_PerezFast input!!!!'
        GHI=1192.0
        DNI=1088.0
        h0=numpy.array([1.47436022778])
        GammaN=0.471238898038
        sinDeltaexp=0.929373090054
        
        sinGammaN=numpy.sin(GammaN)
        cosGammaN=numpy.cos(GammaN)
        sinDeltaexpInput=numpy.array([sinDeltaexp])
        B0c=numpy.array([DNI])
        BHc=DNI*numpy.sin(h0)
        DiffHorInput=GHI-BHc


    if shadow_an==None:
        cosRshadow=1;
        sinRshadow=0;
    else:
        cosRshadow=numpy.cos(shadow_an);
        sinRshadow=numpy.sin(shadow_an);
    
    DiffHor=DiffHorInput.copy()
    sinDeltaexp=sinDeltaexpInput.copy()
    # From Fortran code
    #F11R=[-0.0083117, 0.1299457, 0.3296958, 0.5682053,0.8730280, 1.1326077, 1.0601591, 0.6777470]
    #F12R=[0.5877285, 0.6825954, 0.4868735, 0.1874525,-0.3920403, -1.2367284, -1.5999137, -0.3272588 ]
    #F13R=[-0.0620636,-0.1513752, -0.2210958, -0.2951290,-0.3616149, -0.4118494, -0.3589221, -0.2504286 ]
    #F21R=[-0.0596012,-0.0189325, 0.0554140, 0.1088631,0.2255647, 0.2877813, 0.2642124, 0.1561313 ]
    #F22R=[0.0721249, 0.0659650, -0.0639588, -0.1519229,-0.4620442, -0.8230357, -1.1272340, -1.3765031 ]
    #F23R=[-0.0220216, -0.0288748, -0.0260542, -0.0139754,0.0012448, 0.0558651, 0.1310694, 0.2506212 ]
    #EPSBINS= [ 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2 ]
    
    # from the paper
    F11R=numpy.array([ 0.041, 0.054, 0.227, 0.486, 0.819, 1.020, 1.009, 0.936])
    F12R=numpy.array([ 0.621, 0.966, 0.866, 0.670, 0.106,-0.260,-0.708,-1.121])
    F13R=numpy.array([-0.105,-0.166,-0.250,-0.373,-0.465,-0.514,-0.433,-0.352])
    F21R=numpy.array([-0.040,-0.016, 0.069, 0.148, 0.268, 0.306, 0.287, 0.226])
    F22R=numpy.array([ 0.074, 0.114,-0.002,-0.137,-0.497,-0.804,-1.286,-2.449])
    F23R=numpy.array([-0.031,-0.045,-0.062,-0.056,-0.029, 0.046, 0.166, 0.383])
    EPSBINS=numpy.array([ 1.056, 1.253, 1.586, 2.134, 3.23, 5.98, 10.08,99.99])
    
    # R. Perez naming style conversion______________________________________________________________________
    G=DiffHor+BHc
    Z=(pi/2.)-h0 # rad!!
    
    # vectorisation of surface slope 
    ZH=DiffHor*0
    AIRMASS=DiffHor*0
    MAXZ=DiffHor*0
    
    # vectorisation of other variables
#    DiffInc= numpy.zeros(h0.shape)
#    ReflInc= numpy.zeros(h0.shape)
    
    B2 = 5.534e-6
    
    if sinh0 is not None:
        CZ = sinh0
    else:
        CZ = numpy.cos(Z)
    
    ZENITH = 180./pi* Z
    ZH[CZ<0.0871557]=0.0871557
    ZH[CZ>0.0871557]=CZ[CZ>0.0871557]

    wh_cz0l = CZ <= 0.0
    wh_cz0u = CZ > 0.0

    DiffHor[wh_cz0u] = G[wh_cz0u] - B0c[wh_cz0u]* CZ[wh_cz0u]
    DiffHor[wh_cz0l] = G[wh_cz0l]

    AIRMASS[ZENITH<93.9]=1.0/(CZ[ZENITH<93.9] + 0.15* numpy.power((93.9 - ZENITH[ZENITH<93.9]),-1.253))
    AIRMASS[ZENITH>=93.9]=999
    
    DELTA = DiffHor * AIRMASS / 1367. #Solar constant
    TB2 = B2*numpy.power(ZENITH,3)

    EPS=numpy.zeros_like(DiffHor)
    #added to avoid zero division
    wh_diff_no0=DiffHor!=0
    EPS[wh_diff_no0] = (B0c[wh_diff_no0] + DiffHor[wh_diff_no0]) / DiffHor[wh_diff_no0]
    EPS = (EPS + TB2) / (1.0 + TB2)

    MAXZ[Z<0.4]=0.4 
    MAXZ[Z>=0.4]=Z[Z>=0.4]
    
#    I= numpy.array(DiffHor*0+6, dtype=int) #original by artur    
#    I= numpy.array(DiffHor*0+7, dtype=int) #updated by tomas 20130701
    I= numpy.zeros(DiffHor.shape, dtype=int)  #updated by tomas 20140622
    I= I+7
    I[EPS<=(EPSBINS[6])]=6
    I[EPS<=(EPSBINS[5])]=5
    I[EPS<=(EPSBINS[4])]=4
    I[EPS<=(EPSBINS[3])]=3
    I[EPS<=(EPSBINS[2])]=2
    I[EPS<=(EPSBINS[1])]=1
    I[EPS<=(EPSBINS[0])]=0
        
    F1 = F11R[I] + F12R[I]*DELTA + F13R[I] * MAXZ
    F1[F1<0]=0 
    F2 = F21R[I] + F22R[I] * DELTA + F23R[I] * MAXZ
    
    
    ALBPROD = Albedo * G
    ReflInc= ALBPROD * (1.0 - cosGammaN)/2.0 #reflected inclined
    sinDeltaexp[sinDeltaexp< 0.0]=0.0
    
#    
#    A = DiffHor * ( 1.0 + cosGammaN*cosRshadow-sinGammaN*sinRshadow) / 2.0; #isotropic adapted for shading
##    A = DiffHor * ( 1.0 + cosGammaN) / 2.0  #isotropic original
#    BR = ((sinDeltaexp/ ZH) * DiffHor) - A  #circumsolar
#    C =  DiffHor * sinGammaN # horizontal band
#    DiffInc = A + ((1-shadow)*F1 * BR) + (F2 * C) #diffuse inclined
#    return DiffInc,ReflInc
#
#    #working wersions 20140423
#
#    print 'current version' 
#    A = DiffHor * ( 1.0 + cosGammaN) / 2.0
#    BR = (sinDeltaexp/ ZH) * DiffHor-A
#    C = DiffHor * sinGammaN
#    DiffInc = A + F1 * BR*(1-shadow) + F2 * C
#    return DiffInc,ReflInc
#    
#
#    print 'new simple version - correct circumsolar' 
#    isotropic = (( 1.0 + cosGammaN) / 2.0) * (1-F1)
#    circumsolar = (sinDeltaexp/ ZH) * F1 * (1-shadow) 
#    horizon = sinGammaN * F2
#    DiffInc = DiffHor * (isotropic + circumsolar + horizon) 
#    return DiffInc,ReflInc    

#    

#    print 'new version - correct circumsolar + isotropic' 
#    isotropic = (( 1.0 + cosGammaN) / 2.0) * (1-F1) * isotropic_shading_factor 
#    circumsolar = (sinDeltaexp/ ZH) * F1 * (1-shadow) 
#    horizon = sinGammaN * F2
#    DiffInc = DiffHor * (isotropic + circumsolar + horizon) 
#    return DiffInc,ReflInc    

#    print 'new version - correct circumsolar + isotropic + reflected'

    print '*** ZH = %s'%(ZH) 
    isotropic = (DiffHor * (( 1.0 + cosGammaN) / 2.0) * (1-F1)  * isotropic_shading_factor)  + ((1. - isotropic_shading_factor) *ALBPROD)
    circumsolar = DiffHor * (sinDeltaexp/ ZH) * F1 * (1-shadow) 
    horizon = DiffHor * sinGammaN * F2
    DiffInc = isotropic + circumsolar + horizon

    return DiffInc,ReflInc


# PV_AngularDependence function direct Thomas PVGIS implementation (Martin and Ruiz 2001 approach)
#
# Input data: 
# GammaN_r - the angle the module forms with the horizontal plane
# Binc - Beam Irradiance on inclined
# DiffInc - Diffused Irradiance on inclined
# sinDeltaexp - sin (or cos) of the angle between the surface (or normal to the surface) and the direction of the sun 
# Refl - Reflected Irradiance
# Output:
# Gout - the total real irradiance being seen by module
def PV_Angular(Binc, DiffInc, Refl, sinDeltaexp, GammaN_r, sinGammaN, cosGammaN, ar=None):
    # Beam component
    # ar: the parameter ranging from 0.14 - 0.17 according to the module type
    if ar==None:
        ar=0.16 # The algorithm used in PVGIS uses that value
    B=Binc*(1.-(numpy.exp(-sinDeltaexp/ar)-numpy.exp(-1./ar))/(1.-numpy.exp(-1./ar)))
    # Reflected and diffused componet
    cosGammaN[cosGammaN > 0.98 ]=0.98
    fd=sinGammaN + (pi-GammaN_r-sinGammaN)/(1.+cosGammaN)
    fr=sinGammaN + (GammaN_r-sinGammaN)/(1.-cosGammaN)
    
    c1=4./(3.*pi)
    c2=ar/2.-0.154
    Diff=DiffInc*(1.-numpy.exp((-c1*fd-c2*fd*fd)/ar))
    
    R=Refl*(1.-numpy.exp((-c1*fr-c2*fr*fr)/ar))
    
    R[numpy.isnan(R)]=0
    Diff[numpy.isnan(Diff)]=0
    B[numpy.isnan(B)]=0
    
    R[R<0]=0
    R[R>200]=0
    Diff[Diff<0]=0
    B[B<0]=0
    
    # return total value of irradiance being seen by PV module  
    return B, Diff, R


def testFixedGTI(lati, longi, isoDateTime, elev, GHI, DNI, fixed_module_tilt, fixed_module_azimuth, Albedo, HorAspect, HorHeight):
    
#     atm_type=1
#     aod=0.25
#     w=1.86967025757
#    print "aod",aod, ", w", w, ", elev", elev, ", atm_type", atm_type
#    TL2=aod2TL(aod,w,elev, atm_type)
#    print "TL2 from aod", (TL2)
#    print "aod from TL", (TL2aod(TL2, w, elev, atm_type))
    
    year, doy, time = parseDateTime(isoDateTime)
    
    print parseDateTime(isoDateTime)
    
#     print 'lon:',longi,'lat:',lati,'elevation:',elev
#     print 'time:',time
#     print 'day:',day,'month:',month,'year:',year,'doy:',doy
#     epsil=solar_geom_v5.epsilon_cor(doy)
#     print 'epsilon:',epsil
#     print 'dayangle1 (Heliosat):',degrees(dayangle_r(doy)),'dayangle2 (Gruter):',degrees(dayangle_r2(doy))
    decli=solar_geom_v5.declin_r(year, doy, radians(longi))
#     print 'declination:',decli,'rad  ', declin_d(year,doy,longi),'deg'
#     print 'perturbation:',perturbation(doy), ' perturbation(min)',perturbation(doy)*60.
    LAT = solar_geom_v5.UTC2LAT_r(time,doy,radians(longi))
#     UTC = solar_geom_v5.LAT2UTC_r(LAT,doy,radians(longi))
#     print 'UTC:',UTC
#     print 'local solar time:',LAT
    a0,h0=solar_geom_v5.sunposition_r(decli,radians(lati),LAT)
#     print 'a0 (rad):', a0, 'h0 (rad):', h0
#     a0,h0refr=solar_geom_v5.sunpositionrefr(decli,radians(lati),LAT)
#     z0 = (pi/2) - h0
# #     print 'a0:',degrees(a0),'z0:',degrees(z0),'h0:',degrees(h0),'h0refr:',degrees(h0refr)
#     srss=solar_geom_v5.srss_r(radians(lati),decli)
# #     print 'sunrise,sunset (rad):',srss
# #     print 'sunrise,sunset (hour):',srss_r2srss_t(srss[0],srss[1])
# #     print 'm0,m:',opt_air_mass(elev,h0refr)
#     G0=solar_geom_v5.extr_G0(epsil)
# #     print 'G0',extr_G0(epsil)
# #     print 'G0h',extr_G0h(epsil,h0refr)
# #     print 'GHI_solis_sat', ghc_solis_sat(G0, aod, w, h0refr, elev)
# #     print 'GHI_solis', ghc_solis(G0, aod, w, h0refr, elev)
# #     print 'GHI_solis2', ghc_solis2(G0, aod, w, h0refr, elev,atm_type)
# #     print 'BNI_solis2', bnc_solis2(G0, aod, w, h0refr, elev,atm_type)
# #     print diffinc_PerezFast(DiffHorInput=asVect(0), B0c=asVect(0), BHc=asVect(0), h0=asVect(h0), sinGammaN=asVect(0),cosGammaN=asVect(0), Albedo=0.11, sinDeltaexpInput=asVect(0), shadow=asVect(0) )
#     am=(solar_geom_v5.opt_air_mass_kasten2(elev,h0refr))[1]
#     print 'Finished'
    
    
    
#     aux_DHI=aux_DNI*aux_sinh0
#     aux_DiffHor= aux_GHI-aux_DHI

    DHI = DNI * sin(h0)
    DiffH = GHI - DHI
    
    AN = radians(fixed_module_azimuth)
    GN = radians(fixed_module_tilt)
    
    sin_a0_v = asVect(sin(a0));
    cos_a0_v = asVect(cos(a0));
    a0_v = asVect(a0);
    
    sin_h0_v = asVect(sin(h0));
    cos_h0_v = asVect(cos(h0));
    h0_v = asVect(h0);
    
    GNV = asVect(GN)
    ANV = asVect(AN)
    
    MOUNTING_FIXED = 1
    mounting_angles_result = mounting_geom_angles(MOUNTING_FIXED , sin_a0_v, cos_a0_v, a0_v, sin_h0_v, cos_h0_v, h0_v, GN=GNV, AN=ANV, latitude=lati, rotation_limit=None, rotation_limit2=None,  backtrack=None, relative_spacing_rows=None, relative_spacing_columns=None )
    
    cos_incidence_angle_v = mounting_angles_result[1]
    
    print 'cos_incidence_angle = %s'%(cos_incidence_angle_v)
    
#     print 'Mounting angles result = %s'%str(mounting_angles_result)
    

    
    h0_deg_v = numpy.degrees(h0_v)
    a0_deg_v = numpy.degrees(a0_v)
    
    shading_data_dict = prepare_shading(HorAspect, HorHeight, h0_deg_v, a0_deg_v, longi, lati)
    
    shadow_v = shading_data_dict['shading_direct_vect']
    
    
    sinGammaNV = numpy.sin(GNV) * numpy.ones_like(a0_v)
    cosGammaNV = numpy.cos(GNV) * numpy.ones_like(a0_v)
    sinANV = numpy.sin(ANV) * numpy.ones_like(a0_v)
    cosANV = numpy.cos(ANV) * numpy.ones_like(a0_v)
    
    _isotropic_shading_factor = skydome_isotropic_shading_factor(shading_data_dict, sinGammaNV, cosGammaNV, sinANV, cosANV)
    
#     shadow_v = 0.0
#     _isotropic_shading_factor = 1
    
    diff_perez_result = diffinc_PerezFast(asVect(DiffH), asVect(DNI), asVect(DHI), asVect(h0), asVect(sin(GN)), asVect(cos(GN)), Albedo, cos_incidence_angle_v, shadow = shadow_v, isotropic_shading_factor = _isotropic_shading_factor)

    DiffInc_v, ReflInc_v = diff_perez_result
    
    DIF_inclined, Reflected_inclined = DiffInc_v[0], ReflInc_v[0]
    cos_incidence_angle = cos_incidence_angle_v[0]
    
    GTI = DNI * cos_incidence_angle + DIF_inclined + Reflected_inclined
    
    shadow = shadow_v[0]    
    DTI_shd = DNI * cos_incidence_angle * (1- shadow)
    
    ar = 0.16
    
    B, Diff, R = PV_Angular(asVect(DTI_shd), asVect(DIF_inclined), asVect(Reflected_inclined), asVect(cos_incidence_angle), asVect(GN), asVect(sin(GN)), asVect(cos(GN)), ar=ar)
#     B, Diff, R = PV_Angular(DTI_shd, DIF_inclined, Reflected_inclined, cos_incidence_angle, GN, sin(GN)), cos(GN)), ar=ar)
    
    
    GTI_corrected = B + Diff + R
    
    
#     DTI_shd_v = asVect(DTI_shd)
    
    
       
    
    print 'inclined = %s'%([DIF_inclined, Reflected_inclined])
    
    print 'time = %s'%(isoDateTime)
    print 'isotropic_shading_factor = %s'%str(_isotropic_shading_factor)
    print 'shadow_vect = %s'%(shadow_v)
    print 'GHI = %s'%(GHI) 
    print 'DNI = %s'%(DNI)
    print 'DiffH = %s'%(DiffH)
    print 'GTI = %s'%(GTI)
    print 'GTI corrected = %s'%(GTI_corrected)
    
    
def testBratislava():
    lat = 48.168732
    lon = 17.125851
    elev = 146.0
    
    WS_AZIMUTH = 180
    
    fixed_module_tilt = 36
    fixed_module_azimuth = WS_AZIMUTH - 180
    
#     script_azimuth = WS_AZIMUTH - 180
#     WS_AZIMUTH = script_azimuth + 180
    
    Albedo = 0.11
    
    TESTSET_WINTER = 1
    TESTSET_SUMMER = 2
    
    TESTSET = TESTSET_SUMMER 
   
    horiz_Y = [
           0.0431198991669, 0.0369599135716, 0.0431198991669, 0.0307999279764, 0.0184799567858, 0, 0, 0, 
           0, 0, 0, 0, 0, 0, 0.00615998559527, 0, 
           0, 0, 0, 0, 0, 0.00615998559527, 0.0123199711905, 0.0184799567858, 
           0.0246399423811, 0.0307999279764, 0.0369599135716, 0.0431198991669, 0.0492798847622, 0.0554398703575, 0.0615998559527, 0.0615998559527, 
           0.0800798127386, 0.0985597695244, 0.110879740715, 0.110879740715, 0.10471975512, 0.0985597695244, 0.123199711905, 0.141679668691, 
           0.135519683096, 0.11703972631, 0.110879740715, 0.10471975512, 0.0923997839291, 0.0800798127386, 0.067759841548, 0.067759841548
        ] 



#     user_horizon = []
#     
#     for X, Y in zip(horiz_X, horiz_Y):
#         xx = (degrees(-X + pi))
#         yy = degrees(Y)
#         
#         if xx < 0:
#             xx = 0
#             
#         user_horizon.append((xx , yy))
#         
#     AspectHor,HeightHor = shading_utils.get_horizon(None, user_horizon, None, None)    
    
    
    # code copy pasted from shading_utils.get_horizon
    HeightHor = numpy.asarray(horiz_Y)
    HeightHor = numpy.degrees(HeightHor)
    HorStep   = 360./len(HeightHor)
    AspectHor = numpy.arange(-180, 180+.01, HorStep)
    HeightHor = numpy.hstack((HeightHor, HeightHor[0]))
    
    
    
#     print AspectHor
#     print HeightHor
    
#     print horizon_x_deg
#     print len(horizon_x_deg)
#     print len(horizon_y_deg)
#     print user_horizon
#     0 / 0
    
    
#     import sys
#     for x, i in zip(Y, range(0, len(Y))):
#         if (i % 8 == 0):
#             sys.stdout.write('\n')
#          
#         sys.stdout.write(str(x))
#         sys.stdout.write(', ')
        
        
    
    DTS_SUMMER = [
           '06-06-2015T03:11:00.000Z',
           '06-06-2015T03:26:00.000Z',
           '06-06-2015T03:41:00.000Z',
           '06-06-2015T03:56:00.000Z',
           '06-06-2015T04:11:00.000Z',
           '06-06-2015T04:26:00.000Z',
           '06-06-2015T04:41:00.000Z',
           '06-06-2015T04:56:00.000Z',
           '06-06-2015T05:11:00.000Z',
           '06-06-2015T05:26:00.000Z',
           '06-06-2015T05:41:00.000Z',
           '06-06-2015T05:56:00.000Z',
           '06-06-2015T06:11:00.000Z',
           '06-06-2015T06:26:00.000Z',
           '06-06-2015T06:41:00.000Z',
           '06-06-2015T06:56:00.000Z',
           '06-06-2015T07:11:00.000Z',
           '06-06-2015T07:26:00.000Z',
           '06-06-2015T07:41:00.000Z',
           '06-06-2015T07:56:00.000Z',
           '06-06-2015T08:11:00.000Z',
           '06-06-2015T08:26:00.000Z'
           ]
    
    GHIS_SUMMER = [10, 32, 59, 94, 132, 172, 214, 255, 299, 342, 385, 428, 471, 515, 556, 596, 636, 677, 712, 747, 779, 809]
    DNIS_SUMMER = [0, 155, 244, 342, 438, 517, 569, 607, 649, 683, 707, 728, 752, 775, 792, 809, 826, 845, 857, 870, 880, 889]
    
    DTS_WINTER = [
                  '01-01-2015T06:56:00.000Z',
                  '01-01-2015T07:11:00.000Z',
                  '01-01-2015T07:26:00.000Z',
                  '01-01-2015T07:41:00.000Z',
                  '01-01-2015T07:56:00.000Z',
                  '01-01-2015T08:11:00.000Z',
                  '01-01-2015T08:26:00.000Z',
                  '01-01-2015T08:41:00.000Z',
                  '01-01-2015T08:56:00.000Z',
                  '01-01-2015T09:11:00.000Z',
                  '01-01-2015T09:26:00.000Z',
                  
                  '01-01-2015T12:41:00.000Z',
                  '01-01-2015T12:56:00.000Z',
                  '01-01-2015T13:11:00.000Z',
                  '01-01-2015T13:26:00.000Z',
                  '01-01-2015T13:41:00.000Z',
                  '01-01-2015T13:56:00.000Z',
                  '01-01-2015T14:11:00.000Z',
                  ]
    
    GHIS_WINTER = [3, 15, 28, 37, 59, 53, 44, 62, 68, 61, 46,       117, 29, 49, 54, 11, 39, 25]
    DNIS_WINTER = [0, 0, 2, 1, 36, 6, 0, 0, 0, 0, 0,                72, 0, 0, 24, 0, 4, 15]
    
    
    if TESTSET == TESTSET_WINTER:
        DTS = DTS_WINTER
        GHIS = GHIS_WINTER
        DNIS = DNIS_WINTER
        
    elif TESTSET == TESTSET_SUMMER:
        DTS = DTS_SUMMER
        GHIS = GHIS_SUMMER
        DNIS = DNIS_SUMMER
    else:
        raise 'Unsupported TESTSET'
    
    
    for dt, ghi, dni in zip(DTS, GHIS, DNIS):
        testFixedGTI(lat, lon, dt, elev, ghi, dni, fixed_module_tilt, fixed_module_azimuth, Albedo, AspectHor, HeightHor)
        print '------------------------------------------------------\n'
        
        
def testPhilipines():
    
    lat = 15.360355
    lon = 120.495877
    elev = 112.0
    
        
    fixed_module_tilt = 10
    fixed_module_azimuth = 0
    Albedo = 0.2
    
     
    horiz_Y = [
           0.024639942381096416, 0.024639942381096416, 0.018479956785822312, 0.018479956785822312, 0.024639942381096416, 0.024639942381096416, 0.024639942381096416, 0.018479956785822312, 
           0.018479956785822312, 0.018479956785822312, 0.018479956785822312, 0.012319971190548208, 0.012319971190548208, 0.006159985595274104, 0.006159985595274104, 0.006159985595274104, 
           0, 0.012319971190548208, 0.018479956785822312, 0.018479956785822312, 0.018479956785822312, 0.018479956785822312, 0.018479956785822312, 0.018479956785822312, 
           0.018479956785822312, 0.036959913571644624, 0.04311989916691873, 0.04311989916691873, 0.04927988476219283, 0.03079992797637052, 0.03079992797637052, 0.024639942381096416, 
           0.024639942381096416, 0.03079992797637052, 0.036959913571644624, 0.03079992797637052, 0.04311989916691873, 0.04311989916691873, 0.04311989916691873, 0.036959913571644624, 
           0.036959913571644624, 0.036959913571644624, 0.036959913571644624, 0.03079992797637052, 0.03079992797637052, 0.024639942381096416, 0.024639942381096416, 0.024639942381096416 
        ] 
    
    
    
    # code copy pasted from shading_utils.get_horizon
    HeightHor = numpy.asarray(horiz_Y)
    HeightHor = numpy.degrees(HeightHor)
    HorStep   = 360./len(HeightHor)
    AspectHor = numpy.arange(-180, 180+.01, HorStep)
    HeightHor = numpy.hstack((HeightHor, HeightHor[0]))
    
    
    DTS_WINTER = [
                  '2014-12-31T22:37:00.000Z',
                  '2014-12-31T22:52:00.000Z',
                  '2015-01-01T23:07:00.000Z',
                  '2015-01-01T23:22:00.000Z',
                  '2015-01-01T23:37:00.000Z',
                  '2015-01-01T23:52:00.000Z',
                  '2015-01-01T00:07:00.000Z',
                  '2015-01-01T00:22:00.000Z',
                  '2015-01-01T00:37:00.000Z',
                  '2015-01-01T00:52:00.000Z',
                  '2015-01-01T01:07:00.000Z',
                  '2015-01-01T01:22:00.000Z',
                  '2015-01-01T01:37:00.000Z',
                  '2015-01-01T01:52:00.000Z',
                  '2015-01-01T02:07:00.000Z',
                  '2015-01-01T02:22:00.000Z',
                  
                  ]
    
    GHIS_WINTER = [24, 55, 101, 150, 202, 253, 303, 352, 400, 438, 466, 495, 525, 560, 599, 612]
    DNIS_WINTER = [85, 157, 243, 324, 401, 459, 497, 526, 546, 536, 496, 472, 465, 479, 513, 483]

    DTS = DTS_WINTER
    GHIS = GHIS_WINTER
    DNIS = DNIS_WINTER
    
    for dt, ghi, dni in zip(DTS, GHIS, DNIS):
        testFixedGTI(lat, lon, dt, elev, ghi, dni, fixed_module_tilt, fixed_module_azimuth, Albedo, AspectHor, HeightHor)
        print '------------------------------------------------------\n'
        
        
        
        
def testAfrica():
    
#      <ws:dataDeliveryRequest dateFrom="2015-01-01" dateTo="2015-01-31"
#          xmlns="http://geomodel.eu/schema/data/request"
#          xmlns:ws="http://geomodel.eu/schema/ws/data"
#          xmlns:geo="http://geomodel.eu/schema/common/geo" 
#          xmlns:pv="http://geomodel.eu/schema/common/pv" 
#          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
#          
#          <site id="TEST_SITE" name="Second site" lat="-28.856111" lng="21.783056" >                        
#              <pv:geometry xsi:type="pv:GeometryFixedOneAngle" azimuth="25" tilt="40"/>
#          </site>
#          
#          <processing key="GHI DNI DIF GTI" summarization="MIN_15" terrainShading="true">
#              <timestampType>CENTER</timestampType>        
#          </processing>
#      
#      </ws:dataDeliveryRequest>
    
    lat = -28.856111
    lon = 21.783056
    elev = 912.0;
    
        
    fixed_module_tilt = 40
#     fixed_module_azimuth = 155
    fixed_module_azimuth = 25 - 180
    
    Albedo = 0.2
     
    horiz_Y = [
          0.012319971190548208, 0.012319971190548208, 0.006159985595274104, 0.006159985595274104, 0.006159985595274104, 0.006159985595274104, 0.006159985595274104, 0.006159985595274104, 
          0.006159985595274104, 0.006159985595274104, 0.006159985595274104, 0, 0.006159985595274104, 0.006159985595274104, 0.006159985595274104, 0.012319971190548208, 
          0.012319971190548208, 0.024639942381096416, 0.018479956785822312, 0.024639942381096416, 0.018479956785822312, 0.024639942381096416, 0.012319971190548208, 0.024639942381096416, 
          0.006159985595274104, 0.006159985595274104, 0.006159985595274104, 0.012319971190548208, 0.03079992797637052, 0.04927988476219283, 0.06159985595274104, 0.06159985595274104, 
          0.05543987035746693, 0.05543987035746693, 0.05543987035746693, 0.04311989916691873, 0.036959913571644624, 0.04311989916691873, 0.04927988476219283, 0.03079992797637052, 
          0.036959913571644624, 0.036959913571644624, 0.036959913571644624, 0.024639942381096416, 0.024639942381096416, 0.018479956785822312, 0.012319971190548208, 0.012319971190548208
        ] 
    
    # code copy pasted from shading_utils.get_horizon
    HeightHor = numpy.asarray(horiz_Y)
    HeightHor = numpy.degrees(HeightHor)
    HorStep   = 360./len(HeightHor)
    AspectHor = numpy.arange(-180, 180+.01, HorStep)
    HeightHor = numpy.hstack((HeightHor, HeightHor[0]))
    
    
    DTS_WINTER = [
                  '2015-01-01T04:02:00.000Z',
                  '2015-01-01T04:17:00.000Z',
                  '2015-01-01T04:32:00.000Z',
                  '2015-01-01T04:47:00.000Z',
                  '2015-01-01T05:02:00.000Z',
                  '2015-01-01T05:17:00.000Z',
                  '2015-01-01T05:32:00.000Z',
                  '2015-01-01T05:47:00.000Z',
                  '2015-01-01T06:02:00.000Z',
                  '2015-01-01T06:17:00.000Z',
                  '2015-01-01T06:32:00.000Z',
                  '2015-01-01T06:47:00.000Z',
                  '2015-01-01T07:02:00.000Z',
                  '2015-01-01T07:17:00.000Z',                  
                  '2015-01-01T10:17:00.000Z', 
                  '2015-07-07T11:17:00.000Z'
                                   
                  ]
    
    GHIS_WINTER = [49, 97, 151, 207, 263, 323, 382, 441, 500, 558, 617, 673, 727, 781, 1151, 606]
    DNIS_WINTER = [375, 512, 639, 703, 753, 798, 831, 857, 881, 902, 924, 942, 958, 975, 1062, 836]
    
    DTS = DTS_WINTER
    GHIS = GHIS_WINTER
    DNIS = DNIS_WINTER
    
    for dt, ghi, dni in zip(DTS, GHIS, DNIS):
        testFixedGTI(lat, lon, dt, elev, ghi, dni, fixed_module_tilt, fixed_module_azimuth, Albedo, AspectHor, HeightHor)
        print '------------------------------------------------------\n'


def testChina():
    
#     <ws:dataDeliveryRequest dateFrom="2015-01-01" dateTo="2015-01-01"
#         xmlns="http://geomodel.eu/schema/data/request"
#         xmlns:ws="http://geomodel.eu/schema/ws/data"
#         xmlns:geo="http://geomodel.eu/schema/common/geo" 
#         xmlns:pv="http://geomodel.eu/schema/common/pv" 
#         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
#         
#         <site id="TEST_SITE" name="Second site" lat="49.742937" lng="122.0353" >
#             <pv:geometry xsi:type="pv:GeometryFixedOneAngle" azimuth="110" tilt="33"/>
#         </site>
#         
#         <processing key="GHI DNI DIF GTI" summarization="MIN_15" terrainShading="false">
#             <timestampType>CENTER</timestampType>
#              <timeZone>GMT+08</timeZone>
#         </processing>
#     
#     </ws:dataDeliveryRequest>

    lat = 49.742937
    lon = 122.0353
    elev = 970.0
    
        
    fixed_module_tilt = 33
#     fixed_module_azimuth = 155
    fixed_module_azimuth = 110 - 180
    
    Albedo = 0.11
     
    horiz_Y = [
        0.03079992797637052, 0.012319971190548208, 0, 0, 0, 0.012319971190548208, 0.018479956785822312, 0.012319971190548208, 
        0, 0, 0.006159985595274104, 0.012319971190548208, 0.012319971190548208, 0.006159985595274104, 0.006159985595274104, 0.006159985595274104, 
        0.012319971190548208, 0.012319971190548208, 0.012319971190548208, 0.006159985595274104, 0, 0, 0, 0.006159985595274104, 
        0.012319971190548208, 0.012319971190548208, 0.024639942381096416, 0.036959913571644624, 0.05543987035746693, 0.07391982714328925, 0.08623979833383746, 0.10471975511965977, 
        0.11703972631020798, 0.12319971190548208, 0.1355196830960303, 0.14167966869130438, 0.1478396542865785, 0.1478396542865785, 0.1539996398818526, 0.1539996398818526, 
        0.1478396542865785, 0.1478396542865785, 0.14167966869130438, 0.1355196830960303, 0.12319971190548208, 0.09855976952438567, 0.08007981273856335, 0.05543987035746693,
    ] 
    
    # code copy pasted from shading_utils.get_horizon
    HeightHor = numpy.asarray(horiz_Y)
    HeightHor = numpy.degrees(HeightHor)
    HorStep   = 360./len(HeightHor)
    AspectHor = numpy.arange(-180, 180+.01, HorStep)
    HeightHor = numpy.hstack((HeightHor, HeightHor[0]))
    
    
    DTS_WINTER = [
#                   '2015-01-01T08:07:00.000+08:00',
#                   '2015-01-01T08:22:00.000+08:00'
                  '2015-01-01T00:07:00.000Z',
                  '2015-01-01T00:22:00.000Z',
                  '2015-01-01T03:07:00.000Z',
                  '2015-01-01T03:52:00.000Z'
                  
                  ]
    
    GHIS_WINTER = [26, 40, 288, 298]
    DNIS_WINTER = [235, 305, 849, 859]
    
    DTS = DTS_WINTER
    GHIS = GHIS_WINTER
    DNIS = DNIS_WINTER
    
    for dt, ghi, dni in zip(DTS, GHIS, DNIS):
        testFixedGTI(lat, lon, dt, elev, ghi, dni, fixed_module_tilt, fixed_module_azimuth, Albedo, AspectHor, HeightHor)
        print '------------------------------------------------------\n'
        
        
def testHorizons():
    

    lat = 48.788545
    lon = 20.559025
    elev = 954


    fixed_module_tilt = 80
    fixed_module_azimuth = 0 - 180

    Albedo = 0.11
     
    horiz_Y = [
        0.018479956785822312, 0.012319971190548208, 0, 0, 0, 0, 0, 0,
        0.024639942381096416, 0.04927988476219283, 0.09239978392911156, 0.09239978392911156, 0.08007981273856335, 0.08623979833383746, 0.11087974071493387, 0.14167966869130438,
        0.19711953904877133, 0.23407945262041596, 0.2771993517873347, 0.3018392941684311, 0.32647923654952754, 0.34495919333534986, 0.3634391501211721, 0.36959913571644626,
        0.37575912131172035, 0.3880790925022686, 0.40039906369281675, 0.4065590492880909, 0.412719034883365, 0.41887902047863906, 0.41887902047863906, 0.412719034883365,
        0.40039906369281675, 0.3880790925022686, 0.3634391501211721, 0.3387992077400757, 0.3018392941684311, 0.24023943821569005, 0.17863958226294901, 0.1663196110724008,
        0.1601596254771267, 0.1478396542865785, 0.14167966869130438, 0.12935969750075618, 0.09855976952438567, 0.08007981273856335, 0.036959913571644624, 0.012319971190548208,
    ] 
    
    # code copy pasted from shading_utils.get_horizon
    HeightHor = numpy.asarray(horiz_Y)
    HeightHor = numpy.degrees(HeightHor)
    HorStep   = 360./len(HeightHor)
    AspectHor = numpy.arange(-180, 180+.01, HorStep)
    HeightHor = numpy.hstack((HeightHor, HeightHor[0]))
    
    
    DTS_WINTER = [
#                   '2015-01-01T08:07:00.000+08:00',
#                   '2015-01-01T08:22:00.000+08:00'
        "2015-01-01T09:56:00.000Z",
        "2015-01-01T10:11:00.000Z",
        "2015-01-01T10:26:00.000Z",
        "2015-01-01T10:41:00.000Z",
        "2015-01-01T10:56:00.000Z",
        "2015-01-01T11:11:00.000Z"
                  
                  ]
    
    GHIS_WINTER = [152, 147, 128, 114, 118, 121]
    DNIS_WINTER = [67, 51, 18, 4, 5, 13]
    
    DTS = DTS_WINTER
    GHIS = GHIS_WINTER
    DNIS = DNIS_WINTER
    
    for dt, ghi, dni in zip(DTS, GHIS, DNIS):
        testFixedGTI(lat, lon, dt, elev, ghi, dni, fixed_module_tilt, fixed_module_azimuth, Albedo, AspectHor, HeightHor)
        print '------------------------------------------------------\n'
        
        
        
        

if __name__ == "__main__":
    testBratislava()
#     testPhilipines()
#     testAfrica()
#     testChina()
#     testHorizons()
    
    
