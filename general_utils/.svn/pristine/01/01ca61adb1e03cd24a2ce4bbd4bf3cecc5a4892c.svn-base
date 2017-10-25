'''
Created on Mar 17, 2013

@author: tomas
'''

import math
import numpy

# calculates satellite azimuth angle for given location
# azimuth is measured clockwise, 0 at south, positive on west, negative on east
# note that for Meteosat 8 the satlon=-3.4 for Meteosat 9 and 7 satlon=0.
# algorithm: from Anja Drews (more elaborated as Lucien's)
def sat_azimuth_d(lon, lat, satlon=0.):
    tlat = math.radians(lat)
    tlon = math.radians(lon)
    tsatlon = math.radians(satlon)
    tazi = sat_azimuth_r(tlon, tlat, tsatlon)
    
    azi = math.degrees(tazi)
    return azi

# calculates satellite zenith angle for given location
# note that for Meteosat 8 the satlon=-3.4 (but from which date?) for Meteosat 9 and 7 satlon=0.
# algorithm: from Anja Drews (same as Lucien')
def sat_zenith_d(lon, lat, satlon=0.):
    Re = 6378.15 # earth equatorial radius
    Hs = 35786.0 # altitude of satellite
    Rs = Re + Hs # orbit radius

    tlat = math.radians(lat)
    tlon = math.radians(lon-satlon)
    
    coslatlon = math.cos(tlat)*math.cos(tlon)
    
    cos_zenith = ((Rs * coslatlon) - Re) / math.sqrt((Hs*Hs) + (2.*Re*Rs*(1. - coslatlon)))
    zenith = math.degrees(math.acos(cos_zenith))
    
    return zenith

# calculates satellite azimuth angle for given location - radians
# azimuth is measured clockwise, 0 at south, positive on west, negative on east
# note that for Meteosat 8 the satlon=-3.4 for Meteosat 9 and 7 satlon=0.
# algorithm: from Anja Drews (more elaborated as Lucien's)
def sat_azimuth_r(lon, lat, satlon=0.):
    tlat = lat
    tlon = lon-satlon
    
    if (lat == 0.):
        if (tlon < 0.):
            tazi = math.pi / 2.
        else:
            tazi = 3. * math.pi / 2.
    else:
        tazi = math.atan(math.tan(tlon)/math.sin(tlat)) + math.pi
        if (lat < 0.):
            if (tlon < 0.):
                tazi = tazi - math.pi
            else:
                tazi = tazi + math.pi
    
    tazi -= math.pi
    if (tazi >= (math.pi)):
        tazi = tazi - (2. * math.pi)
    
    if (tazi < -math.pi ):
        tazi = tazi + (2. * math.pi)

    return tazi

# calculates satellite zenith angle for given location - radians
# note that for Meteosat 8 the satlon=-3.4 (but from which date?) for Meteosat 9 and 7 satlon=0.
# algorithm: from Anja Drews (same as Lucien')
def sat_zenith_r(lon, lat, satlon=0.):
    Re = 6378.15 # earth equatorial radius
    Hs = 35786.0 # altitude of satellite
    Rs = Re + Hs # orbit radius
    tlat = lat
    tlon = lon-satlon
    
    coslatlon = math.cos(tlat)*math.cos(tlon)
    
    cos_zenith = ((Rs * coslatlon) - Re) / math.sqrt((Hs*Hs) + (2.*Re*Rs*(1. - coslatlon)))
    zenith_r = math.acos(cos_zenith)
    
    return zenith_r

# calculates satellite horizon angle angle for given location - radians
#version for numpy matrixes
# lon ,lat - longitude abs latitude grid 2D [y,x]
# satlon - lonogitude of satellite 4d [dfb,slot, y,x]
# algorithm: from Anja Drews (same as Lucien')
def sat_hsat_r_vect(lon, lat, satlon=0.):
    Re = 6378.15 # earth equatorial radius
    Hs = 35786.0 # altitude of satellite
    Rs = Re + Hs # orbit radius
    tlat = lat
    tlon = lon-satlon
    
    coslatlon = numpy.cos(tlat)*numpy.cos(tlon)
    
    cos_zenith = ((Rs * coslatlon) - Re) / numpy.sqrt((Hs*Hs) + (2.*Re*Rs*(1. - coslatlon)))
    sath_r = (math.pi/2.) - numpy.arccos(cos_zenith)
    
    return sath_r

#a bit faster version with satlon 2D [day,slot]
def sat_hsat_r_vect2(lon, lat, satlon=0.):
    Re = 6378.15 # earth equatorial radius
    Hs = 35786.0 # altitude of satellite
    Rs = Re + Hs # orbit radius
    sat_lon_unique=numpy.unique(satlon[satlon==satlon])
    days = satlon.shape[0]
    slots = satlon.shape[1]
    rows = lon.shape[0]
    cols = lon.shape[1]
    
    sath_r= numpy.empty((days,slots,rows,cols),dtype=numpy.float64)
    sath_r[:,:,:,:] = numpy.nan
    for slon in sat_lon_unique:
        tlat = lat
        tlon = lon-slon
    
        coslatlon = numpy.cos(tlat)*numpy.cos(tlon)
    
        cos_zenith = ((Rs * coslatlon) - Re) / numpy.sqrt((Hs*Hs) + (2.*Re*Rs*(1. - coslatlon)))
        slon_sath_r = (math.pi/2.) - numpy.arccos(cos_zenith)
        wh_slon = (satlon==slon)
        sath_r[wh_slon ,:,:]=slon_sath_r
    
    return sath_r


# calculates satellite azimuth angle for given location - radians
#version for numpy matrixes
# lon ,lat - longitude abs latitude grid 2D [y,x]
# satlon - lonogitude of satellite 4d [dfb,slot, y,x]
# azimuth is measured clockwise, 0 at south, positive on west, negative on east
# note that for Meteosat 8 the satlon=-3.4 for Meteosat 9 and 7 satlon=0.
# algorithm: from Anja Drews (more elaborated as Lucien's)
def sat_asat_r_vect(lon, lat, satlon):
    tazi = numpy.empty_like(satlon) # assumes the satlon being 4D 
    
    tlat = lat
    tlon = lon-satlon

    wh2d = tlat == 0
    wh2 = tlon[:,:,wh2d] < 0
    tazi[wh2] = math.pi / 2.
    wh2 = tlon[:,:,wh2d] >= 0
    tazi[wh2] = 3. * math.pi / 2.
    
    wh2d = numpy.logical_not(wh2d)
    tazi[:,:,wh2d] = numpy.arctan(numpy.tan(tlon[:,:,wh2d])/numpy.sin(tlat[wh2d])) + math.pi
    
    #wh2d = tlat < 0
    #wh2 = tlon[:,:,wh2d] < 0
    #tazi[wh2] = tazi[wh2] - pi
    #wh2 = tlon[:,:,wh2d] >= 0
    #tazi[wh2] = tazi[wh2] + pi
    
    wh2 = (tlat < 0) & (tlon < 0)
    tazi[wh2] = tazi[wh2] - math.pi
    wh2 = (tlat < 0) & (tlon >= 0)
    tazi[wh2] = tazi[wh2] + math.pi
    
    tazi = tazi - math.pi
    wh2 = tazi >= math.pi
    tazi[wh2] = tazi[wh2] - (2.*math.pi)

    wh2 = tazi < -math.pi
    tazi[wh2] = tazi[wh2] + (2.*math.pi)
    
    return tazi

#a bit faster version with satlon 2D [day,slot]
def sat_asat_r_vect2(lon, lat, satlon):
    sat_lon_unique=numpy.unique(satlon[satlon==satlon])
    days = satlon.shape[0]
    slots = satlon.shape[1]
    rows = lon.shape[0]
    cols = lon.shape[1]
    
    tazi= numpy.empty((days,slots,rows,cols),dtype=numpy.float64)
    tazi[:,:,:,:] = numpy.nan
    
    slon_tazi=numpy.empty((rows,cols),dtype=numpy.float64)
    for slon in sat_lon_unique:
        slon_tazi[:,:]=numpy.nan
        tlat = lat
        tlon = lon-slon
        
        wh2 = (tlat == 0) & (tlon < 0)
        slon_tazi[wh2] = math.pi / 2.
        wh2 = (tlat == 0) & (tlon >= 0)
        slon_tazi[wh2] = 3. * math.pi / 2.
    
        wh2 = (tlat != 0)
        slon_tazi[wh2] =  numpy.arctan(numpy.tan(tlon[wh2])/numpy.sin(tlat[wh2])) + math.pi
    
        wh2 = (tlat < 0) & (tlon < 0)
        slon_tazi[wh2] = slon_tazi[wh2] - math.pi
        wh2 = (tlat < 0) & (tlon >= 0)
        slon_tazi[wh2] = slon_tazi[wh2] + math.pi
    
        slon_tazi = slon_tazi - math.pi
        
        wh2 = slon_tazi >= math.pi
        slon_tazi[wh2] = slon_tazi[wh2] - (2.*math.pi)
        wh2 = slon_tazi < -math.pi
        slon_tazi[wh2] = slon_tazi[wh2] + (2.*math.pi)
    
        wh_slon = (satlon==slon)
        tazi[wh_slon ,:,:]=slon_tazi
    return tazi


#calculate sun-point-satellite angle from 
#radians version
def sat_sun_angle_r(Asun,Hsun, Asat, Hsat ):
    res=numpy.arccos( (numpy.cos(Hsun)*numpy.cos(Hsat)*numpy.cos(Asun-Asat))+(numpy.sin(Hsun)*numpy.sin(Hsat)) )
    return res


def sat_sun_mirror_angle_r(Asun,Hsun, Asat, Hsat ):
    res = numpy.cos(numpy.absolute(Hsun-Hsat)) 
    #difference in sun-sat azim angles
    sun_sat_azim_difference = ((numpy.absolute(((Asun-Asat+math.pi)%(math.pi*2))-math.pi) - (math.pi/2.)))
    sun_sat_azim_difference = (1.25*numpy.sin(sun_sat_azim_difference))-0.25
    res = sun_sat_azim_difference * res
#    res = numpy.radians(res)

    return res

