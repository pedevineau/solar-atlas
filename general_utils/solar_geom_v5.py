#! /usr/bin/env python
# solar_geom
# geometric formulae for sun: coordinates, angles, time ....
#
# last revision: 15/08/2008
#
# I0 - solar constant
# extr_G0h(epsilon,h0_r) - extraterestrial solar irradiance on a horizontal plane
# declin_r(year, doy, longitude_r) - declination parameters and output in rad
# declin_d(year, doy, longitude_d) - declination parameters and output in deg
# dayangle_r(doy) - more precise version with parameters similar to declination calculation
# dayangle2_r(doy)
# perturbation(doy) - Sun perturbation 
# (LMT,doy,longitude_r,ref_longitude_r,summer_cor=0) - local mean (clock) time to local aparent (solar) time, input in radians
# LAT2LMT_r(LAT,doy,longitude_r,ref_longitude_r,summer_corr=0)
# UTC2LAT_r(UTC,doy,longitude_r) - UTC to local aparent (solar) time
# LAT2UTC_r(LAT,doy,longitude_r)
# sunposition_r(declin_r,latitude_r,time_LAT) - calculate sun elevation (h0) and sun azimut (a0) from south
# delta_h0refr(h0r) - calculate refraction
# sunpositionrefr(declin_r,latitude_r,time_LAT) - calculate sun elevation and sun azimuth considering refraction
# srss_r(latitudr_r,declin_r) - calculate sunrise, sunset [rad]
# srss_r2srss_t(sr_r,ss_r) - convert sunrise sunset from rad to time (local solar)
# epsilon_cor(doy) - correction for variation of Earth-Sun distance
# opt_air_mass(elevation,h0refr_r) - optical air mass thickness
# opt_air_mass_kasten1(elevation,h0refr_r) - Kasten pp0 correction
# opt_air_mass_kasten2(elevation,h0refr_r) - Ineichen pp0 correction
# extr_G0(epsilon) - Extraterestrial solar irradiance (normal)
# extr_G0h(epsilon,h0refr_r) - Extraterestrial solar irradiance on a horizontal plane

# ghc(G0, TL, air_mass, alt, h0) - global horizotal clear sky radiation (Ineichen, Perez, 2002)
# bnci(G0, TL, air_mass, alt, h0) - beam normal clear sky radiation (Ineichen, Perez, 2002)
# ghc_sat(G0, TL, air_mass, alt, h0) - global horizotal clear sky radiation (Perez, Ineichen et al., 2002) - this si version used in satellite model



#-------------------
# DO IT: latitude geographic-geocentric correction implemented by Lucien Wald in Heliosat code, 2004
# NOTE: sunrise, sunset do not account for refraction - treba implementovat !!

## CHECK: m and p/p0 in opt_air_mass - OK 
## CHECK: "elif(omegass<-1)" in srss_r - OK

## CHECK: use of more precise declination results in differences of other parameters (h0) - asi netreba!
#-------------------


from math import pi, sin, cos, asin, acos, tan, radians, degrees, exp, pow,log
#from general_utils.daytimeconv import yyyymmdd2doy, yyyymmdd2ymd
from general_utils import daytimeconv
import numpy
import datetime

# solar_constant
I0=1367. #W/m2 

# Earth declination (Bourges 1985, Cavalho and Burges 1991), see ESRA p.107 and Meteonorm manual p 33 
# mean daily value at solar noon - it is accurate enough for solar radiation estimation (ESRA p. 107)
# parameters and output in radians
def declin_r(year, doy, longitude_r):
	b=[0, 0.0064979, 0.4059059, 0.0020054, -0.0029880, -0.0132296, 0.0063809, 0.0003508]
	year_dif = year - 1957
	n0 = 78.8946 + (0.2422 * year_dif) - int(0.25 * year_dif)
	t1 = -0.5 - n0 - (longitude_r/(2*pi))
	omega0 = (pi*2.)/365.2422
	omegat = omega0 * (doy + t1)
	
	decl = b[1] + b[2]*sin(omegat) + b[3]*sin(2.*omegat) + b[4]*sin(3.*omegat) + b[5]*cos(omegat) + b[6]*cos(2.*omegat) + b[7]*cos(3.*omegat)
	return(decl)

#Matrix implementation of declination
def declin_r_arr(year, doy, longitude_r):
	b=[0, 0.0064979, 0.4059059, 0.0020054, -0.0029880, -0.0132296, 0.0063809, 0.0003508]
	year_dif = year - 1957
	aux = 0.25 * year_dif
	aux = numpy.array(aux).astype(numpy.int32)
	n0 = 78.8946 + (0.2422 * year_dif) - aux
	t1 = -0.5 - n0 - (longitude_r/(2*pi))
	omega0 = (pi*2.)/365.2422
	omegat = omega0 * (doy + t1)
	
	decl = b[1] + b[2]*numpy.sin(omegat) + b[3]*numpy.sin(2.0*omegat) + b[4]*numpy.sin(3.*omegat) + b[5]*numpy.cos(omegat) + b[6]*numpy.cos(2.*omegat) + b[7]*numpy.cos(3.*omegat)
	return(decl)


# Ddeclination
# longitude and output in degrees
def declin_d(year, doy, longitude_d):
	return (degrees(declin_r(year, doy, radians(longitude_d))))

def declin_vect_r(year_dim0_vect, doy_dim0_vect, longitude_r):
	'''
    calculate declination angle from set of vectors representing input data at day dimensions and longitude.
    result is vector of one dimension:
        dim0-days (usually day of year)
    inputs:
        year_vect - declination in radians for each individual day
        doy_vect - perturbation for each individual day
        longitude_r - longitude scalar
    outputs:
        decl_vect_r - declination in radians
	Note: longitude can be used as point even for 5x5 deg segment. difference is negligible
	'''

	b=[0, 0.0064979, 0.4059059, 0.0020054, -0.0029880, -0.0132296, 0.0063809, 0.0003508]
	year_dif = year_dim0_vect - 1957
	n0 = 78.8946 + (0.2422 * year_dif) - (0.25 * year_dif).astype(numpy.int16)
	
	t1 = -0.5 - n0 - (longitude_r/(2*pi))
	omega0 = (pi*2.)/365.2422
	omegat = omega0 * (doy_dim0_vect + t1)
	
	decl_vect_r = b[1] + b[2]*numpy.sin(omegat) + b[3]*numpy.sin(2.*omegat) + b[4]*numpy.sin(3.*omegat) + b[5]*numpy.cos(omegat) + b[6]*numpy.cos(2.*omegat) + b[7]*numpy.cos(3.*omegat)
	return(decl_vect_r)



# Day angle: expresses the integral day number chosen as angle from 12:00 on 31 December (heliosat code & Meteonorm)
# output in radians
# preferred equation
def dayangle_r(doy):
	dayangle = pi * 2. * (doy) / 365.2422
	return (dayangle)

# Day angle 
# output in radians (ESRA p. 107, Gruter 1984)
# less acurate equation
def dayangle_r2(doy):
	dayangle = pi * 2. * (doy) / 365.25
	return (dayangle)

# Perturbations in the Earth's rotation (equation of time, ET) (ESRA p. 111, Gruter 1984)
# difference, over the course of a year, between local apparent (solar) time and local mean (clock) time
# the difference between time as read from a sundial (slnecne hodiny) and a clock
# output in decimal hours, day angle in radian
def perturbation(doy):
	dangle=dayangle_r(doy)
	perturb = (-0.128*sin(dangle-0.04887)) - (0.165*sin((2.*dangle) +0.34383))
	return (perturb)

def perturbation_vect(doy_vect):
	dangle=doy_vect * (pi * 2. / 365.2422)
	perturb = (-0.128*numpy.sin(dangle-0.04887)) - (0.165*numpy.sin((2.*dangle) +0.34383))
	return (perturb)


# Local Mean (clock) Time (LMT) to Local Aparent (solar) Time (LAT) (ESRA p. 111)
# doy: day of year
# longitude_r: longitude of the site [rad]
# ref_longitude_r: reference time zone longitude for the site [rad] (0 for GMT)
# summer_corr: corection for summer time winter=0/summer=1
# output LMT and LAT in decimal hours
def LMT2LAT_r(LMT,doy,longitude_r,ref_longitude_r,summer_corr=0):
	ET=perturbation(doy)
	LAT = LMT + ET + (12 * (longitude_r - ref_longitude_r) / pi) - summer_corr
	return LAT

def LAT2LMT_r(LAT,doy,longitude_r,ref_longitude_r,summer_corr=0):
	ET=perturbation(doy)
	LMT = LAT - ET - (12 * (longitude_r - ref_longitude_r) / pi) + summer_corr
	return LMT

#UTC to local aparent (solar) time
def UTC2LAT_r(UTC,doy,longitude_r):
	ET=perturbation(doy)
	LAT = UTC + ET + (12 * (longitude_r) / pi)
	return LAT

def LAT2UTC_r(LAT,doy,longitude_r):
	ET=perturbation(doy)
	UTC = LAT - ET - (12 * (longitude_r) / pi)
	return UTC

def utcDT2LocalApparentDT(utcDT, LongitudeDeg):
	'''
	Function to convert UTC datetime to local apparent datetime - LAT_DT.
	Longitude is in deg, minus for west plus for east.
	'''
	
	#correct to interval <-180,180>
	if LongitudeDeg<-180: LongitudeDeg+=360
	if LongitudeDeg>180: LongitudeDeg-=360
	
	#perturbation
	UTCDoy = daytimeconv.date2doy(utcDT.date())
	ET=perturbation(UTCDoy)
	
	#calculate timedelta
	LAT_Tdelta_dh = ET + (LongitudeDeg / 15.)
	LAT_Tdelta = datetime.timedelta(hours=LAT_Tdelta_dh)
	
	#resulting DT
	LAT_DT =   utcDT + LAT_Tdelta
	
	return LAT_DT

def localApparentDT2utcDT(LAT_DT, LongitudeDeg):
	'''
	Function to convert Local Apparent Datetime - LAT_DT, to UTC date time on zero meridian.
	'''
	#correct to interval <-180,180>
	if LongitudeDeg<-180: LongitudeDeg+=360
	if LongitudeDeg>180: LongitudeDeg-=360
	
	#Calculation of utc without perturbation
	LAT_Tdelta_dh_base = LongitudeDeg / 15.
	LAT_Tdelta_base = datetime.timedelta(hours=LAT_Tdelta_dh_base)
	UTC_DT_base = LAT_DT - LAT_Tdelta_base
	
	#Perturbation time correction
	LAT_Doy = daytimeconv.date2doy(UTC_DT_base.date())
	ET= perturbation(LAT_Doy)
	Pert_TimeCorr = datetime.timedelta(hours=ET)
	UTC_DT = UTC_DT_base - Pert_TimeCorr
	
	#Test of date border within perturbation range - plus one doy
	Test_Diff = datetime.timedelta(days=1)
	DT_Test = UTC_DT_base + Test_Diff
	LAT_Doy_plus = daytimeconv.date2doy(DT_Test.date())    
	ET_plus = perturbation(LAT_Doy_plus)
	Pert_TimeCorr_plus = datetime.timedelta(hours=ET_plus)
	
	#UTC with test plus one day position
	UTC_DT_plus = UTC_DT_base - Pert_TimeCorr_plus
	# Test of reverse calculation for case where date switch is within perturbation interval 
	LAT_Test = utcDT2LocalApparentDT(UTC_DT, LongitudeDeg)
	
	UTC_DT_plus_doy = daytimeconv.date2doy(UTC_DT_plus.date())
	# UTC is one doy shifted if there is date switch, year switch, if reverse function does not give the same result
	if (LAT_Doy > UTC_DT_plus_doy and UTC_DT.year <  UTC_DT_plus.year) or LAT_Doy < UTC_DT_plus_doy or LAT_Test != LAT_DT:
		UTC_DT = UTC_DT_plus
	
	
	return UTC_DT

# Solar horizontal coordinates
# according to ESRA a SOLPOS (NREL) http://rredc.nrel.gov/solar/codes_algs/solpos/
# a0: Solar azimuth measured from South [rad]
# h0: Solar altitude (incidence angle) - an angle between the centre of solar disc and the horizontal plane [rad]
# z0: Solar zenithal angle - an angle between the centre of solar disc and the normal to the horizontal plane [rad]
# declination and site latitude are in rad, 
# time is local apparent (solar) time (LAT); LAT=0.0 to 24.0
# azimuth is measured from the true South in the Northern hemisphere and from the true North in the Southern hemisphere
def sunposition_r(declin_r,latitude_r,time_LAT):
	time_LAT=time_LAT%24
	time_r = radians((time_LAT - 12)*15) #time in radians; 12:00 hour is 0.0; East is negative; West positive
	
	sinfi = sin(latitude_r)
	cosfi = cos(latitude_r)
	sinde = sin(declin_r)
	cosde = cos(declin_r)
#	sint = sin(time_r)
	cost = cos(time_r)
#	print sinfi, cosfi, sinde, cosde, cost
	
	sinh0 = (sinfi*sinde) + (cosfi*cosde*cost)
	h0_r = asin(sinh0)
	cosh0 = cos(h0_r)
#	z0_r = (pi/2) - h0_r
	
	cecl=cosfi*cosh0
	a0_r=pi				#toto je pre J a S pol - nie je definovana - da sa nahradit "time in radians" ???
	if abs(cecl) >= 0.001:
		cosas = ((sinfi*sinh0) - sinde)/(cecl)
		if (cosas>1.):cosas=1.     	#to avoid numerical problems
		if (cosas<-1.):cosas=-1.
		a0_r=pi - acos(cosas)
		if time_r > 0.:
			a0_r =  - a0_r	   	# correction 360 deg
	a0_r = a0_r + pi		   	# correction to have a0 in range from -180 to 180
	if (a0_r > pi): a0_r = a0_r - (2*pi)	
	return([a0_r,h0_r])


#vector version 
def sunposition_vect_r(longit_r_dim3_vect,latit_r_dim2_vect,declin_r_dim0_vect, ET_dim0_vect, UTC_dim1_vect):
	'''
    calculate sun position angles from set of vectors representing input data at relevant dimensions.
    result is matrix of four dimensions. Dimensions have specific meaning and must represent following:
        dim0-days (usually day of year)
        dim1-time
        dim2-longitude
        dim3-latitude
    inputs:
        declin_r_dim0_vect - declination in radians for each individual day
        ET_dim0_vect - perturbation for each individual day
        UTC_dim1_vect - UTC in decimal hours
        latit_r_dim2_vect - latitude in radians
        longit_r_dim3_vect - longitude in radians
    outputs:
        a0_r - sun azimuth in radians
        h0_r - sun elevation angle in radians
        h0_refr_r - sun elevation angle with refraction in radians
        sinh0 - sine of sun elevation angle
    '''
	
	dim0=ET_dim0_vect.shape[0]
	dim1=UTC_dim1_vect.shape[0]
	dim2=latit_r_dim2_vect.shape[0]
	dim3=longit_r_dim3_vect.shape[0]

#    ET_dim01_2d = numpy.repeat(ET_dim0_vect,dim1).reshape(dim0,dim1)
	UTC_dim01_2d = numpy.tile(UTC_dim1_vect,(dim0,1)).reshape(dim0,dim1)
	UTC_dim013_3d = numpy.repeat(UTC_dim01_2d,dim3).reshape(dim0,dim1,dim3)
	ET_dim013_3d = numpy.repeat(ET_dim0_vect,dim1*dim3).reshape(dim0,dim1,dim3)
	longit_dim013_3d = numpy.tile(longit_r_dim3_vect,(dim0,dim1,1)).reshape(dim0,dim1,dim3)

#    print ET_dim01_2d.shape, ET_dim01_2d[0,:], ET_dim01_2d[:,0]
#    print UTC_dim013_3d.shape, UTC_dim013_3d[:,0,0], UTC_dim013_3d[0,:,0], UTC_dim013_3d[0,2,:]
#    print ET_dim013_3d.shape, ET_dim013_3d[:,0,0], ET_dim013_3d[0,:,0], ET_dim013_3d[2,0,:]
#    print longit_dim013_3d.shape, longit_dim013_3d[:,0,0], longit_dim013_3d[0,:,1], longit_dim013_3d[2,0,:]

	#calculate time 
	LAT_dim013_3d = UTC_dim013_3d + ET_dim013_3d + (12 * (longit_dim013_3d) / pi)
	LAT_dim013_3d=numpy.mod(LAT_dim013_3d,24)
	time_r = numpy.radians((LAT_dim013_3d - 12)*15)
	aux=numpy.cos(time_r)
	costime=numpy.empty((dim0,dim1,dim2,dim3),dtype=aux.dtype)
	for dim2_idx in range(dim2):
		costime[:,:,dim2_idx,:]=aux
	
	#declination sin and cos        
	sinde_1d = numpy.sin(declin_r_dim0_vect)
	cosde_1d = numpy.cos(declin_r_dim0_vect)
	sinde = numpy.repeat(sinde_1d,dim1*dim2*dim3).reshape(dim0,dim1,dim2,dim3)
	cosde = numpy.repeat(cosde_1d,dim1*dim2*dim3).reshape(dim0,dim1,dim2,dim3)
	
	
	#latitude scos and sin
	aux1=numpy.cos(latit_r_dim2_vect)
	aux2=numpy.sin(latit_r_dim2_vect)
	
	aux1=numpy.repeat(aux1,dim3).reshape(dim2,dim3)
	cosfi = numpy.tile(aux1,(dim0,dim1,1,1)).reshape(dim0,dim1,dim2,dim3)
	aux2=numpy.repeat(aux2,dim3).reshape(dim2,dim3)
	sinfi = numpy.tile(aux2,(dim0,dim1,1,1)).reshape(dim0,dim1,dim2,dim3)
	
	sinh0 = (sinfi*sinde) + (cosfi*cosde*costime)
	h0_r = numpy.arcsin(sinh0)
	cecl=cosfi*numpy.cos(h0_r)
	
	a0_r=numpy.empty_like(h0_r)
	a0_r[:,:,:,:] = pi                #toto je pre J a S pol - nie je definovana - da sa nahradit "time in radians" ???
	wh1 = numpy.abs(cecl) >= 0.001
	cosas = ((sinfi[wh1]*sinh0[wh1]) - sinde[wh1])/(cecl[wh1])
	cosas[cosas>1.] =1.
	cosas[cosas<-1.] =-1.
	a0_r[wh1]=pi - numpy.arccos(cosas)
	del(cosas, sinfi, sinde, cecl)
	wh1 = time_r > 0
	a0_r[wh1] =  - a0_r[wh1]           # correction 360 deg
	a0_r = a0_r + pi               # correction to have a0 in range from -180 to 180
	wh1=a0_r > pi
	a0_r[wh1] = a0_r[wh1] - (2*pi)
	del(time_r)
	
	#refraction correction
	h0_r2 = h0_r*h0_r
	h0_refr_r = h0_r + 0.061359*(0.1594+(1.1230*h0_r)+(0.065656*h0_r2))/(1.+(28.9344*h0_r)+(277.3971*h0_r2))
	
	
	return (a0_r,h0_r, h0_refr_r, sinh0)



# Atmospheric refraction
# delta_h0refr: atmospheric refraction [rad]
def delta_h0refr(h0_r):
	h0_r2 = h0_r*h0_r
	delta_h0refr = 0.061359*(0.1594+(1.1230*h0_r)+(0.065656*h0_r2))/(1.+(28.9344*h0_r)+(277.3971*h0_r2))
	return(delta_h0refr)

# Solar horizontal coordinates with influence of refraction [rad]
# delta_h0refr: atmospheric refraction [rad]
# h0refr: true solar altitude (incidence angle) h0, when refraction considered [rad]
def sunpositionrefr(declin_r,latitude_r,time_LAT):
	sunpos=sunposition_r(declin_r,latitude_r,time_LAT)
	h0refr=sunpos[1]+delta_h0refr(sunpos[1])
	return ([sunpos[0],h0refr])

# Astronmic time of sunrise/sunset in radians (i.e. no refraction taken into account; ESRA p. 109)
# refraction not implemented!! To be done
def srss_r(latitude_r,declin_r):
	omega_ss=-tan(latitude_r)*tan(declin_r)
	if(omega_ss<1.) and (omega_ss>-1.):
		sr_r=-acos(omega_ss)
	elif(omega_ss<=-1.):
		sr_r=-pi
	else:
		sr_r=0.
	ss_r=-sr_r	
	return([sr_r,ss_r])

# Conversion of srss from angle (radians) to local solar time LAT (decial hours)
def srss_r2srss_t(sr_r,ss_r):
	sr_t=degrees(sr_r)/15.+12.
	ss_t=degrees(ss_r)/15.+12.
	return([sr_t,ss_t])

# Correction for variation of Earth-Sun distance from its mean value
def epsilon_cor(doy):
	eps=1.+(0.0334*cos(dayangle_r(doy)-0.048869))
	return(eps)

def epsilon_cor_vect(doy): #vector version
	eps=1.+(0.0334*numpy.cos(dayangle_r(doy)-0.048869))
	return(eps)

# Correction for variation of Earth-Sun distance from its mean value (by Perez sat. model description)
def epsilon_cor_per(doy):
	eps=1.+(0.033*cos(doy*0.0172))
	return(eps)

# Extraterestrial solar irradiance (normal)
def extr_G0(epsilon):
	G0=epsilon*I0
	return(G0)

# Extraterestrial solar irradiance on a horizontal plane
def extr_G0h(epsilon,h0_r):
	if h0_r >= 0:
		G0h=epsilon*I0*sin(h0_r)
	else:
		G0h=0.
	return(G0h)

# Optical air mass m (Remund, Wald, Lefevre, Ranchin, Page, 2003; Remund, Page 2002)
# m0: for the sea level
# m: corrected for the altitude (pressure)
def opt_air_mass(elevation,h0refr_r):
	pp0=exp(-elevation/8435.2)
	if h0refr_r >= 0:
		m0=1./(sin(h0refr_r)+(0.50572*pow((degrees(h0refr_r)+6.07995),-1.6364)))
		m=pp0*m0
	else:
		m0=37.919608377836333
		m=pp0*m0
	return (m0,m)


# Optical air mass m (Iqbal 1983; Kasten 1966) 
# m0: for the sea level
# m: corrected for the altitude (pressure)
def opt_air_mass_kasten1(elevation,h0refr_r):
	pp0=exp(-elevation*0.0001184)
	if h0refr_r >= 0:
		m0=1./(sin(h0refr_r)+(0.15*pow((3.885 + degrees(h0refr_r)),-1.253)))
		m=pp0*m0
	else:
		m0=36.510324500288753
		m=pp0*m0
	return (m0,m)

# Optical air mass m (Iqbal 1983; Kasten 1966)  
# - with pp0 correction used in dirindex 
# m0: for the sea level
# m: corrected for the altitude (pressure)
def opt_air_mass_kasten2(elevation,h0refr_r):
	pp0=(1.-(elevation/10000.))
	if h0refr_r >= 0:
		m0=1./(sin(h0refr_r)+(0.15*pow((3.885 + degrees(h0refr_r)),-1.253)))
		m=pp0*m0
	else:
		m0=36.510324500288753
		m=pp0*m0
	return (m0,m)


# global horizotal clear sky radiation (Ineichen, Perez, 2002)
# G0- extraterestial solar irradiance
# TL - linke turbidity for airmass 2
# AM - airmass (kasten2) - elevation corrected
# alt - altitude
# h0 - sun height angle
def ghc(G0, TL, air_mass, alt, h0):
	TLcor = TL
	if TL < 2.:
		TLcor = TL - 0.25 * ((2. - TL)**0.5)
	cg1=(0.0000509 * alt) + 0.868
	cg2=(0.0000392 * alt) + 0.0387
	fh1=exp(-alt/8000.)
	fh2=exp(-alt/1250.)
	sinh0=sin(h0)
	aux1=(-cg2)*air_mass*(fh1+(fh2*(TLcor-1.)))
	GHIc=cg1*G0*sinh0*exp(aux1)
	return (GHIc)

# global horizotal clear sky radiation (Perez, Ineichen et al., 2002)
# - this is version used in satellite model
# G0- extraterestial solar irradiance (epsilon corrected)
# TL - linke turbidity for airmass 2
# air_mass - airmass (kasten2) - elevation corrected
# alt - altitude
# h0 - sun height angle
def ghc_sat(G0, TL, air_mass, alt, h0):
	TLcor = TL
	if TL < 2.:
		TLcor = TL - 0.25 * ((2. - TL)**0.5)
	cg1=(0.0000509 * alt) + 0.868
	cg2=(0.0000392 * alt) + 0.0387
	fh1=exp(-alt/8000.)
	fh2=exp(-alt/1250.)
	sinh0=sin(h0)
	aux1=(-cg2)*air_mass*(fh1+(fh2*(TLcor-1.)))
	aux2=0.01*(air_mass**1.8)
	GHIc=cg1*G0*sinh0*exp(aux1)*exp(aux2)
	return (GHIc)


# global horizotal clear sky radiation - Solis - from Richard's description
# - version used in satellite model and used in dirindex
# G0- extraterestial solar irradiance (epsilon corrected)
# aod - broadband
# w - precipitable water
# alt - altitude
# h0 - sun height angle (in radians)
def ghc_solis_sat(G0, aod, w, h0, alt):
	gloa=(-0.071*aod)+(0.0006*w*w)-(0.0091*w)+0.381 
	pp0=(1.-(alt/10000.))
	glotau=0.144*(1.-pp0)+(-(0.0042*w*w)+(0.0103*w)-1.124)*aod+(0.00422*w*w-0.0522*w-0.215)
	G0mod=G0*((0.393*(w**0.348))*aod*aod+1.188*(w**0.036)*aod+1.078*(w**0.00659))-(1.-pp0)*111.
	sinh0=sin(h0)
	GHIc=G0mod*exp(glotau/(sinh0**gloa))*sinh0
	return (GHIc)

# global horizotal clear sky radiation - Solis - Ineichen 2008
# G0- extraterestial solar irradiance (epsilon corrected)
# aod - broadband
# w - precipitable water
# alt - altitude
# h0 - sun height angle
def ghc_solis(G0, aod, w, h0, alt):
	logw=log(w)
	gloa=(-0.0147*logw)-(0.3079*aod*aod)+(0.2846*aod)+0.3798 
	pp0=exp(-alt/8427.7)
	tg1=1.24+0.047*logw+0.0061*logw*logw
	tg0=0.27+0.043*logw+0.0090*logw*logw
	tgp=0.0079*w+0.1
	glotau=tg1*aod+tg0+tgp*log(pp0)
	G0mod=G0*((0.12*(w**0.56))*aod*aod+0.97*(w**0.032)*aod+1.08*(w**0.0051)+0.071*log(pp0))
	sinh0=sin(h0)
	GHIc=G0mod*exp(-glotau/(sinh0**gloa))*sinh0
	return (GHIc)


# global horizotal clear sky radiation - Solis - Ineichen 2008 -from excel sheet
# G0- extraterestial solar irradiance (epsilon corrected)
# aod - broadband
# w - precipitable water
# alt - altitude
# h0 - sun height angle
# atmosphere type (1, 4, 5, 6)
# sinh0 can be used instead of h0 - to speed up processing, in such case let h0  be None value it  won't be used (but must be specified e.g. None)
def ghc_solis2(G0, aod, w, h0, alt, atm_type=5, sinh0=None):
	if not(atm_type in [1,4,5,6]):
		print "atmosphere type must be one of: 1-rural (water and dust-like), 4-maritime (rural and sea salt), 5-urban (rural and soot-like), 6-tropospheric (rural mixture)"
		return (None)
	
	GOmod, taug, g = _ghc_solis2_calc_coeffs(G0, aod, w, alt, atm_type)
	
	if sinh0 is None:
		sinh0=sin(h0)
	
	GHIc=GOmod*exp(taug/(sinh0**g))*sinh0
	return (GHIc)

def ghc_solis2_2atmospheres(G0, aod, w, h0, alt, atm_primary_type=5, atm_secondary_type=1, atm_secondary_weight=0.5, sinh0=None):
	if (atm_primary_type not in [1,4,5,6]) or (atm_secondary_type not in [1,4,5,6]):
		print "atmosphere type must be one of: 1-rural (water and dust-like), 4-maritime (rural and sea salt), 5-urban (rural and soot-like), 6-tropospheric (rural mixture)"
		return (None)
	
	if sinh0 is None:
		sinh0=sin(h0)

	GOmod, taug, g = _ghc_solis2_calc_coeffs(G0, aod, w, alt, atm_primary_type)
	GHIc_p=GOmod*exp(taug/(sinh0**g))*sinh0
	GOmod, taug, g = _ghc_solis2_calc_coeffs(G0, aod, w, alt, atm_secondary_type)
	GHIc_s=GOmod*exp(taug/(sinh0**g))*sinh0
	
	GHIc = GHIc_p + (GHIc_s-GHIc_p)*atm_secondary_weight
	return (GHIc)


def _ghc_solis2_calc_coeffs(G0, aod, w, alt, atm_type):
	pp0=exp(-alt/8427.744)
	logpp0 = log(pp0)
	logw = log(w)

	#calculate GOmodified 
	if atm_type==5:
		aux1=((0.0122*pp0-0.0066)*w)+(0.0793*pp0)+1.0078
		thresh=3.14158*pp0**4.42
		if (w<0.66*thresh):
			GOmod=G0*aux1*exp((0.0616*log(w)+0.798)*aod)
		elif(w<thresh):
			GOmod=G0*aux1*exp((0.0596*log(w)+0.964)*aod)
		else:
			GOmod=1323.*aux1*exp((0.0575*log(w)+1.13)*aod)
			
		a,b,c,d,e,f,g,h=(-1.2392562954263, -0.0497518860970, -0.0095919495011, -0.2675425937758, -0.0440464168321, -0.0084009774214, -0.0112600602564, -0.0963623389601)
		taug=(a+b*logw+c*(logw**2))*aod+(d+e*logw+f*(logw**2))+(g*w+h)*logpp0
		a,b,c,d=(-0.0297037887554, 0.0224753926528, -0.0369893129928, 0.3768602213906)
		g=a*logw+(b*aod*aod+c*aod+d)
	
	elif atm_type==1:
		a,b,c,d,e,f,g=(0.346936572622, 0.566025211173, 1.190442276874, 0.037307833086, 1.078558169864, 0.008196228708, 0.040823530783)
		GOmod=G0*((a*(w**b)*aod*aod) + (c*(w**d)*aod) + (e*(w**f))) + g*logpp0
		a,b,c,d,e,f,g,h=(-1.1895361526998, -0.0617539626603, -0.0104823258746, -0.2690060904092, -0.0438638370950, -0.0083421824470, -0.0116564521232, -0.0912150376486)
		taug=(a+b*logw+c*(logw**2))*aod+(d+e*logw+f*(logw**2))+(g*w+h)*logpp0
		a,b,c,d=(-0.0184556815893, 0.2444853746423, -0.2477170865282, 0.3748303633989)
		g=a*logw+(b*aod*aod+c*aod+d)

	elif atm_type==4:
		a,b,c,d,e,f,g=(0.844502913626, 0.409294156690, 1.288348700846, 0.010128228278, 1.074766404757, 0.010343232268, 0.026523210477)
		GOmod=G0*((a*(w**b)*aod*aod) + (c*(w**d)*aod) + (e*(w**f))) + g*logpp0
		a,b,c,d,e,f,g,h=(-1.4167651946173, -0.0854302012429, -0.0098701659301, -0.2615934974993, -0.0454507769295, -0.0079300401258, -0.0150148671624, -0.0677716871532)
		taug=(a+b*logw+c*(logw**2))*aod+(d+e*logw+f*(logw**2))+(g*w+h)*logpp0
		a,b,c,d=(-0.0174079481645, 0.3849943723180, -0.3906598704130, 0.3737632250464)
		g=a*logw+(b*aod*aod+c*aod+d)

	elif atm_type==6:
		a,b,c,d,e,f,g=(0.271428115990, 0.613663628138, 1.180630668845, 0.043082659710, 1.079283629229, 0.007716214650, 0.044034054965)
		GOmod=G0*((a*(w**b)*aod*aod) + (c*(w**d)*aod) + (e*(w**f))) + g*logpp0
		a,b,c,d,e,f,g,h=(-1.1346416406548, -0.0625413225274, -0.0110028394690, -0.2706357043070, -0.0436166481587, -0.0084126898283, -0.0110354930711, -0.0933361159933)
		taug=(a+b*logw+c*(logw**2))*aod+(d+e*logw+f*(logw**2))+(g*w+h)*logpp0
		a,b,c,d=(-0.0179629517066, 0.2275753317886, -0.2398730087433, 0.3743321192667)
		g=a*logw+(b*aod*aod+c*aod+d)

	return [GOmod, taug, g]

# beam normal clear sky radiation - Solis - Ineichen 2008 -from excel sheet
# G0- extraterestial solar irradiance (epsilon corrected)
# aod - broadband
# w - precipitable water
# alt - altitude
# h0 - sun height angle
# atmosphere type (1, 4, 5, 6)
# sinh0 can be used instead of h0 - to speed up processing, in such case let h0 be None value it won't be used (but must be specified e.g. None)
def bnc_solis2(G0, aod, w, h0, alt, atm_type=5, sinh0=None):
	if not(atm_type in [1,4,5,6]):
		print "atmosphere type must be one of: 1-rural (water and dust-like), 4-maritime (rural and sea salt), 5-urban (rural and soot-like), 6-tropospheric (rural mixture)"
		return (None)
	
	GOmod, taub, b = _bnc_solis2_calc_coeffs(G0, aod, w, alt, atm_type)
	
	#print "G0mod",GOmod, "taub",taub, "bb:",b
	if sinh0 is None:
		sinh0=sin(h0)
		
	BNIc=GOmod*exp(taub/pow(sinh0,b))
	BHIc=BNIc*sinh0
	return (BNIc, BHIc)

def bnc_solis2_2atmospheres(G0, aod, w, h0, alt, atm_primary_type=5, atm_secondary_type=1, atm_secondary_weight=0.5, sinh0=None):
	if (atm_primary_type not in [1,4,5,6]) or (atm_secondary_type not in [1,4,5,6]):
		print "atmosphere type must be one of: 1-rural (water and dust-like), 4-maritime (rural and sea salt), 5-urban (rural and soot-like), 6-tropospheric (rural mixture)"
		return (None)
	
	if sinh0 is None:
		sinh0=sin(h0)

	GOmod, taub, b = _bnc_solis2_calc_coeffs(G0, aod, w, alt, atm_primary_type)
	BNIc_p=GOmod*exp(taub/pow(sinh0,b))
	GOmod, taub, b = _bnc_solis2_calc_coeffs(G0, aod, w, alt, atm_secondary_type)
	BNIc_s=GOmod*exp(taub/pow(sinh0,b))
	
	BNIc = BNIc_p + (BNIc_s-BNIc_p)*atm_secondary_weight
	BHIc=BNIc*sinh0
	return (BNIc, BHIc)



def _bnc_solis2_calc_coeffs(G0, aod, w, alt, atm_type):
	pp0=exp(-alt/8427.744)
	logpp0 = log(pp0)
	logw = log(w)
	
	#calculate GOmodified 
	if atm_type==5:
		aux1=((0.0122*pp0-0.0066)*w)+(0.0793*pp0)+1.0078
		thresh=3.14158*pp0**4.42
		if (w<0.66*thresh):
			GOmod=G0*aux1*exp((0.0616*logw+0.798)*aod)
		elif(w<thresh):
			GOmod=G0*aux1*exp((0.0596*logw+0.964)*aod)
		else:
			GOmod=1323.*aux1*exp((0.0575*logw+1.13)*aod)

		a,b,c,d,e,f,g,h=(-1.965341198054770, -0.163710558004316, -0.022209228926442, -0.323849305498880, -0.050394041722800, -0.008204573001310, -0.022032552519559, -0.058466288111591)
		taub=(a+b*logw+c*(logw**2))*aod+(d+e*logw+f*(logw**2))+(g*w+h)*logpp0
		a,b,c,d, e, f=(0.077891231817823, -0.041877632298790, -0.015090695358747, -0.622091161045139, 0.419010706744548, 0.448656283106022)
		b=((a*aod*aod)+(b*aod)+c)*logw+((d*aod*aod)+(e*aod)+f)
	
	elif atm_type==1:
		a,b,c,d,e,f,g=(0.346936572622, 0.566025211173, 1.190442276874, 0.037307833086, 1.078558169864, 0.008196228708, 0.040823530783)
		GOmod=G0*((a*pow(w,b)*aod*aod) + (c*pow(w,d)*aod) + (e*pow(w,f))) + g*logpp0
		a,b,c,d,e,f,g,h=(-2.059941216979620, -0.107447214120998, -0.020896698392603, -0.324224016657897, -0.047426391848129, -0.008419440679385, -0.012007353196091, -0.100322560148726)
		taub=(a+b*logw+c*(logw**2))*aod+(d+e*logw+f*(logw**2))+(g*w+h)*logpp0
		a,b,c,d, e, f=(0.033949802603157, 0.002688095210234, -0.014528362196273, -0.484417695451910, 0.283659044782658, 0.455207477988204)
		b=((a*aod*aod)+(b*aod)+c)*logw+((d*aod*aod)+(e*aod)+f)
		
	elif atm_type==4:
		a,b,c,d,e,f,g=(0.844502913626, 0.409294156690, 1.288348700846, 0.010128228278, 1.074766404757, 0.010343232268, 0.026523210477)
		GOmod=G0*((a*(w**b)*aod*aod) + (c*(w**d)*aod) + (e*(w**f))) + g*logpp0
		a,b,c,d,e,f,g,h=(-2.5284409186830, -0.1301856537840, -0.0220143218116, -0.3090892922069, -0.0499572528191, -0.0077600188947, -0.0146239529283, -0.0649677195734)
		taub=(a+b*logw+c*(logw**2))*aod+(d+e*logw+f*(logw**2))+(g*w+h)*logpp0
		a,b,c,d, e, f=(0.0208175348892, 0.0085593776689, -0.0153671301864, -0.4729558849757, 0.3542509094000, 0.4551284274735)
		b=((a*aod*aod)+(b*aod)+c)*logw+((d*aod*aod)+(e*aod)+f)

	elif atm_type==6:
		a,b,c,d,e,f,g=(0.271428115990, 0.613663628138, 1.180630668845, 0.043082659710, 1.079283629229, 0.007716214650, 0.044034054965)
		GOmod=G0*((a*(w**b)*aod*aod) + (c*(w**d)*aod) + (e*(w**f))) + g*logpp0
		a,b,c,d,e,f,g,h=(-1.9723411375458, -0.1071330385566, -0.0213875080714, -0.3273104140121, -0.0471306487031, -0.0082429582869, -0.0114060164565, -0.1032214260573)
		taub=(a+b*logw+c*(logw**2))*aod+(d+e*logw+f*(logw**2))+(g*w+h)*logpp0
		a,b,c,d, e, f=(0.0241194476465, 0.0085422295745, -0.0142919988942, -0.5033663242241, 0.2572182074800, 0.4559542533391)
		b=((a*aod*aod)+(b*aod)+c)*logw+((d*aod*aod)+(e*aod)+f)

	return [GOmod, taub, b]




# beam normal clear sky radiation (Ineichen, Perez, 2002)
# G0- extraterestial solar irradiance
# TL - linke turbidity for airmass 2
# air_mass - airmass (kasten2) - elevation corrected
# alt - altitude
# h0 - sun height angle
def bnci(G0, TL, air_mass, alt, h0):
	TLcor = TL
	if TL < 2.:
		TLcor = TL - 0.25 * ((2. - TL)**0.5)
	
	fh1=exp(-alt/8000.)
	b=0.664 + 0.163/fh1
	BNIc = b * G0 * exp(-0.09 * air_mass * (TLcor - 1.))
	if TL < 2.:
		GHc = ghc(G0, TL, air_mass, alt, h0)
		sinh0=sin(h0)
		BNIc=min(BNIc, (GHc * (1. - (0.1 - (0.2*exp(-TL))) / (0.1 + (0.88/fh1)))/sinh0))
	return (BNIc)


# global horizotal clear sky radiation (Duerr, Zelenka, 2008)
# G0- extraterestial solar irradiance
# TL - linke turbidity for airmass 2
# AM - airmass (kasten2) - elevation corrected
# alt - altitude
# h0 - sun height angle
def ghc_duerr(G0, TL, air_mass, alt, h0):
	TLcor = TL
	if TL < 2.:
		TLcor = TL - 0.25 * ((2. - TL)**0.5)
	cg1=(0.0000174 * alt) + 0.868
	cg2=(0.00000681 * alt) + 0.0387
	fh1=exp(-alt/8000.)
	fh2=exp(-alt/1250.)
	sinh0=sin(h0)
	aux1=(-cg2)*air_mass*(fh1+(fh2*(TLcor-1.)))
	GHIc=cg1*G0*sinh0*exp(aux1)
	return (GHIc)



#clearness index 
#extr_G0h - extraterestrial horizontal irraidance 
#Gh - measured (modeled) horizontal irradiance
def clearnes_idx(extr_G0h,Gh):
	kt=0.
	if (Gh is None) or (extr_G0h is None):
		return(kt)
	if (Gh <> 0.) and (extr_G0h <> 0.):
		kt = Gh/extr_G0h
	return(kt)

#modified clrearness index (Perez, Ineichen 1990)
#extr_G0h - extraterestrial horizontal irraidance 
#Gh - measured (modeled) horizontal irradiance
#air_mass - airmass (kasten2) - elevation corrected
def clearnes_idx_modified(extr_G0h,Gh, air_mass):
	kt=0.
	if (Gh is None) or (extr_G0h is None):
		return(kt)
	if (Gh <> 0.) and (extr_G0h <> 0.):
		kt = Gh/extr_G0h
	kt_m=kt/ ((1.031*exp(-1.4/(0.9+(9.4/air_mass))))+0.1)
	return(kt_m)


# approximate linke turbidity from global horizontal clear sky radiation 
# using clear sky model (Ineichen, Perez, 2002)
# Gh - global horizontal irradiance (cloudless) used to assess the TL
# G0- extraterestial solar irradiance
# AM - airmass (kasten2) - elevation corrected
# alt - altitude
# h0 - sun height angle
def Gh2TL(Gh, G0, air_mass, alt, h0):
	cg1=(0.0000509 * alt) + 0.868
	cg2=(0.0000392 * alt) + 0.0387
	fh1=exp(-alt/8000.)
	fh2=exp(-alt/1250.)
	sinh0=sin(h0)
#	aux0=(-cg2)*air_mass
	aux1=cg1*G0*sinh0
	TL_res=1-(((log(Gh/aux1)/(cg2*air_mass))+fh1)/fh2)
	#TL_res <2.0 should be atlered	bcakward TLres = TL - 0.25 * ((2. - TL)**0.5)
	return (TL_res)


# approximate linke turbidity from global horizontal clear sky radiation 
# using clear sky model (Ineichen, Perez, 2002) with parameters from (Duerr, Zelenka, 2008)
# Gh - global horizontal irradiance (cloudless) used to assess the TL
# G0- extraterestial solar irradiance
# AM - airmass (kasten2) - elevation corrected
# alt - altitude
# h0 - sun height angle
def Gh2TL_duerr(Gh, G0, air_mass, alt, h0):
	cg1=(0.0000174 * alt) + 0.868
	cg2=(0.00000681 * alt) + 0.0387
	fh1=exp(-alt/8000.)
	fh2=exp(-alt/1250.)
	sinh0=sin(h0)
#	aux0=(-cg2)*air_mass
	aux1=cg1*G0*sinh0
	TL_res=1-(((log(Gh/aux1)/(cg2*air_mass))+fh1)/fh2)
	#TL_res <2.0 should be atlered	bcakward TLres = TL - 0.25 * ((2. - TL)**0.5)
	return (TL_res)

# approximate linke turbidity from Beam normal clear sky radiation
# formula from (Ineichen, Perez, 2002)
# Bh - global horizontal irradiance (cloudless) used to assess the TL
# G0- extraterestial solar irradiance
# AM - airmass (kasten2) - elevation corrected
# alt - altitude
def Bn2TL(Bn, G0, air_mass, alt):
	fh1=exp(-alt/8000.)
	b=0.664 + (0.163/fh1)
	TL=((11.1*log(b*G0/Bn))/air_mass)+1.
	return (TL)

#linke turbidity form broadband AOD, w, altitude, for atmosphete_type
#from Ineichen 2008, Solar energy 82, 1095-1097
#inputs:
#broadband AOD
#precipitable water column in cm
#altitude
#atmosphere type (1, 4, 5, 6)
def aod2TL(aod,w,alt,atm_type):
	if ((atm_type != 1) and (atm_type != 4) and (atm_type != 5) and (atm_type != 6)):
		print "atmosphere typa must be one of: 1-rural (water and dust-like), 4-maritime (rural and sea salt), 5-urban (rural and soot-like), 6-tropospheric (rural mixture)"
		return (None)
	pp0=(1.-(alt/10000.))
	coefs=[[],[],[],[],[],[],[]]
	coefs[1]=[5.579739734261890, 0.645333991839355, 0.388121452484152, 2.020514217895010, 0.565107645758485, 0.622907263667422, 0.232678347478149]
	coefs[4]=[6.333375731360800, 0.739960955274041, 0.419832729083074, 2.896048313107980, -1.478814440745530, -0.783876449868698, -0.086248595313049]
	coefs[5]=[5.694191652374560, 0.652759736175906, 0.385367617959831, 1.870720942772150, 0.821528259255109, 0.755161500809153, 0.249247215466936]
	coefs[6]=[5.228308000833870, 0.646047664060631, 0.396006037341279, 2.079262744057610, 0.473922342979583, 0.554257911461324, 0.217511901703177]
	a, b, c, d, e, f, g = coefs[atm_type]
	TL2=((a*exp(b/pp0))*aod)+c*log(w)+(d+e/pp0-f/(pp0*pp0)+g/(pp0*pp0*pp0))
	return (TL2)

def TL2aod(TL2,w,alt,atm_type):
	if ((atm_type != 1) and (atm_type != 4) and (atm_type != 5) and (atm_type != 6)):
		print "atmosphere typa must be one of: 1-rural (water and dust-like), 4-maritime (rural and sea salt), 5-urban (rural and soot-like), 6-tropospheric (rural mixture)"
		return (None)
	pp0=(1.-(alt/10000.))
	coefs=[[],[],[],[],[],[],[]]
	coefs[1]=[5.579739734261890, 0.645333991839355, 0.388121452484152, 2.020514217895010, 0.565107645758485, 0.622907263667422, 0.232678347478149]
	coefs[4]=[6.333375731360800, 0.739960955274041, 0.419832729083074, 2.896048313107980, -1.478814440745530, -0.783876449868698, -0.086248595313049]
	coefs[5]=[5.694191652374560, 0.652759736175906, 0.385367617959831, 1.870720942772150, 0.821528259255109, 0.755161500809153, 0.249247215466936]
	coefs[6]=[5.228308000833870, 0.646047664060631, 0.396006037341279, 2.079262744057610, 0.473922342979583, 0.554257911461324, 0.217511901703177]
	a, b, c, d, e, f, g = coefs[atm_type]
	aod=(TL2-(c*log(w))-(d+e/pp0-f/(pp0*pp0)+g/(pp0*pp0*pp0))) / (a*exp(b/pp0))
	return (aod)

#Calculate PAR radiation from GHI
#Alados Olmo Foyo-Moreno Alados-Arboledas 2000 Estimation of PAR under cloudy conditions. Agricultural and Forest Meteorology
# Implemented in: Rubio Lopez Tovar Pozo Batlles 2005 The use of satellite measurements to estimate photosynthetically active radiation. Phusics and gemistry of the Earth
# Implemented in Meteonorm 6
#Gh 
#kt - clearness index
#h0_r - sun height angle (in radians)
#Qp - PAR in W/m2 
#PPFD in photosynthetic photon flux (area) density (PPFD)  micro mol *m-2 *s-1
def Gh2PAR(Gh,kt,h0_r, PPFD=True):
	if kt==0:
		return 0
	PPFD_val=Gh*(1.832 -(0.191*log(kt)) + (0.099*cos(h0_r)))
#	print Gh, kt, h0_r, (0.191*log(kt)),  (0.099*cos(h0_r)), PPFD_val, (1.832 -(0.191*log(kt)) + (0.099*cos(h0_r)))
	if PPFD:
		return PPFD_val
	else:
		return PPFD_val/4.6 #convert to W/m2   # this applies for solar spectra


# diffinc_Perez
#     PROGRAMMED BY:    HOWARD M. BISNER, FOLLOWING R. PEREZ
#						Pythonized and Twisted by SKOCZEK
#     ARGUMENTS:    G - GLOBAL IRRADIANCE (WATTS / SQ. METER)
#                   B - DIRECT IRRADIANCE (WATTS / SQ. METER)
#					derrived from DiffHor, BOc, BHc
#                   Z - SOLAR ZENITH ANGLE (RADIANS) = (pi/2-h0)
#					derrived from h0
#               SLOPE - SURFACE'S SLOPE (RADIANS)
#					derrived from sin and cos cosGammaN, sinGammaN
#              Albedo - GROUND Albedo
#                 INC - SOLAR INCIDANCE ANGLE ON SURFACE (RADIANS)
#					derrived from sinDeltaexp
#     RETURNS:  IRRPZ - TILTED DIFFUSE PLUS REFLECTED IRRADIANCE (WATTS / SQ. METER)
#                       -        CORRECTION ON 11/2/00: SECOND B VARIABLE WAS RENAMED BR
#                                AFTER OBSERVATION FROM W.MILLER AT OAK-RIDGE NTL.LAB
#                       -       ON 5/25/04 REPLACED Z BY MAX(Z,0.4) TO AVOID NON-
#                               SUN-FACING DISTORSION FOR VERY LOW ZENITH ANGLES
#                              AFTER OBSERVATION BY MR. ABIEL DERIZANS
#FUNCTION IRRPZ( G, B, Z, SLOPE, Albedo, INC )
def diffinc_Perez(DiffHorInput, B0c, BHc, h0, sinGammaN,cosGammaN, Albedo, sinDeltaexpInput,shadow):
	'''
	original Perez model extended by interpolation between bins to provide much smoother results
	slower version - not used in real conditions
	parameters described in diffinc_PerezFast
	'''
	
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
#	DiffInc= numpy.zeros(h0.shape)
#	ReflInc= numpy.zeros(h0.shape)
	
	B2 = 5.534e-6
	
	CZ = numpy.cos(Z)
	ZENITH = numpy.degrees(Z)
	ZH[CZ<0.0871557]=0.0871557
	ZH[CZ>0.0871557]=CZ[CZ>0.0871557]
	
	DiffHor[CZ > 0.0] = G[CZ > 0.0] - B0c[CZ > 0.0]* CZ[CZ > 0.0]
	DiffHor[CZ <= 0.0] = G[CZ <= 0.0]

	AIRMASS[ZENITH<93.9]=1.0/(CZ[ZENITH<93.9] + 0.15* numpy.power((93.9 - ZENITH[ZENITH<93.9]),-1.253))
	AIRMASS[ZENITH>=93.9]=999
	
	DELTA = DiffHor * AIRMASS / I0 #Solar constant
	TB2 = B2*numpy.power(ZENITH,3)
	
	EPS = (B0c + DiffHor) / DiffHor
	EPS = (EPS + TB2) / (1.0 + TB2)
		
	#________Insane interpolation of PEREZ bins (bin's centers)________________________________________________________________
	I= numpy.array(DiffHor*0+6, dtype=int)			
	I[EPS<=(EPSBINS[6]+EPSBINS[5])/2]=5
	I[EPS<=(EPSBINS[5]+EPSBINS[4])/2]=4
	I[EPS<=(EPSBINS[4]+EPSBINS[3])/2]=3
	I[EPS<=(EPSBINS[3]+EPSBINS[2])/2]=2
	I[EPS<=(EPSBINS[2]+EPSBINS[1])/2]=1
	I[EPS<=(EPSBINS[1]+EPSBINS[0])/2]=0

	MAXZ[Z<0.4]=0.4 
	MAXZ[Z>=0.4]=Z[Z>=0.4]
	
	Im1=I-1
	Ip1=I+1
		
	BinWith=(EPSBINS[Ip1]+EPSBINS[I])/2-(EPSBINS[I]+EPSBINS[Im1])/2
	DistanceFromBinEdge=EPS-(EPSBINS[I]+EPSBINS[Im1])/2
	BinSlope=DistanceFromBinEdge/BinWith
			
	F11RIp1=F11R[Ip1] 
	F11RIm1=F11R[Im1] 
	F11RI=F11R[I] 
			
	F12RIp1=F12R[Ip1] 
	F12RIm1=F12R[Im1] 
	F12RI=F12R[I] 
			
	F12RIp1DELTA=F12RIp1 * DELTA
	F12RIDELTA=F12RI * DELTA
	F12RIm1DELTA=F12RIm1 * DELTA
			
	F13RIp1=F13R[Ip1] 
	F13RIm1=F13R[Im1] 
	F13RI=F13R[I]
			
	F13RIp1MAXZ=F13RIp1*MAXZ
	F13RIm1MAXZ=F13RIm1*MAXZ
	F13RIMAXZ=F13RI*MAXZ
			
	F21RIp1=F21R[Ip1] 
	F21RIm1=F21R[Im1] 
	F21RI=F21R[I]
			
	F22RIp1=F22R[Ip1] 
	F22RIm1=F22R[Im1] 
	F22RI=F22R[I]
			
	F22RIp1DELTA=F22RIp1 * DELTA
	F22RIDELTA=F22RI * DELTA
	F22RIm1DELTA=F22RIm1 * DELTA
			
	F23RIp1=F23R[Ip1] 
	F23RIm1=F23R[Im1] 
	F23RI=F23R[I]
			
	F23RIp1MAXZ=F23RIp1*MAXZ
	F23RIm1MAXZ=F23RIm1*MAXZ
	F23RIMAXZ=F23RI*MAXZ
				
	F1t = (F11RIp1 + F12RIp1DELTA + F13RIp1MAXZ - F11RIm1 - F12RIm1DELTA - F13RIm1MAXZ)/2
	F2t = (F21RIp1 + F22RIp1DELTA + F23RIp1MAXZ - F21RIm1 - F22RIm1DELTA - F23RIm1MAXZ)/2
			
	F1=(F11RI + F12RIDELTA + F13RIMAXZ +F11RIm1 + F12RIm1DELTA + F13RIm1MAXZ)/2+BinSlope*F1t
	F2=(F21RI + F22RIDELTA + F23RIMAXZ +F21RIm1 + F22RIm1DELTA + F23RIm1MAXZ)/2 + BinSlope*F2t
	#F1orig = max(0.0,F11R[I] + F12R[I]*DELTA + F13R[I] * MAXZ)
	#F2orig = F21R[I] + F22R[I] * DELTA + F23R[I] * MAXZ
	
	ALBPROD = Albedo * G
	sinDeltaexp[sinDeltaexp< 0.0]=0.0
	A = DiffHor * ( 1.0 + cosGammaN) / 2.0
	BR = (sinDeltaexp/ ZH) * DiffHor - A 
	C = DiffHor * sinGammaN
	ReflInc= ALBPROD * (1.0 - cosGammaN)/2.0
	DiffInc = A + (1-shadow)*F1 * BR + F2 * C
	return DiffInc,ReflInc


def diffinc_PerezFast_SkyFraction(DiffHorInput, B0c, BHc, h0, sinGammaN,cosGammaN, Albedo, sinDeltaexpInput, shadow, SkyViewFracton=None, sinh0=None):
	'''
	optional:
	sinh0 - if not specified - calculated from h0 or Z
	shadow_an  - an angle (in radians) of near horizon used to reduce isotropic diffuse by part of shading from previous row 
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


	if SkyViewFracton is None:
		SkyViewFracton=1;

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
#	DiffInc= numpy.zeros(h0.shape)
#	ReflInc= numpy.zeros(h0.shape)
	
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
	
#	I= numpy.array(DiffHor*0+6, dtype=int) #original by artur	
	I= numpy.array(DiffHor*0+7, dtype=int) #updated by tomas 20130701
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
	
	A = DiffHor * ( 1.0 + cosGammaN) / 2.0  * SkyViewFracton #isotropic original * sky view fraction coefficient
#	A = DiffHor * ( 1.0 + cosGammaN) / 2.0  #isotropic original
	BR = ((sinDeltaexp/ ZH) * DiffHor) - A  #circumsolar
	C =  DiffHor * sinGammaN # horizontal band
	DiffInc = A + ((1-shadow)*F1 * BR) + (F2 * C) #diffuse inclined
	return DiffInc, ReflInc




def diffinc_PerezFast(DiffHorInput, B0c, BHc, h0, sinGammaN, cosGammaN, Albedo, sinDeltaexpInput, shadow=0.0, shadow_an=None, sinh0=None, isotropic_shading_factor=1.0):
	'''
	DiffHorInput - diffuse horizontal (VECTOR)
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


	if shadow_an is None:
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
#	DiffInc= numpy.zeros(h0.shape)
#	ReflInc= numpy.zeros(h0.shape)
	
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
	
#	I= numpy.array(DiffHor*0+6, dtype=int) #original by artur	
#	I= numpy.array(DiffHor*0+7, dtype=int) #updated by tomas 20130701
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
#	A = DiffHor * ( 1.0 + cosGammaN*cosRshadow-sinGammaN*sinRshadow) / 2.0; #isotropic adapted for shading
##	A = DiffHor * ( 1.0 + cosGammaN) / 2.0  #isotropic original
#	BR = ((sinDeltaexp/ ZH) * DiffHor) - A  #circumsolar
#	C =  DiffHor * sinGammaN # horizontal band
#	DiffInc = A + ((1-shadow)*F1 * BR) + (F2 * C) #diffuse inclined
#	return DiffInc,ReflInc
#
#	#working wersions 20140423
#
#	print 'current version' 
#	A = DiffHor * ( 1.0 + cosGammaN) / 2.0
#	BR = (sinDeltaexp/ ZH) * DiffHor-A
#	C = DiffHor * sinGammaN
#	DiffInc = A + F1 * BR*(1-shadow) + F2 * C
#	return DiffInc,ReflInc
#	
#
#	print 'new simple version - correct circumsolar' 
#	isotropic = (( 1.0 + cosGammaN) / 2.0) * (1-F1)
#	circumsolar = (sinDeltaexp/ ZH) * F1 * (1-shadow) 
#	horizon = sinGammaN * F2
#	DiffInc = DiffHor * (isotropic + circumsolar + horizon) 
#	return DiffInc,ReflInc	

#	

#	print 'new version - correct circumsolar + isotropic' 
#	isotropic = (( 1.0 + cosGammaN) / 2.0) * (1-F1) * isotropic_shading_factor 
#	circumsolar = (sinDeltaexp/ ZH) * F1 * (1-shadow) 
#	horizon = sinGammaN * F2
#	DiffInc = DiffHor * (isotropic + circumsolar + horizon) 
#	return DiffInc,ReflInc	

#	print 'new version - correct circumsolar + isotropic + reflected' 
	isotropic = (DiffHor * (( 1.0 + cosGammaN) / 2.0) * (1-F1)  * isotropic_shading_factor)  + ((1. - isotropic_shading_factor) *ALBPROD)
	circumsolar = DiffHor * (sinDeltaexp/ ZH) * F1 * (1-shadow) 
	horizon = DiffHor * sinGammaN * F2
	DiffInc = isotropic + circumsolar + horizon

	return DiffInc,ReflInc	




def diffinc_PerezFast2_old(DiffHorInput, B0c, BHc, h0,sinh0, sinGammaN,cosGammaN, Albedo, sinDeltaexpInput,shadow,shadow_an):
	DiffHor=DiffHorInput.copy()
	sinDeltaexp=sinDeltaexpInput.copy()
	if shadow_an is None:
		shadow_an=numpy.zeros_like(B0c)

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
	
	B2 = 5.534e-6
	
	CZ = sinh0
	ZENITH = 180./pi* Z
	ZH[CZ<0.0871557]=0.0871557
	ZH[CZ>0.0871557]=CZ[CZ>0.0871557]

	AIRMASS[ZENITH<93.9]=1.0/(CZ[ZENITH<93.9] + 0.15* numpy.power((93.9 - ZENITH[ZENITH<93.9]),-1.253))
	
	AIRMASS[ZENITH>=93.9]=999
	DELTA = DiffHor * AIRMASS / 1367. #Solar constant
	TB2 = B2*numpy.power(ZENITH,3)
	EPS = (B0c + DiffHor) / DiffHor
	EPS = (EPS + TB2) / (1.0 + TB2)
	
	MAXZ[Z<0.4]=0.4 
	MAXZ[Z>=0.4]=Z[Z>=0.4]
	
	I= numpy.array(DiffHor*0+7, dtype=int)	
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
	
	cosRshadow=numpy.cos(shadow_an);
	sinRshadow=numpy.sin(shadow_an);

	ReflInc= ALBPROD * (1.0 - cosGammaN)/2.0

	A = DiffHor * ( 1.0 + cosGammaN*cosRshadow-sinGammaN*sinRshadow) / 2.0;
	#sinDeltaexp[sinDeltaexp< 0.0]=0.0
	#A = DiffHor * ( 1.0 + cosGammaN) / 2.0
	BR = (sinDeltaexp/ ZH) * DiffHor-A
	C = DiffHor * sinGammaN
	DiffInc = A + F1 * BR*(1-shadow) + F2 * C
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
def PV_Angular(Binc,DiffInc,Refl,sinDeltaexp,GammaN_r,sinGammaN,cosGammaN,ar=None):
	# Beam component
	# ar: the parameter ranging from 0.14 - 0.17 according to the module type
	if ar is None:
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
	return B,Diff,R



def test_sun_geom():
	d="20100101"
	longi,lati=90.1,37.5120278
	year,month,day=(daytimeconv.yyyymmdd2ymd(d))
	doy=daytimeconv.yyyymmdd2doy(d)
	decli=declin_r(year,doy,radians(longi))
	print 'declination:',decli,'rad  ', declin_d(year,doy,longi),'deg'

	for L in range (0,48):
		LAT=L/2.
		a0,h0=sunposition_r(decli,radians(lati),LAT)
		print 'local solar time:',LAT, '   a0 (rad):', a0, 'h0 (rad):', h0
	
	
# Test
if __name__ == "__main__":

	a,h,sa,ca,sh,ch = diffuse_isotropic_shading_meshgrids()
	import pylab
	pylab.imshow()
	pylab.colorbar()
	pylab.show()
	exit()
#	test_sun_geom()
#	exit()
	
	d="20120101"
	longi,lati=21.233333, -28.433333
	elev=119.
	atm_type=1
	aod=0.25
	w=1.86967025757
#	print "aod",aod, ", w", w, ", elev", elev, ", atm_type", atm_type
#	TL2=aod2TL(aod,w,elev, atm_type)
#	print "TL2 from aod", (TL2)
#	print "aod from TL", (TL2aod(TL2, w, elev, atm_type))
	
	time=12.04
	year,month,day=(daytimeconv.yyyymmdd2ymd(d))
	doy=daytimeconv.yyyymmdd2doy(d)
	print 'lon:',longi,'lat:',lati,'elevation:',elev
	print 'time:',time
	print 'day:',day,'month:',month,'year:',year,'doy:',doy
	epsil=epsilon_cor(doy)
	print 'epsilon:',epsil
	print 'dayangle1 (Heliosat):',degrees(dayangle_r(doy)),'dayangle2 (Gruter):',degrees(dayangle_r2(doy))
	decli=declin_r(year,doy,radians(longi))
	print 'declination:',decli,'rad  ', declin_d(year,doy,longi),'deg'
	print 'perturbation:',perturbation(doy), ' perturbation(min)',perturbation(doy)*60.
	LAT = UTC2LAT_r(time,doy,radians(longi))
	UTC = LAT2UTC_r(LAT,doy,radians(longi))
	print 'UTC:',UTC
	print 'local solar time:',LAT
	a0,h0=sunposition_r(decli,radians(lati),LAT)
	print 'a0 (rad):', a0, 'h0 (rad):', h0
	a0,h0refr=sunpositionrefr(decli,radians(lati),LAT)
	z0 = (pi/2) - h0
	print 'a0:',degrees(a0),'z0:',degrees(z0),'h0:',degrees(h0),'h0refr:',degrees(h0refr)
	srss=srss_r(radians(lati),decli)
	print 'sunrise,sunset (rad):',srss
	print 'sunrise,sunset (hour):',srss_r2srss_t(srss[0],srss[1])
	print 'm0,m:',opt_air_mass(elev,h0refr)
	G0=extr_G0(epsil)
	print 'G0',extr_G0(epsil)
	print 'G0h',extr_G0h(epsil,h0refr)
	print 'GHI_solis_sat', ghc_solis_sat(G0, aod, w, h0refr, elev)
	print 'GHI_solis', ghc_solis(G0, aod, w, h0refr, elev)
	print 'GHI_solis2', ghc_solis2(G0, aod, w, h0refr, elev,atm_type)
	print 'BNI_solis2', bnc_solis2(G0, aod, w, h0refr, elev,atm_type)
	print diffinc_PerezFast(DiffHorInput=0, B0c=0, BHc=0, h0=h0, sinGammaN=0,cosGammaN=0, Albedo=0.11, sinDeltaexpInput=0, shadow=0 )
	am=(opt_air_mass_kasten2(elev,h0refr))[1]
#	print 'GHI_sat', ghc_sat(G0, TL2, am, elev, h0refr)
#	print 'GHI', ghc(G0, TL2, am, elev, h0refr)
