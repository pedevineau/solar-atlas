'''
MODULE: daytimeconv:
+Function: archseg2doyminmax    - Arguments: ['aseg', 'year']--Default arguments: [('year', 2004)]
+Function: archsegy2dfbminmax    - Arguments: ['aseg', 'year']
+Function: date2archseg    - Arguments: ['dat']
+Function: date2dfb    - Arguments: ['dat']
+Function: date2doy    - Arguments: ['dat']
+Function: date2ymd    - Arguments: ['dat']
+Function: date2yyyymmdd    - Arguments: ['dat']
+Function: dd2dms    - Arguments: ['dd']
+Function: dfb2archseg    - Arguments: ['dfb']
+Function: dfb2date    - Arguments: ['dfb']
+Function: dfb2doy    - Arguments: ['dfb']
+Function: dfb2doyy    - Arguments: ['dfb']
+Function: dfb2ymd    - Arguments: ['dayfrombeg']
+Function: dfb2yyyymmdd    - Arguments: ['dayfrombeg']
+Function: dfbdh_offset    - Arguments: ['dfb', 'dh', 'hoffset']
+Function: dh2hms    - Arguments: ['dh']
+Function: dh2time    - Arguments: ['dh']
+Function: dh2hms_old    - Arguments: ['dh']
+Function: dh_offset    - Arguments: ['dh', 'hoffset']
+Function: dh2slot    - Arguments: ['dh', ' 'MSG'']
+Function: dms2dd    - Arguments: ['d', 'm', 's']
+Function: doy2archseg    - Arguments: ['d']
+Function: doyy2date    - Arguments: ['yearday', 'year']
+Function: doyy2dfb    - Arguments: ['yearday', 'year']
+Function: doyy2md    - Arguments: ['yearday', 'year']
+Function: doyy2yyyymmdd    - Arguments: ['yearday', 'year']
+Function: hh_mm_ss2time    - Arguments: ['hh_mm_ss']
+Function: hhmm2hm    - Arguments: ['hhmm']
+Function: hhmm2slot    - Arguments: ['hhmm', 'MSG']--Default arguments: [('MSG', True)]
+Function: hhmm2time    - Arguments: ['hhmm']
+Function: hm2hhmm    - Arguments: ['hour', 'minut']
+Function: hm2slot    - Arguments: ['hour', 'minut', 'MSG']--Default arguments: [('MSG', True)]
+Function: hm2time    - Arguments: ['h', 'm']
+Function: hms2dh    - Arguments: ['h', 'm', 's']
+Function: hms2time    - Arguments: ['h', 'm', 's']
+Function: latlon2formatedDms    - Arguments: ['latDD', 'lonDD']
+Function: leapyear    - Arguments: ['year']--Default arguments: [('year', 1500)]
+Function: month_midday    - Arguments: ['month', 'year']--Default arguments: [('year', 2005)]
+Function: month_middoy    - Arguments: ['month', 'year']--Default arguments: [('year', 2005)]
+Function: month_representday Arguments: ['month', 'year']--Default arguments: [('year', 2005)]
+Function: month_representdoy Arguments: ['month', 'year']--Default arguments: [('year', 2005)]
+Function: month_numdays    - Arguments: ['month', 'year']--Default arguments: [('year', 2005)]
+Function: notvalid_dfb    - Arguments: ['dfb']
+Function: notvalid_doyy    - Arguments: ['doy', 'year']--Default arguments: [('year', 2001)]
+Function: notvalid_hhmm    - Arguments: ['hhmm']
+Function: notvalid_hm    - Arguments: ['hour', 'minut']
+Function: notvalid_slot    - Arguments: ['msgslot', 'MSG']--Default arguments: [('MSG', True)]
+Function: notvalid_ymd    - Arguments: ['year', 'month', 'day']
+Function: notvalid_yyyymmdd    - Arguments: ['yyyymmdd']
+Function: slot2datetime    - Arguments: ['msgslot', 'aD', 'MSG']--Default arguments: [('MSG', True)]
+Function: slot2hhmm    - Arguments: ['slot', 'MSG']--Default arguments: [('MSG', True)]
+Function: slot2hm    - Arguments: ['slot', 'MSG']--Default arguments: [('MSG', True)]
+Function: slot2time    - Arguments: ['slot', 'MSG']--Default arguments: [('MSG', True)]
+Function: time2hhmm    - Arguments: ['time']
+Function: time2hm    - Arguments: ['atime']
+Function: time2slot    - Arguments: ['time', 'MSG']--Default arguments: [('MSG', True)]
+Function: ymd2archseg    - Arguments: ['y', 'm', 'd']
+Function: ymd2date    - Arguments: ['year', 'month', 'day']
+Function: ymd2dfb    - Arguments: ['year', 'month', 'day']
+Function: ymd2doy    - Arguments: ['year', 'month', 'day']
+Function: ymd2yyyymmdd    - Arguments: ['year', 'month', 'day']
+Function: yyyymmdd2archseg    - Arguments: ['yyyymmdd']
+Function: yyyymmdd2date    - Arguments: ['yyyymmdd']
+Function: yyyymmdd2dfb    - Arguments: ['yyyymmdd']
+Function: yyyymmdd2doy    - Arguments: ['yyyymmdd']
+Function: yyyymmdd2ymd    - Arguments: ['yyyymmdd']
+Function: doyy2month_interpolation_weight    - Arguments: ['doy', 'year']
+Function: dfb2month_interpolation_weight    - Arguments: ['dfb']
'''

from math import floor, fabs
import datetime


yearbegin=1980
month_days=[[0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],[0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]
month_days_avg=[0, 31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_centerdays=[[0, 16, 14, 16, 15, 16, 15, 16, 16, 15, 16, 15, 16],[0, 16, 15, 16, 15, 16, 15, 16, 16, 15, 16, 15, 16]]
month_centerdoys=[[0, 16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350],[0, 16, 46, 76, 106, 137, 167, 198, 229, 259, 290, 320, 351]]
month_representativedays=[[0, 17, 14, 16, 15, 15, 11, 17, 16, 16, 16, 15, 11],[0, 17, 14, 16, 15, 15, 11, 17, 16, 16, 16, 15, 11]]
month_representativedoys=[[0, 17, 45, 75, 105, 135, 162, 198, 228, 259, 289, 319, 345],[0, 17, 45, 76, 106, 136, 163, 199, 227, 260, 290, 320, 346]]

month_num_to_abbrev3_dict={1:"Jan",2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec", 13:"Year"}

def archseg2doyminmax(aseg, year=2004):
	as_min=(aseg-1)*61+1
	as_max=((aseg-1)*61)+61 
	if leapyear(year):
		year_doys=366
	else:
		year_doys=365
	as_max=min(year_doys,as_max)
	return (as_min, as_max)

def archsegy2dfbminmax(aseg, year):
	#arch_seg to doy conversion
	as_min, as_max = archseg2doyminmax(aseg, year)
	#doy2dfb
	as_min=doyy2dfb(as_min,year)
	as_max=doyy2dfb(as_max,year)
	return (as_min, as_max)

def monthyear2doyminmax(month, year=2004):
	doy_min = ymd2doy(year,month,1)
	day_max = month_days[leapyear(year)][month]
	doy_max = ymd2doy(year,month,day_max)
	return (doy_min, doy_max)

def monthyear2dfbminmax(month, year):
	dfb_min = ymd2dfb(year,month,1)
	day_max = month_days[leapyear(year)][month]
	dfb_max = ymd2dfb(year,month,day_max)
	return (dfb_min, dfb_max)

#date2archseg: converts date to msg archive segment (six 61 days long segments)
def date2archseg(dat):
	y=dat.year
	m=dat.month
	d=dat.day
	doy=ymd2doy(y,m,d)
	return doy2archseg(doy)

#date2dfb: converts date (python daytime format) to day from beginning
def date2dfb(dat):
	y=dat.year
	m=dat.month
	d=dat.day
	return(ymd2dfb(y,m,d))

#date2doy: converts date (python daytime format) to day of the year
def date2doy(dat):
	y=dat.year
	m=dat.month
	d=dat.day
	return(ymd2doy(y,m,d))

#date2ymd: converts date (python daytime format) to year month day
def date2ymd(dat):
	y=dat.year
	m=dat.month
	d=dat.day
	return(y,m,d)

#date2yyyymmdd: converts date (python daytime format) to yyyymmdd (string)
def date2yyyymmdd(dat,separator=''):
	y=dat.year
	m=dat.month
	d=dat.day
	return(ymd2yyyymmdd(y,m,d, separator=separator))

#decimal degrees to degrees, minutes, seconds 
def dd2dms(dd):
	dda=fabs(dd)
	d=int(floor(dda))
	res=(dda-d)*60.
	m=int(floor(res))
	s=(res-m)*60.
	if(dd<0):
		d=-d
	return(d,m,s)
	
#dfb2archseg: converts dfb to msg archive segment (six 61 days long segments)
def dfb2archseg(dfb):
	if notvalid_dfb(dfb):
		return(-1)
	doy=dfb2doy(dfb)
	return doy2archseg(doy)

#dfb2date: converts day from beginning to date (python daytime format)
def dfb2date(dfb):
	if notvalid_dfb(dfb):
		return(-1)
	year,month,day=dfb2ymd(dfb)
	year,month,day = int(year),int(month),int(day)
	d=datetime.date(year,month,day)
	return(d)

#dfb2doy: converts day from begining to day of the year
def dfb2doy(dfb):
	y,m,d=dfb2ymd(dfb)
	return(ymd2doy(y,m,d))

#dfb2doyy: converts day from begining to day of the year and year
def dfb2doyy(dfb):
	y,m,d=dfb2ymd(dfb)
	return(ymd2doy(y,m,d),y)

# dfb2ymd:  converts day from baginning to year month day form  
def dfb2ymd(dayfrombeg):
	global yearbegin
	curyear = yearbegin
	leap = leapyear(curyear)
	curyeardays = 365 + leap
	
	while (dayfrombeg > curyeardays):
		dayfrombeg = dayfrombeg - curyeardays
		curyear=curyear+1
		leap = leapyear(curyear)
		curyeardays = 365 + leap

	pyear = curyear;
	pmonthday=doyy2md(dayfrombeg, curyear);
	if (len(pmonthday)!= 2):
		return ([])
	
	pmonth,pday=pmonthday
	if(notvalid_ymd(pyear,pmonth,pday)):
		return([])
	return(pyear,pmonth,pday)
	
# dfb2yyyymmdd:  converts day from baginning to string yyyymmdd form  
def dfb2yyyymmdd(dayfrombeg,separator=''):
	global yearbegin
	curyear = yearbegin
	leap = leapyear(curyear)
	curyeardays = 365 + leap
	
	while (dayfrombeg > curyeardays):
		dayfrombeg = dayfrombeg - curyeardays
		curyear=curyear+1
		leap = leapyear(curyear)
		curyeardays = 365 + leap

	pyear = curyear;
	pmonthday=doyy2md(dayfrombeg, curyear);
	if (len(pmonthday)!= 2):
		return ("")
	
	pmonth,pday=pmonthday
	if(notvalid_ymd(pyear,pmonth,pday)):
		return("")
	return(str(pyear).zfill(4)+separator+str(pmonth).zfill(2)+separator+str(pday).zfill(2))
	
#in/de-creases time (in decimal hours) by offset , if needed the day is modified as well 
def dfbdh_offset(dfb,dh,hoffset):
	dho=dh+hoffset
	dfbo=dfb
	while(dho>=24.):
		dho=dho-24.
		dfbo=dfbo+1
	while(dho<0.):
		dho=dho+24.
		dfbo=dfbo-1
	return(dfbo,dho)


#decimal hours to hours, minutes, seconds  - rounds to closest second
def dh2hms(dh):
	if dh<0:
		dha=24 - (fabs(dh) % 24)
	else:
		dha=dh % 24
	h=int(floor(dha))
	res=(dha-h)*60.
	m=int(floor(res))
	s=int(floor(((res-m)*60.)))
	if s >= 60:
		s -= 60
		m += 1
	if m>= 60:
		m -= 60
		h += 1
	if h >= 24:
		h-=24
	return(h,m,s)

def dh2hms_old2(dh):
	dha=fabs(dh)
	h=int(floor(dha))
	res=(dha-h)*60.
	m=int(floor(res+0.5))
	s=int(floor(((res-m)*60.)+0.5))
	if s >= 60:
		s -= 60
		m += 1
	if m>= 60:
		m -= 60
		h += 1
	if h >= 24:
		h-=24
	return(h,m,s)

#decimal hours to time  - rounds to closest second
def dh2time(dh):
	h,m,s=dh2hms(dh)
	t=hms2time(h,m,s)
	return(t)

#decimal hours to hours, minutes, seconds  - rounds to closest second
def dh2hms_old(dh):
	dha=fabs(dh)
	h=int(floor(dha))
	res=(dha-h)*60.
	m=int(floor(res+0.5))
	s=int(floor(((res-m)*60.)+0.5))
	return(h,m,s)

#in/de-creases time (in decimal hours) by offset (also in decimal hours)
def dh_offset(dh,hoffset):
	dho=dh+hoffset
	while(dho>=24.):
		dho=dho-24.
	while(dho<0.):
		dho=dho+24.
	return(dho)

# decimal hours to MSG or slot number 
def dh2slot(dh, MSG=True, MFG_nominal=True):
	h,m = dh2hms(dh)[0:2]
	return(hm2slot(h, m, MSG, MFG_nominal))

#degrees,minutes, seconds to decimal degrees
def dms2dd(d,m,s):
	dd=d+(m/60.)+(s/3600.)
	return(dd)

#doy2archseg: converts doy to msg archive segment (six 61 days long segments)
def doy2archseg(d):
	i=1
	while (d>61):
		d-=61
		i+=1
	return i

#doyy2date: converts day of the year and year to  date
def doyy2date(yearday, year):
	if notvalid_doyy(yearday, year):
		return(None)
	m,d=doyy2md(yearday, year)
	return(ymd2date(year,m,d))

#doyy2dfb: converts day of the year and year to  day from beginning
def doyy2dfb(yearday, year):
	if notvalid_doyy(yearday, year):
		return(-1)
	m,d=doyy2md(yearday, year)
	return(ymd2dfb(year,m,d))

# doyy2md: converts day of the year and year (needed to decide leap year) to month and day
def doyy2md(yearday, year):
	global month_days
	leap = leapyear(year)
	i = 1
	while (yearday > month_days[leap][i]) and (i<12):
		yearday = yearday - month_days[leap][i]
		i=i+1
	pmonth = i
	pday = yearday
	
	if(notvalid_ymd(year,pmonth,pday)):
		return([])
	return(pmonth,pday)

# doyy2yyyymmdd: converts day of year and year to yyyymmdd
def doyy2yyyymmdd(yearday, year,separator=''):
	global month_days
	leap = leapyear(year)
	i = 1
	while (yearday > month_days[leap][i]) and (i<12):
		yearday = yearday - month_days[leap][i]
		i=i+1
	pmonth = i
	pday = yearday
	
	if(notvalid_ymd(year,pmonth,pday)):
		return("")
	
	return(str(year).zfill(4)+separator+str(pmonth).zfill(2)+separator+str(pday).zfill(2))
	
# hh_mm_ss2time:  string hh:mm:ss to time (python daytime format)
def hh_mm_ss2time(hh_mm_ss):
	if((len(hh_mm_ss)!=8) or not hh_mm_ss.isdigit()):
		return(-1)
	h=int(hh_mm_ss[0:2])
	m=int(hh_mm_ss[3:5])
	s=int(hh_mm_ss[6:8])
	t=datetime.time(h,m,s)
	return (t)

#hhmm2hh hhmm (string) to hour and minute
def hhmm2hm(hhmm):
	if((len(hhmm)!=4) or not hhmm.isdigit()):
		return(-1,-1)
	h=int(hhmm[0:2])
	m=int(hhmm[2:4])
	return(h,m)

# hhmm2slot:  hhmm to  MSG or MFG slot number 
def hhmm2slot(hhmm, MSG=True, MFG_nominal=True):
	if((len(hhmm)!=4) or not hhmm.isdigit()):
		return(-1)
	hour=int(hhmm[0:2])
	minut=int(hhmm[2:4])
	return (hm2slot(hour, minut, MSG, MFG_nominal))

# hhmm2time:  string hhmm to time (python daytime format)
def hhmm2time(hhmm):
	if((len(hhmm)!=4) or not hhmm.isdigit()):
		return(-1)
	h=int(hhmm[0:2])
	m=int(hhmm[2:4])
	t=datetime.time(h,m)
	return (t)

# hm2hhmm  hour and minute to  hhmm (string)
def hm2hhmm(hour, minut):
	return(str(hour).zfill(2)+str(minut).zfill(2))

# hm2slot  hour and minute to  MSG or MFG slot number, 
# MFG fix Marek Caltik 18.1.2010 hm of scan start
def hm2slot(hour, minut, MSG=True, MFG_nominal=True):
	if MSG:
		# convert min to closest 15 min.
		slot = int(floor(minut/15.0))
		slot = slot + (hour*4) +1
	else:
		# convert min to closest 30 min.
		slot = int(floor(minut/30.0))
		slot = slot + (hour*2)+1
		if not MFG_nominal:
			slot+=1
			if slot>48:
				slot-=48
	return(int(slot))

# hm2time:  hm to time (python daytime format)
def hm2time(h,m):

	t=datetime.time(h,m)
	return (t)

#hours, minutes, seconds to decimal hours
def hms2dh(h,m,s):
	dh=h+(m/60.)+(s/3600.)
	return(dh)

# hms2time:  hm to time (python daytime format)
def hms2time(h,m,s):
	t=datetime.time(h,m,s)
	return (t)

#format deg dec to nice dms format
def latlon2formatedDms(latDD, lonDD):
	'''Marek Caltik added'''
	latD,latM,latS = dd2dms(latDD)
	latS = round(latS,2)
	if(latS == 60.0):
		latM = latM + 1
		latS = 0.0
	latHemi=""
	if(latDD > 0.0): 
		latHemi = "N"
	if(latDD < 0.0): 
		latHemi = "S"
		latD = latD * (-1)
	latMString = str(latM)
	if (len(latMString)==1) and (latMString!='0'):
		latMString = '0'+latMString
	latDString = str(latD)
	if (len(latDString)==1) and (latDString!='0'):
		latDString = '0'+latDString
	latDMS = (latDString,latMString,str(latS),latHemi)
	
	lonD,lonM,lonS = dd2dms(lonDD)
	lonS =round(lonS,2)
	if(lonS == 60.0):
		lonM = lonM + 1
		lonS = 0.0
	lonHemi=""
	if(lonDD > 0.0): 
		lonHemi = "E"
	if(lonDD < 0.0): 
		lonHemi = "W"
		lonD = lonD * (-1)
	lonMString = str(lonM)
	if (len(lonMString)==1) and (lonMString!='0'):
		lonMString = '0'+lonMString
	lonDString = str(lonD)
	if (len(lonDString)==1) and (lonDString!='0'):
		lonDString = '0'+lonDString
	lonDMS = (lonDString,lonMString,str(lonS),lonHemi)
	return (latDMS,lonDMS)

# check whether the year is leap 
# returns 1 for leap year, 0 otherwise 
def leapyear(year=1500):
	leap = (((year%4) == 0) and ((year%100) != 0) or ((year%400) == 0))
	return leap

#return day representing center day in a given month
#for months with even number of days first day in a pair is taken
def month_midday(month, year=2005):
	month=int(month)
	leap = leapyear(year)
	if ((year<1900) or (year>2060)):
		return (-1)
	if ((month<1) or (month>12)):
		return (-1)
	return month_centerdays[leap][month]

#return doy representing center day in a given month
#for months with even number of days first day in a pair is taken
def month_middoy(month, year=2005):
	month=int(month)
	if ((year<1900) or (year>2060)):
		return (-1)
	if ((month<1) or (month>12)):
		return (-1)
	return ymd2doy(year, month, month_midday(month, year))

#return doy representing center day in a given month
#for months with even number of days first day in a pair is taken
def month_representdoy(month, year=2005):
	month=int(month)
	if ((year<1900) or (year>2060)):
		return (-1)
	if ((month<1) or (month>12)):
		return (-1)
	return ymd2doy(year, month, month_midday(month, year))

#return day representing center day in a given month
#for months with even number of days first day in a pair is taken
def month_representday(month, year=2005):
	month=int(month)
	leap = leapyear(year)
	if ((year<1900) or (year>2060)):
		return (-1)
	if ((month<1) or (month>12)):
		return (-1)
	return month_representativedays[leap][month]

#return number of days in given year
def month_numdays(month, year=2005):
	month=int(month)
	leap = leapyear(year)
	if ((year<1900) or (year>2060)):
		return (-1)
	if ((month<1) or (month>12)):
		return (-1)
	return month_days[leap][month]

# test validity of dfb
def notvalid_dfb(dfb):
	if(dfb < 0):
		return(-1)
	return(0)

# test validity of doy
def notvalid_doyy(doy,year=2001):
	leap = leapyear(year)
	if((doy < 1) or (doy>(365+leap))):
		return(-1)
	return(0)

# test validity of hm
def notvalid_hm(hour,minut):
	if ((hour<0) or (hour>23)):
		return (-1)
	if ((minut<0) or (minut>59)):
		return (-1)
	return 0

# test validity of hhmm
def notvalid_hhmm(hhmm):
	if (type(hhmm) != str):
		return(-1)
	if((len(hhmm)!=4) or not hhmm.isdigit()):
		return(-1)
	hour=int(hhmm[0:2])
	minut=int(hhmm[2:4])
	return (notvalid_hm(hour,minut))

# test validity of MSG or MFG slot number
def notvalid_slot(msgslot, MSG=True):
	if MSG:
		if ((msgslot <1) or (msgslot >96)):
			return(-1) # time slot out of range 
	else:
		if ((msgslot <1) or (msgslot >48)):
			return(-1) # time slot out of range 
	return(0)

# test validity of ymd 
def notvalid_ymd(year, month, day):
	leap = leapyear(year)
	if ((year<1900) or (year>2060)):
		return (-1)
	if ((month<1) or (month>12)):
		return (-1)
	if ((day <1) or (day>month_days[leap][month])):
		return (-1)
	return 0

# test validity of yyyymmdd 
def notvalid_yyyymmdd(yyyymmdd):
	if((len(yyyymmdd)!=8) or not yyyymmdd.isdigit()):
		return(-1)
	year=int(yyyymmdd[0:4])
	month=int(yyyymmdd[4:6])
	day=int(yyyymmdd[6:8])
	return(notvalid_ymd(year,month,day))

def yyyy_mm_dd2yyyymmdd(yyyy_mm_dd):
	''' converts yyyy-mm-dd to yyyymmdd '''
	if(len(yyyy_mm_dd)!=10) :
		return(-1)
	return yyyy_mm_dd[0:4]+yyyy_mm_dd[5:7]+yyyy_mm_dd[8:10]

def yyyymmdd2yyyy_mm_dd(yyyymmdd, separator='-'):
	''' converts yyyy-mm-dd to yyyymmdd '''
	if(len(yyyymmdd)!=8) :
		return(-1)
	return yyyymmdd[0:4]+separator+yyyymmdd[4:6]+separator+yyyymmdd[6:8]

def slot2datetime(msgslot,aD,MSG=True, MFG_nominal=True):
	'''Returns datetime from slot num, added by Marek Caltik'''
	if MSG:
		if ((msgslot <1) or (msgslot >96)):
			return(-1) # time slot out of range 
	else:
		if ((msgslot <1) or (msgslot >48)):
			return(-1) # time slot out of range
	h, m = slot2hm(msgslot,MSG,MFG_nominal)
	t=datetime.time(hour=h,minute=m)
	aDT=datetime.datetime.combine(aD,t)
	if (not MSG) & (not MFG_nominal) & (msgslot==1):
		aDT-=datetime.timedelta(days=1)
	return (aDT)
	
# slot2hhmm: converts MSG or MFG slot number to hour and minute of scan start 
# mfg fix Marek Caltik 18.01.2010 hhmm of scan start
# for GOES (simplified 48 slots) MSG=False, MFG_nominal=True is used. slot 1 has value 0.0(00:00), slot 2 0.5(00:30)
def slot2hhmm(slot, MSG=True, MFG_nominal=True):
	if MSG:
		if ((slot <1) or (slot >96)):
			return("") # time slot out of range 
	else:
		if ((slot <1) or (slot >48)):
			return("") # time slot out of range 
	h, m = slot2hm(slot, MSG, MFG_nominal)
	return(str(h).zfill(2)+str(m).zfill(2))
	
# slots are numbered from 1 to 96. 1 is slot starting at 00:00 and 96 at 23:45 
# for MFG 1 to 48 (30 min interval)
# slot2hm:  converts slot number to hour and minute of scan start 
# for mfg: nominal time (end time of image) is written in file metadata, but for us start time of image
# is important, therefore slot 1 = 00:00 (not 00:30 as in file header) Marek Caltik 18.1.2010
# 
# MFG_nominal = False returns time decreased by the 30 min TOmas Cebecauer 2.1.2011 
# for GOES (simplified 48 slots) MSG=False, MFG_nominal=True is used. slot 1 has value 0.0(00:00), slot 2 0.5(00:30)
def slot2hm(slot, MSG=True, MFG_nominal=True):
	if MSG:
		if ((slot <1) or (slot >96)):
			return([]) # time slot out of range 
		hour = int(floor((slot -1)/4))
		minut = int(((slot -1)%4)*15)
	else:
		if ((slot <1) or (slot >48)):
			return([]) # time slot out of range 
		hour = int(floor((slot-1)/2))
		minut = int(((slot-1)%2)*30)
		if not MFG_nominal:
			if minut == 30:
				minut = 0
			else:
				minut=30
				hour-=1
			if hour<0:
				hour+=24
	return(hour, minut)

def slot2time(slot,MSG=True, MFG_nominal=True):
	# mfg fix Marek Caltik 18.01.2010 time of scan start
	if MSG:
		if ((slot <1) or (slot >96)):
			return(-1) # time slot out of range 
	else:
		if ((slot <1) or (slot >48)):
			return(-1) # time slot out of range
	h,m = slot2hm(slot,MSG, MFG_nominal)
	t=datetime.time(hour=h,minute=m)
	return t

def slot2dh(slot,MSG=True, MFG_nominal=True):
	# mfg fix Marek Caltik 18.01.2010 time of scan start
	if MSG:
		if ((slot <1) or (slot >96)):
			return(-1) # time slot out of range 
	else:
		if ((slot <1) or (slot >48)):
			return(-1) # time slot out of range
	h,m = slot2hm(slot,MSG, MFG_nominal)
	dh=hms2dh(h,m,0)
	return dh

# time2hhmm: time (python daytime format) to hm (string)
def time2hhmm(time):
	m=time.minute
	h=time.hour
	return(str(h).zfill(2)+str(m).zfill(2))

# time2hm: time (python daytime format) to hm
def time2hm(atime):
	m=atime.minute
	h=atime.hour
	return(h,m)

# time2dh: time (python daytime format) to dh
def time2dh(atime):
	m=atime.minute
	h=atime.hour
	dh=h+m/60.
	return(dh)

# time2hm: time (python daytime format) to hms
def time2hms(atime):
	s=atime.second
	m=atime.minute
	h=atime.hour
	return(h,m,s)

# time2slot  time (python daytime format) to MSG or slot number 
def time2slot(time, MSG=True, MFG_nominal=True):
	m=time.minute
	h=time.hour
	return(hm2slot(h, m, MSG, MFG_nominal))

#ymd2archseg: converts year month day to msg archive segment (six 61 days long segments)
def ymd2archseg(y, m, d):
	if notvalid_ymd(y,m,d):
		return(-1)
	doy=ymd2doy(y, m, d)
	return doy2archseg(doy)

#ymd2date: converts year month day to date (python daytime format)
def ymd2date(year,month,day):
	year,month,day = int(year),int(month),int(day)
	if notvalid_ymd(year,month,day):
		return(-1)
	d=datetime.date(year,month,day)
	return(d)

# ymd2dfb:  converts date in year month day to  day from the baginning where beginning is defined as  1st Jan YEARBEGIN (in our case 1980) 
def ymd2dfb(year, month, day):
	global yearbegin
	if(notvalid_ymd(year,month,day)):
		return (-1)
	
	if(year<yearbegin):
		return (-1)
	
	doy=ymd2doy(year, month, day)
	if(year==yearbegin):
		return (doy)

	yearcor=int((yearbegin-1) % 4); # correction for leap year
	leapyears=int(floor((year-yearbegin + yearcor)/4.0))
	yeardays=int(floor(year - yearbegin)*365) + leapyears
	dayfbeg=yeardays+doy
	
	return(dayfbeg)

# ymd2doy: converts date in year month day to day of the year 
def ymd2doy(year, month, day):
	global month_days
	if(notvalid_ymd(year,month,day)):
		return (-1)
	leap = leapyear(year)
	for i in range(1,month):
		day += month_days[leap][i];
	return day;

#ymd2yyyymmdd: converts year month day to string yyyymmdd 
def ymd2yyyymmdd(year, month, day, separator=''):
	return(str(year).zfill(4)+separator+str(month).zfill(2)+separator+str(day).zfill(2))

#yyyymmdd2archseg: converts yyyymmdd to msg archive segment (six 61 days long segments)
def yyyymmdd2archseg(yyyymmdd):
	if notvalid_yyyymmdd(yyyymmdd):
		return(-1)
	doy=yyyymmdd2doy(yyyymmdd)
	return doy2archseg(doy)

#yyyymmdd2date: converts yyyymmdd(string) to date (python daytime format)
def yyyymmdd2date(yyyymmdd):
	if notvalid_yyyymmdd(yyyymmdd):
		return(-1)
	year,month,day=yyyymmdd2ymd(yyyymmdd)
	d=datetime.date(year,month,day)
	return(d)

# yyyymmdd2dfb:  converts date in yyyymmdd to day from the baginning where beginning is defined as  1st Jan YEARBEGIN (in our case 1980) 
def yyyymmdd2dfb(yyyymmdd):
	global yearbegin
	if((len(yyyymmdd)!=8) or not yyyymmdd.isdigit()):
		return(-1)
	year=int(yyyymmdd[0:4])
	month=int(yyyymmdd[4:6])
	day=int(yyyymmdd[6:8])
	if(notvalid_ymd(year,month,day)):
		return(-1);
	if(notvalid_ymd(year,month,day)):
		return (-1)
	
	if(year<yearbegin):
		return (-1)
	
	doy=ymd2doy(year, month, day)
	if(year==yearbegin):
		return (doy)

	yearcor=int((yearbegin-1) % 4); # correction for leap year
	leapyears=int(floor((year-yearbegin + yearcor)/4.0))
	yeardays=int(floor(year - yearbegin)*365) + leapyears
	dayfbeg=yeardays+doy
	
	return(dayfbeg)

# yyyymmdd2doy:  converts date in yyyymmdd to day of yaear 
def yyyymmdd2doy(yyyymmdd):
	global yearbegin
	if((len(yyyymmdd)!=8) or not yyyymmdd.isdigit()):
		return(-1)
	year=int(yyyymmdd[0:4])
	month=int(yyyymmdd[4:6])
	day=int(yyyymmdd[6:8])
	if(notvalid_ymd(year,month,day)):
		return(-1);
	if(notvalid_ymd(year,month,day)):
		return (-1)
	
	doy = ymd2doy(year, month, day)
	return doy

#yyyymmdd2ymd: converts string yyyymmdd to  year month day
def yyyymmdd2ymd(yyyymmdd):
	if((len(yyyymmdd)!=8) or not yyyymmdd.isdigit()):
		return([])
	year=int(yyyymmdd[0:4])
	month=int(yyyymmdd[4:6])
	day=int(yyyymmdd[6:8])
	if(notvalid_ymd(year,month,day)):
		return([]);
	return(year,month,day)


#for doy return closest 2 months and weight for the closes one
# the interpolation is done interpolatedVal=val[month1]*weight +val[month2]*(1-weight)
# interpolatedVal=val[month2] +(val[month1]-val[month2])*weight
def doyy2month_interpolation_weight(doy,year):
	leap=leapyear(year)
	cdoys = month_centerdoys[leap]
	cdoys[0] = cdoys[12]-365-leap 
	cdoys.append(cdoys[1]+365+leap )

	m=doyy2md(doy,2007+leap)[0]
	cdoy=cdoys[m]
	if cdoy>=doy:
		month1=m
		month2=m-1
	elif cdoy<doy:
		month1=m
		month2=m+1
	weight=float(cdoys[month2]-doy)/(cdoys[month2]-cdoys[month1])
	if month2 <1: month2=12
	if month2 >12: month2=1
	return (month1,month2,weight)

#for dfb return closest 2 months and weight for the closes one
# the interpolation is done interpolatedVal=val[month1]*weight +val[month2]*(1-weight)
# interpolatedVal=val[month2] +(val[month1]-val[month2])*weight
def dfb2month_interpolation_weight(dfb):
	doy,year=dfb2doyy(dfb)
	return doyy2month_interpolation_weight(doy, year)

#create list of (year, archseg) pairs covering the period defined by dfb_start, dfb_end
def dfb_minmax2yea_archseg_list(dfb_start, dfb_end):
	a1=dfb2archseg(dfb_start)
	a2=dfb2archseg(dfb_end)
	y1=dfb2ymd(dfb_start)[0]
	y2=dfb2ymd(dfb_end)[0]
	year_arch_pais_list=[]
	for y in range(y1,y2+1):
		for a in range(1,6+1):
			if (y==y1) and (a < a1):
				continue
			if (y==y2) and (a > a2):
				continue
			year_arch_pais_list.append((y, a))
	return year_arch_pais_list


#create list of (year, archseg) pairs covering the period defined by dfb_start, dfb_end
def dfb_minmax2year_month_list(dfb_start, dfb_end):
	m1=dfb2ymd(dfb_start)[1]
	m2=dfb2ymd(dfb_end)[1]
	y1=dfb2ymd(dfb_start)[0]
	y2=dfb2ymd(dfb_end)[0]
	year_month_pais_list=[]
	for y in range(y1,y2+1):
		for m in range(1,12+1):
			if (y==y1) and (m < m1):
				continue
			if (y==y2) and (m > m2):
				continue
			year_month_pais_list.append((y, m))
	return year_month_pais_list


#resolves CET / CEST datetime and returns utc datetime (+ a boolean True if DT is in CEST)
# CET slots in March between <2:00 and 3:00) cannot normally occur - they are lost! (but this method is not aware of that)
# CEST slots in October between <2:00 and 3:00) are repeated in CET!
# assuming DT in CET or CEST clock time
# added by Artur+Marek
def central_european_time2utc(DT):
	summr_td=datetime.timedelta(hours=-2)
	wintr_td=datetime.timedelta(hours=-1)
	y=DT.date().year
	
	#resolve days according to year
	day_beg=(31-((((5*y)/4)+4)%7))
	day_end=(31-((((5*y)/4)+1)%7))
	
	summertime=False
	if (DT>=datetime.datetime(year=y, month=3,day=day_beg,hour=2)) and (DT<=datetime.datetime(year=y, month=10,day=day_end,hour=2)):
		DT+=summr_td
		summertime=True
	else:
		DT+=wintr_td
	return DT,summertime




if __name__ == "__main__":
	aD=datetime.date(2007,04,19)
	for s in range(1,48+1):
		print s, slot2hm(s,MSG=False, MFG_nominal=False), slot2hm(s,MSG=False, MFG_nominal=True), slot2datetime(s,aD,MSG=False, MFG_nominal=False)
		h, m=slot2hm(s,MSG=False, MFG_nominal=True)
		print hm2slot(h,m,MSG=False,MFG_nominal=True)
	exit()
	print dh2hms(-0.125)
	exit()

	#some testing
	yyyymmdd='20041231'
	y,m,d=yyyymmdd2ymd(yyyymmdd)
	dat=ymd2date(y,m,d)
	dat2=yyyymmdd2date(yyyymmdd)
	dfb=date2dfb(dat2)
	doy=date2doy(dat2)
	print "doy:",doy, " archseg", doy2archseg(doy), yyyymmdd2archseg(yyyymmdd),  dfb2archseg(dfb), ymd2archseg(y,m,d) , date2archseg(dat)
	print dat, dat2, date2ymd(dat), date2yyyymmdd(dat2)
	print dfb, dfb2date(dfb), doy, doyy2date(doy, y)
	h,mi,s=18,16,0
	t=hm2time(h,mi)
	hhmm=hm2hhmm(h,mi)
	slot=hhmm2slot(hhmm)
	print notvalid_slot(0), notvalid_hm(24,00), notvalid_hhmm('1459')
	print t, time2hhmm(t),hhmm, hhmm2time(hhmm),hhmm2hm(hhmm),slot2hhmm(slot),slot2hm(slot),slot2time(slot)
	print time2slot(t),hhmm2slot(hhmm),hm2slot(h,mi), 
	#dh=dms2dd(h,mi,s)
	#res= dfbdh_offset(yyyymmdd2dfb(yyyymmdd),dh,21.75)
	#print (dfb2ymd(res[0]),dd2dms(res[1]))
	#dd=dms2dd(8,37,37.4)
	#print dd, dd2dms(dd)
	#print month_days[leapyear(2000)][2], ymd2doy(1979,12,15),doyy2md(3849,1979),ymd2dfb(1981,11,30), dfb2ymd(700),slot2hm(50),hm2slot(12,15),yyyymmdd2ymd("20071212"),len(yyyymmdd2ymd("20071201")),slot2hhmm(73)
	print slot2hm(4, MSG=False)
	print hm2slot(23,30, MSG=False)
	print slot2hhmm(1, MSG=False)
	print hhmm2slot("0000", MSG=False)
	print slot2time(48, False)
	print slot2datetime(48, datetime.datetime(1993,01,01),False)
