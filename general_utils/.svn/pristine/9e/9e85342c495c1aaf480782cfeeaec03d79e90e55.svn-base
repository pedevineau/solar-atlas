#! /usr/bin/env python
#NOTE: this version is valid also for MFG data, Marek Caltik, 07/2009
# daytimeconv
#
# last revision: 14/04/2008
#
#
#DATE
#leapyear(year) - test leap year
#notvalid_ymd(year, month, day) - validate date
#notvalid_yyyymmdd(yyyymmdd) -  test validity of yyyymmdd 
#notvalid_dfb(dfb) - test validity of dfb
#notvalid_doyy(doy,year=2001) - test validity of doy

#ymd2doy(year, month, day) - ymd to 'day of year'
#doyy2md(yearday, year) - 'day of year' and year to month day
#ymd2dfb(year, month, day) - ymd to day from beggining  (yearbegin=1980)
#dfb2ymd(dayfrombeg) - day from beggening to ymd
#yyyymmdd2ymd(yyyymmdd) - string yyyymmdd 
#ymd2yyyymmdd(year, month, day) 
#dfb2yyyymmdd(dayfrombeg)
#yyyymmdd2dfb(yyyymmdd)
#doyy2yyyymmdd(yearday, year)
#yyyymmdd2doy(yyyymmdd)
#ymd2date(year,month,day) - converts year month day to date (python daytime format)
#date2ymd(dat) - converts date (python daytime format) to year month day
#yyyymmdd2date(yyyymmdd) - converts yyyymmdd(string) to date (python daytime format)
#date2yyyymmdd(dat) - converts date (python daytime format) to yyyymmdd (string)
#dfb2date(dfb) - converts day from beginning to date (python daytime format)
#date2dfb(dat) - converts date (python daytime format) to day from beginning
#doyy2date(yearday, year) - converts day of the year and year to date (python daytime format)
#doyy2dfb(yearday, year) - converts day of the year and year to  day from beginning
#dfb2doy(dfb) - converts day from begining to day of the year
#date2doy(date) - converts date to day of the year
#doyy2date(yearday, year) - converts day of the year and year to date

#doy2archseg(doy) - converts doy to msg archive segment (six 61 days long segments)
#ymd2archseg(year, month, day) - converts year month day to msg archive segment (six 61 days long segments)
#yyyymmdd2archseg(yyyymmdd) - converts yyyymmdd to msg archive segment (six 61 days long segments)
#dfb2archseg(dfb) - converts dfb to msg archive segment (six 61 days long segments)
#date2archseg(date) - converts date to msg archive segment (six 61 days long segments)


#TIME
#notvalid_hm(hour,minut) - test validity of hm
#notvalid_hhmm(hhmm) - test validity of hhmm
#notvalid_slot(msgslot) - test validity of MSG slot number

#slot2hm(msgslot) - MSG time slot to h,m
#hm2slot(hour, minut)
#slot2hhmm(msgslot) - MSG time slot to hhmm string
#hhmm2slot(hhmm)
#hm2hhmm(hour, minut) - hour and minute to  hhmm (string)
#hhmm2hm(hhmm) - hhmm (string) to hour and minute
#hm2time(h,m) - hm to time (python daytime format)
#time2hm(time) - time (python daytime format) to hm
#hhmm2time(hhmm) - string hhmm to time (python daytime format)
#hh_mm_ss2time(hh_mm_ss) - string hh:mm:ss to time (python daytime format)
#slot2time(msgslot) - MSG slot number to time (python daytime format)
#time2slot(time) - time (python daytime format) to MSG slot number

#dms2dd(d,m,s) - degree, min, sec to decimal degree
#dd2dms(dd)
#hms2dh(h,m,s) - hours, minutes, seconds to decimal hours
#dh2hms(dh) - decimal hours to hours, minutes, seconds  - rounds to closest second
#dh_offset(dh,hoffset) - in/de-creases time (in decimal hours) by offset (in decimal hours)
#dfbdh_offset(dfb,dh,hoffset) - in/de-creases time by offset, if needed the day is modified as well  


from math import floor, fabs
from datetime import *


yearbegin=1980
daytab=[[0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],[0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]]
month_days=daytab
month_centerdays=[[0, 17, 14, 16, 15, 15, 11, 17, 16, 16, 16, 15, 11],[0, 17, 15, 16, 15, 15, 11, 17, 16, 16, 16, 15, 11]]

# check whether the year is leap 
# returns 1 for leap year, 0 otherwise 
def leapyear(year=1500):
	leap = (((year%4) == 0) and ((year%100) != 0) or ((year%400) == 0))
	return leap


# test validity of ymd 
def notvalid_ymd(year, month, day):
	leap = leapyear(year)
	if ((year<1900) or (year>2060)):
		return (-1)
	if ((month<1) or (month>12)):
		return (-1)
	if ((day <1) or (day>daytab[leap][month])):
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


# ymd2doy: converts date in year month day to day of the year 
def ymd2doy(year, month, day):
	global daytab
	if(notvalid_ymd(year,month,day)):
		return (-1)
	leap = leapyear(year)
	for i in range(1,month):
		day += daytab[leap][i];
	return day;

# doyy2md: converts day of the year and year (needed to decide leap year) to month and day
def doyy2md(yearday, year):
	global daytab
	leap = leapyear(year)
	i = 1
	while (yearday > daytab[leap][i]) and (i<12):
		yearday = yearday - daytab[leap][i]
		i=i+1
	pmonth = i
	pday = yearday
	
	if(notvalid_ymd(year,pmonth,pday)):
		return([])
	return(pmonth,pday)

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
		return([]);
	return(pyear,pmonth,pday);

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

#ymd2yyyymmdd: converts year month day to string yyyymmdd 
def ymd2yyyymmdd(year, month, day):
	return(str(year).zfill(4)+str(month).zfill(2)+str(day).zfill(2))

# dfb2yyyymmdd:  converts day from baginning to string yyyymmdd form  
def dfb2yyyymmdd(dayfrombeg):
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
	return(str(pyear).zfill(4)+str(pmonth).zfill(2)+str(pday).zfill(2))

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

# doyy2yyyymmdd: converts day of year and year to yyyymmdd
def doyy2yyyymmdd(yearday, year):
	global daytab
	leap = leapyear(year)
	i = 1
	while (yearday > daytab[leap][i]) and (i<12):
		yearday = yearday - daytab[leap][i]
		i=i+1
	pmonth = i
	pday = yearday
	
	if(notvalid_ymd(year,pmonth,pday)):
		return("")
	
	return(str(year).zfill(4)+str(pmonth).zfill(2)+str(pday).zfill(2))

#ymd2date: converts year month day to date (python daytime format)
def ymd2date(year,month,day):
	if notvalid_ymd(year,month,day):
		return(-1)
	d=date(year,month,day)
	return(d)

#date2ymd: converts date (python daytime format) to year month day
def date2ymd(dat):
	y=dat.year
	m=dat.month
	d=dat.day
	return(y,m,d)

#yyyymmdd2date: converts yyyymmdd(string) to date (python daytime format)
def yyyymmdd2date(yyyymmdd):
	if notvalid_yyyymmdd(yyyymmdd):
		return(-1)
	year,month,day=yyyymmdd2ymd(yyyymmdd)
	d=date(year,month,day)
	return(d)

#date2yyyymmdd: converts date (python daytime format) to yyyymmdd (string)
def date2yyyymmdd(dat):
	y=dat.year
	m=dat.month
	d=dat.day
	return(ymd2yyyymmdd(y,m,d))

#dfb2date: converts day from beginning to date (python daytime format)
def dfb2date(dfb):
	if notvalid_dfb(dfb):
		return(-1)
	year,month,day=dfb2ymd(dfb)
	d=date(year,month,day)
	return(d)

#date2dfb: converts date (python daytime format) to day from beginning
def date2dfb(dat):
	y=dat.year
	m=dat.month
	d=dat.day
	return(ymd2dfb(y,m,d))

#doyy2date: converts day of the year and year to date (python daytime format)
def doyy2date(yearday, year):
	if notvalid_doyy(yearday, year):
		return(-1)
	m,d=doyy2md(yearday, year)
	return(ymd2date(year,m,d))

#doyy2dfb: converts day of the year and year to  day from beginning
def doyy2dfb(yearday, year):
	if notvalid_doyy(yearday, year):
		return(-1)
	m,d=doyy2md(yearday, year)
	return(ymd2dfb(year,m,d))

#date2doy: converts date (python daytime format) to day of the year
#def date2doy(dat):
#	y=dat.year
#	m=dat.month
#	d=dat.day
#	return(ymd2doy(y,m,d))

#dfb2doy: converts day from begining to day of the year
def dfb2doy(dfb):
	y,m,d=dfb2ymd(dfb)
	return(ymd2doy(y,m,d))

#date2doy: converts date to day of the year
def date2doy(adate):
	y,m,d=date2ymd(adate)
	return(ymd2doy(y,m,d))


#doy2archseg: converts doy to msg archive segment (six 61 days long segments)
def doy2archseg(d):
	i=1
	while (d>61):
		d-=61
		i+=1
	return i

#ymd2archseg: converts year month day to msg archive segment (six 61 days long segments)
def ymd2archseg(y, m, d):
	if notvalid_ymd(y,m,d):
		return(-1)
	doy=ymd2doy(y, m, d)
	return doy2archseg(doy)

#yyyymmdd2archseg: converts yyyymmdd to msg archive segment (six 61 days long segments)
def yyyymmdd2archseg(yyyymmdd):
	if notvalid_yyyymmdd(yyyymmdd):
		return(-1)
	doy=yyyymmdd2doy(yyyymmdd)
	return doy2archseg(doy)

#dfb2archseg: converts dfb to msg archive segment (six 61 days long segments)
def dfb2archseg(dfb):
	if notvalid_dfb(dfb):
		return(-1)
	doy=dfb2doy(dfb)
	return doy2archseg(doy)

#date2archseg: converts date to msg archive segment (six 61 days long segments)
def date2archseg(dat):
	y=dat.year
	m=dat.month
	d=dat.day
	doy=ymd2doy(y,m,d)
	return doy2archseg(doy)



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

# slots are numbered from 1 to 96. 1 is slot starting at 00:00 and 96 at 23:45 
# for MFG 1 to 48 (30 min interval)
# slot2hm:  converts slot number to hour and minute of scan start 
# for mfg: nominal time (end time of image) is written in file metadata, but for us start time of image
# is important, therefore slot 1 = 00:00 (not 00:30 as in file header) Marek Caltik 18.1.2010
# 
def slot2hm(slot, MSG=True):
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
	return(hour, minut)

# hm2slot  hour and minute to  MSG or MFG slot number, 
# MFG fix Marek Caltik 18.1.2010 hm of scan start
def hm2slot(hour, minut, MSG=True):
	if MSG:
		# convert min to closest 15 min.
		slot = int(floor(minut/15.0))
		slot = slot + (hour*4) +1
	else:
		# convert min to closest 30 min.
		slot = int(floor(minut/30.0))
		slot = slot + (hour*2)+1
	return(slot)

# slot2hhmm: converts MSG or MFG slot number to hour and minute of scan start 
# mfg fix Marek Caltik 18.01.2010 hhmm of scan start
def slot2hhmm(slot, MSG=True):
	if MSG:
		if ((slot <1) or (slot >96)):
			return("") # time slot out of range 
		hour = int(floor((slot -1)/4))
		minut = int(((slot -1)%4)*15)
	else:
		if ((slot <1) or (slot >48)):
			return("") # time slot out of range 
		hour = int(floor((slot-1)/2))
		minut = int(((slot-1)%2)*30)
	return(str(hour).zfill(2)+str(minut).zfill(2))

# hhmm2slot:  hhmm to  MSG or MFG slot number 
# mfg fix Marek Caltik 18.01.2010 hhmm of scan start
def hhmm2slot(hhmm, MSG=True):
	if((len(hhmm)!=4) or not hhmm.isdigit()):
		return(-1)
	hour=int(hhmm[0:2])
	minut=int(hhmm[2:4])
	if MSG:
		# convert min to closest 15 min.
		slot = int(floor(minut/15.0))
		slot = slot + (hour*4) +1
	else:
		# convert min to closest 30 min.
		slot = int(floor(minut/30.0))
		slot = slot + (hour*2)+1
	return(slot)

# hm2hhmm  hour and minute to  hhmm (string)
def hm2hhmm(hour, minut):
	return(str(hour).zfill(2)+str(minut).zfill(2))

#hhmm2hh hhmm (string) to hour and minute
def hhmm2hm(hhmm):
	if((len(hhmm)!=4) or not hhmm.isdigit()):
		return(-1,-1)
	h=int(hhmm[0:2])
	m=int(hhmm[2:4])
	return(h,m)

# hm2time:  hm to time (python daytime format)
def hm2time(h,m):
	t=time(h,m)
	return (t)

# time2hm: time (python daytime format) to hm
def time2hm(time):
	m=time.minute
	h=time.hour
	return(h,m)

# hhmm2time:  string hhmm to time (python daytime format)
def hhmm2time(hhmm):
	if((len(hhmm)!=4) or not hhmm.isdigit()):
		return(-1)
	h=int(hhmm[0:2])
	m=int(hhmm[2:4])
	t=time(h,m)
	return (t)

# hh_mm_ss2time:  string hh:mm:ss to time (python daytime format)
def hh_mm_ss2time(hh_mm_ss):
	if((len(hh_mm_ss)!=8) or not hh_mm_ss.isdigit()):
		return(-1)
	h=int(hh_mm_ss[0:2])
	m=int(hh_mm_ss[3:5])
	s=int(hh_mm_ss[6:8])
	t=time(h,m,s)
	return (t)

# time2hhmm: time (python daytime format) to hm (string)
def time2hhmm(time):
	m=time.minute
	h=time.hour
	return(str(h).zfill(2)+str(m).zfill(2))

# slot2time: MSG or MFG slot number to time (python daytime format)

def slot2time(slot,MSG=True):
	# mfg fix Marek Caltik 18.01.2010 time of scan start
	if MSG:
		if ((slot <1) or (slot >96)):
			return(-1) # time slot out of range 
	else:
		if ((slot <1) or (slot >48)):
			return(-1) # time slot out of range
	h,m = slot2hm(slot,MSG)
	t=time(hour=h,minute=m)
	return t

def slot2datetime(msgslot,aD,MSG=True):
	'''Returns datetime from slot num, added by Marek Caltik'''
	if MSG:
		if ((msgslot <1) or (msgslot >96)):
			return(-1) # time slot out of range 
	else:
		if ((msgslot <1) or (msgslot >48)):
			return(-1) # time slot out of range
	h,m = slot2hm(msgslot,MSG)
	t=time(hour=h,minute=m)
	aDT=datetime.combine(aD,t)
	return (aDT)

# time2slot  time (python daytime format) to MSG or MFG slot number 
def time2slot(time, MSG=True):
	m=time.minute
	h=time.hour
	return(hm2slot(h, m, MSG))



#degrees,minutes, seconds to decimal degrees
def dms2dd(d,m,s):
	dd=d+(m/60.)+(s/3600.)
	return(dd)

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

#hours, minutes, seconds to decimal hours
def hms2dh(h,m,s):
	dh=h+(m/60.)+(s/3600.)
	return(dh)

#decimal hours to hours, minutes, seconds  - rounds to closest second
def dh2hms(dh):
	dha=fabs(dh)
	h=int(floor(dha))
	res=(dha-h)*60.
	m=int(floor(res+0.5))
	s=int(floor(((res-m)*60.)+0.5))
	return(h,m,s)

def latlon2formatedDms(latDD, lonDD):
	'''Marek Caltik'''
	latD,latM,latS = dd2dms(latDD)
	latS = round(latS,2)
	if(latS == 60.0):
		latM = latM + 1
		latS = 0.0
	latHemi=""
	if(latD > 0): latHemi = "N"
	if(latD < 0): 
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
	if(lonD > 0): lonHemi = "E"
	if(lonD < 0): 
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

#in/de-creases time (in decimal hours) by offset (also in decimal hours)
def dh_offset(dh,hoffset):
	dho=dh+hoffset
	while(dho>=24.):
		dho=dho-24.
	while(dho<0.):
		dho=dho+24.
	return(dho)
	
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

#return number of days in given year
def month_numdays(month, year=2005):
	month=int(month)
	leap = leapyear(year)
	if ((year<1900) or (year>2060)):
		return (-1)
	if ((month<1) or (month>12)):
		return (-1)
	return month_days[leap][month]

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




if __name__ == "__main__":
	#some testing
#	yyyymmdd='20041231'
#	y,m,d=yyyymmdd2ymd(yyyymmdd)
#	dat=ymd2date(y,m,d)
#	dat2=yyyymmdd2date(yyyymmdd)
#	dfb=date2dfb(dat2)
#	doy=date2doy(dat2)
#	print "doy:",doy, " archseg", doy2archseg(doy), yyyymmdd2archseg(yyyymmdd),  dfb2archseg(dfb), ymd2archseg(y,m,d) , date2archseg(dat)
#	print dat, dat2, date2ymd(dat), date2yyyymmdd(dat2)
#	print dfb, dfb2date(dfb), doy, doyy2date(doy, y)
#	h,mi,s=18,16,0
#	t=hm2time(h,mi)
#	hhmm=hm2hhmm(h,mi)
#	slot=hhmm2slot(hhmm)
#	print notvalid_slot(0), notvalid_hm(24,00), notvalid_hhmm('1459')
#	print t, time2hhmm(t),hhmm, hhmm2time(hhmm),hhmm2hm(hhmm),slot2hhmm(slot),slot2hm(slot),slot2time(slot)
#	print time2slot(t),hhmm2slot(hhmm),hm2slot(h,mi), 
	#dh=dms2dd(h,mi,s)
	#res= dfbdh_offset(yyyymmdd2dfb(yyyymmdd),dh,21.75)
	#print (dfb2ymd(res[0]),dd2dms(res[1]))
	#dd=dms2dd(8,37,37.4)
	#print dd, dd2dms(dd)
	#print daytab[leapyear(2000)][2], ymd2doy(1979,12,15),doyy2md(3849,1979),ymd2dfb(1981,11,30), dfb2ymd(700),slot2hm(50),hm2slot(12,15),yyyymmdd2ymd("20071212"),len(yyyymmdd2ymd("20071201")),slot2hhmm(73)
#	print slot2datetime(1,date(2004,12,31), False)
#	print slot2time(1)
#	print hhmm2slot("0000", False)
	print slot2hm(4, MSG=False)
	print hm2slot(23,30, MSG=False)
	print slot2hhmm(1, MSG=False)
	print hhmm2slot("0000", MSG=False)
	print slot2time(48, False)
