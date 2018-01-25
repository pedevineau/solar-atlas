#! /usr/bin/env python
'''
Created on Jan 21, 2011

@author: tomas
'''

import numpy
from general_utils import daytimeconv
import datetime
from pylab import date2num, plot, show, figure, xticks, yticks, setp, title, num2date, legend, scatter, axhspan, axhline, xlabel, ylabel, axvspan
#from matplotlib.dates import DayLocator, HourLocator, DateFormatter, MonthLocator
from general_utils import db_sites
from general_utils import db_utils
from general_utils import solar_geom_v5

from general_utils import funct_fit


def _interpolate(a, b, fraction):
    """Returns the point at the given fraction between a and b, where
    'fraction' must be between 0 and 1.
    """
    return a + (b - a)*fraction;

def scoreatpercentile_multi(a, per_list):
    """Calculate the score at the given 'per' percentile of the
    sequence a.  For example, the score at per=50 is the median.
    
    If the desired quantile lies between two data points, we
    interpolate between them.
    
    """
    # TODO: this should be a simple wrapper around a well-written quantile
    # function.  GNU R provides 9 quantile algorithms (!), with differing
    # behaviour at, for example, discontinuities.
    values = numpy.sort(a,axis=0)
    
    per_out_list=[]

    num_vals = (values.shape[0] - 1)
    for per in per_list:
        idx = per /100. * num_vals
        if (idx % 1 == 0):
            per_out_list.append(values[idx])
        else:
            per_out_list.append( _interpolate(values[int(idx)], values[int(idx) + 1], idx % 1))
    
    return per_out_list


def bins_stat(x, y, xmin, xmax, numbins):
    
    numbins = int(numbins)
    bin_size = (xmax-xmin)/numbins
    
    bins_min = numpy.empty((numbins), dtype=numpy.float32)
    bins_max = numpy.empty((numbins), dtype=numpy.float32)
    bins_mid = numpy.empty((numbins), dtype=numpy.float32)
    bins_P25 = numpy.empty((numbins), dtype=numpy.float32)
    bins_P50 = numpy.empty((numbins), dtype=numpy.float32)
    bins_P75 = numpy.empty((numbins), dtype=numpy.float32)
    bins_P25[:] = numpy.nan
    bins_P50[:] = numpy.nan
    bins_P75[:] = numpy.nan
    
    percentiles=[25, 50, 75]

    for i in range(0,numbins):
        bins_min[i] = xmin + (i*bin_size)
        bins_max[i] = bins_min[i] + bin_size
        bins_mid[i] = bins_min[i] + (bin_size/2.)
        
        wh = (x >= bins_min[i]) & (x < bins_max[i])  
        if wh.sum() > 0:
            result = scoreatpercentile_multi(y[wh], percentiles)
            bins_P25[i] = result[0]
            bins_P50[i] = result[1]
            bins_P75[i] = result[2]
    
    bins_dict = {}
    bins_dict['bins_min'] = bins_min
    bins_dict['bins_max'] = bins_max
    bins_dict['bins_mid'] = bins_mid
    bins_dict['bins_P25'] = bins_P25
    bins_dict['bins_P50'] = bins_P50
    bins_dict['bins_P75'] = bins_P75

    return bins_dict

#add vlaues to force fitting
def enforce(nvals, fix_list_ciktm, ci, ktm):
    for fixci, fixktm in fix_list_ciktm:
        fix_ci_vect=numpy.empty((nvals))
        fix_ci_vect[:]=fixci
        fix_ktm_vect=numpy.empty((nvals))
        fix_ktm_vect[:]=fixktm
        ci=numpy.hstack((ci,fix_ci_vect ))
        ktm=numpy.hstack((ktm,fix_ktm_vect ))
    return ci, ktm


def make_ci_ktm_fit_1(ci, ktm):
    
    #Ktm = ci * (ci * (ci * (ci * (2.36 * ci - 6.2) + 6.22) - 2.63) - .58) + 1.
    # giving initial parameters
    a1 = funct_fit.Parameter(2.3)
    a2 = funct_fit.Parameter(6.2)
    a3 = funct_fit.Parameter(6.2)
    a4 = funct_fit.Parameter(2.6)
    a5 = funct_fit.Parameter(-0.5)
    a6 = funct_fit.Parameter(-1.0)
    # define your function:
    def f(x): return x * (x * (x * (x * (a1() * x + a2()) + a3()) + a4()) + a5()) + a6()
    
    # fit! (given that data is an array with the data to fit)
    funct_fit.fit(f, [a1, a2, a3, a4, a5, a6], ktm, ci)
    
    return a1(), a2(), a3(), a4(), a5(), a6()
     
    


def ci2ktm(ci, a1, a2, a3, a4, a5, a6):

    Ktm = ci * (ci * (ci * (ci * ((a1 * ci) + a2) + a3) + a4) + a5) + a6

    return Ktm


 

def _calculate_DT_minmax(aDTList):
    minDT=datetime.datetime(2200,12,31,23,59,59)
    maxDT=datetime.datetime(200,12,31,23,59,59)
    for aDT in aDTList:
        minDT=min(aDT,minDT)
        maxDT=max(aDT,maxDT)
    return minDT, maxDT
 
def datasources_get_total_dt_minmax(data_sources, overlapping_minmax_D):
    if overlapping_minmax_D:
        maxDT_total=datetime.datetime(2200,12,31,23,59,59)
        minDT_total=datetime.datetime(200,12,31,23,59,59)
    else:
        minDT_total=datetime.datetime(2200,12,31,23,59,59)
        maxDT_total=datetime.datetime(200,12,31,23,59,59)
        
    for data_source_key in data_sources.keys():
        data = data_sources[data_source_key]['DATA']
        minDT = data['minDT']
        maxDT = data['maxDT']
        if overlapping_minmax_D:
            maxDT_total=min(maxDT_total,maxDT)
            minDT_total=max(minDT_total,minDT)
        else:
            minDT_total=min(minDT_total,minDT)
            maxDT_total=max(maxDT_total,maxDT)
    return minDT_total, maxDT_total 

def  datasources_reduce_by_minmax_DT(data_sources, minDT_total, maxDT_total ):
    minD_total=minDT_total.date()
    maxD_total=maxDT_total.date()
    
    for data_source_key in data_sources.keys():
        data_source = data_sources[data_source_key]
        data=data_source['DATA']
        aDTList = data['dt_list']
        ghlist = data['val_list']
        minDT = data['minDT']
        maxDT = data['maxDT']
        aDTList_new=[]
        ghlist_new=[]
        for i in range(0,len(aDTList)):
            if (aDTList[i].date() >= minD_total) and (aDTList[i].date() <= maxD_total):
                aDTList_new.append(aDTList[i])
                ghlist_new.append(ghlist[i])
        data['minDT'] = max(minDT_total, minDT)
        data['maxDT'] = min(maxDT_total, maxDT)
        data['dt_list'] = aDTList_new
        data['val_list'] = ghlist_new
        data_source['DATA'] = data
    return data_sources

def datasources_arrays_from_lists(data_sources):
    for data_source_key in data_sources.keys():
        data_source = data_sources[data_source_key]
        data = data_source['DATA']
        aDTList = data['dt_list']
        vallist = data['val_list']

        data_aDTn=numpy.empty((len(aDTList)),dtype='float64')
        data_aDTn[:]=numpy.NaN
        data_aDTn2004=numpy.empty((len(aDTList)),dtype='float64')
        data_aDTn2004[:]=numpy.NaN
        data_val=numpy.empty((len(aDTList)),dtype='float64')
        data_val[:]=numpy.NaN

#        data_array64=numpy.empty((len(aDTList),3),dtype='float64')
#        data_array64[:,:]=numpy.NaN
        for i in range(0,len(aDTList)):
            aDT=aDTList[i]
            #aDT=datetime.strptime(aDT_item,'%Y-%m-%d %H:%M:%S')
            aDT_2004=datetime.datetime.combine(datetime.date(2004,aDT.month,aDT.day),aDT.time())
            data_aDTn[i]=date2num(aDT)
            data_aDTn2004[i]=date2num(aDT_2004)
#            data_array64[i,0]=date2num(aDT)
#            data_array64[i,1]=date2num(aDT_2004)
            if not (vallist[i] is None):
#                data_array64[i,2]=vallist[i]
                data_val[i]=vallist[i]
                
        #reduce Nan values    
#        wh = data_array64[:,2]==data_array64[:,2]
#        data_array64=data_array64[wh,:]
     
#        data['array']=data_array64
        data['array']={'DTn':data_aDTn,'DTn_2004':data_aDTn2004,'val':data_val}
        data_source['DATA'] = data
    return  data_sources


def calculate_solar_geom(aDT,longitude_r,  latitude_r):
    year,month,day=daytimeconv.date2ymd(aDT.date())
    doy=daytimeconv.date2doy(aDT.date())
    h,m=daytimeconv.time2hm(aDT.time())
    UTCtime_dd=daytimeconv.hms2dh(h,m,0)
    time_LAT=solar_geom_v5.UTC2LAT_r(UTCtime_dd,doy,longitude_r)
    
    #process solar geometry
    declin_r=solar_geom_v5.declin_r(year, doy, longitude_r)
    A0,h0=solar_geom_v5.sunposition_r(declin_r,latitude_r,time_LAT)
    h0refr=h0+solar_geom_v5.delta_h0refr(h0)
    return A0,h0, h0refr
    
def datasources_enforce_hourly(data_sources, minDT_total, maxDT_total, longit, latit ):
    minD = minDT_total.date()
    maxD = maxDT_total.date()
    dfb_begin=daytimeconv.date2dfb(minD)
    dfb_end=daytimeconv.date2dfb(maxD)
    num_dfbs=dfb_end - dfb_begin + 1
    num_times=24
    longitude_r = numpy.radians(longit)
    latitude_r = numpy.radians(latit)

    for data_source_key in data_sources.keys():
        data_source = data_sources[data_source_key]
        data = data_source['DATA']
        data_array = data['array']
        hourly_array=numpy.empty((num_dfbs,num_times,data_array.shape[1]+1),dtype=numpy.float64)
        hourly_array[:,:,:]=numpy.nan
        
        aTdelta=datetime.timedelta(hours=1)
        aTdelta_half=datetime.timedelta(minutes=30)
        for dfb in range(dfb_begin, dfb_end+1):
            dfb_idx=dfb-dfb_begin
            aD=daytimeconv.dfb2date(dfb)
            for hour in range(0,23+1):
                aT=datetime.time(hour)
                aDT_min=datetime.datetime.combine(aD,aT)
                aDTn_min=date2num(aDT_min)
                aDTn_max= date2num(aDT_min+aTdelta)
                aDT_center=aDT_min+(aTdelta_half)
                A0,h0, h0refr = calculate_solar_geom(aDT_center, longitude_r,  latitude_r)
                hourly_array[dfb_idx, hour,3]=h0refr
                aDTn_center=date2num(aDT_center)
                hourly_array[dfb_idx, hour,0] = aDTn_center
                wh = (data_array[:,0]>=aDTn_min) & (data_array[:,0]<aDTn_max)
                if wh.sum()>0:
                    hourly_array[dfb_idx, hour,1:3]=data_array[wh,1:].mean(axis=0)     
        data['array_hourly']=hourly_array
        data_source['DATA']=data
    return data_sources

def datasources_enforce_hourly2(data_sources, minDT_total, maxDT_total, longit, latit ):
    minD = minDT_total.date()
    maxD = maxDT_total.date()
    dfb_begin=daytimeconv.date2dfb(minD)
    dfb_end=daytimeconv.date2dfb(maxD)
    num_dfbs=dfb_end - dfb_begin + 1
    num_times=24
    longitude_r = numpy.radians(longit)
    latitude_r = numpy.radians(latit)

    for data_source_key in data_sources.keys():
        data_source = data_sources[data_source_key]
        data = data_source['DATA']
        data_array = data['array']
        print data_source_key, data_array.keys(), datetime.datetime.now()
        hourly_aDT = numpy.empty((num_dfbs,num_times),dtype=numpy.float64)
        hourly_aDT2004 = numpy.empty((num_dfbs,num_times),dtype=numpy.float64)
        hourly_val = numpy.empty((num_dfbs,num_times),dtype=numpy.float64)
        hourly_h0refr = numpy.empty((num_dfbs,num_times),dtype=numpy.float64)
        hourly_aDT[:,:]=numpy.nan
        hourly_aDT2004[:,:]=numpy.nan
        hourly_val[:,:]=numpy.nan
        hourly_h0refr[:,:]=numpy.nan
        print data_array['val'].shape, hourly_val.shape
#        hourly_array=numpy.empty((num_dfbs,num_times,data_array.shape[1]+1),dtype=numpy.float64)
#        hourly_array[:,:,:]=numpy.nan
        
        aTdelta=datetime.timedelta(hours=1)
        aTdelta_half=datetime.timedelta(minutes=30)
        for dfb in range(dfb_begin, dfb_end+1):
            dfb_idx=dfb-dfb_begin
            aD=daytimeconv.dfb2date(dfb)
            for hour in range(0,23+1):
                aT=datetime.time(hour)
                aDT_min=datetime.datetime.combine(aD,aT)
                aDTn_min=date2num(aDT_min)
                aDTn_max= date2num(aDT_min+aTdelta)
                aDT_center=aDT_min+(aTdelta_half)
                A0,h0, h0refr = calculate_solar_geom(aDT_center, longitude_r,  latitude_r)
#                hourly_array[dfb_idx, hour,3]=h0refr
                hourly_h0refr[dfb_idx, hour]=h0refr
                aDTn_center=date2num(aDT_center)
#                hourly_array[dfb_idx, hour,0] = aDTn_center
                hourly_aDT[dfb_idx, hour]= aDTn_center
                wh = (data_array['DTn']>=aDTn_min) & (data_array['DTn']<aDTn_max)
                if wh.sum()>0:
                    hourly_aDT2004[dfb_idx, hour]=data_array['DTn_2004'][wh].mean(axis=0)
                    hourly_val[dfb_idx, hour]=data_array['val'][wh].mean(axis=0)
#                    hourly_array[dfb_idx, hour,1:3]=data_array[wh,1:].mean(axis=0)     
        print hourly_aDT.mean(), hourly_val[hourly_val==hourly_val].mean(),hourly_h0refr.mean()
#        data['array_hourly']=hourly_array
        data['array_hourly']={'DTn':hourly_aDT,'DTn_2004':hourly_aDT2004,'val':hourly_val,'h0refr':hourly_h0refr}
        data_source['DATA']=data
    return data_sources


def datasources_enforce_hourly3(data_sources, minDT_total, maxDT_total, longit, latit ):
    minD = minDT_total.date()
    maxD = maxDT_total.date()
    dfb_begin=daytimeconv.date2dfb(minD)
    dfb_end=daytimeconv.date2dfb(maxD)
    num_dfbs=dfb_end - dfb_begin + 1
    num_times=24
    longitude_r = numpy.radians(longit)
    latitude_r = numpy.radians(latit)

    for data_source_key in data_sources.keys():
        
        #data to be integrated into hourly 
        data_source = data_sources[data_source_key]
        data = data_source['DATA']
        data_array = data['array']

        #empty hourly data arrays
        hourly_aDT = numpy.empty((num_dfbs,num_times),dtype=numpy.float64)
        hourly_aDT2004 = numpy.empty((num_dfbs,num_times),dtype=numpy.float64)
        hourly_val = numpy.empty((num_dfbs,num_times),dtype=numpy.float64)
        hourly_h0refr = numpy.empty((num_dfbs,num_times),dtype=numpy.float64)
        hourly_aDT[:,:]=numpy.nan
        hourly_aDT2004[:,:]=numpy.nan
        hourly_val[:,:]=numpy.nan
        hourly_h0refr[:,:]=numpy.nan

        # prepare aux arrays representing hour and dfb for faster selection         
        aux_h_array = numpy.empty(data_array['DTn'].shape, dtype=numpy.int16)
        aux_dfb_array = numpy.empty(data_array['DTn'].shape, dtype=numpy.int32)
        aux_h_array[:] = -9999
        aux_dfb_array[:] = -9999
        for i in range(0,data_array['DTn'].shape[0]):
            aux_DT = num2date(data_array['DTn'][i])
            aux_dfb_array[i] = daytimeconv.date2dfb(aux_DT.date())
            aux_h_array[i] = aux_DT.hour

        # do the hourly summarization       
        for dfb in range(dfb_begin, dfb_end+1):
            dfb_idx=dfb-dfb_begin
            aD=daytimeconv.dfb2date(dfb)
            wh_dfb = aux_dfb_array == dfb
            
            D_aux_h_array = aux_h_array[wh_dfb]
            D_aDT2004 = data_array['DTn_2004'][wh_dfb]
            D_val = data_array['val'][wh_dfb]
            
            for hour in range(0,23+1):  
                aDT_center=datetime.datetime.combine(aD,datetime.time(hour,30))
                A0,h0, h0refr = calculate_solar_geom(aDT_center, longitude_r,  latitude_r)
                hourly_h0refr[dfb_idx, hour]=h0refr
                aDTn_center=date2num(aDT_center)
                hourly_aDT[dfb_idx, hour]= aDTn_center
                wh = D_aux_h_array == hour
                if wh.sum()>0:
                    hourly_aDT2004[dfb_idx, hour]=D_aDT2004[wh].mean(axis=0)
                    hourly_val[dfb_idx, hour]=D_val[wh].mean(axis=0)
        
        #put results to data container
        data['array_hourly']={'DTn':hourly_aDT,'DTn_2004':hourly_aDT2004,'val':hourly_val,'h0refr':hourly_h0refr}
        data_source['DATA']=data
    return data_sources

        
def datasources_flatten_hourly(data_sources):
    for data_source_key in data_sources.keys():
        data_source = data_sources[data_source_key]
        data = data_source['DATA']
        data['array']['DTn'] = data['array_hourly']['DTn'].flatten()
        data['array']['DTn_2004'] = data['array_hourly']['DTn_2004'].flatten()
        data['array']['val'] = data['array_hourly']['val'].flatten()
        data['array']['h0refr'] = data['array_hourly']['h0refr'].flatten()
#        data_flat = data_array.reshape((data_array.shape[0]*data_array.shape[1],data_array.shape[2]))
#        data['array']=data_flat
        data_source['DATA']=data
    return data_sources
        
def datasources_array_reduce_to_overlapping_values(data_sources, fill_by_nan=False):
    
    ashape=None
    for data_source_key in data_sources.keys():
        data_source = data_sources[data_source_key]
        data = data_source['DATA']
        data_array = data['array']['val']
        if ashape is None: ashape = data_array.shape
        if ashape != data_array.shape:
            print 'Shape of arrays not equal, skipping reduction to overlapping values', ashape, data_array.shape
            return data_sources
    
    #find overlap of non NAN values
    total_wh=None
    for data_source_key in data_sources.keys():
        data_source = data_sources[data_source_key]
        data = data_source['DATA']
#        data_array = data['array']
        adata= data['array']['val']
        wh=adata==adata
        if total_wh is None: total_wh=wh.copy()
        total_wh&=wh
    
    #reduce
    
    for data_source_key in data_sources.keys():
#        print data['array'].keys()
        data_source = data_sources[data_source_key]
        data = data_source['DATA']
        data_array_dict = data['array']
        for array_key in data_array_dict.keys():
            data_array = data['array'][array_key]
            if fill_by_nan:
                data_array[numpy.logical_not(total_wh)]=numpy.nan
                data['array'][array_key]=data_array
            else: 
                data['array'][array_key]=data_array[total_wh]
        data_source['DATA']=data
    return data_sources
 
 
 
def datasources_read_data(data_sources, aD_begin, aD_end):
    for data_source_key in data_sources.keys(): 
        data_source = data_sources[data_source_key]
        src_type=data_source['SRC_TYPE']

        if src_type=='meteo':
            DSN = data_source['DSN']
            table = data_source['TAB']
            column = data_source['COL']
            print data_source_key, src_type, table,
            if not(db_utils.test_dsn(DSN) and db_utils.db_dtable_exist(DSN,table) and db_utils.db_dtable_descr_exist(DSN,table)):
                print "Can not find data table %s. Exiting,...." % (table)
                return False

            site_id=db_sites.db_dtables_get_siteID_for_dtable(DSN, table)
            if (site_id == None):
                print "Site ID not found for meteo table %s" % (table)
                return False
            name_short=db_sites.db_site_name_s_by_ID(DSN, site_id)
            
            coords = db_sites.db_site_coord(DSN, site_id)
            if (coords == None):
                print "Site coordinates not found for %s" % (table)
                return False
            longit,latit = coords
            
            elev = db_sites.db_site_Z(DSN, site_id)
            if (elev == None):
                print "Site elevation not found for %s" % (table)
                return False
                
            print '\n  ', "Station %s (%d) at %f,%f,%f" % (name_short, site_id, longit,latit,elev),
            
            aDTList, ghlist = db_sites.db_getdata(DSN, table, column, aD_begin, aD_end, use_flag=True)            
            
            if len(aDTList)<1:
                print  "Insufficient data for %s" % (table)
                return False

            minDT, maxDT = _calculate_DT_minmax(aDTList)
#            print '\n  ', len(aDTList), '\n  ', minDT, maxDT
            data_source['DATA']= {'dt_list': aDTList, 'val_list': ghlist, 'minDT': minDT, 'maxDT': maxDT}
        
        elif src_type=='sat_db':
            DSN = data_source['DSN']
            table = data_source['TAB']
            column = data_source['COL']
            site_id = data_source['ID']

            if not(db_utils.test_dsn(DSN) and db_utils.db_dtable_exist(DSN,table)):
                print "Can not find data table %s. Exiting,...." % (table)
                return False

            aDTList, ghlist=db_sites.db_getdata_sat(DSN, table, column, site_id, aD_begin, aD_end, time_start=datetime.time(hour=0,minute=0), time_end=datetime.time(hour=23,minute=59))
            
            if len(aDTList)<1:
                print  "Insufficient data for %s." % (table)
                return False
            
            minDT, maxDT = _calculate_DT_minmax(aDTList)
            print src_type, table, '\n  ',len(aDTList), '\n  ', minDT, maxDT
            data_source['DATA']= {'dt_list': aDTList, 'val_list': ghlist, 'minDT': minDT, 'maxDT': maxDT}
            
        else:
            print 'Data source type %s not supported' % (data_source[0])
            return False
    return data_sources, longit, latit

def prepare_data_sources(data_sources):
    #read data
    result = datasources_read_data(data_sources, aD_begin, aD_end)
    if result == False: return False
    data_sources, longit, latit = result



    # find min and max dates of all datasources 
    minDT_total, maxDT_total = datasources_get_total_dt_minmax(data_sources, True)
    print 'Total_data span',minDT_total, maxDT_total
    data_sources = datasources_reduce_by_minmax_DT(data_sources, minDT_total, maxDT_total )
        
    # convert lists to arrays
    data_sources = datasources_arrays_from_lists(data_sources)       
            

    
    print 'enforce hourly data'
    data_sources = datasources_enforce_hourly3(data_sources, minDT_total, maxDT_total, longit, latit )
    data_sources = datasources_flatten_hourly(data_sources)
    datasources_array_reduce_to_overlapping_values(data_sources, fill_by_nan=True)


    return data_sources

 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
if __name__ == "__main__":
    day_begin='20150801'
    day_end  ='20161231'
    aD_begin=daytimeconv.yyyymmdd2date(day_begin)
    aD_end=daytimeconv.yyyymmdd2date(day_end)

    # North
#    tables_ground = [ 'r_tat_10min_bsrn', 'r_sap_10min_bsrn', 'r_ish_10min_bsrn', 'r_xia_10min_bsrn', 'r_mnm_10min_bsrn', 'r_fua_10min_bsrn','r_yns_10min'  ]
#    # Center
#    tables_ground = [ 'r_kwa_10min_bsrn', 'r_bkt_60min', 'r_pnng_10min' ]
#    #South
#    tables_ground = [ 'r_coc_10min_bsrn', 'r_brm_10min', 'r_dar_10min', 'r_asp_10min', 'r_adl_10min', 'r_rckh_10min' ]
#    #All
    tables_ground = [ 'r_tat_10min_bsrn', 'r_sap_10min_bsrn', 'r_ish_10min_bsrn', 'r_xia_10min_bsrn', 'r_coc_10min_bsrn', 'r_mnm_10min_bsrn', 'r_kwa_10min_bsrn','r_fua_10min_bsrn', 'r_bkt_60min', 'r_pnng_10min', 'r_yns_10min', 'r_brm_10min', 'r_dar_10min', 'r_asp_10min', 'r_adl_10min', 'r_rckh_10min' ]

                     
    mtsat_model_version = 'v20b'
    himawari_model_version = 'v20a'

    DSN_METEO = "dbname=meteo_sites host=dbdata user=gm_user password=ibVal4"
    DSN_MTSAT = "dbname=mtsat_sites host=dbdata user=gm_user password=ibVal4"
    DSN_HIMAWARI = "dbname=himawari_sites host=dbdata user=gm_user password=ibVal4"

    
    
    
    all_mtsat_model_ghi_c = None
    all_mtsat_model_ghi = None
    all_mtsat_model_dni = None
    all_mtsat_model_ci = None
    all_mtsat_model_ktm = None
    all_mtsat_model_h0 = None
    
    all_him_model_ghi_c = None
    all_him_model_ghi = None
    all_him_model_dni = None
    all_him_model_ci = None
    all_him_model_ktm = None
    all_him_model_h0 = None
    
    for tabl_ground in tables_ground:
        #input model results - note that model outputs for all sites are stored in one table
        ID=db_sites.db_dtables_get_siteID_for_dtable(DSN_METEO, tabl_ground)

        data_sources={}
        #METEO
        column="gh"
        data_sources['ground_ghi']={'SRC_TYPE': 'meteo', 'DSN': DSN_METEO, 'TAB': tabl_ground, 'COL': column}
        
        #MTSAT
        table='res_model_%d_%s' % (ID,mtsat_model_version)
        column="ghi_c"
        data_sources['mtsat_model_ghi_c']={'SRC_TYPE': 'sat_db', 'DSN': DSN_MTSAT, 'TAB': table, 'COL': column, 'ID': ID}
        column="ghi"
        data_sources['mtsat_model_ghi']={'SRC_TYPE': 'sat_db', 'DSN': DSN_MTSAT, 'TAB': table, 'COL': column, 'ID': ID}
        column="dni_c"
        data_sources['mtsat_model_dni_c']={'SRC_TYPE': 'sat_db', 'DSN': DSN_MTSAT, 'TAB': table, 'COL': column, 'ID': ID}
        column="dni"
        data_sources['mtsat_model_dni']={'SRC_TYPE': 'sat_db', 'DSN': DSN_MTSAT, 'TAB': table, 'COL': column, 'ID': ID}
        column="ci"
        data_sources['mtsat_model_ci']={'SRC_TYPE': 'sat_db', 'DSN': DSN_MTSAT, 'TAB': table, 'COL': column, 'ID': ID}
        column="ktm"
        data_sources['mtsat_model_ktm']={'SRC_TYPE': 'sat_db', 'DSN': DSN_MTSAT, 'TAB': table, 'COL': column, 'ID': ID}

        #HIMAWARI
        table='res_model_%d_%s' % (ID,himawari_model_version)
        column="ghi_c"
        data_sources['him_model_ghi_c']={'SRC_TYPE': 'sat_db', 'DSN': DSN_HIMAWARI, 'TAB': table, 'COL': column, 'ID': ID}
        column="ghi"
        data_sources['him_model_ghi']={'SRC_TYPE': 'sat_db', 'DSN': DSN_HIMAWARI, 'TAB': table, 'COL': column, 'ID': ID}
        column="dni_c"
        data_sources['him_model_dni_c']={'SRC_TYPE': 'sat_db', 'DSN': DSN_HIMAWARI, 'TAB': table, 'COL': column, 'ID': ID}
        column="dni"
        data_sources['him_model_dni']={'SRC_TYPE': 'sat_db', 'DSN': DSN_HIMAWARI, 'TAB': table, 'COL': column, 'ID': ID}
        column="ci"
        data_sources['him_model_ci']={'SRC_TYPE': 'sat_db', 'DSN': DSN_HIMAWARI, 'TAB': table, 'COL': column, 'ID': ID}
        column="ktm"
        data_sources['him_model_ktm']={'SRC_TYPE': 'sat_db', 'DSN': DSN_HIMAWARI, 'TAB': table, 'COL': column, 'ID': ID}



        #-----------------------------
        
        #read data sources
        data_sources=prepare_data_sources(data_sources)
        print data_sources.keys()

        mtsat_model_ghi_c=data_sources['mtsat_model_ghi_c']['DATA']['array']['val']
        mtsat_model_ghi=data_sources['mtsat_model_ghi']['DATA']['array']['val']
        mtsat_model_dni=data_sources['mtsat_model_dni']['DATA']['array']['val']
        mtsat_model_dni_c=data_sources['mtsat_model_dni_c']['DATA']['array']['val']
        mtsat_model_ci=data_sources['mtsat_model_ci']['DATA']['array']['val']
        mtsat_model_ktm=data_sources['mtsat_model_ktm']['DATA']['array']['val']
        mtsat_model_h0=data_sources['mtsat_model_ghi_c']['DATA']['array']['h0refr']

        him_model_ghi_c=data_sources['him_model_ghi_c']['DATA']['array']['val']
        him_model_ghi=data_sources['him_model_ghi']['DATA']['array']['val']
        him_model_dni=data_sources['him_model_dni']['DATA']['array']['val']
        him_model_dni_c=data_sources['him_model_dni_c']['DATA']['array']['val']
        him_model_ci=data_sources['him_model_ci']['DATA']['array']['val']
        him_model_ktm=data_sources['him_model_ktm']['DATA']['array']['val']
        him_model_h0=data_sources['him_model_ghi_c']['DATA']['array']['h0refr']
        
        
        # ktm for ground 
        wh = (him_model_ghi_c > 15) & (mtsat_model_ghi_c > 15) & (him_model_h0>numpy.radians(6.))
        
        
        # if we want to add additional inital data to CI anf KTM
        if him_model_ktm.shape[0] > 1:
            if all_mtsat_model_ktm is None:
                all_mtsat_model_ghi_c = mtsat_model_ghi_c
                all_mtsat_model_ghi = mtsat_model_ghi
                all_mtsat_model_dni_c = mtsat_model_dni_c
                all_mtsat_model_dni = mtsat_model_dni
                all_mtsat_model_ci = mtsat_model_ci
                all_mtsat_model_ktm = mtsat_model_ktm
                all_mtsat_model_h0 = mtsat_model_h0
                
                all_him_model_ghi_c = him_model_ghi_c
                all_him_model_ghi = him_model_ghi
                all_him_model_dni_c = him_model_dni_c
                all_him_model_dni = him_model_dni
                all_him_model_ci = him_model_ci
                all_him_model_ktm = him_model_ktm
                all_him_model_h0 = him_model_h0
            else:
                all_mtsat_model_ghi_c = numpy.hstack((all_mtsat_model_ghi_c, mtsat_model_ghi_c))
                all_mtsat_model_ghi = numpy.hstack((all_mtsat_model_ghi, mtsat_model_ghi))
                all_mtsat_model_dni_c = numpy.hstack((all_mtsat_model_dni_c, mtsat_model_dni_c))
                all_mtsat_model_dni = numpy.hstack((all_mtsat_model_dni, mtsat_model_dni))
                all_mtsat_model_ci = numpy.hstack((all_mtsat_model_ci, mtsat_model_ci))
                all_mtsat_model_ktm = numpy.hstack((all_mtsat_model_ktm, mtsat_model_ktm))
                all_mtsat_model_h0 = numpy.hstack((all_mtsat_model_h0, mtsat_model_h0))
                
                all_him_model_ghi_c = numpy.hstack((all_him_model_ghi_c, him_model_ghi_c))
                all_him_model_ghi = numpy.hstack((all_him_model_ghi, him_model_ghi))
                all_him_model_dni_c = numpy.hstack((all_him_model_dni_c, him_model_dni_c))
                all_him_model_dni = numpy.hstack((all_him_model_dni, him_model_dni))
                all_him_model_ci = numpy.hstack((all_him_model_ci, him_model_ci))
                all_him_model_ktm = numpy.hstack((all_him_model_ktm, him_model_ktm))
                all_him_model_h0 = numpy.hstack((all_him_model_h0, him_model_h0))
        

    import pylab
    fig1 = figure(num=1,figsize=(15,12),facecolor='w')
    fig1.clear()
    
    
    ax=pylab.subplot(3,3,1)
    ax.plot(all_mtsat_model_ghi_c, all_him_model_ghi_c, 'r.', ms=1)
    pylab.xlabel('MTSAT')
    pylab.ylabel('HIMAWARI')
    pylab.title("GHIc")
    wh=(all_him_model_ghi_c==all_him_model_ghi_c)&(all_mtsat_model_ghi_c==all_mtsat_model_ghi_c)
    bias=100*(all_him_model_ghi_c[wh].sum()-all_mtsat_model_ghi_c[wh].sum())/all_mtsat_model_ghi_c[wh].sum()
    pylab.text(0.02,0.95,"bias: %.1f "%(bias), {'color': 'g', 'fontsize': 8}, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    pylab.grid()

    ax=pylab.subplot(3,3,2)
    ax.plot(all_mtsat_model_ghi, all_him_model_ghi, 'r.', ms=1)
    pylab.xlabel('MTSAT')
    pylab.ylabel('HIMAWARI')
    pylab.title("GHI")
    wh=(all_him_model_ghi==all_him_model_ghi)&(all_mtsat_model_ghi==all_mtsat_model_ghi)
    bias=100*(all_him_model_ghi[wh].sum()-all_mtsat_model_ghi[wh].sum())/all_mtsat_model_ghi[wh].sum()
    pylab.text(0.02,0.95,"bias: %.1f "%(bias), {'color': 'g', 'fontsize': 8}, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    pylab.grid()

    ax=pylab.subplot(3,3,4)
    ax.plot(all_mtsat_model_dni_c, all_him_model_dni_c, 'r.', ms=1)
    pylab.xlabel('MTSAT')
    pylab.ylabel('HIMAWARI')
    pylab.title("DNIc")
    wh=(all_him_model_dni_c==all_him_model_dni_c)&(all_mtsat_model_dni_c==all_mtsat_model_dni_c)
    bias=100*(all_him_model_dni_c[wh].sum()-all_mtsat_model_dni_c[wh].sum())/all_mtsat_model_dni_c[wh].sum()
    pylab.text(0.02,0.95,"bias: %.1f "%(bias), {'color': 'g', 'fontsize': 8}, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    pylab.grid()

    ax=pylab.subplot(3,3,5)
    ax.plot(all_mtsat_model_dni, all_him_model_dni, 'r.', ms=1)
    pylab.xlabel('MTSAT')
    pylab.ylabel('HIMAWARI')
    pylab.title("DNI")
    wh=(all_him_model_dni==all_him_model_dni)&(all_mtsat_model_dni==all_mtsat_model_dni)
    bias=100*(all_him_model_dni[wh].sum()-all_mtsat_model_dni[wh].sum())/all_mtsat_model_dni[wh].sum()
    pylab.text(0.02,0.95,"bias: %.1f "%(bias), {'color': 'g', 'fontsize': 8}, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    pylab.grid()

    ax=pylab.subplot(3,3,3)
    wh=(all_him_model_dni==all_him_model_dni)&(all_him_model_ghi==all_him_model_ghi)
    kt_b = all_him_model_dni[wh]/all_him_model_dni_c[wh]
    kt_g = all_him_model_ghi[wh]/all_him_model_ghi_c[wh]
    
    ax.plot(kt_g, kt_b, 'r.', ms=1)
    pylab.xlabel('kt GHI ')
    pylab.ylabel('kt DNI')
    pylab.title("KTM GHI - DNI")
    pylab.grid()


    ax=pylab.subplot(3,3,7)
    ax.plot(all_mtsat_model_ci, all_him_model_ci, 'r.', ms=1)
    pylab.xlabel('MTSAT')
    pylab.ylabel('HIMAWARI')
    pylab.title("CI")
    wh=(all_him_model_ci==all_him_model_ci)&(all_mtsat_model_ci==all_mtsat_model_ci)
    bias=100*(all_him_model_ci[wh].sum()-all_mtsat_model_ci[wh].sum())/all_mtsat_model_ci[wh].sum()
    pylab.text(0.02,0.95,"bias: %.1f "%(bias), {'color': 'g', 'fontsize': 8}, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    pylab.grid()

    ax=pylab.subplot(3,3,8)
    ax.plot(all_mtsat_model_ktm, all_him_model_ktm, 'r.', ms=1)
    pylab.xlabel('MTSAT')
    pylab.ylabel('HIMAWARI')
    pylab.title("KTM")
    wh=(all_him_model_ktm==all_him_model_ktm)&(all_mtsat_model_ktm==all_mtsat_model_ktm)
    bias=100*(all_him_model_ktm[wh].sum()-all_mtsat_model_ktm[wh].sum())/all_mtsat_model_ktm[wh].sum()
    pylab.text(0.02,0.95,"bias: %.1f "%(bias), {'color': 'g', 'fontsize': 8}, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    pylab.grid()

    ax=pylab.subplot(3,3,6)
    wh=(all_him_model_dni==all_him_model_dni)&(all_mtsat_model_dni==all_mtsat_model_dni)
    kt_him_b = all_him_model_dni[wh]/all_him_model_dni_c[wh]
    kt_mtsat_b = all_mtsat_model_dni[wh]/all_mtsat_model_dni_c[wh]
    
    ax.plot(kt_mtsat_b, kt_him_b, 'r.', ms=1)
    pylab.xlabel('kt_MTSAT_b')
    pylab.ylabel('kt_HIM_b')
    pylab.title("KT_b")
    wh=(kt_him_b==kt_him_b)&(kt_mtsat_b==kt_mtsat_b)
    bias=100*(kt_him_b[wh].sum()-kt_mtsat_b[wh].sum())/kt_mtsat_b[wh].sum()
    pylab.text(0.02,0.95,"bias: %.1f "%(bias), {'color': 'g', 'fontsize': 8}, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    pylab.grid()

    ax=pylab.subplot(3,3,9)
    wh=(all_him_model_dni==all_him_model_dni)&(all_mtsat_model_dni==all_mtsat_model_dni)
    kt_him_b = all_him_model_dni[wh]/all_him_model_dni_c[wh]
    kt_mtsat_b = all_mtsat_model_dni[wh]/all_mtsat_model_dni_c[wh]
    
    ax.plot(all_him_model_ktm[wh], kt_him_b-kt_mtsat_b, 'r.', ms=1)
    pylab.xlabel('KT_HIM_g')
    pylab.ylabel('KT_HIM_b-KT_MTSAT_b')
    pylab.title("KT_HIM_b-KT_MTSAT_b vs KT_HIM_g")
#    wh=(kt_him_b==kt_him_b)&(kt_mtsat_b==kt_mtsat_b)
#    bias=100*(kt_him_b[wh].sum()-kt_mtsat_b[wh].sum())/kt_mtsat_b[wh].sum()
#    pylab.text(0.02,0.95,"bias: %.1f "%(bias), {'color': 'g', 'fontsize': 8}, horizontalalignment='left', verticalalignment='center', transform = ax.transAxes)
    pylab.grid()

    pylab.show()
        
    exit()

    


    


