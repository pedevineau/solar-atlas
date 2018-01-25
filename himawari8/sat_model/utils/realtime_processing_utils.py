'''
Created on Feb 4, 2016

@author: tomas
'''
import netCDF4

import psycopg2
from general_utils import daytimeconv
from general_utils import db_utils
from general_utils import latlon
from himawari8.sat_model.utils import himawari_nc_latlontools
            
def get_max_rawsat_data_slot_from_db(dsn_sat_archive_dict, dfb_end, data_table='processed'):
    dsn_sat_archive= db_utils.DSNDictToDSNString(dsn_sat_archive_dict)
    if not db_utils.test_dsn(dsn_sat_archive,verbose=False):
        print 'Unable to access raw sat data archive DB'
        return None

    date_str = daytimeconv.dfb2yyyymmdd(dfb_end, separator='-')
    query = "select max(slot) from \"%s\"  where datetime::date='%s' and channel_name='VIS064_2000' group by datetime::date " % (data_table, date_str)

    conn = psycopg2.connect(dsn_sat_archive)
    curs = conn.cursor()

    try:
        curs.execute(query)
    except:
        print "Unable to execute db query for most recent slot with available data"
        print query
        return None

    row = curs.fetchone()
    if row != None:
        max_slot = row[0]
    else:
        max_slot = None
    conn.close()
    
    return max_slot

def get_max_processed_data_slot_from_nc(dfb, segment_list, realtime_paths, file_time_segmentation, selection_type='min'):   
    nclist=[]
    dfbStart = dfb
    dfbEnd = dfb
    chan='GHI'
    
    for seg_c, seg_r in segment_list:
        outdata_suffix="_c%d_r%d" % (seg_c, seg_r)
        time_strings = himawari_nc_latlontools.filename_timestrings_create(dfbStart, dfbEnd, file_time_segmentation=file_time_segmentation)
        time_string = time_strings.keys()[0]
        res = himawari_nc_latlontools.create_outnc_filename(time_string, chan, realtime_paths, file_suffix=outdata_suffix, check_exist=True)
        if res[0] != '':
            nclist.append(res[0])

    slotlist = []
    for ncfile in nclist:
        if (ncfile is not None) and (ncfile!=''):
            rootgrp = netCDF4.Dataset(ncfile, 'r', format='NETCDF4')
            data = rootgrp.variables['GHI'][0,:,0,0]
            nc_slot_begin = rootgrp.variables['slot'][:].min()
            rootgrp.close()
            s=-9 
            for i in range(0, data.shape[0]):
                if data[i]==data[i]:
                    s=i+nc_slot_begin
            if s >= 0:
                slotlist.append(s)

    if len(slotlist)<1:
        return None
    if selection_type=='min':
        s = min(slotlist)
    elif selection_type=='max':
        s = max(slotlist)
    
    return s
