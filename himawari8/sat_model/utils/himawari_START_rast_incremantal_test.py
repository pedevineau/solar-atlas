#! /usr/bin/env python

import datetime
import os
import sys

from aerosols.model import atmosphere_param
from general_utils import basic_mail
from general_utils import daytimeconv
from general_utils import latlon
from himawari8.sat_model import himawari_mdl_core
from himawari8.sat_model.utils import himawari_nc_latlontools


if __name__ == "__main__":
    mail_notification='' #email address to send finish notification to, Use '' to avoid mail notification
    
    dfbStart = daytimeconv.yyyymmdd2dfb('20151001')
#    dfbEnd = daytimeconv.yyyymmdd2dfb('20160331')
    dfbEnd = daytimeconv.yyyymmdd2dfb('20160331')
    
    slotStart = 1
    slotEnd = 144

    resolution =  2./60.
    segments_to_calculate = latlon.expand_segments([[63, 63, 10, 10]]) 
    segments_to_calculate = latlon.expand_segments([[63, 63, 11, 11]]) 
#    segments_to_calculate = latlon.expand_segments([[66, 66, 12, 12]]) 
#    segments_to_calculate = latlon.expand_segments([[60, 60, 8, 8]])
#    segments_to_calculate = latlon.expand_segments([[61, 61, 9, 9]]) 
#    segments_to_calculate = latlon.expand_segments([[71, 71, 26, 26]]) #strong forward
#    segments_to_calculate = latlon.expand_segments([[65, 65, 26, 26]])
#    segments_to_calculate = latlon.expand_segments([[61, 61, 22, 22]])
#    segments_to_calculate = latlon.expand_segments([[64, 64, 19, 19]])
#    segments_to_calculate = latlon.expand_segments([[58, 58, 18, 18]])
#    segments_to_calculate = latlon.expand_segments([[59, 59, 17, 17]])
#    segments_to_calculate = latlon.expand_segments([[55, 55, 17, 17]]) #M
#    segments_to_calculate = latlon.expand_segments([[69, 69, 15, 15]]) #M
    
    #if  process_bbox specified - it overrides the segment
    import numpy
    w, e, s, n, resolution = 135.+(0*2./60.), 135. +(1*2./60.), 40.-(1*2./60.), 40.-(0*2./60.), 2./60.   #UL corner
    w, e, s, n, resolution = 135.+(0*2./60.), 135. +(1*2./60.), 35.+(0*2./60.), 35.+(1*2./60.), 2./60.   #LL corner
#    w, e, s, n, resolution = 136., 138. , 35., 36., 2./60.   #LL corner
#    w, e, s, n, resolution = 135., 136. , 35.-(2*2./60.), 35., 2./60.   #LL corner
    process_bbox=latlon.bounding_box(w, e, s, n, int(numpy.floor(((e-w)/resolution)+0.5)), int(numpy.floor(((n-s)/resolution)+0.5)), resolution)
#    process_bbox=None
    
    
    dsnCalibDict={'db':'himawari_archive', 'host':'dbsatarchive', 'user':'sat_user', 'password':'uWrox5'}
    dsnGeomDict={'db':'himawari_archive', 'host':'dbsatarchive', 'user':'sat_user', 'password':'uWrox5'}

    
    #use previous results (if available) to init lower bound calculation (uses LB and LBland parameters)
    init_by_previousday=True

    save_UB = False  #only when UB needs to be recalculated from full time series
    
    do_parallel=True
    ncpus='autodetect'# minimum one worker on local machine if no servers
    
    
    trail_window_size=30
    
    segment_sizex=8; segment_sizey=8
    
    
    #local path where all data are stored
    model_version = 'v20_t'
#    snow_data_path = '/home0/model_data/data_snow/'
    geom_data_path = "/home1/model_data_himawari/data_geom/" 
    dem_data_path = "/home0/model_data_goes/data_geom/" 
    satelliteDataDirs = ["/home1/model_data_himawari/sat_data_procseg/"]
    aod_path = '/home0/model_data/data_aod/'
    out_data_path = '/home1/model_data_himawari/data_output/'+model_version
    file_time_segmentation='month' # archseg month day
    
    #atmosphere section
    aod_ncfiles, wv_ncfiles = himawari_mdl_core.init_aod_nc_files_v21p()
    # ATMOSPHERE TYPES
    # primary_type : 1-rural (water and dust-like), 4-maritime (rural and sea salt), 5-urban (rural and soot-like), 6-tropospheric (rural mixture)
    # secondary_type : -9 for None or atmosphere type code
    # secondary_weight_file : number <0.0,1.0> or NC file with weight data
    atmosph=atmosphere_param.atmosphere_param(primary_type = 6, secondary_type = 5, secondary_weight_file = aod_path+'aod_urban_type_weight_v1.nc')
    atmosph.hourly_data_persistence = 0
    atmosph.aod_ncfiles = aod_ncfiles
    atmosph.wv_ncfiles = wv_ncfiles
    atmosph.aod_path = aod_path
    atmosph.do_smoothing=True
    atmosph.do_extreme_correction=True
#    atmosph.extreme_correction_params=[0.1,0.25,0.85,0.75]
    atmosph.extreme_correction_params=[0.0,0.6,1.0,0.85]
    
    #write progres and result of processing to onitoring database
    datasources_monitor_use = False
    datasources_monitor_params={}
    datasources_monitor_params['datasrc_name'] = 'HIMAWARI'
    datasources_monitor_params['proc_group'] = "production"
    datasources_monitor_params['proc_host'] = "moon"
    datasources_monitor_params['storage_host'] = "moon"
    datasources_monitor_params['version'] = model_version
    datasources_monitor_params['description'] = 'test model'

    
    
    ########################
    #end of inputs
    ########################
    satInfoDict={\
            'satList' : ["H08"],\
            'chanNameList' : ['VIS064_2000', 'VIS160_2000', 'IR124_2000', 'IR390_2000'],\
#            'chanNameList' : ['VIS064_2000', 'VIS160_2000', 'IR124_2000', 'IR390_2000'],\
            'satelliteDataDirs' : satelliteDataDirs,\
                 }
    
    #check AOI to calculate - only one segment (or intrasegemnt subset) is allowed
    
    if process_bbox is not None:
        seg_c, seg_r = latlon.get_5x5_seg(process_bbox.center()[0],process_bbox.center()[1])
        seg_bbox=latlon.get_5x5_seg_bbox(arow=seg_r, acol=seg_c, resolution=resolution)
        if not seg_bbox.contains(process_bbox):
            print ('AOI overlaps 2 or more 5x5 deg. data segments. Limit it one. Exit.')
            exit()
        print 'process box overrides the segments. '  
        segments_to_calculate=(([seg_c, seg_r],))
        
    if dfbStart>dfbEnd:
        print "Requested dates for calculation not coherent. Exit."
        exit()
    
#    if ((not os.path.exists(snow_data_path)) or (os.path.isfile(snow_data_path))):
#        print "Snow data path %s not found" % snow_data_path
#        sys.exit()
        
    if ((not os.path.exists(dem_data_path)) or (os.path.isfile(dem_data_path))):
        print "Dem data path %s not found" % dem_data_path
        sys.exit()
        
    if ((not os.path.exists(geom_data_path)) or (os.path.isfile(geom_data_path))):
        print "Geom (UB) data path %s not found" % geom_data_path
        sys.exit()
        
        
    res=False
    for sat_data_path in satelliteDataDirs:
        res = res or ((os.path.exists(sat_data_path)) and (not os.path.isfile(sat_data_path)))
    if not res:
        print "Satellite data path %s not found" % ' '.join(satelliteDataDirs)
        sys.exit()
    if ((not os.path.exists(aod_path)) or (os.path.isfile(aod_path))):
        print "AOD data path %s not found" % aod_path
        sys.exit()
    if ((not os.path.exists(out_data_path)) or (os.path.isfile(out_data_path))):
        print "Output data path %s not found" % out_data_path
        sys.exit()
    
    list_succ=[]
    list_fail=[]
    aTotalStartTime = datetime.datetime.now()
    for seg_c, seg_r in segments_to_calculate:
        print 'seg(c,r): %d %d' %( seg_c, seg_r)
        if  process_bbox is not None:
            process_bbox_current = process_bbox
        else:
            process_bbox_current=latlon.get_5x5_seg_bbox(seg_r, seg_c, resolution, seg_size=5.)

        #sat data
        # note that order of r,c is different from output files (c,r)
        satdata_suffix="_r%02d_c%02d" % (seg_r, seg_c)
        
    
        #suffix added to output NETCDF file names
        outdata_suffix="_c%d_r%d" % (seg_c, seg_r)
    
        
    #    aux data used in sat data classification %s auto replaced by region suffix
        auxdata_file_dict = {"altitude": [dem_data_path+"/dem_strm120.nc", "dem"], \
#                            "SDWE":[snow_data_path+"/*_sdwe_%s.nc","sdwe"],\
                            "UB":[geom_data_path+"/himawari_UB"+outdata_suffix+".nc","UB"]}
    
        #output data
        outdata_path_dict={"LB": out_data_path, "LBclass": out_data_path, "LBland": out_data_path, "CLI": out_data_path,\
                         "CI": out_data_path, "KTM": out_data_path, "GHIc": out_data_path, "GHI": out_data_path,\
                        "DNIc": out_data_path, "DNI": out_data_path, "GHIcor": out_data_path, "DNIcor": out_data_path, \
                        "CI_flag": out_data_path}
        
        out_channels=("GHI", "GHIc", "DNI", "DNIc", "KTM", "CI", "CLI", "LB", "LBclass", "LBland", "CI_flag") # channels to write to output NC files
    
    
        print "Using settings from", sys.argv[0]
        print "Model version", model_version
        print "Pre-defined AOI %s within 5x5 deg. data segment %d %d" % (str(process_bbox_current), seg_c, seg_r)
        print "Pre-defined dates:",daytimeconv.dfb2yyyymmdd(dfbStart), daytimeconv.dfb2yyyymmdd(dfbEnd)
        print "Pre-defined input slots:", slotStart, slotEnd
        print "Parallel processing:",do_parallel," number of CPUs:",ncpus
        print "Init by previous day (LB, LBclass):",init_by_previousday
        print "Save Upper bound (UB):",save_UB
        print "Path - geom data (UB):", geom_data_path
        print "Path - dem data:", dem_data_path
#        print "Path - snow data:",     snow_data_path
        print "Path - raw sat data:", ' '.join(satelliteDataDirs)
        print "Path - AOD:",aod_path 
        print "Path - output data:", out_data_path
        print "Output data suffix:",outdata_suffix
        print "Trial window size:", trail_window_size
        print '---end of inputs---'
    
        
        seg_bbox=latlon.get_5x5_seg_bbox(seg_r, seg_c, resolution, seg_size=5.)
        
        try:
            out_files_dict = himawari_nc_latlontools.check_output_nc_files(dfbStart, dfbEnd, 1, 144, seg_bbox, outdata_path_dict, out_channels, outdata_suffix, model_version=model_version, file_time_segmentation=file_time_segmentation)
        except:
            list_fail.append("c%d_r%d" % (seg_c, seg_r))
            continue
        if len(out_files_dict)<1:
            list_fail.append("c%d_r%d" % (seg_c, seg_r))
            print 'No output file. Skipping segment.',seg_c, seg_r
            continue
    
    
    
        if mail_notification != '':
            basic_mail.mail_process_message_ssl(reciever_to=mail_notification, message='processing segment started %d, %d.' % (seg_c, seg_r))
        ##################################
        #PROCESSING
        ##################################    
        aStartTime = datetime.datetime.now()
        print 'Start', aStartTime
    
    
        for dfb in range(dfbStart, dfbEnd):
            dfbStart_t = dfb
            dfbEnd_t  = dfb+1 
            result = himawari_mdl_core.sat_model_rast_pp( dsnCalibDict, dsnGeomDict, dfbStart_t, dfbEnd_t, slotStart, slotEnd, satInfoDict, atmosph, out_channels, auxdata_file_dict, satdata_suffix, out_files_dict, outdata_path_dict, outdata_suffix, process_bbox_current, seg_bbox,  do_parallel=do_parallel, ncpus=ncpus, segment_sizex=segment_sizex, segment_sizey=segment_sizey,trail_window_size=trail_window_size, init_by_previousday=init_by_previousday, save_UB=save_UB, file_time_segmentation=file_time_segmentation)

        if not(result):
            print 'Processing failed'
            list_fail.append("c%d_r%d" % (seg_c, seg_r))
        else:
            list_succ.append("c%d_r%d" % (seg_c, seg_r))
            
        segment_processing_time = datetime.datetime.now() - aStartTime
        print 'End', datetime.datetime.now(), "Segment processing time:", segment_processing_time

    total_processing_time = datetime.datetime.now() - aTotalStartTime
    message='Processing all segments finished. \n'
    message+='Time: %s\n'%(str(total_processing_time))
    message+='Successfully processed:%d/%d\n'%(len(list_succ),len(list_succ)+len(list_fail))
    message+='Success segments: %s\n'%(', '.join(list_succ))
    message+='Failure segments: %s\n'%(', '.join(list_fail))
    if mail_notification != '':
        basic_mail.mail_process_message_ssl(reciever_to=mail_notification, message=message)
    print message
    exit()
