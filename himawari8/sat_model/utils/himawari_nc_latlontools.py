'''
Created on Feb 13, 2016

@author: tomas
'''
import sys
import os
import glob
import numpy
import math
import netCDF4
from general_utils import  daytimeconv
from general_utils import  latlon_nctools
from general_utils import  latlon


def isiterable(obj):
    return hasattr(obj, '__iter__')

def check_file_existence(afile):
    if (not os.path.exists(afile)) or (not os.path.isfile(afile)):
        return False
    return True

def filename_timestrings_create(dfb_min,dfb_max,file_time_segmentation='archseg'):    
    #file_time_segmentation  archseg month day
    #create list of timestrings used in nc filenames 
    timestring_dict={}    
    if file_time_segmentation == 'archseg':
        year_arch_pais_list = daytimeconv.dfb_minmax2yea_archseg_list(dfb_min, dfb_max)
        for year, archseg in year_arch_pais_list:
            timestring = str(year)+"_"+str(archseg)
            timestring_dict[timestring] = daytimeconv.archsegy2dfbminmax(archseg, year)
    elif file_time_segmentation == 'month':
        year_month_pais_list = daytimeconv.dfb_minmax2year_month_list(dfb_min, dfb_max)
        for year, month in year_month_pais_list:
            timestring = "%d_%02d"%(year,month)
            timestring_dict[timestring] = daytimeconv.monthyear2dfbminmax(month, year)
    elif file_time_segmentation == 'day':
        for dfb in range(dfb_min,dfb_max+1):
            timestring = daytimeconv.dfb2yyyymmdd(dfb)
            timestring_dict[timestring] = [dfb,dfb]
    else:
        print 'Unsupported file time segmentation %s' % (str(file_time_segmentation))
        return {}
    return timestring_dict

def create_outnc_filename(time_string,chan, disk_preffix_dict, file_suffix="", check_exist=False):

    ncfilebase=chan+"_"+time_string+file_suffix+".nc"
    ncfile=''
    if disk_preffix_dict is not None:
        if chan in disk_preffix_dict.keys():
            ncfilepath=disk_preffix_dict[chan]
        else:
            if 'default' in disk_preffix_dict.keys():
                ncfilepath=disk_preffix_dict['default']
            else:
                return (ncfile,ncfilebase)
            
        if isiterable(ncfilepath):
            print >> sys.stderr , 'warning: multipath nc dictionary not supported, using first path'
            ncfilepath=ncfilepath[0]
        ncfile=os.path.join(ncfilepath,ncfilebase)
    if ((not os.path.exists(ncfile)) or(not os.path.isfile(ncfile))) and (check_exist):
        #print "Warning: nc file", ncfile, "for dfb", str(adfb) ,"not found, skipping...."
        return ('','')
    return(ncfile,ncfilebase)



# if more than one path defined create path for all existing
def create_outnc_filename_multi(time_string,chan,disk_preffix_dict, file_suffix=""):
    output_list=[]
    ncfilebase=chan+"_"+time_string+file_suffix+".nc"
    ncfile=''
    if disk_preffix_dict is not None:
        if chan in disk_preffix_dict.keys():
            ncfilepaths=disk_preffix_dict[chan]
        else:
            if 'default' in disk_preffix_dict.keys():
                ncfilepaths=disk_preffix_dict['default']
            else:
                return ([ncfile,ncfilebase])
        if isiterable(ncfilepaths):
            for ncfilepath in ncfilepaths:
                ncfile=os.path.join(ncfilepath,ncfilebase)
                output_list.append([ncfile,ncfilebase])
        else:
            ncfile=os.path.join(ncfilepaths,ncfilebase)
            output_list.append([ncfile,ncfilebase])
            
    return output_list


def get_outdata_ncfile_parameters(ncfile, channel):
    #get min max slots of nc file
    rootgrp = netCDF4.Dataset(ncfile, 'r')
    dimDict = rootgrp.dimensions
    varDict=rootgrp.variables

    #get min and max slot number from in and out files
    ncslotmin=varDict['slot'].valid_range[0]
    ncslotmax=varDict['slot'].valid_range[1]
    nccolmin=varDict['longitude'].valid_range[0]
    nccolmax=varDict['longitude'].valid_range[1]
    nccols=len(dimDict['longitude'])
    ncrowmin=varDict['latitude'].valid_range[0]
    ncrowmax=varDict['latitude'].valid_range[1]
    ncrows=len(dimDict['latitude'])
    
    ncres=(nccolmax-nccolmin)/nccols
    nc_bbox=latlon.bounding_box(nccolmin, nccolmax, ncrowmin, ncrowmax, nccols, ncrows, ncres)
    
    var_attribs = varDict[channel].__dict__
    if ("missing_value" in var_attribs) :
        noval=varDict[channel].missing_value
    else:
        noval=None
    
    if ("valid_range" in var_attribs) :
        valid_range=varDict[channel].valid_range
    else:
        valid_range=[None, None]
    
    ncdfb_min, ncdfb_max = None, None
    if len(dimDict['dfb'])>=1:
        ncdfbs=varDict['dfb'][:]
        if ncdfbs[ncdfbs>0].sum()>0:
            ncdfb_min=ncdfbs[ncdfbs>0].min()
            ncdfb_max=ncdfbs[ncdfbs>0].max()
    rootgrp.close()

    ncdfb_0=ncdfb_min       

    return ([ncfile, [ncdfb_min, ncdfb_max ],ncdfb_0, [ncslotmin,ncslotmax],nc_bbox , noval, valid_range])


def check_output_nc_files(dfb_begin, dfb_end, aslot_begin, aslot_end, seg_bbox, outdata_path_dict, out_channels, outdata_suffix, model_version='v1.x', file_time_segmentation='archseg'):

    outdata_decsr_dict = {"LB": ["lower bound value for all conditions", "NORPIX"], "LBclass": ["classification result", "UNKNOWN, LAND, SNOW, LANDSNOW = 0, 1, 2, 3"], \
                    "LBland": ["lower band value for snow.cloud free conditions", "NORPIX"], "CI": ["cloud index", ""], "CLI": ["spectral cloud index - auxiliary", ""], "KTM": ["clearsky index", ""], \
                    "GHIc": ["Global horizontal clearsky irradiance", "W/m2"], "GHI": ["Global horizontal realsky irradiance", "W/m2"], "DNIc": ["Direct normal clearsky irradiance", "W/m2"], \
                    "DNI": ["Direct normal realsky irradiance", "W/m2"], "GHIcor": ["Global horizontal realsky irradiance with correction", "W/m2"], "DNIcor": ["Direct normal realsky irradiance with correction", "W/m2"], \
                    "CI_flag": ["cloud index flag: 0-no (h0 <0), 1 - from data, 2 - interpolated, 3 - extrapolated, 4 - inter/extra-polated more than one hour", ""]}

    #for each output variable define no_value, least_significant_digit, NC_data_type 
    #"NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"
    outdata_nc_param_dict = { \
                             "CI":{ 'noval':-9., 'leastSignificantDigit':4, 'nctype':"NC_FLOAT"}, \
                             "CI_flag":{'noval':-9, 'nctype':"NC_BYTE"}, \
                             "CLI":{'noval':-99., 'leastSignificantDigit':4, 'nctype':"NC_FLOAT"}, \
                             "DNIc":{'noval':-99, 'leastSignificantDigit':1, 'nctype':"NC_FLOAT"}, \
                             "DNI":{'noval':-99, 'leastSignificantDigit':1, 'nctype':"NC_FLOAT"}, \
                             "GHIc":{'noval':-99, 'leastSignificantDigit':1, 'nctype':"NC_FLOAT"}, \
                             "GHI":{'noval':-99, 'leastSignificantDigit':1, 'nctype':"NC_FLOAT"}, \
                             "KTM":{'noval':-9., 'leastSignificantDigit':4, 'nctype':"NC_FLOAT"}, \
                             "LB":{'noval':-9., 'leastSignificantDigit':4, 'nctype':"NC_FLOAT"}, \
                             "LBclass":{'noval':-9, 'nctype':"NC_BYTE"}, \
                             "LBland":{'noval':-9., 'leastSignificantDigit':4, 'nctype':"NC_FLOAT"}, \
                            }
    
    out_files_dict = outdata_ncfile_dict(dfb_begin, dfb_end, out_channels, outdata_path_dict, outdata_suffix, outdata_decsr_dict, outdata_nc_param_dict, seg_bbox=seg_bbox, create_missing=True, slot_min=aslot_begin, slot_max=aslot_end, model_version=model_version, file_time_segmentation=file_time_segmentation)


    return out_files_dict



#check existence of the output data files
#optionaly create one if missing
def outdata_ncfile_dict(dfb_start, dfb_end, outchannels, nc_paths_preffix_dict, ncfile_suffix, outdata_decsr_dict, outdata_nc_param_dict=None , seg_bbox=None, create_missing=True, slot_min=1, slot_max=144, model_version='v1.x', file_time_segmentation='archseg'):

    ncfileparamdict={}

    timestring_dict = filename_timestrings_create(dfb_start,dfb_end,file_time_segmentation=file_time_segmentation)
    
    for timestring, [nc_dfb_min, nc_dfb_max] in  timestring_dict.iteritems():
        for channel in outchannels:
            ncfile, nc_basename = create_outnc_filename(timestring, channel, nc_paths_preffix_dict, ncfile_suffix)
            if not (check_file_existence(ncfile)):
                if not create_missing:
                    continue #skip file
                #create missing file
                chan_param_dict = outdata_nc_param_dict[channel]
                nc_type = chan_param_dict['nctype']
                noval = chan_param_dict['noval']
                if chan_param_dict.has_key('leastSignificantDigit'):
                    leastSignificantDigit = chan_param_dict['leastSignificantDigit']
                img_units, img_long_name = outdata_decsr_dict[channel]
                chunksiz=[4,48,16,16]
                compression=True
                
                year = daytimeconv.dfb2ymd(nc_dfb_min)[0]
                metadata = [['title',"HIMAWARI model output"], ['year',year], ['version',model_version], ['channel',channel], ['projection',"geographic coordinates"]]
                SlotDescription = "scan of slot 1 starts at 00:00, slot 2 at 00:10, ..."
                result = latlon_nctools.latlon_make_params_dfb_slot_lat_lon_nc(nc_file_name=ncfile, metadata=metadata, force_overwrite=True, img_channels=[channel], img_types=[nc_type], img_units=[img_units],img_long_names=[img_long_name], novals=[noval], chunksizes=[chunksiz], least_significant_digits=[leastSignificantDigit], tslot_min=slot_min, tslot_max=slot_max, dfb_begin=nc_dfb_min, dfb_end=nc_dfb_max, nc_extent=seg_bbox ,compression=compression, dims_name_colrow=False, SlotDescription=SlotDescription)
                print ('output NC file creation %s: %s'% (ncfile, result))


            #get parameters of the nc file
            if ncfileparamdict.has_key(nc_basename): continue
            
            params = get_outdata_ncfile_parameters(ncfile, channel)

            #check bbox if requested
            nc_bbox = params[4]
            if seg_bbox:
                if not nc_bbox.equals(seg_bbox):
                    print 'WARNING: processing segment bbox %s is not equal with nc file bbox  %s ' % (str(nc_bbox), str(seg_bbox) )
                    continue

            ncfileparamdict[nc_basename]=params

    return (ncfileparamdict)


#check existence of the output data files
def outdata_existingncfile_dict(dfb_start, dfb_end, channels, nc_paths_preffix_dict, ncfile_suffix, file_time_segmentation='month'):
    ncfileparamdict={}
    timestring_dict = filename_timestrings_create(dfb_start,dfb_end,file_time_segmentation=file_time_segmentation)
    for timestring, [nc_dfb_min, nc_dfb_max] in  timestring_dict.iteritems():
        for channel in channels:
            file_name_list = create_outnc_filename_multi(timestring, channel, nc_paths_preffix_dict, ncfile_suffix)
            for ncfile, nc_basename in file_name_list:
                #check file existence            
                if not (check_file_existence(ncfile)):continue
                #get parameters of the nc file
                if ncfileparamdict.has_key(nc_basename): continue

                ncfileparamdict[nc_basename]=get_outdata_ncfile_parameters(ncfile, channel)    
 
    return (ncfileparamdict)



#check existence of the output data files
def outdata_existingncfile_dict_pathpool(dfb_start, dfb_end, channels, data_path_pool, ncfile_suffix, file_time_segmentation='month'):
    ncfileparamdict={}
    timestring_dict = filename_timestrings_create(dfb_start,dfb_end,file_time_segmentation=file_time_segmentation)
    for timestring, [nc_dfb_min, nc_dfb_max] in  timestring_dict.iteritems():
        for channel in channels:
            for out_data_path in data_path_pool:
                outdata_path_dict={"default": out_data_path}
                file_name_list = create_outnc_filename_multi(timestring, channel, outdata_path_dict, ncfile_suffix)
                for ncfile, nc_basename in file_name_list:
                    if ncfileparamdict.has_key(nc_basename): continue
                    #check file existence            
                    if not (check_file_existence(ncfile)):continue
                    #get parameters of the nc file
    
                    ncfileparamdict[nc_basename]=get_outdata_ncfile_parameters(ncfile, channel)    
 
    return (ncfileparamdict)



#write model output to NetCDF 
def write_output_nc(channel, data_to_write, out_files_dict, outdata_suffix, dfb_begin, dfb_end, slot_begin, slot_end, bbox, file_time_segmentation='month', verbose=False):
    timestring_dict = filename_timestrings_create(dfb_begin,dfb_end,file_time_segmentation=file_time_segmentation)
    for timestring, [nc_dfb_min, nc_dfb_max] in  timestring_dict.iteritems():

        dummy, nc_basename = create_outnc_filename(timestring, channel, None, outdata_suffix)
        if not(nc_basename in out_files_dict.keys()):
            print "output NC file %s not found. Skipping" % (nc_basename)
            continue
            
        ncfile, [ncdfb_min, ncdfb_max ],dummy, [dummy,dummy],dummy , dummy, dummy = out_files_dict[nc_basename]

        read_dfb_min=max(ncdfb_min,dfb_begin)
        read_dfb_max=min(ncdfb_max,dfb_end)
        
        if read_dfb_max<read_dfb_min:
            print 'no dfbs to write to %s' %(nc_basename)
            continue
        
        data_dfb_min_idx=read_dfb_min-dfb_begin
        data_dfb_max_idx=data_dfb_min_idx+read_dfb_max-read_dfb_min

        data=data_to_write[data_dfb_min_idx:data_dfb_max_idx+1,:,:,:]
        res=latlon_nctools.latlon_write_dfb_slot_lat_lon_nc(ncfile, channel, (read_dfb_min, read_dfb_max), (slot_begin,slot_end), bbox, data)
        if res is False:
            print 'problem writing %s' %(ncfile)
            continue

    return True



def outdata_nc_read(channel, out_files_dict, outdata_suffix, dfb_begin, dfb_end, slot_min, slot_max, bbox, file_time_segmentation='month'):

    dfbs=dfb_end-dfb_begin+1
    slots=slot_max-slot_min+1
    rows=bbox.height
    cols=bbox.width
    all_data=numpy.empty((dfbs,slots, rows, cols), dtype='float32')
    all_data[:,:,:,:]=-999.

    timestring_dict = filename_timestrings_create(dfb_begin,dfb_end,file_time_segmentation=file_time_segmentation)
    
    for timestring, [nc_dfb_min, nc_dfb_max] in  timestring_dict.iteritems():
        dummy, nc_basename = create_outnc_filename(timestring, channel, None, outdata_suffix)
        if not out_files_dict.has_key(nc_basename):
            continue

        ncfile, [ncdfb_min, ncdfb_max ],dummy, [dummy,dummy],dummy, dummy, dummy = out_files_dict[nc_basename]
        read_dfb_min=max(nc_dfb_min,ncdfb_min,dfb_begin)
        read_dfb_max=min(nc_dfb_max,ncdfb_max,dfb_end)
        
        if read_dfb_max<read_dfb_min:
            print 'no dfbs to read from %s' %(nc_basename)
            continue
      
        res=latlon_nctools.latlon_read_dfb_slot_lat_lon_nc(ncfile, channel, (read_dfb_min, read_dfb_max), (slot_min,slot_max), bbox, interpolate='nearest')
        if res is None:
            print 'problem reading %s' %(ncfile)
            continue

        read_dfbs=read_dfb_max-read_dfb_min+1
        read_dfb_min_idx = read_dfb_min-dfb_begin
        all_data[read_dfb_min_idx:read_dfb_min_idx+read_dfbs,:,:,:] =  res  

    return all_data


def outdata_nc_read_point(channel, out_files_dict, outdata_suffix, dfb_begin, dfb_end, slot_min, slot_max, lon, lat, interpolation='nearest', file_time_segmentation='month'):

    dfbs=dfb_end-dfb_begin+1
    slots=slot_max-slot_min+1
    all_data=numpy.empty((dfbs,slots), dtype='float32')
    all_data[:,:]=-9

    timestring_dict = filename_timestrings_create(dfb_begin,dfb_end,file_time_segmentation=file_time_segmentation)
    
    for timestring, [nc_dfb_min, nc_dfb_max] in  timestring_dict.iteritems():
        dummy, nc_basename = create_outnc_filename(timestring, channel, None, outdata_suffix)
        if not out_files_dict.has_key(nc_basename):
            continue

        ncfile, [ncdfb_min, ncdfb_max ],dummy, [dummy,dummy],dummy, dummy, dummy = out_files_dict[nc_basename]
        read_dfb_min=max(nc_dfb_min,ncdfb_min,dfb_begin)
        read_dfb_max=min(nc_dfb_max,ncdfb_max,dfb_end)
        
        if read_dfb_max<read_dfb_min:
            print 'no dfbs to read from %s' %(nc_basename)
            continue
        res=latlon_nctools.latlon_read_dfb_slot_lat_lon_nc_point(ncfile, channel, (read_dfb_min, read_dfb_max), (slot_min,slot_max), lon, lat, interpolate=interpolation)
        if res is None:
            print 'problem reading %s' %(ncfile)
            continue

        read_dfbs=read_dfb_max-read_dfb_min+1
        read_dfb_min_idx = read_dfb_min-dfb_begin
        all_data[read_dfb_min_idx:read_dfb_min_idx+read_dfbs,:] =  res  

    return all_data

# wrapper putting together path creation and dataread
def outdata_nc_read_point_latlon(nc_var_name, outdata_path_dict, outdata_suffix, lon, lat, dfb_begin, dfb_end, slot_min, slot_max, interpolation='nearest', file_time_segmentation='month'):
            
        seg_col, seg_row = latlon.get_5x5_seg(lon, lat)
        seg_suffix="_c%d_r%d" % (seg_col, seg_row)
        
        if outdata_suffix=='':
            outdata_suffix=seg_suffix
        else:
            outdata_suffix="_%s%s" % (outdata_suffix, seg_suffix)
        out_files_dict = outdata_existingncfile_dict(dfb_begin, dfb_end, [nc_var_name], outdata_path_dict, outdata_suffix, file_time_segmentation=file_time_segmentation)    
#        out_files_dict = outdata_ncfile_dict(dfb_begin, dfb_end, [nc_var_name], outdata_path_dict, outdata_suffix, None, None, None, create_missing=False, slot_min=slot_min, slot_max=slot_max, model_version=None)
        if len(out_files_dict)<1:
            print >> sys.stderr, 'No NC files for %s found' % (nc_var_name) 
            return None
        rast_data=outdata_nc_read_point(nc_var_name, out_files_dict, outdata_suffix, dfb_begin, dfb_end, slot_min, slot_max, lon=lon, lat=lat, interpolation=interpolation, file_time_segmentation=file_time_segmentation)

        return rast_data








def read_multisegment_data(dfb_begin,dfb_end,aslot_begin,aslot_end,roll_slots,himawari_slot_min,himawari_slot_max,segments_to_calculate,nc_var_name,data_path_pool,res, file_time_segmentation='month'):

    #extend data if roll is needed
    if roll_slots==0:
        read_slot_begin = aslot_begin
        read_slot_end = aslot_end
        read_dfb_begin = dfb_begin 
        read_dfb_end = dfb_end 
    elif roll_slots>0:
        read_slot_begin = himawari_slot_min
        read_slot_end = himawari_slot_max
        read_dfb_begin = dfb_begin - 1
        read_dfb_end = dfb_end 
    elif roll_slots<0:
        read_slot_begin = himawari_slot_min
        read_slot_end = himawari_slot_max
        read_dfb_begin = dfb_begin 
        read_dfb_end = dfb_end + 1
    
    
    #calculate subsegments parameters
    xmin_total, xmax_total, ymin_total, ymax_total = 99999, -99999, 99999, -99999
    subsegs={}
#    print 'segments:'
    for seg_c, seg_r in segments_to_calculate:
        seg_rc = seg_r*100+seg_c
        seg_bbox = latlon.get_5x5_seg_bbox(seg_r, seg_c, res)
        xmin_total=min(xmin_total,seg_bbox.xmin)
        xmax_total=max(xmax_total,seg_bbox.xmax)
        ymin_total=min(ymin_total,seg_bbox.ymin)
        ymax_total=max(ymax_total,seg_bbox.ymax)
#            print seg_c, seg_r
        outdata_seg_suffix="_c%d_r%d" % (seg_c, seg_r)
        subsegs[seg_rc]={'seg_c':seg_c, 'seg_r':seg_r, 'suffix':outdata_seg_suffix, 'bbox':seg_bbox}

    #create empty array for all data
    width=int(math.floor(((xmax_total-xmin_total)/res)+0.49))
    height=int(math.floor(((ymax_total-ymin_total)/res)+0.49))
    bbox = latlon.bounding_box(xmin=xmin_total, xmax=xmax_total, ymin=ymin_total, ymax=ymax_total, width=width, height=height, resolution=res)
    
    data_total = numpy.empty((read_dfb_end-read_dfb_begin+1,read_slot_end-read_slot_begin+1, height, width))
    data_total[:,:,:,:] = numpy.nan
    print 'total shape:', data_total.shape


    #read data
    counter=0
    for seg_rc in subsegs.keys():
        counter+=1
        seg_c = subsegs[seg_rc]['seg_c']
        seg_r = subsegs[seg_rc]['seg_r']
        outdata_suffix = subsegs[seg_rc]['suffix']
        seg_bbox = subsegs[seg_rc]['bbox']
        print 'processing %d/%d:' %(counter,len(subsegs.keys())), seg_c, seg_r, seg_bbox
    

        #search output files
        for out_data_path in data_path_pool:
            outdata_path_dict={"default": out_data_path}

            out_files_dict = outdata_existingncfile_dict(read_dfb_begin, read_dfb_end, [nc_var_name], outdata_path_dict, outdata_suffix, file_time_segmentation=file_time_segmentation)
            if len(out_files_dict)>0:
                break
        if len(out_files_dict)==0:
            print 'no files skipping segment'
            continue

        #read             
        data=outdata_nc_read(nc_var_name, out_files_dict, outdata_suffix, read_dfb_begin, read_dfb_end, read_slot_begin, read_slot_end, seg_bbox, file_time_segmentation=file_time_segmentation)
        
        if data is not None:
            px_xmin, px_xmax, px_ymin, px_ymax = bbox.pixel_coords_of_bbox(seg_bbox) 
            try:
                data_total[:,:,px_ymin:px_ymax+1, px_xmin:px_xmax+1] = data
            except:
                print 'shape mismatch'
                print 'read data shape',data.shape
                print 'out data shape',data_total[:,:,px_ymin:px_ymax+1,px_xmin:px_xmax+1].shape

    #roll and trim
    if roll_slots>0:
        aux=data_total.reshape((data_total.shape[0]*data_total.shape[1],data_total.shape[2],data_total.shape[3]))
        aux = numpy.roll(aux,shift=roll_slots, axis=0)
        aux = aux.reshape(data_total.shape)
        data_total = aux[1:,aslot_begin-himawari_slot_min:aslot_end-himawari_slot_min+1,:,:]
    if roll_slots<0:
        aux=data_total.reshape((data_total.shape[0]*data_total.shape[1],data_total.shape[2],data_total.shape[3]))
        aux = numpy.roll(aux,shift=roll_slots, axis=0)
        aux = aux.reshape(data_total.shape)
        data_total = aux[:data_total.shape[0]-1,aslot_begin-himawari_slot_min:aslot_end-himawari_slot_min+1,:,:]

    return data_total, bbox

#based on the x_seg, y_seg indexes, returns path where data reside
def identify_model_output_segment_path(y_seg, x_seg, model_out_dir_pool, file_pattern='GHI_????_*_c%d_r%d.nc'):
    
    model_data_path=None
    for apath in model_out_dir_pool:
        ghi_files = glob.glob(os.path.join(apath, file_pattern%(x_seg, y_seg)))
        if len(ghi_files) > 0:
            model_data_path=apath
            break
    return model_data_path