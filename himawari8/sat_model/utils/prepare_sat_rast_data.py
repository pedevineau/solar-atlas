'''
Created on Feb 14, 2016

@author: tomas
'''
import glob
import os
import numpy
from general_utils import daytimeconv
from general_utils import latlon_nctools

from general_utils.basic_logger import make_logger
logger = make_logger(__name__)
logger.setLevel(20)




def search_for_NC_files(year, month, ncReadDirsList, ncReadFileNamePattern, suffix, satList, chanNameList):
    #get dict of files for given year, month in paths listed in ncReadDirsList
    sat_chan_NCFileDict = {}
    for sat in satList:
        for chan in chanNameList:
            NCFileDict = {}
            ncNamePattern_aux = ncReadFileNamePattern % (sat,chan,year,month,suffix)
            for aPath in ncReadDirsList:
#                print aPath,ncNamePattern_aux
                for fullName in glob.glob(os.path.join(aPath,ncNamePattern_aux)):
                    baseName = os.path.basename(fullName)
                    if baseName not in NCFileDict.keys():
                        NCFileDict[baseName] = fullName
            
            if len(NCFileDict) > 0:
                if not sat_chan_NCFileDict.has_key(sat):
                    sat_chan_NCFileDict[sat] = {}
                sat_chan_NCFileDict[sat][chan] = {'fullName':fullName}
    return sat_chan_NCFileDict



    
def readLonLatSatelliteData(DfbStart, DfbEnd, SlotStart, SlotEnd, SatellitesList, channelNameList, SatelliteRawDataLocations, SatSuffix, BoundingBox):
    '''
    Function to read and merge together data from different satellites from more paths in LonLat nc files of 5x5 deg boxes.
    Returns dictionary with channel number and float32 array of dfbs,slots,lon,lat dimensions. Errors or missing data are 0.
    For calibration purpose returns dict: {"sat":wh_array, ... according SatellitesList and vis channel (number 1)}
    '''
    
#    channelLookupNumName = {1: 'VIS', 2: 'IR2', 4: 'IR4'}
#    channelLookupNameNum = {'VIS':1, 'IR2':2, 'IR4':4}
#
#    #only those requested 
#    channelNameList = []
#    for ch_n in ChannelsNumbersList:
#        channelNameList.append(channelLookupNumName[ch_n])

    
    dfb_array = numpy.arange(DfbStart,DfbEnd+1)

    ncReadFileNamePattern =  "%sLATLON_%s__TMON_%d_%02d__SDEG05%s.nc"    #     H08LATLON_IR390_2000__TMON_2015_10__SDEG05_r09_c60.nc

    
    
    
    #init output dictionaries 
    ChannelRawDataDictionary = {}
    SatDataAvailabilityDict = {}
#    DfbsBordersDict = {}
    
    dfbs=DfbEnd-DfbStart+1
    slots=SlotEnd-SlotStart+1
    latlon_rows=BoundingBox.height
    latlon_cols=BoundingBox.width
    empty_data_mtx =  numpy.empty((dfbs,slots,latlon_rows,latlon_cols), dtype=numpy.float64)
    empty_data_mtx[:,:,:,:] = numpy.nan
    empty_availability_mtx =  numpy.empty((dfbs,slots,latlon_rows,latlon_cols), dtype=numpy.bool_)
    empty_availability_mtx[:,:,:,:] = False
    
    # data dict with data read individually for satellites
    SatChannelRawDataDictionary = {}
    for sat in SatellitesList:
        SatChannelRawDataDictionary[sat]={}
        for chanName in channelNameList:
            SatChannelRawDataDictionary[sat][chanName] = empty_data_mtx.copy()
    

    # final output data dict with sat data availability
    for sat in SatellitesList:
        SatDataAvailabilityDict[sat] = empty_availability_mtx.copy()
            
    # final output data dict with data merged from all satellites
    for chanName in channelNameList:
        ChannelRawDataDictionary[chanName] = empty_data_mtx.copy()


    #loop years, months and process data
    year_month_list = daytimeconv.dfb_minmax2year_month_list(DfbStart, DfbEnd)
    
    for year, month in year_month_list:

        sat_chan_NCFileDict = search_for_NC_files(year, month, SatelliteRawDataLocations, ncReadFileNamePattern, SatSuffix, SatellitesList, channelNameList)
        if len(sat_chan_NCFileDict.keys()) ==0:
            logger.warning('Processing year %d, month %d. No input NC files found', year, month)
#            logger.warning('No input NC files found')
            continue
        else:
#            logger.info('Found NC files for: %s'," ".join(sat_chan_NCFileDict.keys()))
            logger.debug('Processing year %d, month %d. Found data: %s', year, month," ".join(sat_chan_NCFileDict.keys()))

        
        #calculate dfbs to read            
        dfb_archseg_start, dfb_archseg_end = daytimeconv.monthyear2dfbminmax(month, year)
        read_dfb_min = max(DfbStart,dfb_archseg_start)
        read_dfb_max = min(DfbEnd,dfb_archseg_end)
        #and dfb indexis of arch segment in whole dfb range
        dfb_idx_min = read_dfb_min - DfbStart
        dfb_idx_max = read_dfb_max - DfbStart+1
        
        #read data from NC files
        for sat in sat_chan_NCFileDict.keys():
            #multiple channel data for given satellite
            chan_NCFileDict = sat_chan_NCFileDict[sat]
            for chan in chan_NCFileDict.keys():
                ncfile = chan_NCFileDict[chan]['fullName']
                res=latlon_nctools.latlon_read_dfb_slot_lat_lon_nc(ncfile, chan, (read_dfb_min, read_dfb_max), (SlotStart,SlotEnd), BoundingBox, interpolate='nearest')
#                chanNum = channelLookupNameNum[chan]
                if (res==res).sum() > 0:
                    res[res>5000] = numpy.nan
                res[res==0] = numpy.nan
                SatChannelRawDataDictionary[sat][chan][dfb_idx_min:dfb_idx_max,:,:,:] = res 


    #merge data from various satellites to one dataset - make final ChannelRawDataDictionary
    for chan in channelNameList:
        for sat in SatellitesList:
            data_sat_chan = SatChannelRawDataDictionary[sat][chan]
            wh = data_sat_chan==data_sat_chan
            if wh.sum()> 1:
                ChannelRawDataDictionary[chan][wh] = SatChannelRawDataDictionary[sat][chan][wh]
            if chan == 'VIS064_2000':
                SatDataAvailabilityDict[sat][wh] = True

    #clean up -  delete if no data from given satellite - make final SatDataAvailabilityDict and DfbsBordersDict
    total_availability_arr = numpy.empty((dfbs), dtype=numpy.bool_)
    total_availability_arr[:] = False
    for sat in SatellitesList:
        if (SatDataAvailabilityDict[sat] == True).sum() < 1:
            del SatDataAvailabilityDict[sat]
        else:
            sat_dfb_wh = SatDataAvailabilityDict[sat].any(axis=3).any(axis=2).any(axis=1)
#            sat_dfb_list = list(dfb_array[sat_dfb_wh])
#            DfbsBordersDict[sat] = sat_dfb_list
            total_availability_arr |= sat_dfb_wh 
            
    no_data_dfbs = dfb_array[numpy.logical_not(total_availability_arr)]
    if len(no_data_dfbs) > 35:
        logger.warning('No raw satellite data for %d days. ', len(no_data_dfbs))
    elif len(no_data_dfbs) > 0:
        no_data_dates = []
        for dfb in list(no_data_dfbs):
            no_data_dates.append(daytimeconv.dfb2yyyymmdd(dfb))
        logger.warning('No raw satellite data for %d days: %s ', len(no_data_dfbs), " ".join(no_data_dates))
            


#    import pylab
#    pylab.plot(SatChannelRawDataDictionary['H08']['IR390_2000'][:,:,0,0].flatten())
#    pylab.plot(SatChannelRawDataDictionary['H08']['IR124_2000'][:,:,0,0].flatten())
#    pylab.show()
#  
#   
#        
#    for chan in channelNameList:
#        ChannelRawDataDictionary[chan][numpy.isnan(ChannelRawDataDictionary[chan])] = 0.
#    logger.info("Raw satellite data read.")
#    
    return ChannelRawDataDictionary, SatDataAvailabilityDict

