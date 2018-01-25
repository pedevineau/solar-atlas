import numpy
from general_utils import daytimeconv
from general_utils import solar_geom_v5
from himawari8 import slotmapping

from himawari8.utils import compute_scan_time_offset2



def calculate_realscan_UTCtimes(DfbStart,DfbEnd, SlotStart,SlotEnd, bbox):
    # calculate real scan time given by nominal scan time and scan time offset for each pixel - for each satellite image is specific
 
    dfbs = DfbEnd-DfbStart+1
    slots = SlotEnd-SlotStart+1
    lats = bbox.height
    lons = bbox.width
    
    slot_dh = slotmapping.slot2dh(numpy.arange(SlotStart, SlotEnd+1))
    slot_dh_3D = numpy.repeat(slot_dh, lats*lons).reshape( slots, lats, lons)

    
    scantime_offset_dh = compute_scan_time_offset2(bbox,  channel_name='VIS064_2000')/3600.
    scantime_offset_dh_3D  = numpy.tile(scantime_offset_dh, (slots,1,1)).reshape(slots, lats, lons)
    
    UTC_dh_3D = slot_dh_3D + scantime_offset_dh_3D
    UTC_dh_4D = numpy.tile(UTC_dh_3D, (dfbs,1,1,1)).reshape(dfbs,slots, lats, lons)

    return UTC_dh_4D

    
def calculate_realscan_UTCtimes_slotcenter(DfbStart,DfbEnd, SlotStart,SlotEnd, bbox):
    # calculate real scan time given by nominal scan time and scan time offset for each pixel - for each satellite image is specific
 
    dfbs = DfbEnd-DfbStart+1
    slots = SlotEnd-SlotStart+1
    lats = bbox.height
    lons = bbox.width
    
    slot_dh = slotmapping.slot2dh(numpy.arange(SlotStart, SlotEnd+1))
    slot_dh_3D = numpy.repeat(slot_dh, lats*lons).reshape( slots, lats, lons)

    
    slotcenter_offset_dh = 5./60.  # five minutes
    
    UTC_dh_3D = slot_dh_3D + slotcenter_offset_dh
    UTC_dh_4D = numpy.tile(UTC_dh_3D, (dfbs,1,1,1)).reshape(dfbs,slots, lats, lons)

    
    return UTC_dh_4D
    


def prepare_doy_year_arrays(subseg_bbox,DfbStart,DfbEnd,SlotStart,SlotEnd,dfbs,slots):
    
    #calculate DOY[dfb,slot] and YEAR[dfb,slot]
    #this is tricky - but we need to get DOY taking into consideration of longitudinal time shift - just to avoid change of DOY during the local day on the given location
    #therefore DOY and YEAR derived from DFB are "adapted" in the slot dimension to fit with local day   
    #find which time slot represents local time offset - it is used to shift the DOY calculated in UTC to local time zone DOY

#    dfbs = DfbEnd - DfbStart +1
#    slots = SlotEnd - SlotStart +1
    
    
    c_lon, c_lat = subseg_bbox.center()
    if (c_lon > 0) :
        c_lon = c_lon - 360
    time_off = -c_lon/15.
    slot_shift = 1
    for s in range(1,144+1):
        slot_shift = s
#        print s, c_lon, daytimeconv.time2dh(slotmapping.slot2time(s)),time_off 
        if daytimeconv.time2dh(slotmapping.slot2time(s)) > time_off:
            break
    doy_2d_arr = numpy.empty((dfbs,slots),dtype=numpy.float32)
    year_2d_arr = numpy.empty((dfbs,slots),dtype=numpy.float32)
    doy_2d_arr[:,:] = numpy.nan
    year_2d_arr[:,:] = numpy.nan
    
    s_idx_to_shift = slot_shift - SlotStart
    for dfb in range(DfbStart,DfbEnd+1):
        dfb_idx=dfb-DfbStart
        doy_2d_arr[dfb_idx,:], year_2d_arr[dfb_idx,:] = daytimeconv.dfb2doyy(dfb) 
        if s_idx_to_shift >= 0: 
            doy_2d_arr[dfb_idx,0:s_idx_to_shift], year_2d_arr[dfb_idx,0:s_idx_to_shift] =   daytimeconv.dfb2doyy(dfb-1)

    return year_2d_arr,doy_2d_arr


def declination(subseg_bbox,dfbs,slots,doy_2d_arr,year_2d_arr):
    #calculate declination
    #returns sin and cos of declination, as the declination itself is not needed
    Longitues_d_2D = subseg_bbox.longitudes(array2d=True,degrees=True)
    Longitues_d_2D[Longitues_d_2D<-180] += 360
    Longitues_d_2D[Longitues_d_2D<+180] += -360
    
    Longitues_r_4D = numpy.tile(numpy.radians(Longitues_d_2D),(dfbs*slots,1,1)).reshape(dfbs,slots,subseg_bbox.height,subseg_bbox.width)

    doy_4D = numpy.repeat(doy_2d_arr, subseg_bbox.height*subseg_bbox.width).reshape((dfbs,slots,subseg_bbox.height,subseg_bbox.width))
    year_4D = numpy.repeat(year_2d_arr, subseg_bbox.height*subseg_bbox.width).reshape((dfbs,slots,subseg_bbox.height,subseg_bbox.width))
    
    aux = solar_geom_v5.declin_r_arr(year_4D, doy_4D, Longitues_r_4D)
    cosde = numpy.cos(aux)
    sinde = numpy.sin(aux)
    
    return sinde,cosde
     
def LAT(subseg_bbox,dfbs,slots,doy_2d_arr,UTC_dh_4D):
    Longitues_d_2D = subseg_bbox.longitudes(array2d=True,degrees=True)
    Longitues_d_2D[Longitues_d_2D<-180] += 360
    Longitues_d_2D[Longitues_d_2D<+180] += -360
#    Longitues_r_4D = numpy.tile(numpy.radians(Longitues_d_2D),(dfbs*slots,1)).reshape(dfbs,slots,subseg_bbox.height,subseg_bbox.width)
    Longitues_r_4D = numpy.tile(Longitues_d_2D,(dfbs*slots,1)).reshape(dfbs,slots,subseg_bbox.height,subseg_bbox.width)
    
    #LAT - local apparent time = UTC + perturbation + longitudianl_offset
    #LAT is in decimal hours 0.-24.
    #perturbation ET
    ET_2D = solar_geom_v5.perturbation_vect(doy_2d_arr)
    ET_4D = numpy.repeat(ET_2D, subseg_bbox.height*subseg_bbox.width).reshape((dfbs,slots,subseg_bbox.height,subseg_bbox.width))
    LAT_dh_4D = ET_4D + (Longitues_r_4D / 15.) + UTC_dh_4D

    return numpy.radians((numpy.mod(LAT_dh_4D,24) - 12.)*15.)

def solargeom(subseg_bbox,DfbStart,DfbEnd,SlotStart,SlotEnd, UTC_dh_4D):
    dfbs = DfbEnd - DfbStart +1
    if dfbs<=366:
        h0_r, h0_r_ref, a0_r = solargeom_core(subseg_bbox,DfbStart,DfbEnd,SlotStart,SlotEnd, UTC_dh_4D)
    else:
        #to avoid low memory problem we use segmented processing (in dfb dimension)      
        h0_r = numpy.empty_like(UTC_dh_4D)
        h0_r_ref = numpy.empty_like(UTC_dh_4D)
        a0_r = numpy.empty_like(UTC_dh_4D)
        h0_r[:,:,:,:] = numpy.nan
        h0_r_ref[:,:,:,:] = numpy.nan
        a0_r[:,:,:,:] = numpy.nan
        
        dfb_step=400
        dfb_range_min = numpy.arange(DfbStart, DfbEnd+1, dfb_step, dtype=numpy.int32)
        for DfbStart_seg in dfb_range_min:
            DfbEnd_seg = min(DfbEnd,DfbStart_seg+dfb_step-1)
            DfbStart_seg_idx = DfbStart_seg-DfbStart
            DfbEnd_seg_idx = DfbEnd_seg-DfbStart
            h0_r[DfbStart_seg_idx:DfbEnd_seg_idx+1,:,:,:],h0_r_ref[DfbStart_seg_idx:DfbEnd_seg_idx+1,:,:,:], a0_r[DfbStart_seg_idx:DfbEnd_seg_idx+1,:,:,:] = solargeom_core(subseg_bbox,DfbStart_seg,DfbEnd_seg,SlotStart,SlotEnd, UTC_dh_4D[DfbStart_seg_idx:DfbEnd_seg_idx+1,:,:,:])

    return h0_r, h0_r_ref, a0_r


def solargeom_core(subseg_bbox,DfbStart,DfbEnd,SlotStart,SlotEnd, UTC_dh_4D):
    dfbs = DfbEnd - DfbStart +1
    slots = SlotEnd - SlotStart +1

    #calculate DOY[dfb,slot] and YEAR[dfb,slot]
    year_2d_arr,doy_2d_arr = prepare_doy_year_arrays(subseg_bbox,DfbStart,DfbEnd,SlotStart,SlotEnd,dfbs,slots)

    #LAT - local apparent time = UTC + perturbation + longitudianl_offset
    # converted to radians
    LAT_r_4D = LAT(subseg_bbox,dfbs,slots,doy_2d_arr,UTC_dh_4D)
    cost = numpy.cos(LAT_r_4D)
    
    # declination
    sinde,cosde = declination(subseg_bbox,dfbs,slots,doy_2d_arr,year_2d_arr)


    #latitude sin and cos 
    Latitues_r_2D  = subseg_bbox.latitudes(array2d=True,degrees=False)
    sinfi = numpy.tile(numpy.sin(Latitues_r_2D),(dfbs*slots,1,1)).reshape(dfbs,slots,subseg_bbox.height,subseg_bbox.width)
    cosfi = numpy.tile(numpy.cos(Latitues_r_2D),(dfbs*slots,1,1)).reshape(dfbs,slots,subseg_bbox.height,subseg_bbox.width)

    #finally calculate solar geometry
    sinh0 = (sinfi*sinde) + (cosfi*cosde*cost)
    h0_r = numpy.arcsin(sinh0)
    cosh0 = numpy.cos(h0_r)
    cecl=cosfi*cosh0
    
    a0_r = numpy.empty_like(h0_r) 
    a0_r[:,:,:,:] = numpy.pi   #toto je pre J a S pol - nie je definovana - da sa nahradit "time in radians" ???
    wh1 = numpy.abs(cecl) >= 0.001 #if abs(cecl) >= 0.001:
    cosas = ((sinfi[wh1]*sinh0[wh1]) - sinde[wh1])/cecl[wh1]
    cosas[cosas>1.] = 1.
    cosas[cosas<-1.] = -1.
    a0_r[wh1] = numpy.pi - numpy.arccos(cosas)
    wh1 = LAT_r_4D>0.
    a0_r[wh1] = -a0_r[wh1]
    a0_r = a0_r+numpy.pi
    wh1 = a0_r>numpy.pi
    a0_r[wh1] = a0_r[wh1] - (2.*numpy.pi)

    wh = h0_r != h0_r
    a0_r[wh] = numpy.nan
    #h0_r_ref - h0 (sun elevation) with refraction (in radians)
    h0_r_ref=h0_r+solar_geom_v5.delta_h0refr(h0_r)
    #solar geometry done - results are in a0_r (sun aspect), h0_r, h0_r_ref 


    return h0_r, h0_r_ref, a0_r
