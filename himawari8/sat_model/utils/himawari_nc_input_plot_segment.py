'''
Created on Sep 11, 2011

@author: tomas
'''

import os

import numpy
from general_utils import daytimeconv
from general_utils import latlon
from general_utils import latlon_nctools
from himawari8.sat_model.utils import prepare_sat_rast_data


if __name__ == '__main__':
    segment_list = latlon.expand_segments([[56,57,12,13]])

    dfb_begin = daytimeconv.yyyymmdd2dfb('20170902')
    dfb_end = daytimeconv.yyyymmdd2dfb('20170902')

    slot_begin = 1 # 1 
    slot_end = 144  # 144
    
    sat_data_path = '/data/model_data_himawari/sat_data_procseg/'

    chan = 'VIS064_2000' #["VIS064_2000", "VIS160_2000", "IR390_2000", "IR124_2000"]
    Satellite = "H08"   # in order of preference


    # H08LATLON_VIS064_2000__TMON_2017_09__SDEG05_r12_c58.nc
    filename_pattern = '%sLATLON_%s__TMON_%d_%02d__SDEG05_r%d_c%d.nc'


    vmin=0
    vmax=0.9
    
    skip_empty = True
    
    resolution = 2./60.


    y,m,d = daytimeconv.dfb2ymd(dfb_begin)

    total_bbox = latlon.seg_list_to_bbox(segment_list, resolution)
    total_data = numpy.empty((dfb_end-dfb_begin+1,slot_end-slot_begin+1,total_bbox.height,total_bbox.width))

    for seg in segment_list:
        seg_col,seg_row = seg
        bbox = latlon.get_5x5_seg_bbox(seg_row, seg_col, resolution)
        #ncfilebasename = MSG1IODC_2017_02_VIS006_c55_r08.nc
        ncfilebasename = filename_pattern % (Satellite, chan, y, m, seg_row, seg_col)
        ncfile = os.path.join(sat_data_path,ncfilebasename)

        data = latlon_nctools.latlon_read_dfb_slot_lat_lon_nc(ncfile, chan,[dfb_begin,dfb_end], [slot_begin,slot_end], bbox, interpolate='nearest')

        px_xmin, px_xmax, px_ymin, px_ymax = total_bbox.pixel_coords_of_bbox(bbox)
        total_data[:,:,px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data


    print "chan:%s" % (chan)
    print "dir:", sat_data_path
    print "valid values [perc]: %.1f" % (100*(data==data).sum()/data.size)
    print "shape:", data.shape
    print "min:%.2f, max:%.2f" % (data[data==data].min(), data[data==data].max())
    
    map_data_3d=data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3]))
    

    aDS_list = []
    for dfb in range(dfb_begin,dfb_end+1):
        aDs = daytimeconv.dfb2yyyymmdd(dfb)
        for slot in range(slot_begin,slot_end+1):
            aDSs = aDs + "_%03d"%slot
            aDS_list.append(aDSs)

    aDS_arr = numpy.array(aDS_list)


    if skip_empty:     
        aux=map_data_3d.mean(axis=2).mean(axis=1)
        wh=(aux> 0) & (aux < 32000)
        map_data_3d = map_data_3d[wh,:,:]
        # print aux
        aDS_arr = aDS_arr[wh]
        
    
    print 'preparing plot...'
    
    
    latlon.visualize_map_3d(map_data_3d, total_bbox, vmin=vmin, vmax=vmax, interpolation='nearest', subplot_titles_list=aDS_arr,color='gray')
    print 'DONE'
