import math

def get_5x5_seg(lon,lat):
    seg_w,seg_e,seg_s,seg_n,seg_res=-180,180,-90,90,5. #GLOBAL segment region
    seg_col=int(math.floor((lon-seg_w)/seg_res))
    seg_row=int(math.floor((seg_n - lat)/seg_res))
    return seg_col, seg_row

def get_segment_name(longitude,latitude):
    # Determine the 5x5 deg segment name
    seg_col, seg_row=get_5x5_seg(longitude,latitude)
    seg_name='_ns'+str(seg_row)+'_ew'+str(seg_col)
    return seg_name

def get_segment_name_simple(longitude,latitude):
    # Determine the 5x5 deg segment name
    seg_col, seg_row=get_5x5_seg(longitude,latitude)
    seg_name='_'+str(seg_row)+'_'+str(seg_col)
    return seg_name