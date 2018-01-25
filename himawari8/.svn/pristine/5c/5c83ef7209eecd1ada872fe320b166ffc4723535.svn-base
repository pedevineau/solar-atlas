'''
Created on Feb 13, 2016

@author: tomas
'''

import numpy
import math



def plot_LB_data(site, name, datalist, titlelist=['','',''],limitslist=[[None,None],[None,None],[None,None]], counours=[None, None, None], colormaps=[None, None, None], saveplot=False, showplot=False):
    from pylab import cm, figure, imshow,colorbar, title, grid, xticks, yticks, savefig, show
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import LinearSegmentedColormap
    #colorbars
    cdict3 = {'red':  ((0.0, 0.0, 0.0),
                       (0.35,0.5, 0.5),
                       (0.5, 0.97, 0.97),
                       (0.65, 0.97, 0.97),
                       (1.0, 0.5, 0.5)), 
    
             'green': ((0.0, 0., 0.),
                       (0.35,0.6, 0.6),
                       (0.5, 0.97, 0.97),  
                       (0.65,0.6, 0.6),
                       (1.0, 0., 0.)), 
    
             'blue':  ((0.0, 0.5, 0.5),
                       (0.35, 0.97, 0.97),
                       (0.5, 0.97, 0.97),
                       (0.65,0.5, 0.5),
                       (1.0, 0.0, 0.0))  
            } 
    blue_red3 = LinearSegmentedColormap('BlueRed3', cdict3) 

    
    axes=[]
    y_step=1./len(datalist)
    # image plots snow
    fig3 = figure(num=11,figsize=(11.0,10.),facecolor='w')
    fig3.clear()
    for i in range(0,len(datalist)):
        adata=datalist[i]
        atitle=titlelist[i] + ' (%d )' % (site)
        alimits=limitslist[i]
#        acountour=counours[i]
        acolormap_name=colormaps[i]
        if acolormap_name=="jet":
            acolormap=cm.jet
        elif acolormap_name=="blue_red3":
            acolormap=blue_red3
        else:
            acolormap=cm.jet
        if acolormap is None: acolormap=cm.jet
        y_bottom=i*y_step + 0.07*y_step
        if i ==0:
            ax = fig3.add_axes([0.06,y_bottom,1.,0.83*y_step])
        else:
            ax = fig3.add_axes([0.06,y_bottom,1.,0.83*y_step], sharex=axes[0], sharey=axes[0])
        zma=numpy.ma.masked_where(numpy.isnan(adata) , adata)
        zma=numpy.rot90(zma)
        imshow(zma, cmap=acolormap, interpolation='nearest', vmin=alimits[0], vmax=alimits[1])
        cb=colorbar(shrink=0.65)
        #if not (acountour is None):
            #contour(rot90(adata.copy()), np.arange(alimits[0], alimits[1], acountour), hold='on', colors = 'k',origin='lower',linewidths=0.5, alpha=.5 )

        fp=FontProperties(size='x-small')
        
        title(atitle, fontproperties=fp )
        grid(True)
        xticks(rotation=0, size='xx-small')
        yticks(rotation=0, size='xx-small')
        for t in cb.ax.get_yticklabels():
            t.set_fontsize('xx-small')
        axes.append(ax)
            
    if saveplot:
        savefig ('LB_quant_regres_%d.png'%site)
    if showplot:
        show()


#generate numpy array with weights in slot dimension
#the size in s dimension is doubled to have highest value centered, general slot weights are created. 
def slot_weight_general(window_size=60, slots=48, s_std=3):
    s_weight= numpy.empty((window_size, slots*2), dtype= "float64")
    for i in range(0,slots*2):
        x_dist=slots-i
        s_weight[:,i]=gaussian_kernel(x_dist, s_std)
    s_weight = s_weight / s_weight.max()
    return s_weight

#select weights of given slot from general slot weights 
def slot_weight_for_slot(s_weight_general, slots, my_slot):
    s_weight_slot=s_weight_general[:,(slots-my_slot):(slots+slots-my_slot)]
    return s_weight_slot


#make dictionary for slot weights
def slot_weight_for_slot_dict(slot_weight_general,slots):
    slot_weight_dict={}
    for s in range(0,slots):
        slot_weight_dict[s]=slot_weight_for_slot(slot_weight_general, slots, s)
    return slot_weight_dict

#generate numpy array with weights in day dimension
def day_weight(window_size=60, slots=48, d_std=40.):
    d_weight= numpy.empty((window_size, slots), dtype= "float64")
    for i in range(0,window_size):
        x_dist=window_size-1-i
        d_weight[i,:]=gaussian_kernel(x_dist, d_std)
    d_weight = d_weight / d_weight.max()
    return d_weight


#calculate gausian kernel for given distance and std
def gaussian_kernel(x_dist, x_std):
    return (1./(x_std*math.sqrt(2.*math.pi)))*math.exp(-0.5*pow(x_dist,2.)/pow(x_std,2.))


def day_slot_dict(s_weight_sharp_limit_dict, d_weight_land):
    slots=s_weight_sharp_limit_dict.keys()
    d_s_dict = {}
    for s in slots:
        d_s_dict[s]=s_weight_sharp_limit_dict[s]*d_weight_land
    return d_s_dict

def LB_quantile_regression_init_weights(window_size, slots, gauss_fltr_day_std, gauss_fltr_slot_std_sharp, gauss_fltr_slot_std_flat):
    #calculate weights for moving window
    #in slot direction
    s_weight_sharp_general = slot_weight_general(window_size=window_size, slots=slots, s_std=gauss_fltr_slot_std_sharp)
    s_weight_sharp_dict=slot_weight_for_slot_dict(s_weight_sharp_general,slots)
    
    s_weight_sharp_general_limit =s_weight_sharp_general.copy() 
    s_weight_sharp_general_limit[s_weight_sharp_general_limit<0.00001]=0.00001
    s_weight_sharp_limit_dict=slot_weight_for_slot_dict(s_weight_sharp_general_limit,slots)
    
    s_weight_flat_general = 0.5*slot_weight_general(window_size=window_size, slots=slots, s_std=gauss_fltr_slot_std_flat)
#    s_weight_flat_general=s_weight_flat_general * sum(s_weight_sharp_general_limit[numpy.logical_not(numpy.isnan(s_weight_sharp_general_limit))])/sum(s_weight_flat_general[numpy.logical_not(numpy.isnan(s_weight_flat_general))])
    s_weight_flat_dict=slot_weight_for_slot_dict(s_weight_flat_general,slots)

    #in day direction
    d_weight_land = day_weight(window_size=window_size, slots=slots, d_std=gauss_fltr_day_std)

    d_s_sharp_weight_land_dict=day_slot_dict(s_weight_sharp_limit_dict, d_weight_land)
    return [s_weight_sharp_dict, s_weight_sharp_limit_dict, s_weight_flat_dict, d_weight_land, d_s_sharp_weight_land_dict]


def LB_quantile_regression_init_weights_with_param(window_size, num_slots):
    gauss_fltr_day_std = 15.5 # standard deviation of gaussian filter in day dimension 
    gauss_fltr_slot_std_sharp = 1.5 # standard deviation of gaussian filter in slot dimension for snow/cloud free 
    gauss_fltr_slot_std_flat = 9.0
    # standard deviation of gaussian filter in slot dimension for specific cases where more  influence in slot dimension is required
    LB_QR_weights = LB_quantile_regression_init_weights(window_size, num_slots, gauss_fltr_day_std, gauss_fltr_slot_std_sharp, gauss_fltr_slot_std_flat)
    return LB_QR_weights

def sat_data_missing_init_days(sat_data, window_size):
    min_number_of_valid_slots = 15
    missing_days=numpy.zeros((window_size),dtype=numpy.int8)
    for i in range(0, window_size):
        if numpy.logical_not(numpy.isnan(sat_data[i,:])).sum() <= min_number_of_valid_slots:
            missing_days[i]=1
    return missing_days
 
def LB_quantile_regression_init_window_data(window_size, land_data, snow_data, full_data, mask_data, missing_days = None):
    verbose = False
    landsnow_data=land_data.copy()
    landsnow_data[numpy.logical_not(numpy.isnan(snow_data))]=snow_data[numpy.logical_not(numpy.isnan(snow_data))]

    window_data_landsnow = landsnow_data[0:window_size,:].copy().astype(numpy.float64)
    window_data_full = full_data[0:window_size,:].copy().astype(numpy.float64)
    window_data_land = land_data[0:window_size,:].copy().astype(numpy.float64)
    window_data_mask = mask_data[0:window_size,:].copy().astype(numpy.float64)
    
    data_len=len(land_data)


    if missing_days is not None:
        wh_missing_days = missing_days ==1
        if (missing_days.sum()>1):
            if data_len>=365+window_size:
                if verbose: print "init window missing days %d filled from more than 365 days" % (missing_days.sum())
                window_data_landsnow[wh_missing_days,:]=landsnow_data[365:365+window_size,:][wh_missing_days,:]
                window_data_full[wh_missing_days,:]=full_data[365:365+window_size,:][wh_missing_days,:]
                window_data_land[wh_missing_days,:]=land_data[365:365+window_size,:][wh_missing_days,:]
                window_data_mask[wh_missing_days,:]=mask_data[365:365+window_size,:][wh_missing_days,:]
            elif (data_len>=365) and (data_len<(365+window_size)):
                if verbose: print "init window missing days %d filled from 365 days" % (missing_days.sum())
                window_data_landsnow[wh_missing_days,:]=landsnow_data[-window_size:,:][wh_missing_days,:]
                window_data_full[wh_missing_days,:]=full_data[-window_size:,:][wh_missing_days,:]
                window_data_land[wh_missing_days,:]=land_data[-window_size:,:][wh_missing_days,:]
                window_data_mask[wh_missing_days,:]=mask_data[-window_size:,:][wh_missing_days,:]
            elif data_len>=window_size+window_size:
                if verbose: print "init window missing days %d filled from beginning days" % (missing_days.sum())
                window_data_landsnow[wh_missing_days,:]=landsnow_data[window_size:window_size+window_size,:][wh_missing_days,:]
                window_data_full[wh_missing_days,:]=full_data[window_size:window_size+window_size,:][wh_missing_days,:]
                window_data_land[wh_missing_days,:]=land_data[window_size:window_size+window_size,:][wh_missing_days,:]
                window_data_mask[wh_missing_days,:]=mask_data[window_size:window_size+window_size,:][wh_missing_days,:]
            elif data_len>=window_size:
                if verbose: print "init window missing days %d filled from recent days" % (missing_days.sum())
                window_data_landsnow[wh_missing_days,:]=landsnow_data[-window_size:,:][wh_missing_days,:]
                window_data_full[wh_missing_days,:]=full_data[-window_size:,:][wh_missing_days,:]
                window_data_land[wh_missing_days,:]=land_data[-window_size:,:][wh_missing_days,:]
                window_data_mask[wh_missing_days,:]=mask_data[-window_size:,:][wh_missing_days,:]

    del landsnow_data
    
    return [window_data_landsnow, window_data_land, window_data_mask, window_data_full]


##average filter (average of three subsequent values) on vector data
def vector_avg_filter(vector_data, onlyNaN=True):
    out_data2=vector_data.copy()
    day_data=numpy.empty((vector_data.shape[0],4),dtype=vector_data.dtype)
    day_data[:,0]=numpy.roll(vector_data,+1)
    day_data[:,1]=numpy.roll(vector_data,-1)
#    day_data[:,2]=vector_data.copy()

    day_data[0,0]=vector_data[0]
    day_data[-1,1]=vector_data[-1]

#    wh = numpy.isnan(vector_data)
#    day_data=numpy.ma.masked_where(day_data != day_data, day_data)
#    out_data2[wh] = (day_data[wh,0]+day_data[wh,1]) *0.5

    #fill nans
    wh = numpy.isnan(vector_data)
    wh2 = wh & (day_data[:,0]==day_data[:,0]) &  (day_data[:,1]!=day_data[:,1]) 
    out_data2[wh2] = day_data[wh2,0]
    wh2 = wh & (day_data[:,0]!=day_data[:,0]) &  (day_data[:,1]==day_data[:,1]) 
    out_data2[wh2] = day_data[wh2,1]
    wh2 = wh & (day_data[:,0]==day_data[:,0]) &  (day_data[:,1]==day_data[:,1]) 
    out_data2[wh2] = (day_data[wh2,0]+day_data[wh2,1]) *0.5

    #make weighted average of all values
    if not onlyNaN:
        day_data[:,:] = numpy.nan
        day_data[:,0]=numpy.roll(out_data2,+1)
        day_data[:,1]=numpy.roll(out_data2,-1)
        day_data[:,2]=out_data2.copy()  # it is twice to have double weight to original value 
        day_data[:,3]=out_data2.copy()  
        day_data_ma =numpy.ma.masked_where((day_data != day_data) , day_data)
        out_data2 = day_data_ma.mean(axis=1) 
        out_data2 = numpy.ma.filled(out_data2, numpy.nan)
        
    return out_data2



#init flags and last_days
def _init_flags(window_size, window_data, window_data_land):
    #daily flags
    window_land_flag=numpy.any(numpy.logical_not (numpy.isnan(window_data_land)), axis=1)
    window_data_snow=window_data.copy()
    window_data_snow[numpy.logical_not(numpy.isnan(window_data_land))] = numpy.nan
    window_snow_flag=numpy.any(numpy.logical_not (numpy.isnan(window_data_snow)), axis=1)
    del(window_data_snow)

    #last day of snow and land occurrence
    snow_last_day=-window_size-1
    land_last_day=-window_size-1
    snow_cumul=0
    snow_cumul_stop=False
    for i in range(0, window_size):
        if window_snow_flag[-i-1] and (snow_last_day<-i-1):
            snow_last_day=-i-1
        if window_land_flag[-i-1] and (land_last_day<-i-1):
            land_last_day=-i-1
            if not window_snow_flag[-i-1]:
                snow_cumul_stop=True
        if window_snow_flag[-i-1] and not (snow_cumul_stop):
            snow_cumul+=1

    #set last flag: 
    UNKNOWN, LAND, SNOW, LANDSNOW = 0, 1, 2, 3
    if snow_last_day>land_last_day:
        last_flag=SNOW 
#        snow_cumul=1
    elif land_last_day>snow_last_day:
        last_flag=LAND
        snow_cumul=0
    elif (snow_last_day==land_last_day) and (snow_last_day>-window_size-1):
        last_flag=LANDSNOW
#        snow_cumul=1
    else: 
        last_flag=UNKNOWN #

    return window_land_flag, window_snow_flag, snow_last_day, land_last_day, last_flag, snow_cumul


        
def percentile_based_LB_for_window(slots, window_size, window_data_full, avg_filter_runs=2):
    import msgmdl #@UnresolvedImport
    perc_lb_day=numpy.empty((slots), dtype=numpy.float64)
    perc_lb_day[:]=numpy.nan
    min_size=window_size*0.2
    window_size_half=int(window_size*0.5)
    
    for s in range(0,slots):
        wh = window_data_full[:,s]==window_data_full[:,s]
        sel_data = window_data_full[wh,s].copy()
        if wh.sum() > min_size:
            perc_lb_day[s]=msgmdl.scoreatpercentile(sel_data, 10.)
        wh = window_data_full[-window_size_half:,s]==window_data_full[-window_size_half:,s]
        sel_data = window_data_full[-window_size_half:,s]
        sel_data=sel_data[wh].copy()
        if wh.sum() > min_size:
            perc_lb_day[s]=(perc_lb_day[s]*0.5) + (msgmdl.scoreatpercentile(sel_data, 15.)*0.5)

    for i in range(0,avg_filter_runs):
        msgmdl.vector_avg_filter(perc_lb_day)
    
    return perc_lb_day



def _scoreatpercentile(a, per):
    #return percentile
    values = numpy.sort(a,axis=0)

    idx = per /100. * (values.shape[0] - 1)
    if (idx % 1 == 0):
        return values[idx]
    else:
        return values[int(idx)] + (values[int(idx) + 1] - values[int(idx)])*(idx % 1);



def _rolling_window_update(d, window_size, window_data, window_data_land,  window_data_full, window_data_mask, window_land_flag, window_snow_flag, all_data, land, snow, full, mask ):
    
    #re-init previous days -  restore original state if data from previous days were changed by excessive filter
    if d>0:
        in_idx_b=max(0,(d-1)-(window_size-1))
        in_idx_e=d-1+1
        indxs=(in_idx_e-in_idx_b)
        out_idx_b=-indxs
        if indxs==1:
            window_data[out_idx_b,:]=all_data[in_idx_b,:]
            window_data_land[out_idx_b,:]=land[in_idx_b,:]
            window_data_full[out_idx_b,:]=full[in_idx_b,:]
            window_data_mask[out_idx_b,:]=mask[in_idx_b,:]
        else:
            window_data[out_idx_b:,:]=all_data[in_idx_b:in_idx_e,:]
            window_data_land[out_idx_b:,:]=land[in_idx_b:in_idx_e,:]
            window_data_full[out_idx_b:,:]=full[in_idx_b:in_idx_e,:]
            window_data_mask[out_idx_b:,:]=mask[in_idx_b:in_idx_e,:]

    #init current day
    # data window
    window_data=numpy.roll(window_data,shift=-1,axis=0)
    window_data[-1,:]=all_data[d,:]
    
    window_data_land=numpy.roll(window_data_land,shift=-1,axis=0)
    window_data_land[-1,:]=land[d,:]

    window_data_full=numpy.roll(window_data_full,shift=-1,axis=0)
    window_data_full[-1,:]=full[d,:]

    window_data_mask=numpy.roll(window_data_mask,shift=-1,axis=0)
    window_data_mask[-1,:]=mask[d,:]

    # daily flags
    window_land_flag=numpy.roll(window_land_flag,shift=-1,axis=0)
    window_land_flag[-1]=numpy.any(numpy.logical_not (numpy.isnan(land[d,:])))
    window_snow_flag=numpy.roll(window_snow_flag,shift=-1,axis=0)
    window_snow_flag[-1]=numpy.any(numpy.logical_not (numpy.isnan(snow[d,:])))

    return window_data, window_data_land, window_data_full, window_data_mask, window_land_flag, window_snow_flag

def _rolling_window_counts(window_data, window_data_land, window_data_mask):
    #counts for last day
    day_land_count=(numpy.logical_not(numpy.isnan(window_data_land[-1,:]))).sum()
    day_all_count=(numpy.logical_not(numpy.isnan(window_data[-1,:]))).sum()
    day_mask_count=(window_data_mask[-1,:]).sum()

    #calculate ratio of valid LAND for given day
    if day_mask_count > 0:
        day_land_count_ratio=float(day_land_count)/day_mask_count
        all_count_ratio=float(day_all_count)/day_mask_count
    else:
        day_land_count_ratio=-1
        all_count_ratio=-1
    #ratio of the land class from whole day valid timeslots recalculated to <0,1> from <0.2, 0.8>
    wght_land_ratio=(min(0.80, max(0.10,day_land_count_ratio))-0.10)/0.7
    #ratio of the all class from whole day valid timeslots recalculated to <0.6,1.2>
    wght_all_ratio=((min(0.85, max(0.25,all_count_ratio))-0.25))+0.6
    
    return day_land_count,day_land_count_ratio, wght_land_ratio, wght_all_ratio



def _rolling_window_excessive_filter(d,window_data_land, wght_land_ratio, LB, LB_previous, do_LB_loaded, day_land_count):
    #EXCESSIVE filter - LAND ONLY mask out land data from window with exceptionally high land value (higher than previous LBland)
    #dynamic thresholds:
    #filter depends on wght_land_ratio - number of present classified data in given day: less data > lower limit
    land_excessive_count=0
    limit_min, limit_max = 1.6, 3.0
    limit=((wght_land_ratio*limit_max)+((1.-wght_land_ratio)*limit_min))
    if ((d>0) or do_LB_loaded) and (day_land_count>0): 
        LB_previous2 = LB_previous.copy()
        LB_previous2[LB_previous2<10]=10 # calculate it from at least 1.5
        wh = window_data_land[-1,:] > (LB_previous2*limit)
        land_excessive_count=wh.sum()
        if land_excessive_count > 0:
            window_data_land[-1,wh]=numpy.NaN
    return window_data_land, land_excessive_count




#version with removed uncertainty calculation
def LB_quantile_regression3(land, snow, full, mask5, mask0, window_size, window_history_data, weights ,lb_optimize_min_lim, lb_optimize_max_lim, lb_optimize_converg_lim, LBquantile, LB_loaded=None, LBland_loaded=None,  dfb_corr_quantile=None):

    #try to load msgmdl module - C optimized quantile regression calculation
    LB_slot_step=1
    try:
        import msgmdl #@UnresolvedImport
    except:
        print 'error. unable to import msgmdl.so module'
        exit()
        
    #note land is used for cloud/snow free, and includes also sea and other water bodies
#    verbose=True
    days, slots = land.shape
    
    #---------input variables-------------
    #read pre-calculated weights and trail window data
    s_weight_sharp_dict, s_weight_sharp_limit_dict, s_weight_flat_dict, d_weight_land, d_s_sharp_weight_land_dict = weights
    window_landsnow, window_data_land, window_data_mask,  window_data_full = window_history_data
    d_weight_land_last_day = d_weight_land[-1,:].copy()
#    combined_weight_last_day_arr=numpy.empty_like(d_weight_land_last_day)

    aux1_day_arr=numpy.empty_like(d_weight_land_last_day)
    aux2_day_arr=numpy.empty_like(d_weight_land_last_day)
    aux3_day_arr=numpy.empty_like(d_weight_land_last_day)
    
    #merge snow and snow/cloud free data
    landsnow=land.copy()
    landsnow[numpy.logical_not(numpy.isnan(snow))]=snow[numpy.logical_not(numpy.isnan(snow))]


    if dfb_corr_quantile is None:
        dfb_corr_quantile=numpy.zeros((days))
    
    #check loaded (history) LB
    do_LB_loaded = (LB_loaded is not None) and (LBland_loaded is not None) and (len(LB_loaded) == slots) and (len(LBland_loaded) == slots)
    if do_LB_loaded:
        LB_loaded = vector_avg_filter(LB_loaded,onlyNaN=True)
        LBland_loaded = vector_avg_filter(LBland_loaded,onlyNaN=True)
    
    
    #-----------output variables--------------
    #make out total LB grid
    LB=numpy.empty(land.shape,dtype=numpy.float64)
    LB[:,:]=numpy.NaN
    
    #LB output for cloud/snow free only - it is later modified by snow albedo and put to LB[:,:]
    LBland=numpy.empty(land.shape,dtype=numpy.float64)
    LBland[:,:]=numpy.NaN
    LBland_previous=LBland[-1,:]

    #--------- temporary variables ---------------
    #last day LB source: True - calculated or taken from previous day, False - extrapolated (by avg filter) 
    LB_source = numpy.empty((slots), dtype=numpy.bool)
    LB_source[:] = False
    
    # if input value is NaN , then If it is in between two valid values then true, otherwise NaN is at tail
    middle_nan = numpy.empty((slots), dtype=numpy.bool)
    middle_nan[:] = False

    #init flags and last days
#    UNKNOWN, LAND, SNOW, LANDSNOW = 0, 1, 2, 3
    window_land_flag, window_snow_flag, snow_last_day, land_last_day, last_flag, snow_cumul = _init_flags(window_size, window_landsnow, window_data_land)


    #INIT LBland for FIRST run
    #if possible init LB land with mean value of all land data; only FIRST run NOT CONTINUE RUN (reading old LBs)
    LBland_previous_initiated = False
    if (not do_LB_loaded):
        if ((land==land).sum()>350): #at least 350 valid pixels.must be present to do this
            #print 'available 350 pixels'
            wh=mask5[0,:]==True
            land_ma =numpy.ma.masked_where((land != land) | numpy.logical_not(mask5), land)
            amean=land_ma.mean(axis=0)
            amean = numpy.ma.filled(amean, numpy.nan)
            amean=vector_avg_filter(amean, onlyNaN=False)
            amean=vector_avg_filter(amean, onlyNaN=False)
            LBland_previous[wh]=amean[wh].astype(numpy.float32)
            LBland_previous_initiated = (LBland_previous==LBland_previous).sum() > 0
        else:
            full_sel=full[0:min(90, full.shape[0])]
            for s in range(0,slots):
                wh = full_sel[:,s] == full_sel[:,s]
                if wh.sum()>1:
                    LBland_previous[s]=_scoreatpercentile(full_sel[wh,s], 7)
            msgmdl.vector_avg_filter(LBland_previous)
            msgmdl.vector_avg_filter(LBland_previous)
            msgmdl.vector_avg_filter(LBland_previous)
            LBland_previous=vector_avg_filter(LBland_previous, onlyNaN=False)
            wh=mask5[0,:]==False
            LBland_previous[wh]=numpy.nan
            LBland_previous_initiated = (LBland_previous==LBland_previous).sum() > 0
            
        #if LBland was initiated then remove all pixels that are too high
        #: apply this also on all data??? Limit should be more relaxed
        if (LBland_previous==LBland_previous).sum() > 0:
            lim_value = LBland_previous[LBland_previous==LBland_previous].max() *3.5
            window_data_land[(window_data_land > lim_value)] = numpy.NaN

    #INIT for CONTINUE run
    if do_LB_loaded:
        LBland_previous=LBland_loaded
        LBland_previous_initiated=True
        LB_previous=LB_loaded
        del(LBland_loaded, LB_loaded)
        for s in range(0,slots,LB_slot_step):
            if mask5[0,s] == 1:
                LB_source[s]=True
    else: #OR FIRST RUN
        LB_previous=LBland_previous.copy()



    #--------PROCESS day by day----------
    for d in range(0,days):
#        a_DT=datetime.datetime.now()
#        if verbose: print "\n",d
        
        #check 3 consecutive days for high albedo
        aux_high_albed0_lim = 0.45
        d_high_count=0
        dl=land[d,:]
        wh_dl=(dl==dl)
        if ((wh_dl).sum()>0) and (dl[wh_dl].mean()>aux_high_albed0_lim):
            d_high_count+=1
        if d<(days-1):
            dl=land[d+1,:]
            wh_dl=(dl==dl)
            if ((wh_dl).sum()>0) and (dl[wh_dl].mean()>aux_high_albed0_lim):
                d_high_count+=1
        if d<(days-2):
            dl=land[d+2,:]
            wh_dl=(dl==dl)
            if ((wh_dl).sum()>0) and (dl[wh_dl].mean()>aux_high_albed0_lim):
                d_high_count+=1
        
        
        #put current day data to the window
        result =_rolling_window_update(d, window_size, window_landsnow, window_data_land,  window_data_full, window_data_mask, window_land_flag, window_snow_flag, landsnow, land, snow, full, mask5 )
        window_landsnow, window_data_land, window_data_full, window_data_mask, window_land_flag, window_snow_flag = result

        #calculate Percentile based LB
        LB_bypercentile_day = percentile_based_LB_for_window(slots, window_size, window_data_full, avg_filter_runs=1)
#        print d, 'perc_lb_day', LB_bypercentile_day
        
        #calculate ratio of valid LAND for given day and associated weights
        result = _rolling_window_counts(window_landsnow, window_data_land, window_data_mask)
        day_land_count,day_land_count_ratio, wght_land_ratio, wght_all_ratio = result
#        print d, 'land_count_ratio', day_land_count_ratio
        
        #EXCESSIVE filter - LAND mask out land data from window with exceptionally high land value (higher than previous LBland)
        window_data_land, land_excessive_count = _rolling_window_excessive_filter(d, window_data_land, wght_land_ratio, LBland,  LBland_previous, do_LB_loaded, day_land_count)
        
        #window_data_land valid data
        wh_d_land = window_data_land==window_data_land
        wh_d_land_sum = wh_d_land.sum()
        
        #daily average of previous LB 
        LBland_previous_mean = LBland_previous[LBland_previous==LBland_previous].mean()

        #prioritize current day weight
        #less land class data - lower priority
        priority_min, priority_max = 0.90, 2.1
        prioritize_current_day=((wght_land_ratio*priority_max)+((1.-wght_land_ratio)*priority_min))
#        print d, 'prioritize_current_day', wght_land_ratio, prioritize_current_day

#        a_Tdetla+=datetime.datetime.now()-a_DT
        #ADAPT QUANTILE by the amount of available data (in a given day???). If enough data, then increase quantile by 25%
        LB_quant=dfb_corr_quantile[d]+LBquantile
        if day_land_count_ratio>0.5:
            LB_quant=(LB_quant+((day_land_count_ratio-0.5)/(0.5))*(0.25)) 
#        print d, 'LB_quant', LBquantile, dfb_corr_quantile[d]+LBquantile, day_land_count_ratio, LB_quant


        #----------optimize LBland for each  slot-----
        for s in range(0,slots,LB_slot_step):
            # to speed up, skip calculation if we are outside the mask
            if not mask5[d,s]:
                continue

            #WEIGHT in SLOT direction
            s_weight_slot_flat=s_weight_flat_dict[s]
            s_weight_slot_flat_last_day=s_weight_slot_flat[-1,:]
            s_weight_slot_flat_last_day_max=s_weight_slot_flat_last_day.max()
            s_weight_slot_sharp=s_weight_sharp_limit_dict[s]
            s_weight_slot_sharp_last_day=s_weight_slot_sharp[-1,:]
            
            # combined weight in both directions (day and slot)
            # make working copy for current calculation - in which data will be adapted  
            day_slot_weight_land=d_s_sharp_weight_land_dict[s].copy()

            #adapt flattening of the current day weights in slot dimension according to the number of data available
            #less land class data > flatter
            #WEIGHT CURRENT DAY
            if wh_d_land[-1,s]: #pixel has valid value
                #if in given slot are available data
                w1=wght_land_ratio
                w2=1.1*(1.-w1)
            else:
                #if in given slot are NOT available data - be more flat
                w1=wght_land_ratio/2.
                w2=1.1*(1.-w1)
            #result is returned by reference in aux1_day_arr
            #combined_weight_last_day(in1_vect, w1, in2_vect, w2, in3_vect, w3, out_vect)   out_vect = ((in1_vect*w1) + (in2_vect*w2))*in2_vect*w3
            #day_slot_weight_land[-1,:]=d_weight_land_last_day * ((s_weight_slot_sharp_last_day*w1) + (s_weight_slot_flat_last_day*w2))* prioritize
            last_day_wght_max = msgmdl.combined_weight_last_day(s_weight_slot_sharp_last_day, w1, s_weight_slot_flat_last_day, w2, d_weight_land_last_day, prioritize_current_day, aux1_day_arr)
            day_slot_weight_land[-1,:] = aux1_day_arr
                    

            if wh_d_land_sum>0:
                max_wght=msgmdl.max_under_mask_2d(day_slot_weight_land, wh_d_land)
            else:
                max_wght=last_day_wght_max
            
            
            #ADD LAST LBland  with combination of flat and sharp weight
            if LBland_previous_initiated:
                #weight_sum_last_day(in1_vect, w1, in2_vect, w2, w3, out_vect)   out_vect = ((in1_vect*w1) + (in2_vect*w2))*w3
                w1,w2,w3 = 0.8, 0.05, max_wght
                msgmdl.weight_sum_last_day(s_weight_slot_sharp_last_day, w1, s_weight_slot_flat_last_day, w2, w3, aux2_day_arr)
                #constant 0.9 added to decrease the influence (15. Nov 2015) 
#                day_slot_weight_land[0,:]=aux2_day_arr
                day_slot_weight_land[0,:]=aux2_day_arr*0.95
                window_data_land[0,:]=LBland_previous
                
            #ADD LB - statistically derived (percentile)
            wh1=True
            if LBland_previous_initiated:
                wh1=window_data_land[-1,s]<(LBland_previous[s]*1.2)  
            if (LB_bypercentile_day[s]==LB_bypercentile_day[s]) and wh1 and True:
                #weight_sum_last_day(in1_vect, w1, in2_vect, w2, w3, out_vect)   out_vect = ((in1_vect*w1) + (in2_vect*w2))*w3
                w1,w2,w3 = 0.7, 0.05, max_wght*(1.-day_land_count_ratio)*(1.0-wght_land_ratio)
#                print w1,w2,w3, max_wght, land_count_ratio, wght_land_ratio
                msgmdl.weight_sum_last_day(s_weight_slot_sharp_last_day, w1, s_weight_slot_flat_last_day, w2, w3, aux3_day_arr) 
                day_slot_weight_land[1,:]=aux3_day_arr
                window_data_land[1,:]=LB_bypercentile_day
                

            #if we have high albedo days - change weights
            if (d_high_count>1):
                day_slot_weight_land[-1,:]=s_weight_slot_flat_last_day*4.5
                day_slot_weight_land[-1,s]*=1.2
                day_slot_weight_land[:-1,:]*=0.5
                
            #
            if LBland_previous_initiated and (LBland_previous_mean >0.45) and (wght_land_ratio>0.1) and (d_high_count==0):
                wh=window_data_land>0.45
                #decrease influence of high albedo days
                day_slot_weight_land[wh]*=0.1
                #????????
                day_slot_weight_land[-1,:]=day_slot_weight_land[-1,:]*2+(s_weight_slot_flat_last_day/s_weight_slot_flat_last_day_max)
                    

#            import pylab
#            # all_data, land, snow, full, mask
#            pylab.imshow(day_slot_weight_land.transpose(), interpolation='nearest')
#            pylab.colorbar()
#            pylab.show()
#            exit()

            #GET  available data and weights for current window for land
            #OPTIMIZE - reduce data if amount is high - remove those with low weights
#            if True: #this optimization seems to be faster, 
            wght_limit=max_wght/1000.
#            wh_lim_land = wh_d_land & (day_slot_weight_land>wght_limit)
            wh_lim_land =(day_slot_weight_land>wght_limit) & (window_data_land==window_data_land)

            sel_data_land=window_data_land[wh_lim_land]
            sel_weight_land=day_slot_weight_land[wh_lim_land]
               
                
            #ADAPT LIMITS - optimization limits to the last LB 
#            if ((d>0) or do_LB_loaded) and numpy.logical_not(numpy.isnan((LBland_previous[s]))):
            if LBland_previous_initiated and (LBland_previous[s]==LBland_previous[s]):
                lb_opt_min_lim, lb_opt_max_lim = max(0.01,LBland_previous[s] - 0.15), LBland_previous[s] + 0.25
            else:
                lb_opt_min_lim, lb_opt_max_lim = lb_optimize_min_lim, lb_optimize_max_lim*0.55

           
            #if high albedo days
            if (d_high_count>1):
                lb_opt_max_lim=1.2
            if LBland_previous_initiated and (LBland_previous_mean >90) and (wght_land_ratio>0.1) and (d_high_count==0):
                lb_opt_min_lim=0.25
                LB_quant=LB_quant-0.075
            lb_opt_max_lim = max(lb_opt_max_lim, (1.1 * lb_opt_min_lim))
             
            
            #OPTIMIZE LAND !!!!!
            LBland[d,s]=numpy.NaN
            if len(sel_data_land)>2:
                aLB_land = 0.
                aLB_land = msgmdl.fminbound_QR(lb_opt_min_lim, lb_opt_max_lim, lb_optimize_converg_lim, sel_data_land, sel_weight_land, LB_quant)

                LBland[d,s] = aLB_land

                # if enough data in given day then even more prioritize it use in LB (e.g. too high values are ignored in quantile regression)
                if wght_land_ratio> 0.95:
                    s_weight_slot_sharp_oneday=s_weight_sharp_dict[s][-1,:]
                    wh = wh_d_land[-1,:]
                    data_oneday=window_landsnow[-1,:]
                    LBland[d,s]=msgmdl.prioritize_current_day(s_weight_slot_sharp_oneday, data_oneday, wh, wght_land_ratio, LBland[d,s])


        #postprocessing - moving window filter
#        LBland[d,:] = vector_avg_filter(LBland[d,:],onlyNaN=True)
        msgmdl.vector_avg_filter(LBland[d,:])
        

        #-----PROCESS SNOW BASED ON DECISSION------
#        if (use_LBland):
        if (LBland[d,:]==LBland[d,:]).sum()>0: #LBland is defined, copy
            LB[d,:]=LBland[d,:]
        elif (d>0) or do_LB_loaded: #get LB from day-1
            LB[d,:]=LB_previous
        else: #no LB to use
            LB[d,:]=numpy.NaN

        LB_source[:]=True
                

        #postprocess filter (in s dimension)
        LB_input=LB[d,:].copy()
        LB[d,numpy.logical_not(LB_source)]=numpy.NaN
        msgmdl.vector_avg_filter(LB[d,:])
        msgmdl.vector_avg_filter(LB[d,:])
#        LB[d,:] = vector_avg_filter(LB[d,:],onlyNaN=True)
#        LB[d,:] = vector_avg_filter(LB[d,:],onlyNaN=True) #do it twice to extrapolate into NaN towards the tails


        wh = numpy.logical_not(mask0[d,:])
        LB[d,wh] = numpy.NaN
        LBland[d,wh] = numpy.NaN
        
#        wh = numpy.isnan(LB_input) & numpy.logical_not(numpy.isnan(LB[d,:]))
        wh = (LB_input!=LB_input) & (LB[d,:]==LB[d,:])
        LB_source[wh]=False
        del(LB_input)
        
        #this is very important - replace init data
        LB_previous=LB[d,:]
        LBland_previous=LBland[d,:]
        

    return (LB, LBland)


