#! /usr/bin/env python
# visualisation of processing segments with basemap
#
#

import netCDF4

import numpy
from general_utils import daytimeconv
from general_utils import latlon
from himawari8.sat_model.utils import himawari_nc_latlontools

#------------------------------------------------------------
# main
#------------------------------------------------------------
if __name__ == "__main__":
	himawari_slot_min = 1
	himawari_slot_max = 144


#	dfb_begin = daytimeconv.yyyymmdd2dfb('20140209')
#	dfb_end = daytimeconv.yyyymmdd2dfb('20140209')
	dfb_begin = daytimeconv.yyyymmdd2dfb('20170902')
	dfb_end = daytimeconv.yyyymmdd2dfb('20170902')

	slot_begin, slot_end = 1, 144
	roll_slots=+30 #  
	roll_slots=0 #
	
	segments_to_calculate = latlon.expand_segments([[56,57,12,13]])

	nc_var_name='GHI'
	nc_var_name='LBclass'
	data_path_pool=["/net/loge/data/model_data_himawari/data_output/v20b/" ]
	file_time_segmentation = "month"
	# data_path_pool=["/home0/model_data_himawari/data_output/nowcast_v20/" ]
	# data_path_pool=["/net/loge/data/model_data_himawari/data_output/nowcast_v20b/" ]
	# file_time_segmentation = "day"

	skip_empty=False

	vmin=0; vmax=None

	resolution=2./60.


	#-----------------

	#read data	
	data_total, bbox = himawari_nc_latlontools.read_multisegment_data(dfb_begin, dfb_end, slot_begin, slot_end, roll_slots, himawari_slot_min, himawari_slot_max, segments_to_calculate, nc_var_name, data_path_pool, resolution, file_time_segmentation=file_time_segmentation)
	
	

	print 'preparing plot'
	shp = data_total.shape
	map_data_3d=data_total.reshape((shp[0]*shp[1],shp[2],shp[3]))
	
	map_data_3d[map_data_3d==-99] = numpy.nan

	if skip_empty:	 
		aux=numpy.ma.masked_where(map_data_3d!=map_data_3d,map_data_3d)
		aux = aux.mean(axis=2).mean(axis=1)
		wh=aux> 0
		map_data_3d = map_data_3d[wh,:,:]

	
	title=daytimeconv.dfb2yyyymmdd(dfb_begin)+' - '+daytimeconv.dfb2yyyymmdd(dfb_end)
	latlon.visualize_map_3d(map_data_3d, bbox, vmin=vmin, vmax=vmax, interpolation='nearest', title=title)

	print 'done' 
