#! /usr/bin/env python
# visualisation of processing segments with basemap
#
# last revision: 10/28/2009
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

	seg_col, seg_row = 71, 21

	dfb_begin = daytimeconv.yyyymmdd2dfb('20160818')
	dfb_end = daytimeconv.yyyymmdd2dfb('20160818')
	slot_begin, slot_end = 1, 144


	nc_var_name='GHI'

	vmin=0; vmax=None
	out_data_path = '/home1/model_data_himawari/data_output/v20b/'
	file_time_segmentation='month'

#	out_data_path = '/home1/model_data_himawari/data_output/realtime_v20_test/'
#	file_time_segmentation='day'

	skip_empty = False

	#-----------------

	#sat data
	outdata_suffix="_c%d_r%d" % (seg_col, seg_row)

		
	#output data
	outdata_path_dict={"LB": out_data_path, "LBclass": out_data_path, "LBland": out_data_path,\
					 "CI": out_data_path, "KTM": out_data_path, "GHIc": out_data_path, "GHI": out_data_path,\
					"DNIc": out_data_path, "DNI": out_data_path, "GHIcor": out_data_path, "DNIcor": out_data_path, \
					"CI_flag": out_data_path}

	seg_bbox=latlon.get_5x5_seg_bbox(arow=seg_row, acol=seg_col, resolution=2./60.)

	out_files_dict = himawari_nc_latlontools.outdata_existingncfile_dict(dfb_begin, dfb_end, [nc_var_name], outdata_path_dict, outdata_suffix, file_time_segmentation=file_time_segmentation)
	print 'files: ',len(out_files_dict)
	
	
	data= himawari_nc_latlontools.outdata_nc_read(nc_var_name, out_files_dict, outdata_suffix, dfb_begin, dfb_end, slot_begin, slot_end, seg_bbox, file_time_segmentation=file_time_segmentation)
	
	
	map_data_3d=data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3]))

	aDS_list = []
	for dfb in range(dfb_begin,dfb_end+1):
		aDs = daytimeconv.dfb2yyyymmdd(dfb)
		for slot in range(slot_begin,slot_end+1):
			aDSs = aDs + "_%03d"%slot
			aDS_list.append(aDSs)

	aDS_arr = numpy.array(aDS_list)

	map_data_3d[map_data_3d==-99] = numpy.nan

	if skip_empty:	 
		aux=numpy.ma.masked_where(map_data_3d!=map_data_3d,map_data_3d)
		aux = aux.mean(axis=2).mean(axis=1)
		
		wh=aux> 0
		map_data_3d = map_data_3d[wh,:,:]
		aDS_arr = aDS_arr[wh]
	
	latlon.visualize_map_3d(map_data_3d, seg_bbox, vmin=vmin, vmax=vmax, interpolation='nearest', subplot_titles_list=aDS_arr)


		

