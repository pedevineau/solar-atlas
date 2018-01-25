#! /usr/bin/env python
# visualisation of processing segments with basemap
#
# last revision: 10/28/2009
#


import netCDF4
import numpy
import math
import os

from general_utils import latlon
from general_utils import latlon_nctools






#------------------------------------------------------------
# main
#------------------------------------------------------------
if __name__ == "__main__":

	seg_c_min, seg_c_max, seg_r_min, seg_r_max = 63, 63, 10, 11

	data_path_pool=["/home1/model_data_himawari/data_geom/"]

	
	nc_var_name='UB'
	
	out_prefix='himawari_UB'
	out_suffix='.nc'

	res=1./30.
	
	#calculate subsegments parameters
	xmin_total, xmax_total, ymin_total, ymax_total = 99999, -99999, 99999, -99999
	subsegs={}
#	print 'segments:'
	for seg_c in range(seg_c_min, seg_c_max+1):
		for seg_r in range(seg_r_min, seg_r_max+1):
			seg_rc = seg_r*100+seg_c
			seg_bbox = latlon.get_5x5_seg_bbox(seg_r, seg_c, res)
			xmin_total=min(xmin_total,seg_bbox.xmin)
			xmax_total=max(xmax_total,seg_bbox.xmax)
			ymin_total=min(ymin_total,seg_bbox.ymin)
			ymax_total=max(ymax_total,seg_bbox.ymax)
#			print seg_c, seg_r
			outdata_seg_suffix="_c%d_r%d" % (seg_c, seg_r)
			subsegs[seg_rc]={'seg_c':seg_c, 'seg_r':seg_r, 'suffix':outdata_seg_suffix, 'bbox':seg_bbox}

	#create empty array for all data
	width=int(math.floor(((xmax_total-xmin_total)/res)+0.49))
	height=int(math.floor(((ymax_total-ymin_total)/res)+0.49))
	bbox = latlon.bounding_box(xmin=xmin_total, xmax=xmax_total, ymin=ymin_total, ymax=ymax_total, width=width, height=height, resolution=res)
	
	data_total = numpy.empty((12, height, width))
	print 'total:', data_total.shape
	
	#read the data
	for seg_rc in subsegs.keys():
		seg_c = subsegs[seg_rc]['seg_c']
		seg_r = subsegs[seg_rc]['seg_r']
		outdata_seg_suffix = subsegs[seg_rc]['suffix']
		seg_bbox = subsegs[seg_rc]['bbox']
		print seg_c, seg_r, seg_bbox
		
		nc_file_found=None
		for data_path in data_path_pool:
			base_name = "%s%s%s" % (out_prefix,outdata_seg_suffix, out_suffix)
			nc_file=os.path.join(data_path,base_name)
			if (not(os.access(nc_file,os.F_OK)) or (os.path.isdir(nc_file))):
				continue
			else:
				nc_file_found=nc_file
				print ' ', nc_file
				
		if nc_file_found is None:
			print 'file not found:',  base_name
			continue

		print 'reading file:',  nc_file_found
		data=latlon_nctools.latlon_read_lat_lon_nc_bbox(nc_file_found, nc_var_name, seg_bbox=seg_bbox, interpolate='nearest')
		if data is not None:
			px_xmin, px_xmax, px_ymin, px_ymax = bbox.pixel_coords_of_bbox(seg_bbox) 
			try:
				data_total[:,px_ymin:px_ymax+1, px_xmin:px_xmax+1] = data
			except:
				print data.shape
				print data_total[:,px_ymin:px_ymax+1,px_xmin:px_xmax+1].shape
				print 'shape mismatch'

	print 'pixel 0, 0 values', data_total[:,0,0]
	print 'preparing plot'
	shp=data_total.shape
#	vals=data_total.reshape(shp[0]*shp[1],shp[2],shp[3])
	vals_ma=numpy.ma.masked_where(numpy.isnan(data_total),data_total)
	latlon.visualize_map_3d(vals_ma, bbox, vmin=0, vmax=None, interpolation='nearest', show_grid=False)

