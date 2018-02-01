#! /usr/bin/env python
# point value from satelite model to db
#
# last revision: 18/02/2010
#
import datetime
import os
import sys

import numpy
from general_utils import daytimeconv
from general_utils import daytimeconv_num
from general_utils import latlon
from himawari8.sat_model.utils import himawari_nc_latlontools




#------------------------------------------------------------
# main
#------------------------------------------------------------
if __name__ == "__main__":


#	dfb_begin = daytimeconv.yyyymmdd2dfb('20130101')
#	dfb_end = daytimeconv.yyyymmdd2dfb('20141231')
	dfb_begin = daytimeconv.yyyymmdd2dfb('20150801')
	dfb_begin = daytimeconv.yyyymmdd2dfb('20170701')
	dfb_end = daytimeconv.yyyymmdd2dfb('20170430')
	dfb_end = daytimeconv.date2dfb(datetime.datetime.now().date())+1


	slot_begin, slot_end = 1, 144
	
	interpolation='bilinear' #bilinear or nearest
	
	sat_suff=""

#	nc_var_names=['GHI', 'DNI', 'GHIc', 'DNIc', 'CI_flag', 'CI', 'KTM']
	nc_var_names=['GHI']
	latitude = -32.65986
	longitude = 176.67472
	latitude = 39.65986
	longitude = 139.67472
	site_dict={\
#			'site19':{'latitude':40.-(1./60.), 'longitude':125.+(1./60.), 'source_dir':'/home0/model_data_mtsat/data_output/v19/'},\
#			'site20UL':{'latitude':26.+(17*2./60.)-(1./60.), 'longitude':102.+(19*2./60.)+(1./60.), 'source_dir':'/home0/model_data_mtsat/data_output/v20/'},\
#			'site20UR':{'latitude':26.+(17*2./60.)-(1./60.), 'longitude':102.+(20*2./60.)+(1./60.), 'source_dir':'/home0/model_data_mtsat/data_output/v20/'},\
#			'site20LL':{'latitude':26.+(16*2./60.)-(1./60.), 'longitude':102.+(19*2./60.)+(1./60.), 'source_dir':'/home0/model_data_mtsat/data_output/v20/'},\
#			'site20LR':{'latitude':26.+(16*2./60.)-(1./60.), 'longitude':102.+(20*2./60.)+(1./60.), 'source_dir':'/home0/model_data_mtsat/data_output/v20/'},\
#			'site63_10_LL ':{'latitude':35.+(0*2./60.)+(1./60.), 'longitude':135.+(0*2./60.)+(1./60.), 'source_dir':'/home1/model_data_himawari/data_output/v20/'},\
#			'site63_10_LL ':{'latitude':35.+(1*2./60.)+(1./60.), 'longitude':135.+(1*2./60.)+(1./60.), 'source_dir':'/home1/model_data_himawari/data_output/v20/'},\
#			'site63_10_LLt':{'latitude':35.+(0*2./60.)+(1./60.), 'longitude':135.+(0*2./60.)+(1./60.), 'source_dir':'/home1/model_data_himawari/data_output/v20_t/'},\
#			'site20':{'latitude':44.65986, 'longitude':141.67472, 'source_dir':'/net/carme/data/model_data_himawari/data_output/v20/'},\
#			'site20t':{'latitude':44.65986, 'longitude':141.67472, 'source_dir':'/net/carme/data/model_data_himawari/data_output/v20_test/'},\
#			'site20s':{'latitude':39.65986, 'longitude':141.67472, 'source_dir':'/net/surtur/data/model_data_himawari/data_output/v20/'},\
#			'site20m':{'latitude':39.65986, 'longitude':141.67472, 'source_dir':'/net/miranda/data/model_data_himawari/data_output/v20/'},\
#			'site20a':{'latitude':39.65986, 'longitude':141.67472, 'source_dir':'/net/ariel/data/model_data_himawari/data_output/v20a/'},\
#			'site20l':{'latitude':39.65986, 'longitude':141.67472, 'source_dir':'/net/loge/data/model_data_himawari/data_output/v20a/'},\
#			'site20m':{'latitude':44.65986, 'longitude':142.67472, 'source_dir':'/net/carme/data/model_data_himawari/data_output/v20/'},\
#			'site20m':{'latitude':-28.905830, 'longitude':115.111110, 'source_dir':'/net/carme/data/model_data_himawari/data_output/v20/'},\
#			'site20a':{'latitude':-28.905830, 'longitude':115.111110, 'source_dir':'/net/ariel/data/model_data_himawari/data_output/v20a/'},\
#			'site20l':{'latitude':-28.905830, 'longitude':115.111110, 'source_dir':'/net/loge/data/model_data_himawari/data_output/v20a/'},\
			# 'site20b_a':{'latitude':latitude, 'longitude':longitude, 'source_dir':'/net/ariel/data/model_data_himawari/data_output/v20b/'},\
			# 'site20b_l':{'latitude':latitude, 'longitude':longitude, 'source_dir':'/net/loge/data/model_data_himawari/data_output/v20b/'},\
			'site20b_r_a':{'latitude':latitude, 'longitude':longitude, 'source_dir':'/net/ariel/data/model_data_himawari/data_output/realtime_v20b/', 'file_time_segmentation':'day'},\
			'site20b_r_l':{'latitude':latitude, 'longitude':longitude, 'source_dir':'/net/loge/data/model_data_himawari/data_output/realtime_v20b/', 'file_time_segmentation':'day'},\
			'site20b_n_a':{'latitude':latitude, 'longitude':longitude, 'source_dir':'/net/ariel/data/model_data_himawari/data_output/nowcast_v20b/', 'file_time_segmentation':'day'},\
			'site20b_n_l':{'latitude':latitude, 'longitude':longitude, 'source_dir':'/net/loge/data/model_data_himawari/data_output/nowcast_v20b/', 'file_time_segmentation':'day'},\

			}
	

		


	#######################################################################
	# processing
	#######################################################################
	site_data_dict = {}
	for siteid, siteparams in site_dict.iteritems():
		lon = siteparams['longitude']
		lat = siteparams['latitude'] 
		file_time_segmentation = 'month'
		if siteparams.has_key('file_time_segmentation'):
			file_time_segmentation = siteparams['file_time_segmentation']
		print "Processing site", siteid, lon, lat

		
		if (lon is None):
			print "   Cannot read site coordinates, skipping site"
			continue

		
		seg_col, seg_row = latlon.get_5x5_seg(lon, lat)
		seg_suffix="_c%d_r%d" % (seg_col, seg_row)
		
		print 'reading site %s from c%d, r%d' % (siteid, seg_col, seg_row)
		print lon, lat
		print 'segment col %d  , row %d' % (seg_col, seg_row)
		#suffix added to output NETCDF file names
		if sat_suff=='':
			outdata_suffix=seg_suffix
		else:
			outdata_suffix="_%s%s" % (sat_suff, seg_suffix)


		#output data
		source_dir = siteparams['source_dir'] 
		outdata_path_dict={"LB": source_dir, "LBclass": source_dir, "LBland": source_dir,\
						 "CI": source_dir, "KTM": source_dir, "GHIc": source_dir, "GHI": source_dir,\
						"DNIc": source_dir, "DNI": source_dir, "GHIcor": source_dir, "DNIcor": source_dir, \
						"CI_flag": source_dir}

		seg_bbox=latlon.get_5x5_seg_bbox(arow=seg_row, acol=seg_col, resolution=2./60.)
			
		out_files_dict = himawari_nc_latlontools.outdata_existingncfile_dict(dfb_begin, dfb_end, nc_var_names, outdata_path_dict, outdata_suffix, file_time_segmentation=file_time_segmentation)

		if len(out_files_dict.keys())<1:
			print 'No NC files found'
			continue

			
		print 'Model outputs read:'
		data_dict={}
		for nc_var_name in nc_var_names:
			print nc_var_name,
			#To read data from raw netcdf model output
			rast_data= himawari_nc_latlontools.outdata_nc_read_point(nc_var_name, out_files_dict, outdata_suffix, dfb_begin, dfb_end, slot_begin, slot_end, lon, lat, interpolation=interpolation, file_time_segmentation=file_time_segmentation)
			data_dict[nc_var_name] = rast_data
		print ''

		aDTn=numpy.empty([dfb_end-dfb_begin+1, slot_end-slot_begin+1], numpy.float64)
		aT_offset=datetime.timedelta(hours=5./60.)
		for dfb_idx in range(0, dfb_end-dfb_begin+1):
			aD=daytimeconv.dfb2date(dfb_idx + dfb_begin)
			for slot_idx in range(0, slot_end-slot_begin+1):
				slot=slot_idx+slot_begin
	
#				aT = daytimeconv.slot2time(slot_idx+slot_begin,MSG=False, MFG_nominal=False)  #BAD
				aT = daytimeconv.dh2time((slot-1.)/6.)
				
				aDTreal = datetime.datetime.combine(aD, aT) + aT_offset
				aDTn[dfb_idx, slot_idx] = daytimeconv_num.date2num(aDTreal)
#				data_dict['aDT']=aDTreal
		data_dict['aDTn']=aDTn

		site_data_dict[siteid]=data_dict

		
	if len(site_data_dict.keys())<1:
		print 'nothing to plot, exit.'
		exit()

#	print site_data_dict.keys()

	import pylab
	from matplotlib.dates import DayLocator, DateFormatter, MonthLocator
	fig1 = pylab.figure(num=1,figsize=(14,5),facecolor='w')
	fig1.clear()
	ax1 = fig1.add_subplot(111)
	locatorM=MonthLocator()
	daysFmt = DateFormatter('%y-%m-%d')
	daysFmt = DateFormatter('%y-%m-%d %H:%M')
	
	for siteid,data_dict in site_data_dict.iteritems():
		aDTn = data_dict['aDTn'].flatten()
		for param, data in data_dict.iteritems():
			if param=='aDTn':
				continue
			pylab.plot(aDTn,data.flatten(),label=siteid+' '+param)
			print siteid, param, data[data==data].mean()
	ax1.xaxis.set_major_locator(locatorM)
	ax1.xaxis.set_major_formatter(daysFmt)
	pylab.legend()
	pylab.show()
		
	print "DONE."
	exit()
		
