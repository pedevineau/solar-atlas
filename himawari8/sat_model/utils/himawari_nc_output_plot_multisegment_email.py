#! /usr/bin/env python
# visualisation of processing segments with basemap
#
#
import datetime
import matplotlib
import os

import numpy
from general_utils import basic_mail
from general_utils import daytimeconv
from general_utils import latlon
matplotlib.use('Agg')

from himawari8.sat_model.utils import himawari_nc_latlontools

#------------------------------------------------------------
# main
#------------------------------------------------------------
if __name__ == "__main__":
	himawari_slot_min = 1
	himawari_slot_max = 144


	dfb_begin = daytimeconv.yyyymmdd2dfb('20160301')
	dfb_end = daytimeconv.yyyymmdd2dfb('20160303')

#	dfb_today = daytimeconv.date2dfb(datetime.datetime.now().date())
#	dfb_begin = dfb_today
#	dfb_end = dfb_today

	slot_begin, slot_end = 1, 144
	roll_slots=+30 #  

	
	segmentsDict={\
					'jp':latlon.expand_segments([[63,63,10,11]]),\
				  }

	#EMAIL subject	
	if dfb_begin == dfb_end:
		subject_date=daytimeconv.dfb2yyyymmdd(dfb_begin)
	else:
		subject_date=daytimeconv.dfb2yyyymmdd(dfb_begin)+'-'+daytimeconv.dfb2yyyymmdd(dfb_end)
	subject = 'HIMAWARI8 operational processing for %s' % (subject_date)  # '' automatic subject from script name


	nc_var_name='GHI'
	data_path_pool=["/home1/model_data_himawari/data_output/v20/" ]


	vmin=0; vmax=None
	resolution=2./60.
	file_time_segmentation = "month"


	to=['tomas.cebecauer@geomodel.eu']
	slot_to_mail = 20
	msg=''
	
	do_slot=False
	do_all_slots=False
	do_day=True

	#-----------------
	attachments=[]

	keys = segmentsDict.keys()
	for key in  keys:
		segments_to_calculate = segmentsDict[key]

		#read data	
		data_total, bbox = himawari_nc_latlontools.read_multisegment_data(dfb_begin, dfb_end, slot_begin, slot_end, roll_slots, himawari_slot_min, himawari_slot_max, segments_to_calculate, nc_var_name, data_path_pool, resolution, file_time_segmentation=file_time_segmentation)

		
		print 'prepare email attachments'
		data_total=numpy.ma.masked_where(numpy.isnan(data_total),data_total)
	
		if do_slot:
			slot_idx = slot_to_mail - slot_begin
			data=data_total[:,slot_idx,:,:].mean(axis=0)
			temp_file_slot='/tmp/%s_%s_%s_%s_%d.png'%(key,nc_var_name, daytimeconv.dfb2yyyymmdd(dfb_begin), daytimeconv.dfb2yyyymmdd(dfb_end), slot_to_mail)
			latlon.visualize_map_2d(data, bbox, vmin, vmax, interpolation='nearest', openfile=temp_file_slot)
			attachments.append(temp_file_slot)
		
		if do_all_slots:
			data=data_total[:,:,:,:].mean(axis=0)
			temp_file_day_slots='/tmp/%s_%s_%s_%s_%d_%d.png'%(key,nc_var_name, daytimeconv.dfb2yyyymmdd(dfb_begin), daytimeconv.dfb2yyyymmdd(dfb_end), slot_begin, slot_end)
			latlon.visualize_map_3d_subplots(data, bbox, vmin, vmax, interpolation='nearest', img_width=12, img_height=10, subplot_rows=7, subplot_cols=5, openfile=temp_file_day_slots)
			attachments.append(temp_file_day_slots)
			
		if do_day:
			data=data_total[:,:,:,:].mean(axis=0).sum(axis=0)*10./60.
			temp_file_day='/tmp/%s_%s_%s_%s.png'%(key,nc_var_name, daytimeconv.dfb2yyyymmdd(dfb_begin), daytimeconv.dfb2yyyymmdd(dfb_end))
			latlon.visualize_map_2d(data, bbox, vmin, vmax, interpolation='nearest', openfile=temp_file_day)
			attachments.append(temp_file_day)
	

	print 'send mail'
	basic_mail.mail_process_message_ssl(reciever_to=to, message=msg, subject=subject, attachments=attachments)

	
	for temp_file in attachments:
		os.remove(temp_file)

	print 'DONE'