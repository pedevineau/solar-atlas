#! /usr/bin/env python
# point value from satelite model to db
#
# last revision: 18/02/2010
#
import StringIO
import datetime
import os
import sys

import numpy
import psycopg2
from general_utils import daytimeconv
from general_utils import db_sat_model
from general_utils import db_sites
from general_utils import db_utils
from general_utils import latlon
from himawari8.sat_model.utils import himawari_nc_latlontools



#drop existing data from output table
def table_drop_data(dsn, out_table, site_id, minslot, maxslot, minD, maxD):
	try:
		conn = psycopg2.connect(dsn)
	except:
		print "Unable to connect to the database, exiting."
		return False
	curs = conn.cursor()
	query = "DELETE FROM " + out_table + " WHERE siteid=" + str(site_id) + " AND date>='" + str(minD) + "' AND date<='" + str(maxD) + "' AND slot>=" + str(minslot) + " AND slot<=" + str(maxslot)
	try:
		curs.execute(query)
	except:
		print "Unable to execute the query, skipping."
		print query

	conn.commit()
	conn.close()
	return True


#write output data to output table
#def write_out_data(dsn, output_table, asite, avalidData, outValidData, aRealtimeValidData, inData_col_dict, outData_col_dict, vacuum=False, verbose=False):
def write_out_data(dsn, output_table, siteid, lon, lat, dfb_begin, dfb_end, slot_begin, slot_end, data_dict, vacuum=False, verbose=True):
	if verbose: print "DB write", datetime.datetime.now()

	aT_offset=datetime.timedelta(hours=(5./60.)+0.000001)

	#check whether table holds some data for given site. If not then directly insert all data, othervise try to update
	#tab_has_site_data = check_site_in_out_table(dsn, output_table, asite)
	#best performance is acheived by removing data and INSERTing new ones
	minD = daytimeconv.dfb2date(dfb_begin)
	maxD = daytimeconv.dfb2date(dfb_end)
	table_drop_data(dsn, output_table, siteid, slot_begin, slot_end, minD, maxD)

	#prepare db cursor
	try:
		conn = psycopg2.connect(dsn)
	except:
		print "Unable to connect to the database, exiting."
		return False
	curs = conn.cursor()
	data_keys=data_dict.keys()
	
	channel2dbcol_dict={"GHI":"ghi","GHIc": "ghi_c", "DNI":"dni","DNIc":"dni_c","KTM":"ktm","CI":"ci","LB":"lb","LBclass":"lbclass","LBland":"lbland","CI_flag":"ci_flag"}
	for akey in data_keys:
		if akey not in  channel2dbcol_dict.keys():
			del(data_dict[akey])
	
	outflds = ['siteid', 'date', 'time', 'datetime', 'slot']
	data_keys=data_dict.keys()
	for akey in data_keys:
		outflds.append( channel2dbcol_dict[akey])
			
	buffer=StringIO.StringIO()
	
	inserts = 0
	for dfb_idx in range(0, dfb_end-dfb_begin+1):
		aD=daytimeconv.dfb2date(dfb_idx + dfb_begin)
		for slot_idx in range(0, slot_end-slot_begin+1):
			slot=slot_idx+slot_begin

			aT = daytimeconv.dh2time((slot-1.)/6.)
			
			aDTreal = datetime.datetime.combine(aD, aT) + aT_offset
			
#			outflds = 'siteid, date, time, datetime, slot'
			outvals =str(siteid) + "\t'" + aDTreal.strftime("%Y-%m-%d") + "'\t'" + aDTreal.strftime("%H:%M:") + str(aDTreal.second) + "'\t'" + aDTreal.strftime("%Y-%m-%d %H:%M:") + str(aDTreal.second)+ "'\t" + str(slot)
			do_insert=True
			for akey in data_keys:
				val = data_dict[akey][dfb_idx, slot_idx]
				if numpy.isnan(val):
					do_insert=False
					continue
				if (akey == 'CI_flag') or (akey == 'LBclass'):
					val = int(val)
				outvals += "\t" +str(val)
#				print aDTreal, akey, val
			if do_insert:
				inserts+=1
				buffer.write(outvals+'\n')	

	buffer.seek(0,0)
				
	try:
		curs.copy_from(buffer, "\""+output_table+"\"", sep='\t', null='\N', columns=outflds)
	except:
		print sys.exc_info()
		print "Unable to execute copy to db"
	conn.commit()
	conn.close()
	buffer.close()

	if vacuum:
		if verbose: print 'Vacuum', datetime.datetime.now()
		db_utils.db_vacuum_table(dsn, atable=output_table)
		#indx=output_table+"_date_idx"
		#db_cluster_table(dsn,output_table,indx)

	if verbose: print  "inserts:" + str(inserts)


#------------------------------------------------------------
# main
#------------------------------------------------------------
if __name__ == "__main__":
	DSN_HIMAWARI = "dbname=himawari_sites host=dbdata user=gm_user password=ibVal4"
	DSN_SITES = "dbname=site_coordinates host=dbdata user=gm_user password=ibVal4"


	dfb_begin = daytimeconv.yyyymmdd2dfb('20150801')
	dfb_end = daytimeconv.yyyymmdd2dfb('20160331')
#	dfb_end = daytimeconv.yyyymmdd2dfb('19991231')

	slot_begin, slot_end = 1, 144
	
	interpolation='nearest' #bilinear or nearest
	

	sites=[ 2006811 ]
	
	site_table="site_coordinates"

	
	output_table_suffix = "v20a_rast"
	output_table_descr = "sat model v20 - output for report sites from raster model"
#	output_table_suffix = "v19_raster"
#	output_table_descr = "sat model v1.9 - output for report sites from raster model"
#	dsn = "dbname=msg_sites host=triton user=thebe password=bI12op"

	dsn_data = DSN_HIMAWARI
	sat_suff=""

#	nc_var_names=['GHI', 'DNI', 'GHIc', 'DNIc', 'CI_flag', 'CI', 'KTM']
#	output_fields=["ghi", "ghi_c", "dni", "dni_c", "ktm", "ci", "ci_flag", "lbclass", "lb", "lbland"]
	nc_var_names=['GHI', 'DNI', 'GHIc', 'DNIc', 'KTM']
	output_fields=["ghi", "dni", "ghi_c", "dni_c", "ktm"]
	nc_var_names=['GHI', 'DNI', 'GHIc', 'DNIc', 'CI_flag']
	output_fields=["ghi", "dni", "ghi_c", "dni_c", "ci_flag"]

	
	
	
	model_out_dir_pool=[ '/home1/model_data_himawari/data_output/v20/']
	


	#######################################################################
	# processing
	#######################################################################
	
	for siteid in sites:
		lon, lat, alt = db_sites.db_get_site_lonlatalt(siteid, DSN_SITES, site_table)
		print "Processing site", siteid, lon, lat, alt
		
		output_table = "res_model_%d_%s" % (siteid, output_table_suffix)

		out_table_exists=db_utils.db_dtable_exist(dsn_data,output_table)
		if not out_table_exists:
			result = db_sat_model.out_tab_init(dsn_data, output_table, output_table_descr, output_fields)
			if not result:
				print "cannot check/create output table, exit"
				exit()
	
	
		#remove indexes
		db_utils.db_remove_indexes(dsn_data, db_utils.db_table_get_indexes(dsn_data, output_table))

		
		if (lon is None):
			print "   Cannot read site coordinates, skipping site"
			continue


		seg_col, seg_row = latlon.get_5x5_seg(lon, lat)
		seg_suffix="_c%d_r%d" % (seg_col, seg_row)
		
		print 'reading site %d from c%d, r%d' % (siteid, seg_col, seg_row)
		print lon, lat
		#suffix added to output NETCDF file names
		if sat_suff=='':
			outdata_suffix=seg_suffix
		else:
			outdata_suffix="_%s%s" % (sat_suff, seg_suffix)

		
		seg_bbox=latlon.get_5x5_seg_bbox(arow=seg_row, acol=seg_col, resolution=2./60.)

		#find output dir in model_out_dir_pool
		for out_data_path in model_out_dir_pool:
				#output data
			outdata_path_dict={"LB": out_data_path, "LBclass": out_data_path, "LBland": out_data_path,\
							 "CI": out_data_path, "KTM": out_data_path, "GHIc": out_data_path, "GHI": out_data_path,\
							"DNIc": out_data_path, "DNI": out_data_path, "GHIcor": out_data_path, "DNIcor": out_data_path, \
							"CI_flag": out_data_path}
			out_files_dict = himawari_nc_latlontools.outdata_existingncfile_dict(dfb_begin, dfb_end, nc_var_names, outdata_path_dict, outdata_suffix, file_time_segmentation='month')
			if len(out_files_dict.keys()) > 0:
				print "data from segment found in ", out_data_path
				break

		if len(out_files_dict.keys())<1:
			print 'No NC files found'
			continue

		print 'Model outputs read:'
		data_dict={}
		for nc_var_name in nc_var_names:
			print nc_var_name,
			#To read data from raw netcdf model output
			rast_data=himawari_nc_latlontools.outdata_nc_read_point(nc_var_name, out_files_dict, outdata_suffix, dfb_begin, dfb_end, slot_begin, slot_end, lon, lat, interpolation=interpolation, file_time_segmentation='month')
			data_dict[nc_var_name] = rast_data
		print ''

		print 'Model outputs write DB', output_table
		write_out_data(dsn_data, output_table, siteid, lon, lat, dfb_begin, dfb_end, slot_begin, slot_end, data_dict, vacuum=False, verbose=True)

		#recreate indexes
		print 'DB rebuilding output table indexes', datetime.datetime.now()
		indexes={}
		for fld in ("date", "datetime"):
			db_indx=output_table + "_"+fld+"_idx"
			db_query = "CREATE INDEX " + db_indx + " ON " + output_table + " USING hash ("+fld+")"
			indexes[db_indx] = db_query
		db_utils.db_create_indexes(dsn_data, indexes)
	
	
	print "DONE."
	exit()
		
