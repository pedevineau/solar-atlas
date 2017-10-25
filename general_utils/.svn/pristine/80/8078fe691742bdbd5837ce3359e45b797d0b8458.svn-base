#! /usr/bin/env python
'''
Created on Aug 17, 2009

@author: tomas
'''
import os
import sys
import math
import numpy
import datetime
import netCDF4

from general_utils import latlon
from general_utils import daytimeconv

from general_utils.basic_logger import make_logger, ObjectWithLogger, inject_logger, with_logger
logger = make_logger(__name__)
import logging
logging.getLogger().setLevel(logging.WARNING)






#create new NetCDF file with [month, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels = [var_name,var_type,var_description,var_units,-9999., ('month','row','col'),[1,256,256]]  OR
#img_channels = [var_name,var_type,var_description,var_units,-9999., ('month','row','col'),[1,256,256],3]

def latlon_make_month_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[], nc_extent=None ,compression=True,dims_name_colrow=True, skip_dimension_check=False):
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'
	


	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
#	for chan_name, chan_type, description, units, noval, dims, chunksizes in img_channels:
	for img_ch in img_channels:
		chan_type = img_ch[1]
		chan_name = img_ch[0]
		if not (chan_type in datatypes):
			logger.warning("dat_type  of chan %s is not one of predefined NetCDF data types",chan_name)
			return False
		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height

	if not skip_dimension_check:
		if (xmin < -180) or (xmin > 180):
			logger.warning("xmin outside of range")
			return False
		if (xmax < -180) or (xmax > 180):
			logger.warning("xmax outside of range")
			return False
		if (ymin < -90) or (ymin > 90):
			logger.warning("ymin outside of range")
			return False
		if (ymax < -90) or (ymax > 90):
			logger.warning("ymax outside of range")
			return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('month', 12)
	except:
		logger.error("unable to create NetCDF dimension: %s",'month')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable month
	try:
		month_var=rootgrp.createVariable('month','i1',('month',))
	except:
		logger.error("unable to create NetCDF variable: %s",'month')
		rootgrp.close()
		return False
	#and attributes
	try:
		month_var.units="month"
		month_var.valid_range=numpy.array([1,12],dtype='int8')
	except:
		logger.error("unable to add %s variable attributes",'month')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	months=numpy.arange(1,12+1,1,dtype='int8')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		month_var[:]=months
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
	for img_ch in img_channels:
		if len(img_ch) == 7:
			img_channel, chan_type, description, units, noval, dims, chunksizes = img_ch
			least_significant_digit = None
		elif len(img_ch) == 8:
			img_channel, chan_type, description, units, noval, dims, chunksizes, least_significant_digit = img_ch
			
		try:
			if (chunksizes is None):
				if least_significant_digit is None:
					img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression)
				else:
					img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, least_significant_digit=least_significant_digit)
			else:
				if least_significant_digit is None:
					img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksizes)
				else:
					img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksizes, least_significant_digit=least_significant_digit)
					
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
	logger.info("nc create file OK")
	return True



#create new NetCDF file with [month, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_month_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,256,256]], least_significant_digits=[], nc_extent=None ,compression=True, dims_name_colrow=True):
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'
	

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.debug("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.debug("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False
		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	
	
	if type(nc_extent) == type(latlon.uneven_bounding_box()):	
		xresolution=nc_extent.xresolution
		yresolution=nc_extent.yresolution
	else:
		xresolution=nc_extent.resolution
		yresolution=nc_extent.resolution
	
#	print nc_file_name
#	print xresolution, yresolution
	
	img_width=nc_extent.width
	img_height=nc_extent.height
	if (xmin < -360) or (xmin > 360):
		logger.warning("xmin %f outside of range",xmin)
		return False
	if (xmax < -360) or (xmax > 360):
		logger.warning("xmax %f outside of range",xmax)
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90-yresolution) or (ymin > 90+yresolution):
		logger.warning("ymin %f outside of range",ymin)
		return False
	if (ymax < -90-yresolution) or (ymax > 90+yresolution):
		logger.warning("ymax %f outside of range",ymax)
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('month', 12)
	except:
		logger.error("unable to create NetCDF dimension: %s",'month')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable month
	try:
		month_var=rootgrp.createVariable('month','i1',('month',))
	except:
		logger.error("unable to create NetCDF variable: %s",'month')
		rootgrp.close()
		return False
	#and attributes
	try:
		month_var.units="month"
		month_var.valid_range=numpy.array([1,12],dtype='int8')
	except:
		logger.error("unable to add %s variable attributes",'month')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(xresolution/2),xmax,xresolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(yresolution/2),ymin,-yresolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	months=numpy.arange(1,12+1,1,dtype='int8')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		month_var[:]=months
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['month', row_name, col_name]
		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]

#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, least_significant_digit=ls_digit)
			else:
				chsize=chunksize[:]
				chsize[0] = min(chunksize[0],12)
				chsize[1] = min(chunksize[1],img_height)
				chsize[2] = min(chunksize[2],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chsize, least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
			if ls_digit is not None:
				img_var.least_significant_digit=ls_digit
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
	logger.info("nc create file OK")
	return True


#create new NetCDF file with [dfb, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_dfb_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,256,256]], dfb_begin=None, dfb_end=None, nc_extent=None ,compression=True):
	if (dfb_begin is None) or (dfb_end is None) or (dfb_end<dfb_begin):
		logger.error('cannot create output file. dfb values problem')
		return False
	
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False
		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	if (xmin < -180) or (xmin > 180):
		logger.warning("xmin outside of range")
		return False
	if (xmax < -180) or (xmax > 180):
		logger.warning("xmax outside of range")
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin outside of range")
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax outside of range")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension('row', img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",'row')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('col', img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",'col')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('dfb',  dfb_end-dfb_begin+1)
	except:
		logger.error("unable to create NetCDF dimension: %s",'dfb')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable('row','f8',('row',))
	except:
		logger.error("unable to create NetCDF variable: %s",'row')
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",'row')
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable('col','f8',('col',))
	except:
		logger.error("unable to create NetCDF variable: %s",'col')
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",'col')
		rootgrp.close()
		return False
	
	# coordinate variable dfb
	try:
		dfb_var=rootgrp.createVariable('dfb','i2',('dfb',))
	except:
		logger.error("unable to create NetCDF variable: %s",'dfb')
		rootgrp.close()
		return False
	#and attributes
	try:
		dfb_var.units="dfb"
		dfb_var.valid_range=numpy.array([dfb_begin,dfb_end],dtype='int16')
		dfb_var.long_name='days since 1980-01-01'
	except:
		logger.error("unable to add %s variable attributes",'dfb')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	dfbs=numpy.arange(dfb_begin,dfb_end+1,1,dtype='int16')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		dfb_var[:]=dfbs
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['dfb', 'row', 'col']
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression)
			else:
				chunksize[0] = min(chunksize[0],12)
				chunksize[1] = min(chunksize[1],img_height)
				chunksize[2] = min(chunksize[2],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
	logger.info("nc create file OK")
	return True



#create new NetCDF file with [dfb, slot, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_dfb_slot_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,1,256,256]], least_significant_digits=[None], tslot_min=10, tslot_max=80, dfb_begin=None, dfb_end=None, nc_extent=None ,compression=True, dims_name_colrow=True, SlotDescription = "scan of slot 1 starts at 00:00, slot 2 at 00:15, ..."):
	TIMESLOT_MIN=1
	TIMESLOT_MAX=144

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'

	if (dfb_begin is None) or (dfb_end is None) or (dfb_end<dfb_begin):
		logger.error('cannot create output file. dfb values problem')
		return False
	
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
#			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False

	if ((type(tslot_min) != int)) or tslot_min<TIMESLOT_MIN:
		logger.warning("tslot_min not integer or lower then min %d",TIMESLOT_MIN)
		return False
	if (type(tslot_max) != int) or tslot_max>TIMESLOT_MAX:
		logger.warning("tslot_max not integer or bigger then max %d",TIMESLOT_MAX)
		return False
	if (tslot_max < tslot_min):
		logger.warning("tslot_max < tslot_min")
		return False

		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	tslot_count = (tslot_max -tslot_min)+1

	if (xmin < -360) or (xmin > 360):
		logger.warning("xmin outside of range")
		return False
	if (xmax < -360) or (xmax > 360):
		logger.warning("xmax outside of range")
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin outside of range")
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax outside of range")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False

	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False

	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('slot', tslot_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'slot')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('dfb', dfb_end-dfb_begin+1)
	except:
		logger.error("unable to create NetCDF dimension: %s",'dfb')
		rootgrp.close()
		return False

	# coordinate variable row 
	try:

		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8', (col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable slot
	try:

		slot_var=rootgrp.createVariable('slot',datatypes_dict['NC_SHORT'],('slot',))
	except:
		logger.error("unable to create NetCDF variable: %s",'slot')
		rootgrp.close()
		return False
	#and attributes
	try:
		slot_var.units="slot number"
		slot_var.valid_range=numpy.array([tslot_min,tslot_max],dtype='int16')
		slot_var.long_name="time slot number"
		slot_var.description=SlotDescription
	except:
		logger.error("unable to add %s variable attributes",'slot')
		rootgrp.close()
		return False
	
	# coordinate variable dfb
	try:

		dfb_var=rootgrp.createVariable('dfb','i2',('dfb',))
	except:
		logger.error("unable to create NetCDF variable: %s",'dfb')
		rootgrp.close()
		return False
	#and attributes
	try:
		dfb_var.units="dfb"
		dfb_var.valid_range=numpy.array([dfb_begin,dfb_end],dtype='int16')
		dfb_var.long_name='days since 1980-01-01'
	except:
		logger.error("unable to add %s variable attributes",'dfb')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	slots=numpy.arange(tslot_min,tslot_max+1,1,dtype='uint8')
	dfbs=numpy.arange(dfb_begin,dfb_end+1,1,dtype='int16')

	try:

		row_var[:]=rows
		col_var[:]=cols
		slot_var[:]=slots
		dfb_var[:]=dfbs
	except Exception as e:
		print e, e.__class__
		logger.error("unable to add variable values")
		rootgrp.close()
		return False


	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['dfb', 'slot', row_name, col_name]
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]

		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression,least_significant_digit=ls_digit)
			else:
				chunksize[0] = min(chunksize[0],dfb_end-dfb_begin+1)
				chunksize[1] = min(chunksize[1],tslot_count)
				chunksize[2] = min(chunksize[2],img_height)
				chunksize[3] = min(chunksize[3],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize,least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			print sys.exc_info()
			print img_channel, datatypes_dict[chan_type], dims, noval, compression, chunksize
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
			if ls_digit is not None:
				img_var.least_significant_digit=ls_digit
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
			
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
#	logger.info("nc create file OK")
	return True

#create new NetCDF file with [dfb, rslot, fslot, lat, lon] dimensions
#used to store forecasts rslot - reference slot of forecast start, fslot - forecast slot
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_dfb_rslot_fslot_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,1,256,256]], least_significant_digits=[None], rslot_min=10, rslot_max=80, fslot_min=10, fslot_max=80, dfb_begin=None, dfb_end=None, nc_extent=None ,compression=True, SlotDescription = "scan of slot 1 starts at 00:00, slot 2 at 00:15, ..."):
	TIMESLOT_MIN=1
	TIMESLOT_MAX=144

	col_name='longitude'
	row_name='latitude'

	if (dfb_begin is None) or (dfb_end is None) or (dfb_end<dfb_begin):
		logger.error('cannot create output file. dfb values problem')
		return False
	
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
#			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False

	if ((type(rslot_min) != int)) or rslot_min<TIMESLOT_MIN:
		logger.warning("rslot_min not integer or lower then min %d",TIMESLOT_MIN)
		return False
	if ((type(fslot_min) != int)) or fslot_min<TIMESLOT_MIN:
		logger.warning("fslot_min not integer or lower then min %d",TIMESLOT_MIN)
		return False
	if (type(rslot_max) != int) or rslot_max>TIMESLOT_MAX:
		logger.warning("rslot_max not integer or bigger then max %d",TIMESLOT_MAX)
		return False
	if (rslot_max < rslot_min):
		logger.warning("rslot_max < rslot_min")
		return False
	if (fslot_max < fslot_min):
		logger.warning("fslot_max < fslot_min")
		return False

		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	rslot_count = (rslot_max -rslot_min)+1
	fslot_count = (fslot_max -fslot_min)+1

	if (xmin < -360) or (xmin > 360):
		logger.warning("xmin outside of range")
		return False
	if (xmax < -360) or (xmax > 360):
		logger.warning("xmax outside of range")
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin outside of range")
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax outside of range")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False

	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False

	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('rslot', rslot_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'rslot')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('fslot', fslot_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'fslot')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('dfb', dfb_end-dfb_begin+1)
	except:
		logger.error("unable to create NetCDF dimension: %s",'dfb')
		rootgrp.close()
		return False

	# coordinate variable row 
	try:

		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8', (col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable rslot
	try:

		rslot_var=rootgrp.createVariable('rslot',datatypes_dict['NC_SHORT'],('rslot',))
	except:
		logger.error("unable to create NetCDF variable: %s",'rslot')
		rootgrp.close()
		return False
	#and attributes
	try:
		rslot_var.units="slot number"
		rslot_var.valid_range=numpy.array([rslot_min,rslot_max],dtype='int16')
		rslot_var.long_name="reference time slot number (start of forecast)"
		rslot_var.description=SlotDescription
	except:
		logger.error("unable to add %s variable attributes",'rslot')
		rootgrp.close()
		return False
	
	# coordinate variable fslot
	try:

		fslot_var=rootgrp.createVariable('fslot',datatypes_dict['NC_SHORT'],('fslot',))
	except:
		logger.error("unable to create NetCDF variable: %s",'fslot')
		rootgrp.close()
		return False
	#and attributes
	try:
		fslot_var.units="slot number"
		fslot_var.valid_range=numpy.array([fslot_min,fslot_max],dtype='int16')
		fslot_var.long_name="forecast time slot number"
		fslot_var.description=SlotDescription
	except:
		logger.error("unable to add %s variable attributes",'fslot')
		rootgrp.close()
		return False
	
	# coordinate variable dfb
	try:

		dfb_var=rootgrp.createVariable('dfb','i2',('dfb',))
	except:
		logger.error("unable to create NetCDF variable: %s",'dfb')
		rootgrp.close()
		return False
	#and attributes
	try:
		dfb_var.units="dfb"
		dfb_var.valid_range=numpy.array([dfb_begin,dfb_end],dtype='int16')
		dfb_var.long_name='days since 1980-01-01'
	except:
		logger.error("unable to add %s variable attributes",'dfb')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	rslots=numpy.arange(rslot_min,rslot_max+1,1,dtype='uint8')
	fslots=numpy.arange(fslot_min,fslot_max+1,1,dtype='uint8')
	dfbs=numpy.arange(dfb_begin,dfb_end+1,1,dtype='int16')

	try:
		row_var[:]=rows
		col_var[:]=cols
		rslot_var[:]=rslots
		fslot_var[:]=fslots
		dfb_var[:]=dfbs
	except Exception as e:
		print e, e.__class__
		logger.error("unable to add variable values")
		rootgrp.close()
		return False


	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['dfb', 'rslot', 'fslot', row_name, col_name]
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]

		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression,least_significant_digit=ls_digit)
			else:
				chunksize[0] = min(chunksize[0],dfb_end-dfb_begin+1)
				chunksize[1] = min(chunksize[1],rslot_count)
				chunksize[2] = min(chunksize[2],fslot_count)
				chunksize[3] = min(chunksize[3],img_height)
				chunksize[4] = min(chunksize[4],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize,least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			print sys.exc_info()
			print img_channel, datatypes_dict[chan_type], dims, noval, compression, chunksize
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
			if ls_digit is not None:
				img_var.least_significant_digit=ls_digit
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
			
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
#	logger.info("nc create file OK")
	return True



#create new NetCDF file with [dfb, slot, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_dfb_hour_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,1,256,256]], least_significant_digits=[None], hour_min=1, hour_max=24, dfb_begin=None, dfb_end=None, nc_extent=None ,compression=True, dims_name_colrow=True):

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'
	
	if (dfb_begin is None) or (dfb_end is None) or (dfb_end<dfb_begin):
		logger.error('cannot create output file. dfb values problem')
		return False
	
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False

	if ((type(hour_min) != int)) or hour_min<1:
		logger.warning("hour_min not integer or lower then min %d",1)
		return False
	if (type(hour_max) != int) or hour_max>24:
		logger.warning("hour_max not integer or bigger then max %d",24)
		return False
	if (hour_max < hour_min):
		logger.warning("hour_max < hour_min")
		return False


		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	hour_count = (hour_max -hour_min)+1

	if (xmin < -180) or (xmin > 180):
		logger.warning("xmin outside of range")
		return False
	if (xmax < -180) or (xmax > 180):
		logger.warning("xmax outside of range")
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin outside of range")
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax outside of range")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('hour', hour_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'hour')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('dfb', dfb_end-dfb_begin+1)
	except:
		logger.error("unable to create NetCDF dimension: %s",'dfb')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable hour
	try:
		hour_var=rootgrp.createVariable('hour',datatypes_dict['NC_SHORT'],('hour',))
	except:
		logger.error("unable to create NetCDF variable: %s",'hour')
		rootgrp.close()
		return False
	#and attributes
	try:
		hour_var.units="hour"
		hour_var.valid_range=numpy.array([hour_min,hour_max],dtype='int16')
		hour_var.long_name="hour"
		hour_var.description="end of integration interval"
	except:
		logger.error("unable to add %s variable attributes",'hour')
		rootgrp.close()
		return False
	
	# coordinate variable dfb
	try:
		dfb_var=rootgrp.createVariable('dfb','i2',('dfb',))
	except:
		logger.error("unable to create NetCDF variable: %s",'dfb')
		rootgrp.close()
		return False
	#and attributes
	try:
		dfb_var.units="dfb"
		dfb_var.valid_range=numpy.array([dfb_begin,dfb_end],dtype='int16')
		dfb_var.long_name='days since 1980-01-01'
	except:
		logger.error("unable to add %s variable attributes",'dfb')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	hours=numpy.arange(hour_min,hour_max+1,1,dtype='int8')
	dfbs=numpy.arange(dfb_begin,dfb_end+1,1,dtype='int16')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		hour_var[:]=hours
		dfb_var[:]=dfbs
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['dfb', 'hour', row_name, col_name]
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]

		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression,least_significant_digit=ls_digit)
			else:
				chunksize[0] = min(chunksize[0],16)
				chunksize[1] = min(chunksize[1],hour_count)
				chunksize[2] = min(chunksize[2],img_height)
				chunksize[3] = min(chunksize[3],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize,least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			print sys.exc_info()
			print img_channel, datatypes_dict[chan_type], dims, noval, compression, chunksize
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
			if ls_digit is not None:
				img_var.least_significant_digit=ls_digit
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
#	logger.info("nc create file OK")
	return True



#create new NetCDF file with [dfb, slot, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_doy_slot_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,256,256]], least_significant_digits=[None], tslot_min=10, tslot_max=80, doy_begin=None, doy_end=None, nc_extent=None ,compression=True, dims_name_colrow=False):
	TIMESLOT_MIN=1
	TIMESLOT_MAX=96

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'
	
	if (doy_begin is None) or (doy_end is None) or (doy_end<doy_begin):
		logger.error('cannot create output file. doy values problem')
		return False
	
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False

	if ((type(tslot_min) != int)) or tslot_min<TIMESLOT_MIN:
		logger.warning("tslot_min not integer or lower then min %d",TIMESLOT_MIN)
		return False
	if (type(tslot_max) != int) or tslot_max>TIMESLOT_MAX:
		logger.warning("tslot_max not integer or bigger then max %d",TIMESLOT_MAX)
		return False
	if (tslot_max < tslot_min):
		logger.warning("tslot_max < tslot_min")
		return False

	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	tslot_count = (tslot_max -tslot_min)+1

	if (xmin < -180) or (xmin > 180):
		logger.warning("xmin outside of range")
		return False
	if (xmax < -180) or (xmax > 180):
		logger.warning("xmax outside of range")
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin outside of range")
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax outside of range")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('slot', tslot_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'slot')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('doy', doy_end-doy_begin+1)
	except:
		logger.error("unable to create NetCDF dimension: %s",'doy')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable slot
	try:
		slot_var=rootgrp.createVariable('slot',datatypes_dict['NC_SHORT'],('slot',))
	except:
		logger.error("unable to create NetCDF variable: %s",'slot')
		rootgrp.close()
		return False
	#and attributes
	try:
		slot_var.units="slot number"
		slot_var.valid_range=numpy.array([tslot_min,tslot_max],dtype='int16')
		slot_var.long_name="time slot number"
		slot_var.description="scan of slot 1 starts at 00:00, slot 2 at 00:15, ..."
	except:
		logger.error("unable to add %s variable attributes",'slot')
		rootgrp.close()
		return False
	
	# coordinate variable dfb
	try:
		doy_var=rootgrp.createVariable('doy','i2',('doy',))
	except:
		logger.error("unable to create NetCDF variable: %s",'doy')
		rootgrp.close()
		return False
	#and attributes
	try:
		doy_var.units="doy"
		doy_var.valid_range=numpy.array([doy_begin,doy_end],dtype='int16')
		doy_var.long_name='day of the year'
	except:
		logger.error("unable to add %s variable attributes",'doy')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	slots=numpy.arange(tslot_min,tslot_max+1,1,dtype='int8')
	doys=numpy.arange(doy_begin,doy_end+1,1,dtype='int16')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		slot_var[:]=slots
		doy_var[:]=doys
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['doy', 'slot', row_name, col_name]
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]
		
		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression,least_significant_digit=ls_digit)
			else:
				chunksize[0] = min(chunksize[0],16)
				chunksize[1] = min(chunksize[1],tslot_count)
				chunksize[2] = min(chunksize[2],img_height)
				chunksize[3] = min(chunksize[3],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize,least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			print sys.exc_info()
			print img_channel, datatypes_dict[chan_type], dims, noval, compression, chunksize,ls_digit
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
			if ls_digit is not None:
				img_var.least_significant_digit=ls_digit
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
#	logger.info("nc create file OK")
	return True


#create new NetCDF file with [doy, hour, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_doy_hour_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,1,256,256]], least_significant_digits=[None], hour_min=1, hour_max=24, doy_begin=None, doy_end=None, nc_extent=None ,compression=True, dims_name_colrow=True):

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'
	
	if (doy_begin is None) or (doy_end is None) or (doy_end<doy_begin):
		logger.error('cannot create output file. doy values problem')
		return False
	
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False

	if ((type(hour_min) != int)) or hour_min<1:
		logger.warning("hour_min not integer or lower then min %d",1)
		return False
	if (type(hour_max) != int) or hour_max>24:
		logger.warning("hour_max not integer or bigger then max %d",24)
		return False
	if (hour_max < hour_min):
		logger.warning("hour_max < hour_min")
		return False


		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	hour_count = (hour_max -hour_min)+1

	if (xmin < -180) or (xmin > 180):
		logger.warning("xmin outside of range")
		return False
	if (xmax < -180) or (xmax > 180):
		logger.warning("xmax outside of range")
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin outside of range")
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax outside of range")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('hour', hour_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'hour')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('doy', doy_end-doy_begin+1)
	except:
		logger.error("unable to create NetCDF dimension: %s",'doy')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable hour
	try:
		hour_var=rootgrp.createVariable('hour',datatypes_dict['NC_SHORT'],('hour',))
	except:
		logger.error("unable to create NetCDF variable: %s",'hour')
		rootgrp.close()
		return False
	#and attributes
	try:
		hour_var.units="hour"
		hour_var.valid_range=numpy.array([hour_min,hour_max],dtype='int16')
		hour_var.long_name="hour"
		hour_var.description="end of integration interval"
	except:
		logger.error("unable to add %s variable attributes",'hour')
		rootgrp.close()
		return False
	
	# coordinate variable dfb
	try:
		doy_var=rootgrp.createVariable('doy','i2',('doy',))
	except:
		logger.error("unable to create NetCDF variable: %s",'doy')
		rootgrp.close()
		return False
	#and attributes
	try:
		doy_var.units="doy"
		doy_var.valid_range=numpy.array([doy_begin,doy_end],dtype='int16')
		doy_var.long_name='Day of year'
	except:
		logger.error("unable to add %s variable attributes",'doy')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	hours=numpy.arange(hour_min,hour_max+1,1,dtype='int8')
	doys=numpy.arange(doy_begin,doy_end+1,1,dtype='int16')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		hour_var[:]=hours
		doy_var[:]=doys
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['doy', 'hour', row_name, col_name]
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]

		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression,least_significant_digit=ls_digit)
			else:
				chunksize[0] = min(chunksize[0],16)
				chunksize[1] = min(chunksize[1],hour_count)
				chunksize[2] = min(chunksize[2],img_height)
				chunksize[3] = min(chunksize[3],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize,least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			print sys.exc_info()
			print img_channel, datatypes_dict[chan_type], dims, noval, compression, chunksize
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
			if ls_digit is not None:
				img_var.least_significant_digit=ls_digit
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
#	logger.info("nc create file OK")
	return True



#create new NetCDF file with [month, slot, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_month_slot_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,256,256]], least_significant_digits=[], tslot_min=10, tslot_max=80, nc_extent=None ,compression=True, isMSG=True, dims_name_colrow=True, slot_represents_center=False, slot_time_step_min=None):

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'
	
	TIMESLOT_MIN=1
	
	if slot_time_step_min is None:
		logger.warning("Depricated use of isMSG for slots count in latlon_make_params_month_slot_lat_lon_nc function. Use slot_time_step_min instead.")
		if isMSG:
			TIMESLOT_MAX=96
		else:
			TIMESLOT_MAX=48
	else:
		TIMESLOT_MAX = int(24*60/slot_time_step_min)

	
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.debug("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.debug("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False

	if (type(tslot_min) != int) or tslot_min<TIMESLOT_MIN:
		logger.warning("tslot_min not integer or lower then min %d",TIMESLOT_MIN)
		return False
	if (type(tslot_max) != int) or tslot_max>TIMESLOT_MAX:
		logger.warning("tslot_max not integer or bigger then max %d",TIMESLOT_MAX)
		return False
	if (tslot_max < tslot_min):
		logger.warning("tslot_max < tslot_min")
		return False
		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	tslot_count = (tslot_max -tslot_min)+1

	if (xmin < -360) or (xmin > 360):
		logger.warning("xmin %f outside of range",xmin)
		return False
	if (xmax < -360) or (xmax > 360):
		logger.warning("xmax %f outside of range",xmax)
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin %f outside of range",ymin)
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax %f outside of range",ymax)
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('slot', tslot_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'slot')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('month', 12)
	except:
		logger.error("unable to create NetCDF dimension: %s",'month')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable slot
	try:
		slot_var=rootgrp.createVariable('slot',datatypes_dict['NC_SHORT'],('slot',))
	except:
		logger.error("unable to create NetCDF variable: %s",'slot')
		rootgrp.close()
		return False
	#and attributes
	try:
		slot_var.units="slot number"
		slot_var.valid_range=numpy.array([tslot_min,tslot_max],dtype='int16')
		slot_var.long_name="time slot number"
		if isMSG:
			if slot_represents_center:
				slot_var.description="slot 1 represents at 00:07:30, slot 2 at 00:22:30, ..."
			else:
				slot_var.description="scan of slot 1 starts at 00:00, slot 2 at 00:15, ..."
	except:
		logger.error("unable to add %s variable attributes",'slot')
		rootgrp.close()
		return False
	
	# coordinate variable month
	try:
		month_var=rootgrp.createVariable('month','i1',('month',))
	except:
		logger.error("unable to create NetCDF variable: %s",'month')
		rootgrp.close()
		return False
	#and attributes
	try:
		month_var.units="month"
		month_var.valid_range=numpy.array([1,12],dtype='int8')
	except:
		logger.error("unable to add %s variable attributes",'month')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	slots=numpy.arange(tslot_min,tslot_max+1,1,dtype='int8')
	months=numpy.arange(1,12+1,1,dtype='int8')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		slot_var[:]=slots
		month_var[:]=months
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['month', 'slot', row_name, col_name]
		
		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]

#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, least_significant_digit=ls_digit)
			else:
				chunksize[0] = min(chunksize[0],12)
				chunksize[1] = min(chunksize[1],tslot_count)
				chunksize[2] = min(chunksize[2],img_height)
				chunksize[3] = min(chunksize[3],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize, least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
#	logger.info("nc create file OK")
	return True



#create new NetCDF file with [month, slot, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_slot_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,256,256]], least_significant_digits=[], tslot_min=10, tslot_max=80, nc_extent=None ,compression=True, isMSG=True, dims_name_colrow=True, slot_represents_center=False, slot_time_step_min=None):

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'
	
	TIMESLOT_MIN=1
	
	if slot_time_step_min is None:
		logger.warning("Depricated use of isMSG for slots count in latlon_make_params_month_slot_lat_lon_nc function. Use slot_time_step_min instead.")
		if isMSG:
			TIMESLOT_MAX=96
		else:
			TIMESLOT_MAX=48
	else:
		TIMESLOT_MAX = int(24*60/slot_time_step_min)

	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False

	if (type(tslot_min) != int) or tslot_min<TIMESLOT_MIN:
		logger.warning("tslot_min not integer or lower then min %d",TIMESLOT_MIN)
		return False
	if (type(tslot_max) != int) or tslot_max>TIMESLOT_MAX:
		logger.warning("tslot_max not integer or bigger then max %d",TIMESLOT_MAX)
		return False
	if (tslot_max < tslot_min):
		logger.warning("tslot_max < tslot_min")
		return False
		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	tslot_count = (tslot_max -tslot_min)+1

	if (xmin < -180) or (xmin > 180):
		logger.warning("xmin outside of range")
		return False
	if (xmax < -180) or (xmax > 180):
		logger.warning("xmax outside of range")
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin outside of range")
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax outside of range")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('slot', tslot_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'slot')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable slot
	try:
		slot_var=rootgrp.createVariable('slot',datatypes_dict['NC_SHORT'],('slot',))
	except:
		logger.error("unable to create NetCDF variable: %s",'slot')
		rootgrp.close()
		return False
	#and attributes
	try:
		slot_var.units="slot number"
		slot_var.valid_range=numpy.array([tslot_min,tslot_max],dtype='int16')
		slot_var.long_name="time slot number"
		if isMSG:
			if slot_represents_center:
				slot_var.description="slot 1 represents at 00:07:30, slot 2 at 00:22:30, ..."
			else:
				slot_var.description="scan of slot 1 starts at 00:00, slot 2 at 00:15, ..."
	except:
		logger.error("unable to add %s variable attributes",'slot')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	slots=numpy.arange(tslot_min,tslot_max+1,1,dtype='int8')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		slot_var[:]=slots
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['slot', row_name, col_name]
		
		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]

#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, least_significant_digit=ls_digit)
			else:
				chunksize[0] = min(chunksize[0],tslot_count)
				chunksize[1] = min(chunksize[1],img_height)
				chunksize[2] = min(chunksize[2],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize, least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
#	logger.info("nc create file OK")
	return True


#create new NetCDF file with [dfb, time, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_dfb_time_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,1,256,256]], least_significant_digits=[None], hour_min=0, hour_max=24,hour_step= None, dfb_begin=None, dfb_end=None, nc_extent=None ,compression=True, dims_name_colrow=True,slot_dim=False):

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'
	
	if (dfb_begin is None) or (dfb_end is None) or (dfb_end<dfb_begin) or (hour_step is None):
		logger.error('cannot create output file. dfb values problem')
		return False
	
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False

	if hour_min<0:
		logger.warning("hour_min lower then min %d",1)
		return False
	if (hour_max < hour_min):
		logger.warning("hour_max < hour_min")
		return False

	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	hour_count = (hour_max -hour_min)/hour_step+1

	if (xmin < -180) or (xmin > 180):
		logger.warning("xmin outside of range")
		return False
	if (xmax < -180) or (xmax > 180):
		logger.warning("xmax outside of range")
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin outside of range")
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax outside of range")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	if slot_dim:
		try:
			dummy=rootgrp.createDimension('slot', hour_count)
		except:
			logger.error("unable to create NetCDF dimension: %s",'slot')
			rootgrp.close()
			return False
	else:
		try:
			dummy=rootgrp.createDimension('hour', hour_count)
		except:
			logger.error("unable to create NetCDF dimension: %s",'hour')
			rootgrp.close()
			return False
	try:
		dummy=rootgrp.createDimension('dfb', dfb_end-dfb_begin+1)
	except:
		logger.error("unable to create NetCDF dimension: %s",'dfb')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable hour
	if slot_dim:
		try:
			hour_var=rootgrp.createVariable('slot',datatypes_dict['NC_SHORT'],('slot',))
		except:
			logger.error("unable to create NetCDF variable: %s",'slot')
			rootgrp.close()
			return False
	else:
		try:
			hour_var=rootgrp.createVariable('hour',datatypes_dict['NC_FLOAT'],('hour',))
		except:
			logger.error("unable to create NetCDF variable: %s",'hour')
			rootgrp.close()
			return False
	#and attributes
	if slot_dim:
		try:
			hour_var.units="slot"
			hour_var.valid_range=numpy.array([hour_min,hour_max],dtype='int16')
			hour_var.long_name="slot"
			hour_var.description="time slot"
		except:
			logger.error("unable to add %s variable attributes",'slot')
			rootgrp.close()
			return False
	else:
		try:
			hour_var.units="time"
			hour_var.valid_range=numpy.array([hour_min,hour_max],dtype='float')
			hour_var.long_name="time"
			hour_var.description="center of interval"
		except:
			logger.error("unable to add %s variable attributes",'time')
			rootgrp.close()
			return False
	
	# coordinate variable dfb
	try:
		dfb_var=rootgrp.createVariable('dfb','i2',('dfb',))
	except:
		logger.error("unable to create NetCDF variable: %s",'dfb')
		rootgrp.close()
		return False
	#and attributes
	try:
		dfb_var.units="dfb"
		dfb_var.valid_range=numpy.array([dfb_begin,dfb_end],dtype='int16')
		dfb_var.long_name='days since 1980-01-01'
	except:
		logger.error("unable to add %s variable attributes",'dfb')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	
	if slot_dim:
		hours=numpy.arange(1,hour_max+1,hour_step,dtype='int16')
	else:
		hours=numpy.arange(hour_min+hour_step/2.,hour_max+hour_step,hour_step)
	
	dfbs=numpy.arange(dfb_begin,dfb_end+1,1,dtype='int16')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		hour_var[:]=hours
		dfb_var[:]=dfbs
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		if slot_dim:
			dims=['dfb', 'slot', row_name, col_name]
		else:
			dims=['dfb', 'hour', row_name, col_name]
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]

		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression,least_significant_digit=ls_digit)
			else:
				chunksize[0] = min(chunksize[0],16)
				chunksize[1] = min(chunksize[1],hour_count)
				chunksize[2] = min(chunksize[2],img_height)
				chunksize[3] = min(chunksize[3],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize,least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			print sys.exc_info()
			print img_channel, datatypes_dict[chan_type], dims, noval, compression, chunksize
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
			if ls_digit is not None:
				img_var.least_significant_digit=ls_digit
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
#	logger.info("nc create file OK")
	return True


def latlon_make_params_month_hour_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,256,256]], hour_min=0, hour_max=23, nc_extent=None ,compression=True, least_significant_digit=None):
	MONTH_NOVALUE=-9999
	HOUR_MIN=0
	HOUR_MAX=23

	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False

	if (type(hour_min) != int) or hour_min<HOUR_MIN:
		logger.warning("hour_min not integer or lower then min %d",HOUR_MIN)
		return False
	if (type(hour_max) != int) or hour_max>HOUR_MAX:
		logger.warning("hour_max not integer or bigger then max %d",HOUR_MAX)
		return False
	if (hour_max < hour_min):
		logger.warning("hour_max < hour_min")
		return False


		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	hour_count = (hour_max -hour_min)+1

	if (xmin < -180) or (xmin > 180):
		logger.warning("xmin outside of range")
		return False
	if (xmax < -180) or (xmax > 180):
		logger.warning("xmax outside of range")
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin outside of range")
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax outside of range")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		row_dim=rootgrp.createDimension('latitude', img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",'latitude')
		rootgrp.close()
		return False
	try:
		col_dim=rootgrp.createDimension('longitude', img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",'longitude')
		rootgrp.close()
		return False
	try:
		hour_dim=rootgrp.createDimension('hour', hour_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'hour')
		rootgrp.close()
		return False
	try:
		month_dim=rootgrp.createDimension('month', 12)
	except:
		logger.error("unable to create NetCDF dimension: %s",'month')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable('latitude','f8',('latitude',))
	except:
		logger.error("unable to create NetCDF variable: %s",'latitude')
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",'latitude')
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable('longitude','f8',('longitude',))
	except:
		logger.error("unable to create NetCDF variable: %s",'longitude')
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",'longitude')
		rootgrp.close()
		return False
	
	# coordinate variable hour
	try:
		hour_var=rootgrp.createVariable('hour',datatypes_dict['NC_SHORT'],('hour',))
	except:
		logger.error("unable to create NetCDF variable: %s",'hour')
		rootgrp.close()
		return False
	#and attributes
	try:
		hour_var.units="hour UTC"
		hour_var.valid_range=numpy.array([hour_min,hour_max],dtype='int16')
		hour_var.long_name="hour in UTC"
		hour_var.description="hour"
	except:
		logger.error("unable to add %s variable attributes",'hour')
		rootgrp.close()
		return False
	
	# coordinate variable month
	try:
		month_var=rootgrp.createVariable('month','i1',('month',))
	except:
		logger.error("unable to create NetCDF variable: %s",'month')
		rootgrp.close()
		return False
	#and attributes
	try:
		month_var.units="month"
		month_var.valid_range=numpy.array([1,12],dtype='int8')
	except:
		logger.error("unable to add %s variable attributes",'month')
		rootgrp.close()
		return False

	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	hours=numpy.arange(hour_min,hour_max+1,1,dtype='int8')
	months=numpy.arange(1,12+1,1,dtype='int8')
	
	try:
		row_var[:]=rows
		col_var[:]=cols
		hour_var[:]=hours
		month_var[:]=months
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['month', 'hour', 'longitude', 'latitude']
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, least_significant_digit=least_significant_digit)
			else:
				chunksize[0] = min(chunksize[0],12)
				chunksize[1] = min(chunksize[1],hour_count)
				chunksize[2] = min(chunksize[2],img_height)
				chunksize[3] = min(chunksize[3],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize, least_significant_digit=least_significant_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
#	logger.info("nc create file OK")
	return True






#create new NetCDF file with [year, month, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels - channels (data variables) names []
#img_types - data variables types []
#img_units - data variables units []
#img_long_names - data variables long description []
#novals - data variables novalues []
#chunksizes - data variables chunksizes
def latlon_make_params_year_month_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[''], img_types=["NC_FLOAT"], img_units=[''],img_long_names=[''], novals=[-9999], chunksizes=[[1,256,256]], least_significant_digits=[], year_min=1994, year_max=2011, nc_extent=None ,compression=True, dims_name_colrow=True):

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'

	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.debug("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.debug("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
	for chan_type in img_types:
		if not (chan_type in datatypes):
			logger.warning("data_type %s is not one of predefined NetCDF data types", chan_type)
			return False
		
	
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution

#	print nc_file_name
#	print resolution


	year_count = int(year_max - year_min)+1
	img_width=nc_extent.width
	img_height=nc_extent.height
	if (xmin < -360) or (xmin > 360):
		logger.warning("xmin %f outside of range",xmin)
		return False
	if (xmax < -360) or (xmax > 360):
		logger.warning("xmax %f outside of range",xmax)
		return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin < -90) or (ymin > 90):
		logger.warning("ymin %f outside of range",ymin)
		return False
	if (ymax < -90) or (ymax > 90):
		logger.warning("ymax %f  outside of range",ymax)
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		dummy=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",row_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",col_name)
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('month', 12)
	except:
		logger.error("unable to create NetCDF dimension: %s",'month')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('year', year_count)
	except:
		logger.error("unable to create NetCDF dimension: %s",'year')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	# coordinate variable month
	try:
		month_var=rootgrp.createVariable('month','i1',('month',))
	except:
		logger.error("unable to create NetCDF variable: %s",'month')
		rootgrp.close()
		return False
	#and attributes
	try:
		month_var.units="month"
		month_var.valid_range=numpy.array([1,12],dtype='int8')
	except:
		logger.error("unable to add %s variable attributes",'month')
		rootgrp.close()
		return False

	# coordinate variable year 
	try:
		year_var=rootgrp.createVariable('year',datatypes_dict['NC_SHORT'],('year',))
	except:
		logger.error("unable to create NetCDF variable: %s",'year')
		rootgrp.close()
		return False
	#and attributes
	try:
		year_var.units="year"
		year_var.valid_range=numpy.array([year_min,year_max],dtype='int16')
		year_var.long_name="year"
	except:
		logger.error("unable to add %s variable attributes",'year')
		rootgrp.close()
		return False


	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	months=numpy.arange(1,12+1,1,dtype='int8')
	years=numpy.arange(year_min,year_max+1)

	try:
		row_var[:]=rows
		col_var[:]=cols
		month_var[:]=months
		year_var[:]=years
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	
	# define data variables
	for indx in range(0,len(img_channels)):
		img_channel=img_channels[indx]
		chan_type=img_types[indx]
		units=img_units[indx]
		description=img_long_names[indx]
		noval=novals[indx]	
		chunksize=chunksizes[indx]
		dims=['year' ,'month', row_name, col_name]

		ls_digit=None
		if chan_type in ["NC_FLOAT", "NC_DOUBLE"]: 
			if len(least_significant_digits) == len(img_channels):
				ls_digit=least_significant_digits[indx]

#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
		try:
			if (chunksize is None):
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, least_significant_digit=ls_digit)
			else:
				chunksize[0] = min(chunksize[0],year_count)
				chunksize[1] = min(chunksize[1],12)
				chunksize[2] = min(chunksize[2],img_height)
				chunksize[3] = min(chunksize[3],img_width)
				img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksize, least_significant_digit=ls_digit)
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)

	rootgrp.close()
	logger.info("nc create file OK")
	return True




#crate new NetCDF file with [month, lat, lon] dimensions
#metadata = [[name, value],[],[] ...] - whole netcdf metadata e.g. ['projection',"geographic coordinates"]
#img_channels = [var_name,var_type,var_description,var_units,-9999., ('month','row','col'),[1,256,256]]

def latlon_make_lat_lon_nc(nc_file_name='', metadata=[], force_overwrite=True, img_channels=[], nc_extent=None ,compression=True, dims_name_colrow=False, skip_dimension_check=False):
	datatypes = ["NC_BYTE", "NC_SHORT", "NC_INT", "NC_FLOAT", "NC_DOUBLE", "NC_UBYTE", "NC_USHORT", "NC_UINT", "NC_INT64", "NC_UINT64"]
	
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}

	if dims_name_colrow:
		col_name='col'
		row_name='row'
	else:
		col_name='longitude'
		row_name='latitude'
	
	if (nc_file_name is None) or (nc_file_name==''):
		logger.warning("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite is set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True
	
#	for chan_name, chan_type, description, units, noval, dims, chunksizes in img_channels:
	for img_ch in img_channels:
		chan_name=img_ch[0]
		chan_type=img_ch[1]
		if not (chan_type in datatypes):
			logger.warning("dat_type  of chan %s is not one of predefined NetCDF data types",chan_name)
			return False
		
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height
	if not skip_dimension_check:
		if (xmin < -180) or (xmin > 180):
			logger.warning("xmin outside of range")
			return False
		if (xmax < -180) or (xmax > 180):
			logger.warning("xmax outside of range")
			return False
		if (ymin < -90) or (ymin > 90):
			logger.warning("ymin outside of range")
			return False
		if (ymax < -90) or (ymax > 90):
			logger.warning("ymax outside of range")
			return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False

	# define dimensions
	try:
		row_dim=rootgrp.createDimension(row_name, img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s",'row')
		rootgrp.close()
		return False
	try:
		col_dim=rootgrp.createDimension(col_name, img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s",'col')
		rootgrp.close()
		return False
	
	# coordinate variable row 
	try:
		row_var=rootgrp.createVariable(row_name,'f8',(row_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",row_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		row_var.units="degrees north"
		
		row_var.valid_range=numpy.array([ymin,ymax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",row_name)
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		col_var=rootgrp.createVariable(col_name,'f8',(col_name,))
	except:
		logger.error("unable to create NetCDF variable: %s",col_name)
		rootgrp.close()
		return False
	#and attributes
	try:
		col_var.units="degrees east"
		col_var.valid_range=numpy.array([xmin,xmax],dtype='float64')
	except:
		logger.error("unable to add %s variable attributes",col_name)
		rootgrp.close()
		return False
	
	#create coordinate variables values
	cols=numpy.arange(xmin+(resolution/2),xmax,resolution,dtype='float64')
	cols=numpy.around(cols,decimals=6)
	rows=numpy.arange(ymax-(resolution/2),ymin,-resolution,dtype='float64')
	rows=numpy.around(rows,decimals=6)
	
	try:
		row_var[:]=rows
		col_var[:]=cols
	except:
		logger.error("unable to add variable values")
		rootgrp.close()
		return False
	

	
	# define data variables
#	for img_channel, chan_type, description, units, noval, dims, chunksizes in img_channels:
	for img_ch in img_channels:
		if len(img_ch) == 7:
			img_channel, chan_type, description, units, noval, dims, chunksizes = img_ch
			least_significant_digit = None
		elif len(img_ch) == 8:
			img_channel, chan_type, description, units, noval, dims, chunksizes, least_significant_digit = img_ch
#		print img_channel, chan_type, description, units, noval, dims, chunksizes, least_significant_digit
		dims = [row_name, col_name]
		if (len(dims) >2):
			logger.error("variables can have at max 2 dimensions. Skipping variable")
			continue
		
		try:
			if (chunksizes is None):
				if least_significant_digit is None:
					img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression)
				else:
					img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, least_significant_digit=least_significant_digit)
			else:
				if chunksizes[0] > img_height: chunksizes[0]=img_height
				if chunksizes[1] > img_width: chunksizes[1]=img_width
				if least_significant_digit is None:
					img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksizes)
				else:
					img_var=rootgrp.createVariable(img_channel, datatypes_dict[chan_type], dims, fill_value=noval, zlib=compression, contiguous=False, chunksizes=chunksizes, least_significant_digit=least_significant_digit)
					
		except:
			logger.error("unable to create NetCDF variable: %s",img_channel)
			rootgrp.close()
			return False
		#and attributes
		
		try:
			if not(units is None) and (units != ''):
				img_var.units=units
			if not(description is None) and (description != ''):
				img_var.description=description
			img_var.missing_value=numpy.array(noval,dtype=datatypes_dict[chan_type])
			img_var.compression=str(compression)
		except:
			logger.error("unable to add %s variable attributes",img_channel)
			rootgrp.close()
			return False
	
	#global attributes
	for item,value in metadata:
		rootgrp.__setattr__(item,value)
	rootgrp.close()
	logger.info("nc create file OK")
	return True


def latlon_update_history_attribute(nc_file_name,history_string=''):
	if not(os.access(nc_file_name, os.F_OK)):
		logger.warning("file %s not found",nc_file_name)
		return False
	
	if len(history_string) <1:
		logger.warning("empty history string")
		return False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r+')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		return False

	item=u'history'
	ncattrs=rootgrp.ncattrs()
	if item in ncattrs:
		histarrt = rootgrp.getncattr(item)
		history_string = [histarrt,history_string]
		rootgrp.__delattr__(item)
	rootgrp.setncattr(item,history_string)
	rootgrp.close()
	logger.info("nc update attribute OK")
	return True


#writes data to the file
def latlon_write_month_lat_lon_nc(nc_file_name, varname, month, data):

	if (varname is None) or (varname==''):
		logger.warning("empty variable name")
		return False
	if not(os.access(nc_file_name, os.F_OK)):
		logger.warning("file %s not found",nc_file_name)
		return False
	
	month_idx=month-1
	if (month_idx < 0) or (month_idx > 11):
		logger.warning("wrong day index for %s file",nc_file_name)
		return False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r+')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		return False

	ncvariabs=rootgrp.variables
	
	if len(ncvariabs)<1:
		logger.error("no variables found in NetCDF file: %s",nc_file_name)
		rootgrp.close()
		return False
	
	if not(varname in ncvariabs.keys()):
		logger.error("variable %s not found in NetCDF  %s",varname ,nc_file_name)
		rootgrp.close()
		return False
	
	ncvarimg=ncvariabs[varname]
	ncvarimg.set_auto_maskandscale(False)
	ncvarimg[month_idx,:,:]=data
	
	rootgrp.close()
	return True


def latlon_write_months_lat_lon_nc(nc_file_name, varname, data):

	if (varname is None) or (varname==''):
		logger.warning("empty variable name")
		return False
	if not(os.access(nc_file_name, os.F_OK)):
		logger.warning("file %s not found",nc_file_name)
		return False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r+')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		return False

	ncvariabs=rootgrp.variables
	
	if len(ncvariabs)<1:
		logger.error("no variables found in NetCDF file: %s",nc_file_name)
		rootgrp.close()
		return False
	
	if not(varname in ncvariabs.keys()):
		logger.error("variable %s not found in NetCDF  %s",varname ,nc_file_name)
		rootgrp.close()
		return False
	
	ncvarimg=ncvariabs[varname]
	ncvarimg.set_auto_maskandscale(False)
	ncvarimg[:,:,:]=data
	
	rootgrp.close()
	return True


#writes data to the file
def latlon_write_lat_lon_nc(nc_file_name, varname, data):
	if (varname is None) or (varname==''):
		logger.warning("empty variable name")
		return False
	if not(os.access(nc_file_name, os.F_OK)):
		logger.warning("file %s not found",nc_file_name)
		return False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r+')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		return False

	ncvariabs=rootgrp.variables
	
	if len(ncvariabs)<1:
		logger.error("no variables found in NetCDF file: %s",nc_file_name)
		rootgrp.close()
		return False
	
	if not(varname in ncvariabs.keys()):
		logger.error("variable %s not found in NetCDF  %s",varname ,nc_file_name)
		rootgrp.close()
		return False
	
	ncvarimg=ncvariabs[varname]
	ncvarimg.set_auto_maskandscale(False)
	ncvarimg[:,:]=data
	
	rootgrp.close()
	return True



def latlon_write_doy_slot_lat_lon_nc(ncfile_out, chan, data, Doys, Slots):		
	#open NC for writting
	rootgrpout = netCDF4.Dataset(ncfile_out, 'r+')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'doy' in varDictout.keys():
		doy_var_name = 'doy'
	else:
		logger.error('Variable doy not found in nc %s . Skipping' % (ncfile_out))
		rootgrpout.close()
		return False
	doy_var_out=varDictout[doy_var_name]

	if 'slot' in varDictout.keys():
		slot_var_name = 'slot'
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile_out))
		rootgrpout.close()
		return False
	slot_var_out=varDictout[slot_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile_out))
		rootgrpout.close()
		return False
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile_out))
		rootgrpout.close()
		return False
	cols = len(dimDictout[col_name])
	rows = len(dimDictout[row_name])
	
	if (data.shape[2] != rows) or (data.shape[3] != cols):
		logger.error('Rows/cols size of data to write to nc %s differs from nc dimensions. Skipping' % (ncfile_out))
		rootgrpout.close()
		return False


	#recalculate in and out dfb indexes					
	nc_doy_min, nc_doy_max = doy_var_out.valid_range
	doy_min_out=max(nc_doy_min, Doys[0])
	doy_max_out=min(nc_doy_max, Doys[1])
	data_doy_min_idx=doy_min_out-Doys[0]
	data_doy_max_idx=data_doy_min_idx+doy_max_out-doy_min_out
	nc_doy_min_idx=doy_min_out-nc_doy_min
	nc_doy_max_idx=nc_doy_min_idx+doy_max_out-doy_min_out
	
	
	#recalculate in and out slot indexes					
	nc_slot_min, nc_slot_max = slot_var_out.valid_range
	slot_min_out=max(nc_slot_min, Slots[0])
	slot_max_out=min(nc_slot_max, Slots[1])
	data_slot_min_idx=slot_min_out-Slots[0]
	data_slot_max_idx=data_slot_min_idx+slot_max_out-slot_min_out
	nc_slot_min_idx=slot_min_out-nc_slot_min
	nc_slot_max_idx=nc_slot_min_idx+slot_max_out-slot_min_out
	
	#vrite data
	data_var_out[nc_doy_min_idx:nc_doy_max_idx+1, nc_slot_min_idx:nc_slot_max_idx+1,:,:]=data[data_doy_min_idx:data_doy_max_idx+1, data_slot_min_idx:data_slot_max_idx+1,:,:]
	try:
		data_var_out[nc_doy_min_idx:nc_doy_max_idx+1, nc_slot_min_idx:nc_slot_max_idx+1,:,:]=data[data_doy_min_idx:data_doy_max_idx+1, data_slot_min_idx:data_slot_max_idx+1,:,:]
	except:
		print sys.exc_info()
		logger.error('Problem writing data to nc %s. Skipping' % (ncfile_out))
		rootgrpout.close()
		return False
		
	doys = numpy.arange(doy_min_out,doy_max_out+1)
	
	try:
		doy_var_out[nc_doy_min_idx:nc_doy_max_idx+1]=doys
	except:
		print sys.exc_info()
		logger.error('Problem writing doys to nc %s. Skipping' % (ncfile_out))
		rootgrpout.close()
		return False
	
	rootgrpout.close()
	return True




def latlon_write_dfb_slot_lat_lon_nc(ncfile, chan, Dfbs, Slots, bbox, data):		

	if (bbox is None):
		logger.error("no or None bbox to read : %s",ncfile)
		return False
	
#	dfbs=Dfbs[1]-Dfbs[0]+1
#	slots=Slots[1]-Slots[0]+1
#	latlon_rows=bbox.height
#	latlon_cols=bbox.width
	
	#open NC for reading

	rootgrpout = netCDF4.Dataset(ncfile, 'r+')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'dfb' in varDictout.keys():
		dfb_var_name = 'dfb'
	elif 'day' in varDictout.keys():
		dfb_var_name = 'day'
	else:
		logger.error('Variable dfb not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	dfb_var_out=varDictout[dfb_var_name]

	if 'slot' in varDictout.keys():
		slot_var_name = 'slot'
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	slot_var_out=varDictout[slot_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return False
	data_var_out=varDictout[chan]
	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	#recalculate in and out dfb indexes
	#print Dfbs
	nc_dfb_min, nc_dfb_max = dfb_var_out.valid_range
	#print nc_dfb_min, nc_dfb_max
	dfb_min_out=max(nc_dfb_min, Dfbs[0])
	dfb_max_out=min(nc_dfb_max, Dfbs[1])
	#print dfb_min_out, dfb_max_out
	data_dfb_min_idx=dfb_min_out-Dfbs[0]
	data_dfb_max_idx=data_dfb_min_idx+dfb_max_out-dfb_min_out
	nc_dfb_min_idx=dfb_min_out-nc_dfb_min
	nc_dfb_max_idx=nc_dfb_min_idx+dfb_max_out-dfb_min_out
	#print nc_dfb_min_idx, nc_dfb_max_idx+1
	
	#recalculate in and out slot indexes					
#	nc_slot_min, nc_slot_max = slot_var_out.valid_range
	nc_slot_min, nc_slot_max = slot_var_out[:].min(), slot_var_out[:].max()
	slot_min_out=max(nc_slot_min, Slots[0])
	slot_max_out=min(nc_slot_max, Slots[1])
	data_slot_min_idx=slot_min_out-Slots[0]
	data_slot_max_idx=data_slot_min_idx+slot_max_out-slot_min_out
	nc_slot_min_idx=slot_min_out-nc_slot_min
	nc_slot_max_idx=nc_slot_min_idx+slot_max_out-slot_min_out



	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range

	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)

	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation
		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data[data_dfb_min_idx:data_dfb_max_idx+1,data_slot_min_idx:data_slot_max_idx+1,:,:]

	else:
		logger.warning("problems to write outputs - check dimensions and resolution")
		return False
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	
	rootgrpout.close()
	return True



def latlon_write_dfb_hour_lat_lon_nc(ncfile, chan, Dfbs, Hours, bbox, data):		
	
	if (bbox is None):
		logger.error("no or None bbox to read : %s",ncfile)
		return False
	
	#open NC for reading/writting
	rootgrpout = netCDF4.Dataset(ncfile, 'r+')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables

	if 'dfb' in varDictout.keys():
		dfb_var_name='dfb'
	elif 'day' in varDictout.keys():
		dfb_var_name = 'day'
	else:
		logger.error('Variable dfb or day not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	dfb_var_out=varDictout[dfb_var_name]

	if 'hour' in varDictout.keys():
		hour_var_name = 'hour'
	if 'time' in varDictout.keys():
		hour_var_name = 'time'
	else:
		logger.error('No variable for hour or time found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	hour_var_out=varDictout[hour_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return False
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	#recalculate in and out dfb indexes					
	nc_dfb_min, nc_dfb_max = dfb_var_out.valid_range
	dfb_min_out=max(nc_dfb_min, Dfbs[0])
	dfb_max_out=min(nc_dfb_max, Dfbs[1])
	data_dfb_min_idx=dfb_min_out-Dfbs[0]
	data_dfb_max_idx=data_dfb_min_idx+dfb_max_out-dfb_min_out
	nc_dfb_min_idx=dfb_min_out-nc_dfb_min
	nc_dfb_max_idx=nc_dfb_min_idx+dfb_max_out-dfb_min_out
	
	
	#recalculate in and out slot indexes					
	nc_hour_min, nc_hour_max = hour_var_out.valid_range
	hour_min_out=max(nc_hour_min, Hours[0])
	hour_max_out=min(nc_hour_max, Hours[1])
	data_hour_min_idx=hour_min_out-Hours[0]
	data_hour_max_idx=data_hour_min_idx+hour_max_out-hour_min_out
	nc_hour_min_idx=hour_min_out-nc_hour_min
	nc_hour_max_idx=nc_hour_min_idx+hour_max_out-hour_min_out
	
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation

		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_hour_min_idx:nc_hour_max_idx+1, px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data[data_dfb_min_idx:data_dfb_max_idx+1,data_hour_min_idx:data_hour_max_idx+1,:,:]	
	else:
		logger.warning("problems to write outputs - check dimensions and resolution")
		return False
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	
	rootgrpout.close()
	return True



def latlon_write_dfb_lat_lon_nc(ncfile, chan, Dfbs,  bbox, data):		
	
	if (bbox is None):
		logger.error("no or None bbox to read : %s",ncfile)
		return False
	
	#open NC for reading/writting
	rootgrpout = netCDF4.Dataset(ncfile, 'r+')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'dfb' in varDictout.keys():
		dfb_var_name = 'dfb'
	else:
		logger.error('Variable dfb not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	dfb_var_out=varDictout[dfb_var_name]

	if 'hour' in varDictout.keys():
		hour_var_name = 'hour'
	elif 'slot' in varDictout.keys():
		hour_var_name = 'slot'
	else:
		logger.error('No variable for hour found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	hour_var_out=varDictout[hour_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return False
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	#recalculate in and out dfb indexes					
	nc_dfb_min, nc_dfb_max = dfb_var_out.valid_range
	dfb_min_out=max(nc_dfb_min, Dfbs[0])
	dfb_max_out=min(nc_dfb_max, Dfbs[1])
	data_dfb_min_idx=dfb_min_out-Dfbs[0]
	data_dfb_max_idx=data_dfb_min_idx+dfb_max_out-dfb_min_out
	nc_dfb_min_idx=dfb_min_out-nc_dfb_min
	nc_dfb_max_idx=nc_dfb_min_idx+dfb_max_out-dfb_min_out
	
	
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation

		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,:, px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data[data_dfb_min_idx:data_dfb_max_idx+1,:,:,:]	
	else:
		logger.warning("problems to write outputs - check dimensions and resolution")
		return False
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	
	rootgrpout.close()
	return True





def latlon_write_month_hour_lat_lon_nc(ncfile, chan, Months, Hours, bbox, data):		
	
	if (bbox is None):
		logger.error("no or None bbox to read : %s",ncfile)
		return False
	
	#open NC for reading/writting
	rootgrpout = netCDF4.Dataset(ncfile, 'r+')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'month' in varDictout.keys():
		month_var_name = 'month'
	else:
		logger.error('Variable month not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	month_var_out=varDictout[month_var_name]

	if 'time' in varDictout.keys():
		time_var_name = 'time'
	elif 'hour' in varDictout.keys():
		time_var_name = 'hour'
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	time_var_out=varDictout[time_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return False
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	elif 'lon' in dimDictout.keys():
		col_name='lon'
		row_name='lat'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	#recalculate in and out month indexes					
	nc_month_min = month_var_out[:].min()
	nc_month_max = month_var_out[:].max()
	month_min_out=max(nc_month_min, Months[0])
	month_max_out=min(nc_month_max, Months[1])
	data_month_min_idx=month_min_out-Months[0]
	data_month_max_idx=data_month_min_idx+month_max_out-month_min_out
	nc_month_min_idx=month_min_out-nc_month_min
	nc_month_max_idx=nc_month_min_idx+month_max_out-month_min_out
	
	
	#recalculate in and out slot indexes					
	nc_hour_min = time_var_out[:].min()
	nc_hour_max = time_var_out[:].max()
	hour_min_out=max(nc_hour_min, Hours[0])
	hour_max_out=min(nc_hour_max, Hours[1])
	data_hour_min_idx=hour_min_out-Hours[0]
	data_hour_max_idx=data_hour_min_idx+hour_max_out-hour_min_out
	nc_hour_min_idx=hour_min_out-nc_hour_min
	nc_hour_max_idx=nc_hour_min_idx+hour_max_out-hour_min_out
	
		
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation

		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		data_var_out[nc_month_min_idx:nc_month_max_idx+1,nc_hour_min_idx:nc_hour_max_idx+1, px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data[data_month_min_idx:data_month_max_idx+1,data_hour_min_idx:data_hour_max_idx+1,:,:]	
	else:
		logger.warning("problems to write outputs - check dimensions and resolution")
		return False
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	
	rootgrpout.close()
	return True


def latlon_write_slot_lat_lon_nc(ncfile, chan, Slots, bbox, data):		
	
	if (bbox is None):
		logger.error("no or None bbox to read : %s",ncfile)
		return False
	
	#open NC for reading/writting
	rootgrpout = netCDF4.Dataset(ncfile, 'r+')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	

	if 'time' in varDictout.keys():
		time_var_name = 'time'
	elif 'slot' in varDictout.keys():
		time_var_name = 'slot'
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	time_var_out=varDictout[time_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return False
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	elif 'lon' in dimDictout.keys():
		col_name='lon'
		row_name='lat'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	#recalculate in and out slot indexes					
	nc_slot_min = time_var_out[:].min()
	nc_slot_max = time_var_out[:].max()
	slot_min_out=max(nc_slot_min, Slots[0])
	slot_max_out=min(nc_slot_max, Slots[1])
	data_slot_min_idx=slot_min_out-Slots[0]
	data_slot_max_idx=data_slot_min_idx+slot_max_out-slot_min_out
	nc_slot_min_idx=slot_min_out-nc_slot_min
	nc_slot_max_idx=nc_slot_min_idx+slot_max_out-slot_min_out
	
		
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation

		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		data_var_out[nc_slot_min_idx:nc_slot_max_idx+1, px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data[data_slot_min_idx:data_slot_max_idx+1,:,:]	
	else:
		logger.warning("problems to write outputs - check dimensions and resolution")
		return False
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	
	rootgrpout.close()
	return True





def latlon_write_month_slot_lat_lon_nc(ncfile, chan, Months, Slots, bbox, data):		
	
	if (bbox is None):
		logger.error("no or None bbox to read : %s",ncfile)
		return False
	
	#open NC for reading/writting
	rootgrpout = netCDF4.Dataset(ncfile, 'r+')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'month' in varDictout.keys():
		month_var_name = 'month'
	else:
		logger.error('Variable month not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	month_var_out=varDictout[month_var_name]

	if 'time' in varDictout.keys():
		time_var_name = 'time'
	elif 'slot' in varDictout.keys():
		time_var_name = 'slot'
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	time_var_out=varDictout[time_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return False
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	elif 'lon' in dimDictout.keys():
		col_name='lon'
		row_name='lat'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	#recalculate in and out month indexes					
	nc_month_min = month_var_out[:].min()
	nc_month_max = month_var_out[:].max()
	month_min_out=max(nc_month_min, Months[0])
	month_max_out=min(nc_month_max, Months[1])
	data_month_min_idx=month_min_out-Months[0]
	data_month_max_idx=data_month_min_idx+month_max_out-month_min_out
	nc_month_min_idx=month_min_out-nc_month_min
	nc_month_max_idx=nc_month_min_idx+month_max_out-month_min_out
	
	
	#recalculate in and out slot indexes					
	nc_slot_min = time_var_out[:].min()
	nc_slot_max = time_var_out[:].max()
	slot_min_out=max(nc_slot_min, Slots[0])
	slot_max_out=min(nc_slot_max, Slots[1])
	data_slot_min_idx=slot_min_out-Slots[0]
	data_slot_max_idx=data_slot_min_idx+slot_max_out-slot_min_out
	nc_slot_min_idx=slot_min_out-nc_slot_min
	nc_slot_max_idx=nc_slot_min_idx+slot_max_out-slot_min_out
	
		
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation

		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		data_var_out[nc_month_min_idx:nc_month_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data[data_month_min_idx:data_month_max_idx+1,data_slot_min_idx:data_slot_max_idx+1,:,:]	
	else:
		logger.warning("problems to write outputs - check dimensions and resolution")
		return False
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	
	rootgrpout.close()
	return True



def latlon_write_whole_lat_lon_nc_bbox(ncfile, chan, bbox, data):		
	
	if (bbox is None):
		logger.error("no or None bbox to read : %s",ncfile)
		return False
	
	#open NC for reading
	rootgrpout = netCDF4.Dataset(ncfile, 'r+')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return False
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return False
	
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()


	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation
		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)

		if data.ndim ==4:
			data_var_out[:,:, px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data[:,:,:,:]	
		elif data.ndim ==3:
			data_var_out[:, px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data[:,:,:]
		elif data.ndim ==2:
			data_var_out[px_ymin:px_ymax+1,px_xmin:px_xmax+1] = data[:,:]	
		else:
			logger.warning("problems to write outputs - check dimensions")
	else:
		logger.warning("problems to write outputs - check dimensions and resolution")
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	
	rootgrpout.close()
	return True



def latlon_write_months_hour_lat_lon_nc_point(nc_file_name, varname,input_data, col):
	
	if (nc_file_name is None) or (nc_file_name==''):
		logger.error( "No NC filename set")
		return None
	#print nc_file_name
	if not(os.access(nc_file_name,os.F_OK)):
#		logger.error("unable to find NetCDF file: %s",nc_file_name)
		print("unable to find NetCDF file: %s",nc_file_name)
		return None
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r+')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		
		return None
	ncvariabs=rootgrp.variables
	try:
		ncvar_data=ncvariabs[varname]
		ncvar_data.set_auto_maskandscale(False)
	except:
		logger.error("one of variables: %s, latitude, longitude, not found in %s" ,varname,  nc_file_name)
		
		rootgrp.close()
		return None
	ncvar_data[:,:,:, col]=input_data
	rootgrp.close()
	return True




def latlon_read_doy_slot_lat_lon_nc(ncfile, chan, Doys, Slots, bbox, interpolate='nearest'):		
	if (interpolate not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolate)
		return None
	
	if (bbox is None):
		logger.error("no or None bbox to read soecified: %s",interpolate)
		return None
	
	doys=Doys[1]-Doys[0]+1
	slots=Slots[1]-Slots[0]+1
	latlon_rows=bbox.height
	latlon_cols=bbox.width
	output_data = numpy.zeros((doys,slots,latlon_rows,latlon_cols), dtype=numpy.float32)
	output_data[:,:,:,:] = numpy.nan

	
	#open NC for reading
	try:
		rootgrpout = netCDF4.Dataset(ncfile, 'r')
	except:
		logger.error('Unable to open file %s. %s Skipping' % (ncfile, str(sys.exc_info())))
		return None
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'doy' in varDictout.keys():
		doy_var_name = 'doy'
	elif 'month' in varDictout.keys():
		doy_var_name = 'month'
	else:
		logger.error('Variable doy not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	doy_var_out=varDictout[doy_var_name]

	if 'slot' in varDictout.keys():
		slot_var_name = 'slot'
	
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	slot_var_out=varDictout[slot_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return None
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	#recalculate in and out dfb indexes					
	nc_doy_min, nc_doy_max = doy_var_out.valid_range
	doy_min_out=max(nc_doy_min, Doys[0])
	doy_max_out=min(nc_doy_max, Doys[1])
	data_doy_min_idx=doy_min_out-Doys[0]
	data_doy_max_idx=data_doy_min_idx+doy_max_out-doy_min_out
	nc_doy_min_idx=doy_min_out-nc_doy_min
	nc_doy_max_idx=nc_doy_min_idx+doy_max_out-doy_min_out
	
	
	#recalculate in and out slot indexes					
	nc_slot_min, nc_slot_max = slot_var_out.valid_range
	slot_min_out=max(nc_slot_min, Slots[0])
	slot_max_out=min(nc_slot_max, Slots[1])
	data_slot_min_idx=slot_min_out-Slots[0]
	data_slot_max_idx=data_slot_min_idx+slot_max_out-slot_min_out
	nc_slot_min_idx=slot_min_out-nc_slot_min
	nc_slot_max_idx=nc_slot_min_idx+slot_max_out-slot_min_out
	

	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation
		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		res=data_var_out[nc_doy_min_idx:nc_doy_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, px_ymin:px_ymax+1,px_xmin:px_xmax+1]
		output_data[data_doy_min_idx:data_doy_max_idx+1,data_slot_min_idx:data_slot_max_idx+1,:,:] = res	

	else:
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
		col_idxs1, row_idxs1, col_idxs2, row_idxs2, col_wghts, row_wghts = seg_ubbox.px_idx_grid_of_second_bbox(nc_ubbox)
		wh = (col_idxs1 != -9) & (col_idxs2 != -9) & (row_idxs1 != -9) & (row_idxs2 != -9)
		if interpolate=='nearest':
			wh = (col_idxs1 == col_idxs1) & (row_idxs1==row_idxs1) 
			col_min=col_idxs1[col_idxs1==col_idxs1].min()
			col_max=col_idxs1[col_idxs1==col_idxs1].max()
			row_min=row_idxs1[row_idxs1==row_idxs1].min()
			row_max=row_idxs1[row_idxs1==row_idxs1].max()
			data=data_var_out[nc_doy_min_idx:nc_doy_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, row_min:row_max+1,col_min:col_max+1]
			res=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
		else:
			wh = (col_idxs1 == col_idxs1) & (col_idxs2 == col_idxs2) & (row_idxs1==row_idxs1) & (row_idxs2==row_idxs2)
			col_min=min(col_idxs1[col_idxs1==col_idxs1].min(), col_idxs2[col_idxs2==col_idxs2].min())
			col_max=max(col_idxs1[col_idxs1==col_idxs1].max(), col_idxs2[col_idxs2==col_idxs2].max())
			row_min=min(row_idxs1[row_idxs1==row_idxs1].min(), row_idxs2[row_idxs2==row_idxs2].min())
			row_max=max(row_idxs1[row_idxs1==row_idxs1].max(), row_idxs2[row_idxs2==row_idxs2].max())
	
			data=data_var_out[nc_doy_min_idx:nc_doy_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1,row_min:row_max+1,col_min:col_max+1]
			
			out_data11=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
			out_data12=data[:,:,row_idxs1[wh]-row_min,col_idxs2[wh]-col_min]
			out_data21=data[:,:,row_idxs2[wh]-row_min,col_idxs1[wh]-col_min]
			out_data22=data[:,:,row_idxs2[wh]-row_min,col_idxs2[wh]-col_min]
	
			aux1 = out_data11 + row_wghts[wh]*(out_data21 - out_data11)
			aux2 = out_data12 + row_wghts[wh]*(out_data22 - out_data12)
			res=aux1 + col_wghts[wh]*(aux2 - aux1)

		output_data[data_doy_min_idx:data_doy_max_idx+1,data_slot_min_idx:data_slot_max_idx+1,wh] = res	
	
	rootgrpout.close()
	return output_data



def latlon_read_dfb_slot_lat_lon_nc(ncfile, chan, Dfbs, Slots, bbox, interpolate='nearest'):		
	if (interpolate not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolate)
		return None
	
	if (bbox is None):
		logger.error("no or None bbox to read : %s",ncfile)
		return None
	
	dfbs=Dfbs[1]-Dfbs[0]+1
	slots=Slots[1]-Slots[0]+1
	latlon_rows=bbox.height
	latlon_cols=bbox.width
	
	output_data = numpy.zeros((dfbs,slots,latlon_rows,latlon_cols), dtype=numpy.float32)
	output_data[:,:,:,:] = numpy.nan

	
	#open NC for reading
	try:
		rootgrpout = netCDF4.Dataset(ncfile, 'r')
	except:
		logger.error('Unable to open file %s. %s Skipping' % (ncfile, str(sys.exc_info())))
		return None
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'dfb' in varDictout.keys():
		dfb_var_name = 'dfb'
	elif 'month' in varDictout.keys():
		dfb_var_name = 'month'	
	elif 'day' in varDictout.keys():
		dfb_var_name = 'day'	
	else:
		logger.error('Variable dfb not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	dfb_var_out=varDictout[dfb_var_name]

	if 'slot' in varDictout.keys():
		slot_var_name = 'slot'
	elif 'time' in varDictout.keys():
		slot_var_name = 'time'
	elif 'hour' in varDictout.keys():
		slot_var_name = 'hour'
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	slot_var_out=varDictout[slot_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return None
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	#recalculate in and out dfb indexes					
	nc_dfb_min, nc_dfb_max = dfb_var_out.valid_range
	dfb_min_out=max(nc_dfb_min, Dfbs[0])
	dfb_max_out=min(nc_dfb_max, Dfbs[1])
	data_dfb_min_idx=dfb_min_out-Dfbs[0]
	data_dfb_max_idx=data_dfb_min_idx+dfb_max_out-dfb_min_out
	nc_dfb_min_idx=dfb_min_out-nc_dfb_min
	nc_dfb_max_idx=nc_dfb_min_idx+dfb_max_out-dfb_min_out
	
	
	#recalculate in and out slot indexes					
	nc_slot_min, nc_slot_max = slot_var_out.valid_range
	slot_min_out=max(nc_slot_min, Slots[0])
	slot_max_out=min(nc_slot_max, Slots[1])
	data_slot_min_idx=slot_min_out-Slots[0]
	data_slot_max_idx=data_slot_min_idx+slot_max_out-slot_min_out
	nc_slot_min_idx=slot_min_out-nc_slot_min
	nc_slot_max_idx=nc_slot_min_idx+slot_max_out-slot_min_out
	

	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()
	
	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation
		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
#		print px_xmin, px_xmax, px_ymin, px_ymax
		
		try:
			res=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, px_ymin:px_ymax+1,px_xmin:px_xmax+1]
		except:
			logger.error("unable to read from NetCDF file: %s",ncfile)
			print sys.exc_info()
			print 'try to read coordinates:', nc_dfb_min_idx,nc_dfb_max_idx+1,nc_slot_min_idx,nc_slot_max_idx+1, px_ymin,px_ymax+1,px_xmin,px_xmax+1
			rootgrpout.close()
			return None
		output_data[data_dfb_min_idx:data_dfb_max_idx+1,data_slot_min_idx:data_slot_max_idx+1,:,:] = res	

	else:
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
#		if not nc_ubbox.contains(seg_ubbox) and False:
#			logger.error("unable to read from NetCDF file: %s",ncfile)
#			px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
#			print 'coordinates out of nc bbox:', nc_dfb_min_idx,nc_dfb_max_idx+1,nc_slot_min_idx,nc_slot_max_idx+1, px_ymin,px_ymax+1,px_xmin,px_xmax+1
#			rootgrpout.close()
#			return None
			
		col_idxs1, row_idxs1, col_idxs2, row_idxs2, col_wghts, row_wghts = seg_ubbox.px_idx_grid_of_second_bbox(nc_ubbox)
		
#		print col_idxs1, row_idxs1, col_idxs2, row_idxs2, col_wghts, row_wghts, interpolate
		if interpolate=='nearest':
			wh = (col_idxs1 == col_idxs1) & (row_idxs1==row_idxs1) & (col_idxs1 != -9) & (row_idxs1 != -9)
			col_min=col_idxs1[wh].min()
			col_max=col_idxs1[wh].max()
			row_min=row_idxs1[wh].min()
			row_max=row_idxs1[wh].max()
#			print  col_min,col_max,row_min,row_max
#			print nc_dfb_min_idx,  nc_dfb_max_idx, nc_slot_min_idx, nc_slot_max_idx
			try:
				data=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, row_min:row_max+1,col_min:col_max+1]
			except:
				logger.error("unable to read from NetCDF file: %s",ncfile)
				print sys.exc_info()
				rootgrpout.close()
				return None
			res=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
		else:
			wh = (col_idxs1 != -9) & (col_idxs2 != -9) & (row_idxs1 != -9) & (row_idxs2 != -9)
			wh = wh & (col_idxs1 == col_idxs1) & (col_idxs2 == col_idxs2) & (row_idxs1==row_idxs1) & (row_idxs2==row_idxs2)
			col_min=min(col_idxs1[wh].min(), col_idxs2[wh].min())
			col_max=max(col_idxs1[wh].max(), col_idxs2[wh].max())
			row_min=min(row_idxs1[wh].min(), row_idxs2[wh].min())
			row_max=max(row_idxs1[wh].max(), row_idxs2[wh].max())

			try:	
				data=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1,row_min:row_max+1,col_min:col_max+1]
			except:
				logger.error("unable to read from NetCDF file: %s",ncfile)
				print sys.exc_info()
				rootgrpout.close()
				return None
			
			out_data11=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
			out_data12=data[:,:,row_idxs1[wh]-row_min,col_idxs2[wh]-col_min]
			out_data21=data[:,:,row_idxs2[wh]-row_min,col_idxs1[wh]-col_min]
			out_data22=data[:,:,row_idxs2[wh]-row_min,col_idxs2[wh]-col_min]
	
			aux1 = out_data11 + row_wghts[wh]*(out_data21 - out_data11)
			aux2 = out_data12 + row_wghts[wh]*(out_data22 - out_data12)
			res=aux1 + col_wghts[wh]*(aux2 - aux1)

		output_data[data_dfb_min_idx:data_dfb_max_idx+1,data_slot_min_idx:data_slot_max_idx+1,wh] = res	
	
	rootgrpout.close()
	return output_data


def latlon_read_dfb_hour_lat_lon_nc_bbox(ncfile, varname=None,Hours=None, Dfbs=None, bbox=None, interpolate='nearest'):		
	if (interpolate not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolate)
		return None


	if (bbox is None or varname==None or Hours==None or Dfbs==None):
		logger.error("give me real stuff not nones for file: %s",ncfile)
		return None
	
	dfbs=Dfbs[1]-Dfbs[0]+1
	latlon_rows=bbox.height
	latlon_cols=bbox.width
	hours=Hours[1]-Hours[0]+1

	#open NC for reading
	try:
		rootgrpout = netCDF4.Dataset(ncfile, 'r')
	except:
		logger.error('Unable to open file %s. %s Skipping' % (ncfile, str(sys.exc_info())))
		return None
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'dfb' in varDictout.keys():
		dfb_var_name = 'dfb'
	elif 'day' in varDictout.keys():
		dfb_var_name = 'day'
	else:
		logger.error('Variable dfb not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	dfb_var_out=varDictout[dfb_var_name]

	if 'time' in varDictout.keys():
		time_var_name = 'time'
	elif 'hour' in varDictout.keys():
		time_var_name = 'hour'
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	time_var_out=varDictout[time_var_name]

	if varname not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (varname, ncfile))
		rootgrpout.close()
		return None
	data_var_out=varDictout[varname]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	times=len(dimDictout[time_var_name])

	#recalculate in and out dfb indexes					
	nc_dfb_min, nc_dfb_max = dfb_var_out.valid_range
	dfb_min_out=max(nc_dfb_min, Dfbs[0])
	dfb_max_out=min(nc_dfb_max, Dfbs[1])
	data_dfb_min_idx=dfb_min_out-Dfbs[0]
	data_dfb_max_idx=data_dfb_min_idx+dfb_max_out-dfb_min_out
	nc_dfb_min_idx=dfb_min_out-nc_dfb_min
	nc_dfb_max_idx=nc_dfb_min_idx+dfb_max_out-dfb_min_out
	
	#recalculate in and out slot indexes
	nc_hour_min, nc_hour_max = time_var_out.valid_range
	sub_hour_step=	(nc_hour_max-nc_hour_min)	/(times-1)
#	print 	sub_hour_step	
	hour_min_out=max(nc_hour_min, Hours[0])
	hour_max_out=min(nc_hour_max, Hours[1])
	data_hour_min_idx=int(round((hour_min_out)/sub_hour_step))
	data_hour_max_idx=int(round(data_hour_min_idx+(hour_max_out-hour_min_out)/sub_hour_step))
	nc_hour_min_idx=int(round((hour_min_out-nc_hour_min)/sub_hour_step))
	nc_hour_max_idx=int(round(nc_hour_min_idx+(hour_max_out-hour_min_out)/sub_hour_step))
#	
#	print Hours
#	print nc_hour_min, nc_hour_max
#	print hour_min_out, hour_max_out
#	print data_hour_min_idx, data_hour_max_idx
#	print nc_hour_min_idx, nc_hour_max_idx
#	print nc_dfb_min_idx,nc_dfb_max_idx
	
	
	hours=len(time_var_out) 
	output_data = numpy.zeros((dfbs,hours,latlon_rows,latlon_cols), dtype=numpy.float32)
	output_data[:,:,:,:] = numpy.nan
	
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation
		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		try:
			res=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_hour_min_idx:nc_hour_max_idx+1, px_ymin:px_ymax+1,px_xmin:px_xmax+1]
		except:
			logger.error("unable to read from NetCDF file: %s",ncfile)
			print sys.exc_info()
			rootgrpout.close()
			return None
		output_data[data_dfb_min_idx:data_dfb_max_idx+1,data_hour_min_idx:data_hour_max_idx+1,:,:] = res	

	else:
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
		col_idxs1, row_idxs1, col_idxs2, row_idxs2, col_wghts, row_wghts = seg_ubbox.px_idx_grid_of_second_bbox(nc_ubbox)
		wh = (col_idxs1 != -9) & (col_idxs2 != -9) & (row_idxs1 != -9) & (row_idxs2 != -9)
		if interpolate=='nearest':
			wh = (col_idxs1 == col_idxs1) & (row_idxs1==row_idxs1) 
			col_min=col_idxs1[col_idxs1==col_idxs1].min()
			col_max=col_idxs1[col_idxs1==col_idxs1].max()
			row_min=row_idxs1[row_idxs1==row_idxs1].min()
			row_max=row_idxs1[row_idxs1==row_idxs1].max()
			try:
				data=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_hour_min_idx:nc_hour_max_idx+1, row_min:row_max+1,col_min:col_max+1]
			except:
				logger.error("unable to read from NetCDF file: %s",ncfile)
				print sys.exc_info()
				rootgrpout.close()
				return None
			res=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
		else:
			wh = (col_idxs1 == col_idxs1) & (col_idxs2 == col_idxs2) & (row_idxs1==row_idxs1) & (row_idxs2==row_idxs2)
			col_min=min(col_idxs1[col_idxs1==col_idxs1].min(), col_idxs2[col_idxs2==col_idxs2].min())
			col_max=max(col_idxs1[col_idxs1==col_idxs1].max(), col_idxs2[col_idxs2==col_idxs2].max())
			row_min=min(row_idxs1[row_idxs1==row_idxs1].min(), row_idxs2[row_idxs2==row_idxs2].min())
			row_max=max(row_idxs1[row_idxs1==row_idxs1].max(), row_idxs2[row_idxs2==row_idxs2].max())

			try:	
				data=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_hour_min_idx:nc_hour_max_idx+1,row_min:row_max+1,col_min:col_max+1]
			except:
				logger.error("unable to read from NetCDF file: %s",ncfile)
				print sys.exc_info()
				rootgrpout.close()
				return None
			
			out_data11=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
			out_data12=data[:,:,row_idxs1[wh]-row_min,col_idxs2[wh]-col_min]
			out_data21=data[:,:,row_idxs2[wh]-row_min,col_idxs1[wh]-col_min]
			out_data22=data[:,:,row_idxs2[wh]-row_min,col_idxs2[wh]-col_min]
	
			aux1 = out_data11 + row_wghts[wh]*(out_data21 - out_data11)
			aux2 = out_data12 + row_wghts[wh]*(out_data22 - out_data12)
			res=aux1 + col_wghts[wh]*(aux2 - aux1)

		output_data[data_dfb_min_idx:data_dfb_max_idx+1,data_hour_min_idx:data_hour_max_idx+1,wh] = res	
	
	rootgrpout.close()
	return output_data


def latlon_read_dfb_lat_lon_nc_bbox(ncfile, varname=None,Dfbs=None, bbox=None, interpolate='nearest'):		
	if (interpolate not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolate)
		return None


	if (bbox is None or varname==None or Dfbs==None):
		logger.error("give me real stuff not nones for file: %s",ncfile)
		return None
	
	dfbs=Dfbs[1]-Dfbs[0]+1
	latlon_rows=bbox.height
	latlon_cols=bbox.width


	#open NC for reading
	try:
		rootgrpout = netCDF4.Dataset(ncfile, 'r')
	except:
		logger.error('Unable to open file %s. %s Skipping' % (ncfile, str(sys.exc_info())))
		return None
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'dfb' in varDictout.keys():
		dfb_var_name = 'dfb'
	elif 'day' in varDictout.keys():
		dfb_var_name = 'day'
	else:
		logger.error('Variable dfb not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	dfb_var_out=varDictout[dfb_var_name]

	if 'time' in varDictout.keys():
		time_var_name = 'time'
	elif 'hour' in varDictout.keys():
		time_var_name = 'hour'
	elif 'slot' in varDictout.keys():
		time_var_name = 'slot'	
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	time_var_out=varDictout[time_var_name]

	if varname not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (varname, ncfile))
		rootgrpout.close()
		return None
	data_var_out=varDictout[varname]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])


	#recalculate in and out dfb indexes					
	nc_dfb_min, nc_dfb_max = dfb_var_out.valid_range
	dfb_min_out=max(nc_dfb_min, Dfbs[0])
	dfb_max_out=min(nc_dfb_max, Dfbs[1])
	data_dfb_min_idx=dfb_min_out-Dfbs[0]
	data_dfb_max_idx=data_dfb_min_idx+dfb_max_out-dfb_min_out
	nc_dfb_min_idx=dfb_min_out-nc_dfb_min
	nc_dfb_max_idx=nc_dfb_min_idx+dfb_max_out-dfb_min_out
	
	
	
	
	times=len(time_var_out)
	output_data = numpy.zeros((dfbs,times,latlon_rows,latlon_cols), dtype=numpy.float32)
	output_data[:,:,:,:] = numpy.nan
	
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation
		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		try:
			res=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,:, px_ymin:px_ymax+1,px_xmin:px_xmax+1]
		except:
			logger.error("unable to read from NetCDF file: %s",ncfile)
			print sys.exc_info()
			rootgrpout.close()
			return None
		output_data[data_dfb_min_idx:data_dfb_max_idx+1,:,:,:] = res	

	else:
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
		col_idxs1, row_idxs1, col_idxs2, row_idxs2, col_wghts, row_wghts = seg_ubbox.px_idx_grid_of_second_bbox(nc_ubbox)
		wh = (col_idxs1 != -9) & (col_idxs2 != -9) & (row_idxs1 != -9) & (row_idxs2 != -9)
		if interpolate=='nearest':
			wh = (col_idxs1 == col_idxs1) & (row_idxs1==row_idxs1) 
			col_min=col_idxs1[col_idxs1==col_idxs1].min()
			col_max=col_idxs1[col_idxs1==col_idxs1].max()
			row_min=row_idxs1[row_idxs1==row_idxs1].min()
			row_max=row_idxs1[row_idxs1==row_idxs1].max()
			try:
				data=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,:, row_min:row_max+1,col_min:col_max+1]
			except:
				logger.error("unable to read from NetCDF file: %s",ncfile)
				print sys.exc_info()
				rootgrpout.close()
				return None
			res=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
		else:
			wh = (col_idxs1 == col_idxs1) & (col_idxs2 == col_idxs2) & (row_idxs1==row_idxs1) & (row_idxs2==row_idxs2)
			col_min=min(col_idxs1[col_idxs1==col_idxs1].min(), col_idxs2[col_idxs2==col_idxs2].min())
			col_max=max(col_idxs1[col_idxs1==col_idxs1].max(), col_idxs2[col_idxs2==col_idxs2].max())
			row_min=min(row_idxs1[row_idxs1==row_idxs1].min(), row_idxs2[row_idxs2==row_idxs2].min())
			row_max=max(row_idxs1[row_idxs1==row_idxs1].max(), row_idxs2[row_idxs2==row_idxs2].max())

			try:	
				data=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,:,row_min:row_max+1,col_min:col_max+1]
			except:
				logger.error("unable to read from NetCDF file: %s",ncfile)
				print sys.exc_info()
				rootgrpout.close()
				return None
			
			out_data11=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
			out_data12=data[:,:,row_idxs1[wh]-row_min,col_idxs2[wh]-col_min]
			out_data21=data[:,:,row_idxs2[wh]-row_min,col_idxs1[wh]-col_min]
			out_data22=data[:,:,row_idxs2[wh]-row_min,col_idxs2[wh]-col_min]
	
			aux1 = out_data11 + row_wghts[wh]*(out_data21 - out_data11)
			aux2 = out_data12 + row_wghts[wh]*(out_data22 - out_data12)
			res=aux1 + col_wghts[wh]*(aux2 - aux1)

		output_data[data_dfb_min_idx:data_dfb_max_idx+1,:,wh] = res	
	
	rootgrpout.close()
	return output_data



def latlon_read_month_hour_lat_lon_nc_bbox(ncfile, varname=None,Hours=None, Months=None, bbox=None, interpolate='nearest'):		
	if (interpolate not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolate)
		return None


	if (bbox is None or varname==None or Hours==None or Months==None):
		logger.error("give me real stuff not nones for file: %s",ncfile)
		return None
	
	months=Months[1]-Months[0]+1
	latlon_rows=bbox.height
	latlon_cols=bbox.width
	hours=Hours[1]-Hours[0]+1

	#open NC for reading
	try:
		rootgrpout = netCDF4.Dataset(ncfile, 'r')
	except:
		logger.error('Unable to open file %s. %s Skipping' % (ncfile, str(sys.exc_info())))
		return None

	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'month' in varDictout.keys():
		month_var_name = 'month'
	else:
		logger.error('Variable month not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	month_var_out=varDictout[month_var_name]

	if 'time' in varDictout.keys():
		time_var_name = 'time'
	elif 'hour' in varDictout.keys():
		time_var_name = 'hour'
	elif 'slot' in varDictout.keys():
		time_var_name = 'slot'
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	time_var_out=varDictout[time_var_name]

	if varname not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (varname, ncfile))
		rootgrpout.close()
		return None
	data_var_out=varDictout[varname]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	elif 'lon' in dimDictout.keys():
		col_name='lon'
		row_name='lat'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])


	#recalculate in and out month indexes					
	nc_month_min = month_var_out[:].min()
	nc_month_max = month_var_out[:].max()
	month_min_out=max(nc_month_min, Months[0])
	month_max_out=min(nc_month_max, Months[1])
	data_month_min_idx=month_min_out-Months[0]
	data_month_max_idx=data_month_min_idx+month_max_out-month_min_out
	nc_month_min_idx=month_min_out-nc_month_min
	nc_month_max_idx=nc_month_min_idx+month_max_out-month_min_out
	
	
	#recalculate in and out slot indexes					
	nc_hour_min = time_var_out[:].min()
	nc_hour_max = time_var_out[:].max()
	hour_min_out=max(nc_hour_min, Hours[0])
	hour_max_out=min(nc_hour_max, Hours[1])
	data_hour_min_idx=hour_min_out-Hours[0]
	data_hour_max_idx=data_hour_min_idx+hour_max_out-hour_min_out
	nc_hour_min_idx=hour_min_out-nc_hour_min
	nc_hour_max_idx=nc_hour_min_idx+hour_max_out-hour_min_out
	
#	hours=len(time_var_out)
	output_data = numpy.zeros((months,hours,latlon_rows,latlon_cols), dtype=numpy.float32)
	output_data[:,:,:,:] = numpy.nan
	
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = bbox.get_uneven_bbox()

	if nc_ubbox.equal_resolution(seg_ubbox) and nc_ubbox.contains(seg_ubbox):
		#equal resolution of bboxes - use simplified reader without interpolation
		px_xmin, px_xmax, px_ymin, px_ymax = nc_ubbox.pixel_coords_of_bbox(seg_ubbox)
		try:
			res=data_var_out[nc_month_min_idx:nc_month_max_idx+1,nc_hour_min_idx:nc_hour_max_idx+1, px_ymin:px_ymax+1,px_xmin:px_xmax+1]
		except:
			logger.error("unable to read from NetCDF file: %s",ncfile)
			print sys.exc_info()
			rootgrpout.close()
			return None
		output_data[data_month_min_idx:data_month_max_idx+1,data_hour_min_idx:data_hour_max_idx+1,:,:] = res	

	else:
		#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
		col_idxs1, row_idxs1, col_idxs2, row_idxs2, col_wghts, row_wghts = seg_ubbox.px_idx_grid_of_second_bbox(nc_ubbox)
		wh = (col_idxs1 != -9) & (col_idxs2 != -9) & (row_idxs1 != -9) & (row_idxs2 != -9)
		if interpolate=='nearest':
			wh = (col_idxs1 == col_idxs1) & (row_idxs1==row_idxs1) 
			col_min=col_idxs1[col_idxs1==col_idxs1].min()
			col_max=col_idxs1[col_idxs1==col_idxs1].max()
			row_min=row_idxs1[row_idxs1==row_idxs1].min()
			row_max=row_idxs1[row_idxs1==row_idxs1].max()
			try:
				data=data_var_out[nc_month_min_idx:nc_month_max_idx+1,nc_hour_min_idx:nc_hour_max_idx+1, row_min:row_max+1,col_min:col_max+1]
			except:
				logger.error("unable to read from NetCDF file: %s",ncfile)
				print sys.exc_info()
				rootgrpout.close()
				return None
			res=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
		else:
			wh = (col_idxs1 == col_idxs1) & (col_idxs2 == col_idxs2) & (row_idxs1==row_idxs1) & (row_idxs2==row_idxs2)
			col_min=min(col_idxs1[col_idxs1==col_idxs1].min(), col_idxs2[col_idxs2==col_idxs2].min())
			col_max=max(col_idxs1[col_idxs1==col_idxs1].max(), col_idxs2[col_idxs2==col_idxs2].max())
			row_min=min(row_idxs1[row_idxs1==row_idxs1].min(), row_idxs2[row_idxs2==row_idxs2].min())
			row_max=max(row_idxs1[row_idxs1==row_idxs1].max(), row_idxs2[row_idxs2==row_idxs2].max())

			try:	
				data=data_var_out[nc_month_min_idx:nc_month_max_idx+1,nc_hour_min_idx:nc_hour_max_idx+1,row_min:row_max+1,col_min:col_max+1]
			except:
				logger.error("unable to read from NetCDF file: %s",ncfile)
				print sys.exc_info()
				rootgrpout.close()
				return None
			
			out_data11=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
			out_data12=data[:,:,row_idxs1[wh]-row_min,col_idxs2[wh]-col_min]
			out_data21=data[:,:,row_idxs2[wh]-row_min,col_idxs1[wh]-col_min]
			out_data22=data[:,:,row_idxs2[wh]-row_min,col_idxs2[wh]-col_min]
	
			aux1 = out_data11 + row_wghts[wh]*(out_data21 - out_data11)
			aux2 = out_data12 + row_wghts[wh]*(out_data22 - out_data12)
			res=aux1 + col_wghts[wh]*(aux2 - aux1)

		output_data[data_month_min_idx:data_month_max_idx+1,data_hour_min_idx:data_hour_max_idx+1,wh] = res	
	
	rootgrpout.close()
	return output_data

def latlon_read_dfb_slot_lat_lon_nc_point(ncfile, chan, Dfbs, Slots, lon, lat, interpolate='nearest',lon_circular=False):		
	if (interpolate not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolate)
		return None
	
	
	dfbs=Dfbs[1]-Dfbs[0]+1
	slots=Slots[1]-Slots[0]+1
	output_data = numpy.zeros((dfbs,slots), dtype=numpy.float32)
	output_data[:,:] = numpy.nan

	
	#open NC for reading
	try:
		rootgrpout = netCDF4.Dataset(ncfile, 'r')
	except:
		logger.error('Unable to open file %s. %s Skipping' % (ncfile, str(sys.exc_info())))
		return None
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	if 'dfb' in varDictout.keys():
		dfb_var_name = 'dfb'
	elif 'month' in varDictout.keys():
		dfb_var_name = 'month'	
	elif 'day' in varDictout.keys():
		dfb_var_name = 'day'	
	else:
		logger.error('Variable dfb not found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	dfb_var_out=varDictout[dfb_var_name]

	if 'slot' in varDictout.keys():
		slot_var_name = 'slot'
	elif 'time' in varDictout.keys():
		slot_var_name = 'time'
	else:
		logger.error('No variable for slot found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None
	slot_var_out=varDictout[slot_var_name]

	if chan not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (chan, ncfile))
		rootgrpout.close()
		return None
	data_var_out=varDictout[chan]

	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	else:
		logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (ncfile))
		rootgrpout.close()
		return None

	ncvar_col=varDictout[col_name]
	ncvar_row=varDictout[row_name]
	cols=len(dimDictout[col_name])
	rows=len(dimDictout[row_name])

	#recalculate in and out dfb indexes					
	nc_dfb_min, nc_dfb_max = dfb_var_out.valid_range
	dfb_min_out=max(nc_dfb_min, Dfbs[0])
	dfb_max_out=min(nc_dfb_max, Dfbs[1])
	data_dfb_min_idx=dfb_min_out-Dfbs[0]
	data_dfb_max_idx=data_dfb_min_idx+dfb_max_out-dfb_min_out
	nc_dfb_min_idx=dfb_min_out-nc_dfb_min
	nc_dfb_max_idx=nc_dfb_min_idx+dfb_max_out-dfb_min_out
	
	
	#recalculate in and out slot indexes					
#	nc_slot_min, nc_slot_max = slot_var_out.valid_range
	nc_slot_min, nc_slot_max = slot_var_out[:].min(), slot_var_out[:].max()
	slot_min_out=max(nc_slot_min, Slots[0])
	slot_max_out=min(nc_slot_max, Slots[1])
	data_slot_min_idx=slot_min_out-Slots[0]
	data_slot_max_idx=data_slot_min_idx+slot_max_out-slot_min_out
	nc_slot_min_idx=slot_min_out-nc_slot_min
	nc_slot_max_idx=nc_slot_min_idx+slot_max_out-slot_min_out
	

	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range

#   TODO:TST IT ON ALL readers we are using 
#	aux = (ncvar_col[:].max() - ncvar_col[:].min())
#	if cols>1:
#		colres=aux/(cols-1)
#	else:
#		colres=aux
#	col_min = ncvar_col[:].min() - (colres/2.)
#	col_max = ncvar_col[:].max() + (colres/2.)
#
#	aux = (ncvar_row[:].max() - ncvar_row[:].min())
#	if rows>1:
#		rowres=aux/(rows-1)
#	else:
#		rowres=aux
#	row_min = ncvar_row[:].min() - (rowres/2.)
#	row_max = ncvar_row[:].max() + (rowres/2.)

	
	colres=float(col_max-col_min)/(cols)
	rowres=float(row_max-row_min)/(rows)
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	pxidxs = nc_ubbox.px_idxs_of_latlon(lon, lat, lon_circular=lon_circular)
	
	
	if pxidxs is None:
		return None
	
	x1, y1, x2, y2, x_wght, y_wght = pxidxs
	y1 = max(y1,0)
	x1 = max(x1,0)
	y2 = min(y2,data_var_out.shape[2]-1)
	x2 = min(x2,data_var_out.shape[3]-1)
	

	#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	if interpolate=='nearest':
		data=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, y1, x1]
	else:
		px_x1y1=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, y1, x1]
		px_x1y2=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, y2, x1]
		px_x2y1=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, y1, x2]
		px_x2y2=data_var_out[nc_dfb_min_idx:nc_dfb_max_idx+1,nc_slot_min_idx:nc_slot_max_idx+1, y2, x2]
		
		if (px_x1y1 is None) or (px_x1y2 is None) or (px_x1y2 is None) or (px_x2y2 is None):
			return None
		#bilinear interpolation
		aux1=px_x1y1 + (px_x2y1-px_x1y1)*x_wght
		aux2=px_x1y2 + (px_x2y2-px_x1y2)*x_wght
		data=aux1 + (aux2-aux1)*y_wght

	output_data[data_dfb_min_idx:data_dfb_max_idx+1,data_slot_min_idx:data_slot_max_idx+1] = data	
	
	rootgrpout.close()
	return output_data

#read data from the the NC file
def latlon_read_lat_lon_nc_whole(nc_file_name, varname, return_bbox=False):

	if (varname is None) or (varname==''):
		logger.warning("empty variable name")
		return None
	if not(os.access(nc_file_name, os.F_OK)):
		logger.warning("file %s not found",nc_file_name)
		return None
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		return None

	varDictout=rootgrp.variables
	if return_bbox:
		dimDictout=rootgrp.dimensions
		if 'col' in dimDictout.keys():
			col_name='col'
			row_name='row'
		elif 'longitude' in dimDictout.keys():
			col_name='longitude'
			row_name='latitude'
		elif 'lon' in dimDictout.keys():
			col_name='lon'
			row_name='lat'
		else:
			logger.error('No longitude/latitude or col/row dimensions found in nc %s . Skipping' % (nc_file_name))
			rootgrp.close()
			return False
		
		if 'col' in varDictout.keys():
			var_col_name='col'
			var_row_name='row'
		elif 'longitude' in varDictout.keys():
			var_col_name='longitude'
			var_row_name='latitude'
		elif 'lon' in varDictout.keys():
			var_col_name='lon'
			var_row_name='lat'
		else:
			logger.error('No longitude/latitude or col/row variables found in nc %s . Skipping' % (nc_file_name))
			rootgrp.close()
			return False

		ncvar_col=varDictout[var_col_name]
		ncvar_row=varDictout[var_row_name]
		cols=len(dimDictout[col_name])
		rows=len(dimDictout[row_name])
		if 'valid_range' in ncvar_col.ncattrs():  
			col_min, col_max = ncvar_col.valid_range
			row_min, row_max = ncvar_row.valid_range
			colres=(col_max-col_min)/(cols)
			rowres=(row_max-row_min)/(rows)
		else:
			col_min = ncvar_col[:].min()
			col_max = ncvar_col[:].max()
			row_min = ncvar_row[:].min()
			row_max = ncvar_row[:].max()
			colres=(col_max-col_min)/(cols-1)
			rowres=(row_max-row_min)/(rows-1)
			col_min -= colres/2.
			col_max += colres/2.
			row_min -= rowres/2.
			row_max += rowres/2.
		nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	
	
	
	if len(varDictout)<1:
		logger.error("no variables found in NetCDF file: %s",nc_file_name)
		rootgrp.close()
		return None
	
	if not(varname in varDictout.keys()):
		logger.error("variable %s not found in NetCDF  %s",varname ,nc_file_name)
		rootgrp.close()
		return None
	
	ncvarimg=varDictout[varname]
	img = ncvarimg[:,:]
	
	rootgrp.close()
	if return_bbox:
		return img, nc_ubbox
	return img


#read variable from nc files
def latlon_read_lat_lon_nc_bbox(nc_file_name, varname, seg_bbox=None, interpolate='nearest'):
	
	if (seg_bbox is None):
		logger.error("no or None bbox to read soecified: %s",interpolate)
		return None
	
	
	if (interpolate not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolate)
		return None
		
	#print nc_file_name
	if not(os.access(nc_file_name,os.F_OK)):
		logger.error("unable to find NetCDF file: %s",nc_file_name)
		return None

	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		
		return None

	dimDictout=rootgrp.dimensions
	if 'col' in dimDictout.keys():
		col_name='col'
		row_name='row'
	elif 'longitude' in dimDictout.keys():
		col_name='longitude'
		row_name='latitude'
	elif 'lon' in dimDictout.keys():
		col_name='lon'
		row_name='lat'
	try:
		cols=len(dimDictout[col_name])
		rows=len(dimDictout[row_name])
	except:
		logger.error("one of dimensions longitude, latitude not found in "+ nc_file_name)
		
		rootgrp.close()
		return None

	ncvariabs=rootgrp.variables
	if 'col' in ncvariabs.keys():
		vcol_name='col'
		vrow_name='row'
	elif 'longitude' in ncvariabs.keys():
		vcol_name='longitude'
		vrow_name='latitude'
	elif 'lon' in ncvariabs.keys():
		vcol_name='lon'
		vrow_name='lat'
	try:
		ncvar_data=ncvariabs[varname]
		ncvar_data.set_auto_maskandscale(False)
		ncvar_col=ncvariabs[vcol_name]
		ncvar_row=ncvariabs[vrow_name]
	except:
		logger.error("one of variables: %s, latitude, longitude, not found in %s" ,varname,  nc_file_name)
		
		rootgrp.close()
		return None
		
#	col_min, col_max = ncvar_col.valid_range
#	row_min, row_max = ncvar_row.valid_range
#	
#	colres=(col_max-col_min)/(cols)
#	rowres=(row_max-row_min)/(rows)


	if 'valid_range' in ncvar_col.ncattrs():  
		col_min, col_max = ncvar_col.valid_range
		row_min, row_max = ncvar_row.valid_range
		colres=(col_max-col_min)/(cols)
		rowres=(row_max-row_min)/(rows)
	else:
		col_min = ncvar_col[:].min()
		col_max = ncvar_col[:].max()
		row_min = ncvar_row[:].min()
		row_max = ncvar_row[:].max()
		colres=(col_max-col_min)/(cols-1)
		rowres=(row_max-row_min)/(rows-1)
		col_min -= colres/2.
		col_max += colres/2.
		row_min -= rowres/2.
		row_max += rowres/2.




	
	#this is test whether modulo recalculation of the column coordinate may be done
	#in case the data are global (360 deg.)
#	lon_circular=False
#	epsilon=0.001
#	total_width = col_max - col_min + colres
#	if (total_width > (360-epsilon)) and  (total_width < (360+epsilon)):
#		lon_circular=True
	
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)

	seg_ubbox = seg_bbox.get_uneven_bbox()
	col_idxs1, row_idxs1, col_idxs2, row_idxs2, col_wghts, row_wghts = seg_ubbox.px_idx_grid_of_second_bbox(nc_ubbox)

	wh = (col_idxs1 != -9) & (col_idxs2 != -9) & (row_idxs1 != -9) & (row_idxs2 != -9)
	if interpolate=='nearest':
		wh = (col_idxs1 == col_idxs1) & (row_idxs1==row_idxs1) 
		col_min=col_idxs1[col_idxs1==col_idxs1].min()
		col_max=col_idxs1[col_idxs1==col_idxs1].max()
		row_min=row_idxs1[row_idxs1==row_idxs1].min()
		row_max=row_idxs1[row_idxs1==row_idxs1].max()
		if ncvar_data.ndim == 2:
			data=ncvar_data[row_min:row_max+1,col_min:col_max+1]
			res=data[row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
		elif ncvar_data.ndim == 3:
			data=ncvar_data[:,row_min:row_max+1,col_min:col_max+1]
			res=data[:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
		elif ncvar_data.ndim == 4:
			data=ncvar_data[:,:,row_min:row_max+1,col_min:col_max+1]
			res=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]

	else:
		wh = (col_idxs1 == col_idxs1) & (col_idxs2 == col_idxs2) & (row_idxs1==row_idxs1) & (row_idxs2==row_idxs2)
		col_min=min(col_idxs1[col_idxs1==col_idxs1].min(), col_idxs2[col_idxs2==col_idxs2].min())
		col_max=max(col_idxs1[col_idxs1==col_idxs1].max(), col_idxs2[col_idxs2==col_idxs2].max())
		row_min=min(row_idxs1[row_idxs1==row_idxs1].min(), row_idxs2[row_idxs2==row_idxs2].min())
		row_max=max(row_idxs1[row_idxs1==row_idxs1].max(), row_idxs2[row_idxs2==row_idxs2].max())

		if ncvar_data.ndim == 2:
			data=ncvar_data[row_min:row_max+1,col_min:col_max+1]
			out_data11=data[row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
			out_data12=data[row_idxs1[wh]-row_min,col_idxs2[wh]-col_min]
			out_data21=data[row_idxs2[wh]-row_min,col_idxs1[wh]-col_min]
			out_data22=data[row_idxs2[wh]-row_min,col_idxs2[wh]-col_min]
		elif ncvar_data.ndim == 3:
			data=ncvar_data[:,row_min:row_max+1,col_min:col_max+1]
			out_data11=data[:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
			out_data12=data[:,row_idxs1[wh]-row_min,col_idxs2[wh]-col_min]
			out_data21=data[:,row_idxs2[wh]-row_min,col_idxs1[wh]-col_min]
			out_data22=data[:,row_idxs2[wh]-row_min,col_idxs2[wh]-col_min]
		elif ncvar_data.ndim == 4:
			data=ncvar_data[:,:,row_min:row_max+1,col_min:col_max+1]
			out_data11=data[:,:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
			out_data12=data[:,:,row_idxs1[wh]-row_min,col_idxs2[wh]-col_min]
			out_data21=data[:,:,row_idxs2[wh]-row_min,col_idxs1[wh]-col_min]
			out_data22=data[:,:,row_idxs2[wh]-row_min,col_idxs2[wh]-col_min]

		aux1 = out_data11 + row_wghts[wh]*(out_data21 - out_data11)
		aux2 = out_data12 + row_wghts[wh]*(out_data22 - out_data12)
		res=aux1 + col_wghts[wh]*(aux2 - aux1)

	latlon_rows=seg_bbox.height
	latlon_cols=seg_bbox.width
	if ncvar_data.ndim == 2:
		output_data = numpy.zeros((latlon_rows,latlon_cols), dtype=numpy.float32)
		output_data[:,:] = numpy.nan
		output_data[wh] = res	
	elif ncvar_data.ndim == 3:
		output_data = numpy.zeros((ncvar_data.shape[0],latlon_rows,latlon_cols), dtype=numpy.float32)
		output_data[:,:,:] = numpy.nan
		output_data[:,wh] = res	
	elif ncvar_data.ndim == 4:
		output_data = numpy.zeros((ncvar_data.shape[0],ncvar_data.shape[1],latlon_rows,latlon_cols), dtype=numpy.float32)
		output_data[:,:,:,:] = numpy.nan
		output_data[:,:,wh] = res	
	
	#rescale data
	ncvar_attrs = ncvar_data.ncattrs()
	if ('scale_factor' in ncvar_attrs) and ('add_offset' in ncvar_attrs):
		scale=ncvar_data.scale_factor
		offset=ncvar_data.add_offset
		output_data = offset + (scale*output_data)
	
	rootgrp.close()
	return output_data


#read variable from nc files
def latlon_read_lat_lon_nc_point(nc_file_name, varname, lon=None, lat=None, interpolate='nearest'):
	
	if (lon is None) or (lat is None):
		logger.error("problem with point coordinates")
		return None
	
	if (interpolate not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolate)
		return None
		
	#print nc_file_name
	if not(os.access(nc_file_name,os.F_OK)):
		logger.error("unable to find NetCDF file: %s",nc_file_name)
		return None

	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		
		return None

	ncdims=rootgrp.dimensions
	if 'longitude' in ncdims.keys():
		col_name='longitude'
		row_name='latitude'
	elif 'lon' in ncdims.keys():
		col_name='lon'
		row_name='lat'
	
	
	try:
		cols=len(ncdims[col_name])
		rows=len(ncdims[row_name])
	except:
		logger.error("one of dimensions longitude, latitude not found in "+ nc_file_name)
		
		rootgrp.close()
		return None

	ncvariabs=rootgrp.variables
	try:
		ncvar_data=ncvariabs[varname]
		ncvar_data.set_auto_maskandscale(False)
		ncvar_col=ncvariabs[col_name]
		ncvar_row=ncvariabs[row_name]
	except:
		logger.error("one of variables: %s, latitude, longitude, not found in %s" ,varname,  nc_file_name)
		
		rootgrp.close()
		return None


#	if 	'valid_range' not in ncvar_col.ncattrs():
#		aux = (ncvar_col[:].max() - ncvar_col[:].min())
#		if cols>1:
#			colres=aux/(cols-1)
#		else:
#			colres=aux
#		col_min = ncvar_col[:].min() - (colres/2.)
#		col_max = ncvar_col[:].max() + (colres/2.)
#	else:
#		col_min, col_max = ncvar_col.valid_range
#		colres=(col_max-col_min)/(cols)
#
#	if 	'valid_range' not in ncvar_row.ncattrs():
#		aux = (ncvar_row[:].max() - ncvar_row[:].min())
#		if rows>1:
#			rowres=aux/(rows-1)
#		else:
#			rowres=aux
#		row_min = ncvar_row[:].min() - (rowres/2.)
#		row_max = ncvar_row[:].max() + (rowres/2.)
#	else:
#		row_min, row_max = ncvar_row.valid_range
#		if row_max < row_min:
#			aux = row_min
#			row_min = row_max
#			row_max = aux
#		rowres=(row_max-row_min)/(rows)

	aux = (ncvar_col[:].max() - ncvar_col[:].min())
	if cols>1:
		colres=aux/(cols-1)
	else:
		colres=aux
	col_min = ncvar_col[:].min() - (colres/2.)
	col_max = ncvar_col[:].max() + (colres/2.)

	aux = (ncvar_row[:].max() - ncvar_row[:].min())
	if rows>1:
		rowres=aux/(rows-1)
	else:
		rowres=aux
	row_min = ncvar_row[:].min() - (rowres/2.)
	row_max = ncvar_row[:].max() + (rowres/2.)
	

	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	pxidxs = nc_ubbox.px_idxs_of_latlon(lon, lat, True)
	if pxidxs is None:
		logger.error("point out of region %s, %.f, %.f, %s" ,nc_file_name, lat, lon, str(nc_ubbox))
		return None
	
	x1, y1, x2, y2, x_wght, y_wght = pxidxs
	print nc_ubbox
	print lon, lat , x1, y1, x2, y2, x_wght, y_wght
	if (ncvar_row[0]==ncvar_row[0]) and (ncvar_row[-1]==ncvar_row[-1]) and (ncvar_row[0]<ncvar_row[-1]):
		logger.warning("latitude in bottom to top direction, adapting read" )
		y1=rows-y1
		y2=rows-y2
	
	#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	if interpolate=='nearest':
		if ncvar_data.ndim == 2:
			data=ncvar_data[y1, x1]
		if ncvar_data.ndim == 3:
			data=ncvar_data[:,y1, x1]
		if ncvar_data.ndim == 4:
			data=ncvar_data[:,:,y1, x1]
		if ncvar_data.ndim == 5:
			data=ncvar_data[:,:,y1, x1]
	else:
		if ncvar_data.ndim == 2:
			px_x1y1=ncvar_data[y1, x1]
			px_x1y2=ncvar_data[y2, x1]
			px_x2y1=ncvar_data[y1, x2]
			px_x2y2=ncvar_data[y2, x2]
		if ncvar_data.ndim == 3:
			px_x1y1=ncvar_data[:, y1, x1]
			px_x1y2=ncvar_data[:, y2, x1]
			px_x2y1=ncvar_data[:, y1, x2]
			px_x2y2=ncvar_data[:, y2, x2]
		if ncvar_data.ndim == 4:
			px_x1y1=ncvar_data[:,:, y1, x1]
			px_x1y2=ncvar_data[:,:, y2, x1]
			px_x2y1=ncvar_data[:,:, y1, x2]
			px_x2y2=ncvar_data[:,:, y2, x2]
		if ncvar_data.ndim == 5:
			px_x1y1=ncvar_data[:,:,:, y1, x1]
			px_x1y2=ncvar_data[:,:,:, y2, x1]
			px_x2y1=ncvar_data[:,:,:, y1, x2]
			px_x2y2=ncvar_data[:,:,:, y2, x2]

		if (px_x1y1 is None) or (px_x1y2 is None) or (px_x1y2 is None) or (px_x2y2 is None):
			logger.error("point out of region %s, %.f, %.f" ,nc_file_name, lat, lon)
			return None
		#bilinear interpolation
		aux1=px_x1y1 + (px_x2y1-px_x1y1)*x_wght
		aux2=px_x1y2 + (px_x2y2-px_x1y2)*x_wght
		data=aux1 + (aux2-aux1)*y_wght

	output_data=data

	#rescale data
	ncvar_attrs = ncvar_data.ncattrs()
	if ('scale_factor' in ncvar_attrs) and ('add_offset' in ncvar_attrs):
		scale=ncvar_data.scale_factor
		offset=ncvar_data.add_offset
		output_data = offset + (scale*output_data)
	
	rootgrp.close()
	return output_data

#read variable from nc files
def latlon_read_months_lat_lon_nc_bbox(nc_file_name, varname, seg_bbox=None, interpolate='nearest'):
	
	if (seg_bbox is None):
		logger.error("no or None bbox to read soecified: %s",interpolate)
		return None
	
	latlon_rows=seg_bbox.height
	latlon_cols=seg_bbox.width
	output_data = numpy.zeros((12,latlon_rows,latlon_cols), dtype=numpy.float32)
	
	if (interpolate not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolate)
		return None
		
	#print nc_file_name
	if not(os.access(nc_file_name,os.F_OK)):
		logger.error("unable to find NetCDF file: %s",nc_file_name)
		return None

	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		
		return None

	ncdims=rootgrp.dimensions
	try:
		cols=len(ncdims['longitude'])
		rows=len(ncdims['latitude'])
	except:
		logger.error("one of dimensions longitude, latitude not found in "+ nc_file_name)
		
		rootgrp.close()
		return None

	ncvariabs=rootgrp.variables
	try:
		ncvar_data=ncvariabs[varname]
		ncvar_data.set_auto_maskandscale(False)
		ncvar_col=ncvariabs['longitude']
		ncvar_row=ncvariabs['latitude']
	except:
		logger.error("one of variables: %s, latitude, longitude, not found in %s" ,varname,  nc_file_name)
		
		rootgrp.close()
		return None
	
	if 'valid_range' in ncvar_col.ncattrs():  
		col_min, col_max = ncvar_col.valid_range
		row_min, row_max = ncvar_row.valid_range
		colres=(col_max-col_min)/(cols)
		rowres=(row_max-row_min)/(rows)
	else:
		col_min = ncvar_col[:].min()
		col_max = ncvar_col[:].max()
		row_min = ncvar_row[:].min()
		row_max = ncvar_row[:].max()
		colres=(col_max-col_min)/(cols-1)
		rowres=(row_max-row_min)/(rows-1)
		col_min -= colres/2.
		col_max += colres/2.
		row_min -= rowres/2.
		row_max += rowres/2.
		
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	seg_ubbox = seg_bbox.get_uneven_bbox()
	
	col_idxs1, row_idxs1, col_idxs2, row_idxs2, col_wghts, row_wghts = seg_ubbox.px_idx_grid_of_second_bbox(nc_ubbox)


	wh = (col_idxs1 != -9) & (col_idxs2 != -9) & (row_idxs1 != -9) & (row_idxs2 != -9)
	if interpolate=='nearest':
		wh = (col_idxs1 == col_idxs1) & (row_idxs1==row_idxs1) 
		col_min=col_idxs1[col_idxs1==col_idxs1].min()
		col_max=col_idxs1[col_idxs1==col_idxs1].max()
		row_min=row_idxs1[row_idxs1==row_idxs1].min()
		row_max=row_idxs1[row_idxs1==row_idxs1].max()
		data=ncvar_data[:,row_min:row_max+1,col_min:col_max+1]
		res=data[:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
	else:
		wh = (col_idxs1 == col_idxs1) & (col_idxs2 == col_idxs2) & (row_idxs1==row_idxs1) & (row_idxs2==row_idxs2)
		col_min=min(col_idxs1[col_idxs1==col_idxs1].min(), col_idxs2[col_idxs2==col_idxs2].min())
		col_max=max(col_idxs1[col_idxs1==col_idxs1].max(), col_idxs2[col_idxs2==col_idxs2].max())
		row_min=min(row_idxs1[row_idxs1==row_idxs1].min(), row_idxs2[row_idxs2==row_idxs2].min())
		row_max=max(row_idxs1[row_idxs1==row_idxs1].max(), row_idxs2[row_idxs2==row_idxs2].max())

		data=ncvar_data[:,row_min:row_max+1,col_min:col_max+1]
		
		out_data11=data[:,row_idxs1[wh]-row_min,col_idxs1[wh]-col_min]
		out_data12=data[:,row_idxs1[wh]-row_min,col_idxs2[wh]-col_min]
		out_data21=data[:,row_idxs2[wh]-row_min,col_idxs1[wh]-col_min]
		out_data22=data[:,row_idxs2[wh]-row_min,col_idxs2[wh]-col_min]

		aux1 = out_data11 + row_wghts[wh]*(out_data21 - out_data11)
		aux2 = out_data12 + row_wghts[wh]*(out_data22 - out_data12)
		res=aux1 + col_wghts[wh]*(aux2 - aux1)

	output_data[:,:,:] = numpy.nan
	output_data[:,wh] = res	
	
	#rescale data
	ncvar_attrs = ncvar_data.ncattrs()
	if ('scale_factor' in ncvar_attrs) and ('add_offset' in ncvar_attrs):
		scale=ncvar_data.scale_factor
		offset=ncvar_data.add_offset
		output_data = offset + (scale*output_data)
	
	rootgrp.close()
	return output_data


#read variable from nc files
def latlon_read_months_hour_lat_lon_nc_point(nc_file_name, varname, lon=None, lat=None, interpolation='nearest'):

	output_data = numpy.zeros((12,24), dtype=numpy.float32)
	
	if (lon is None) or (lat is None):
		logger.error("problem with point coordinates")
		return None
	
	if (nc_file_name is None) or (nc_file_name==''):
		logger.error( "No NC filename set")
		return None
	
	if (interpolation not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolation)
		return None
		
	#print nc_file_name
	if not(os.access(nc_file_name,os.F_OK)):
#		logger.error("unable to find NetCDF file: %s",nc_file_name)
		print("unable to find NetCDF file: %s",nc_file_name)
		return None

	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		
		return None

	ncdims=rootgrp.dimensions
	ncvariabs=rootgrp.variables

	try:
		cols=len(ncdims['longitude'])
		rows=len(ncdims['latitude'])
	except:
		logger.error("one of dimensions longitude, latitude not found in "+ nc_file_name)
		
		rootgrp.close()
		return None

	try:
		ncvar_data=ncvariabs[varname]
		ncvar_data.set_auto_maskandscale(False)
		ncvar_col=ncvariabs['longitude']
		ncvar_row=ncvariabs['latitude']
	except:
		logger.error("one of variables: %s, latitude, longitude, not found in %s" ,varname,  nc_file_name)
		
		rootgrp.close()
		return None
		
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	#this is test whether modulo recalculation of the column coordinate may be done
	#in case the data are global (360 deg.)
#	lon_circular=False
#	epsilon=0.001
#	total_width = col_max - col_min + colres
#	if (total_width > (360-epsilon)) and  (total_width < (360+epsilon)):
#		lon_circular=True
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	
	pxidxs = nc_ubbox.px_idxs_of_latlon(lon, lat, True)
	if pxidxs is None:
		return None
	
	x1, y1, x2, y2, x_wght, y_wght = pxidxs
	#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	if interpolation=='nearest':
		data=ncvar_data[:, y1, x1]
	else:

		px_x1y1=ncvar_data[:,:, y1, x1]
		px_x1y2=ncvar_data[:,:, y2, x1]
		px_x2y1=ncvar_data[:,:, y1, x2]
		px_x2y2=ncvar_data[:,:, y2, x2]
		
		if (px_x1y1 is None) or (px_x1y2 is None) or (px_x1y2 is None) or (px_x2y2 is None):
			return None
		#bilinear interpolation
		aux1=px_x1y1 + (px_x2y1-px_x1y1)*x_wght
		aux2=px_x1y2 + (px_x2y2-px_x1y2)*x_wght
		data=aux1 + (aux2-aux1)*y_wght

	output_data[:,:] = data
	
	#rescale data
	ncvar_attrs = ncvar_data.ncattrs()
	if ('scale_factor' in ncvar_attrs) and ('add_offset' in ncvar_attrs):
		scale=ncvar_data.scale_factor
		offset=ncvar_data.add_offset
		output_data = offset + (scale*output_data)
	
	rootgrp.close()
	return output_data
	

def latlon_read_months_hour_nc(nc_file_name, varname, month=None,hour=None):
	
	if (month is None) or (hour is None):
		logger.error("problem with month and hour data ranges")
		return None
	if (nc_file_name is None) or (nc_file_name==''):
		logger.error( "No NC filename set")
		return None
	if not(os.access(nc_file_name,os.F_OK)):

		print("unable to find NetCDF file: %s",nc_file_name)
		return None
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		
		return None
	ncvariabs=rootgrp.variables
	try:
		ncvar_data=ncvariabs[varname]
		ncvar_data.set_auto_maskandscale(False)
	except:
		logger.error("one of variables: %s, latitude, longitude, not found in %s" ,varname,  nc_file_name)
		
		rootgrp.close()
		return None
	return ncvar_data[month-1,hour,:,:]


#read variable from nc files
def latlon_read_months_lat_lon_nc_point(nc_file_name, varname, lon=None, lat=None, interpolation='nearest'):

	output_data = numpy.zeros((12), dtype=numpy.float32)
	
	if (lon is None) or (lat is None):
		logger.error("problem with point coordinates")
		return None
	
	if (nc_file_name is None) or (nc_file_name==''):
		logger.error( "No NC filename set")
		return None
	
	if (interpolation not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolation)
		return None
		
	#print nc_file_name
	if not(os.access(nc_file_name,os.F_OK)):
#		logger.error("unable to find NetCDF file: %s",nc_file_name)
		print("unable to find NetCDF file: %s",nc_file_name)
		return None

	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		
		return None

	ncdims=rootgrp.dimensions
	ncvariabs=rootgrp.variables

	try:
		cols=len(ncdims['longitude'])
		rows=len(ncdims['latitude'])
	except:
		logger.error("one of dimensions longitude, latitude not found in "+ nc_file_name)
		
		rootgrp.close()
		return None

	try:
		ncvar_data=ncvariabs[varname]
		ncvar_data.set_auto_maskandscale(False)
		ncvar_col=ncvariabs['longitude']
		ncvar_row=ncvariabs['latitude']
	except:
		logger.error("one of variables: %s, latitude, longitude, not found in %s" ,varname,  nc_file_name)
		
		rootgrp.close()
		return None
		
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	
	#this is test whether modulo recalculation of the column coordinate may be done
	#in case the data are global (360 deg.)
#	lon_circular=False
#	epsilon=0.001
#	total_width = col_max - col_min + colres
#	if (total_width > (360-epsilon)) and  (total_width < (360+epsilon)):
#		lon_circular=True
	
	nc_ubbox = latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	
	pxidxs = nc_ubbox.px_idxs_of_latlon(lon, lat, True)
	if pxidxs is None:
		return None
	
	x1, y1, x2, y2, x_wght, y_wght = pxidxs
	#no equal resolution of bboxes or not full overlap - use extended reader with interpolation
	if interpolation=='nearest':
		data=ncvar_data[:, y1, x1]
	else:

		px_x1y1=ncvar_data[:, y1, x1]
		px_x1y2=ncvar_data[:, y2, x1]
		px_x2y1=ncvar_data[:, y1, x2]
		px_x2y2=ncvar_data[:, y2, x2]
		
		if (px_x1y1 is None) or (px_x1y2 is None) or (px_x1y2 is None) or (px_x2y2 is None):
			return None
		#bilinear interpolation
		aux1=px_x1y1 + (px_x2y1-px_x1y1)*x_wght
		aux2=px_x1y2 + (px_x2y2-px_x1y2)*x_wght
		data=aux1 + (aux2-aux1)*y_wght

	output_data[:] = data
	
	#rescale data
	ncvar_attrs = ncvar_data.ncattrs()
	if ('scale_factor' in ncvar_attrs) and ('add_offset' in ncvar_attrs):
		scale=ncvar_data.scale_factor
		offset=ncvar_data.add_offset
		output_data = offset + (scale*output_data)
	
	rootgrp.close()
	return output_data



def horizon_file_name_create(horizon_path='', horizon_file_pattern='horizon_ns%d_ew%d_dir48.nc', longitude=None, latitude=None):
	col, row = get_5x5_seg(longitude,latitude)
	horizon_file_name=horizon_file_pattern % (row, col)
	if horizon_path!='':
		horizon_file_name = os.path.join(horizon_path, horizon_file_name)
	
	return horizon_file_name


def read_horizon_nc_file_lonlat(nc_file_name,var_name='horizons',longit=None, latit=None):
	try:
		rootgrp = netCDF4.Dataset(nc_file_name, 'r')
	except Exception, e:
		#logger.error(str(type(e)) + ": "+ str(e)+" "+nc_file_name)
		return None
	ncdims=rootgrp.dimensions
	try:
		cols=len(ncdims['lon'])
		rows=len(ncdims['lat'])
		dirs=len(ncdims['dir'])
	except Exception, e:
		#logger.error(str(type(e)) + ": "+ str(e))
		rootgrp.close()
		return None
	ncvariabs=rootgrp.variables
	try:
		ncvar_data=ncvariabs[var_name]
		ncvar_data.set_auto_maskandscale(False)
		ncvar_col=ncvariabs['longitude']
		ncvar_row=ncvariabs['latitude']
	except Exception, e:
		#logger.error(str(type(e)) + ": "+ str(e))
		return None
	add_offset=ncvariabs[var_name].add_offset
	scale_factor=ncvariabs[var_name].scale_factor
	col_min, col_max = ncvar_col.valid_range
	row_min, row_max = ncvar_row.valid_range	

	colres=(col_max-col_min)/cols
	rowres=(row_max-row_min)/rows
	col_idx=int(math.floor((longit - col_min) / colres))
	row_idx=int(math.floor((row_max-latit) / rowres))

	if (col_idx < 0) or (col_idx >= cols) or (row_idx < 0) or (row_idx >= rows):
		#print "point location out of nc data range "+ nc_file_name
		rootgrp.close()
		return None
	try:
		Horizons_vector=ncvar_data[:,row_idx,col_idx]
	except Exception, e:
		#logger.error(str(type(e)) + ": "+ str(e))
		rootgrp.close()
		return None
	Horizons_vector = ( Horizons_vector.astype(numpy.float32)  * scale_factor) + add_offset
	rootgrp.close()
	return Horizons_vector


def read_projected_data_for_lonlat_point(ncfile='', var_name='', lon=None, lat=None, interpolation='nearest'):
		
	if (lon is None) or (lat is None):
		logger.error("problem with point coordinates")
		return None
	
	if (interpolation not in ['nearest', 'bilinear']):
		logger.error("unsupported interpolation type: %s",interpolation)
		return None
		
	if (ncfile is None) or (ncfile==''):
		logger.error( "No NC filename set")
		return None

	if not(os.access(ncfile, os.F_OK)):
		logger.warning("Input data NetCDF file %s not found",ncfile)
		return None

	try:
		rootgrp=netCDF4.Dataset(ncfile, 'r')
	except:
		logger.error('Failed to open input data NetCDF %s', ncfile)
		return None
	
	dimDict = rootgrp.dimensions
	varDict = rootgrp.variables

	dkeys=dimDict.keys()
	if not(('row'in dkeys) and ('col' in dkeys)) and not(('lon'in dkeys) and ('lat' in dkeys)) and not(('longitude'in dkeys) and ('latitude' in dkeys)):
		logger.error('Dimensions row, col or lat, lon not found in input data NetCDF %s', ncfile)
		rootgrp.close()
		return None
	if ('row'in dkeys) and ('col' in dkeys):
		nc_height = len(dimDict['row'])
		nc_width = len(dimDict['col'])
		nc_w, nc_e = varDict['col'].valid_range
		nc_s, nc_n = varDict['row'].valid_range
	elif ('latitude'in dkeys) and ('longitude' in dkeys):
		nc_height = len(dimDict['latitude'])
		nc_width = len(dimDict['longitude'])
		nc_w, nc_e = varDict['longitude'].valid_range
		nc_s, nc_n = varDict['latitude'].valid_range
	else:
		nc_height = len(dimDict['lat'])
		nc_width = len(dimDict['lon'])
		nc_w, nc_e = varDict['lon'].valid_range
		nc_s, nc_n = varDict['lat'].valid_range

	#create bbox with avial lowres data based on the size of highres bbox
	#print dkeys
	#print nc_w, nc_e, nc_s, nc_n, nc_width, nc_height, (nc_e-nc_w)/nc_width
	nc_latlon_bbox = latlon.bounding_box(nc_w, nc_e, nc_s, nc_n, nc_width, nc_height, (nc_e-nc_w)/nc_width) #nc bbox
	#print 	nc_latlon_bbox 
	#print lat, lon
	#now finally read the data
	if not(var_name in varDict.keys()):
		logger.error('Variable %s in input data NetCDF %s'%(var_name, ncfile))
		rootgrp.close()
		return None
	
	dimensions = varDict[var_name].ndim
	if (dimensions <2) or (dimensions >4):
		logger.error('Unsuported number of dimensions for variable %s in NetCDF %s'% (var_name, ncfile))
		rootgrp.close()
		return None
	
	if (interpolation== 'nearest') or (interpolation== 'n'):
		try:
			px_x, px_y = nc_latlon_bbox.pixel_coords_of_lonlat(lon,lat)
			if dimensions==2:
				data=varDict[var_name][px_y, px_x]
			elif dimensions==3:
				data=varDict[var_name][:, px_y, px_x]
			else:
				data=varDict[var_name][:, :, px_y, px_x]
		except:
			logger.error('Problem reading input data from NetCDF %s', ncfile)
			rootgrp.close()
			return None
				
	elif (interpolation== 'bilinear') or (interpolation== 'b'):
		px_x1, px_y1, px_x2, px_y2, x_wght, y_wght = nc_latlon_bbox.px_idxs_of_latlon(lon=lon, lat=lat)
		#print px_x1, px_y1, px_x2, px_y2, x_wght, y_wght
		try:
			if dimensions == 2:
				aux1=varDict[var_name][px_y1, px_x1]
				aux2=varDict[var_name][px_y1, px_x2]
				aux3=varDict[var_name][px_y2, px_x1]
				aux4=varDict[var_name][px_y2, px_x2]
			elif dimensions == 3:
				aux1=varDict[var_name][:, px_y1, px_x1]
				aux2=varDict[var_name][:, px_y1, px_x2]
				aux3=varDict[var_name][:, px_y2, px_x1]
				aux4=varDict[var_name][:, px_y2, px_x2]
			else:
				aux1=varDict[var_name][:, :, px_y1, px_x1]
				aux2=varDict[var_name][:, :, px_y1, px_x2]
				aux3=varDict[var_name][:, :, px_y2, px_x1]
				aux4=varDict[var_name][:, :, px_y2, px_x2]
		except:
			logger.error('Problem reading input data from NetCDF %s', ncfile)
			rootgrp.close()
			return None
		aux5a=aux1 + (aux2 - aux1)*x_wght
		aux5b=aux3 + (aux4 - aux3)*x_wght
		data = aux5a + (aux5b - aux5a)*y_wght
			
		del (aux1, aux2, aux3, aux4)
		del (aux5a, aux5b)
	return data



#read variable from nc files
def latlon_get_ubboxfrom_nc(nc_file_name, varname=None):
	if (nc_file_name is None) or (nc_file_name==''):
		logger.error( "No NC filename set")
		return None
	#print nc_file_name
	if not(os.access(nc_file_name,os.F_OK)):
#		logger.error("unable to find NetCDF file: %s",nc_file_name)
		print("unable to find NetCDF file: %s",nc_file_name)
		return None
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		
		return None
	ncdims=rootgrp.dimensions
	ncvariabs=rootgrp.variables
	try:
		cols=len(ncdims['longitude'])
		rows=len(ncdims['latitude'])
	except:
		logger.error("one of dimensions longitude, latitude not found in "+ nc_file_name)
		
		rootgrp.close()
		return None
	try:
		ncvar_col=ncvariabs['longitude']
		ncvar_row=ncvariabs['latitude']
	except:
		logger.error("one of variables: latitude, longitude, not found in %s" ,  nc_file_name)
		
		rootgrp.close()
		return None
		
	col_min = ncvar_col[0]
	col_max = ncvar_col[-1]
	pxsize_x =(col_max - col_min) / float(cols-1)
	col_min -= pxsize_x/2.
	col_max += pxsize_x/2.
	row_min = ncvar_row[:].min()
	row_max = ncvar_row[:].max()
	pxsize_y =(row_max - row_min) / float(rows-1)
	row_min -= pxsize_y/2.
	row_max += pxsize_y/2.
	
	colres=(col_max-col_min)/(cols)
	rowres=(row_max-row_min)/(rows)
	rootgrp.close()
	return latlon.uneven_bounding_box(xmin=col_min, xmax=col_max, ymin=row_min, ymax=row_max, width=cols, height=rows, xresolution=colres, yresolution=rowres)
	

#read boundingbox from nc files
def latlon_get_bboxfrom_nc(nc_file_name, varname=None):
	if (nc_file_name is None) or (nc_file_name==''):
		logger.error( "No NC filename set")
		return None
	#print nc_file_name
	if not(os.access(nc_file_name,os.F_OK)):
#		logger.error("unable to find NetCDF file: %s",nc_file_name)
		print("unable to find NetCDF file: %s",nc_file_name)
		return None
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		
		return None
	dimDict=rootgrp.dimensions
	varDict=rootgrp.variables

	dkeys=dimDict.keys()

	if not(('row'in dkeys) and ('col' in dkeys)) and not(('lon'in dkeys) and ('lat' in dkeys)) and not(('longitude'in dkeys) and ('latitude' in dkeys)):
		logger.error('Dimensions row, col or lat, lon or latitude, longitude not found in input data NetCDF %s', nc_file_name)
		rootgrp.close()
		return None
	
	if ('row'in dkeys) and ('col' in dkeys):
		nc_height = len(dimDict['row'])
		nc_width = len(dimDict['col'])
		nc_w, nc_e = varDict['col'].valid_range
		nc_s, nc_n = varDict['row'].valid_range
	elif ('latitude'in dkeys) and ('longitude' in dkeys):
		nc_height = len(dimDict['latitude'])
		nc_width = len(dimDict['longitude'])
		nc_w = varDict['longitude'][0]
		nc_e = varDict['longitude'][-1] 
		pxsize_x =(nc_e - nc_w) / numpy.double(nc_width-1)
		nc_w -= pxsize_x/2.
		nc_e += pxsize_x/2.
		nc_s = varDict['latitude'][:].min()
		nc_n = varDict['latitude'][:].max()
		pxsize_y =(nc_n - nc_s) / numpy.double(nc_height-1)
		nc_s -= pxsize_y/2.
		nc_n += pxsize_y/2.
	else:
		nc_height = len(dimDict['lat'])
		nc_width = len(dimDict['lon'])
		try:
			nc_w, nc_e = varDict['lon'].valid_range
			nc_s, nc_n = varDict['lat'].valid_range
		except:
			try:
				nc_w = varDict['lon'][0]
				nc_e = varDict['lon'][-1]
				nc_s = varDict['lat'][:].min()
				nc_n = varDict['lat'][:].max()
			except:
				nc_w = varDict['longitude'][0]
				nc_e = varDict['longitude'][-1]
				nc_s = varDict['latitude'][:].min()
				nc_n = varDict['latitude'][:].max()

			pxsize_x =(nc_e - nc_w) / numpy.double(nc_width-1)
			nc_w -= pxsize_x/2.
			nc_e += pxsize_x/2.
			pxsize_y =(nc_n - nc_s) / numpy.double(nc_height-1)
			nc_s -= pxsize_y/2.
			nc_n += pxsize_y/2.
		
	
	colres=(float(nc_e)-float(nc_w))/float(nc_width)
	rootgrp.close()
	return latlon.bounding_box(xmin=nc_w, xmax=nc_e, ymin=nc_s, ymax=nc_n, width=nc_width, height=nc_height, resolution=colres)


#read boundingbox from nc files
def latlon_get_times_nc(nc_file_name):
	if (nc_file_name is None) or (nc_file_name==''):
		logger.error( "No NC filename set")
		return None
	#print nc_file_name
	if not(os.access(nc_file_name,os.F_OK)):
#		logger.error("unable to find NetCDF file: %s",nc_file_name)
		print("unable to find NetCDF file: %s",nc_file_name)
		return None
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'r')
	except:
		logger.error("unable to open NetCDF file: %s",nc_file_name)
		return None

	dimDict=rootgrp.dimensions
	varDict=rootgrp.variables

	dkeys=dimDict.keys()

	if not('time' in dkeys) :
		logger.error('Dimension time not found in input data NetCDF %s', nc_file_name)
		rootgrp.close()
		return None
	
	times = varDict["time"]
	times = netCDF4.num2date(times[:],units=times.units,calendar=times.calendar)
	
	rootgrp.close()
	return times


def get_5x5_seg(lon,lat):
	seg_w,seg_e,seg_s,seg_n,seg_res=-180,180,-90,90,5. #GLOBAL segment region 
	seg_col=int(math.floor((lon-seg_w)/seg_res))
	seg_row=int(math.floor((seg_n - lat)/seg_res))
	return seg_col, seg_row

#create new NetCDF file with [time, lat, lon] dimensions
#	version for days
#
#	data_channel_dict={}
#	data_channel_dict["GHI"] = {}
#	data_channel_dict["GHI"]["datatype"]="NC_FLOAT"
#	data_channel_dict["GHI"]["long_name"]="Global Horizontal Irradiation (GHI) average daily sum"
#	data_channel_dict["GHI"]["cell_methods"]="time: mean (interval: 1 day comment: daily sum) latitude: longitude: mean"
#	data_channel_dict["GHI"]["units"]="kilowatthour/meter2"
#	data_channel_dict["GHI"]["fill_value"]=-9
#	data_channel_dict["GHI"]["missing_value"]=-9
#	data_channel_dict["GHI"]["zlib"]=True
#	data_channel_dict["GHI"]["chunksizes"]=[8,16,16]
#	data_channel_dict["GHI"]["least_significant_digit"]=2 
#	data_channel_dict={}
#	data_channel_dict["GHI"] = {}
#	data_channel_dict["GHI"]["datatype"]="NC_FLOAT"
#	data_channel_dict["GHI"]["long_name"]="Global Horizontal Irradiation (GHI) average daily sum"
#	data_channel_dict["GHI"]["cell_methods"]="time: mean (interval: 1 day comment: daily sum) latitude: longitude: mean"
#	data_channel_dict["GHI"]["units"]="kilowatthour/meter2"
#	data_channel_dict["GHI"]["fill_value"]=-9
#	data_channel_dict["GHI"]["missing_value"]=-9
#	data_channel_dict["GHI"]["zlib"]=True
#	data_channel_dict["GHI"]["chunksizes"]=[8,16,16]
#	data_channel_dict["GHI"]["least_significant_digit"]=2 
def make_TimeD_lat_lon_nc(nc_file_name='', nc_metadata_dict=[], force_overwrite=True, data_channel_dict={}, date_begin=None, date_end=None, nc_extent=None, skip_dimension_check=False):
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}
	datatypes_short = ('i1', 'i2', 'i4', 'f4', 'f8', 'u1', 'u2', 'u4', 'i8', 'u8')

	lat_lon_precission = 12

	if (date_begin is None) or (date_end is None):
		logger.error("missing date_begin date_end")
		return False
	
	time_dim_size = daytimeconv.date2dfb(date_end) -daytimeconv.date2dfb(date_begin)+1 

	#check filename and existance	
	if (nc_file_name is None) or (nc_file_name==''):
		logger.error("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True

	#check types of data variables	
	for chan_name, chan_param_dict in data_channel_dict.iteritems():
		chan_type = chan_param_dict['datatype']
		if chan_type in datatypes_dict.keys():
			chan_param_dict['datatype'] = datatypes_dict[chan_type]
		elif chan_type not in datatypes_short:
			logger.warning("data_type %s of chan %s is not one of predefined NetCDF data types",str(chan_type), chan_name)
			return False

		
	# extent of the NC file - (bounding box)
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height

	if not skip_dimension_check:
		if (xmin < -180) or (xmin > 180):
			logger.warning("xmin outside of range")
			return False
		if (xmax < -180) or (xmax > 180):
			logger.warning("xmax outside of range")
			return False
		if (ymin < -90) or (ymin > 90):
			logger.warning("ymin outside of range")
			return False
		if (ymax < -90) or (ymax > 90):
			logger.warning("ymax outside of range")
			return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False




	# define dimensions
	try:
		dummy=rootgrp.createDimension("latitude", img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s","latitude")
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension("longitude", img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s","longitude")
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('time', time_dim_size)
	except:
		logger.error("unable to create NetCDF dimension: %s",'time')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('nv', 2)
	except:
		logger.error("unable to create NetCDF dimension: %s",'nv')
		rootgrp.close()
		return False
	



	
	
	# coordinate variable row 
	try:
		lat_var=rootgrp.createVariable("latitude",'f8',("latitude",))
	except:
		logger.error("unable to create NetCDF variable: %s","latitude")
		rootgrp.close()
		return False
	#and attributes
	try:
		lat_var.units="degrees_north"
		lat_var.standard_name="latitude"
		lat_var.long_name = "Central latitude of the pixel"
		amin=round(ymin+(resolution/2.), lat_lon_precission)
		amax=round(ymax-(resolution/2.), lat_lon_precission)
		lat_var.valid_range=numpy.array([amin,amax],dtype='float64')
		lat_var.bounds = "latitude_bounds"
	except:
		logger.error("unable to add %s variable attributes","latitude")
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		lon_var=rootgrp.createVariable("longitude",'f8',("longitude",))
	except:
		logger.error("unable to create NetCDF variable: %s","longitude")
		rootgrp.close()
		return False
	#and attributes
	try:
		lon_var.units="degrees_east"
		lon_var.standard_name="longitude"
		lon_var.long_name = "Central longitude of the pixel" 
		amin=round(xmin+(resolution/2.), lat_lon_precission)
		amax=round(xmax-(resolution/2.), lat_lon_precission)
		lon_var.valid_range=numpy.array([amin,amax],dtype='float64')
		lon_var.bounds = "longitude_bounds"
	except:
		logger.error("unable to add %s variable attributes","longitude")
		rootgrp.close()
		return False
	
	
	# coordinate variable time
	try:
		time_var=rootgrp.createVariable('time','f8',('time',))
	except:
		logger.error("unable to create NetCDF variable: %s",'time')
		rootgrp.close()
		return False
	#and attributes
	try:
		time_var.units="days since 1900-01-01"
		time_var.standard_name="time"
		time_var.bounds="time_bounds"
		time_var.long_name="Middle hour of the day" 
		time_var.calendar="gregorian"
	except:
		logger.error("unable to add %s variable attributes",'time')
		rootgrp.close()
		return False

	# variables for bounds of dimension variables
	try:
		lat_bounds_var=rootgrp.createVariable('latitude_bounds','f8',('latitude','nv',))
	except:
		logger.error("unable to create NetCDF variable: %s",'latitude_bounds')
		rootgrp.close()
		return False
	try:
		lon_bounds_var=rootgrp.createVariable('longitude_bounds','f8',('longitude','nv',))
	except:
		logger.error("unable to create NetCDF variable: %s",'longitude_bounds')
		rootgrp.close()
		return False
	try:
		time_bounds_var=rootgrp.createVariable('time_bounds','f8',('time','nv',))
	except:
		logger.error("unable to create NetCDF variable: %s",'time_bounds')
		rootgrp.close()
		return False
	
	
	

	#create dimension variables values
	lons=numpy.arange(xmin+(resolution/2.),xmax,resolution,dtype='float64')
	lons_bounds=numpy.empty((lons.shape[0],2),dtype=lons.dtype)
	lons_bounds[:,0] = lons-(resolution/2.)
	lons_bounds[:,1] = lons+(resolution/2.)
	lons=numpy.around(lons,decimals=lat_lon_precission)
	lons_bounds=numpy.around(lons_bounds,decimals=lat_lon_precission)
	
	lats=numpy.arange(ymax-(resolution/2.),ymin,-resolution,dtype='float64')
	lats_bounds=numpy.empty((lats.shape[0],2),dtype=lats.dtype)
	lats_bounds[:,1] = lats-(resolution/2.)
	lats_bounds[:,0] = lats+(resolution/2.)
	lats=numpy.around(lats,decimals=lat_lon_precission)
	lats_bounds=numpy.around(lats_bounds,decimals=lat_lon_precission)
	
	dates=[]
	dates_bound_low=[]
	dates_bound_up=[]
	aDT_begin = datetime.datetime.combine(date_begin, datetime.time(0,0,0))
	for n in range(0,time_dim_size):
		dates_bound_low.append(aDT_begin+n*datetime.timedelta(hours=24)) 
		dates_bound_up.append(dates_bound_low[n] + datetime.timedelta(hours=24))
		dates.append(dates_bound_low[n] + datetime.timedelta(hours=12))
	dates = netCDF4.date2num(dates,units=time_var.units,calendar=time_var.calendar)
	dates_bounds = numpy.empty((dates.shape[0],2),dtype=dates.dtype)
	dates_bounds[:,0]=netCDF4.date2num(dates_bound_low,units=time_var.units,calendar=time_var.calendar)
	dates_bounds[:,1]=netCDF4.date2num(dates_bound_up,units=time_var.units,calendar=time_var.calendar)

	#fill values for dimension variables
	try:
		lat_var[:]=lats
		lat_bounds_var[:]=lats_bounds
		lon_var[:]=lons
		lon_bounds_var[:]=lons_bounds
		time_var[:]=dates
		time_bounds_var[:]=dates_bounds
#		print lat_var[:][0], lat_var[:][-1], lat_var[:].min(), lat_var[:].max()
#		print lon_var[:][0], lon_var[:][-1], lon_var[:].min(), lon_var[:].max()
	except:
		logger.error("unable to add variable values")
		print sys.exc_info()
		rootgrp.close()
		return False
	
	
	#create data variables	
	for chan_name, chan_param_dict in data_channel_dict.iteritems():

		allowed_create_keywords = [ 'varname', 'datatype', 'zlib' , 'chunksizes', 'least_significant_digit', 'fill_value', 'complevel','dimensions']
		
		data_attributes = {}
		variable_create_keywords = {} 
		for chan_param, chan_param_value in chan_param_dict.iteritems():
			if chan_param in allowed_create_keywords:
				variable_create_keywords[chan_param] = chan_param_value
			else:
				data_attributes[chan_param] = chan_param_value
	
		if variable_create_keywords.has_key('chunksizes'):
			chunksizes=list(variable_create_keywords['chunksizes'])
			chunksizes[0]=min(time_var.shape[0],chunksizes[0])
			chunksizes[1]=min(lat_var.shape[0],chunksizes[1])
			chunksizes[2]=min(lon_var.shape[0],chunksizes[2])
			variable_create_keywords['chunksizes']=chunksizes
	
		if not variable_create_keywords.has_key('dimensions'):
			variable_create_keywords['dimensions'] = ('time','latitude','longitude')

		if not variable_create_keywords.has_key('varname'):
			variable_create_keywords['varname'] = chan_name
		
		try:
			img_var=rootgrp.createVariable(**variable_create_keywords)
		except:
			logger.error("unable to create NetCDF variable: %s",chan_name)
			rootgrp.close()
			return False

		#and attributes
		
		try:
			for chan_attr, chan_attr_value in data_attributes.iteritems():
				img_var.__setattr__(chan_attr, chan_attr_value)
		except:
			logger.error("unable to add %s variable attributes",chan_name)
			rootgrp.close()
			return False



	#global attributes
	for item,value in nc_metadata_dict.iteritems():
		rootgrp.__setattr__(item,value)

	if not nc_metadata_dict.has_key("history"):
		rootgrp.__setattr__("history","Created "+str(datetime.datetime.utcnow()))
	if not nc_metadata_dict.has_key("Conventions"):
		rootgrp.__setattr__("Conventions","CF-1.6")
		

	rootgrp.sync()
	
	rootgrp.close()
	logger.info("NetCDF file %s create OK",nc_file_name) 
	
	return True
	

def make_TimeH_lat_lon_nc(nc_file_name='', nc_metadata_dict=[], force_overwrite=True, data_channel_dict={}, date_begin=None, date_end=None, nc_extent=None, skip_dimension_check=False):
	datatypes_dict = {"NC_BYTE": 'i1', "NC_SHORT": 'i2', "NC_INT": 'i4', "NC_FLOAT": 'f4', "NC_DOUBLE": 'f8', "NC_UBYTE": 'u1', "NC_USHORT": 'u2', "NC_UINT": 'u4', "NC_INT64": 'i8', "NC_UINT64": 'u8'}
	datatypes_short = ('i1', 'i2', 'i4', 'f4', 'f8', 'u1', 'u2', 'u4', 'i8', 'u8')

	lat_lon_precission = 12

	if (date_begin is None) or (date_end is None):
		logger.error("missing date_begin date_end")
		return False
	
	#hourly = num_days*24
	num_days = daytimeconv.date2dfb(date_end) -daytimeconv.date2dfb(date_begin)+1
	time_dim_size = (num_days) * 24 

	#check filename and existance	
	if (nc_file_name is None) or (nc_file_name==''):
		logger.error("empty file name")
		return False
	
	if (os.access(nc_file_name, os.F_OK)):
		if force_overwrite:
			logger.warning("file %s exists and will be overwritten (force_overwrite set to True)",nc_file_name)
		else:
			logger.warning("file %s exists. Skipping. You can overwrite it seting \"force_overwrite=True\"", nc_file_name)
			return True

	#check types of data variables	
	for chan_name, chan_param_dict in data_channel_dict.iteritems():
		chan_type = chan_param_dict['datatype']
		if chan_type in datatypes_dict.keys():
			chan_param_dict['datatype'] = datatypes_dict[chan_type]
		elif chan_type not in datatypes_short:
			logger.warning("data_type %s of chan %s is not one of predefined NetCDF data types",str(chan_type), chan_name)
			return False

		
	# extent of the NC file - (bounding box)
	xmin=nc_extent.xmin
	xmax=nc_extent.xmax
	ymin=nc_extent.ymin
	ymax=nc_extent.ymax
	resolution=nc_extent.resolution
	img_width=nc_extent.width
	img_height=nc_extent.height

	if not skip_dimension_check:
		if (xmin < -180) or (xmin > 180):
			logger.warning("xmin outside of range")
			return False
		if (xmax < -180) or (xmax > 180):
			logger.warning("xmax outside of range")
			return False
		if (ymin < -90) or (ymin > 90):
			logger.warning("ymin outside of range")
			return False
		if (ymax < -90) or (ymax > 90):
			logger.warning("ymax outside of range")
			return False
	if (xmin>xmax):
		logger.warning("xmax < xmin")
		return False
	if (ymin>ymax):
		logger.warning("ymax < ymin")
		return False
	
	#create nc file
	if (force_overwrite):
		a_clobber=True
	else:
		a_clobber=False
	
	try:
		rootgrp=netCDF4.Dataset(nc_file_name, 'w', clobber=a_clobber, format='NETCDF4')
	except:
		logger.error("unable to create NetCDF file: %s",nc_file_name)
		return False




	# define dimensions
	try:
		dummy=rootgrp.createDimension("latitude", img_height)
	except:
		logger.error("unable to create NetCDF dimension: %s","latitude")
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension("longitude", img_width)
	except:
		logger.error("unable to create NetCDF dimension: %s","longitude")
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('time', time_dim_size)
	except:
		logger.error("unable to create NetCDF dimension: %s",'time')
		rootgrp.close()
		return False
	try:
		dummy=rootgrp.createDimension('nv', 2)
	except:
		logger.error("unable to create NetCDF dimension: %s",'nv')
		rootgrp.close()
		return False
	



	
	
	# coordinate variable row 
	try:
		lat_var=rootgrp.createVariable("latitude",'f8',("latitude",))
	except:
		logger.error("unable to create NetCDF variable: %s","latitude")
		rootgrp.close()
		return False
	#and attributes
	try:
		lat_var.units="degrees_north"
		lat_var.standard_name="latitude"
		lat_var.long_name = "Central latitude of the pixel"
		amin=round(ymin+(resolution/2.), lat_lon_precission)
		amax=round(ymax-(resolution/2.), lat_lon_precission)
		lat_var.valid_range=numpy.array([amin,amax],dtype='float64')
		lat_var.bounds = "latitude_bounds"
	except:
		logger.error("unable to add %s variable attributes","latitude")
		rootgrp.close()
		return False
		
	# coordinate variable col
	try:
		lon_var=rootgrp.createVariable("longitude",'f8',("longitude",))
	except:
		logger.error("unable to create NetCDF variable: %s","longitude")
		rootgrp.close()
		return False
	#and attributes
	try:
		lon_var.units="degrees_east"
		lon_var.standard_name="longitude"
		lon_var.long_name = "Central longitude of the pixel" 
		amin=round(xmin+(resolution/2.), lat_lon_precission)
		amax=round(xmax-(resolution/2.), lat_lon_precission)
		lon_var.valid_range=numpy.array([amin,amax],dtype='float64')
		lon_var.bounds = "longitude_bounds"
	except:
		logger.error("unable to add %s variable attributes","longitude")
		rootgrp.close()
		return False
	
	
	# coordinate variable time
	try:
		time_var=rootgrp.createVariable('time','f8',('time',))
	except:
		logger.error("unable to create NetCDF variable: %s",'time')
		rootgrp.close()
		return False
	#and attributes
	try:
		time_var.units="hours since 1900-01-01"
		time_var.standard_name="time"
		time_var.bounds="time_bounds"
		time_var.long_name="Middle minute of the hour" 
		time_var.calendar="gregorian"
	except:
		logger.error("unable to add %s variable attributes",'time')
		rootgrp.close()
		return False

	# variables for bounds of dimension variables
	try:
		lat_bounds_var=rootgrp.createVariable('latitude_bounds','f8',('latitude','nv',))
	except:
		logger.error("unable to create NetCDF variable: %s",'latitude_bounds')
		rootgrp.close()
		return False
	try:
		lon_bounds_var=rootgrp.createVariable('longitude_bounds','f8',('longitude','nv',))
	except:
		logger.error("unable to create NetCDF variable: %s",'longitude_bounds')
		rootgrp.close()
		return False
	try:
		time_bounds_var=rootgrp.createVariable('time_bounds','f8',('time','nv',))
	except:
		logger.error("unable to create NetCDF variable: %s",'time_bounds')
		rootgrp.close()
		return False
	
	
	

	#create dimension variables values
	lons=numpy.arange(xmin+(resolution/2.),xmax,resolution,dtype='float64')
	lons_bounds=numpy.empty((lons.shape[0],2),dtype=lons.dtype)
	lons_bounds[:,0] = lons-(resolution/2.)
	lons_bounds[:,1] = lons+(resolution/2.)
	lons=numpy.around(lons,decimals=lat_lon_precission)
	lons_bounds=numpy.around(lons_bounds,decimals=lat_lon_precission)
	
	lats=numpy.arange(ymax-(resolution/2.),ymin,-resolution,dtype='float64')
	lats_bounds=numpy.empty((lats.shape[0],2),dtype=lats.dtype)
	lats_bounds[:,1] = lats-(resolution/2.)
	lats_bounds[:,0] = lats+(resolution/2.)
	lats=numpy.around(lats,decimals=lat_lon_precission)
	lats_bounds=numpy.around(lats_bounds,decimals=lat_lon_precission)
	
	hours=[]
	hours_bound_low=[]
	hours_bound_up=[]
	aDT_begin = datetime.datetime.combine(date_begin, datetime.time(0,0,0))
	for d in range(0,num_days):
		aux_DT = aDT_begin+d*datetime.timedelta(hours=24)
		for h in range(0,23+1):
			hours_bound_low.append(aux_DT + h*datetime.timedelta(hours=1)) 
			hours_bound_up.append(hours_bound_low[-1] + datetime.timedelta(hours=1))
			hours.append(hours_bound_low[-1] + datetime.timedelta(minutes=30))
			
	hours = netCDF4.date2num(hours,units=time_var.units,calendar=time_var.calendar)
	hours_bounds = numpy.empty((hours.shape[0],2),dtype=hours.dtype)
	hours_bounds[:,0]=netCDF4.date2num(hours_bound_low,units=time_var.units,calendar=time_var.calendar)
	hours_bounds[:,1]=netCDF4.date2num(hours_bound_up,units=time_var.units,calendar=time_var.calendar)

	#fill values for dimension variables
	try:
		lat_var[:]=lats
		lat_bounds_var[:]=lats_bounds
		lon_var[:]=lons
		lon_bounds_var[:]=lons_bounds
		time_var[:]=hours
		time_bounds_var[:]=hours_bounds
#		print lat_var[:][0], lat_var[:][-1], lat_var[:].min(), lat_var[:].max()
#		print lon_var[:][0], lon_var[:][-1], lon_var[:].min(), lon_var[:].max()
	except:
		logger.error("unable to add variable values")
		print sys.exc_info()
		rootgrp.close()
		return False
	
	
	#create data variables	
	for chan_name, chan_param_dict in data_channel_dict.iteritems():

		allowed_create_keywords = [ 'varname', 'datatype', 'zlib' , 'chunksizes', 'least_significant_digit', 'fill_value', 'complevel','dimensions']
		
		data_attributes = {}
		variable_create_keywords = {} 
		for chan_param, chan_param_value in chan_param_dict.iteritems():
			if chan_param in allowed_create_keywords:
				variable_create_keywords[chan_param] = chan_param_value
			else:
				data_attributes[chan_param] = chan_param_value
	
		if variable_create_keywords.has_key('chunksizes'):
			chunksizes=list(variable_create_keywords['chunksizes'])
			chunksizes[0]=min(time_var.shape[0],chunksizes[0])
			chunksizes[1]=min(lat_var.shape[0],chunksizes[1])
			chunksizes[2]=min(lon_var.shape[0],chunksizes[2])
			variable_create_keywords['chunksizes']=chunksizes
	
		if not variable_create_keywords.has_key('dimensions'):
			variable_create_keywords['dimensions'] = ('time','latitude','longitude')

		if not variable_create_keywords.has_key('varname'):
			variable_create_keywords['varname'] = chan_name
		
		try:
			img_var=rootgrp.createVariable(**variable_create_keywords)
		except:
			logger.error("unable to create NetCDF variable: %s",chan_name)
			rootgrp.close()
			return False

		#and attributes
		
		try:
			for chan_attr, chan_attr_value in data_attributes.iteritems():
				img_var.__setattr__(chan_attr, chan_attr_value)
		except:
			logger.error("unable to add %s variable attributes",chan_name)
			rootgrp.close()
			return False



	#global attributes
	for item,value in nc_metadata_dict.iteritems():
		rootgrp.__setattr__(item,value)

	if not nc_metadata_dict.has_key("history"):
		rootgrp.__setattr__("history","Created "+str(datetime.datetime.utcnow()))
	if not nc_metadata_dict.has_key("Conventions"):
		rootgrp.__setattr__("Conventions","CF-1.6")
		

	rootgrp.sync()
	
	rootgrp.close()
	logger.info("NetCDF file %s create OK",nc_file_name) 
	
	return True
	


def write_data_to_nc(nc_file_name,var_name, data, position_indexes={} ):
	
	#open NC for reading/writting
	rootgrpout = netCDF4.Dataset(nc_file_name, 'r+')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	
	if var_name not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (var_name, nc_file_name))
		rootgrpout.close()
		return False
	data_var=varDictout[var_name]
	
	data_var_dims = data_var.dimensions
	
	#check requested  dimensions and indexes
	for key in position_indexes.keys():
		if key not in data_var_dims:
			logger.error('Index dimension %s not found in variable %s in nc %s . Skipping' % (key, var_name, nc_file_name))
			rootgrpout.close()
			return False
		idx_b, idx_e = position_indexes[key]
		if (idx_b < 0) or (idx_e >= len(dimDictout[key])): 
			logger.error('Indices %d, %d for %s dimension in nc %s out of dimension size 0, %d. Skipping' % (idx_b,idx_e, key, nc_file_name, len(dimDictout[key])))
			rootgrpout.close()
			return False
		
	#create final slicelist		 
	slice_list=[]
	for data_var_dim in data_var_dims:
		if data_var_dim in position_indexes.keys():
			slice_list.append(slice(position_indexes[data_var_dim][0],position_indexes[data_var_dim][1]+1))
		else:
			slice_list.append(slice(0,len(dimDictout[data_var_dim])))
			
	#check if size of data array fits NC indexes
	if data.ndim != len(slice_list):
		logger.error('Number of dimensions of data to write (%d) differs from dimensions (%d) of NC variable %s in nc %s . Skipping' % (  data.ndim, len(slice_list), var_name, nc_file_name))
		rootgrpout.close()
		return False
	for i in range(0,data.ndim):
		if data.shape[i] != (slice_list[i].stop - slice_list[i].start):
			logger.error('Size of %s dimension of data (%d) differs from nc write slice size (%d) of NC variable %s in nc %s . Skipping' % (  data_var_dims[i], data.shape[i], (slice_list[i].stop - slice_list[i].start), var_name, nc_file_name))
			rootgrpout.close()
			return False
			
	#write data	 
	data_var[slice_list] = data
	
	rootgrpout.close()
	
	return True


def read_data_from_nc(nc_file_name,var_name, position_indexes={} ):
	
	#open NC for reading/writting
	rootgrpout = netCDF4.Dataset(nc_file_name, 'r')
	dimDictout=rootgrpout.dimensions
	varDictout=rootgrpout.variables
	
	if var_name not in varDictout.keys():
		logger.error('No variable for %s found in nc %s . Skipping' % (var_name, nc_file_name))
		rootgrpout.close()
		return False
	data_var=varDictout[var_name]
	
	data_var_dims = data_var.dimensions
	
	#check requested  dimensions and indexes
	for key in position_indexes.keys():
		if key not in data_var_dims:
			logger.error('Index dimension %s not found in variable %s in nc %s . Skipping' % (key, var_name, nc_file_name))
			rootgrpout.close()
			return False
		idx_b, idx_e = position_indexes[key]
		if (idx_b < 0) or (idx_e >= len(dimDictout[key])): 
			logger.error('Indices %d, %d for %s dimension in nc %s out of dimension size 0, %d. Skipping' % (idx_b,idx_e, key, nc_file_name, len(dimDictout[key])))
			rootgrpout.close()
			return False
		
	#create final slicelist		 
	slice_list=[]
	for data_var_dim in data_var_dims:
		if data_var_dim in position_indexes.keys():
			slice_list.append(slice(position_indexes[data_var_dim][0],position_indexes[data_var_dim][1]+1))
		else:
			slice_list.append(slice(0,len(dimDictout[data_var_dim])))
			
	#read data	 
	data = data_var[slice_list]
	
	rootgrpout.close()
	
	return data



def example_create_NC_timeDlatlon():
	nc_file_name= '/tmp/testD.nc'
	force_overwrite=True
	date_begin=daytimeconv.ymd2date(2014, 1, 1)
	date_end=daytimeconv.ymd2date(2014, 1, 31)
	xmin, xmax, ymin, ymax, res = -6, -2, 55, 58, 5./60.
	nc_extent=latlon.bounding_box(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax, width=int(math.floor((xmax-xmin)/res)),height=int(math.floor((ymax-ymin)/res)),resolution=res )

	data_channel_dict={}
	data_channel_dict["GHI"] = {}
	data_channel_dict["GHI"]["datatype"]="NC_FLOAT"
	data_channel_dict["GHI"]["long_name"]="Global Horizontal Irradiation (GHI) average daily sum"
	data_channel_dict["GHI"]["cell_methods"]="time: sum (interval: 1 day comment: daily sum) latitude: longitude: mean"
	data_channel_dict["GHI"]["units"]="kilowatthour/meter2"
	data_channel_dict["GHI"]["fill_value"]=-9
	data_channel_dict["GHI"]["missing_value"]=numpy.array((-9.),dtype=numpy.float32)
	data_channel_dict["GHI"]["zlib"]=True
	data_channel_dict["GHI"]["chunksizes"]=[8,16,16]
	data_channel_dict["GHI"]["least_significant_digit"]=3 


	nc_metadata_dict={}
	nc_metadata_dict["title"]="SolarGIS solar radiation data"
	nc_metadata_dict["institution"]="GeoModel Solar s.r.o."
	nc_metadata_dict["source"]="SolarGIS solar radiation model"
#	nc_metadata_dict["history"]=""
	nc_metadata_dict["references"]="http://geomodelsolar.eu/"
	nc_metadata_dict["comments"]="Daily time series of Global Horizontal Irradiation"
#	nc_metadata_dict["Conventions"]="CF-1.6"

	res = make_TimeD_lat_lon_nc(nc_file_name=nc_file_name, nc_metadata_dict=nc_metadata_dict, force_overwrite=force_overwrite, data_channel_dict=data_channel_dict, date_begin=date_begin, date_end=date_end, nc_extent=nc_extent, skip_dimension_check=False)
	print 'create file',res
	
	
	times = latlon_get_times_nc(nc_file_name)
	bbox = latlon_get_bboxfrom_nc(nc_file_name)
	print 'bbox', bbox
	print 'number of times', len(times)
	
	data = numpy.random.rand(31, 30, 38)
	data = numpy.round(data,2)
	position_indexes={'latitude':(0,29),'longitude':(10,47)}
#	position_indexes={}
	var_name='GHI'
	res = write_data_to_nc(nc_file_name,var_name, data, position_indexes)
	print 'write file',res
	data2 = read_data_from_nc(nc_file_name,var_name, position_indexes)
	print data[:,0,0]
	print data2[:,0,0]




def example_create_NC_timeHlatlon():
	nc_file_name= '/tmp/testH.nc'
	force_overwrite=True
	date_begin=daytimeconv.ymd2date(2014, 1, 1)
	date_end=daytimeconv.ymd2date(2014, 1, 31)
	xmin, xmax, ymin, ymax, res = 32.5, 36.5, -17.5, -9.0, 0.1
	
	nc_extent=latlon.bounding_box(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax, width=int(math.floor((xmax-xmin)/res)),height=int(math.floor((ymax-ymin)/res)),resolution=res )
	
	data_channel_dict={}
	data_channel_dict["GHI"] = {}
	data_channel_dict["GHI"]["datatype"]="NC_FLOAT"
	data_channel_dict["GHI"]["long_name"]="Global Horizontal Irradiation (GHI)"
	data_channel_dict["GHI"]["cell_methods"]="time: sum (interval: 1 hour comment: hourly sum) latitude: longitude: mean"
	data_channel_dict["GHI"]["units"]="watthour/meter2"
	data_channel_dict["GHI"]["fill_value"]=-9.
	data_channel_dict["GHI"]["missing_value"]=numpy.array((-9.),dtype=numpy.float32)
	data_channel_dict["GHI"]["zlib"]=True
	data_channel_dict["GHI"]["chunksizes"]=[24,16,16]
	data_channel_dict["GHI"]["least_significant_digit"]=1 


	nc_metadata_dict={}
	nc_metadata_dict["title"]="SolarGIS solar radiation data"
	nc_metadata_dict["institution"]="GeoModel Solar s.r.o."
	nc_metadata_dict["source"]="SolarGIS solar radiation model"
#	nc_metadata_dict["history"]=""
	nc_metadata_dict["references"]="http://geomodelsolar.eu/"
	nc_metadata_dict["comments"]="Hourly time series of Global Horizontal Irradiation"
#	nc_metadata_dict["Conventions"]="CF-1.6"

	res = make_TimeH_lat_lon_nc(nc_file_name=nc_file_name, nc_metadata_dict=nc_metadata_dict, force_overwrite=force_overwrite, data_channel_dict=data_channel_dict, date_begin=date_begin, date_end=date_end, nc_extent=nc_extent, skip_dimension_check=False)
	print 'create file',res
	
	times = latlon_get_times_nc(nc_file_name)
	bbox = latlon_get_bboxfrom_nc(nc_file_name)
	print 'bbox', bbox
	print 'number of times', len(times), times[:2],times[-2:]
	
	
	data = numpy.random.rand(744, 85, 40 )
	data = numpy.round(data*100,1)
#	position_indexes={'latitude':(0,29),'longitude':(10,47)}
	position_indexes={}
	var_name='GHI'
	res = write_data_to_nc(nc_file_name,var_name, data, position_indexes)
	print 'write file',res
	data2 = read_data_from_nc(nc_file_name,var_name, position_indexes)
	print data[0:10,0,0]
	print data2[0:10,0,0]

		

def test():
	lat,lon = 44.99864, -0.688407
	lat,lon = 45.000000, -0.688407
	print get_5x5_seg(lon,lat)


if __name__=="__main__":
#	test()
#	example_create_NC_timeDlatlon()
	example_create_NC_timeHlatlon()
	pass