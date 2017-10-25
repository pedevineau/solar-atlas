#! /usr/bin/env python
'''
Created on Jan 13, 2011
rechunking NC data

@author: tomas
'''

print 'TO BE DONE - INPUT from command line'
print 'TO BE DONE - change of other parameters e.g. complevel, shuffle'
import netCDF4
import math



def get_total_variable_size(var):
	dims=var.shape
	if len(dims)==1:
		elements=dims[0]
	else:
		elements=dims[0]
		for d in dims[1:]: elements*=d
	return elements*var.dtype.itemsize
	
def reorder_variable_names(variab, dimens):
	vars_dims=[]
	vars_others=[]
	for var_name in variab:
		if var_name in dimens:
			vars_dims.append(var_name)
		else:
			vars_others.append(var_name)
	return vars_dims+vars_others
	
	
def copy_netcdf2(in_file, out_file, out_rewrite=False, out_file_format=None, chunks_dict={}, least_significant_digit_dict={}):
	'''
	copy netCDF file to new netCDF file
	in_file, out_file
	out_rewrite - True if existing file can be rewritten
	out_file_format -  if format to be changed (eg. to older NC formats see. NetCDF documentation)
	chunks_dict - dictionary with new chunks to be used for each variable chunks_dict={'tcwv':[1024,64,64]}
	'''

	if in_file==out_file:
		print 'Joking? in and out files are the same'
		exit()
		
	rootgrp_in=netCDF4.Dataset(in_file, 'r')
	
	if out_file_format is None:
		out_file_format=rootgrp_in.file_format
	
	attributes=rootgrp_in.ncattrs()
	dimensions=rootgrp_in.dimensions
	variables=rootgrp_in.variables
	
	rootgrp_out=netCDF4.Dataset(out_file, 'w', clobber=out_rewrite, format=out_file_format)
	
	print '***global attributes***'
	for attrib in attributes:
		print attrib, rootgrp_in.__getattr__(attrib)
		rootgrp_out.__setattr__(attrib,rootgrp_in.__getattr__(attrib))
	
	print '***dimensions***'
	for dimens_name in dimensions.keys():
		dimens_length = len(dimensions[dimens_name])
		print dimens_name, dimens_length
		rootgrp_out.createDimension(dimens_name, size=dimens_length)
	
	print '***variables***'
	for variab_name in variables.keys():
		var=variables[variab_name]
		dtype=var.dtype
		dimens=var.dimensions
		endian=var.endian()
		fill_value=None
		if '_FillValue' in var.ncattrs():
			fill_value=var.__getattr__('_FillValue')
		
		print variab_name, var.dtype, var.dimensions,  var.endian(), fill_value
		
		chunks=None
		if variab_name in chunks_dict.keys():
			chunks_user=chunks_dict[variab_name]
			if len(chunks_user) != len(dimens):
					print 'WARNING wrong chunks for',variab_name
			else:
					for ch_idx in range(0,len(chunks_user)):
						dim_len=len(dimensions[dimens[ch_idx]])
						if dim_len>0:
							chunks_user[ch_idx]=min(len(dimensions[dimens[ch_idx]]), chunks_user[ch_idx])
					chunks=chunks_user
					print '   using chunks ',chunks ,' for variable',  variab_name

		least_significant_digit=None
		if variab_name in least_significant_digit_dict.keys():
			least_significant_digit=least_significant_digit_dict[variab_name]
		
		
		if (chunks is None) or (chunks == 'contiguous'):
			out_var=rootgrp_out.createVariable(variab_name, dtype, dimensions=dimens, contiguous=True, endian=endian, fill_value=fill_value, least_significant_digit=least_significant_digit)
		else:
			out_var=rootgrp_out.createVariable(variab_name, dtype, dimensions=dimens, zlib=True, complevel=6, chunksizes=chunks, endian=endian, fill_value=fill_value, least_significant_digit=least_significant_digit)
			
		for attrib in var.ncattrs():
			if attrib=='_FillValue':
					continue
			print '   ',attrib, var.__getattr__(attrib)
			out_var.__setattr__(attrib,var.__getattr__(attrib))
		
	print '***data***'
	
	
	variables_out=rootgrp_out.variables
	variables_list=reorder_variable_names(variables.keys(), dimensions.keys())
	for variab_name in variables_list:
		var=variables[variab_name]
		var_out=variables_out[variab_name]
		
		print 'variable: %s, dimensions: %d, shape %s, item size: %d, total size: %d' % (variab_name,  var.ndim, str(var.shape), var.dtype.itemsize, get_total_variable_size(var))
		
		if get_total_variable_size(var)<= (1024*1024*1024):
			print '	copying whole array a once'
			var_out[:]=var[:]
		else:
			print '	variable size exceedes 1GB: copying by chunks'
			#decide step in dim 0 used to segment data for reading
			d0_step = (var.shape[0]/(get_total_variable_size(var)/(1024*1024*1024)))
			#align with chung sizes
			if variab_name in chunks_dict.keys():
				d0_chunk_size=chunks_dict[variab_name][0]
				d0_step = d0_chunk_size*int(round(d0_step/d0_chunk_size))
			d0_step=min(var.shape[0], max(1,d0_step))
			d0_steps=int(math.ceil(var.shape[0]/float(d0_step)))
			for s in range(0, d0_steps):
				d0_min=s*d0_step
				d0_max=min(var.shape[0],(d0_min+d0_step))
				print '	 copying part in dim0:%d-%d'%(d0_min, d0_max)
				var_out[d0_min:d0_max]=var[d0_min:d0_max]
				#else:
					#var_out[d0_min:d0_max,:]=var[d0_min:d0_max,:]

	rootgrp_out.close()
	rootgrp_in.close()
	print 'DONE'
	exit()




if __name__ == "__main__":
		
	in_file='/home3/CFSR/netcdf/CFSR_rh_2002.nc'
	out_file='/home3/CFSR/netcdf/CFSR_rh_2002.nc2'
	
	mdl_chunks=[16,4,16,16]
	chunks_dict={'rh': mdl_chunks}
	least_significant_digit_dict={'rh': 1}
	copy_netcdf2(in_file, out_file, chunks_dict=chunks_dict, least_significant_digit_dict=least_significant_digit_dict, out_rewrite=True)
		
	print 'DONE'
	