#! /usr/bin/env python
'''
Created on Aug 17, 2009

@author: tomas
'''
import math
import numpy


class bounding_box(object):
	def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None, width=None, height=None, resolution=None):
		'''
		TODO: change width to cols
		width - number of columns in px
		height - number of rows in px
		xmin, xmax, ymin, ymax - out boundary of the bounding box (notcenters of pixels) in geographical coordinates
		resolution - pixel size in geographical coordinates
		'''
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		if width is not None:
			self.width = int(width)
		else:
			self.width = None
		if height is not None:
			self.height = int(height)
		else:
			self.height = None
		self.resolution = resolution

	def __repr__(self):
		return "%.12f %.12f %.12f %.12f %d %d %.12f" % (self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.resolution)

	def intersects(self, bbox=None):
		'''
		true/false - has overlap with other bounding box
		exclude_boundary - if overlap is only on boundary (no aerial overlap) then terurn None
		'''
		if bbox is None:
			return False
		if (self.xmax <= bbox.xmin) or (self.xmin >= bbox.xmax) or (self.ymax <= bbox.ymin) or (self.ymin >= bbox.ymax):
			return False
			
		return True

	def contains(self, bbox=None):
		'''
		true/false - has overlap with other bounding box
		'''
		epsilon=min(self.resolution, bbox.resolution)/10000.
		
		if bbox is None:
			return False
		if (self.xmax >= bbox.xmax-epsilon) and (self.xmin <= bbox.xmin+epsilon) and (self.ymax >= bbox.ymax-epsilon) and (self.ymin <= bbox.ymin+epsilon):
			return True
		return False


	def move(self, lon_shift=0.0, lat_shift=0.0):
		'''
		returns new bounding box moved to new position by lon_shift, lat_shift
		lon_shift - shift in longitude
		lat_shift - shift in latitude
		'''
		out_bbox = bounding_box(self.xmin+lon_shift, self.xmax+lon_shift, self.ymin+lat_shift, self.ymax+lat_shift, self.width, self.height, self.resolution)
		return out_bbox


	def intersect(self, bbox=None, epsilon_rate=0.0001):
		'''
		return intersection (bbox) of two bounding_boxes
		'''
		epsilon=epsilon_rate*self.resolution
		if bbox is None:
			return None
		if (self.xmax < bbox.xmin) or (self.xmin > bbox.xmax):
			return None
		if (self.ymax < bbox.ymin) or (self.ymin > bbox.ymax):
			return None
		
		temp_xmin=max(self.xmin, bbox.xmin)
		temp_xmax=min(self.xmax, bbox.xmax)
		temp_ymin=max(self.ymin, bbox.ymin)
		temp_ymax=min(self.ymax, bbox.ymax)
		out_xmin=max(self.xmin, self.resolution * math.floor((temp_xmin/self.resolution)+epsilon))
		out_xmax=min(self.xmax, self.resolution * math.ceil((temp_xmax/self.resolution)-epsilon))
		out_ymin=max(self.ymin, self.resolution * math.floor((temp_ymin/self.resolution)+epsilon))
		out_ymax=min(self.ymax, self.resolution * math.ceil((temp_ymax/self.resolution)-epsilon))
		
		out_height = int(round((out_ymax - out_ymin)/self.resolution))
		out_width = int(round((out_xmax - out_xmin)/self.resolution))
		if (out_height < 1) or (out_width < 1):
			return None

		out_bbox = bounding_box(out_xmin, out_xmax, out_ymin, out_ymax, out_width, out_height, self.resolution)
		
		return out_bbox

	def equals(self, bbox=None):
		'''
		test if the bounding boxes are equal
		returns: True/False
		'''
		if bbox is None:
			return False
		result=True
		result &= (self.xmin == bbox.xmin)
		result &= (self.xmax == bbox.xmax)
		result &= (self.ymin == bbox.ymin)
		result &= (self.ymax == bbox.ymax)
		result &= (self.width == bbox.width)
		result &= (self.height == bbox.height)
		result &= (self.resolution == bbox.resolution)
		return result


	def equal_resolution(self, bbox=None, epsilon=0.0001):
		'''
		test if the bounding boxes are equal
		returns: True/False
		'''
		if bbox is None:
			return False
		result=True
		eps=self.resolution * epsilon
		result &= ((self.resolution-eps) < bbox.resolution) and ((self.resolution+eps) > bbox.resolution)
		return result

	
	def buffer_px(self, px=0):
		'''
		enlarge the bounding box by number of pixels
		returns: bbox
		'''
		px=int(math.floor(px))
		if px==0:
			return self
		
		out_xmin=self.xmin - (self.resolution*px)
		out_xmax=self.xmax + (self.resolution*px) 
		out_ymin=self.ymin - (self.resolution*px)
		out_ymax=self.ymax + (self.resolution*px) 
		
		out_height = self.height + (2*px)
		out_width = self.width + (2*px)
		
		if (out_height < 1) or (out_width < 1):
			return None
		
		out_bbox = bounding_box(out_xmin, out_xmax, out_ymin, out_ymax, out_width, out_height, self.resolution)
		
		return out_bbox

		
	def buffer_size(self, size=0):
		'''
		enlarge the bounding box by buffer size in geographical coordinates
		rounds to pixel
		returns: bbox
		'''
		if size==0:
			return self
		px=int(math.ceil(size/self.resolution))
		return self.buffer_px(px=px)

		
	def pixel_coords_of_lonlat(self, lon=None, lat=None, lon_circular=False):
		'''
		returns pixel indexes of (nearest) point with lon, lat in bounding box
		inputs:
		lon, lat - longitude and latitude of point
		lon_circular [True/False] - work with longitude as circular (resolve different definition 
									of circular coordinate system longitudes in -180,180 or 0,360 range)  
		returns:
		px_x, px_y
		'''
		if (lon is None) or (lat is None):
			return None
		if ((lon < self.xmin) or (lon > self.xmax)) and not(lon_circular):
			return None
		if (lat < self.ymin) or (lat > self.ymax):
			return None
			
		px_x = int(math.floor((lon-self.xmin)/self.resolution))
		px_y = min(self.height-1, max(0, int(math.floor((self.ymax-lat)/self.resolution))))
		
		if (px_x < 0) & lon_circular:
			px_x+=self.width
		if (px_x > self.width-1) & lon_circular:
			px_x-=self.width

		if (px_x < 0) | (px_x > (self.width-1)):
			return None
		
		return px_x, px_y

	
	def px_idxs_of_latlon(self, lon=None, lat=None, lon_circular=False):
		'''
		returns closest pixel indexes and weights of points (bilinear) with lon, lat in bounding box
		inputs:
		lon, lat - longitude and latitude of point
		lon_circular [True/False] - work with longitude as circular (resolve different definition 
									of circular coordinate system longitudes in -180,180 or 0,360 range)  
		returns:
		x1, y1, x2, y2, x_wght, y_wght
		'''
		if (lon is None) or (lat is None):
			return None
		if ((lon < self.xmin) or (lon > self.xmax)) and not(lon_circular):
			return None
		if (lat < self.ymin) or (lat > self.ymax):
			return None

		position = (lon - self.xmin )/self.resolution
		x1 = int(math.floor(position))
		x_wght=(x1+0.5)-position
		if (x_wght > 0):
			x2 = x1 - 1
		else:
			x_wght = -x_wght
			x2 = x1 + 1

		position = (self.ymax - lat)/self.resolution
		y1 = int(math.floor(position))
		y_wght=(y1+0.5)-position
		if (y_wght > 0):
			y2 = y1 - 1
		else:
			y_wght = -y_wght
			y2 = y1 + 1


		#apply circular 'corection' (move coordinate to 0,self.width-1 interval)
		if (x1 < 0) & lon_circular:
			x1+=self.width
		if (x1 > self.width-1) & lon_circular:
			x1-=self.width
		if (x1 < 0) | (x1 > (self.width-1)):
			return None
		if (x2 < 0) & lon_circular:
			x2+=self.width
		if (x2 > self.width-1) & lon_circular:
			x2-=self.width
		if (x2 == -1):
			x2 = 0
		if (x2 == (self.width)):
			x2=self.width-1
		if (x2 < 0) | (x2 > (self.width-1)):
			return None

		return (x1, y1, x2, y2, x_wght, y_wght)
	
	
	def pixel_coords_of_bbox(self, bbox=None, epsilon_rate=0.001):
		'''
		returns px coordinates of other bounding box 
		'''
		epsilon = epsilon_rate* min(self.resolution,bbox.resolution)
		if not(self.intersects(bbox)):
			return None
		px_xmin = max(0, int(math.floor(((bbox.xmin-self.xmin)/self.resolution)+epsilon)))
		px_ymin = max(0, int(math.floor(((self.ymax-bbox.ymax)/self.resolution)+epsilon)))
		if abs(self.resolution - bbox.resolution) < epsilon :
			px_xmax = int(px_xmin + bbox.width - 1)
			px_ymax = int(px_ymin + bbox.height - 1)
		else:
			px_xmax = int(min(self.width -1, int(math.ceil(((bbox.xmax-self.xmin)/self.resolution)-epsilon))))
			px_ymax = int(min(self.height -1, int(math.ceil(((bbox.ymax-self.ymin)/self.resolution)-epsilon))))
		return px_xmin, px_xmax, px_ymin, px_ymax

	
	def longitudes(self, px_order=True, array2d=False, degrees=True):
		'''
		returns 1D array representing longitudes of centers of pixels
		if  array2d - returns array [height, width], otherwise [width]
		'''
		longs=numpy.arange(self.xmin+(self.resolution/2),self.xmax,self.resolution,dtype='float64')
		if px_order:
			longs=numpy.arange(self.xmin+(self.resolution/2),self.xmax,self.resolution,dtype='float64')
		else:
			longs=numpy.arange(self.xmax-(self.resolution/2),self.xmin,-self.resolution,dtype='float64')
		
		if array2d:
			longs = numpy.tile(longs, self.height).reshape((self.height,self.width))

		if not degrees:
			longs = numpy.radians(longs)

		return longs


	def latitudes(self, px_order=True, array2d=False, degrees=True):
		'''
		returns array representing latitudes of centers of pixels 
		if  array2d - returns array [height, width], otherwise [height]
		'''
		if (px_order):
			lats=numpy.arange(self.ymax-(self.resolution/2),self.ymin,-self.resolution,dtype='float64')
#			if lats.shape < self.height:
#				lats=lats.hstack((lats,numpy.array(self.ymin)))
		else:
			lats=numpy.arange(self.ymin+(self.resolution/2),self.ymax,self.resolution,dtype='float64')

		if array2d:
			lats = numpy.repeat(lats, self.width).reshape((self.height,self.width))

		if not degrees:
			lats = numpy.radians(lats)

		return lats

	
	def center(self):
		'''
		returns lon lat of the bounding box center 
		'''
		c_lon=(self.xmin+self.xmax )/ 2.
		c_lat=(self.ymin+self.ymax )/ 2.
		return c_lon, c_lat
	
	def lonlat_of_px(self, px_x, px_y):
		'''
		returns lon lat of the pixel 
		'''
		lon, lat = None, None
		if (px_x >= 0) and (px_x <= (self.width -1)):
			lon = self.xmin + (px_x*self.resolution) + (self.resolution/2.)
		if (px_y >= 0) and (px_y <= (self.height -1)):
			lat = self.ymax - (px_y*self.resolution) - (self.resolution/2.)
		return lon, lat

	def subsegments(self, subseg_size=0.5):
		'''
		subseg_size - ratio of subsegment size to the size of bbox
		returns list of bounding boxes - representing subsegments 
		'''
		x_segs = int(math.ceil((self.xmax-self.xmin) / subseg_size))
		y_segs = int(math.ceil((self.ymax-self.ymin) / subseg_size))
		subseg_list=[]
		for x_seg in range(0,x_segs):
			for y_seg in range(0,y_segs):
				seg_xmin=max(self.xmin, self.xmin + x_seg*subseg_size)
				seg_xmax=min(self.xmax,seg_xmin+subseg_size )
				seg_ymax=min(self.ymax,self.ymax - y_seg*subseg_size)
				seg_ymin=max(self.ymin, seg_ymax - subseg_size)
				
				seg_width = int(math.floor(0.0000001+(seg_xmax-seg_xmin)/self.resolution))
				seg_height = int(math.floor(0.0000001+(seg_ymax-seg_ymin)/self.resolution))
#				seg_width = int((seg_xmax-seg_xmin)/self.resolution)
#				seg_height = int((seg_ymax-seg_ymin)/self.resolution)
				if (seg_width < 1) or (seg_height<1):
					continue
				seg_bbox = bounding_box(seg_xmin, seg_xmax, seg_ymin, seg_ymax, seg_width, seg_height, self.resolution)
				subseg_list.append(seg_bbox)
		return subseg_list
	
	def subsegments_xy(self, subseg_size_x=0.5, subseg_size_y=0.5):
		'''
		subseg_size - ratio of subsegment size to the size of bbox
		returns list of bounding boxes - representing subsegments 
		'''
		x_segs = int(math.ceil((self.xmax-self.xmin) / subseg_size_x))
		y_segs = int(math.ceil((self.ymax-self.ymin) / subseg_size_y))
		subseg_list=[]
		for x_seg in range(0,x_segs):
			for y_seg in range(0,y_segs):
				seg_xmin=max(self.xmin, self.xmin + x_seg*subseg_size_x)
				seg_xmax=min(self.xmax,seg_xmin+subseg_size_x )
				seg_ymax=min(self.ymax,self.ymax - y_seg*subseg_size_y)
				seg_ymin=max(self.ymin, seg_ymax - subseg_size_y)
				
				seg_width = int(math.floor(0.0000001+(seg_xmax-seg_xmin)/self.resolution))
				seg_height = int(math.floor(0.0000001+(seg_ymax-seg_ymin)/self.resolution))
#				seg_width = int((seg_xmax-seg_xmin)/self.resolution)
#				seg_height = int((seg_ymax-seg_ymin)/self.resolution)
				if (seg_width < 1) or (seg_height<1):
					continue
				seg_bbox = bounding_box(seg_xmin, seg_xmax, seg_ymin, seg_ymax, seg_width, seg_height, self.resolution)
				subseg_list.append(seg_bbox)
		return subseg_list
	
	def subsegments_by_pxsize(self, subseg_pxsize=128, return_with_indexes=False):
		'''
		subseg_pxsize - size of subsegment in px
		returns list of bounding boxes - representing subsegments 
		if return_with_indexes, then list contains triplets of [x_index,y_index,subseg_bbox]
		'''
		x_segs = int(math.ceil(self.width / float(subseg_pxsize)))
		y_segs = int(math.ceil(self.height / float(subseg_pxsize)))
#		x_segs = int(math.ceil(self.width) / subseg_pxsize)
#		y_segs = int(math.ceil(self.height) / subseg_pxsize)
		subseg_size = (float(subseg_pxsize)/self.width) * (self.xmax-self.xmin)
		subseg_list=[]
		for x_seg in range(0,x_segs):
			for y_seg in range(0,y_segs):
				seg_xmin=max(self.xmin, self.xmin + x_seg*subseg_size)
				seg_xmax=min(self.xmax,seg_xmin+subseg_size )
				seg_ymax=min(self.ymax,self.ymax - y_seg*subseg_size)
				seg_ymin=max(self.ymin, seg_ymax - subseg_size)
				seg_width = int(math.floor(0.0000001+(seg_xmax-seg_xmin)/self.resolution))
				seg_height = int(math.floor(0.0000001+(seg_ymax-seg_ymin)/self.resolution))
#				seg_width = int((seg_xmax-seg_xmin)/self.resolution)
#				seg_height = int((seg_ymax-seg_ymin)/self.resolution)
				if (seg_width < 1) or (seg_height<1):
					continue
				seg_bbox = bounding_box(seg_xmin, seg_xmax, seg_ymin, seg_ymax, seg_width, seg_height, self.resolution)
				if return_with_indexes:
					subseg_list.append([x_seg,y_seg,seg_bbox])
				else:
					subseg_list.append(seg_bbox)
		return subseg_list
	
	def subsegments_by_pxsize_xy(self, subseg_pxsize_x=16, subseg_pxsize_y=16):
		'''
		subseg_pxsize - size of subsegment in px in x and y dimension
		returns list of bounding boxes - representing subsegments 
		'''
		x_segs = int(math.ceil(self.width / float(subseg_pxsize_x)))
		y_segs = int(math.ceil(self.height / float(subseg_pxsize_y)))
#		x_segs = int(math.ceil(self.width) / subseg_pxsize)
#		y_segs = int(math.ceil(self.height) / subseg_pxsize)
		subseg_size_x = (float(subseg_pxsize_x)/self.width) * (self.xmax-self.xmin)
		subseg_size_y = (float(subseg_pxsize_y)/self.height) * (self.ymax-self.ymin)
		subseg_list=[]
		for x_seg in range(0,x_segs):
			seg_xmin=max(self.xmin, self.xmin + x_seg*subseg_size_x)
			seg_xmax=min(self.xmax,seg_xmin+subseg_size_x )
			for y_seg in range(0,y_segs):
				seg_ymax=min(self.ymax,self.ymax - y_seg*subseg_size_y)
				seg_ymin=max(self.ymin, seg_ymax - subseg_size_y)
				seg_width = int(math.floor(0.0000001+(seg_xmax-seg_xmin)/self.resolution))
				seg_height = int(math.floor(0.0000001+(seg_ymax-seg_ymin)/self.resolution))
#				seg_width = int((seg_xmax-seg_xmin)/self.resolution)
#				seg_height = int((seg_ymax-seg_ymin)/self.resolution)
				if (seg_width < 1) or (seg_height<1):
					continue
				seg_bbox = bounding_box(seg_xmin, seg_xmax, seg_ymin, seg_ymax, seg_width, seg_height, self.resolution)
				subseg_list.append(seg_bbox)
		return subseg_list
	
	
	def px_idx_grid_of_second_bbox(self, lowres_bbox):
		'''
		'''
		if not(lowres_bbox.is_lon_circular()) and not(self.intersects(lowres_bbox)):
			return None
		prjgrd_x1 = numpy.empty((self.height, self.width),dtype=numpy.int16)
		prjgrd_x2 = numpy.empty((self.height, self.width),dtype=numpy.int16)
		prjgrd_y1 = numpy.empty((self.height, self.width),dtype=numpy.int16)
		prjgrd_y2 = numpy.empty((self.height, self.width),dtype=numpy.int16)
		prjgrd_x_wght = numpy.zeros((self.height, self.width),dtype=numpy.float64)
		prjgrd_y_wght = numpy.zeros((self.height, self.width),dtype=numpy.float64)
		prjgrd_x1[:,:] = -1
		prjgrd_x2[:,:] = -1
		prjgrd_y1[:,:] = -1
		prjgrd_y2[:,:] = -1
		
		lons = self.longitudes(True)
		x1_vect = numpy.empty((self.width),dtype=numpy.int16)
		x2_vect = numpy.empty((self.width),dtype=numpy.int16)
		x_wght_vect = numpy.empty((self.width),dtype=numpy.float64)
		for i in range(0, self.width):
			lon = lons[i]
			position = (lon - lowres_bbox.xmin )/lowres_bbox.resolution
			x1_vect[i] = int(math.floor(position))
			x_wght_vect[i]=(x1_vect[i]+0.5)-position
			if (x_wght_vect[i] > 0):
				x2_vect[i] = x1_vect[i] - 1
			else:
				x_wght_vect[i] = -x_wght_vect[i]
				x2_vect[i] = x1_vect[i] + 1
				
		lats = self.latitudes(True)
		y1_vect = numpy.empty((self.height),dtype=numpy.int16)
		y2_vect = numpy.empty((self.height),dtype=numpy.int16)
		y_wght_vect = numpy.empty((self.height),dtype=numpy.float64)
		for i in range(0, self.height):
			lat = lats[i]
			position = (lowres_bbox.ymax - lat)/lowres_bbox.resolution
			y1_vect[i] = int(math.floor(position))
			y_wght_vect[i]=(y1_vect[i]+0.5)-position
			if (y_wght_vect[i] > 0):
				y2_vect[i] = y1_vect[i] - 1
			else:
				y_wght_vect[i] = -y_wght_vect[i]
				y2_vect[i] = y1_vect[i] + 1
		
		#if bbox is circular 
		if lowres_bbox.is_lon_circular():
			x1_vect%=lowres_bbox.width
			x2_vect%=lowres_bbox.width
		else:
			x1_vect[x1_vect<0]=-9
			wh = (x2_vect<0) & (x1_vect==0) #use x1_vect for x2_vect  if x1_vect is 0  
			x2_vect[wh] = x1_vect[wh]
			wh = (x2_vect<0) & (x1_vect<0)
			x2_vect[wh] = -9
			
		y1_vect[y1_vect<0]=-9
		wh = (y2_vect<0) & (y1_vect==0)
		y2_vect[wh] = y1_vect[wh]
		wh = (y2_vect<0) & (y1_vect<0)
		y2_vect[wh] = -9
		
		width_idx = lowres_bbox.width-1
		x1_vect[x1_vect>width_idx]=-9
		wh = (x2_vect>width_idx) & (x1_vect==width_idx)
		x2_vect[wh]=width_idx
		wh = (x2_vect>width_idx) & (x1_vect>width_idx)
		x2_vect[wh]=-9
		
		height_idx = lowres_bbox.height-1
		y1_vect[y1_vect>height_idx]=-9
		wh = (y2_vect>height_idx) & (y1_vect==height_idx)
		y2_vect[wh]=height_idx
		wh = (y2_vect>height_idx) & (y1_vect>height_idx)
		y2_vect[wh]=-9


#		else:
#			x1_vect[x1_vect<0]=0
#			x2_vect[x2_vect<0]=0
#
#		y1_vect[y1_vect<0]=0
#		y2_vect[y2_vect<0]=0
#
#		x1_vect[x1_vect>lowres_bbox.width-1]=lowres_bbox.width-1
#		x2_vect[x2_vect>lowres_bbox.width-1]=lowres_bbox.width-1
#		y1_vect[y1_vect>lowres_bbox.height-1]=lowres_bbox.height-1
#		y2_vect[y2_vect>lowres_bbox.height-1]=lowres_bbox.height-1
		
#		print prjgrd_x1.shape, x1_vect.shape
#		print prjgrd_x2.shape, x2_vect.shape
#		print x_wght_vect.shape
		for i in range(0, self.height):
			prjgrd_x1[i,:] = x1_vect
			prjgrd_x2[i,:] = x2_vect
			prjgrd_x_wght[i,:] = x_wght_vect

		for i in range(0, self.width):
			prjgrd_y1[:,i] = y1_vect
			prjgrd_y2[:,i] = y2_vect
			prjgrd_y_wght[:,i] = y_wght_vect

		return (prjgrd_x1, prjgrd_y1, prjgrd_x2, prjgrd_y2, prjgrd_x_wght, prjgrd_y_wght)

	def get_uneven_bbox(self):
		return uneven_bounding_box(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, width=self.width, height=self.height, xresolution=self.resolution, yresolution=self.resolution)


	def is_lon_circular(self, epsilon=0.001):
		total_width = self.xmax - self.xmin
		if (total_width > (360-epsilon)) and  (total_width < (360+epsilon)):
			return(True)
		else:
			return(False)
	def copy(self):
		import copy
		return copy.deepcopy(self)




class uneven_bounding_box(object):
	def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None, width=None, height=None, xresolution=None, yresolution=None):
		'''
		TODO: change width to cols
		width - number of columns in px
		height - number of rows in px
		xmin, xmax, ymin, ymax - out boundary of the bounding box (notcenters of pixels) in geographical coordinates
		resolution - pixel size in geographical coordinates
		'''
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		if width is not None:
			self.width = int(width)
		else:
			self.width = None
		if height is not None:
			self.height = int(height)
		else:
			self.height = None
		self.xresolution = xresolution
		self.yresolution = yresolution

	def __repr__(self):
		return "%f %f %f %f %d %d %f %f" % (self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.xresolution, self.yresolution)

	def intersects(self, bbox=None):
		'''
		true/false - has overlap with other bounding box
		'''
		if bbox is None:
			return False
		if bbox is None:
			return False
		if (self.xmax <= bbox.xmin) or (self.xmin >= bbox.xmax) or (self.ymax <= bbox.ymin) or (self.ymin >= bbox.ymax):
			return False
		return True

	def contains(self, bbox=None):
		'''
		true/false - has overlap with other bounding box
		'''
		epsilon=min(self.xresolution, bbox.xresolution)/10000.
		
		if bbox is None:
			return False
		if (self.xmax >= bbox.xmax-epsilon) and (self.xmin <= bbox.xmin+epsilon) and (self.ymax >= bbox.ymax-epsilon) and (self.ymin <= bbox.ymin+epsilon):
			return True
		return False

	def intersect(self, bbox=None, epsilon_rate=0.0001):
		'''
		return intersection (bbox) of two bounding_boxes
		'''
		epsilon=epsilon_rate*self.xresolution
		if bbox is None:
			return None
		if (self.xmax < bbox.xmin) or (self.xmin > bbox.xmax):
			return None
		if (self.ymax < bbox.ymin) or (self.ymin > bbox.ymax):
			return None
		
		temp_xmin=max(self.xmin, bbox.xmin)
		temp_xmax=min(self.xmax, bbox.xmax)
		temp_ymin=max(self.ymin, bbox.ymin)
		temp_ymax=min(self.ymax, bbox.ymax)
		
		out_xmin=max(self.xmin, self.xresolution * math.floor((temp_xmin/self.xresolution)+epsilon))
		out_xmax=min(self.xmax, self.xresolution * math.ceil((temp_xmax/self.xresolution)-epsilon))
		out_ymin=max(self.ymin, self.yresolution * math.floor((temp_ymin/self.yresolution)+epsilon))
		out_ymax=min(self.ymax, self.yresolution * math.ceil((temp_ymax/self.yresolution)-epsilon))
		
		out_height = int(round((out_ymax - out_ymin)/self.yresolution))
		out_width = int(round((out_xmax - out_xmin)/self.xresolution))
		if (out_height < 1) or (out_width < 1):
			return None

		out_bbox = uneven_bounding_box(out_xmin, out_xmax, out_ymin, out_ymax, out_width, out_height, self.xresolution, self.yresolution)
		
		return out_bbox

	def equals(self, bbox=None):
		'''
		test if the bounding boxes are equal
		returns: True/False
		'''
		if bbox is None:
			return False
		result=True
		result &= (self.xmin == bbox.xmin)
		result &= (self.xmax == bbox.xmax)
		result &= (self.ymin == bbox.ymin)
		result &= (self.ymax == bbox.ymax)
		result &= (self.width == bbox.width)
		result &= (self.height == bbox.height)
		result &= (self.xresolution == bbox.xresolution)
		result &= (self.yresolution == bbox.yresolution)
		return result

	
	def equal_resolution(self, bbox=None, epsilon=0.0001):
		'''
		test if the bounding boxes are equal
		returns: True/False
		'''
		if bbox is None:
			return False
		result=True
		eps=self.xresolution * epsilon
		result &= ((self.xresolution-eps) < bbox.xresolution) and ((self.xresolution+eps) > bbox.xresolution)
		eps=self.yresolution * epsilon
		result &= ((self.yresolution-eps) < bbox.yresolution) and ((self.yresolution+eps) > bbox.yresolution)
		return result

	
	def buffer_px(self, px=0):
		'''
		enlarge the bounding box by number of pixels
		returns: bbox
		'''
		px=int(math.floor(px))
		if px==0:
			return self
		
		out_xmin=self.xmin - (self.xresolution*px)
		out_xmax=self.xmax + (self.xresolution*px) 
		out_ymin=self.ymin - (self.yresolution*px)
		out_ymax=self.ymax + (self.yresolution*px) 
		
		out_height = self.height + (2*px)
		out_width = self.width + (2*px)
		
		if (out_height < 1) or (out_width < 1):
			return None
		
		out_bbox = bounding_box(out_xmin, out_xmax, out_ymin, out_ymax, out_width, out_height, self.xresolution, self.yresolution)
		
		return out_bbox

		
	def buffer_size(self, size=0):
		'''
		enlarge the bounding box by buffer size in geographical coordinates
		rounds to pixel
		returns: bbox
		'''
		if size==0:
			return self
		px=int(math.ceil(size/self.xresolution))
		return self.buffer_px(px=px)

		
	def pixel_coords_of_lonlat(self, lon=None, lat=None, lon_circular=False):
		'''
		returns pixel indexes of (nearest) point with lon, lat in bounding box
		inputs:
		lon, lat - longitude and latitude of point
		lon_circular [True/False] - work with longitude as circular (resolve different definition 
									of circular coordinate system longitudes in -180,180 or 0,360 range)  
		returns:
		px_x, px_y
		'''
		if (lon is None) or (lat is None):
			return None
		if ((lon < self.xmin) or (lon > self.xmax)) and not(lon_circular):
			return None
		if (lat < self.ymin) or (lat > self.ymax):
			return None
			
		px_x = int(math.floor((lon-self.xmin)/self.xresolution))
		px_y = min(self.height-1, max(0, int(math.floor((self.ymax-lat)/self.yresolution))))
		
		if (px_x < 0) & lon_circular:
			px_x+=self.width
		if (px_x > self.width-1) & lon_circular:
			px_x-=self.width

		if (px_x < 0) | (px_x > (self.width-1)):
			return None
		
		return px_x, px_y

	
	def px_idxs_of_latlon(self, lon=None, lat=None, lon_circular=False):
		'''
		returns closest pixel indexes and weights of points (bilinear) with lon, lat in bounding box
		inputs:
		lon, lat - longitude and latitude of point
		lon_circular [True/False] - work with longitude as circular (resolve different definition 
									of circular coordinate system longitudes in -180,180 or 0,360 range)  
		returns:
		x1, y1, x2, y2, x_wght, y_wght
		'''
		if (lon is None) or (lat is None):
			return None
		if ((lon < self.xmin) or (lon > self.xmax)) and not(lon_circular):
			return None
		if (lat < self.ymin) or (lat > self.ymax):
			return None

		position = (lon - self.xmin )/self.xresolution
		x1 = int(math.floor(position))
		x_wght=(x1+0.5)-position
		if (x_wght > 0):
			x2 = x1 - 1
		else:
			x_wght = -x_wght
			x2 = x1 + 1
		
		position = (self.ymax - lat)/self.yresolution
		y1 = int(math.floor(position))
		y_wght=(y1+0.5)-position
		if (y_wght > 0):
			y2 = y1 - 1
		else:
			y_wght = -y_wght
			y2 = y1 + 1

		#apply circular 'corection' (move coordinate to 0,self.width-1 interval)
		if (x1 < 0) & lon_circular:
			x1+=self.width
		if (x1 > self.width-1) & lon_circular:
			x1-=self.width
		if (x1 < 0) | (x1 > (self.width-1)):
			return None
		if (x2 < 0) & lon_circular:
			x2+=self.width
		if (x2 > self.width-1) & lon_circular:
			x2-=self.width
		if (x2 == -1):
			x2 = 0
		if (x2 == (self.width)):
			x2=self.width-1
		if (x2 < 0) | (x2 > (self.width-1)):
			return None

		return (x1, y1, x2, y2, x_wght, y_wght)
	
	
	def pixel_coords_of_bbox(self, bbox=None, epsilon_rate=0.001):
		'''
		returns px coordinates of other bounding box 
		'''
		epsilon = epsilon_rate* min(self.xresolution,bbox.xresolution)
		if not(self.intersects(bbox)):
			return None
		px_xmin = max(0, int(math.floor(((bbox.xmin-self.xmin)/self.xresolution)+epsilon)))
		px_ymin = max(0, int(math.floor(((self.ymax-bbox.ymax)/self.yresolution)+epsilon)))
		if abs(self.xresolution - bbox.xresolution) < epsilon :
			px_xmax = int(px_xmin + bbox.width - 1)
			px_ymax = int(px_ymin + bbox.height - 1)
		else:
			px_xmax = int(min(self.width -1, int(math.ceil(((bbox.xmax-self.xmin)/self.xresolution)-epsilon))))
			px_ymax = int(min(self.height -1, int(math.ceil(((bbox.ymax-self.ymin)/self.yresolution)-epsilon))))
		return px_xmin, px_xmax, px_ymin, px_ymax

	
	def longitudes(self, px_order=True):
		'''
		returns array representing longitudes of centers of pixels 
		'''
#		longs=numpy.arange(self.xmin+(self.xresolution/2),self.xmax,self.xresolution,dtype='float64')
		if px_order:
			longs=numpy.arange(self.xmin+(self.xresolution/2),self.xmax,self.xresolution,dtype='float64')
		else:
			longs=numpy.arange(self.xmax-(self.xresolution/2),self.xmin,-self.xresolution,dtype='float64')
		return longs

	def latitudes(self, px_order=True):
		'''
		returns array representing latitudes of centers of pixels 
		'''
		if (px_order):
			lats=numpy.arange(self.ymax-(self.yresolution/2),self.ymin,-self.yresolution,dtype='float64')
		else:
			lats=numpy.arange(self.ymin+(self.yresolution/2),self.ymax,self.yresolution,dtype='float64')
		return lats
	
	def center(self):
		'''
		returns lon lat of the bounding box center 
		'''
		c_lon=(self.xmin+self.xmax )/ 2.
		c_lat=(self.ymin+self.ymax )/ 2.
		return c_lon, c_lat
	
	def lonlat_of_px(self, px_x, px_y):
		'''
		returns lon lat of the pixel 
		'''
		lon, lat = None, None
		if (px_x >= 0) and (px_x <= (self.width -1)):
			lon = self.xmin + (px_x*self.xresolution) + (self.xresolution/2.)
		if (px_y >= 0) and (px_y <= (self.height -1)):
			lat = self.ymax - (px_y*self.yresolution) - (self.yresolution/2.)
		return lon, lat

	def subsegments(self, subseg_size=0.5):
		'''
		subseg_size - ratio of subsegment size to the size of bbox
		returns list of bounding boxes - representing subsegments 
		'''
		x_segs = int(math.ceil((self.xmax-self.xmin) / subseg_size))
		y_segs = int(math.ceil((self.ymax-self.ymin) / subseg_size))
		subseg_list=[]
		for x_seg in range(0,x_segs):
			for y_seg in range(0,y_segs):
				seg_xmin=max(self.xmin, self.xmin + x_seg*subseg_size)
				seg_xmax=min(self.xmax,seg_xmin+subseg_size )
				seg_ymax=min(self.ymax,self.ymax - y_seg*subseg_size)
				seg_ymin=max(self.ymin, seg_ymax - subseg_size)
				
				seg_width = int(math.floor(0.0000001+(seg_xmax-seg_xmin)/self.xresolution))
				seg_height = int(math.floor(0.0000001+(seg_ymax-seg_ymin)/self.yresolution))
#				seg_width = int((seg_xmax-seg_xmin)/self.resolution)
#				seg_height = int((seg_ymax-seg_ymin)/self.resolution)
				if (seg_width < 1) or (seg_height<1):
					continue
				seg_bbox = bounding_box(seg_xmin, seg_xmax, seg_ymin, seg_ymax, seg_width, seg_height, self.xresolution, self.yresolution)
				subseg_list.append(seg_bbox)
		return subseg_list
	
	def subsegments_by_pxsize(self, subseg_pxsize=128):
		'''
		subseg_pxsize - size of subsegment in px
		returns list of bounding boxes - representing subsegments 
		'''
		x_segs = int(math.ceil(self.width / float(subseg_pxsize)))
		y_segs = int(math.ceil(self.height / float(subseg_pxsize)))
#		x_segs = int(math.ceil(self.width) / subseg_pxsize)
#		y_segs = int(math.ceil(self.height) / subseg_pxsize)
		subseg_size = (float(subseg_pxsize)/self.width) * (self.xmax-self.xmin)
		subseg_list=[]
		for x_seg in range(0,x_segs):
			for y_seg in range(0,y_segs):
				seg_xmin=max(self.xmin, self.xmin + x_seg*subseg_size)
				seg_xmax=min(self.xmax,seg_xmin+subseg_size )
				seg_ymax=min(self.ymax,self.ymax - y_seg*subseg_size)
				seg_ymin=max(self.ymin, seg_ymax - subseg_size)
				seg_width = int(math.floor(0.0000001+(seg_xmax-seg_xmin)/self.xresolution))
				seg_height = int(math.floor(0.0000001+(seg_ymax-seg_ymin)/self.yresolution))
#				seg_width = int((seg_xmax-seg_xmin)/self.resolution)
#				seg_height = int((seg_ymax-seg_ymin)/self.resolution)
				if (seg_width < 1) or (seg_height<1):
					continue
				seg_bbox = bounding_box(seg_xmin, seg_xmax, seg_ymin, seg_ymax, seg_width, seg_height, self.xresolution, self.yresolution)
				subseg_list.append(seg_bbox)
		return subseg_list
	
	
	def px_idx_grid_of_second_bbox(self, lowres_bbox):
		'''
		'''
		if not(lowres_bbox.is_lon_circular()) and not(self.intersects(lowres_bbox)):
			return None
		prjgrd_x1 = numpy.empty((self.height, self.width),dtype=numpy.int16)
		prjgrd_x2 = numpy.empty((self.height, self.width),dtype=numpy.int16)
		prjgrd_y1 = numpy.empty((self.height, self.width),dtype=numpy.int16)
		prjgrd_y2 = numpy.empty((self.height, self.width),dtype=numpy.int16)
		prjgrd_x_wght = numpy.zeros((self.height, self.width),dtype=numpy.float64)
		prjgrd_y_wght = numpy.zeros((self.height, self.width),dtype=numpy.float64)
		prjgrd_x1[:,:] = -1
		prjgrd_x2[:,:] = -1
		prjgrd_y1[:,:] = -1
		prjgrd_y2[:,:] = -1
		
		lons = self.longitudes(True)
		x1_vect = numpy.empty((self.width),dtype=numpy.int16)
		x2_vect = numpy.empty((self.width),dtype=numpy.int16)
		x_wght_vect = numpy.empty((self.width),dtype=numpy.float64)
		for i in range(0, self.width):
			lon = lons[i]
			position = (lon - lowres_bbox.xmin )/lowres_bbox.xresolution
			x1_vect[i] = int(math.floor(position))
			x_wght_vect[i]=(x1_vect[i]+0.5)-position
			if (x_wght_vect[i] > 0):
				x2_vect[i] = x1_vect[i] - 1
			else:
				x_wght_vect[i] = -x_wght_vect[i]
				x2_vect[i] = x1_vect[i] + 1
				
		lats = self.latitudes(True)
		y1_vect = numpy.empty((self.height),dtype=numpy.int16)
		y2_vect = numpy.empty((self.height),dtype=numpy.int16)
		y_wght_vect = numpy.empty((self.height),dtype=numpy.float64)
		for i in range(0, self.height):
			lat = lats[i]
			position = (lowres_bbox.ymax - lat)/lowres_bbox.yresolution
			y1_vect[i] = int(math.floor(position))
			y_wght_vect[i]=(y1_vect[i]+0.5)-position
			if (y_wght_vect[i] > 0):
				y2_vect[i] = y1_vect[i] - 1
			else:
				y_wght_vect[i] = -y_wght_vect[i]
				y2_vect[i] = y1_vect[i] + 1
		
		#if bbox is circular 
		if lowres_bbox.is_lon_circular():
			x1_vect%=lowres_bbox.width
			x2_vect%=lowres_bbox.width
		else:
			x1_vect[x1_vect<0]=-9
			wh = (x2_vect<0) & (x1_vect==0) #use x1_vect for x2_vect  if x1_vect is 0  
			x2_vect[wh] = x1_vect[wh]
			wh = (x2_vect<0) & (x1_vect<0)
			x2_vect[wh] = -9
			
		y1_vect[y1_vect<0]=-9
		wh = (y2_vect<0) & (y1_vect==0)
		y2_vect[wh] = y1_vect[wh]
		wh = (y2_vect<0) & (y1_vect<0)
		y2_vect[wh] = -9
		
		width_idx = lowres_bbox.width-1
		x1_vect[x1_vect>width_idx]=-9
		wh = (x2_vect>width_idx) & (x1_vect==width_idx)
		x2_vect[wh]=width_idx
		wh = (x2_vect>width_idx) & (x1_vect>width_idx)
		x2_vect[wh]=-9
		
		height_idx = lowres_bbox.height-1
		y1_vect[y1_vect>height_idx]=-9
		wh = (y2_vect>height_idx) & (y1_vect==height_idx)
		y2_vect[wh]=height_idx
		wh = (y2_vect>height_idx) & (y1_vect>height_idx)
		y2_vect[wh]=-9

#
#		x1_vect[x1_vect>lowres_bbox.width-1]=lowres_bbox.width-1
#		x2_vect[x2_vect>lowres_bbox.width-1]=lowres_bbox.width-1
#		y1_vect[y1_vect>lowres_bbox.height-1]=lowres_bbox.height-1
#		y2_vect[y2_vect>lowres_bbox.height-1]=lowres_bbox.height-1
		
#		print prjgrd_x1.shape, x1_vect.shape
#		print prjgrd_x2.shape, x2_vect.shape
#		print x_wght_vect.shape
		for i in range(0, self.height):
			prjgrd_x1[i,:] = x1_vect
			prjgrd_x2[i,:] = x2_vect
			prjgrd_x_wght[i,:] = x_wght_vect

		for i in range(0, self.width):
			prjgrd_y1[:,i] = y1_vect
			prjgrd_y2[:,i] = y2_vect
			prjgrd_y_wght[:,i] = y_wght_vect
		return (prjgrd_x1, prjgrd_y1, prjgrd_x2, prjgrd_y2, prjgrd_x_wght, prjgrd_y_wght)


	def get_bbox(self):
		epsilon=0.00001
		delta= (self.xresolution - self.yresolution)
		if abs(delta) > epsilon:
			return None
		return bounding_box(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, width=self.width, height=self.height, resolution=self.xresolution)


	def is_lon_circular(self, epsilon=0.001):
		total_width = self.xmax - self.xmin
		if (total_width > (360-epsilon)) and  (total_width < (360+epsilon)):
			return(True)
		else:
			return(False)
		

def get_5x5_seg(lon,lat):
	'''
	
	'''
	seg_w,seg_e,seg_s,seg_n,seg_res=-180,180,-90,90,5. #GLOBAL segment region 
	seg_col=int(math.floor((lon-seg_w)/seg_res))
	seg_row=int(math.floor((seg_n - lat)/seg_res))
	return seg_col, seg_row

#calculate extent of segment
def get_5x5_seg_bbox(arow, acol , resolution, seg_size=5.):
	seg_w,seg_e,seg_s,seg_n=-180,180,-90,90 #segment region 
	w=seg_w+(acol*seg_size)
	e=seg_w+((acol+1)*seg_size)
	s=seg_n-((arow+1)*seg_size)
	n=seg_n-(arow*seg_size)
	width = int(math.floor(((e-w)/resolution)+0.5))
	height = int(math.floor(((n-s)/resolution)+0.5))
	bb=bounding_box(xmin=w, xmax=e, ymin=s, ymax=n, width=width, height=height, resolution=resolution)
	return bb



def get_5x5_seg_list_for_bbox(bbox,dateline_correction=False):
	'''
	inputs: bbox - bounding box for which to identify segments
	        correct_dateline - if segment is out of -180 - 180 range roll it to this range
	returns: list of 5x5 degree segments [[c,r],[c,r],...] within the bbox 
	'''
	xmin=bbox.xmin
	xmax=bbox.xmax
	ymin=bbox.ymin
	ymax=bbox.ymax
	seg_col_min, seg_row_min = get_5x5_seg(xmin, ymax)
	seg_col_max, seg_row_max = get_5x5_seg(xmax, ymin)
	segment_list=[[seg_col_min, seg_col_max, seg_row_min, seg_row_max ]]
	out_segment_list=[]
	for segment_c,segment_r in expand_segments(segment_list):
		seg_bbox = get_5x5_seg_bbox(segment_r,segment_c,bbox.resolution)
		if  (bbox.intersect(seg_bbox) is not None):
			if dateline_correction:
				if seg_bbox.center()[0] >180:
					segment_c-=72
				elif seg_bbox.center()[0]<-180:
					segment_c+=72
			out_segment_list.append([segment_c,segment_r])
	return out_segment_list


def expand_segments(segment_list):
	new_segment_list=[]
	for seg in segment_list:
		if len(seg) == 2:
			new_segment_list.append(seg)
		elif len(seg) == 4:
			for c in range(seg[0],seg[1]+1):
				for r in range(seg[2],seg[3]+1):
					new_segment_list.append([c,r])
	return new_segment_list
	
def seg_list_to_bbox(segment_list,resolution):
	if len(segment_list) <1:
		return None

	aux_bb = get_5x5_seg_bbox(segment_list[0][1], segment_list[0][0],resolution)
	if len(segment_list) == 1:
		return aux_bb
	
	xmin=aux_bb.xmin
	xmax=aux_bb.xmax
	ymin=aux_bb.ymin
	ymax=aux_bb.ymax
	for i in range(1,len(segment_list)):
		aux_bb = get_5x5_seg_bbox(segment_list[i][1], segment_list[i][0],resolution)
		xmin=min(xmin,aux_bb.xmin)
		xmax=max(xmax,aux_bb.xmax)
		ymin=min(ymin,aux_bb.ymin)
		ymax=max(ymax,aux_bb.ymax)

	width = int(math.floor(((xmax-xmin)/resolution)+0.5))
	height = int(math.floor(((ymax-ymin)/resolution)+0.5))
	bb=bounding_box(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width=width, height=height, resolution=resolution)
	return bb
	



def visualize_map_2d(map_data_2d, bbox, vmin=None, vmax=None, interpolation='nearest', title=None, img_width=10,img_height=8, show_grid=True, color='jet',countries_color='#999999', coast_color='#bbbbbb', openfile=None,  left=0.05, bottom=0.05, right=0.92, top=0.92, wspace=0.1, hspace=0.1, legend_padding=0.01, legend_width=0.018, legend_height=0.28, legend_bottom=None, legend_left=None, legend_title=None , grid_step=5.,xtickssize=10,ytickssize=10,xticksrotation=0,yticksrotation=0):
	from pylab import colorbar, show, figure,axes, setp, xticks, yticks
	from matplotlib import cm
	import matplotlib.pyplot as plt
	try:
		from mpl_toolkits.basemap import Basemap
	except:
		from matplotlib.toolkits.basemap import Basemap

	predefined_color_ramp_dict = plt.cm.datad

	if color in predefined_color_ramp_dict.keys():
		cmap=plt.get_cmap(color)
	else:
		cmap=cm.jet

	fig = figure(num=1,figsize=(img_width,img_height),facecolor='w')
	fig.clear()
#	ax=fig.add_subplot(111)
	
	#plot map
	map_data = numpy.ma.masked_where((numpy.isnan(map_data_2d)) ,map_data_2d)
	m = Basemap(projection='cyl',llcrnrlon=bbox.xmin,llcrnrlat=bbox.ymin, urcrnrlon=bbox.xmax,urcrnrlat=bbox.ymax, resolution='h')
	if show_grid:
		#decide resolution
		
		m.drawparallels(numpy.arange(-90.,90.,grid_step),labels=[1,0,0,0],color='k',fontsize=ytickssize, rotation=yticksrotation)
		if bbox.xmax>180:
			m.drawmeridians(numpy.arange(-180.,360.,grid_step),labels=[0,0,0,1],color='k',fontsize=xtickssize,rotation=xticksrotation)
		else:
			m.drawmeridians(numpy.arange(-180.,180.,grid_step),labels=[0,0,0,1],color='k',fontsize=xtickssize,rotation=xticksrotation)
			
	map_data=numpy.flipud(map_data)
	m.imshow(map_data, vmin=vmin, vmax=vmax, extent=(bbox.xmin, bbox.xmax, bbox.ymax, bbox.ymin), interpolation=interpolation, cmap=cmap)
	m.drawcoastlines(color=coast_color)
	m.drawcountries(color=countries_color)

	#legend
	if legend_left is None:
		legend_left=right+legend_padding
	if legend_bottom is None:
		legend_bottom=bottom
		
	cax = axes([legend_left, legend_bottom, legend_width, legend_height])
	cb=colorbar(cax=cax)
	for t in cb.ax.get_yticklabels():
		t.set_fontsize(9)

#	locs,labels=xticks(Visible=False)
#	setp(labels,rotation=90,fontsize=8)
##	locs,labels=yticks()
##	setp(labels,fontsize=9)
	
	if legend_title is not None:
		cb.ax.set_title(legend_title, fontsize=10.5, horizontalalignment='center')

	if title is not None:
		fig.suptitle(title, fontsize=14)
	
	if openfile is not None:
		fig.canvas.print_png(openfile)
	else:
		show()


def visualize_map_3d_subplots(map_data_3d, bbox, vmin=None, vmax=None, interpolation='nearest', img_width=10,img_height=8, axis_frame=True, title=None, show_grid=True, grid_step=5., color='jet', subplot_rows=None, subplot_cols=None, subplot_titles_list=[], subplot_title_xshift=None, subplot_title_yshift=None, countries_color='#777777', coast_color='#bbbbbb', draw_lsmask=True, openfile=None, left=0.05, bottom=0.05, right=0.92, top=0.92, wspace=0.1, hspace=0.1,  legend_padding=0.01, legend_width=0.018, legend_height=0.28,  legend_bottom=None, legend_left=None, legend_title=None, map_resolution='l'):
	from pylab import colorbar, show, figure, axes,close
	from matplotlib import cm
	import matplotlib.pyplot as plt
	try:
		from mpl_toolkits.basemap import Basemap
	except:
		from matplotlib.toolkits.basemap import Basemap

	predefined_color_ramp_dict = plt.cm.datad

	if color in predefined_color_ramp_dict.keys():
		cmap=plt.get_cmap(color)
	else:
		cmap=cm.jet


	#calculate number of subplots
	num_images=map_data_3d.shape[0]
	if (subplot_rows is None) and (subplot_cols is None):
		subplot_rows=max(1,int(math.ceil(math.sqrt(num_images))))
		subplot_cols=int(math.ceil(num_images/float(subplot_rows)))
	elif (subplot_rows is None) and (subplot_cols is not None):
		subplot_rows=int(math.ceil(num_images/float(subplot_cols)))
	elif (subplot_rows is not None) and (subplot_cols is None):
		subplot_cols=int(math.ceil(num_images/float(subplot_rows)))
	if num_images > (subplot_rows,subplot_cols):
		subplot_cols=int(math.ceil(num_images/float(subplot_rows)))
#	print subplot_cols, subplot_rows

	map_data_3d_ma = numpy.ma.masked_where((numpy.isnan(map_data_3d[:,:,:])) ,map_data_3d[:,:,:])
	data_min=map_data_3d_ma.min()
	data_max=map_data_3d_ma.max()
	if vmin is None:
		vmin = data_min - 0.1*(data_max-data_min)
	if vmax is None:
		vmax = data_max + 0.1*(data_max-data_min)

	fig = figure(num=1,figsize=(img_width,img_height),facecolor='w')
	fig.clear()
	
	my_axes=[]
	#plot map
	for indx in range(0,num_images):
#		ax=fig.add_subplot(subplot_rows,subplot_cols,indx+1)
		if indx ==0:
			ax=fig.add_subplot(subplot_rows,subplot_cols,indx+1, adjustable='box-forced')
		else:
			ax=fig.add_subplot(subplot_rows,subplot_cols,indx+1, sharex=my_axes[0], sharey=my_axes[0], adjustable='box-forced')
		
		map_data = map_data_3d_ma[indx,:,:]
		m = Basemap(projection='cyl',llcrnrlon=bbox.xmin,llcrnrlat=bbox.ymin, urcrnrlon=bbox.xmax,urcrnrlat=bbox.ymax, resolution=map_resolution) # f, h, i, l, c
		if show_grid:
			m.drawparallels(numpy.arange(-90.,90.,grid_step),labels=[1,0,0,0],color='k', fontsize=10)
			m.drawmeridians(numpy.arange(-180.,360.,grid_step),labels=[0,0,0,1],color='k', fontsize=10)
		map_data=numpy.flipud(map_data)
		m.imshow(map_data, vmin=vmin, vmax=vmax, extent=(bbox.xmin, bbox.xmax, bbox.ymax, bbox.ymin), interpolation=interpolation, cmap=cmap)
		m.drawcoastlines(color=coast_color)
		m.drawcountries(color=countries_color)
		if draw_lsmask:
			m.drawlsmask(land_color='None',ocean_color='grey',lakes=False)
		if len(subplot_titles_list):
			axtitle=ax.set_title(subplot_titles_list[indx], fontsize=11)
			if subplot_title_xshift is not None:
				x,y = axtitle.get_position()
				axtitle.set_position([x+subplot_title_xshift,y])
			if subplot_title_yshift is not None:
				x,y = axtitle.get_position()
				axtitle.set_position([x,y+subplot_title_yshift])
		if not axis_frame:
			ax.axis('off')
		my_axes.append(ax)
	
	#legend
	if legend_left is None:
		legend_left=right+legend_padding
	if legend_bottom is None:
		legend_bottom=bottom
		
	cax = axes([legend_left, legend_bottom, legend_width, legend_height])
	cb=colorbar(cax=cax)
	for t in cb.ax.get_yticklabels():
		t.set_fontsize(10)
	if legend_title is not None:
		cb.ax.set_title(legend_title, fontsize=10.5, horizontalalignment='center')
	
	#title
	if title is not None:
		fig.suptitle(title, fontsize=14)
	

	fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
		
	if openfile is not None:
		fig.canvas.print_png(openfile)
		close()
	else:
		show()
	
	

def visualize_map_3d(map_data_3d, bbox, vmin=None, vmax=None, interpolation='bilinear', img_width=10,img_height=8, show_grid=True, title=None, color='jet', countries_color='#999999', coast_color='#bbbbbb', openfile=None, grid_step=20, subplot_titles_list=[], ocean_mask=False, resolution='i', quiver_args=None, quiver_kwargs=None):
	from pylab import colorbar, show
	from matplotlib.widgets import Button
	from matplotlib import cm
	from pylab import axes, draw, figure
	import matplotlib.pyplot as plt
	
	try:
		from mpl_toolkits.basemap import Basemap
	except:
		from matplotlib.toolkits.basemap import Basemap

	predefined_color_ramp_dict = plt.cm.datad

	if color in predefined_color_ramp_dict.keys():
		cmap=plt.get_cmap(color)
	else:
		cmap=cm.jet
		

	fig = figure(num=1,figsize=(img_width,img_height),facecolor='w')

	map_data_3d_ma=numpy.ma.masked_where(numpy.isnan(map_data_3d), map_data_3d)
	
	if vmin is None:
		vmin=map_data_3d_ma.min()
	if vmax is None:
		vmax=map_data_3d_ma.max()
	vmin = vmin - 0.1*(vmax-vmin)
	vmax = vmax + 0.1*(vmax-vmin)

	m = Basemap(projection='cyl',llcrnrlon=bbox.xmin,llcrnrlat=bbox.ymin, urcrnrlon=bbox.xmax,urcrnrlat=bbox.ymax, resolution=resolution)
	if show_grid:
		m.drawparallels(numpy.arange(-90.,90.,grid_step),labels=[1,0,0,0],color='k')
		m.drawmeridians(numpy.arange(-180.,360.,grid_step),labels=[0,0,0,1],color='k')
	map_data=numpy.flipud(map_data_3d_ma[0,:,:])
	img = m.imshow(map_data, vmin=vmin, vmax=vmax, extent=(bbox.xmin, bbox.xmax, bbox.ymax, bbox.ymin), interpolation=interpolation, cmap=cmap)
	m.drawcoastlines(color=coast_color)
	m.drawcountries(color=countries_color)
	if ocean_mask:
		m.drawlsmask(land_color='None', ocean_color='w')
	colorbar()

	if (quiver_args is not None) and (len(quiver_args) >3):
		if quiver_kwargs is None:
			quiver_kwargs={}
#		m.quiver(*quiver_args)
#		quiver_args[0]=quiver_args[0]
#		quiver_args[1]=quiver_args[1]
		m.quiver(*quiver_args, **quiver_kwargs)


	#title
	if title is not None:
		fig.suptitle(title, fontsize=14)
	
		
	axprev = axes([0.85, 0.05, 0.06, 0.060])
	axnext = axes([0.92, 0.05, 0.06, 0.060])
	bnext = Button(axnext, '>')
	bprev = Button(axprev, '<')

	if len(subplot_titles_list) == map_data_3d.shape[0]:
		txt = fig.text(0.91, 0.02,'%d/%d: %s'%(1,map_data_3d.shape[0],str(subplot_titles_list[0])), size=8, ha="center")
	else:
		txt = fig.text(0.91, 0.02,'%d/%d'%(1,map_data_3d.shape[0]), size=8, ha="center")


	class Index:
		ind = 0
		def next(self, event):
			self.ind += 1
			i = self.ind%map_data_3d_ma.shape[0]
			img_data = numpy.flipud(map_data_3d_ma[i,:,:])
			img.set_data(img_data)
			if len(subplot_titles_list) == map_data_3d.shape[0]:
				txt.set_text('%d/%d: %s'%(i+1,map_data_3d.shape[0], str(subplot_titles_list[i])))
			else:
				txt.set_text('%d/%d'%(i+1,map_data_3d.shape[0]))
			draw()
			
		def prev(self, event):
			self.ind -= 1
			i = self.ind%map_data_3d_ma.shape[0]
			img_data = numpy.flipud(map_data_3d_ma[i,:,:])
			img.set_data(img_data)
			if len(subplot_titles_list) == map_data_3d.shape[0]:
				txt.set_text('%d/%d: %s'%(i+1,map_data_3d.shape[0], str(subplot_titles_list[i])))
			else:
				txt.set_text('%d/%d'%(i+1,map_data_3d.shape[0]))
			draw()
	
	callback = Index()
	bnext.on_clicked(callback.next)
	bprev.on_clicked(callback.prev)

	if openfile is not None:
		fig.canvas.print_png(openfile)
	else:
		show()



def visualize_map_3d_anim(map_data_3d, bbox, vmin=None, vmax=None, interpolation='bilinear', img_width=None,img_height=None, show_grid=True, title=None, color='jet', countries_color='#777777', coast_color='#888888', output_file=None, delay=50, grid_step=20, subplot_titles_list=[], plot_colorbar=True, ocean_mask=True, resolution='i', area_thresh=None, coastline_lw=2, country_lw=1.5, colorbar_height=0.4):
	from pylab import colorbar, show
	from matplotlib.widgets import Button
	from matplotlib import cm
	from pylab import axes, draw, figure, xticks, yticks, setp
	import matplotlib.pyplot as plt
	
	try:
		from mpl_toolkits.basemap import Basemap
	except:
		from matplotlib.toolkits.basemap import Basemap

	predefined_color_ramp_dict = plt.cm.datad
	if color in predefined_color_ramp_dict.keys():
		cmap=plt.get_cmap(color)
	else:
		cmap=cm.jet


	if output_file is None:
		'Missing animation (GIF) output_file '
		return False


	
	data_length=map_data_3d.shape[0]
	data_width=map_data_3d.shape[2]
	data_height=map_data_3d.shape[1]
	
	# image width, height
	wh_ratio = float(data_width)/float(data_height)
	if img_height is None:
		if img_width is None:
			img_width = 10.
		img_height = img_width/wh_ratio
	if img_width is None:
		if img_height is None:
			img_height = 10.
		img_width = img_height * wh_ratio


	frame_titles =  False
	if len(subplot_titles_list) == data_length:
		frame_titles =  True

	axes_width=0.80
	axes_center=0.5
	if plot_colorbar :
		axes_width=0.85
		axes_center=0.475
	#figure
	fig = figure(num=1,figsize=(img_width,img_height),facecolor='w')
	fig.clear()
	ax = axes([.05, .05, axes_width, .90], axisbg='w')

	map_data_3d_ma=numpy.ma.masked_where(numpy.isnan(map_data_3d), map_data_3d)
	if vmin is None:
		vmin=map_data_3d_ma.min()
	if vmax is None:
		vmax=map_data_3d_ma.max()

	
	vmin = vmin - 0.01*(vmax-vmin)
	vmax = vmax + 0.01*(vmax-vmin)

	m = Basemap(projection='cyl',llcrnrlon=bbox.xmin,llcrnrlat=bbox.ymin, urcrnrlon=bbox.xmax,urcrnrlat=bbox.ymax, resolution=resolution, area_thresh=area_thresh)
	
	if show_grid:
		m.drawparallels(numpy.arange(-90.,90.,grid_step),labels=[1,0,0,0],color='k')
		m.drawmeridians(numpy.arange(-180.,360.,grid_step),labels=[0,0,0,1],color='k')
		locs,labels=xticks()
		setp(labels,fontsize=10)
		locs,labels=yticks()
		setp(labels,fontsize=10)

	map_data=numpy.flipud(map_data_3d_ma[0,:,:])
	img = m.imshow(map_data, vmin=vmin, vmax=vmax, extent=(bbox.xmin, bbox.xmax, bbox.ymax, bbox.ymin), interpolation=interpolation, cmap=cmap)
	m.drawcoastlines(color=coast_color, linewidth=coastline_lw)
	m.drawcountries(color=countries_color, linewidth=country_lw)
	if ocean_mask:
		m.drawlsmask(land_color='None', ocean_color='w')
	
		
	if plot_colorbar:
		cax = axes([0.925, 0.1, 0.025, colorbar_height])
		cb=colorbar(cax=cax)
		for t in cb.ax.get_yticklabels():
			t.set_fontsize(9)
		

	if frame_titles:
		text=subplot_titles_list[0]
		txt = fig.text(axes_center, 0.96,'%s'%(text), size=14, ha="center")

	for i in range(0,data_length):
		img_data = numpy.flipud(map_data_3d_ma[i,:,:])
		img.set_data(img_data)
		draw()

		tmpfilename = str('/tmp/tmp_anim%03d' % i) + '.png'
		#title
		if frame_titles:
			txt.set_text('%s'%(subplot_titles_list[i]))
		fig.canvas.print_png(tmpfilename, dpi=100)
#		print tmpfilename
		
	#convert set of temp images into avi
	from subprocess import Popen,PIPE
	import sys, os

	cmd = 'convert -loop 50 -delay %d /tmp/tmp_anim*.png %s'% (delay, output_file)
#	cmd = 'ffmpeg -delay %d -i /tmp/msg_anim*.png %s'% (delay, output_file)
	print cmd
	p = Popen([cmd],  shell=True, stdout=PIPE,stderr=PIPE)
	out, outerr = p.communicate()
	if outerr:
		print 'avi conversion failure'
		print outerr
		print sys.exc_info()
		result = False
	else:
		print 'animation created: %s' % (output_file)
		result =  True

	#remove set of temp images
	for idx in range(0,data_length ):	
		tmpfilename = str('/tmp/tmp_anim%03d' % idx) + '.png'
#		print tmpfilename
		os.remove(tmpfilename)
	
	return result
	
def visualize_map_3d_multimap_anim(multi_map_data_3d, bbox, vmin=None, vmax=None, interpolation='bilinear', img_width=10,img_height=8, show_grid=True, title=None, color='jet', countries_color='#999999', coast_color='#bbbbbb', grid_step=20, subplot_titles_list=[], ocean_mask=False, resolution='i', quiver_args=None, quiver_kwargs=None, output_file=None, delay=50):
	from pylab import colorbar, show
	from matplotlib.widgets import Button
	from matplotlib import cm
	from pylab import axes, draw, figure
	import matplotlib.pyplot as plt
	
	try:
		from mpl_toolkits.basemap import Basemap
	except:
		from matplotlib.toolkits.basemap import Basemap

	predefined_color_ramp_dict = plt.cm.datad

	if color in predefined_color_ramp_dict.keys():
		cmap=plt.get_cmap(color)
	else:
		cmap=cm.jet
	
	if output_file is None:
		'Missing animation (GIF) output_file '
		return False


	fig = figure(num=1,figsize=(img_width,img_height),facecolor='w')

	map_data_3d_ma_lst = []
	multi_vmin = 9999999999
	multi_vmax =-9999999999
	for map_data_3d in multi_map_data_3d:
		map_data_3d_ma = numpy.ma.masked_where(numpy.isnan(map_data_3d), map_data_3d)
		map_data_3d_ma_lst.append(map_data_3d_ma)
		multi_vmin = min(multi_vmin, map_data_3d_ma.min())
		multi_vmax = max(multi_vmax, map_data_3d_ma.max())

	if vmin is None:
		vmin=multi_vmin
	if vmax is None:
		vmax=multi_vmax

	vmin = vmin - 0.1*(vmax-vmin)
	vmax = vmax + 0.1*(vmax-vmin)

	num_images = len(map_data_3d_ma_lst)	
	img_lst = []
	for indx in range(0,num_images):
		x_rel_size = (1.)/(num_images)
		x_max = ((indx+1-0.02)*x_rel_size)				  
		x_min = ((indx+0.02)*x_rel_size) 
		ax = fig.add_axes([x_min,0.035,  x_max-x_min, 0.95])
#		ax=fig.add_subplot(1,num_images,indx+1)

		m = Basemap(projection='cyl',llcrnrlon=bbox.xmin,llcrnrlat=bbox.ymin, urcrnrlon=bbox.xmax,urcrnrlat=bbox.ymax, resolution=resolution)
		if show_grid:
			m.drawparallels(numpy.arange(-90.,90.,grid_step),labels=[1,0,0,0],color='k')
			m.drawmeridians(numpy.arange(-180.,360.,grid_step),labels=[0,0,0,1],color='k')

	
		map_data=numpy.flipud(map_data_3d_ma_lst[indx][0,:,:])
		img = m.imshow(map_data, vmin=vmin, vmax=vmax, extent=(bbox.xmin, bbox.xmax, bbox.ymax, bbox.ymin), interpolation=interpolation, cmap=cmap)
		m.drawcoastlines(color=coast_color)
		m.drawcountries(color=countries_color)
		if ocean_mask:
			m.drawlsmask(land_color='None', ocean_color='w')
#		colorbar()
	
		if (quiver_args is not None) and (len(quiver_args) >3):
			if quiver_kwargs is None:
				quiver_kwargs={}
			m.quiver(*quiver_args, **quiver_kwargs)

		img_lst.append(img)
		
	#title
	if title is not None:
		fig.suptitle(title, fontsize=14)

	if len(subplot_titles_list) == map_data_3d.shape[0]:
		txt = fig.text(0.91, 0.005,'%d/%d: %s'%(1,map_data_3d.shape[0],str(subplot_titles_list[0])), size=8, ha="center")
	else:
		txt = fig.text(0.91, 0.005,'%d/%d'%(1,map_data_3d.shape[0]), size=8, ha="center")



	data_length = map_data_3d_ma_lst[0].shape[0] 
	for i in range(0,data_length):
		for indx in range(0,num_images):
			img_data = numpy.flipud(map_data_3d_ma_lst[indx][i,:,:])
			img_lst[indx].set_data(img_data)
		draw()
			
		tmpfilename = str('/tmp/tmp_anim%03d' % i) + '.png'
		#title
		if len(subplot_titles_list) == data_length:
			txt.set_text('%d/%d: %s'%(i+1,data_length, str(subplot_titles_list[i])))
		else:
			txt.set_text('%d/%d'%(i+1,data_length))

		fig.canvas.print_png(tmpfilename, dpi=100)


	#convert set of temp images into avi
	from subprocess import Popen,PIPE
	import sys, os

	cmd = 'convert -loop 50 -delay %d /tmp/tmp_anim*.png %s'% (delay, output_file)
#	cmd = 'ffmpeg -delay %d -i /tmp/msg_anim*.png %s'% (delay, output_file)
	print cmd
	p = Popen([cmd],  shell=True, stdout=PIPE,stderr=PIPE)
	out, outerr = p.communicate()
	if outerr:
		print 'avi conversion failure'
		print outerr
		print sys.exc_info()
		result = False
	else:
		print 'animation created: %s' % (output_file)
		result =  True

	#remove set of temp images
	for idx in range(0,data_length ):	
		tmpfilename = str('/tmp/tmp_anim%03d' % idx) + '.png'
#		print tmpfilename
		os.remove(tmpfilename)
	
	return result



def visualize_map_3d_multimap(multi_map_data_3d, bbox, vmin=None, vmax=None, interpolation='bilinear', img_width=10,img_height=8, show_grid=True, title=None, color='jet', countries_color='#999999', coast_color='#bbbbbb', grid_step=20, subplot_titles_list=[], ocean_mask=False, resolution='i', quiver_args=None, quiver_kwargs=None):
	from pylab import colorbar, show
	from matplotlib.widgets import Button
	from matplotlib import cm
	from pylab import axes, draw, figure
	import matplotlib.pyplot as plt
	
	try:
		from mpl_toolkits.basemap import Basemap
	except:
		from matplotlib.toolkits.basemap import Basemap

	predefined_color_ramp_dict = plt.cm.datad

	if color in predefined_color_ramp_dict.keys():
		cmap=plt.get_cmap(color)
	else:
		cmap=cm.jet
	

	fig = figure(num=1,figsize=(img_width,img_height),facecolor='w')

	global map_data_3d_ma_lst
	map_data_3d_ma_lst = []
	multi_vmin = 9999999999
	multi_vmax =-9999999999
	for map_data_3d in multi_map_data_3d:
		map_data_3d_ma = numpy.ma.masked_where(numpy.isnan(map_data_3d), map_data_3d)
		map_data_3d_ma_lst.append(map_data_3d_ma)
		multi_vmin = min(multi_vmin, map_data_3d_ma.min())
		multi_vmax = max(multi_vmax, map_data_3d_ma.max())

	if vmin is None:
		vmin=multi_vmin
	if vmax is None:
		vmax=multi_vmax

	vmin = vmin - 0.1*(vmax-vmin)
	vmax = vmax + 0.1*(vmax-vmin)

	num_images = len(map_data_3d_ma_lst)	
	img_lst = []
	for indx in range(0,num_images):
		x_rel_size = (1.)/(num_images)
		x_max = ((indx+1-0.02)*x_rel_size)				  
		x_min = ((indx+0.02)*x_rel_size) 
		ax = fig.add_axes([x_min,0.035,  x_max-x_min, 0.95])

		m = Basemap(projection='cyl',llcrnrlon=bbox.xmin,llcrnrlat=bbox.ymin, urcrnrlon=bbox.xmax,urcrnrlat=bbox.ymax, resolution=resolution)
		if show_grid:
			m.drawparallels(numpy.arange(-90.,90.,grid_step),labels=[1,0,0,0],color='k')
			m.drawmeridians(numpy.arange(-180.,360.,grid_step),labels=[0,0,0,1],color='k')

	
		map_data=numpy.flipud(map_data_3d_ma_lst[indx][0,:,:])
		img = m.imshow(map_data, vmin=vmin, vmax=vmax, extent=(bbox.xmin, bbox.xmax, bbox.ymax, bbox.ymin), interpolation=interpolation, cmap=cmap)
		m.drawcoastlines(color=coast_color)
		m.drawcountries(color=countries_color)
		if ocean_mask:
			m.drawlsmask(land_color='None', ocean_color='w')
#		colorbar()
	
		if (quiver_args is not None) and (len(quiver_args) >3):
			if quiver_kwargs is None:
				quiver_kwargs={}
			m.quiver(*quiver_args, **quiver_kwargs)

		img_lst.append(img)
		
	#title
	if title is not None:
		fig.suptitle(title, fontsize=14)
	
		
	axprev = axes([0.85, 0.025, 0.06, 0.060])
	axnext = axes([0.92, 0.025, 0.06, 0.060])
	bnext = Button(axnext, '>')
	bprev = Button(axprev, '<')

	if len(subplot_titles_list) == map_data_3d.shape[0]:
		txt = fig.text(0.91, 0.005,'%d/%d: %s'%(1,map_data_3d.shape[0],str(subplot_titles_list[0])), size=8, ha="center")
	else:
		txt = fig.text(0.91, 0.005,'%d/%d'%(1,map_data_3d.shape[0]), size=8, ha="center")


	class Index:
		ind = 0
		def next(self, event):
			self.ind += 1
			i = self.ind%map_data_3d_ma.shape[0]
			for indx in range(0,num_images):
				img_data = numpy.flipud(map_data_3d_ma_lst[indx][i,:,:])
				img_lst[indx].set_data(img_data)
			if len(subplot_titles_list) == map_data_3d.shape[0]:
				txt.set_text('%d/%d: %s'%(i+1,map_data_3d.shape[0], str(subplot_titles_list[i])))
			else:
				txt.set_text('%d/%d'%(i+1,map_data_3d.shape[0]))
			draw()
			
		def prev(self, event):
			self.ind -= 1
			i = self.ind%map_data_3d_ma.shape[0]
			for indx in range(0,num_images):
				img_data = numpy.flipud(map_data_3d_ma_lst[indx][i,:,:])
				img_lst[indx].set_data(img_data)
			if len(subplot_titles_list) == map_data_3d.shape[0]:
				txt.set_text('%d/%d: %s'%(i+1,map_data_3d.shape[0], str(subplot_titles_list[i])))
			else:
				txt.set_text('%d/%d'%(i+1,map_data_3d.shape[0]))
			draw()
	
	callback = Index()
	bnext.on_clicked(callback.next)
	bprev.on_clicked(callback.prev)

	show()










def globe_lonlatdelta_xydelta(latitude,lon_delta,lat_delta):
	'''
	calculate approximate x and y distance in meters for distance in lon and lat for given latitude
	distance in longitude decreases towards the poles  
	'''
	R=6371000 #Earth radius in m
	earth_perimeter_per_degree = 2.*math.pi*R/360.
	cos_lat = math.cos(math.radians(latitude))
	y_delta = lat_delta*earth_perimeter_per_degree
	x_delta = lon_delta*earth_perimeter_per_degree*cos_lat
	
	return x_delta, y_delta
	
	



def _test_bb():
#	bbox1 = bounding_box(-80.100000, -58.000000, 11.000000, 28.100000, 221, 171, 0.100000)
#	print bbox1
#	bbox2 = bounding_box(-80.000000, -75.000000, 25.000000, 30.000000, 600, 600, 1/120.)
#	print bbox2
#	bbox3=bbox1.intersect(bbox2)
#	print bbox3
	
	w, e, s, n, res = -70, -69.73333, -35., -34.80, 2./60.
	bbox4=bounding_box(w, e, s, n, int(numpy.floor(((e-w)/res)+0.5)), int(numpy.floor(((n-s)/res)+0.5)), res)
	
	print bbox4 
	print w+bbox4.width*bbox4.resolution
	subsegs=bbox4.subsegments_by_pxsize_xy(4, 4)
	for subseg_bbox in subsegs:
		print subseg_bbox

def _test():
	lat,lon = 44.99864, -0.688407
	print lat,lon
	seg_col, seg_row = get_5x5_seg(lon,lat)
	print seg_col, seg_row
	bbox=get_5x5_seg_bbox(seg_row, seg_col , 1./30., seg_size=5.)
	print bbox
	print bbox.px_idxs_of_latlon(lon=lon, lat=lat)
	print 

def _test3():
	
	n=45.320815
	s=45.31529
	w=8.392295
	e=8.435945
	res=0.000005
	out_latlon_bbox=bounding_box(w, e, s, n, int((e-w+0.000000001)/res), int((n-s+0.000000001)/res), res)
	ss_bboxes = out_latlon_bbox.subsegments_by_pxsize(subseg_pxsize=128)
	for i in range(0,2):
		ss_bbox=ss_bboxes[i]
		print ss_bbox, out_latlon_bbox.pixel_coords_of_bbox(ss_bbox)
		
	print 
	
	ss_bbox=bounding_box(8.392295, 8.392935, 45.320175, 45.320815, 128, 128, 0.000005)
	print ss_bbox, out_latlon_bbox.pixel_coords_of_bbox(ss_bbox)
	ss_bbox=bounding_box(8.392295, 8.392935, 45.319535, 45.320175, 128, 128, 0.000005)
	print ss_bbox, out_latlon_bbox.pixel_coords_of_bbox(ss_bbox)
	

def _test4():
	lon_delta = 0.1 #
	lat_delta = 0.1
	print 'original lon, lat distance', lon_delta,lat_delta
	for latitude in range(-90,90,5):
		print latitude, globe_lonlatdelta_xydelta(latitude,lon_delta,lat_delta)

	
if __name__=="__main__":
#	_test()
#	_test_bb()
#	_test3()
	_test4()
	
	pass