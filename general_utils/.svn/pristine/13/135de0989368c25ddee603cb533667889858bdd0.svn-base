'''
Created on Apr 19, 2011

@author: tomas

utils to read and write arcvie binary grid data
'''



import os

import numpy
import pylab

from general_utils import latlon

class AVraster_hdr():
    def __init__(self, ncols=None, nrows=None, xllcorner=None, yllcorner=None, cellsize=None, NODATA_value=None, byteorder=None):
        self.ncols = ncols
        self.nrows = nrows
        self.xllcorner = xllcorner
        self.yllcorner = yllcorner
        self.cellsize = cellsize
        self.NODATA_value = NODATA_value
        self.byteorder = byteorder

    def load(self, filename):
        afile = open(filename)
        counter=0
        for line in afile:
            counter+=1
            if counter>7: #maximum size of header
                break
            tokens=line.split()
            if len(tokens)==2:
                if tokens[0] == 'ncols':
                    self.ncols = int(tokens[1])
                if tokens[0] == 'nrows':
                    self.nrows = int(tokens[1])
                if tokens[0] == 'xllcorner':
                    self.xllcorner = float(tokens[1])
                if tokens[0] == 'yllcorner':
                    self.yllcorner = float(tokens[1])
                if tokens[0] == 'cellsize':
                    self.cellsize = float(tokens[1])
                if tokens[0] == 'NODATA_value':
                    self.NODATA_value = float(tokens[1])
                if tokens[0] == 'byteorder':
                    self.byteorder = tokens[1]
        afile.close()
    
    def write(self, filename, type_ascii=False, precision=12):
        afile = open(filename, 'w')
#        afile.write('ncols '+str(self.ncols)+'\r\n')
#        afile.write('nrows '+str(self.nrows)+'\r\n')
        afile.write('ncols %d\r\n' %(self.ncols))
        afile.write('nrows %d\r\n'%((self.nrows)))
#        afile.write('xllcorner '+str(self.xllcorner)+'\r\n')
#        afile.write('yllcorner '+str(self.yllcorner)+'\r\n')
#        afile.write('cellsize '+str(self.cellsize)+'\r\n')
        aux='xllcorner %.'+str(int(precision))+'f\r\n'
        afile.write(aux%(self.xllcorner))
        aux='yllcorner %.'+str(int(precision))+'f\r\n'
        afile.write(aux%(self.yllcorner))
        aux='cellsize %.'+str(int(precision))+'f\r\n'
        afile.write(aux%(self.cellsize))
        afile.write('NODATA_value '+str(self.NODATA_value)+'\r\n')
        if not type_ascii:
            afile.write('byteorder '+str(self.byteorder)+'\r\n')
        afile.close()
     
    def __repr__(self):
        return "%d %d %f %f %f %f %s" % (self.ncols, self.nrows, self.xllcorner, self.yllcorner, self.cellsize, self.NODATA_value, self.byteorder)


class AVraster(object):
    def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None, width=None, height=None, resolution=None, data=None, filename=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        self.resolution = resolution
        self.data = data
        self.filename = filename
        self.filename_hdr = None
        self.filename_flt = None
        self.filename_asc = None
        self.hdr = None
        
        
        if filename is not None:
            self._filename2fielname_hdrflt()

        if None not in (xmin, xmax, ymin, ymax, width, height, resolution) :
            self.hdr = AVraster_hdr(ncols=width, nrows=height, xllcorner=xmin, yllcorner=ymin, cellsize=resolution, NODATA_value=-9999, byteorder='LSBFIRST')
        
    def get_basename(self):
        if len(self.filename) < 5:
            basename=self.filename
        if (self.filename[-3:] == 'txt') or (self.filename[-3:] == 'asc') or (self.filename[-3:] == 'flt') or (self.filename[-3:] == 'hdr'):
            basename=self.filename[:-4]
        else:
            basename=self.filename
        return basename
    
    def _filename2fielname_hdrflt(self):
        #create HDR and FLT file names
        basename=self.get_basename()
        self.filename_flt=basename+'.flt'
        self.filename_hdr=basename+'.hdr'
    
    def _filename2fielname_asc(self):
        #create ASC file name
        basename=self.get_basename()
        self.filename_asc=basename+'.asc'
    
    def _fielname_hdrflt_exist(self):
        if ((not os.path.exists(self.filename_flt)) or (not os.path.isfile(self.filename_flt))):
            return False
        if ((not os.path.exists(self.filename_hdr)) or (not os.path.isfile(self.filename_hdr))):
            return False
        return True
    
    def _fielname_asc_exist(self):
        if ((not os.path.exists(self.filename_asc)) or (not os.path.isfile(self.filename_asc))):
            return False
        return True
    
    def get_bbox(self):
        bbox = latlon.bounding_box(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, width=self.width, height=self.height, resolution=self.resolution)
        return bbox

    def set_bbox(self, bbox):
        self.xmin=bbox.xmin
        self.xmax=bbox.xmax
        self.ymin=bbox.ymin
        self.ymax=bbox.ymax
        self.width=bbox.width
        self.height=bbox.height
        self.resolution=bbox.resolution
        
        if None not in (self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.resolution) :
            self.hdr = AVraster_hdr(ncols=self.width, nrows=self.height, xllcorner=self.xmin, yllcorner=self.ymin, cellsize=self.resolution, NODATA_value=-9999, byteorder='LSBFIRST')

    
    def get_ubbox(self):
        ubbox = latlon.uneven_bounding_box(xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax, width=self.width, height=self.height, xresolution=self.resolution, yresolution=self.resolution)
        return ubbox

    
    def __repr__(self):
        return "%f %f %f %f %d %d %f" % (self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.resolution)
    
    def load_avbin(self, filename=None):
        if (self.filename is None) and (filename is None):
            print 'AVraster load error: missing file name'
            return False
        
        # process filename
        if filename is not None:
            self.filename=filename
            self._filename2fielname_hdrflt()
        
        #check existence of files
        if not self._fielname_hdrflt_exist():
            print 'AVraster load error: file does not exists: %s %s' % (self.filename_flt, self.filename_hdr)
            return False
        
        #read HDR file
        self.hdr=AVraster_hdr()
        self.hdr.load(self.filename_hdr)
        self.xmin = self.hdr.xllcorner
        self.ymin = self.hdr.yllcorner
        self.resolution = self.hdr.cellsize
        self.width = self.hdr.ncols
        self.height = self.hdr.nrows
        self.xmax = self.xmin + (self.resolution * self.width)
        self.ymax = self.ymin + (self.resolution * self.height)
        
        #read DATA file
        fileobj = open(self.filename_flt, mode='rb')
        self.data = numpy.fromfile(file=fileobj, dtype = "float32", count=self.width*self.height)
        fileobj.close()
        self.data = self.data.reshape(self.height,self.width)
        self.data[self.data == self.hdr.NODATA_value] = numpy.nan
        
        return True
    
    
    

    def write_avbin(self, filename=None,header_decimals=17):
        if (self.filename is None) and (filename is None):
            print 'AVraster write error: missing file name'
            return False
        
        # process filename
        if filename is not None:
            self.filename=filename
            self._filename2fielname_hdrflt()
        
        #write header
        self.hdr.write(self.filename_hdr, precision=header_decimals)
        
        #write flt
        adata=self.data.copy()
        adata=adata.astype(numpy.float32)
        adata[adata!=adata]=self.hdr.NODATA_value
        fileobj = open(self.filename_flt, mode='wb')
        adata.tofile(fileobj)
        fileobj.close()


    def load_avascii(self, filename=None):
        if (self.filename is None) and (filename is None):
            print 'AVraster load error: missing file name'
            return False
        
        # process filename
        if filename is not None:
            self.filename=filename
            self._filename2fielname_asc()
        
        #check existence of files
        if not self._fielname_asc_exist():
            print 'AVraster load error: file does not exists: %s' % (self.filename_asc)
            return False
        
        
        #read HDR file
        self.hdr=AVraster_hdr()
        self.hdr.load(self.filename_asc)
        self.xmin = self.hdr.xllcorner
        self.ymin = self.hdr.yllcorner
        self.resolution = self.hdr.cellsize
        self.width = self.hdr.ncols
        self.height = self.hdr.nrows
        self.xmax = self.xmin + (self.resolution * self.width)
        self.ymax = self.ymin + (self.resolution * self.height)

      
        #read DATA from file
        data_list=[]
        afile = open(filename)
        for line in afile:
            tokens=line.split()
            #skip header lines
            if len(tokens)==2 and tokens[0].lower() in ['ncols','nrows', 'xllcorner','yllcorner', 'xllcenter','yllcenter', 'cellsize', 'nodata_value', 'byteorder']:
                continue
            #data_list
            for token in tokens: 
                data_list.append(float(token))
        afile.close()

        if len(data_list) != (self.height*self.width):
            print 'Error reading ASCII file: number of file data elements %d does not fit height %d * size %d size from header (%d).' % (len(data_list),self.height, self.width,self.height*self.width) 
            return False
        
        #convert to array
        self.data = numpy.array(data_list)      
        self.data = self.data.reshape(self.height,self.width)
        self.data[self.data == self.hdr.NODATA_value] = numpy.nan
        
        return True


        
    def write_avascii(self, filename=None, decimals=0,header_decimals=17):
        if (self.filename is None) and (filename is None):
            print 'AVraster write error: missing file name'
            return False
        
        # process filename
        if filename is not None:
            self.filename=filename
            self._filename2fielname_asc()
        
        #write header
        self.hdr.write(self.filename_asc,type_ascii=True, precision=header_decimals)
        
        #write data (to the same file)
        adata=self.data.copy()
        adata[adata!=adata]=self.hdr.NODATA_value
        adata = numpy.around(adata, decimals=decimals)
        nrows, ncols = adata.shape

        fileobj = open(self.filename_asc, mode='a')
        for r in range(0,nrows):
            aline_list = list(adata[r,:])
            aline_str_list=[]
            for val in aline_list:
                aline_str_list.append(str(val))
            aline_str = ' '.join(aline_str_list)
            fileobj.write(aline_str+'\r\n')
        fileobj.close()
        
             
    def show_data(self):
        if self.data is None:
            print 'Error: no data to show'
            return False
        pylab.imshow(self.data, interpolation='nearest', extent=(self.xmin, self.xmax, self.ymin, self.ymax))
        pylab.colorbar()
        pylab.show()
        
    def calculate_slopes(self, type=None):
        if self.data is None:
            print 'Error: no data to process'
            return None
        
        if type is None:
            print 'Error: no slope type defined'
            return None

        if type not in ['EW', 'NS']:
            print 'Error: slope type must be either EW or NS'
            return None

        if type == 'EW':
            dem = self.data
            dem2 = numpy.roll(dem.copy(),axis=1, shift=-1)
            dem2[:,-1]=numpy.nan
            aux1=numpy.arcsin((dem2-dem)/self.resolution)
            dem2 = numpy.roll(dem.copy(),axis=1, shift=-2)
            dem2[:,-2:]=numpy.nan
            aux11=numpy.arcsin((dem2-dem)/(2.*self.resolution))
            dem2 = numpy.roll(dem.copy(),axis=1, shift=1)
            dem2[:,0]=numpy.nan
            aux2=numpy.arcsin((dem-dem2)/self.resolution)
            dem22 = numpy.roll(dem.copy(),axis=1, shift=2)
            dem22[:,:2]=numpy.nan
            aux22=numpy.arcsin((dem-dem2)/(2.*self.resolution))
            
            slope = numpy.degrees((aux11+aux1+aux2+aux22) / 4.)

        if type == 'NS':
            # the NS slope calculation is shifted by one pixel to south to see what shading is in southwards dirtection 
            dem = self.data
            dem2 = numpy.roll(dem.copy(),axis=0, shift=-1)
            dem2[-1,:]=numpy.nan
            aux1=numpy.arcsin((dem-dem2)/self.resolution)
            dem2 = numpy.roll(dem.copy(),axis=0, shift=-2)
            dem2[-2:,:]=numpy.nan
            aux11=numpy.arcsin((dem-dem2)/(2.*self.resolution))
            dem2 = numpy.roll(dem.copy(),axis=0, shift=-3)
            dem2[-3:,:]=numpy.nan
            aux111=numpy.arcsin((dem-dem2)/(3.*self.resolution))
            dem2 = numpy.roll(dem.copy(),axis=0, shift=1)
            dem2[0,:]=numpy.nan
            aux2=numpy.arcsin((dem2-dem)/self.resolution)
#            dem2 = numpy.roll(dem.copy(),axis=0, shift=2)
#            dem2[:2,:]=numpy.nan
#            aux22=numpy.arcsin((dem2-dem)/(2.*self.resolution))
#            slope = numpy.degrees((aux11+aux1+aux2+aux22) / 4.)
            
            slope = numpy.degrees((aux11+aux1+aux2+aux111) / 4.)
            
        slp_obj=self.copy(clear_filename=True)
        slp_obj.data = slope

        return slp_obj


    def copy(self, clear_filename=False, clear_data=False):
        import copy
        new_obj=copy.deepcopy(self)
        
        if clear_filename:
            new_obj.filename=None
            new_obj.filename_flt=None
            new_obj.filename_hdr=None
        if clear_data:
            new_obj.data=None
        return new_obj
    
    
def example1():
    avdata_obj=AVraster()
    avdata_obj.load_avbin(filename='/home/tomas/Documents/sg_regions/cz_regs_v0.flt')
    print avdata_obj.get_ubbox()
    print avdata_obj.data.shape, avdata_obj.data.dtype
    print avdata_obj.data
    avdata_obj.show_data()
    
    
    #calculate slopes in EW and NS directions
#    EW_obj=dem_obj.calculate_slopes(type='EW')
#    EW_obj.show_data()
    
def example2():
    avdata_obj=AVraster(xmin=0, xmax=2, ymin=0, ymax=2, width=2, height=2, resolution=1)
#    avdata_obj.data=numpy.zeros((2,2))
    avdata_obj.data=numpy.array([[1,3],[4,2]])
    avdata_obj.show_data()
    print avdata_obj.get_ubbox()
    avdata_obj.write_avbin(filename='/tmp/test_avbin_raster.flt')
    
def example3():
    #create array
    avdata_obj=AVraster(xmin=0, xmax=2, ymin=0, ymax=2, width=2, height=2, resolution=1)
    avdata_obj.data=numpy.array([[1.12,3.11],[4.56,2.55]])
    #show array
    avdata_obj.show_data()
    #write array
    avdata_obj.write_avascii(filename='/tmp/test_avbin_raster.asc', decimals=1)
    
    #create second array and load data
    avdata_obj2=AVraster()
    avdata_obj2.load_avascii(filename='/tmp/test_avbin_raster.asc')
    #show array
    avdata_obj2.show_data()
    #get bbox and data
    bbox = avdata_obj2.get_bbox()
    data = avdata_obj2.data
    print bbox
    print data 


def example4():
    bbox=latlon.bounding_box(xmin=2, xmax=3, ymin=6, ymax=7, width=2, height=2, resolution=1)
    data=numpy.array([[1.12,3.11],[4.56,2.55]])
    
    #create AV object
    avdata_obj=AVraster()
    avdata_obj.set_bbox(bbox)
    avdata_obj.data=data
    
    #show array
    avdata_obj.show_data()
    
    
def test():    
    from general_utils import av_utils
    from general_utils import latlon
    
    #some fake data and bounding box
    bbox=latlon.bounding_box(xmin=2, xmax=3.5, ymin=5.5, ymax=7.5, width=3, height=4, resolution=0.5)
    data=numpy.array([[1.12,3.11,6.5],[1.12,3.11,2.9],[4.56,2.55,0.7],[1.12,4.85,0.1]])
    
    
    #EXPORTS
    #create AVraster instance
    avdata_obj=av_utils.AVraster()
    avdata_obj.set_bbox(bbox)
    avdata_obj.data=data

    #write ASCII raster
    avdata_obj.write_avascii(filename='/tmp/test_av_raster.asc', decimals=1)
    
    #write float32 data
    avdata_obj.write_avbin(filename='/tmp/test_av_raster.flt')
    
    #IMPORTS
    #read ASCII raster
    avdata_obj2 = av_utils.AVraster()
    avdata_obj2.load_avascii(filename='/tmp/test_av_raster.asc')
    bbox = avdata_obj2.get_bbox()
    data = avdata_obj2.data
    
    #read float32 raster
    avdata_obj3 = av_utils.AVraster()
    avdata_obj3.load_avbin(filename='/tmp/test_av_raster.flt')
    
    #show array
    avdata_obj.show_data()
    
if __name__=="__main__":
#    example1()
#    example2()
#    example3()
#    example4()
    test()
