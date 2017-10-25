'''
Created on Jan 21, 2011

@author: duro, tomas
'''
import os
from subprocess import Popen,PIPE
import numpy

##logger section
#from general_utils import basic_logger
#logger = basic_logger.basic_logger_make(basic_logger.get_run_module_name()+'.grass_utils', 'info')

#logger section
from general_utils.basic_logger import make_logger
logger = make_logger(__name__)
import logging
logging.getLogger().setLevel(logging.DEBUG)


def processPopenResults(p, verbose=True):
    out, outerr = p.communicate()
    if outerr:
        return(False, out, outerr)
    else:
        return(True, out, outerr)


def r_out_arc(grs_raster, out_filename, dec_places , verbose=False):
    cmd=["r.out.arc input=%s output=%s.asc dp=%s\n" % (grs_raster,out_filename,dec_places)]
    #print cmd
    if verbose:
        logger.info( "\ncreating %s.asc" % (out_filename))
    p = Popen(cmd,  shell=True, stdout=PIPE,stderr=PIPE)
    res=processPopenResults(p)
    if verbose:
        logger.info((';'.join(cmd)))
        
        logger.info(res[1]+';'+res[2])
    return True

def r_out_bin(input, output):
    cmd_inp=["r.out.bin","input="+input, "output="+output,"--q"]
    p = Popen(cmd_inp,  stdout=PIPE, stderr=PIPE)
    result=processPopenResults(p)


def r_in_bin(bin_filename, grs_raster, north, south, west, east, rows, cols, bytes=4 , overwrite=False, verbose=False, remove_bin=False, anull=-9999):
    ovwrt=''
    if overwrite:
        ovwrt='--o'
    
    cmd=["r.in.bin -f -s input=%s output=%s bytes=%d north=%f south=%f west=%f east=%f rows=%d cols=%d %s anull=%d" % \
         ( bin_filename, grs_raster, bytes, north, south, west, east, rows, cols, ovwrt, anull)]
    
    p = Popen(cmd,  shell=True,  stdout=PIPE, stderr=PIPE)
    res=processPopenResults(p)
    if verbose:
        logger.info((';'.join(cmd)))
        if len(res[1])>0:
            logger.info(res[1])
        if len(res[2])>0:
            logger.error(res[2])
            
    if remove_bin:
        cmd = ["rm "+bin_filename]
        p = Popen(cmd,  shell=True, stdout=PIPE, stderr=PIPE)
        res=processPopenResults(p)
    return True

#g.region wrapper
def g_region(w=None,e=None,s=None,n=None,res=None,rast=None):
    cmd_inp=["g.region"]
    if not (w is None):
        cmd_inp.append("w="+str(w))
    if not (e is None):
        cmd_inp.append("e="+str(e))
    if not (s is None):
        cmd_inp.append("s="+str(s))
    if not (n is None):
        cmd_inp.append("n="+str(n))
    if not (res is None):
        cmd_inp.append("res="+str(res))
    if not (rast is None):
        cmd_inp.append("rast="+rast)

    p = Popen(cmd_inp,  stdout=PIPE, stderr=PIPE)
    result=processPopenResults(p)
    return result

def g_region_bbox(bbox):
    return g_region(w=bbox.xmin,e=bbox.xmax,s=bbox.ymin,n=bbox.ymax,res=bbox.resolution)


def grass_import_data_for_bbox(map_name, bbox):
    temp_file=os.tmpnam() #to store bin fire exported from grass
    #export temp raster
    g_region_bbox(bbox)
    r_out_bin(map_name, temp_file)
    seg_pixels=bbox.width*bbox.height
    
    #check presence of temp.file
    if ((not os.path.exists(temp_file)) or (not os.path.isfile(temp_file))):
        print "Error: ", temp_file, "not found"
        return None
    
    #check file size
    fsize= os.path.getsize(temp_file)
    if (fsize < 1):
        print "Warning: empty file"
        return None
    fileobj = open(temp_file, mode='rb')
    if (fsize== (1*seg_pixels)):
        img = numpy.fromfile(file=fileobj, dtype = "uint8", count=seg_pixels)
    elif (fsize== (2*seg_pixels)):
        img = numpy.fromfile(file=fileobj, dtype = "int16", count=seg_pixels)
    elif (fsize== (4*seg_pixels)):
        img = numpy.fromfile(file=fileobj, dtype = "float32", count=seg_pixels)
    elif (fsize== (8*seg_pixels)):
        img = numpy.fromfile(file=fileobj, dtype = "float64", count=seg_pixels)
    else:
        print 'Error, grass export file size %d is not multiple of bbox pixels %d' % (fsize, seg_pixels)
        return None
    fileobj.close()

    os.remove(temp_file)

    img = img.reshape(bbox.height,bbox.width)
    return img



def grass_export_data(grs_file, data, bbox, anull=-9999):
#    temp_file=os.tmpnam() #to store bin fire exported from grass
    temp_file='/tmp/'+grs_file+'.bin'
    #export temp raster
    aux=data.astype(numpy.float32)
    aux[aux!=aux]=anull
    bin_file=temp_file
    fileobj = open(bin_file, mode='wb')
    aux.tofile(fileobj)
    fileobj.close()

    result=r_in_bin(bin_file, grs_file, bbox.ymax, bbox.ymin, bbox.xmin, bbox.xmax, rows=bbox.height, cols=bbox.width, bytes=4 , overwrite=True, verbose=False, remove_bin=True, anull=anull)
    
    return result
