'''
Created on May 22, 2017

@author: tomas
'''
import datetime

from general_utils import daytimeconv
from general_utils import latlon
from himawari8.sat_model.utils import segmented_solar_geometry



if __name__ == "__main__":

    dfbStart = daytimeconv.ymd2dfb(2017, 1, 1)
    dfbEnd = daytimeconv.ymd2dfb(2017, 1, 15)
    
    #himawari 10-min data slots 1-144
    slotStart = 1
    slotEnd = 144
    
    
    bbox = latlon.bounding_box(xmin=150, xmax=155,ymin=35,ymax=40, width=150, height=150, resolution=2./60.)

    print 'start',datetime.datetime.now()
    UTC_dh_4D_slotcenter = segmented_solar_geometry.calculate_realscan_UTCtimes_slotcenter(dfbStart, dfbEnd, slotStart, slotEnd, bbox)
    h0_r_slotcenter, h0_r_ref_slotcenter, a0_r_slotcenter = segmented_solar_geometry.solargeom_core(bbox, dfbStart, dfbEnd, slotStart, slotEnd, UTC_dh_4D_slotcenter)
    print 'end',datetime.datetime.now()

    import pylab
    aux = h0_r_slotcenter[:,:,0,0].flatten()
    pylab.plot(aux)
    aux = a0_r_slotcenter[:,:,0,0].flatten()
    pylab.plot(aux)
    pylab.show()