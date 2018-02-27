from goesr.navigation.abi_navigation import ll2lc
from general_utils import latlon

import netCDF4 as nc
import pylab
if __name__ == '__main__':

    latmin = -50
    latmax = 60
    lonmin = -130
    lonmax = -10
    llres = 1 / 33.
    w = round(120 / llres)
    h = 110 / llres
    bb = latlon.bounding_box(lonmin, lonmax, latmin, latmax, w, h, llres)
    lats = bb.latitudes(array2d=True)
    lons = bb.longitudes(array2d=True)

    src_file = '/home/jano/Downloads/tmp/OR_ABI-L2-ACMF-M3_G16_s20180470000385_e20180470011151_c20180470011318.nc'
    with nc.Dataset(src_file) as ncd:
        v = ncd.variables['BCM'][:]
        rectl, rectc = ll2lc(lats, lons,lon0=-75, res_km=2)
        latlon_data = v[rectl,rectc]
        pylab.imshow(v, interpolation='nearest')
        pylab.show()
        latlon.visualize_map_2d(latlon_data,bbox=bb, color='gray')


