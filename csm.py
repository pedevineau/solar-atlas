from proj4_navigation import ll2lc
from general_utils import latlon
import os

from netCDF4 import Dataset
import numpy as np

try:
    from numba import jit
    @jit
    def is_range_array(array, bmin, bmax):
        """is_01range_array"""
        for i in array.ravel():
            if (i <= bmax) or (i >= bmin):
                return False
        return True
except Exception as e:
    def is_range_array(array, bmin, bmax):
        """is_01range_array"""
        return np.all(array >= bmin) and np.all(array <= bmax)


if __name__ == '__main__':

    lonmin = 115.
    lonmax = 155.
    latmin = -30
    latmax = 60
    llres = 1 / 33.
    w = round(40 / llres)
    h = 90 / llres
    bb = latlon.bounding_box(lonmin, lonmax, latmin, latmax, w, h, llres)
    lats = bb.latitudes(array2d=True)
    lons = bb.longitudes(array2d=True)

    dir_in = '/data/test_data/BOM_cloud_data'
    dir_csp_out = '/data/test_data/BOM_clear_mask_latlon'
    dir_ct_out = '/data/test_data/BOM_cloud_type_latlon'
    nc_csp = 'clear_sky_probability'
    nc_ct = 'cloud_type'

    for file_ in (i for i in os.listdir(dir_in)):
        # file_ is just filename of .nc file in the directory (no dir in the string)
        if not "_CLD-" in file_:
            continue  # IGNORE not .nc files
        if not file_.endswith(".nc"):
            continue  # IGNORE not .nc files
        file_path = os.path.join(dir_in, file_)
       # print("Opening ", file_path)
        try:
            with Dataset(file_path) as ncF:
                dimensions = ncF.dimensions
                csp_ = ncF.variables[nc_csp]
                csp_data = csp_[:]
                if is_range_array(csp_data, bmin=0., bmax=1.):
                    newcsp = Dataset(os.path.join(dir_csp_out, file_.replace('P1S-ABOM_CLD-PRJ_GEOS141_2000', 'CSP_LATLON')), 'w', 'NETCDF4')
                    rectl, rectc = ll2lc(lats, lons, ssp=140.7, gt=(5500000, 2000, 0, 5500000, 0, -2000))
                    csp_latlon_data = csp_data[0, rectl, rectc]
                    newcsp.createDimension('t', None)
                    newcsp.createDimension('lat', csp_latlon_data.shape[0])
                    newcsp.createDimension('lon', csp_latlon_data.shape[1])
                    newcsp.createVariable(nc_csp, float, ('t', 'lat', 'lon',))
                    newcsp.variables[nc_csp][:] = csp_latlon_data
                    newcsp.close()
                else:
                    print("Clear-sky probability - Check this file! ", file_path)
                ct_ = ncF.variables[nc_ct]
                ct_data = ct_[:]
                if is_range_array(ct_data, bmin=0., bmax=10.):
                    newct = Dataset(os.path.join(dir_ct_out, file_.replace('P1S-ABOM_CLD-PRJ_GEOS141_2000', 'CT_LATLON')), 'w', 'NETCDF4')
                    newct.createVariable(nc_ct, ct_.datatype, ct_.dimensions)
                    rectl, rectc = ll2lc(lats, lons, ssp=140.7, gt=(5500000, 2000, 0, 5500000, 0, -2000))
                    newct.createDimension('t', None)
                    newct.createDimension('lat', csp_latlon_data.shape[0])
                    newct.createDimension('lon', csp_latlon_data.shape[1])
                    newct.createVariable(nc_csp, float, ('t', 'lat', 'lon',))
                    ct_latlon_data = csp_data[rectl, rectc]
                    newct.variables[nc_ct][:] = ct_latlon_data
                    newct.close()
                else:
                    print("Cloud type - Check this file! ", file_path)
        except IOError as e:
            print("Exception", file_path, type(e), e)
            os.remove(file_path)
            continue
        except Exception as e:
            print("Exception", file_path, type(e), e)
        break


