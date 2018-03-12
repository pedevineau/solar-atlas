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

    name = 'GOES16'

    if name == 'H08':
        ### Himawari08
        llres = 1 / 33.
        lonmin = 115.
        lonmax = 155.
        latmin = -30
        latmax = 60
        ssp = 140.7
        w = round((lonmax - lonmin) / llres)
        h = round((latmax - latmin) / llres)
        bb = latlon.bounding_box(lonmin, lonmax, latmin, latmax, w, h, llres)
        lats = bb.latitudes(array2d=True)
        lons = bb.longitudes(array2d=True)

    elif name == 'GOES16':
        ### GOES16
        latmin = -50
        latmax = 60
        lonmin = -130
        lonmax = -10
        llres = 1 / 33.
        lon0 = 75.2
        w = round((lonmax - lonmin) / llres)
        h = round((latmax - latmin) / llres)
        bb = latlon.bounding_box(lonmin, lonmax, latmin, latmax, w, h, llres)
        lats = bb.latitudes(array2d=False)
        lons = bb.longitudes(array2d=False)

    if name == 'H08':
        dir_in = '/data/test_data/BOM_cloud_data'
        dir_csp_out = '/data/test_data/BOM_clear_mask_latlon'
        dir_ct_out = '/data/test_data/BOM_cloud_type_latlon'
        nc_csp = 'clear_sky_probability'
        nc_ct = 'cloud_type'

        import proj4_navigation

        for file_ in (i for i in os.listdir(dir_in)):
            # file_ is just filename of .nc file in the directory (no dir in the string)
            if not "_CLD-" in file_:
                continue  # IGNORE not .nc files
            if not file_.endswith(".nc"):
                continue  # IGNORE not .nc files
            file_path = os.path.join(dir_in, file_)
            print("Opening ", file_path)
            try:
                with Dataset(file_path) as ncF:
                    dimensions = ncF.dimensions
                    csp_ = ncF.variables[nc_csp]
                    csp_data = csp_[:]
                    if is_range_array(csp_data, bmin=0., bmax=1.):
                        newcsp = Dataset(os.path.join(dir_csp_out, file_.replace('P1S-ABOM_CLD-PRJ_GEOS141_2000', 'CSP_LATLON')), 'w', 'NETCDF4')
                        rectl, rectc = proj4_navigation.ll2lc(lats, lons, ssp=ssp, gt=(-5500000, 2000, 0, 5500000, 0, -2000))
                        csp_latlon_data = csp_data[0, rectl, rectc]
                        # t = newcsp.createDimension('t', 1)
                        lat = newcsp.createDimension('lat', csp_latlon_data.shape[0])
                        lon = newcsp.createDimension('lon', csp_latlon_data.shape[1])
                        newcsp.createVariable(nc_csp, float, ('lat', 'lon',))
                        newcsp.variables[nc_csp][:] = csp_latlon_data
                        newcsp.close()
                    else:
                        print("Clear-sky probability - Check this file! ", file_path)
                    ct_ = ncF.variables[nc_ct]
                    ct_data = ct_[:]
                    if is_range_array(ct_data, bmin=0., bmax=10.):
                        newct = Dataset(os.path.join(dir_ct_out, file_.replace('P1S-ABOM_CLD-PRJ_GEOS141_2000', 'CT_LATLON')), 'w', 'NETCDF4')
                        rectl, rectc = proj4_navigation.ll2lc(lats, lons, ssp=140.7, gt=(-5500000, 2000, 0, 5500000, 0, -2000))
                        ct_latlon_data = ct_data[0, rectl, rectc]
                        # t = newct.createDimension('t', 1)
                        lat = newct.createDimension('lat', ct_latlon_data.shape[0])
                        lon = newct.createDimension('lon', ct_latlon_data.shape[1])
                        newct.createVariable(nc_ct, int, ('lat', 'lon',))
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

    elif name == 'GOES16':
        dir_in = '/data/test_data/goes_clear_mask'
        dir_csp_out = '/data/test_data/GOES_clear_mask_latlon'
        nc_csp = 'BCM'

        from general_utils.daytimeconv import doyy2yyyymmdd
        import abi_navigation

        for file_ in (i for i in os.listdir(dir_in)):
            # file_ is just filename of .nc file in the directory (no dir in the string)
            if not file_.endswith(".nc"):
                continue  # IGNORE not .nc files
            file_path = os.path.join(dir_in, file_)
            print("Opening ", file_path)
            try:
                with Dataset(file_path) as ncF:
                    dimensions = ncF.dimensions
                    csp_ = ncF.variables[nc_csp]
                    csp_data = csp_[:]
                    if is_range_array(csp_data, bmin=0., bmax=1.):
                        rad = file_.replace('OR_ABI-L2-ACMF-M3_G16_s', '')
                        year = rad[:4]
                        doy = rad[4:7]
                        hour = rad[7:9]
                        minu = rad[9:11]
                        str_date = doyy2yyyymmdd(int(doy), int(year)) + hour + minu + '00'
                        str_date = str_date + '-CSP_LATLON-GOES16.nc'
                        newcsp = Dataset(os.path.join(dir_csp_out, str_date), 'w', 'NETCDF4')
                        rectl, rectc = abi_navigation.ll2lc(lats, lons, lon0=lon0)
                        csp_latlon_data = csp_data[0, rectl, rectc]
                        lat = newcsp.createDimension('lat', csp_latlon_data.shape[0])
                        lon = newcsp.createDimension('lon', csp_latlon_data.shape[1])
                        newcsp.createVariable('clear_sky_probability', float, ('lat', 'lon',))
                        newcsp.variables['clear_sky_probability'][:] = csp_latlon_data
                        newcsp.close()
                    else:
                        print("Clear-sky probability - Check this file! ", file_path)
            except IOError as e:
                print("Exception", file_path, type(e), e)
                os.remove(file_path)
                continue
            except Exception as e:
                print("Exception", file_path, type(e), e)



