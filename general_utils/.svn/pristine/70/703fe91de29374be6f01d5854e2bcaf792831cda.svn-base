'''
Created on 23 May 2011

@author: Artur
'''
import sys
import numpy as np
from scipy.interpolate import interp1d

from general_utils import solar_geom_v5
from general_utils import mounting_geom
from general_utils import latlon_nctools


pi2=2*np.pi

# Variant of the script for 2 Rows, with x,y,z coordinates and  rotation matrix for 1 given day 
def row_mtx_fxed(GHI,DHI,DNI,DiffHor,lati,a0,h0refr,sina0,cosa0,sinh0,cosh0,sinDeltaexp,sinGammaN,cosGammaN,\
                        sinTiltSN,cosTiltSN, module_x,module_y,spacing_y,shadow,vert_sect=11,hor_sect=3,Albedo=None,AlterAzimuth=False):

    lati_r=np.radians(lati)
    if lati_r<0:
        a0tmp=a0.copy()
        a0tmp[a0>0]=np.pi-a0[a0>0]
        a0tmp[a0<0]=-np.pi-a0[a0<0]
        a0=a0tmp
    tana0=np.tan(a0)

    linX=np.linspace(-module_x/2.,module_x/2.,hor_sect)
    linY=np.linspace(-module_y/2.,module_y/2.,vert_sect)
    
    MTX_X,MTX_Y=np.meshgrid(linX,linY)
    lenY=len(linY)
    lenX=len(linX)
    M=np.zeros((lenY,lenX,3))
    center=int(hor_sect/2)

    M[:,:,0]=MTX_X
    M[:,:,1]=MTX_Y
    RM=mounting_geom.rotXmatrix(M.copy(),sinGammaN[0],cosGammaN[0]) 
    NCornersMatrix=mounting_geom.neighbour_corners_rows(RM,spacing_y,sinTiltSN/cosTiltSN)

    kx=NCornersMatrix[0,0]
    ky=NCornersMatrix[0,1]
    kz=NCornersMatrix[0,2]
    
    px=NCornersMatrix[1,0]
    py=NCornersMatrix[1,1]
    pz=NCornersMatrix[1,2]
     
    sinDeltaexp_M=np.reshape(np.tile(sinDeltaexp,lenX*lenY),(DNI.shape[0],lenY,lenX),order='F')
    GHI_M=np.reshape(np.tile(GHI,lenX*lenY),(DNI.shape[0],lenY,lenX),order='F')
    DNI_M=np.reshape(np.tile(DNI,lenX*lenY),(DNI.shape[0],lenY,lenX),order='F')
    DHI_M=np.reshape(np.tile(DHI,lenX*lenY),(DNI.shape[0],lenY,lenX),order='F')
    DiffHor_M=np.reshape(np.tile(DiffHor,lenX*lenY),(DNI.shape[0],lenY,lenX),order='F')
    h0refr_M=np.reshape(np.tile(h0refr,lenX*lenY),(DNI.shape[0],lenY,lenX),order='F')
    a0_M=np.reshape(np.tile(a0,lenX*lenY),(DNI.shape[0],lenY,lenX),order='F')
    tana0_M=np.reshape(np.tile(tana0,lenX*lenY),(DNI.shape[0],lenY,lenX),order='F')
    shadow_M=np.reshape(np.tile(np.zeros_like(a0),lenX*lenY),(shadow.shape[0],lenY,lenX),order='F')
    
    ox=(RM[:,:,0])
    oy=(RM[:,:,1])
    oz=(RM[:,:,2])

    minX=(px-ox)
    minY=(py-oy)
    maxX=(kx-ox)
    maxY=(ky-oy)
    min_a0=np.arctan2(minX,minY)
    max_a0=np.arctan2(maxX,maxY)
   
    a=(tana0_M*(py-oy)+ox-px)/(kx-px-tana0_M*(ky-py))
    X=((kx-px)*a+px-ox)
    Y=((ky-py)*a+py-oy)
    Z=((kz-pz)*a+pz-oz)
    D=np.sqrt(Y**2+Z**2)

    h0_row=np.arcsin(Z/D)
    horNULL=h0_row.copy()
    cond=(a0_M<min_a0)|(a0_M>max_a0)
    h0_row[cond]=0
    shadow_M[h0refr_M<h0_row]=1

    horNULL[horNULL<0]=0

    DiffInc_M=np.zeros_like(DHI_M)
    ReflInc_M=np.zeros_like(DHI_M)
    
    del(GHI_M, h0refr_M, a0_M, tana0_M, min_a0, max_a0, a, X, Y, Z, D, h0_row, cond) 
         
    for iy in range(0,lenY): 
        for ix in range(0,lenX):
            DiffInc_M[:,iy,ix],ReflInc_M[:,iy,ix]=solar_geom_v5.diffinc_PerezFast_PV(DiffHor_M[:,iy,ix],DNI_M[:,iy,ix],DHI_M[:,iy,ix],h0refr,sinGammaN*np.ones_like(a0),\
                                                cosGammaN*np.ones_like(a0), Albedo, sinDeltaexp, shadow_M[:,iy,ix],shadow_an=horNULL[:,iy,ix], sinh0=sinh0) 
    DNI_M[shadow_M==1]=0.  
    DInc_M= DNI_M*sinDeltaexp_M 
 
    DiffInc,ReflInc = solar_geom_v5.diffinc_PerezFast_PV(DiffHor,DNI, DHI,h0refr,sinGammaN*np.ones_like(a0),cosGammaN*np.ones_like(a0),Albedo,sinDeltaexp, shadow, \
                                    shadow_an=np.zeros_like(a0), sinh0=sinh0)    

    DInc= DNI*sinDeltaexp 

    ReflInc_M[:, :, 0]=ReflInc_M[:, :, 1]
    ReflInc_M[:, :, -1]=ReflInc_M[:, :, 1]
    DiffInc_M[:, :, 0]=DiffInc_M[:, :, 1]
    DiffInc_M[:, :, -1]=DiffInc_M[:, :, 1]
    DInc_M[:, :, 0]=DInc_M[:, :, 1]
    DInc_M[:, :, -1]=DInc_M[:, :, 1]
 
    return DInc_M, DiffInc_M, ReflInc_M, DInc, DiffInc, ReflInc


### Variant of the script for Tracker with 5 neighbours, with x,y,z coordinates and rotation matrix
def row_mtx_1xNS(GHI,DHI,DNI,DiffHor,lati_r,a0_r,h0refr_r,sina0,cosa0,sinh0,cosh0,sinDeltaexp,tilt,sinGammaNV,cosGammaNV,\
                        sinrot,cosrot,module_x,module_y,spacing_y,spacing_x,shadow=None, Albedo=None,AlterAzimuth=False):

    start_lst,end_lst,noon_lst=mounting_geom.find_start_end_lst(sinh0,sina0)
    lat_sign_n=np.sign(lati_r)
    # Horizontal NS (1xD) and inclined NS (1xB) Tracker rotation limit
    hor_sect=5 
    vert_sect=5
    rows_front=0
    rows_back=0
    rows_leftright=1
    
    DInc_M=np.zeros((len(GHI),hor_sect,vert_sect))
    DiffInc_M=np.zeros((len(GHI),hor_sect,vert_sect))
    ReflInc_M=np.zeros((len(GHI),hor_sect,vert_sect))
    if shadow==None: 
        shadow=np.zeros_like(GHI)

    Mref,MTX_Y,MTX_X=mounting_geom.prepare_reference_module(module_x,module_y,hor_sect,vert_sect)
    a0_r[a0_r<=0]=a0_r[a0_r<=0]+pi2

    sinGammaN=np.sin(tilt)
    cosGammaN=np.cos(tilt)
    noshadow=np.zeros_like(DHI)
    
    DiffInc_sh,ReflInc_sh=solar_geom_v5.diffinc_PerezFast(DiffHor,DNI,DHI,h0refr_r,sinGammaNV,cosGammaNV, Albedo, sinDeltaexp, \
                              shadow=np.ones_like(DNI),shadow_an=noshadow, sinh0=sinh0)
    DiffInc,ReflInc      =solar_geom_v5.diffinc_PerezFast(DiffHor,DNI,DHI,h0refr_r,sinGammaNV,cosGammaNV, Albedo, sinDeltaexp, \
                              shadow=shadow,shadow_an=noshadow, sinh0=sinh0)
    
    for idx_1 in range(len(start_lst)):
        for t_idx in range(start_lst[idx_1],end_lst[idx_1]): 
            M=Mref.copy()
            M=mounting_geom.rotYmatrix(M,-sinrot[t_idx],cosrot[t_idx])    
            M=mounting_geom.rotXmatrix(M,sinGammaN*lat_sign_n,cosGammaN)
            Ms=mounting_geom.prepare_3D_scene(M,rows_back,rows_front,rows_leftright,spacing_x,spacing_y,lat_sign_n)
            corner_pairs=mounting_geom.reduce_scene_to_corner_points(Ms,rows_back,rows_leftright)
            cpX= np.reshape(np.tile(corner_pairs[:,0],M.shape[0]*M.shape[1]),[corner_pairs.shape[0],M.shape[0],M.shape[1]],order='F')
            cpY= np.reshape(np.tile(corner_pairs[:,1],M.shape[0]*M.shape[1]),[corner_pairs.shape[0],M.shape[0],M.shape[1]],order='F')
            cpZ= np.reshape(np.tile(corner_pairs[:,2],M.shape[0]*M.shape[1]),[corner_pairs.shape[0],M.shape[0],M.shape[1]],order='F')
            cp=np.zeros((cpX.shape[0],cpX.shape[1],cpX.shape[2],3))
            cp[:,:,:,0]=cpX
            cp[:,:,:,1]=cpY
            cp[:,:,:,2]=cpZ
            a_r,h_r=mounting_geom.cartesian2polar_v(cp-M)
            a_r[a_r<=0]=a_r[a_r<=0]+pi2    
            sun=np.array((np.tile(a0_r[t_idx],a_r.shape[0]/4),np.tile(h0refr_r[t_idx],a_r.shape[0]/4)))
            point_v=np.array((a_r,h_r))
            p1v=point_v[:,:-3:4,:,:]
            p2v=point_v[:,1:-2:4,:,:]
            p3v=point_v[:,2:-1:4,:,:]
            p4v=point_v[:,3::4,:,:]
            sun_v=np.reshape(np.tile(sun,p1v.shape[2]*p1v.shape[3]),p1v.shape,order='F')
            shadow_detected=mounting_geom.check_point_position_v(p1v,p2v,p3v,p4v,sun_v).sum(0)
            shadow_detected[shadow_detected>1]=1
            no_shadow_detected=np.zeros_like(shadow_detected)
            no_shadow_detected[shadow_detected==0]=1
            DInc_M[t_idx,:,:]=float(DNI[t_idx])*no_shadow_detected*sinDeltaexp[t_idx]
            
            diff=DiffInc_sh[t_idx]*shadow_detected
            refl=ReflInc_sh[t_idx]*shadow_detected
            diff[shadow_detected==0]=DiffInc[t_idx]
            refl[shadow_detected==0]=ReflInc[t_idx]
            DiffInc_M[t_idx,:,:]=diff
            ReflInc_M[t_idx,:,:]=refl

    DInc= DNI*sinDeltaexp
    return DInc_M, DiffInc_M, ReflInc_M, DInc, DiffInc, ReflInc


def irrad2loss(EnergyDirect_MTX,EnergyDiffuse_MTX,EnergyDirectMax,EnergyDiffuseMax,BypassLoss):
        
        meanDirect=EnergyDirect_MTX.mean(axis=1).mean(axis=1)
        meanDiffuse=EnergyDiffuse_MTX.mean(axis=1).mean(axis=1)
        minDiffuse=EnergyDiffuse_MTX.min(axis=1).min(axis=1)
        minDirect=EnergyDirect_MTX.min(axis=1).min(axis=1)
        maxDiffuse=EnergyDiffuseMax
        maxDirect=EnergyDirectMax
        lossEle=((meanDiffuse+meanDirect-(meanDirect-minDirect)*BypassLoss-(meanDiffuse-minDiffuse)*BypassLoss)/(maxDiffuse+maxDirect))
        lossEle[np.isnan(lossEle)]=1
         
        lossMean=((meanDiffuse+meanDirect)/(maxDiffuse+maxDirect))
        lossMean[np.isnan(lossMean)]=1
          
        lossMax=((minDiffuse+minDirect)/(maxDiffuse+maxDirect))    
        lossMax[np.isnan(lossMax)]=1
        
        return lossEle, lossMean, lossMax
    
#used for inter-row shading
def calculate_shadow_losses(mounting,terrainSlope, terrainAzimuth, rel_spacing_rows,rel_spacing_columns, BypassLoss, GHI, DHI, DNI, DiffHor,Refl,\
                            lati, a0, h0refr, sina0, cosa0, sinh0, cosh0, tilt, sinGNV, cosGNV, sinDeltaexp, sinrot,cosrot,shadow, Albedo=None,AlterAzimuth=False):    
    #terrain Slope and Azimuth to  slope(tilt) in SN and EW direction   
    TiltWE_d, TiltSN_d = mounting_geom.inc_azim2ew_ns(terrainSlope, terrainAzimuth)
    cosTiltSN_d=np.cos(np.radians(TiltSN_d))
    sinTiltSN_d=np.sin(np.radians(TiltSN_d))
  
    if mounting==1:
        EnergyDirect_MTX, EnergyDiffuse_MTX, EnergyRefl_MTX, EnergyDirectMax, EnergyDiffuseMax, EnergyReflMax = row_mtx_fxed(\
                        GHI,DHI,DNI,DiffHor,lati,a0,h0refr,sina0,cosa0,sinh0,cosh0, sinDeltaexp,sinGNV,cosGNV,\
                        sinTiltSN_d,cosTiltSN_d,module_x=100,module_y=1,spacing_y=rel_spacing_rows,shadow=shadow, Albedo=Albedo,AlterAzimuth=AlterAzimuth)
        lossEle, lossMean, lossMax = irrad2loss(EnergyDirect_MTX, EnergyDiffuse_MTX+EnergyRefl_MTX, EnergyDirectMax, EnergyDiffuseMax+EnergyReflMax, BypassLoss)
    elif mounting==6 or mounting==7:
        EnergyDirect_MTX, EnergyDiffuse_MTX, EnergyRefl_MTX, EnergyDirectMax, EnergyDiffuseMax, EnergyReflMax = row_mtx_1xNS(\
                        GHI,DHI,DNI,DiffHor,lati,a0,h0refr,sina0,cosa0,sinh0,cosh0, sinDeltaexp,tilt,sinGNV,cosGNV,\
                        sinrot,cosrot,module_x=1,module_y=100,spacing_y=100,spacing_x=rel_spacing_columns,shadow=shadow, Albedo=Albedo,AlterAzimuth=AlterAzimuth)
    
        lossEle, lossMean, lossMax = irrad2loss(EnergyDirect_MTX, EnergyDiffuse_MTX+EnergyRefl_MTX, EnergyDirectMax, EnergyDiffuseMax+EnergyReflMax, BypassLoss)
    else:
        print "WARNING: Shading losses not supported for mounting 5, 8, 9"
        lossEle = np.ones_like(GHI)
        EnergyDirectMax = DNI.copy()
        EnergyDiffuseMax = DiffHor.copy()
        EnergyReflMa = Refl.copy()
        
        

    return EnergyDirectMax, EnergyDiffuseMax, EnergyReflMax, lossEle




# routines for calculation of shading for GTI delivery

def get_horizon(horizon_file_pattern, user_horizon, longit, latit):
    if user_horizon is not None:
        #use user defined horizon
        # User shading list has pairs of azimuth values 0 - 360 degs and heights 0 - 90 degs
        # 0 degrees means NORTH
        # it has the form of list of azim, height pairs  [(0.0, 5.0),(7.5, 3.0), (15.0, 7.0), (22.5, 0.0)]

        # first convert to np.array, radians, check ranges and sort (all can be removed if input is clear and well formated)
        try: 
            user_shad=np.array(user_horizon)
        except:
            raise Exception, "Unable to convert user horizon: "+ str(user_horizon)
        # Conversion to internal standard (-180) South = 0
        if user_shad.ndim !=2:
            raise Exception, "Wrong user horizon input: "+ str(user_horizon)
        AspectHor=user_shad[:,0]-180
        HeightHor=user_shad[:,1]
        AspectHor[AspectHor>180]-=360
        HeightHor[HeightHor>90.]=90.
        AspectHor[AspectHor<-180]+=360
        idxs=AspectHor.argsort()
        AspectHor=AspectHor[idxs]
        HeightHor=HeightHor[idxs]
        AspectHor = np.hstack((AspectHor[-1]-360,AspectHor ,AspectHor[0]+360))
        HeightHor = np.hstack((HeightHor[-1],HeightHor ,HeightHor[0]))
    
    else:
        #read horizon from file
        nc_file_name=latlon_nctools.horizon_file_name_create(horizon_file_pattern=horizon_file_pattern, longitude=longit, latitude=latit)
        HeightHor=latlon_nctools.read_horizon_nc_file_lonlat(nc_file_name,var_name='horizons',longit=longit, latit=latit)
        if HeightHor is None:
            raise Exception, "No horizon read from "+ str(nc_file_name)
        HeightHor=np.degrees(HeightHor)
        HorStep = 360./len(HeightHor)
#        AzimHor_r=np.arange(-pi,pi+.01,math.radians(HorStep))
        AspectHor=np.arange(-180,180+.01,HorStep)
        HeightHor = np.hstack((HeightHor ,HeightHor[0]))
        
    return (AspectHor,HeightHor)
        

def diffuse_isotropic_shading_meshgrids(skydomeAzimsNum=48, skydomeHeightsNum=60):
    AzimuthStep=(np.pi*2.)/skydomeAzimsNum
    HeightStep=(np.pi/2.)/skydomeHeightsNum
    h0s_r=np.arange((HeightStep/2.),np.pi/2.,HeightStep)
    a0s_r=np.arange(-np.pi+(AzimuthStep/2.),np.pi,AzimuthStep)
    
    sinh0s=np.sin(h0s_r)
    cosh0s=np.cos(h0s_r)
    
    sina0s=np.sin(a0s_r)
    cosa0s=np.cos(a0s_r)
    
    a0,h0=np.meshgrid(a0s_r,h0s_r)
    sina0,sinh0=np.meshgrid(sina0s,sinh0s)
    cosa0,cosh0=np.meshgrid(cosa0s,cosh0s)
    
    return a0,h0,sina0,cosa0,sinh0,cosh0

    

def prepare_shading(HorAspect, HorHeight, h0_sat, a0_sat, longit, latit):

    # interpolate horizon 
    # returns 
    # shading_direct - shading for direct component
    #     array of the length of all data (e.g. a0_sat) 
    #     0/1 values indicate if for given time there will be direct shading 1 - shading, 0 - no shading
   
    HorizonInterpolator = interp1d(HorAspect, HorHeight,kind=1)

    #shading for direct component and circumsolar diffuse
    horiz_sat=HorizonInterpolator(a0_sat)
    shading_direct=np.zeros_like(h0_sat)
    shading_direct=shading_direct.astype(np.int16)
    wh=h0_sat<horiz_sat
    shading_direct[wh]=1

    #shading for isotropic diffuse
    # define model of skydome and calculate which part is shaded. this is later used to calculate  
    skydomeAspectsNum=48
    skydomeHeightsNum=60
    skydome_a0_arr,skydome_h0_arr,skydome_sina0_arr,skydome_cosa0_arr,skydome_sinh0_arr,skydome_cosh0_arr = diffuse_isotropic_shading_meshgrids(skydomeAzimsNum=skydomeAspectsNum, skydomeHeightsNum=skydomeHeightsNum)
    
    skydomeAspectStep = 360./skydomeAspectsNum
    AspectHor=np.arange(-180.+skydomeAspectStep/2.,180.,skydomeAspectStep)
    horiz_isotrop_vect = HorizonInterpolator(AspectHor)
    horiz_isotrop_vect = np.radians(horiz_isotrop_vect)
    horiz_isotrop_arr = np.reshape(np.tile(horiz_isotrop_vect,skydomeHeightsNum),(skydomeHeightsNum,skydomeAspectsNum))
    
    skydome_shading_isotrop_arr = np.zeros((skydomeHeightsNum,skydomeAspectsNum),dtype=np.int16)
    
    wh = skydome_h0_arr < horiz_isotrop_arr
    skydome_shading_isotrop_arr[wh] = 1
    
    # shading must be weighted by incidence angle from skydome
    # cosh0_arr - used for pixel weight - with increasing h0 the pixel size is shorter in lon direction   
    # sina0_arr,cosa0_arr,sinh0_arr,cosh0_arr - used for calculation of the incidence angle from every segment of sky dome
    # calculation is done
    #    calculate skydome_incidence_angle_arr
    #    aux = skydome_incidence_angle_arr*cosh0_arr
    #    isotropic_shading_factor = ((1-skydome_shading_isotrop_arr)*aux).sum()/aux.sum()
    
    shading_data_dict = {}
    shading_data_dict['skydome_sina0_arr'] = skydome_sina0_arr
    shading_data_dict['skydome_cosa0_arr'] = skydome_cosa0_arr
    shading_data_dict['skydome_sinh0_arr'] = skydome_sinh0_arr
    shading_data_dict['skydome_cosh0_arr'] = skydome_cosh0_arr
    shading_data_dict['skydome_shading_isotrop_arr'] = skydome_shading_isotrop_arr
    shading_data_dict['skydome_a0_arr'] = skydome_a0_arr
    shading_data_dict['skydome_h0_arr'] = skydome_h0_arr
    shading_data_dict['shading_direct_vect'] = shading_direct
    return shading_data_dict 



def skydome_isotropic_shading_factor(shading_data_dict, sinGammaNV, cosGammaNV, sinANV, cosANV):

    if not shading_data_dict.has_key('skydome_cosh0_arr'):
        return 1.0
    
    skydome_sina0_arr = shading_data_dict['skydome_sina0_arr']
    skydome_cosa0_arr = shading_data_dict['skydome_cosa0_arr']
    skydome_sinh0_arr = shading_data_dict['skydome_sinh0_arr']
    skydome_cosh0_arr = shading_data_dict['skydome_cosh0_arr']
    skydome_shading_isotrop_arr = shading_data_dict['skydome_shading_isotrop_arr']

    skydome_incidence_angle_arr = mounting_geom.Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,skydome_sina0_arr,skydome_cosa0_arr,skydome_sinh0_arr,skydome_cosh0_arr)  # calculate incidence angle for all skydome pointsi
    aux = skydome_incidence_angle_arr * skydome_cosh0_arr
    isotropic_shading_factor = ((1.0-skydome_shading_isotrop_arr)*aux).sum()/aux.sum()

    return isotropic_shading_factor



def skydome_isotropic_shading_factor_tracker(shading_data_dict, GammaNV, ANV):
   
    if not shading_data_dict.has_key('skydome_cosh0_arr'):
        return 1.0

    # Function for calculation of isotropic diffuse reduction for moving collector surfaces - solar trackers
    # Becauase the position of solar tracker is changing with every time step the computational burden is significant
    # The function below first find the solution for every combination of azimuth/tilt (2 dimentional matrix)
    # but not for all values but for discrete bins (azimuth and tilt steps)  - look-up table is created
    # Having calculated values of all incident angles, given value of sky view fraction is determined by look-up table

    skydome_sina0_arr = shading_data_dict['skydome_sina0_arr']
    skydome_cosa0_arr = shading_data_dict['skydome_cosa0_arr']
    skydome_sinh0_arr = shading_data_dict['skydome_sinh0_arr']
    skydome_cosh0_arr = shading_data_dict['skydome_cosh0_arr']
    skydome_shading_isotrop_arr = shading_data_dict['skydome_shading_isotrop_arr']

    timeseries_len=len(GammaNV)
    dome_len=skydome_sina0_arr.shape[0]*skydome_sina0_arr.shape[1]

    # This part of code prepares grid points for Azimuths and Tilts.
    # The density of grid is determined by step variable 
    step=np.pi/48.
    GNs_tracker=np.arange(0,np.pi/2.+step,step)+step/2.
    ANs_tracker=np.arange(-np.pi,np.pi+step,step)+step/2.
    skydome_incidence_angle_arr=np.zeros((len(ANs_tracker),len(GNs_tracker),dome_len)).astype(np.float64)
    aux=np.zeros_like(skydome_incidence_angle_arr).astype(np.float64)
    
    sinGNs=np.sin(GNs_tracker)
    cosGNs=np.cos(GNs_tracker)
    sinANs=np.sin(ANs_tracker)
    cosANs=np.cos(ANs_tracker)

    # The main loo for calculaton of incidence angles between every gridpoint of tracker Azimuths/Tilts and every gridpoint of skydome 
    for gi in range(len(GNs_tracker)):
        for ai in range(len(ANs_tracker)):
            skydome_cos_incidence_angle_arr = mounting_geom.Rel_sunpos_V1(sinGNs[gi],cosGNs[gi],sinANs[ai],cosANs[ai],skydome_sina0_arr,skydome_cosa0_arr,skydome_sinh0_arr,skydome_cosh0_arr)
            aux[ai,gi,:]=(skydome_cos_incidence_angle_arr * skydome_cosh0_arr).flatten()

    # Look-up table indices are created from real tracker Azimut/Tilt pairs
    GammaNVidx=np.round((GammaNV)/step,0).astype(np.int64)
    ANidx=np.round((ANV+np.pi)/step,0).astype(np.int64)

    # Isotropic 
    isotropic_shading_factor=np.zeros_like(GammaNV).astype(np.float64)
    aux2= (aux*(1-skydome_shading_isotrop_arr.flatten())).sum(2)
    aux=aux.sum(2)
    idx = range(timeseries_len)
    isotropic_shading_factor[idx]=aux2[ANidx[idx],GammaNVidx[idx]]/aux[ANidx[idx],GammaNVidx[idx]]

    return isotropic_shading_factor


#no inclination
def apply_shading(ghi_sat, dni_sat, h0_sat, shading_data_dict,Albedo):
    
    if shading_data_dict.has_key('shading_direct_vect'):
        shading_direct_vect = shading_data_dict['shading_direct_vect']
    else:
        shading_direct_vect = ghi_sat * 0.0 # no shading  
    #prepare data    
    dni_sh=dni_sat.copy()
    ghi_sh=ghi_sat.copy()
    dif_sh=ghi_sat.copy()
    wh=(dni_sat>-9) & (shading_direct_vect==1)
    dni_sh[wh]=0
    wh=(ghi_sat>=0) & (dni_sat >= -9) & (h0_sat > 0)
    cos_incidence_angle_input=np.sin(np.radians(h0_sat[wh]))
    dhi=dni_sat[wh]*cos_incidence_angle_input
    dif_sh[wh]=ghi_sat[wh] - dhi

    #isotropic shading factor for 
    sinGammaNV, cosGammaNV, sinANV, cosANV = 0.0, 1.0, 0.0, 1.0
    isotropic_shading_factor = skydome_isotropic_shading_factor(shading_data_dict, sinGammaNV, cosGammaNV, sinANV, cosANV)
    print 'horizontal shading - isotropic_shading_factor', isotropic_shading_factor
    
    
    DiffInc, ReflInc = solar_geom_v5.diffinc_PerezFast(DiffHorInput=dif_sh[wh], B0c=dni_sat[wh], BHc=dhi, h0=np.radians(h0_sat[wh]), sinGammaN=0, cosGammaN=1, \
                                                       Albedo=Albedo, sinDeltaexpInput=cos_incidence_angle_input, shadow=shading_direct_vect[wh], isotropic_shading_factor=isotropic_shading_factor)
    DiffInc=DiffInc.astype(np.float32)
    ReflInc=ReflInc.astype(np.float32)
    dif_sh[wh]=DiffInc
    ghi_sh[wh]=DiffInc + ReflInc + dni_sh[wh] * cos_incidence_angle_input
    
    return ghi_sh, dni_sh, dif_sh


def apply_shading_inplane(ghi_sat, dni_sat, h0_sat, a0_sat , shading_data_dict, mounting, aspect, inclination, latit, Albedo=0.11, radiation_config_dict=None, pvParamDict=None):

    if shading_data_dict.has_key('shading_direct_vect'):
        shading_direct_vect = shading_data_dict['shading_direct_vect']
    else:
        shading_direct_vect = ghi_sat * 0.0 # no shading  
    #prepare data
    dni_sh=dni_sat.copy()
    gti_sh=ghi_sat.copy()
    dif_sh=ghi_sat.copy()
    inc_sh=ghi_sat.copy()
    asp_sh=ghi_sat.copy()
    tilt_sh=ghi_sat.copy()
    inc_sh[:]=np.nan
    asp_sh[:]=np.nan
    tilt_sh[:]=np.nan
    wh=(dni_sat>-9) & (shading_direct_vect==1) 
    dni_sh[wh]=0
    wh=(ghi_sat>=0) & (dni_sat >= -9) & (h0_sat > 0)
    h0=np.radians(h0_sat[wh])
    a0=np.radians(a0_sat[wh])
    sinh0=np.sin(h0)
    cosh0=np.cos(h0)
    sina0=np.sin(a0)
    cosa0=np.cos(a0)
    dhi=dni_sat[wh]*sinh0
    dif_sh[wh]=ghi_sat[wh] - dhi

    if (radiation_config_dict is not None) and (pvParamDict is not None):
        #parse parameters
        result = mounting_geom.PV_extract_mounting_details(radiation_config_dict, pvParamDict)
        mounting, modul_asp, modul_incl, terrainSlope,  terrainAzimuth, do_interrow_shading,  do_backtrack, relative_spacing_columns, relative_spacing_rows, rotation_limit_EW, rotation_limit_TILT = result
        #calculate module geometry
        incidence_angle, cos_incidence_angle, GammaNV, sinGammaNV,cosGammaNV,ANV, sinANV, cosANV ,rot, sinrot, cosrot = \
        mounting_geom.mounting_geom_angles(mounting, sina0, cosa0, a0, sinh0, cosh0, h0, GN=modul_incl, AN=modul_asp, latitude=latit, rotation_limit_EW=rotation_limit_EW, rotation_limit_TILT=rotation_limit_TILT,  backtrack=do_backtrack, relative_spacing_rows=relative_spacing_rows, relative_spacing_columns=relative_spacing_columns )
    else:
        modul_incl = np.radians(inclination) 
        modul_asp = np.radians(aspect)
        #calculate module geometry
        incidence_angle, cos_incidence_angle, GammaNV, sinGammaNV,cosGammaNV,ANV, sinANV, cosANV ,rot, sinrot, cosrot=\
        mounting_geom.mounting_geom_angles(mounting, sina0, cosa0, a0, sinh0, cosh0, h0, GN=modul_incl, AN=modul_asp, latitude=latit, rotation_limit_EW=[-np.pi*2.,np.pi*2.], rotation_limit_TILT=[-np.pi,np.pi], backtrack=False, relative_spacing_rows=1e10, relative_spacing_columns=1e10)

    #isotropic shading factor for 
    if mounting == 1: 
        sinGammaNV = np.sin(modul_incl)
        cosGammaNV = np.cos(modul_incl)
        sinANV = np.sin(modul_asp)
        cosANV = np.cos(modul_asp)

        isotropic_shading_factor = skydome_isotropic_shading_factor(shading_data_dict, sinGammaNV, cosGammaNV, sinANV, cosANV)
        print 'inclined shading - isotropic_shading_factor', isotropic_shading_factor
    else:
        isotropic_shading_factor = skydome_isotropic_shading_factor_tracker(shading_data_dict, GammaNV, ANV)
        print 'inclined shading - isotropic_shading_factor', isotropic_shading_factor.mean()

        
    #calculate inclined diffuse and reflected
    DiffInc, ReflInc = solar_geom_v5.diffinc_PerezFast(DiffHorInput=dif_sh[wh], B0c=dni_sat[wh], BHc=dhi, h0=h0, sinGammaN=sinGammaNV, \
                                                       cosGammaN=cosGammaNV, Albedo=Albedo, sinDeltaexpInput=cos_incidence_angle, shadow=shading_direct_vect[wh], isotropic_shading_factor=isotropic_shading_factor)

    DiffInc=DiffInc.astype(np.float32)
    ReflInc=ReflInc.astype(np.float32)
    dif_sh[wh]=DiffInc
    gti_sh[wh]=DiffInc + ReflInc + dni_sh[wh] * cos_incidence_angle

    inc_sh[wh]=np.degrees(incidence_angle)
    asp_sh[wh]=np.degrees(ANV)
    tilt_sh[wh]=np.degrees(GammaNV)
    
    return gti_sh, dni_sh, dif_sh, inc_sh, tilt_sh, asp_sh




def apply_noshading_inplane(ghi_sat, dni_sat, h0_sat, a0_sat , shading_data_dict, mounting, aspect, inclination, latit, Albedo=0.11, radiation_config_dict=None, pvParamDict=None):

    shading_direct_vect = ghi_sat * 0.
    isotropic_shading_factor = 1.0
    #prepare data
    dni_sh=dni_sat.copy()
    gti_sh=ghi_sat.copy()
    dif_sh=ghi_sat.copy()
    inc_sh=ghi_sat.copy()
    tilt_sh=ghi_sat.copy()
    asp_sh=ghi_sat.copy()
    inc_sh[:]=np.nan
    tilt_sh[:]=np.nan
    asp_sh[:]=np.nan
    wh=(dni_sat>-9) & (shading_direct_vect==1) 
    dni_sh[wh]=0
    wh=(ghi_sat>=0) & (dni_sat >= -9) & (h0_sat > 0)
    h0=np.radians(h0_sat[wh])
    a0=np.radians(a0_sat[wh])
    sinh0=np.sin(h0)
    cosh0=np.cos(h0)
    sina0=np.sin(a0)
    cosa0=np.cos(a0)
    dhi=dni_sat[wh]*sinh0
    dif_sh[wh]=ghi_sat[wh] - dhi

    if (radiation_config_dict is not None) and (pvParamDict is not None):
        #parse parameters
        result = mounting_geom.PV_extract_mounting_details(radiation_config_dict, pvParamDict)
        mounting, modul_asp, modul_incl, terrainSlope,  terrainAzimuth, do_interrow_shading,  do_backtrack, relative_spacing_columns, relative_spacing_rows, rotation_limiti_EW, rotation_limit_TILT = result
        #calculate module geometry
        incidence_angle, cos_incidence_angle, GammaNV, sinGammaNV,cosGammaNV, ANV, sinANV, cosANV, rot ,sinrot, cosrot= \
        mounting_geom.mounting_geom_angles(mounting, sina0, cosa0, a0, sinh0, cosh0, h0, GN=modul_incl, AN=modul_asp, latitude=latit, rotation_limit_EW=rotation_limiti_EW, rotation_limit_TILT=rotation_limit_TILT,  backtrack=do_backtrack, relative_spacing_rows=relative_spacing_rows, relative_spacing_columns=relative_spacing_columns )
    else:
        modul_incl = np.radians(inclination) 
        modul_asp = np.radians(aspect)
        #calculate module geometry
        incidence_angle, cos_incidence_angle, GammaNV, sinGammaNV,cosGammaNV,ANV, sinANV, cosANV ,rot, sinrot, cosrot=\
        mounting_geom.mounting_geom_angles(mounting, sina0, cosa0, a0, sinh0, cosh0, h0, GN=modul_incl, AN=modul_asp, latitude=latit, rotation_limit_EW=np.pi*2., rotation_limit_TILT=[-np.pi,np.pi], backtrack=False, relative_spacing_rows=1e10, relative_spacing_columns=1e10)

           

    #calculate inclined diffuse and reflected
    DiffInc, ReflInc = solar_geom_v5.diffinc_PerezFast(DiffHorInput=dif_sh[wh], B0c=dni_sat[wh], BHc=dhi, h0=h0, sinGammaN=sinGammaNV, \
                                                       cosGammaN=cosGammaNV, Albedo=Albedo, sinDeltaexpInput=cos_incidence_angle, shadow=shading_direct_vect[wh], isotropic_shading_factor=isotropic_shading_factor)
    DiffInc=DiffInc.astype(np.float32)
    ReflInc=ReflInc.astype(np.float32)
    dif_sh[wh]=DiffInc
    gti_sh[wh]=DiffInc + ReflInc + dni_sh[wh] * cos_incidence_angle
    inc_sh[wh]=np.degrees(incidence_angle)
    asp_sh[wh]=np.degrees(ANV)
    tilt_sh[wh]=np.degrees(GammaNV)
    
    return gti_sh, dni_sh, dif_sh, inc_sh, tilt_sh, asp_sh

