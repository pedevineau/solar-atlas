#! /usr/bin/env python
#encoding:UTF-8
'''
'''

import numpy
from general_utils import mounting_geom
from general_utils import daytimeconv
from general_utils import solar_geom_v5
from general_utils import pv_utils
from general_utils import shading_utils 
import pickle
import scipy.interpolate as scinterp

#import pv_utils_diode as pv_utils

sto=100.

modes=[\
['Canadian_Solar_CS5A_195M']]


def PV_extract_terrainShading_details(rad_configDict):
    #force the shading to False  if radiation_config_dict not found
    try:
        shading_params=rad_configDict['shading_params']
        do_shading=shading_params['shading']
    except:
        do_shading=False
        
    if do_shading:
#        print >> sys.stderr, ' reading horizon'
        userHorizon=shading_params['horizon']
        horizon_file_pattern=shading_params['file_pattern']
    else:
        userHorizon = None
        horizon_file_pattern = None
    return do_shading, userHorizon, horizon_file_pattern

def remnan(m):
    m[numpy.isnan(m)]=0
    return m


def PV_performance(data_3D_array, ekeys, pvDict, rad_configDict, longitude,  latitude, dfb_begin, dfb_end,process_STC_25=False):

    keys_auxilary=['GTI_nshd','GTI_hor','GTI_ang','TEMP_mod','PVOUT_stc','PVOUT_dc','PVOUT_shd_el','PVOUT_rot']

    if pvDict['pvModuleTechnology'] == 'CDTE':
       keys_required=['TEMP','SE','SA','DNI_raw','GHI_raw','PVOUT','PWAT']
    else:
       keys_required=['TEMP','SE','SA','DNI_raw','GHI_raw','PVOUT']

    ekeys.extend(keys_auxilary)

    # extend ourput array with n keys, for intermediate results
    col1= numpy.zeros((len(keys_auxilary),data_3D_array.shape[1],data_3D_array.shape[2]))
    data_3D_array=numpy.append(data_3D_array,col1,axis=0)

    idct={}
    keys_required.extend(keys_auxilary)
    for key in keys_required:
        idct[key]=ekeys.index(key)

    if pvDict['pvModuleTechnology'] == 'CDTE':
        PWAT=data_3D_array[idct['PWAT'],:,:]

    TempAmbient=data_3D_array[idct['TEMP'],:,:]
    h0_sat_deg=data_3D_array[idct['SE'],:,:]
    a0_sat_deg=data_3D_array[idct['SA'],:,:]
    a0_sat=numpy.radians(a0_sat_deg)
    h0_sat=numpy.radians(h0_sat_deg)

    DNI=data_3D_array[idct['DNI_raw'],:,:]
    GHI=data_3D_array[idct['GHI_raw'],:,:]

    #parser pvDict to internal variables
    result = pv_dict_2_internal_params(pvDict,data_3D_array,dfb_begin, dfb_end)
    
    Albedo, ar, TempCoeffPmax, ThermCoeff, dcLosses, acLosses, PVdegrad, BypassLoss, installedPower,\
    pvModuleTechnology,pvInverterEff,pvInverterLimitLow,pvInverterLimitHigh,pvInverterCount,pvAvailability = result
    #mounting params    
    result = mounting_geom.PV_extract_mounting_details(rad_configDict, pvDict)
    mounting, aspect_r, tilt_r, terrainSlope,  terrainAzimuth, do_interrow_shading,  do_backtrack, relative_spacing_columns, relative_spacing_rows, rotation_limit_EW, rotation_limit_TILT = result 
    print 'mounting', mounting_geom.mounting_geometrycode_to_name(mounting), '   pvFieldSelfShading',do_interrow_shading, '   pvTrackerBackTrack', do_backtrack
    
    ONES=numpy.ones_like(GHI)
    NANS=numpy.empty(GHI.shape,dtype=numpy.float64)
    NANS[:]=numpy.nan

    #terrain shading params
    result = PV_extract_terrainShading_details(rad_configDict)
    do_shading, userHorizon, horizon_file_pattern = result

    if do_shading:
        HorAspect, HorHeight = shading_utils.get_horizon(horizon_file_pattern, userHorizon, longitude, latitude)
        shading_data_dict = shading_utils.prepare_shading(HorAspect, HorHeight,   h0_sat_deg, a0_sat_deg, longit=longitude, latit=latitude)
    else:
        shading_data_dict = {}

    #calculate geometry   
    co1=(h0_sat>0)&(GHI>0)
    aux_h0_sat=h0_sat[co1]
    aux_a0_sat=a0_sat[co1]
    aux_sinh0=numpy.sin(aux_h0_sat)
    aux_cosh0=numpy.cos(aux_h0_sat)
    aux_sina0=numpy.sin(aux_a0_sat)
    aux_cosa0=numpy.cos(aux_a0_sat)
    aux_DNI=DNI[co1]
    aux_GHI=GHI[co1]
    aux_DHI=aux_DNI*aux_sinh0
    aux_DiffHor= aux_GHI-aux_DHI

    aux_PWAT=None
    if pvDict['pvModuleTechnology'] == 'CDTE':
        aux_PWAT=PWAT[co1]

    GammaNV,sinGNV,cosGNV,ANV,sinANV,cosANV,incidence_angle, cos_incidence_angle,sinrot,cosrot,rot=\
    NANS.copy(),NANS.copy(),NANS.copy(),NANS.copy(),NANS.copy(),NANS.copy(),NANS.copy(),NANS.copy(),NANS.copy(),NANS.copy(),NANS.copy()

    incidence_angle[co1], cos_incidence_angle[co1], GammaNV[co1], sinGNV[co1],cosGNV[co1], ANV[co1],sinANV[co1], cosANV[co1], rot[co1], sinrot[co1], cosrot[co1]=\
    mounting_geom.mounting_geom_angles(mounting,aux_sina0,aux_cosa0, aux_a0_sat, aux_sinh0,aux_cosh0, aux_h0_sat, GN=tilt_r, AN=aspect_r, latitude=latitude, rotation_limit_EW=rotation_limit_EW, rotation_limit_TILT=rotation_limit_TILT,  backtrack=do_backtrack, relative_spacing_rows=relative_spacing_rows, relative_spacing_columns=relative_spacing_columns )

    aux_cos_incidence_angle = cos_incidence_angle[co1] 
    aux_GammaNV=GammaNV[co1]
    aux_sinGNV=sinGNV[co1]
    aux_cosGNV=cosGNV[co1]
    aux_ANV=ANV[co1]
    aux_sinANV=sinANV[co1]
    aux_cosANV=cosANV[co1]
    aux_incidence_angle= incidence_angle[co1]
    aux_sinrot=sinrot[co1]
    aux_cosrot=cosrot[co1]
    aux_rot=rot[co1]

    if aux_sinrot is not None:
        rot[co1] =  aux_rot
        data_3D_array[idct['PVOUT_rot'],:,:] = rot
    if aux_sinrot is not None:
        sinrot[co1] =  aux_sinrot
    if aux_cosrot is not None:
        cosrot[co1] = aux_cosrot
    if do_shading:
        aux_shading_direct_vect = shading_data_dict['shading_direct_vect'][co1]
    else:
        aux_shading_direct_vect = numpy.zeros_like(aux_h0_sat)

    #isotropic shading factor for PV
    isotropic_shading_factor = 1.0
    if mounting ==1:
        if do_shading: 
            sinGammaNV_shd,cosGammaNV_shd,sinANV_shd,cosANV_shd = numpy.sin(tilt_r),numpy.cos(tilt_r),numpy.sin(aspect_r),numpy.cos(aspect_r)
            isotropic_shading_factor = shading_utils.skydome_isotropic_shading_factor(shading_data_dict, sinGammaNV_shd, cosGammaNV_shd, sinANV_shd, cosANV_shd)
    else:
        if do_shading:
            isotropic_shading_factor=shading_utils.skydome_isotropic_shading_factor_tracker(shading_data_dict, aux_GammaNV, aux_ANV)
    
    DTI_shd,DifTI_shd,RFL_shd = NANS.copy(),NANS.copy(),NANS.copy()
    DifTI_shd[co1],RFL_shd[co1]=solar_geom_v5.diffinc_PerezFast_PV(aux_DiffHor, aux_DNI, aux_DHI, aux_h0_sat, aux_sinGNV,aux_cosGNV, \
            Albedo, aux_cos_incidence_angle, aux_shading_direct_vect*0, shadow_an=None, sinh0=aux_sinh0, isotropic_shading_factor=1)
    DTI_shd[co1]=aux_DNI * aux_cos_incidence_angle
    data_3D_array[idct['GTI_nshd'],:,:] = remnan(DTI_shd+DifTI_shd+RFL_shd)

    DTI_shd,DifTI_shd,RFL_shd = NANS.copy(),NANS.copy(),NANS.copy()
    DifTI_shd[co1],RFL_shd[co1]=solar_geom_v5.diffinc_PerezFast_PV(aux_DiffHor, aux_DNI, aux_DHI, aux_h0_sat, aux_sinGNV,aux_cosGNV, \
            Albedo, aux_cos_incidence_angle, aux_shading_direct_vect, shadow_an=None, sinh0=aux_sinh0, isotropic_shading_factor=isotropic_shading_factor)
    
    DTI_shd[co1] = aux_DNI * aux_cos_incidence_angle*(1-aux_shading_direct_vect)
    data_3D_array[idct['GTI_hor'],:,:]=remnan(DTI_shd+DifTI_shd+RFL_shd)

    #interrow shading
    lossEle=ONES.copy()
    if (mounting==1):
        if not rad_configDict.has_key("tilt"):
            do_interrow_shading=False
    if do_interrow_shading:     
        DTI_shd[co1], DifTI_shd[co1], RFL_shd[co1], lossEle[co1] = shading_utils.calculate_shadow_losses(mounting,\
                               terrainSlope, terrainAzimuth,relative_spacing_rows,relative_spacing_columns,BypassLoss,\
                               aux_GHI, aux_DHI, aux_DNI, aux_DiffHor, RFL_shd[co1], numpy.radians(latitude), aux_a0_sat, aux_h0_sat,\
                               aux_sina0, aux_cosa0, aux_sinh0, aux_cosh0, tilt_r, aux_sinGNV, aux_cosGNV, aux_cos_incidence_angle,\
                               aux_sinrot, aux_cosrot, aux_shading_direct_vect, Albedo)

    #global tilted irradiation - with reduced angular losses
    DTI_ang,DifT_ang,RFL_ang=NANS.copy(),NANS.copy(),NANS.copy()
    DTI_ang[co1],DifT_ang[co1],RFL_ang[co1]=\
    solar_geom_v5.PV_Angular(DTI_shd[co1],DifTI_shd[co1],RFL_shd[co1],aux_cos_incidence_angle, aux_GammaNV, aux_sinGNV, aux_cosGNV, ar=ar)
    GTI_ang=DTI_ang+DifT_ang+RFL_ang
    data_3D_array[idct['GTI_ang'],:,:]=remnan(GTI_ang)

    #temperature of the module and irradiance PVOUT 
    TmpModule,PV1=NANS.copy(),ONES.copy()*0

    if process_STC_25==False:
        TmpModule[:,:]=25.
        PV1[co1]=pv_utils.PVModule(installedPower,GTI_ang[co1],TmpModule[co1],pvModuleTechnology,TempCoeffPmax,aux_PWAT)
        data_3D_array[idct['PVOUT_stc'],:,:]=remnan(PV1)

    TmpModule[co1]=TempAmbient[co1] + GTI_ang[co1] * ThermCoeff
    PV1[co1]=pv_utils.PVModule(installedPower,GTI_ang[co1],TmpModule[co1],pvModuleTechnology,TempCoeffPmax,aux_PWAT)
    print PV1[co1].sum()/4000.

    #with open('PVGIS_Coeff_Interpolator.pickle', 'rb') as handle:
    #    PVGIS_Coeff_Interpolator = pickle.load(handle)

    #for mod in modes:
    #    PV1=ONES.copy()*0
    #	Pmax_STC=pv_utils.get_pmax_STC(mod)
    #    PV1[co1] = installedPower * pv_utils_diode.get_curve_pmax(GTI_ang[co1],TmpModule[co1],mod)/float(Pmax_STC)
    #    lt=len(TmpModule[co1])
    #	T=TmpModule[co1]
    #	G=GTI_ang[co1]
    #	P=PV1[co1]
    # 	for i in range(0,lt):
    #  	    P[i]= P[i]/ PVGIS_Coeff_Interpolator(T[i], G[i])
    #	PV1[co1]=P
    #    print mod, PV1[co1].sum()/4000.
    #exit()

    data_3D_array[idct['PVOUT_dc'],:,:]=remnan(PV1)
    data_3D_array[idct['TEMP_mod'],:,:]=TmpModule

    #apply interrow shading
    PV1[co1] *= lossEle[co1]
    data_3D_array[idct['PVOUT_shd_el'],:,:]=remnan(PV1)

    #pv degradation
    PV1*=PVdegrad

    # DCMismatch  & DCCables & PollutionSnow as 1 or 12 numbers
    PV1*=dcLosses

    #inverter output   
    PV1_inv=NANS.copy()
    PV1_inv[co1]=pv_utils.Inverter(PV1[co1],pvInverterEff,pvInverterLimitLow,pvInverterLimitHigh, pvInverterCount)

    #ac losses
    PV1_inv*=acLosses
    PV1_inv*=pvAvailability
    data_3D_array[idct['PVOUT'],:,:]=remnan(PV1_inv)

    data_3D_array=data_3D_array.astype(numpy.float32)

    return data_3D_array


def pv_dict_2_internal_params(pvDict,data_3D_array,dfb_begin, dfb_end):

    pvModuleTechnology=pvDict["pvModuleTechnology"]
    dTCoeff=0.0

    #installed power is in kilowats
    installedPower=pvDict["pvInstalledPower"]

    #surface reflectnace coefficient
    if pvDict.has_key("pvModuleSurfaceReflectance"):
        ar=pvDict["pvModuleSurfaceReflectance"]# Martin Ruiz angular coefficient
    else:
        ar=0.16

    if pvDict.has_key("pvGroundAlbedo"):
        Albedo=pvDict["pvGroundAlbedo"]# ground albedo
    else:
        Albedo=0.2

    #module NOCT Temperature
    if not pvDict.has_key("pvModuleTempNOCT"):
        pvDict["pvModuleTempNOCT"] = pv_utils.TempNOCT_defaults_by_technology(pvModuleTechnology)
    #calculate ThermCoeff
    try:
        ThermCoeff=(pvDict["pvModuleTempNOCT"]-20)/800.
    except:
        print 'problem calculating ThermCoeff from "pvModuleTempNOCT" input paremeter. Using default 44'
        ThermCoeff=0.030
    #adapt ThermCoef
    if pvDict["pvInstallationType"] == 'BUILDING_INTEGRATED':
        ThermCoeff+=0.015
    elif pvDict["pvInstallationType"] == 'ROOF_MOUNTED':
        ThermCoeff+=0.01

    #TempCoeffPmax
    if pvDict.has_key("pvModuleTempCoeffPmax"):
        TempCoeffPmax = pvDict['pvModuleTempCoeffPmax']
    else:
        TempCoeffPmax = pv_utils.TempCoeffPmax_defaults_by_technology(pvModuleTechnology)

    #inverter
    if pvDict["pvInverterEffModel"]==".EfficiencyCurve":
        pvInverterEff=numpy.array(pvDict["pvInverterEffCurveDataPairs"])
    elif pvDict["pvInverterEffModel"]==".EfficiencyConstant":
        pvInverterEff=pvDict["pvInverterEffConstant"]
    else:
        pvInverterEff=pvDict.pvInverterEffConstant_Default

    if pvDict.has_key("pvInverterLimitationACPower"):
        pvInverterLimitHigh=pvDict["pvInverterLimitationACPower"]
    else:
        pvInverterLimitHigh=None
    if pvDict.has_key("pvInverterStartPower"):
        pvInverterLimitLow=pvDict["pvInverterStartPower"]
    else:
        pvInverterLimitLow=None

    if pvDict.has_key("pvInverterCount"):
        pvInverterCount=pvDict["pvInverterCount"]
    else:
        pvInverterCount=1

    if pvDict.has_key("pvLossesDCOther"):
        dcLosses=(1-pvDict["pvLossesDCOther"]/sto)
    else:
        if pvDict.has_key("pvLossesDCPollutionSnowMonth"):
            if pvDict.has_key("pvLossesDCPollutionSnow"):
                pvDict.pop("pvLossesDCPollutionSnow")
        else:
            if pvDict.has_key("pvLossesDCPollutionSnow"):
                pvDict["pvLossesDCPollutionSnowMonth"]=pvDict["pvLossesDCPollutionSnow"]*numpy.ones(12)
                pvDict.pop("pvLossesDCPollutionSnow")
            else:
                pvDict["pvLossesDCPollutionSnowMonth"]=pv_utils.pvLossesDCPollutionSnow_Default*numpy.ones(12)

        dcLossesPollutionSnowMonth=numpy.array(pvDict["pvLossesDCPollutionSnowMonth"])
        dcLossesPollutionSnowDaily=pv_utils.MonthlySnowLosses_2_DailySnowLosses(dfb_begin, dfb_end,dcLossesPollutionSnowMonth)
        dim0 = data_3D_array.shape[1]
        dim1 = data_3D_array.shape[2]
        dcLossesPollutionSnowDaily=numpy.repeat(dcLossesPollutionSnowDaily, dim1).reshape(dim0, dim1)

        if (pvDict.has_key("pvLossesDCMismatch") or pvDict.has_key("pvLossesDCCables")):
            if not pvDict.has_key("pvLossesDCMismatch"):
                pvDict["pvLossesDCMismatch"] = pv_utils.pvLossesDCMismatch_Default
            if not pvDict.has_key("pvLossesDCCables"):
                pvDict["pvLossesDCCables"] = pv_utils.pvLossesDCCables_Default
        else:
            pvDict["pvLossesDCMismatch"] = pv_utils.pvLossesDCMismatch_Default
            pvDict["pvLossesDCCables"] = pv_utils.pvLossesDCCables_Default
        dcLosses=(1-pvDict["pvLossesDCMismatch"]/sto)*(1-pvDict["pvLossesDCCables"]/sto)*(1-dcLossesPollutionSnowDaily/sto)


    pvAvailability=1.
    if pvDict.has_key("pvAvailability"):
        pvAvailability=(pvDict["pvAvailability"]/sto)


    if (pvDict.has_key("pvLossesACTransformer") or pvDict.has_key("pvLossesACCable")):
        if not pvDict.has_key("pvLossesACTransformer"):
            pvDict["pvLossesACTransformer"]=pv_utils.pvLossesACTransformer_Default
        if not pvDict.has_key("pvLossesACCable"):
            pvDict["pvLossesACCable"]=pv_utils.pvLossesACCable_Default
    if (pvDict.has_key("pvLossesACTransformer") and pvDict.has_key("pvLossesACCable")):
        acLosses=(1-pvDict["pvLossesACTransformer"]/sto)*(1-pvDict["pvLossesACCable"]/sto)
    elif pvDict.has_key("pvLossesAC"):
        acLosses=(1-pvDict["pvLossesAC"]/sto)
    else:
        if (pvDict["pvInstallationType"] == 'BUILDING_INTEGRATED') or (pvDict["pvInstallationType"] == 'ROOF_MOUNTED'):
            acLosses=1-(pv_utils.pvLossesAC_Default_building/sto) # no losses
        else:
            acLosses=1-(pv_utils.pvLossesAC_Default/sto) # no losses

    if pvDict.has_key("pvModuleDegradation") and pvDict.has_key("dateStartup"):
        dfbStartup = daytimeconv.yyyymmdd2dfb(pvDict["dateStartup"])
        pvModuleDegradation = pvDict["pvModuleDegradation"]
        if pvDict.has_key("pvModuleDegradationFirstYear"):
            pvModuleDegradationFirstYear = pvDict["pvModuleDegradationFirstYear"]
        else:
            pvModuleDegradationFirstYear = pvModuleDegradation
        PVdegradDfbs = pv_utils.PVdegrad_for_dfbs(dfb_begin, dfb_end, dfbStartup, pvModuleDegradationFirstYear, pvModuleDegradation)
        dim0 = data_3D_array.shape[1]
        dim1 = data_3D_array.shape[2]
        PVdegrad=numpy.repeat(PVdegradDfbs, dim1).reshape(dim0, dim1)
    else:
        PVdegrad=0.0
    PVdegrad = 1.-(PVdegrad/sto)

    #influence of loss
    if pvDict.has_key("pvFieldTopologyType"):
        BypassLoss=pv_utils.topology_type_to_shading_influence(pvDict["pvFieldTopologyType"])
    else:
        if pvModuleTechnology=='CSI':
            BypassLoss=pv_utils.topology_type_to_shading_influence('UNPROPORTIONAL_1')
        else:
            BypassLoss=pv_utils.topology_type_to_shading_influence('PROPORTIONAL')

    return Albedo, ar, TempCoeffPmax, ThermCoeff, dcLosses, acLosses, PVdegrad, BypassLoss, installedPower,pvModuleTechnology,pvInverterEff,pvInverterLimitLow,pvInverterLimitHigh,pvInverterCount,pvAvailability



