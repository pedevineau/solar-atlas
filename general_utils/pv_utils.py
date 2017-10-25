'''
Created on Jul 5, 2012
author: tomas
'''

import numpy
from general_utils import daytimeconv


pvInstallationType_allowed = ['BUILDING_INTEGRATED', 'ROOF_MOUNTED', 'FREE_STANDING']
pvModuleTechnology_allowed = ['CSI', 'ASI', 'CDTE', 'CIS']
pvInverterInterconnection_allowed = ['SERIAL', 'PARALLEL', 'SERIOPARALLEL']
pvInverterEffModel_allowed = [".EfficiencyConstant", ".EfficiencyCurve"]
pvFieldModelType_allowed = [".TopologySimple", ".TopologyExact", ".TopologyShadingAngle"]
pvFieldTopologyType_allowed = ['PROPORTIONAL', 'UNPROPORTIONAL_1', 'UNPROPORTIONAL_2', 'UNPROPORTIONAL_3']

pvModuleSurfaceReflectance_Default=0.16
pvModuleDegradation_Default=0.5
pvModuleDegradationFirstYear_Default=0.8

pvInverterEffConstant_Default = 97.5

pvLossesDC_Default = 5.4
pvLossesDCMismatch_Default = 1.0
pvLossesDCCables_Default =2.0
pvLossesDCPollutionSnow_Default = 2.5 

pvLossesAC_Default=1.5
pvLossesAC_Default_building=0.0
pvLossesACCable_Default = 0.5
pvLossesACTransformer_Default = 1.0 
mandatory_keys  = ['pvInstallationType', 'pvInstalledPower', 'pvModuleTechnology' ]



def topology_type_to_shading_influence(topology_type_enum):
    #interrow shading influence in % - 0-no influence of shading 100- full influence of shading    
    topology_type=topology_type_enum.lower().strip().encode('ascii', 'ignore')
    if topology_type == 'proportional': 
        shading_influence=.20
    elif topology_type == 'unproportional_1': 
        shading_influence=.40
    elif topology_type == 'unproportional_2': 
        shading_influence=.60
    elif topology_type == 'unproportional_3': 
        shading_influence=.80
    else:
        shading_influence=.05
    return shading_influence


def ModulePowerToleranceLow_defaults_by_technology(pvModuleTechnology):
    if pvModuleTechnology=="CSI": # Crystalline Silicon
        PowerToleranceLow_def = 3
#    elif pvModuleTechnology=="ASI":# Amorphous Silicon
#        PowerToleranceLow_def = 5
#    elif pvModuleTechnology=="CDTE":#CdTe
#        PowerToleranceLow_def = 5
#    elif  pvModuleTechnology=="CIS":#CIS
#        PowerToleranceLow_def = 5
    else:
        PowerToleranceLow_def = 5
    return PowerToleranceLow_def
    
def TempNOCT_defaults_by_technology(pvModuleTechnology):
    if pvModuleTechnology=="CSI": # Crystalline Silicon
        TempNOCT_def = 46
    elif pvModuleTechnology=="ASI":# Amorphous Silicon
        TempNOCT_def = 44
    elif pvModuleTechnology=="CDTE":#CdTe
        TempNOCT_def = 48
    elif  pvModuleTechnology=="CIS":#CIS
        TempNOCT_def = 47
    else:
        TempNOCT_def = 44
    return TempNOCT_def
    

def TempCoeffPmax_defaults_by_technology(pvModuleTechnology):
    if pvModuleTechnology=="CSI": # Crystalline Silicon
        TempCoeffPmax_def = -0.438
    elif pvModuleTechnology=="ASI":# Amorphous Silicon
        TempCoeffPmax_def = -0.18
    elif pvModuleTechnology=="CDTE":#CdTe
        TempCoeffPmax_def = -0.297
    elif  pvModuleTechnology=="CIS":#CIS
        TempCoeffPmax_def = -0.36
    else:
        TempCoeffPmax_def = -0.438
    return TempCoeffPmax_def
    

#PV degradation for given dfb
def PVdegrad_for_dfb(dfb,dfb_startup,year_1_degrad,year_2_degrad):
    if (dfb-dfb_startup)<365:
        cumul_degrad=max(0, (dfb-dfb_startup)/365.*year_1_degrad)
    else:
        cumul_degrad=max(0, year_1_degrad+(dfb-dfb_startup-356)/365.*(year_2_degrad))
    return cumul_degrad


#PV degradation for range of dfbs
def PVdegrad_for_dfbs(dfb_begin, dfb_end,dfb_systemstartup,year_1_degrad,year_2_degrad):
    numdfbs = dfb_end - dfb_begin + 1
    PVdegrad = numpy.empty((numdfbs))

    for dfb_idx in range(0,numdfbs):
        dfb = dfb_begin+dfb_idx
        if (dfb-dfb_systemstartup)<365:
            PVdegrad[dfb_idx]=max(0, (dfb-dfb_systemstartup)/365.*year_1_degrad)
        else:
            PVdegrad[dfb_idx]=max(0, year_1_degrad+(dfb-dfb_systemstartup-356)/365.*(year_2_degrad))
    return PVdegrad

#PV degradation for range of dfbs
def MonthlySnowLosses_2_DailySnowLosses(dfb_begin, dfb_end,dcLossesPollutionSnowMonth):
    numdfbs = dfb_end - dfb_begin + 1
    dcLossesSnowDaily = numpy.empty((numdfbs))

    for dfb_idx in range(0,numdfbs):
        dcLossesSnowDaily[dfb_idx]=dcLossesPollutionSnowMonth[daytimeconv.dfb2date(dfb_begin+dfb_idx).month-1]
    return dcLossesSnowDaily


def Inverter(DCIn,pvInverterEff,pvInverterLimitLow,pvInverterLimitHigh,pvInverterCount):
    verbose = False
    
    sto=100.
    zero=0.
    try:
        efftype=pvInverterEff.shape[1]
        effln=pvInverterEff.shape[0]
    except:
        efftype=1

    if efftype==1:
        pvInverterEuroEff=max(min(pvInverterEff,sto),zero)
        ACOut=(float(pvInverterEuroEff)/sto)*DCIn
        if verbose: print "euro", float(pvInverterEuroEff)/sto
    elif efftype==2:
        pvInverterEff[:,0]=pvInverterCount*pvInverterEff[:,0]
        if verbose: print "points"
        util_ranges=numpy.empty(effln+2)
        efficiences=numpy.empty(effln+2)
        
        util_ranges[0]=zero
        efficiences[0]=zero
        
        if pvInverterLimitHigh is not None:
            util_ranges[-1]=pvInverterLimitHigh
        else:
            util_ranges[-1]=pvInverterEff[:,0].max()
        efficiences[-1]=pvInverterEff[-1][1]
        
        util_ranges[1:effln+1]=pvInverterEff[:,0]
        efficiences[1:effln+1]=pvInverterEff[:,1]

        ACOut=(numpy.interp(DCIn,util_ranges,efficiences)/sto)*DCIn
    else:
        return DCIn
    
    if pvInverterLimitHigh is not None:
        ACOut[ACOut>pvInverterLimitHigh]=pvInverterLimitHigh
    if pvInverterLimitLow is not None:
        ACOut[DCIn<pvInverterLimitLow]=zero
    return ACOut
   

def first_solar_spectral_correction(pw, airmass_absolute, module_type='CDTE', coefficients=None):
    if numpy.min(pw) < 0.1:
        pw = numpy.maximum(pw, 0.1)
        print 'Exceptionally low Pwat values replaced with 0.1 cm to prevent' + ' model divergence'
    if numpy.max(pw) > 8:
        print 'Exceptionally high Pwat values. Check input data:' + ' model may diverge in this range'
    if numpy.max(airmass_absolute) > 10:
        airmass_absolute = numpy.minimum(airmass_absolute, 10)
    if numpy.min(airmass_absolute) < 0.58:
        print 'Exceptionally low air mass: ' + 'model not intended for extra-terrestrial use'
    _coefficients = {}
    _coefficients['cdte'] = (0.86273, -0.038948, -0.012506, 0.098871, 0.084658, -0.0042948)
    _coefficients['monocsi'] = (0.85914, -0.02088, -0.0058853, 0.12029, 0.026814, -0.001781)
    _coefficients['xsi'] = _coefficients['monocsi']
    _coefficients['polycsi'] = (0.8409, -0.027539, -0.0079224, 0.1357, 0.038024, -0.0021218) # POLY-CSI
    _coefficients['csi'] = (0.8409, -0.027539, -0.0079224, 0.1357, 0.038024, -0.0021218)  # POLY-CSI
    _coefficients['multicsi'] = _coefficients['csi']
    _coefficients['cis'] = (0.85252, -0.022314, -0.0047216, 0.13666, 0.013342, -0.0008945)
    _coefficients['asi'] = (1.12094, -0.04762, -0.0083627, -0.10443, 0.098382, -0.0033818)
    if module_type is not None and coefficients is None:
        coefficients = _coefficients[module_type.lower()]
    elif module_type is None and coefficients is not None:
        pass
    else:
        raise TypeError('ambiguous input, must supply only 1 of ' + 'module_type and coefficients')
    coeff = coefficients
    ama = airmass_absolute

    modifier = coeff[0] + coeff[1] * ama + coeff[2] * pw + coeff[3] * numpy.sqrt(ama) + coeff[4] * numpy.sqrt(pw) + coeff[5] * ama / numpy.sqrt(pw)
    return modifier



    
def PVModule(P_STC,Gin,Tin,pvModuleTechnology,TempCoeffPmax, AM = 1.5, wv_cm = None):
    if wv_cm is None:
        AM = 1.5
        wv_cm = 32.588685 # NO CORRECTION FOR CDTE (spectral_correction = 1)

    Tstc=25.
    G=Gin/1000.
    TempCoeffPmax_def = TempCoeffPmax_defaults_by_technology(pvModuleTechnology)    
    T=(Tin-Tstc)*(TempCoeffPmax/(TempCoeffPmax_def))
    doy=180
    if pvModuleTechnology=="ASI":# FAKE TO BE DONE
    #Thin film oscilations --0Wp
#        P_STC=P_STC*(.97+.06*numpy.sin(5.-2.*numpy.pi*doy/365.))  #removed on 20170614 - Artur's comment on not-functional correction for all ASI obsolete technologies
        T=T/2.
        spectral_correction = 1.0    

    if pvModuleTechnology=="CSI" or pvModuleTechnology=="ASI":# Crystalline Silicon
        k=[-0.017162,-0.04028,-0.00468,1.48e-4,1.69e-4,5.0e-6]
        spectral_correction = 1.0
    elif pvModuleTechnology=="CDTE":#CdTe
#       # OBSOLETE HULD COEFFS FROM PVPERFORMANCE [-0.103251,-0.040446,-0.001667,-0.002075,-0.001445,-0.000023]
        # Newest coefs from FS, May 2017 model '4120_3' from PVSyst full matrix weighted with IEC 61853-1 matrix
        k=[-1.83901462e-02, -2.09929902e-02,-2.56930126e-03,1.31518994e-04, -2.26722667e-05, -7.72601269e-06]
        spectral_correction = first_solar_spectral_correction(wv_cm, AM, module_type = pvModuleTechnology)
        print 'mean spectral correction for CDTE',spectral_correction.mean()
    elif  pvModuleTechnology=="CIS":#CIS
        k=[-0.005521,-0.038492,-0.003701,-0.000899,-0.001248,0.000001]
        spectral_correction = 1.0
    G[G<=0]=1e-10
    lG=numpy.log(G)
    lG2=lG*lG
    PV1=(spectral_correction*G*P_STC*(1.+(k[0]*lG + k[1]*lG2 + k[2]*T + k[3]*T*lG + k[4]*T*lG2 + k[5]*numpy.power(T,2))))
    PV1[PV1<0]=0
    return PV1



def CalcEURO_from_curve(pvInverterEff):
    pvInverterLimitLow, pvInverterLimitHigh = 0, 100
    EURO_in = numpy.array([5, 10, 20, 30, 50, 100])
    EURO_weights = numpy.array([0.03, 0.06, 0.13, 0.1, 0.48, 0.2])
    EURO_powers = Inverter(EURO_in, pvInverterEff, pvInverterLimitLow, pvInverterLimitHigh)
    return (EURO_weights * EURO_powers).sum()

def CalcCEC_from_curve(pvInverterEff):
    pvInverterLimitLow, pvInverterLimitHigh = 0, 100
    CEC_in = numpy.array([10, 20, 30, 50, 75, 100])
    CEC_weights = numpy.array([0.04, 0.05, 0.12, 0.21, 0.53, 0.05])
    CEC_powers = Inverter(CEC_in, pvInverterEff, pvInverterLimitLow, pvInverterLimitHigh)
    return (CEC_weights * CEC_powers).sum()