'''
Created on Apr 13, 2010

@author: artur, tomas
'''


import numpy
import numpy as np

#import math


# Sun position in relation to the inclined surface 
# Version for fixed plane and 2 and 1 axis tracker with vertical rotating axis (Strategy A)  
# scalar prodact version
# Input
# sinGammaN,cosGammaN - sin and cos of the surface slope (inclination) angle
# sinAN,cosAN - sin and cos of the surface azimuth angle
# sina0,cosa0 - sin and cos of the sun azimuth angle
# sinh0,cosh0 - sin and cos of the sun height angle 
# OUtput
# sinDeltaexp - sin (or cos) of the angle between the surface (or normal to the surface) and the direction of the sun 
def Rel_sunpos_V1(sinGammaN,cosGammaN,sinAN,cosAN,sina0,cosa0,sinh0,cosh0):
    
    x=cosAN*sinGammaN
    y=sinAN*sinGammaN
    z=cosGammaN
    
    x0=cosa0*cosh0
    y0=sina0*cosh0
    z0=sinh0
    
    cosINC=x*x0+y*y0+z*z0
    cosINC[cosINC<0]=0
    return cosINC

    
# Sun position in relation to the inclined surface 
# Version 1 axies trackers with horizontal and inclined rotating axis (Strategy B, C, D)
# scalar prodact version
# Input
# sinphi, cosphi - sin and cos of the rotation angle 
# sinpsi, cospsi - sin and cos of the rotating axis inclination angle 
# sinAN,cosAN - sin and cos of the surface azimuth angle
# sina0,cosa0 - sin and cos of the sun azimuth angle
# sinh0,cosh0 - sin and cos of the sun height angle 
# OUtput
# sinDeltaexp - sin (or cos) of the angle between the surface (or normal to the surface) and the direction of the sun 
def Rel_sunpos_V2(sinphi,cosphi,sinpsi,cospsi,sina0,cosa0,sinh0,cosh0):
    
    x=sinpsi*cosphi
    y=sinphi
    z=cospsi*cosphi
    
    x0=cosa0*cosh0
    y0=sina0*cosh0
    z0=sinh0
    
    cosINC=x*x0+y*y0+z*z0
    cosINC[cosINC<0]=0
    return cosINC
    
# Optimum rotation angle Phi for 1x tracker with rotating inclined axis (Strategy B,D)
# sina0,cosa0 - sin and cos of the sun azimuth angle
# sinh0,cosh0 - sin and cos of the sun height angle
# sinpsi,cospsi - sin and cos of the axis inclination angle
def optPhiB(sina0,cosa0,sinh0,cosh0,sinpsi,cospsi):
    phiopt=numpy.arctan2((sina0*cosh0),(cosa0*cosh0*sinpsi+sinh0*cospsi))
    sinphiopt=numpy.sin(phiopt)
    cosphiopt=numpy.cos(phiopt)
    return sinphiopt,cosphiopt  



# Optimum rotation angle Phi for 1x tracker with horizontal east-west axies (Strategy C)
# sina0,cosa0 - sin and cos of the sun azimuth angle
# sinh0,cosh0 - sin and cos of the sun height angle 
def optPhiC(sina0,cosa0,sinh0,cosh0):
    phiopt=numpy.arctan2(cosa0,(sinh0/cosh0))
    cosphiopt=numpy.cos(phiopt)
    sinphiopt=numpy.sin(phiopt)
    return sinphiopt,cosphiopt



#def NS_TrackerBacktrack_core_bak(a0, h0refr, sinh0, sina0, RotRange, RelativeSpacing, phioptB ):
#    modules_n=10
#    module_x=10
#    row_length=module_x*modules_n
#    module_y=1  
#    
#    start_lst=[]
#    end_lst=[]
#    start=1
#    AN_morning=np.radians(-90)
#    Tracker0B_Position=phioptB.copy()
#    
#    # Search for start and and of full days
#    for idx in range(1,len(sinh0)-1):
#        if (sinh0[idx]<sinh0[idx+1]) and (sinh0[idx]<sinh0[idx-1]) and sina0[idx]<0 and start==1:
#            start=0
#            if idx>1:
#                start_lst.append(0)
#                end_lst.append(idx-1)
#            start_lst.append(idx)
#            continue
#        elif (sinh0[idx]<sinh0[idx+1]) and (sinh0[idx]<sinh0[idx-1]) and sina0[idx+1]<0 and start==1:
#            start=0
#            if idx>1:
#                start_lst.append(0)
#                end_lst.append(idx)
#            start_lst.append(idx+1)
#            continue
#        if (sinh0[idx]<sinh0[idx+1]) and(sinh0[idx]<sinh0[idx-1]) and sina0[idx]>0 and start==0:
#            end_lst.append(idx)
#            start_lst.append(idx+1)
#        elif (sinh0[idx]<sinh0[idx+1]) and (sinh0[idx]<sinh0[idx-1]) and sina0[idx]<0 and start==0:
#            end_lst.append(idx-1)
#            start_lst.append(idx)
#    if len(start_lst)<2:
#        start_lst.append(0)
#    end_lst.append(len(sinh0)-1)
#
#    
#    for idx_1 in range(len(start_lst)):
#        vlen=end_lst[idx_1]-start_lst[idx_1]
#        phiopt_0 = RotRange
#        # applaying back track for before noon
#        for idx in range(start_lst[idx_1],start_lst[idx_1]+vlen/2+1):
#            for phiopt in np.arange(RotRange,0,-numpy.radians(1)):
#                shadow=0
#                valid_h0refr=h0refr[idx]
#                valid_a0=a0[idx]-AN_morning
#                spacings_V=RelativeSpacing-module_y*np.cos(phiopt)
#                heights_V=module_y*np.sin(phiopt)
#                tan_alfa=heights_V/spacings_V
#                RightBoundary0=np.arctan(row_length/spacings_V)
#                LeftBoundaryM=-np.arctan(row_length/spacings_V)
#                h0_row=np.arctan(tan_alfa*np.cos(valid_a0))
#                if valid_a0>RightBoundary0:
#                    h0_row=0
#                if valid_a0<LeftBoundaryM:
#                    h0_row=0
#                if valid_h0refr<h0_row:
#                    shadow=1
#                if shadow:
#                    Tracker0B_Position[idx]=-phiopt
#                    phiopt_0 = phiopt
#                else:
#                    break  
##            print idx, numpy.degrees(Tracker0B_Position[idx])
##        print
#        # applaying back track for after noon
#        phiopt_0 = RotRange
#        for idx in range(end_lst[idx_1],end_lst[idx_1]-vlen/2-1,-1):
#            for phiopt in np.arange(RotRange,0,-numpy.radians(1)):
#                shadow=0
#                valid_h0refr=h0refr[idx]
#                valid_a0=a0[idx]+AN_morning
#                spacings_V=RelativeSpacing-module_y*np.cos(phiopt)
#                heights_V=module_y*np.sin(phiopt)
#                tan_alfa=heights_V/spacings_V
#                RightBoundary0=np.arctan(row_length/spacings_V)
#                LeftBoundaryM=-np.arctan(row_length/spacings_V)
#                h0_row=np.arctan(tan_alfa*np.cos(valid_a0))
#                if valid_a0>RightBoundary0:
#                    h0_row=0
#                if valid_a0<LeftBoundaryM:
#                    h0_row=0
#                if valid_h0refr<h0_row:
#                    shadow=1
#                if shadow:
#                    Tracker0B_Position[idx]=phiopt
#                    phiopt_0 = phiopt
#                else:
#                    break
#    
#    return Tracker0B_Position
#    
#    



def find_start_end_lst(sinh0,sina0):
    
    start_lst=[]
    end_lst=[]
    noon_lst=[]
    # First case when nights (h0<0) are excluded from time-series
    if len(sinh0[sinh0<0])==0:
        start=1
        for idx in range(1,len(sinh0)-1):
            if (sinh0[idx]<sinh0[idx+1]) and (sinh0[idx]<sinh0[idx-1]) and sina0[idx]<0 and start==1:
                start=0
                if idx>1:
                    start_lst.append(0)
                    end_lst.append(idx-1)
                start_lst.append(idx)
                continue
            elif (sinh0[idx]<sinh0[idx+1]) and (sinh0[idx]<sinh0[idx-1]) and sina0[idx+1]<0 and start==1:
                start=0
                if idx>1:
                    start_lst.append(0)
                    end_lst.append(idx)
                start_lst.append(idx+1)
                continue
            if (sinh0[idx]<sinh0[idx+1]) and(sinh0[idx]<sinh0[idx-1]) and sina0[idx]>0 and start==0:
                end_lst.append(idx)
                start_lst.append(idx+1)
            elif (sinh0[idx]<sinh0[idx+1]) and (sinh0[idx]<sinh0[idx-1]) and sina0[idx]<0 and start==0:
                end_lst.append(idx-1)
                start_lst.append(idx)
        if len(start_lst)<2:
            start_lst.append(0)
        end_lst.append(len(sinh0)-1)
        noon_lst=list((np.array(start_lst)+np.array(end_lst))/2)
    else:
        # Search for start and and of full days with nights
        for idx in range(1,len(sinh0)-1):
            if (sinh0[idx-1]<=0) and (sinh0[idx]>=0):
                start_lst.append(idx-1)
            if (sinh0[idx-1]>=0) and (sinh0[idx]<=0):
                end_lst.append(idx)
    
            if (sinh0[idx]>=sinh0[idx-1]) and (sinh0[idx]>=sinh0[idx+1]) and (sinh0[idx]>0):
                noon_lst.append(idx)
    
        if sinh0[0]>0:
            if noon_lst[0]< end_lst[0]:
                start_lst.insert(0,0)
            else:
                start_lst.insert(0,0)
                noon_lst.insert(0,0)
    
        if sinh0[-1]>0:
            if noon_lst[-1]> start_lst[-1]:
                end_lst.append(len(sinh0)-1)
            else:
                end_lst.append(len(sinh0)-1)
                noon_lst.append(len(sinh0)-1)
    
    return start_lst,end_lst,noon_lst
            
    

def NS_TrackerBacktrack_core(a0, h0refr, sinh0, sina0, RotRange, RelativeSpacing, phioptB ):
    modules_n=10
    module_x=10
    row_length=module_x*modules_n
    module_y=1  

    AN_morning=np.radians(-90)
    Tracker0B_Position=phioptB.copy()
    
    start_lst,end_lst,noon_lst=find_start_end_lst(sinh0,sina0)
#    start_lst=[]
#    end_lst=[]
#    noon_lst=[]
##    start=1
#
#    # Search for start and and of full days
#
#    for idx in range(1,len(sinh0)-1):
#        if (sinh0[idx-1]<=0) and (sinh0[idx]>=0):
#            start_lst.append(idx-1)
#        if (sinh0[idx-1]>=0) and (sinh0[idx]<=0):
#            end_lst.append(idx)
#
#        if (sinh0[idx]>=sinh0[idx-1]) and (sinh0[idx]>=sinh0[idx+1]) and (sinh0[idx]>0):
#            noon_lst.append(idx)
#
#    if sinh0[0]>0:
#        if noon_lst[0]< end_lst[0]:
#            start_lst.insert(0,0)
#        else:
#            start_lst.insert(0,0)
#            noon_lst.insert(0,0)
#
#    if sinh0[-1]>0:
#        if noon_lst[-1]> start_lst[-1]:
#            end_lst.append(len(sinh0)-1)
#        else:
#            end_lst.append(len(sinh0)-1)
#            noon_lst.append(len(sinh0)-1)

#    print len(start_lst)
#    print len(noon_lst)
#    print len(end_lst)
#
#    if len(start_lst) != len(noon_lst):
#        print (start_lst)
#        print (noon_lst)
#        print (end_lst)
#         


#    import pylab
#    pylab.plot(sinh0)
#    pylab.grid()
#    pylab.show()


    #precalculated arrays
    phiopt_V=[]
    indexes=[]
    RightBoundary0_V=[]
    tan_alfa_V=[]
    counter=0
    for phiopt in np.arange(RotRange,0,-numpy.radians(1)):
        indexes.append(counter)
        spacings=(RelativeSpacing-module_y*np.cos(phiopt))
        heights=(module_y*np.sin(phiopt))
        phiopt_V.append(phiopt)
        RightBoundary0_V.append(np.arctan(row_length/spacings))
        tan_alfa_V.append(heights/spacings)
        counter+=1


    
    for idx_1 in range(len(start_lst)):
        # applying back track for before noon
        for idx in range(start_lst[idx_1],noon_lst[idx_1]):
            valid_h0refr=h0refr[idx]
            if valid_h0refr < -0.1:
                Tracker0B_Position[idx]=0
                continue
            valid_a0=a0[idx]-AN_morning
            valid_a0_cos = np.cos(valid_a0)
            for phi_idx in indexes:    
                shadow=0
                tan_alfa=tan_alfa_V[phi_idx]
                RightBoundary0=RightBoundary0_V[phi_idx]
                LeftBoundaryM=-(RightBoundary0)
                h0_row=np.arctan(tan_alfa*valid_a0_cos)
                if valid_a0>RightBoundary0:
                    h0_row=0
                if valid_a0<LeftBoundaryM:
                    h0_row=0
                if valid_h0refr<h0_row:
                    shadow=1
                if shadow:
                    Tracker0B_Position[idx]=-phiopt_V[phi_idx]
                else:
                    break  


        # applying back track for after noon
        for idx in range(end_lst[idx_1],noon_lst[idx_1],-1):
            valid_h0refr=h0refr[idx]
            if valid_h0refr < -0.1:
                Tracker0B_Position[idx]=0
                continue
            valid_a0=a0[idx]+AN_morning
            valid_a0_cos=np.cos(valid_a0)
            for phi_idx in indexes:    
                shadow=0
                tan_alfa=tan_alfa_V[phi_idx]
                RightBoundary0=RightBoundary0_V[phi_idx]
                LeftBoundaryM=-(RightBoundary0)
                h0_row=np.arctan(tan_alfa*valid_a0_cos)
                if valid_a0>RightBoundary0:
                    h0_row=0
                if valid_a0<LeftBoundaryM:
                    h0_row=0
                if valid_h0refr<h0_row:
                    shadow=1
                if shadow:
                    Tracker0B_Position[idx]=phiopt_V[phi_idx]
                else:
                    break

    wh = sinh0<0
    Tracker0B_Position[wh]=0
    return Tracker0B_Position
    
    
    
def NS_TrackerBacktrack(sinGammaN,cosGammaN,sinh0,cosh0,h0refr,sina0,cosa0,RotRange,RelativeSpacing):
    a0=np.arcsin(sina0)#calculate advanced (selfshading based) backtracking position

    # Horizontal NS (1xD) and inclined NS (1xB) Tracker rotation limit  
    cosRotRange=np.cos(RotRange)
    sinRotRange=np.sin(RotRange)
    sinphioptB,cosphioptB=optPhiB(sina0,cosa0,sinh0,cosh0,sinGammaN,cosGammaN)
    sinphioptB[cosphioptB<=cosRotRange]=np.sign(sinphioptB[cosphioptB<=cosRotRange])*sinRotRange        
    cosphioptB[cosphioptB<=cosRotRange]=cosRotRange
    phioptB=(numpy.arcsin(sinphioptB))
    np.savetxt("opt_oryg.out",np.round(np.degrees(phioptB),1),fmt='%1.2f',delimiter=';')
    np.savetxt("h0refr.out",np.round(np.degrees((h0refr)),1),fmt='%1.2f',delimiter=';')
    input_dims = sina0.ndim
    if input_dims == 4:
        Tracker0B_Position = numpy.empty_like(sina0)
        months = sina0.shape[0]
        times = sina0.shape[1]
        cols=sina0.shape[2]
        rows=sina0.shape[3]
#        import datetime
        for c in range(0,cols):
#            print c,'/',cols,  datetime. datetime.now()
            for r in range(0,rows):
                a0_1D = a0[:,:,c,r].flatten()
                h0refr_1D = h0refr[:,:,c,r].flatten() 
                sinh0_1D = sinh0[:,:,c,r].flatten()
                sina0_1D = sina0[:,:,c,r].flatten()
                phioptB_1D = phioptB[:,:,c,r].flatten()
                res = NS_TrackerBacktrack_core(a0_1D, h0refr_1D, sinh0_1D, sina0_1D, RotRange, RelativeSpacing, phioptB_1D )
                res = res.reshape([months, times])
                Tracker0B_Position [:,:,c,r] = res
    elif input_dims == 1:
        Tracker0B_Position = NS_TrackerBacktrack_core(a0, h0refr, sinh0, sina0, RotRange, RelativeSpacing, phioptB )

    np.savetxt("opt_modif.out",np.round(np.degrees(Tracker0B_Position),1),fmt='%1.2f',delimiter=';')
    return np.sin(Tracker0B_Position),np.cos(Tracker0B_Position)

def mounting_incidencecos_gamma_core(mounting, sinh0, cosh0, sina0, cosa0, GN_r=None, sinGNV=None, cosGNV=None, sinANV=None, cosANV=None, lat=None, h0=None, RotationRange=numpy.pi/2., Backtrack=False, BacktrackRelativeSpacing=2.25):
    '''
    calculate cosine of incidence angle and module inclination angle for different mounting methods - core calculation function
    inputs precalculated sin's and cos's
    inputs:
        mounting - type of mounting geometry:
            1 - fixed
            5 - t1xV 1-axis tracker, vertical axis, inclined module 
            6 - t1xI 1-axis tracker NS, inclined axis
            7 - t1xH 1-axis tracker NS, horizontal axis
            8 - t1xEW 1-axis tracker, horizontal EW axis
            9 - t2x 2-axis tracker
        sinh0, cosh0 - sin, cos of sun height (radians, matrix)
        sina0, cosa0 - sin, cos of sun azimuth (radians, matrix)
        GN_r - module (axis) inclination(0- horizontal) (radians) 
        sinGN, cosGN - sin, cos of module (axis) inclination(0- horizontal) (radians) 
        sinANV, cosANV - like sinAN, cosAN but in vector form
        sinAN, cosAN - sin, cos of module azimuth (radians)
        lat - latitude to know if we are on Northern or Southern hemisphere
        h0 - sun height in radians
        RotRange - single value in radians, the limiting angle of horizontal or inclined tracker axis rotation
        Backtrack - bool True/False condition if back tracking algorithm should be applied 
        BacktrackRelativeSpacing - the relative spacing calculated as (distance between axles)/(tracker wing width) ; 
    outputs:
        GammaNV - module inclination (considering tracking) (matrix, radians)
        sinDeltaexp = cos_incidence_angle - cosine of incidence angle (angle between module normal and sun) (matrix)
    '''
    
    ZEROS = numpy.zeros_like(sinh0)
    ONES = numpy.ones_like(sinh0)

    if mounting==1:
        #Fixed 1 position system
        GammaNV=GN_r*ONES
        sinGammaNV=sinGNV*ONES
        cosGammaNV=cosGNV*ONES
        sinDeltaexp=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,sina0,cosa0,sinh0,cosh0)

    elif mounting==5:
        #"PV1xA - Tracker vertical, inclined:"
        GammaNV=GN_r*ONES
        sinGammaNV=sinGNV*ONES
        cosGammaNV=cosGNV*ONES
        sinDeltaexp=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sina0,cosa0,sina0,cosa0,sinh0,cosh0)

    elif mounting==6 or mounting==7:
        #"PV1xB - Tracker inclined NS and special case horizontal NS tracker mounting=7"
        cosRotRange=numpy.cos(RotationRange)
        sinRotRange=numpy.sin(RotationRange)
        if mounting==7:
            sinGNV=0
            cosGNV=1
        sinGammaNV=sinGNV*ONES
        cosGammaNV=cosGNV*ONES
        lat_sign=numpy.sign(lat)
        if Backtrack:
            sinphioptB,cosphioptB=NS_TrackerBacktrack(sinGammaNV*lat_sign,cosGammaNV,sinh0,cosh0,h0,sina0,cosa0,RotationRange,BacktrackRelativeSpacing)
        else:
            sinphioptB,cosphioptB=optPhiB(sina0.copy(),cosa0.copy(),sinh0.copy(),cosh0.copy(),sinGammaNV*lat_sign,cosGammaNV.copy())      
            sinphioptB[cosphioptB<cosRotRange]=numpy.sign(sinphioptB[cosphioptB<cosRotRange])*sinRotRange
            cosphioptB[cosphioptB<cosRotRange]=cosRotRange

        sinDeltaexp=Rel_sunpos_V2(sinphioptB,cosphioptB,sinGammaNV*lat_sign,cosGammaNV,sina0,cosa0,sinh0,cosh0)
        cosGammaNV=cosphioptB*cosGammaNV
        sinGammaNV=numpy.power(1-numpy.power(cosGammaNV,2),0.5)
        GammaNV=numpy.arccos(cosGammaNV)

        exit()














    elif mounting==8:
        #"PV1xC - Tracker horizontal EW:"
        sinphioptC,cosphioptC=optPhiC(sina0,cosa0,sinh0,cosh0)
        cosGammaNV=cosphioptC
        sinGammaNV=sinphioptC
        GammaNV=numpy.arccos(cosphioptC)
        sinDeltaexp=Rel_sunpos_V2(ZEROS,ONES,sinphioptC,cosphioptC,sina0,cosa0,sinh0,cosh0)

    elif mounting==9:
        # PV2x - Tracker two axis
        GammaNV = numpy.pi/2.- h0
        sinDeltaexp = ONES
    else:
        return None

    return sinDeltaexp, GammaNV


def mounting_incidencecos_gamma(mounting, a0, h0, GN=None, AN=None, lat=None, RotationRange=numpy.pi/2., Backtrack=False, BacktrackRelativeSpacing=2.5):
    '''
    calculate cosine of incidence angle and module inclination angle for different mounting methods
    inputs:
        a0 - sun azimuth (radians, matrix)
        h0 - sun height (radians, matrix)
        GN - module (axis) inclination(0- horizontal) (radians) 
        AN - module azimuth (radians)
        mounting - type of mounting geometry:
            1 - fixed
            5 - t1xV 1-axis tracker, vertical axis, inclined module 
            6 - t1xI 1-axis tracker, inclined axis
            7 - t1xNS 1-axis tracker, horizontal NS axis
            8 - t1xEW 1-axis tracker, horizontal EW axis
            9 - t2x 2-axis tracker
        RotationRange - single value in radians, the limiting angle of horizontal or inclined tracker axis rotation
        Backtrack - bool True/False condition if back tracking algorithm should be applied 
        BacktrackRelativeSpacing - the relative spacing calculated as (distance between axles)/(tracker wing width) 
     outputs:
        GammaNV - module inclination (considering tracking) (matrix, radians)
        sinDeltaexp = cos_incidence_angle - cosine of incidence angle (angle between module normal and sun) (matrix)
    '''

    sina0 = numpy.sin(a0)
    cosa0 = numpy.cos(a0)
    sinh0 = numpy.sin(h0)
    cosh0 = numpy.cos(h0)

    sinAN, cosAN, sinGN, cosGN = None, None, None, None
    if mounting == 1:
        sinAN = numpy.sin(AN)
        cosAN = numpy.cos(AN)
    if (mounting == 1) or (mounting == 5) or (mounting == 6):
        sinGN = numpy.sin(GN)
        cosGN = numpy.cos(GN)
        
    sinDeltaexp, GammaNV = mounting_incidencecos_gamma_core(mounting, sinh0, cosh0, sina0, cosa0, GN, sinGN, cosGN, sinAN, cosAN, lat, h0, RotationRange, Backtrack, BacktrackRelativeSpacing)
    
    return(sinDeltaexp, GammaNV)


def mounting_geom_angles_core(mounting, sinh0, cosh0, sina0, cosa0, GN_r=None, sinGN=None, cosGN=None, sinANV=None, cosANV=None, lat=None, h0=None, \
                                     RotationRange=numpy.pi,TiltRange=numpy.pi, Backtrack=False, RelativeSpacingRows=2,RelativeSpacingColumns=2):
    '''
    calculate cosine of incidence angle and module inclination angle for different mounting methods - core calculation function
    inputs precalculated sin's and cos's
    inputs:
        mounting - type of mounting geometry:
            1 - fixed
            5 - t1xV 1-axis tracker, vertical axis, inclined module 
            6 - t1xI 1-axis tracker NS, inclined axis
            7 - t1xH 1-axis tracker NS, horizontal axis
            8 - t1xEW 1-axis tracker, horizontal EW axis
            9 - t2x 2-axis tracker
        sinh0, cosh0 - sin, cos of sun height (radians, matrix)
        sina0, cosa0 - sin, cos of sun azimuth (radians, matrix)
        GN_r - module (axis) inclination(0- horizontal) (radians) 
        sinGN, cosGN - sin, cos of module (axis) inclination(0- horizontal) (radians) 
        sinANV, cosANV - like sinAN, cosAN but in vector form
        sinAN, cosAN - sin, cos of module azimuth (radians)
        lat - latitude to know if we are on Northern or Southern hemisphere
        h0 - sun height in radians
        RotRange - single value in radians, the limiting angle of horizontal or inclined tracker axis rotation
        Backtrack - bool True/False condition if back tracking algorithm should be applied 
        relativeSpacingRows - the relative spacing calculated as (distance between axles)/(tracker wing width)
        relativeSpacingColumns - the relative spacing calculated as (distance between axles)/(tracker wing width)
    outputs:
        GammaNV - module inclination (considering tracking) (matrix, radians)
        rot - NEEDS TO BE DESCRIBED
        incidenceAngle - angle between module normal and sun (radians, matrix)
        sinDeltaexp = cos_incidence_angle - cosine of incidence angle (matrix)
        sinGammaNV, cosGammaNV
        sinrot, cosrot
    '''
    sinrot=None
    cosrot=None
    rot=None
    ZEROS = numpy.zeros_like(sinh0)
    ONES = numpy.ones_like(sinh0)

    if mounting==1:
        #Fixed 1 position system
        GammaNV=GN_r*ONES
        sinGammaNV=sinGN*ONES
        cosGammaNV=cosGN*ONES
        sinDeltaexp=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,sina0,cosa0,sinh0,cosh0)

    elif mounting==5:
        #"PV1xA - Tracker vertical, inclined:"
        GammaNV=GN_r*ONES
        sinGammaNV=sinGN*ONES
        cosGammaNV=cosGN*ONES
        sinDeltaexp=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sina0,cosa0,sina0,cosa0,sinh0,cosh0)

    elif mounting==6 or mounting==7:
        #"PV1xB - Tracker inclined NS and special case horizontal NS tracker mounting=7"
        cosRotRange=numpy.cos(RotationRange)
        sinRotRange=numpy.sin(RotationRange)
        if mounting==7:
            sinGN=0
            cosGN=1

        sinGammaNV=sinGN*ONES
        cosGammaNV=cosGN*ONES
        lat_sign=numpy.sign(lat)
        if Backtrack:
            sinphioptB,cosphioptB=NS_TrackerBacktrack(sinGammaNV*lat_sign,cosGammaNV,sinh0,cosh0,h0,sina0,cosa0,RotationRange,RelativeSpacingColumns)
        else:
            sinphioptB,cosphioptB=optPhiB(sina0.copy(),cosa0.copy(),sinh0.copy(),cosh0.copy(),sinGammaNV*lat_sign,cosGammaNV.copy())      
            sinphioptB[cosphioptB<cosRotRange]=numpy.sign(sinphioptB[cosphioptB<cosRotRange])*sinRotRange
            cosphioptB[cosphioptB<cosRotRange]=cosRotRange

        sinDeltaexp=Rel_sunpos_V2(sinphioptB,cosphioptB,sinGammaNV*lat_sign,cosGammaNV,sina0,cosa0,sinh0,cosh0)
        cosGammaNV=cosphioptB*cosGammaNV
        sinGammaNV=numpy.power(1-numpy.power(cosGammaNV,2),0.5)
        GammaNV=numpy.arcsin(sinGammaNV)
        sinrot=sinphioptB
        cosrot=cosphioptB

    elif mounting==8:
        #"PV1xC - Tracker horizontal EW:"
        sinphioptC,cosphioptC=optPhiC(sina0,cosa0,sinh0,cosh0)
        cosGammaNV=cosphioptC
        sinGammaNV=sinphioptC
        GammaNV=numpy.arccos(cosphioptC)
        sinDeltaexp=Rel_sunpos_V2(ZEROS,ONES,sinphioptC,cosphioptC,sina0,cosa0,sinh0,cosh0)
        sinrot=sinphioptC
        cosrot=cosphioptC

    elif mounting==9:
        
        # PV2x - Tracker two axis
    
        GammaNVmin=numpy.radians(10)
        GammaNVmax=numpy.radians(80)
        GammaNV = numpy.pi/2.- h0
        a=np.degrees(np.arcsin(sina0))
        limit=180
        a[a>limit]=limit
        a[a<=-limit]=-limit
        GammaNV[GammaNV<GammaNVmin]=GammaNVmin
        GammaNV[GammaNV>GammaNVmax]=GammaNVmax
        #GammaNV = NS_TrackerBacktrack_core(np.radians(a), h0, sinh0, sina0, np.radians(limit), RelativeSpacingColumns, GammaNV )
        sinGammaNV=np.sin(GammaNV)
        cosGammaNV=np.cos(GammaNV)
        sinANV=sina0.copy()
        sinANV[a==limit]=np.sin(np.radians(limit))
        cosANV=cosa0.copy()
        cosANV[a==-limit]=np.sin(np.radians(-limit))
        sinDeltaexp=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,sina0,cosa0,sinh0,cosh0)
        
        #GammaNV = numpy.pi/2.- h0
            #sinDeltaexp = ONES
            #sinGammaNV=np.sin(GammaNV)
        #cosGammaNV=np.cos(GammaNV)

    else:
        return None

    if cosrot is not None:
        rot=numpy.arccos(cosrot)
    
    #Note:  sinDeltaexp = cos_incidence_angle
    incidenceAngle = numpy.arccos(sinDeltaexp)
    
    return incidenceAngle, GammaNV, rot, sinDeltaexp, sinGammaNV,cosGammaNV, sinrot, cosrot


def mounting_geom_angles(mounting, a0, h0, GN=None, AN=None, latitude=None, rotationRange=numpy.pi, tiltRange=numpy.pi, backtrack=False, relativeSpacingRows=2.0,relativeSpacingColumns=2.0):
    '''
    calculate cosine of incidence angle and module inclination angle for different mounting methods
    inputs:
        a0 - sun aspect (radians, matrix) 0-south
        h0 - sun height (radians, matrix)
        GN - module (axis) inclination(0- horizontal) (radians) 
        AN - module aspect (radians) 0-south
        latitude - latitude of the site
        mounting - type of mounting geometry:
            1 - fixed
            5 - t1xV 1-axis tracker, vertical axis, inclined module 
            6 - t1xI 1-axis tracker, inclined axis
            7 - t1xNS 1-axis tracker, horizontal NS axis
            8 - t1xEW 1-axis tracker, horizontal EW axis
            9 - t2x 2-axis tracker
        rotationRange - PRECISE EXPLANATION single value in radians, the limiting angle of horizontal or inclined tracker axis rotation
        tiltRange - NEEDS EXPLANATION  
        backtrack - bool True/False condition if back tracking algorithm should be applied 
        relativeSpacingRows - the relative spacing calculated as (distance between axles)/(tracker wing width)
        relativeSpacingColumns - the relative spacing calculated as (distance between axles)/(tracker wing width)
     outputs:
        incidenceAngle - angle between module normal and sun (radians, matrix)
        GammaNV - module inclination (considering tracking) (matrix, radians)
        rot - NEEDS EXPLANATION
        sinDeltaexp = cos_incidence_angle - cosine of incidence angle (angle between module normal and sun) (matrix)
        sinGammaNV, cosGammaNV
        sinrot, cosrot - NEEDS EXPLANATION
    '''

    sina0 = numpy.sin(a0)
    cosa0 = numpy.cos(a0)
    sinh0 = numpy.sin(h0)
    cosh0 = numpy.cos(h0)

    sinAN, cosAN, sinGN, cosGN = None, None, None, None
    if mounting == 1:
        sinAN = numpy.sin(AN)
        cosAN = numpy.cos(AN)
    if (mounting == 1) or (mounting == 5) or (mounting == 6):
        sinGN = numpy.sin(GN)
        cosGN = numpy.cos(GN)
    
#    sinDeltaexp, GammaNV, rot, sinGammaNV, cosGammaNV, sinrot, cosrot = mounting_geom_angles_core(mounting, sinh0, cosh0, sina0, cosa0, GN, sinGN, cosGN, sinAN, cosAN, latitude, h0, \
#                                                                      rotationRange, tiltRange, backtrack, relativeSpacingRows, relativeSpacingColumns)
    incidenceAngle, GammaNV, rot, sinDeltaexp, sinGammaNV,cosGammaNV, sinrot, cosrot  = mounting_geom_angles_core(mounting, sinh0, \
                                             cosh0, sina0, cosa0, GN, sinGN, cosGN, sinAN, cosAN, latitude, h0, \
                                             rotationRange, tiltRange, backtrack, relativeSpacingRows, relativeSpacingColumns)
    return incidenceAngle, GammaNV, rot, sinDeltaexp, sinGammaNV,cosGammaNV, sinrot, cosrot
    

def mounting_shortname(mounting, preffix="t"):
    mounting_name_dict={\
                    1 : "fix",\
                    5 : "1xV",\
                    6 : "1xI",\
                    7 : "1xNS",\
                    8 : "1xEW",\
                    9 : "2x",\
                    }
    if mounting in mounting_name_dict.keys():
        return preffix+mounting_name_dict[mounting]
    return ""


def mounting_longname(mounting, inclination, aspect):
    mounting_name_dict={\
                    1 : "fixed inclination: %d deg. azimuth: %d deg." % (inclination, aspect),\
                    5 : "one-axis tracker, vertical axis, module inclination: %d deg." % (inclination),\
                    6 : "one-axis tracker, axis inclination: %d deg." % (inclination),\
                    7 : "one-axis tracker, horizontal axis in NS direction",\
                    8 : "one-axis tracker, horizontal axis in EW direction",\
                    9 : "two-axis tracker",\
                    }
    if mounting in mounting_name_dict.keys():
        return mounting_name_dict[mounting]
    return ""

def mounting_geometryname_to_code(geometryname):
    geomstr=geometryname.lower().strip()
    if geomstr == 'fixedoneangle': # FixedOneAngle
        geomcode = 1
    elif geomstr == 'oneaxisvertical': # OneAxisVertical
        geomcode = 5
    elif geomstr == 'oneaxisinclined': # OneAxisInclined
        geomcode = 6
    elif geomstr == 'oneaxishorizontalns': # OneAxisHorizontalNS
        geomcode = 7
    elif geomstr == 'oneaxishorizontalew': # OneAxisHorizontalEW
        geomcode = 8
    elif geomstr == 'twoaxisastronomical': # TwoAxisAstronomical
        geomcode = 9
    else:
        geomcode=-1
    
    return geomcode
    

def mounting_geometrycode_to_name(geometrycode):
    if geometrycode == 1: # FixedOneAngle
        geomname = 'FixedOneAngle'
    elif geometrycode == 5: # OneAxisVertical
        geomname = 'OneAxisVertical'
    elif geometrycode == 6: # OneAxisInclined
        geomname = 'OneAxisInclined'
    elif geometrycode == 7: # OneAxisHorizontalNS
        geomname = 'OneAxisHorizontalNS'
    elif geometrycode == 8: # OneAxisHorizontalEW
        geomname = 'OneAxisHorizontalEW'
    elif geometrycode == 9: # TwoAxisAstronomical
        geomname = 'TwoAxisAstronomical'
    else:
        geomname = ''
    
    return geomname
    


    
def angles2rowspac(TiltTerrainSN,GammaN,ShadAng,l=1):
    # Convert angles to row spacing
    # l - sirka panelu (if l=1, output is relative spacing)
    # TiltNS - sklon svahu v smere NS [degrees] (SEVER - Positive angle, JUH - Negative value)
    # GammaN - Naklon panelu [degrees]
    # ShadAng - shading angle [degrees]
    # spac_horiz - Rows spacing (bottom to bottom) at the map (horizontal)
    # spac_surf - Rows spacing (bottom to bottom) in the terrain (surface)
    tanShad=numpy.tan(numpy.radians(ShadAng))
    tanTiltTerrainSN=numpy.tan(numpy.radians(TiltTerrainSN))
    cosTiltTerrainSN=numpy.cos(numpy.radians(TiltTerrainSN))
    cosGammaN=numpy.cos(numpy.radians(GammaN))
    sinGammaN=numpy.sin(numpy.radians(GammaN))
    spac_horiz=l*(cosGammaN*tanShad+sinGammaN)/(tanShad-tanTiltTerrainSN)
    
    spac_surf=spac_horiz/cosTiltTerrainSN
    return spac_horiz,spac_surf

def rowspac2angles(TiltTerrainSN,GammaN,spac_horiz,l):
    tanTiltTerrainSN=numpy.tan(numpy.radians(TiltTerrainSN))
    cosGammaN=numpy.cos(numpy.radians(GammaN))
    sinGammaN=numpy.sin(numpy.radians(GammaN))
    spl=spac_horiz/l
    tanShad=(spl*tanTiltTerrainSN+sinGammaN)/(spl-cosGammaN)
    return numpy.degrees(numpy.arctan(tanShad))

    



    
def rotXmatrix(M,salfa,calfa):
    RotMtx=np.array([[1,0,0],[0,calfa,salfa],[0,-salfa,calfa]])
    #RM=np.zeros_like(M)
    #for i in range(0,M.shape[0]):
    #    for j in range(0,M.shape[1]):
    #        RM[i,j,:]=np.dot(RotMtx,M[i,j,:])
    RM=matvec(RotMtx,M)
    return RM

def rotYmatrix(M,salfa,calfa):
    RotMtx=np.array([[calfa,0,-salfa],[0,1,0],[salfa,0,calfa]])
    #RM=np.zeros_like(M)
    #for i in range(0,M.shape[0]):
    #    for j in range(0,M.shape[1]):
    #        RM[i,j,:]=np.dot(RotMtx,M[i,j,:])
    RM=matvec(RotMtx,M)
    return RM

def rotZmatrix(M,salfa,calfa):
    RotMtx=np.array([[calfa,salfa,0],[-salfa,calfa,0],[0,0,1]])
    #RM=np.zeros_like(M)
    #for i in range(0,M.shape[0]):
    #    for j in range(0,M.shape[1]):
    #        RM[i,j,:]=np.dot(RotMtx,M[i,j,:])
    RM=matvec(RotMtx,M)
    return RM

def matvec(m,v):
    ax=range(len(v.shape))[1:]+[0]
    return np.transpose(np.inner(m,v),axes=ax)


def inc_azim2ew_ns(GammaN,AN):
    sinGammaN=numpy.sin(numpy.radians(GammaN))
    cosGammaN=numpy.cos(numpy.radians(GammaN))
    sinAN=numpy.sin(numpy.radians(-AN))
    cosAN=numpy.cos(numpy.radians(-AN))
    
    V=numpy.array([0,0,1])
    V=rotXmatrix(V,sinGammaN,cosGammaN)
    V=rotZmatrix(V,sinAN,cosAN)
   
    incEW_d=numpy.degrees(numpy.arctan(V[0]/V[2]))
    incNS_d=numpy.degrees(numpy.arctan(V[1]/V[2]))
    return incEW_d,incNS_d
     
    
    
    
    
    
    



#for shading calculation

def cartesian2polar(vectors):
    if vectors.ndim==3:
        h0= np.arcsin((vectors[:,:,2])/np.sqrt(np.square(vectors[:,:,0])+np.square(vectors[:,:,1])+np.square(vectors[:,:,2])))
        a0= np.arctan2(vectors[:,:,0],vectors[:,:,1])
    else:
        h0= np.arcsin((vectors[:,2])/np.sqrt(np.square(vectors[:,0])+np.square(vectors[:,1])+np.square(vectors[:,2])))
        a0= np.arctan2(vectors[:,0],vectors[:,1])
    return a0,h0

def polar2cartesian(r,a0,h0):
    sy = r*np.sin(numpy.pi/2.)*numpy.cos(a0)
    sx = r*np.sin(numpy.pi/2.)*numpy.sin(a0)
    sz = r*np.cos(numpy.pi/2.-h0)
    return sx,sy,sz

def neighbour_corners_rows(M,off_y,tanTiltSN,sinAN=0,cosAN=1):
    corn0=M[0,0,:]
    corn1=M[0,-1,:]
    # off_x - spacing between rows
    # shift_z - height difference between tows due to slope
    shift_z = tanTiltSN*off_y
    #offsetMTX=rotZmatrix(array(([0,-off_y, -shift_z])),sinAN,cosAN)
    offsetMTX=(numpy.array(([0,-off_y, -shift_z])))
    return numpy.array([corn1-offsetMTX,corn0-offsetMTX])


def prepare_reference_module(module_x,module_y,hor_sect,vert_sect):
    linX=numpy.linspace(-module_x/2.,module_x/2.,hor_sect)
    linY=numpy.linspace(-module_y/2.,module_y/2.,vert_sect)
    MTX_Y,MTX_X=numpy.meshgrid(linY,linX)
    M=numpy.zeros((len(linX),len(linY),3),dtype=numpy.float64)
    M[:,:,0]=MTX_X
    M[:,:,1]=MTX_Y
    return M,MTX_Y,MTX_X


def prepare_3D_scene(M,rows_back,rows_front,rows_leftright,off_x,off_y,lat_sign_n):
    #Ms matrx with coordinates of all points within all neighbeuring heliostats
    # DIMS(grid_points_x,grid_points_y,3 dimensions xyz, number of neighbours left-right,number of neighbours front )  
    Ms=numpy.zeros((M.shape[0],M.shape[1],3,rows_front+1+rows_back,rows_leftright*2+1))
    for fb in range(-rows_back,rows_front+1):
        for lr in range(-rows_leftright,rows_leftright+1):
            lr_idx=rows_leftright+lr
            fb_idx=fb+rows_back
            Ms[:,:,0,fb_idx,lr_idx]=M[:,:,0]+lr*off_x
            Ms[:,:,1,fb_idx,lr_idx]=M[:,:,1]+fb*off_y*lat_sign_n
            Ms[:,:,2,fb_idx,lr_idx]=M[:,:,2]
    return Ms


def reduce_scene_to_edges_points(Ms,rows_back,rows_leftright):
    start=True
    for f_idx in range(0,Ms.shape[3]):
        for lr_idx in range(0,Ms.shape[4]):
            if  f_idx==rows_back and lr_idx==rows_leftright:continue
            if start:   
                edges=Ms[ :, 0,:,f_idx,lr_idx]
                edges=numpy.vstack((edges,Ms[ :,-1,:,f_idx,lr_idx]))
                edges=numpy.vstack((edges,Ms[-1, :,:,f_idx,lr_idx]))
                edges=numpy.vstack((edges,Ms[ 0, :,:,f_idx,lr_idx]))
            else:
                edges=numpy.vstack((edges,Ms[ :, 0,:,f_idx,lr_idx]))
                edges=numpy.vstack((edges,Ms[ :,-1,:,f_idx,lr_idx]))
                edges=numpy.vstack((edges,Ms[-1, :,:,f_idx,lr_idx]))
                edges=numpy.vstack((edges,Ms[ 0, :,:,f_idx,lr_idx]))
            start=False    
    return edges


def reduce_scene_to_corner_points(Ms,rows_back,rows_leftright):

    cp=numpy.empty((Ms.shape[3]*Ms.shape[4]*4-4,3))
    idx=0
    for f_idx in range(0,Ms.shape[3]):
        for lr_idx in range(0,Ms.shape[4]):
            if  f_idx==rows_back and lr_idx==rows_leftright:continue
            cp[idx,:]=Ms[0,0,:,f_idx,lr_idx]
            idx+=1       
            cp[idx,:]=Ms[-1,0,:,f_idx,lr_idx]
            idx+=1
            cp[idx,:]=Ms[-1,-1,:,f_idx,lr_idx]
            idx+=1
            cp[idx,:]=Ms[0,-1,:,f_idx,lr_idx]
            idx+=1
    return cp


def check_point_position(p1,p2,p3,p4,sun):
    v1= p1-sun
    v2= p2-sun
    angle=vect_angle(v1,v2)
    v1=p2-sun
    v2=p3-sun
    angle+=vect_angle(v1,v2)
    v1=p3-sun
    v2=p4-sun
    angle+=vect_angle(v1,v2)
    v1=p4-sun
    v2=p1-sun
    angle+=vect_angle(v1,v2)
    if angle>1.95*numpy.pi:
        return True
    else:
        return False
    

def vect_angle(v1,v2):
    dot_prod=v1[0]*v2[0]+v1[1]*v2[1]
    norm=numpy.power(sum(v1*v1)*sum(v2*v2),.5)
    try:
        acos=numpy.arccos(dot_prod/norm)
    except:
        return 0
    return acos


def check_point_position_v(p1,p2,p3,p4,sun):
    v1= p1-sun
    v2= p2-sun
    angle=vect_angle_v(v1,v2)
    v3=p3-sun
    angle+=vect_angle_v(v2,v3)
    v4=p4-sun
    angle+=vect_angle_v(v3,v4)
    angle+=vect_angle_v(v4,v1)
    angle[angle<=1.95*numpy.pi]=0
    angle[angle>1.95*numpy.pi]=1
    return angle
    

def vect_angle_v(v1,v2): 
    dot_prod=v1[0,:,:,:]*v2[0,:,:,:]+v1[1,:,:,:]*v2[1,:,:,:]
    norm=numpy.sqrt((numpy.square(v1[0,:,:,:])+numpy.square(v1[1,:,:,:]))*(numpy.square(v2[0,:,:,:])+numpy.square(v2[1,:,:,:])))
    try:
        acos=numpy.arccos(dot_prod/norm)
    except:
        return 0
    return acos


def cartesian2polar_v(vectors):
    h0= np.arcsin((vectors[:,:,:,2])/np.sqrt(np.square(vectors[:,:,:,0])+np.square(vectors[:,:,:,1])+np.square(vectors[:,:,:,2])))
    a0= np.arctan2(vectors[:,:,:,0],vectors[:,:,:,1])
    return a0,h0
                  






def PV_extract_mounting_details(rad_configDict, pvDict):    
    mounting=rad_configDict["mounting"]
    
    try:
        aspect_r=numpy.radians(rad_configDict["azimuth"]-180.)
    except:
        aspect_r=0
    try:
        tilt_r=numpy.radians(rad_configDict["tilt"])
    except:
        tilt_r=0

    terrainSlope = 0
    terrainAzimuth = 180
    if pvDict.has_key("pvFieldTerrainSlope"):
        terrainSlope = pvDict["pvFieldTerrainSlope"]
    if pvDict.has_key("pvFieldTerrainAzimuth"):
        terrainAzimuth = pvDict["pvFieldTerrainAzimuth"]

    do_interrow_shading=True
    if pvDict.has_key("pvFieldSelfShading"): 
        do_interrow_shading=pvDict["pvFieldSelfShading"]
    
    do_backtrack=False
    if pvDict.has_key("pvTrackerBackTrack"):
        do_backtrack=pvDict["pvTrackerBackTrack"]
    
    relativeSpacingColumns=100
    relativeSpacingRows=100
    
    
    #rows and columns spacing
    if mounting==1: # FIX
        if not rad_configDict.has_key("tilt"):
            do_interrow_shading=False
        if pvDict.has_key("pvFieldRowSpacingRelative"):
            relativeSpacingRows=pvDict["pvFieldRowSpacingRelative"]
        elif pvDict.has_key("pvFieldRowSpacingShadingAngle") and do_interrow_shading:
            #adapt for terrain slope and orientation
            TiltTerrainSN_d = inc_azim2ew_ns(terrainSlope, terrainAzimuth)[1]
            relativeSpacingRows = angles2rowspac(TiltTerrainSN_d,numpy.degrees(tilt_r),pvDict["pvFieldRowSpacingShadingAngle"],l=1)[1]
        elif pvDict.has_key("pvFieldRowSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
            relativeSpacingRows=pvDict["pvFieldRowSpacingAbsolute"]/pvDict["pvFieldTableWidth"]
        else:
            do_interrow_shading=False
    elif mounting==6: # t1x NS inclined
        if pvDict.has_key("pvFieldColumnSpacingRelative"):
            relativeSpacingColumns=pvDict["pvFieldColumnSpacingRelative"]
        elif pvDict.has_key("pvFieldColumnSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
            relativeSpacingColumns=pvDict["pvFieldColumnSpacingAbsolute"]/pvDict["pvFieldTableWidth"]
        else:
            do_interrow_shading=False
#        if pvDict.has_key("pvFieldRowSpacingRelative"):
#            relativeSpacingRows=pvDict["pvFieldRowSpacingRelative"]
#        elif pvDict.has_key("pvFieldRowSpacingAbsolute") and pvDict.has_key("pvFieldTableLength"):
#            relativeSpacingRows=pvDict["pvFieldRowSpacingAbsolute"]/pvDict["pvFieldTableLength"]
#        else:
#            do_interrow_shading=False
    elif mounting==7: # t1x NS horizontal
        if pvDict.has_key("pvFieldColumnSpacingRelative"):
            relativeSpacingColumns=pvDict["pvFieldColumnSpacingRelative"]
        elif pvDict.has_key("pvFieldColumnSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
            relativeSpacingColumns=pvDict["pvFieldColumnSpacingAbsolute"]/pvDict["pvFieldTableWidth"]
        else:
            do_interrow_shading=False
    elif mounting==8: # t1x EW horizontal
        do_interrow_shading=False
#        if pvDict.has_key("pvFieldRowSpacingRelative"):
#            relativeSpacingRows=pvDict["pvFieldRowSpacingRelative"]
#        elif pvDict.has_key("pvFieldRowSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
#            relativeSpacingRows=pvDict["pvFieldRowSpacingAbsolute"]/pvDict["pvFieldTableWidth"]
#        else:
#            do_interrow_shading=False
    elif (mounting==9) or (mounting==5): # t2x and t1xV (vertical axis, inclined module)
        do_interrow_shading=False 
#        if pvDict.has_key("pvFieldColumnSpacingRelative"):
#            relativeSpacingColumns=pvDict["pvFieldColumnSpacingRelative"]
#        elif pvDict.has_key("pvFieldColumnSpacingAbsolute") and pvDict.has_key("pvFieldTableLength"):
#            relativeSpacingColumns=pvDict["pvFieldColumnSpacingAbsolute"]/pvDict["pvFieldTableLength"]
#        else:
#            do_interrow_shading=False
#
#        if pvDict.has_key("pvFieldRowSpacingRelative"):
#            relativeSpacingRows=pvDict["pvFieldRowSpacingRelative"]
#        elif pvDict.has_key("pvFieldRowSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
#            relativeSpacingRows=pvDict["pvFieldRowSpacingAbsolute"]/pvDict["pvFieldTableWidth"]
#        else:
#            do_interrow_shading=False

    # rotation limit
    rotation_limit_r=numpy.pi
    rotation_limit2_r=numpy.pi #not implemented yet
    if pvDict.has_key("pvTrackerRotMin"):
        rotation_limit_r=numpy.radians(pvDict['pvTrackerRotMin'])
    if pvDict.has_key("pvTrackerRot2Min"):
        rotation_limit2_r=numpy.radians(pvDict['pvTrackerRot2Min'])

    return mounting, aspect_r, tilt_r, terrainSlope,  terrainAzimuth, do_interrow_shading,  do_backtrack, relativeSpacingColumns, relativeSpacingRows, rotation_limit_r, rotation_limit2_r





