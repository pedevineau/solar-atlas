'''
Created on Apr 13, 2010
@author: artur, tomas
'''
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
# cos_incidence_angle - sin (or cos) of the angle between the surface (or normal to the surface) and the direction of the sun 
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
# Output
# cos_incidence_angle - sin (or cos) of the angle between the surface (or normal to the surface) and the direction of the sun 
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
    return np.arctan2((sina0*cosh0),(cosa0*cosh0*sinpsi+sinh0*cospsi))

# Optimum rotation angle Phi for 1x tracker with horizontal east-west axes (Strategy C)
# sina0,cosa0 - sin and cos of the sun azimuth angle
# sinh0,cosh0 - sin and cos of the sun height angle 
def optPhiC(sina0,cosa0,sinh0,cosh0):
    return np.arctan2(cosa0,(sinh0/cosh0))

def mounting_geom_angles( mounting, sina0, cosa0, a0, sinh0,cosh0, h0, GN, AN, latitude, rotation_limit, rotation_limit2, backtrack, relative_spacing_rows, relative_spacing_columns):

    '''
    calculate cosine of incidence angle and module inclination angle for different mounting methods - core calculation function
    inputs pre-calculated sin's and cos's
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
        GN - module (axis) inclination(0- horizontal) (radians) 
        sinGN, cosGN - sin, cos of module (axis) inclination(0- horizontal) (radians) 
        sinAN, cosANV - like sinAN, cosAN but in vector form
        sinAN, cosAN - sin, cos of module azimuth (radians)
        lat - latitude to know if we are on Northern or Southern hemisphere
        h0 - sun height in radians
        RotRange - single value in radians, the limiting angle of horizontal or inclined tracker axis rotation
        backtrack - bool True/False condition if back tracking algorithm should be applied 
        relative_spacing_rows - the relative spacing calculated as (distance between axles)/(tracker wing width)
        relative_spacing_columns - the relative spacing calculated as (distance between axles)/(tracker wing width)
    outputs:
        GammaNV - module inclination (considering tracking) (matrix, radians)
        rot - tracker axis rotation angle in degrees
    tracker axis rotation angle in degrees
        incidence_angle - angle between module normal and sun (radians, matrix)
        cos_incidence_angle = cos_incidence_angle - cosine of incidence angle (matrix)
        sinGammaNV, cosGammaNV
    '''

    sinAN, cosAN, sinGN, cosGN = None, None, None, None
    if mounting in [1,2,3,4]:
        sinAN = np.sin(AN)
        cosAN = np.cos(AN)
    if mounting in [1,2,3,4,5,6]:
        sinGN = np.sin(GN)
        cosGN = np.cos(GN)

    if mounting in [1,2,3,4]: #Fixed 1 position system - FixedOneAngle
        GammaNV,sinGammaNV,cosGammaNV = GN*np.ones_like(a0), sinGN*np.ones_like(a0), cosGN*np.ones_like(a0)
        ANV,    sinANV,    cosANV     = AN*np.ones_like(a0), sinAN*np.ones_like(a0), cosAN*np.ones_like(a0)
        rot,    sinrot,    cosrot     = GammaNV,             sinGammaNV,             cosGammaNV
        cos_incidence_angle=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,sina0,cosa0,sinh0,cosh0)

    elif mounting==5: #"PV1xA - Tracker vertical, inclined:" - OneAxisVertical
        GammaNV,sinGammaNV,cosGammaNV,ANV,sinANV,cosANV,cos_incidence_angle,rot,sinrot,cosrot=\
        Tracker_1x_VI(latitude,a0,sina0,cosa0,h0,sinh0,cosh0,GN,sinGN,cosGN, rotation_limit,rotation_limit2,relative_spacing_columns,relative_spacing_rows,backtrack)

    elif mounting in [6,7]: #"PV1xB - Tracker inclined NS and special case horizontal NS tracker mounting=7" - OneAxisInclined, OneAxisHorizontalNS
        if mounting==7: sinGN,cosGN=0,1
        GammaNV,sinGammaNV,cosGammaNV,ANV,sinANV,cosANV,cos_incidence_angle,rot,sinrot,cosrot=\
        Tracker_1x_NS(latitude,a0,sina0,cosa0,h0,sinh0,cosh0,GN,sinGN,cosGN, rotation_limit,rotation_limit2,relative_spacing_columns,relative_spacing_rows,backtrack)

    elif mounting==8: #"PV1xC - Tracker horizontal EW:" - OneAxisHorizontalEW
        GammaNV,sinGammaNV,cosGammaNV,ANV,sinANV,cosANV,cos_incidence_angle,rot,sinrot,cosrot=\
        Tracker_1x_EW(latitude,a0,sina0,cosa0,h0,sinh0,cosh0,GN,sinGN,cosGN, rotation_limit,rotation_limit2,relative_spacing_columns,relative_spacing_rows,backtrack)

    elif mounting in [9,10]:  # PV2x - Tracker two axis - ???? TwoAxisAstronomical - not only astronomical
        GammaNV,sinGammaNV,cosGammaNV,ANV,sinANV,cosANV,cos_incidence_angle,rot,sinrot,cosrot=\
        Tracker_2x(latitude,a0,sina0,cosa0,h0,sinh0,cosh0,GN,sinGN,cosGN, rotation_limit,rotation_limit2,relative_spacing_columns,relative_spacing_rows,backtrack)
    else:
        return None

    cos_incidence_angle[cos_incidence_angle>1]=1
    cos_incidence_angle[cos_incidence_angle<-1]=-1
    incidence_angle = np.arccos(cos_incidence_angle)

    return incidence_angle, cos_incidence_angle, GammaNV ,sinGammaNV, cosGammaNV, ANV, sinANV, cosANV, rot, sinrot, cosrot


def Tracker_1x_VI(latitude,a0,sina0,cosa0,h0,sinh0,cosh0,GN,sinGN,cosGN, rotation_limit,rotation_limit2,relative_spacing_columns,relative_spacing_rows,backtrack):
    #"PV1xA - Tracker vertical, inclined:"

    if rotation_limit is None:
        rotation_limit=np.array([-np.pi,np.pi])

    GammaNV=GN*np.ones_like(a0)
    sinGammaNV=sinGN*np.ones_like(a0)
    cosGammaNV=cosGN*np.ones_like(a0)

    rot=a0.copy()
    if latitude<0:
        rot[a0<0]+=np.pi
        rot[a0>=0]-=np.pi

    if backtrack:
        # Azimuthal backtracking
        a=np.cos(rot)*relative_spacing_columns
        a[a<-1]=-1
        a[a>1]=1
        rotation_corrector=np.arccos(a)
        rotation_corrected=rot.copy()
        rotation_corrected[rot>=0]=rot[rot>=0]-rotation_corrector[rot>=0]
        rotation_corrected[rot<0]=rot[rot<0]+rotation_corrector[rot<0]
        rot=rotation_corrected

    rot[rot>rotation_limit[1]]=rotation_limit[1]
    rot[rot<rotation_limit[0]]=rotation_limit[0]

    ANV=rot.copy()

    if latitude<0:
        ANV[rot>0]=ANV[rot>0]-np.pi
        ANV[rot<=0]=ANV[rot<=0]+np.pi

    sinANV,cosANV=np.sin(ANV),np.cos(ANV)
    sinrot,cosrot=np.sin(rot),np.cos(rot)

    cos_incidence_angle=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,sina0,cosa0,sinh0,cosh0)

    return GammaNV,sinGammaNV,cosGammaNV,ANV,sinANV,cosANV,cos_incidence_angle,rot,sinrot,cosrot


def Tracker_1x_NS(latitude,a0,sina0,cosa0,h0,sinh0,cosh0,GN,sinGN,cosGN, rotation_limit,rotation_limit,relative_spacing_columns,relative_spacing_rows,backtrack):
    #"PV1xB - Tracker inclined NS and special case horizontal NS tracker mounting=7"
    lat_sign=np.sign(latitude)
    if lat_sign==0:lat_sign=1
    
    sinGammaNV=sinGN*np.ones_like(a0)
    cosGammaNV=cosGN*np.ones_like(a0)

    if rotation_limit is None:
        rotation_limit=np.array([-np.pi,np.pi])

    phioptB=optPhiB(sina0,cosa0,sinh0,cosh0,sinGammaNV*lat_sign,cosGammaNV)
    sinphioptB,cosphioptB=np.sin(phioptB),np.cos(phioptB)

    if backtrack:
        a=relative_spacing_columns*cosphioptB
        a[a>1]=1
        a[a<-1]=-1
        phioptB_corrector=np.arccos(a)
        phioptB_corrected=phioptB.copy()
        phioptB_corrected[phioptB>=0]=phioptB[phioptB>=0]-phioptB_corrector[phioptB>=0]
        phioptB_corrected[phioptB<0]=phioptB[phioptB<0]+phioptB_corrector[phioptB<0]
        phioptB=phioptB_corrected

    phioptB[phioptB>rotation_limit[1]]=rotation_limit[1]
    phioptB[phioptB<rotation_limit[0]]=rotation_limit[0]

    sinrot, cosrot = np.sin(phioptB), np.cos(phioptB)
    rot=phioptB

    cos_incidence_angle=Rel_sunpos_V2(sinrot,cosrot,sinGammaNV*lat_sign,cosGammaNV,sina0,cosa0,sinh0,cosh0)

    cosGammaNV=cosrot*cosGammaNV
    sinGammaNV=np.power(1-np.power(cosGammaNV,2),0.5)
    GammaNV=np.arcsin(sinGammaNV)

    if lat_sign==-1:
       ANV=np.arctan2(-sinrot,-sinGN*cosrot)*lat_sign
       sinANV=np.sin(ANV)
       cosANV=np.cos(ANV)
    else:
       ANV=np.arctan2(sinrot,sinGN*cosrot)
       sinANV=np.sin(ANV)
       cosANV=np.cos(ANV)

    return GammaNV,sinGammaNV,cosGammaNV,ANV,sinANV,cosANV,cos_incidence_angle,rot,sinrot,cosrot



def Tracker_1x_EW(latitude,a0,sina0,cosa0,h0,sinh0,cosh0,GN,sinGN,cosGN, rotation_limit,rotation_limit2,relative_spacing_columns,relative_spacing_rows,backtrack):

    #"PV1xC - Tracker horizontal EW:"
    lat_sign=np.sign(latitude)
    if lat_sign==0:lat_sign=1

    GammaNV=optPhiC(sina0,cosa0,sinh0,cosh0)
    cosGammaNV=np.cos(GammaNV)
    
    if backtrack:
        # Tilt backtracking
        a=cosGammaNV*relative_spacing_rows
        a[a>1]=1
        a[a<-1]=-1

        GammaNV_corrector=np.arccos(a)
        GammaNV_corrected=GammaNV.copy()
        GammaNV_corrected[GammaNV>=0]=GammaNV[GammaNV>=0]-GammaNV_corrector[GammaNV>=0]
        GammaNV_corrected[GammaNV<0]=GammaNV[GammaNV<0]+GammaNV_corrector[GammaNV<0]
        GammaNV=GammaNV_corrected

    if lat_sign>0:
        GammaNV[GammaNV<rotation_limit2[0]]=rotation_limit2[0]
        GammaNV[GammaNV>rotation_limit2[1]]=rotation_limit2[1]
    else:
        GammaNV[GammaNV>rotation_limit2[0]*lat_sign]=rotation_limit2[0]*lat_sign
        GammaNV[GammaNV<rotation_limit2[1]*lat_sign]=rotation_limit2[1]*lat_sign

    ANV=np.zeros_like(a0)
    ANV[GammaNV<0]=np.pi
    sinANV=np.sin(ANV)
    cosANV=np.cos(ANV)

    rot=GammaNV.copy()
    sinrot,cosrot=np.sin(GammaNV),np.cos(GammaNV)
    GammaNV[GammaNV<0]=-GammaNV[GammaNV<0]
    sinGammaNV,cosGammaNV=np.sin(GammaNV),np.cos(GammaNV)

    cos_incidence_angle=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,sina0,cosa0,sinh0,cosh0)# from FIX

    return GammaNV,sinGammaNV,cosGammaNV,ANV,sinANV,cosANV,cos_incidence_angle,rot,sinrot,cosrot



def Tracker_2x(latitude,a0,sina0,cosa0,h0,sinh0,cosh0,GN,sinGN,cosGN, rotation_limit,rotation_limit2,relative_spacing_columns,relative_spacing_rows,backtrack):

    # PV2x - Tracker two axis
    if rotation_limit==None:
        rotation_limit=np.array([-np.pi,np.pi])
    if rotation_limit2==None:
        rotation_limit2=np.array((-np.pi/2.,np.pi/2.))

    GammaNV = np.pi/2.- h0
    rot=a0.copy()

    if latitude<0:
        rot[a0<0]+=np.pi
        rot[a0>=0]-=np.pi

    cosGammaNV=np.cos(GammaNV)

    rot[rot>rotation_limit[1]]=rotation_limit[1]
    rot[rot<rotation_limit[0]]=rotation_limit[0]
    
    if backtrack:
        # Tilt backtracking
        a=cosGammaNV*relative_spacing_columns
        a[a>1]=1
        GammaNV_corrector=np.arccos(a)
        GammaNV=GammaNV-GammaNV_corrector

    GammaNV[GammaNV<rotation_limit2[0]]=rotation_limit2[0]
    GammaNV[GammaNV>rotation_limit2[1]]=rotation_limit2[1]

    ANV=rot.copy()

    if latitude<0:
        ANV[rot>0]=rot[rot>0]-np.pi
        ANV[rot<=0]=rot[rot<=0]+np.pi

    sinGammaNV,cosGammaNV=np.sin(GammaNV),np.cos(GammaNV)
    sinANV,cosANV=np.sin(ANV),np.cos(ANV)
    sinrot,cosrot=np.sin(rot),np.cos(rot)
    
    cos_incidence_angle=Rel_sunpos_V1(sinGammaNV,cosGammaNV,sinANV,cosANV,sina0,cosa0,sinh0,cosh0)
    return GammaNV,sinGammaNV,cosGammaNV,ANV,sinANV,cosANV,cos_incidence_angle,rot,sinrot,cosrot
    



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
    # TiltTerrainSN - sklon svahu v smere NS [degrees] (SEVER - Positive angle, JUH - Negative value)
    # GammaN - Naklon panelu [degrees]
    # ShadAng - shading angle [degrees]
    # spac_horiz - Rows spacing (bottom to bottom) at the map (horizontal)
    # spac_surf - Rows spacing (bottom to bottom) in the terrain (surface)
    tanShad=np.tan(np.radians(ShadAng))
    tanTiltTerrainSN=np.tan(np.radians(TiltTerrainSN))
    cosTiltTerrainSN=np.cos(np.radians(TiltTerrainSN))
    cosGammaN=np.cos(np.radians(GammaN))
    sinGammaN=np.sin(np.radians(GammaN))
    spac_horiz=l*(cosGammaN*tanShad+sinGammaN)/(tanShad-tanTiltTerrainSN)
    
    spac_surf=spac_horiz/cosTiltTerrainSN
    return spac_horiz,spac_surf

def rowspac2angles(TiltTerrainSN,GammaN,spac_horiz,l):
    # Convert  row spacing 2 angles
    # l - sirka panelu (if l=1, output is relative spacing)
    # TiltTerrainSN - sklon svahu v smere NS [degrees] (SEVER - Positive angle, JUH - Negative value)
    # GammaN - Naklon panelu [degrees]
    # ShadAng - shading angle [degrees]
    # spac_horiz - Rows spacing (bottom to bottom) at the map (horizontal)
    # spac_surf - Rows spacing (bottom to bottom) in the terrain (surface)

    tanTiltTerrainSN=np.tan(np.radians(TiltTerrainSN))
    cosGammaN=np.cos(np.radians(GammaN))
    sinGammaN=np.sin(np.radians(GammaN))
    spl=spac_horiz/l
    tanShad=(spl*tanTiltTerrainSN+sinGammaN)/(spl-cosGammaN)
    return np.degrees(np.arctan(tanShad))

    
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
    sinGammaN=np.sin(np.radians(GammaN))
    cosGammaN=np.cos(np.radians(GammaN))
    sinAN=np.sin(np.radians(-AN))
    cosAN=np.cos(np.radians(-AN))
    
    V=np.array([0,0,1])
    V=rotXmatrix(V,sinGammaN,cosGammaN)
    V=rotZmatrix(V,sinAN,cosAN)
   
    incEW_d=np.degrees(np.arctan(V[0]/V[2]))
    incNS_d=np.degrees(np.arctan(V[1]/V[2]))
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
    sy = r*np.sin(np.pi/2.)*np.cos(a0)
    sx = r*np.sin(np.pi/2.)*np.sin(a0)
    sz = r*np.cos(np.pi/2.-h0)
    return sx,sy,sz

def neighbour_corners_rows(M,off_y,tanTiltSN,sinAN=0,cosAN=1):
    corn0=M[0,0,:]
    corn1=M[0,-1,:]
    # off_x - spacing between rows
    # shift_z - height difference between tows due to slope
    shift_z = tanTiltSN*off_y
    #offsetMTX=rotZmatrix(array(([0,-off_y, -shift_z])),sinAN,cosAN)
    offsetMTX=(np.array(([0,-off_y, -shift_z])))
    return np.array([corn1-offsetMTX,corn0-offsetMTX])


def prepare_reference_module(module_x,module_y,hor_sect,vert_sect):
    linX=np.linspace(-module_x/2.,module_x/2.,hor_sect)
    linY=np.linspace(-module_y/2.,module_y/2.,vert_sect)
    MTX_Y,MTX_X=np.meshgrid(linY,linX)
    M=np.zeros((len(linX),len(linY),3),dtype=np.float64)
    M[:,:,0]=MTX_X
    M[:,:,1]=MTX_Y
    return M,MTX_Y,MTX_X


def prepare_3D_scene(M,rows_back,rows_front,rows_leftright,off_x,off_y,lat_sign_n):
    #Ms matrx with coordinates of all points within all neighbeuring heliostats
    # DIMS(grid_points_x,grid_points_y,3 dimensions xyz, number of neighbours left-right,number of neighbours front )  
    Ms=np.zeros((M.shape[0],M.shape[1],3,rows_front+1+rows_back,rows_leftright*2+1))
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
                edges=np.vstack((edges,Ms[ :,-1,:,f_idx,lr_idx]))
                edges=np.vstack((edges,Ms[-1, :,:,f_idx,lr_idx]))
                edges=np.vstack((edges,Ms[ 0, :,:,f_idx,lr_idx]))
            else:
                edges=np.vstack((edges,Ms[ :, 0,:,f_idx,lr_idx]))
                edges=np.vstack((edges,Ms[ :,-1,:,f_idx,lr_idx]))
                edges=np.vstack((edges,Ms[-1, :,:,f_idx,lr_idx]))
                edges=np.vstack((edges,Ms[ 0, :,:,f_idx,lr_idx]))
            start=False    
    return edges


def reduce_scene_to_corner_points(Ms,rows_back,rows_leftright):

    cp=np.empty((Ms.shape[3]*Ms.shape[4]*4-4,3))
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
    if angle>1.95*np.pi:
        return True
    else:
        return False
    

def vect_angle(v1,v2):
    dot_prod=v1[0]*v2[0]+v1[1]*v2[1]
    norm=np.power(sum(v1*v1)*sum(v2*v2),.5)
    try:
        acos=np.arccos(dot_prod/norm)
    except:
        return 0
    return acos


def check_point_position_v(p1,p2,p3,p4,sun):
    v1 = p1-sun
    v2 = p2-sun
    angle = vect_angle_v(v1,v2)
    v3 = p3-sun
    angle += vect_angle_v(v2,v3)
    v4 = p4-sun
    angle += vect_angle_v(v3,v4)
    angle += vect_angle_v(v4,v1)
    angle[angle <= 1.95 * np.pi]=0
    angle[angle > 1.95 * np.pi]=1
    return angle
    

def vect_angle_v(v1,v2):
    dot_prod = v1[0,:,:,:]*v2[0,:,:,:]+v1[1,:,:,:]*v2[1,:,:,:]
    norm = np.sqrt((np.square(v1[0,:,:,:])+np.square(v1[1,:,:,:]))*(np.square(v2[0,:,:,:])+np.square(v2[1,:,:,:])))
    try:
        acos = np.arccos(dot_prod/norm)
    except:
        return 0
    return acos


def cartesian2polar_v(vectors):
    h0 = np.arcsin((vectors[:,:,:,2])/np.sqrt(np.square(vectors[:,:,:,0])+np.square(vectors[:,:,:,1])+np.square(vectors[:,:,:,2])))
    a0 = np.arctan2(vectors[:,:,:,0],vectors[:,:,:,1])
    return a0, h0
                  


def PV_extract_mounting_details(rad_configDict, pvDict):    
    mounting=rad_configDict["mounting"]
    
    try:
        aspect_r=np.radians(rad_configDict["azimuth"]-180.)
    except:
        aspect_r=0
    try:
        tilt_r=np.radians(rad_configDict["tilt"])
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
    if mounting not in [1,6,7]: do_interrow_shading = False
    
    do_backtrack=False
    if pvDict.has_key("pvTrackerBackTrack"):
        do_backtrack=pvDict["pvTrackerBackTrack"]

    relative_spacing_columns=1e10
    relative_spacing_rows=1e10

    if mounting==5:
        rotation_limit=np.array([-np.pi,np.pi])
    else:
        rotation_limit=np.array([-np.pi/2.,np.pi/2.])
        rotation_limit2=np.array([-np.pi,np.pi])
    
    #rows and columns spacing
    if mounting==1: # FIX
        if not rad_configDict.has_key("tilt"):
            do_interrow_shading=False
        if pvDict.has_key("pvFieldRowSpacingRelative"):
            relative_spacing_rows=pvDict["pvFieldRowSpacingRelative"]
        elif pvDict.has_key("pvFieldRowSpacingShadingAngle") and do_interrow_shading:
            #adapt for terrain slope and orientation
            TiltTerrainSN_d = inc_azim2ew_ns(terrainSlope, terrainAzimuth)[1]
            relative_spacing_rows = angles2rowspac(TiltTerrainSN_d,np.degrees(tilt_r),pvDict["pvFieldRowSpacingShadingAngle"],l=1)[1]
        elif pvDict.has_key("pvFieldRowSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
            relative_spacing_rows=pvDict["pvFieldRowSpacingAbsolute"]/pvDict["pvFieldTableWidth"]
        else:
            do_interrow_shading=False

    elif mounting==6: # t1x NS inclined
        if pvDict.has_key("pvFieldColumnSpacingRelative"):
            relative_spacing_columns=pvDict["pvFieldColumnSpacingRelative"]
        elif pvDict.has_key("pvFieldColumnSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
            relative_spacing_columns=pvDict["pvFieldColumnSpacingAbsolute"]/pvDict["pvFieldTableWidth"]
        if pvDict.has_key("pvFieldRowSpacingRelative"):
            relative_spacing_rows=pvDict["pvFieldRowSpacingRelative"]
        elif pvDict.has_key("pvFieldRowSpacingAbsolute") and pvDict.has_key("pvFieldTableLength"):
            relative_spacing_rows=pvDict["pvFieldRowSpacingAbsolute"]/pvDict["pvFieldTableLength"]


    elif mounting==7: # t1x NS horizontal
        if pvDict.has_key("pvFieldColumnSpacingRelative"):
            relative_spacing_columns=pvDict["pvFieldColumnSpacingRelative"]
        elif pvDict.has_key("pvFieldColumnSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
            relative_spacing_columns=pvDict["pvFieldColumnSpacingAbsolute"]/pvDict["pvFieldTableWidth"]


    elif mounting==8: # t1x EW horizontal
        if pvDict.has_key("pvFieldRowSpacingRelative"):
            relative_spacing_rows=pvDict["pvFieldRowSpacingRelative"]
        elif pvDict.has_key("pvFieldRowSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
            relative_spacing_rows=pvDict["pvFieldRowSpacingAbsolute"]/pvDict["pvFieldTableWidth"]


    elif (mounting==9) or (mounting==5): # t2x and t1xV (vertical axis, inclined module)
        if pvDict.has_key("pvFieldColumnSpacingRelative"):
            relative_spacing_columns=pvDict["pvFieldColumnSpacingRelative"]
        elif pvDict.has_key("pvFieldColumnSpacingAbsolute") and pvDict.has_key("pvFieldTableLength"):
            relative_spacing_columns=pvDict["pvFieldColumnSpacingAbsolute"]/pvDict["pvFieldTableLength"]
        if pvDict.has_key("pvFieldRowSpacingRelative"):
            relative_spacing_rows=pvDict["pvFieldRowSpacingRelative"]
        elif pvDict.has_key("pvFieldRowSpacingAbsolute") and pvDict.has_key("pvFieldTableWidth"):
            relative_spacing_rows=pvDict["pvFieldRowSpacingAbsolute"]/pvDict["pvFieldTableWidth"]

    if pvDict.has_key("pvTrackerRotLimEast") and pvDict.has_key("pvTrackerRotLimWest"):
        rotation_limit=np.radians([pvDict["pvTrackerRotLimWest"],pvDict["pvTrackerRotLimEast"]])
    if pvDict.has_key("pvTrackerTiltRotLimToEquator") and pvDict.has_key("pvTrackerTiltRotLimFromEquator"):
        rotation_limit2=np.radians([pvDict["pvTrackerTiltRotLimFromEquator"], pvDict["pvTrackerTiltRotLimToEquator"]])

    return mounting, aspect_r, tilt_r, terrainSlope,  terrainAzimuth, do_interrow_shading,  do_backtrack, relative_spacing_columns, relative_spacing_rows, rotation_limit, rotation_limit2


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

    #precalculated arrays
    phiopt_V=[]
    indexes=[]
    RightBoundary0_V=[]
    tan_alfa_V=[]
    counter=0
    for phiopt in np.arange(RotRange,0,-np.radians(1)):
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
    optPhiB=optPhiB(sina0,cosa0,sinh0,cosh0,sinGammaN,cosGammaN)
    sinphioptB,cosphioptB=np.sin(optPhiB),np.cos(optPhiB)
    sinphioptB,cosphioptB=optPhiB(sina0,cosa0,sinh0,cosh0,sinGammaN,cosGammaN)
    sinphioptB[cosphioptB<=cosRotRange]=np.sign(sinphioptB[cosphioptB<=cosRotRange])*sinRotRange
    cosphioptB[cosphioptB<=cosRotRange]=cosRotRange
    phioptB=(np.arcsin(sinphioptB))
    #np.savetxt("opt_oryg.out",np.round(np.degrees(phioptB),1),fmt='%1.2f',delimiter=';')



    #np.savetxt("opt_orig.out",np.round(np.degrees(phioptB),1),fmt='%1.2f',delimiter=';')
    #np.savetxt("h0refr.out",np.round(np.degrees((h0refr)),1),fmt='%1.2f',delimiter=';')
    input_dims = sina0.ndim
    if input_dims == 4:
        Tracker0B_Position = np.empty_like(sina0)
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


    #np.savetxt("opt_modif.out",np.round(np.degrees(Tracker0B_Position),1),fmt='%1.2f',delimiter=';')
    return np.sin(Tracker0B_Position),np.cos(Tracker0B_Position)
