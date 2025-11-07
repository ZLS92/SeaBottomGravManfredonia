# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:51:25 2019

@author: lzampa
"""

# -----------------------------------------------------------------------------
# Import libraries

from . import utils as utl
from . import shp_tools as shp
from . import raster_tools as rt
from . import grav_model as gm

import harmonica as hm
import multiprocessing as mpr
import argparse
import functools
import platform
import matplotlib as mplt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Set the aliases for some libraries from the utils and other module

np = utl.np
os = utl.os
sys = utl.sys
ogr = utl.ogr
osr = utl.osr
gdal = utl.gdal
plt = utl.plt
time = utl.time
copy = utl.copy

# -----------------------------------------------------------------------------
# Alias for the current directory (main dir.)
mdir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Constants
G = utl.G
M = utl.M
a_wgs84 = utl.a_wgs84
c_wgs84 = utl.c_wgs84
R_wgs84 = utl.R_wgs84
J2_wgs84 = utl.J2_wgs84
w_wgs84 = utl.w_wgs84 
  
# -----------------------------------------------------------------------------
def Radi1( herr=5, R=R_wgs84 ): 
    
    R1 = np.sqrt( herr * 2 * R ) 
    
    return R1

# -----------------------------------------------------------------------------
def GrdStep2( R1, h, errg, gs=25, R=R_wgs84, dc=2670, 
              plot=False, xmax=None, ymax=None, fs=14 ):  
    """
    Empirical estimate of far-field gridstep. 
    Computation starts from a distance 
    R1 and calculates the grav. effect of a squared 3D ring.
    h = constant heigh of the ring
    gs = incremental step to extend the ring outward.
    errg = accuracy of gravity data.
    N.B Computation is made using Tesseroids (i.e. sherical coordinates).
    """  
             
    R1d = utl.m2deg(R1)
    gd = utl.m2deg(gs)
    rd = np.arange(gd, 3*R1d, gd)
    bottom, top = R, R+h
    t_st = (0, 0, top)
    den = [dc,dc,dc,dc]  
    r_eff = np.zeros(np.size(rd))
    d_eff = np.zeros(np.size(rd))
    for i, ri in enumerate(rd):
        e1 = [[-R1d, R1d, R1d, R1d+ri, bottom, top]]
        e2 = [[R1d, R1d+ri, -R1d-ri, R1d+ri, bottom, top]]
        e3 = [[-R1d, R1d, -R1d-ri, -R1d, bottom, top]]
        e4 = [[-R1d-ri, -R1d, -R1d-ri, R1d+ri, bottom, top]]
        e = e1+e2+e3+e4
        r_eff[i] = hm.tesseroid_gravity(t_st, e, density=den, field="g_z")
        n=i-1
        if i==0: n=0     
        d_eff[i] = r_eff[i]-r_eff[n]
    
    r = utl.deg2m(rd)    
    gs2d=np.interp(errg, r_eff, rd)
    gs2=np.interp(errg, r_eff, r)
    
    # Plot --------------------------------------------------------------------
    if plot==True:
        rstr = f'''gs2 = {np.round(gs2/1e3, 2)} km 
        g = {np.round(errg, 4)} mGal
        h = {np.round(h,0)} m'''
        plt.plot(np.array(r)/1e3,r_eff)
        plt.xlabel('gs2 [km]',fontsize=fs); plt.ylabel('g [mGal]',fontsize=fs)
        plt.vlines(gs2/1e3, 0, errg, linestyle="dashed")
        plt.hlines(errg, 0, gs2/1e3, linestyle="dashed")
        plt.scatter(gs2/1e3, errg, zorder=2);plt.xlim(left=0);plt.ylim(bottom=0)
        box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        if xmax== None: xmax=np.mean(r)/1e3
        if ymax== None: ymax=np.mean(r_eff)
        plt.xlim(0,xmax); plt.ylim(0,ymax);         
        plt.text(0.05, 0.95, rstr, fontsize=fs, transform=plt.gca().transAxes,
                 ha='left', va='top', linespacing=2, bbox=box_props)
        plt.tick_params(labelsize=fs)     

    return (gs2,gs2d), r_eff, d_eff  

# -----------------------------------------------------------------------------
def Radi2(R1, h, errg, gs=100, Rmax=166735, R=R_wgs84, dc=2670, 
          plot=False, xmax=None, ymax=None, font_size=14):
    """
    Empirical estimate of the far-field R2 radius, starting from a distance 
    Rmax and calculating the grav. effect of a squared 3D-ring
    h = constant heigh of the ring
    gs = incremental step to extend the ring inward.
    errg = accuracy of gravity data.
    N.B Computation is made by mean of Tesseroids (i.e. spherical coordinates)
    """  
       
    Rmaxd = utl.m2deg(Rmax)
    gsd = utl.m2deg(gs)
    R1d = utl.m2deg(R1)
    rd = np.arange(R1d, Rmaxd-gsd, gsd)
    bottom, top = R, R+h
    t_st = (0, 0, top)
    den = [dc,dc,dc,dc]  
    r_eff = np.zeros(np.size(rd))
    d_eff = np.zeros(np.size(rd))
    
    for i, ri in enumerate(rd):
        e1 = [[ -ri, ri, ri, Rmaxd, bottom, top ]]
        e2 = [[ ri, Rmaxd, -Rmaxd, Rmaxd, bottom, top ]]
        e3 = [[ -ri, +ri, -Rmaxd, -ri, bottom, top ]]
        e4 = [[ -Rmaxd, -ri, -Rmaxd, Rmaxd, bottom, top ]]
        e = e1+e2+e3+e4
        r_eff[i] = hm.tesseroid_gravity(t_st, e, density=den, field="g_z")     
        n = i-1
        if i == 0: 
            n = 0     
        d_eff[i] = r_eff[i]-r_eff[n]
    
    r = utl.deg2m(rd)  
    R2d = np.interp(errg, r_eff[::-1], rd[::-1])
    R2 = np.interp(errg, r_eff[::-1], r[::-1])
    
    # -------------------------------------------------------------------------
    # Plot Result
    if plot == True:
        rstr = f'''R2 = {np.round(R2/1e3, 2)} km 
        g = {np.round(errg, 4)} mGal
        h = {np.round(h,0)} m'''
        plt.plot( np.array( r ) / 1e3, r_eff )
        plt.xlabel( 'R2 [km]', fontsize = font_size )  
        plt.ylabel('g [mGal]', fontsize = font_size )
        plt.vlines(R2/1e3, 0, errg, linestyle="dashed")
        plt.hlines(errg, 0, R2/1e3, linestyle="dashed")
        plt.scatter(R2/1e3, errg, zorder=2);plt.xlim(left=0);plt.ylim(bottom=0)
        box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        if xmax == None: 
            xmax = np.max( r ) / 1e3
        
        if ymax== None: 
            ymax = np.max( r_eff )
            
        plt.xlim( 0, xmax ) 
        plt.ylim( 0, ymax )
        
        plt.text( 0.95, 
                  0.95, 
                  rstr, 
                  fontsize = font_size, 
                  transform = plt.gca().transAxes,
                  ha = 'right', 
                  va = 'top', 
                  linespacing = 2, 
                  ma = 'left', 
                  bbox = box_props )  
        
        plt.tick_params( labelsize = font_size )  

    return (R2, R2d), r_eff, d_eff  

# -----------------------------------------------------------------------------  
def adj_mesh( gs1, R1, gs2, R2 ):
    """
    Adjust mesh borders (R1,R2) and far field grid_step (gs2) 
    to prevent overlapping or gaps between near and far field areas  
    """
    
    R1n = ( round( R1 / gs1 ) * gs1 ) + ( gs1 / 2 )

    k = round( 2 * R1n / gs2 )
    if k % 2 == 0 :
        k -= 1
    gs2n = 2 * R1n / k
    R2n = ( round( R2 / gs2n ) * gs2n ) + ( gs2n / 2 )
    
    return gs1, R1n, gs2n, R2n  

# -----------------------------------------------------------------------------
def mesh_R3( pr2, R3, prjcode_g=None, prjcode_m=None, R=utl.R_wgs84 ) :
    
    if ( prjcode_g != None ) and ( prjcode_m != None ) :
        x1, y1 = utl.prjxy( prjcode_g, prjcode_m, pr2[0][:,0], pr2[0][:,2] )
        x2, y2 = utl.prjxy( prjcode_g, prjcode_m, pr2[0][:,1], pr2[0][:,3] ) 
    else :
       x1, y1 = pr2[0][:,0], pr2[0][:,2] 
       x2, y2 = pr2[0][:,1], pr2[0][:,3] 

    idx = ( x2 >= R3 ) | ( x2 <= -R3 ) | ( y2 >= R3 ) | ( y2 <= -R3 ) 
        
    dx = x2[idx] - x1[idx]
    dy = y2[idx] - y1[idx]
    dz = pr2[0][idx,5] - pr2[0][idx,4]   
    
    den = pr2[1][idx]
    
    xp = x1[idx] + ( dx / 2 )
    yp = y1[idx] + ( dy / 2 )
    zp = pr2[0][idx,4] + ( dz / 2 )  
    
    mass = den * dx * dy * dz
    points = np.vstack( ( xp, yp, zp ) )
    
    pr2 = ( pr2[0][~idx], pr2[1][~idx] )
    pr3 = ( points, mass )
    
    return pr2, pr3
    
# -----------------------------------------------------------------------------
def mesh_xyz( x, y, R1, R2, gs1, gs2, dtm1, dtm2, st_num=0, z=9999, 
              prjcode_in=4326, prjcode_m=None, R=R_wgs84, adjust_mesh=True,
              plot=None, vmin=None, vmax=None, adjust_dtm=True ) :
    """
    Creates the near-field and far-field array starting from a central point (x,y). 
    Returns:
    Mx = lon. of elements borders
    My = lat. of elements borders
    Zc = heght of elemrnts centre
    hst = dtm height of the central point
    """
    
    # Set reference codes obgect for coordinates transformation ---------------
    if prjcode_m is None:
        prjcode_m = f'+proj=ortho +lat_0={y} +lon_0={x} +ellps=sphere +R={R}'
    xm, ym = utl.prjxy( prjcode_in, prjcode_m, x, y ) # metric coordinates
    
    r_val = rt.xy2rasterVal( dtm1, x, y, prjcode=prjcode_in, close=False )[0]
    
    # Adjust mesh borders and step2 to prevent overlapping or gaps between near 
    # and far field     
    if adjust_mesh == True : 
        gs1, R1, gs2, R2 = adj_mesh( gs1, R1, gs2, R2 )  
    
    # Near field lat lon grid -------------------------------------------------
    lim1 = [ -R1, R1, -R1, R1 ] # borders coordinates [m]
    dtm1m = rt.raster_warp( dtm1, 
                            lim = lim1, 
                            method = gdal.GRA_Average, 
                            width = int( R1*2/gs1 ), 
                            height = int( R1*2/gs1 ), 
                            out_prjcode = prjcode_m, 
                            lim_prjcode = prjcode_m, 
                            new_name = 'mesh_dtm1_' + str( int( st_num ) ),
                            new_path = None,
                            extension = 'tif' ) 
    
    Zc1 = np.flipud( dtm1m.GetRasterBand(1).ReadAsArray() ) # elevetion grid [m]
    cpx = int( ( Zc1.shape[0] + 1 ) / 2 ) 
    
    # dtm station height ------------------------------------------------------
    if ( z != 9999 ) and ( adjust_dtm == True )  : 
         Zc1[cpx, cpx] = z 
    else :
         Zc1[cpx, cpx] = r_val
    
    if dtm1m.RasterCount == 2 :
        msk1 = np.round( np.flipud( dtm1m.GetRasterBand(2).ReadAsArray() ) )
    else: 
        msk1 = None    
    
    # Far field lat lon grid --------------------------------------------------
    lim2 = [ -R2, R2, -R2, R2 ] # borders coordinates [m]
    dtm2m = rt.raster_warp( dtm2, 
                            lim = lim2, 
                            method = gdal.GRA_Average,
                            width = int( R2*2/gs2 ), 
                            height = int( R2*2/gs2 ), 
                            out_prjcode = prjcode_m, 
                            lim_prjcode = prjcode_m, 
                            new_name = 'mesh_dtm2_' + str( int( st_num ) ),
                            new_path = None, 
                            extension = 'tif' )
    Zc2 =  np.flipud( dtm2m.GetRasterBand(1).ReadAsArray() ) # elevetion grid [m]

    # set nodata values to 0
    Zc2[ Zc2 == dtm2m.GetRasterBand(1).GetNoDataValue() ] = 0
    Zc1[ Zc1 == dtm1m.GetRasterBand(1).GetNoDataValue() ] = 0

    grid1 = np.linspace( -R1, R1, int( R1*2/gs1 ) + 1 )
    # grid1 = np.arange( -R1, R1, gs1 )
    grid2 = np.linspace( -R2, R2, int( R2*2/gs2 ) + 1 )
    # grid2 = np.arange( -R2, R2, gs2 )
    Mx_1m, My_1m = np.meshgrid( grid1, grid1 ) # near field mesh
    Mx_2m, My_2m = np.meshgrid( grid2, grid2 ) # far field mesh



    if dtm2m.RasterCount == 2 :
        msk2 = np.round( np.flipud( dtm2m.GetRasterBand(2).ReadAsArray() ) )
    else: 
        msk2 = None
  
    # Plot mesh ---------------------------------------------------------------
    if plot is not None:
        print( '====================================================================\n'     
               f'R1={round(R1)} | R2={round(R2)} | gs1={round(gs1)} | gs2={round(gs2)}\n'
               '====================================================================' )
        plt.figure()
        
        if plot == 1: 
            plot_1, plot_2, n = True, False, 0
            
        if plot == 2: 
            plot_1, plot_2, n = False, True, 0
        
        if plot == 12: 
            plot_1, plot_2, n = True, True, 1
            
        if plot_1 == True:
            p1 = plt.subplot( 1, 1 + n, 1 )            
            for i in range( Mx_2m.shape[1] ) :
                p1.plot( Mx_2m[:,i], My_2m[:,i], c='k' )
            for i in range( Mx_2m.shape[0] ) :
                p1.plot( Mx_2m[i,:], My_2m[i,:], c='k' )    
            Mxc, Myc = Mx_2m[ 1:, 1:] - gs2/2, My_2m[ 1:, 1: ] - gs2/2 # conrdinates of prism's center (array)
            p1.scatter( Mxc, Myc, c='b', marker='o' )
            rect = mplt.patches.Rectangle( ( Mx_1m.min(), My_1m.min()),
                                      (Mx_1m.shape[1]-1)*gs1, (Mx_1m.shape[0]-1)*gs1,
                                      linewidth=1, edgecolor='k', facecolor='w',
                                      zorder=2 )
            p1.add_patch( rect )
            for i in range( Mx_1m.shape[1] ) :
                p1.plot( Mx_1m[:,i], My_1m[:,i], c='k' )
            for i in range( Mx_1m.shape[0] ) :
                p1.plot( Mx_1m[i,:], My_1m[i,:], c='k' )    
            Mxc, Myc = Mx_1m[ 1:, 1:] - gs1/2, My_1m[ 1:, 1: ] - gs1/2 # conrdinates of prism's center (array)
            p1.scatter( Mxc, Myc, c='r', marker='o', zorder=2 )
            p1.scatter( 0, 0, c='k', marker='^', zorder=2 )
            xticks = p1.get_xticks() 
            yticks = p1.get_yticks()
            p1.xaxis.set_ticks( xticks )
            p1.set_xticklabels( xticks/1e3 ) 
            p1.yaxis.set_ticks( yticks )
            p1.set_yticklabels( yticks/1e3 ) 
            p1.set_aspect( 'equal', 'box' )
            plt.xlabel( 'Longitude [km]' ) 
            plt.ylabel( 'Latitude [km]' )
               
        if plot_2 == True:
            p2 = plt.subplot(1,1+n,1+n)
            if vmin == None: 
                vmin=np.mean(Zc2)-2*np.std(Zc2)
            if vmax == None: 
                vmax=np.mean(Zc2)+2*np.std(Zc2)    
            im = p2.pcolor(Mx_2m, My_2m, Zc2, 
                           cmap='terrain', edgecolors='white',
                           vmin=vmin,vmax=vmax)
            p2.pcolor(Mx_1m, My_1m, Zc1, cmap='terrain', 
                      edgecolors='white', vmin=vmin, vmax=vmax)
            p2.scatter(0, 0, s=gs2/15, c='m', marker='^')
            xticks = p2.get_xticks() 
            yticks = p2.get_yticks()
            p2.xaxis.set_ticks( xticks )
            p2.set_xticklabels( xticks/1e3 ) 
            p2.yaxis.set_ticks( yticks )
            p2.set_yticklabels( yticks/1e3 ) 
            p2.set_aspect( 'equal', 'box' )
            plt.xlabel('Longitude [km]') 
            plt.ylabel('Latitude [km]')
                
        if plot_2 == True:
            divider = make_axes_locatable(p2)
            cax = divider.append_axes('right', size='3%', pad=0.1)
            plt.gcf().colorbar(im, cax=cax, label='m')
        
        plt.tight_layout()
    # -------------------------------------------------------------------------
    # Remove temporary gdal .vrt files (from both python RAM and disk memory)
    dtm1m_path_name = dtm1m.GetDescription()
    dtm2m_path_name = dtm2m.GetDescription()
    dtm1m = None 
    dtm2m = None
    os.remove( dtm1m_path_name )
    os.remove( dtm2m_path_name )
    
    # -------------------------------------------------------------------------
    # return   
    return ( Mx_1m, Mx_2m ), ( My_1m, My_2m ), ( Zc1, Zc2 ), ( msk1, msk2 )
   
# -----------------------------------------------------------------------------
def mesh_ls( Mx, My, Zc, msk=None, R1=None, R=R_wgs84, dc=2670, dw=1030, z_shift=True ) :
    """
    Crate a prisms mesh valid for both Land and Sea surface stations.
    R1 = limit between near and far field
    R = earth radius (spherical approximation)
    dc, dw = water and crustal densities
    """    
    
    dhx, dhy = np.mean( np.diff( Mx ) ) / 2, np.mean( np.diff( My, axis=0 ) ) / 2
    Mxc, Myc = Mx[ 1:, 1:] - dhx, My[ 1:, 1: ] - dhy # conrdinates of prism's center (array)
    
    if R1 != None :
        Indx = ( Mxc < -R1 ) | ( Mxc > R1 ) | ( Myc < -R1 ) | ( Myc > R1 )
    else: 
        Indx = Zc == Zc 
    
    Me, Mw = Mxc-dhx, Mxc+dhx
    Ms, Mn = Myc-dhy, Myc+dhy

    Z_pos = ( Zc > 0 ) & Indx
    Z_neg = ( Zc < 0 ) & Indx	
    
    # (A,B) compartment sorting -----------------------------------------------
    if msk is not None : # with coastline mask array
        IA = ( msk == 0 ) & Indx
        IB = ( msk == 1 ) & Indx
    if msk is None : # without coastline array mask
        IA = Z_pos
        IB = Z_neg	
    # Bottom 
    MbA, MtA = np.copy( Zc ), np.copy( Zc )
    # Top
    MbB, MtB = np.copy( Zc ), np.copy( Zc ) 
    
    MbA[Z_pos & IA], MtA[Z_pos & IA] = 0, MtA[Z_pos & IA] # compartment A Z_positive
    MbA[Z_neg & IA], MtA[Z_neg & IA] = 0, 0 # compartment A Z_negative
    MbB[Z_pos & IB], MtB[Z_pos & IB] = 0, 0 # compartment B Z_positive
    MbB[Z_neg & IB], MtB[Z_neg & IB] = MtB[Z_neg & IB], 0 # compartment B Z_negative
    
    # desnsity array 
    DA = np.full(Zc.shape, dc) 
    DB = np.full(Zc.shape, -dc+dw) 
    
    # z_shift (curvature correction for prisms) -------------------------------
    if z_shift == True:
        Mxb, Myb = np.copy(Mxc), np.copy(Myc)
        Mxb[Mxb<0], Myb[Myb<0] = Mxb[Mxb<0]-dhx, Myb[Myb<0]-dhy
        Mxb[Mxb>0], Myb[Myb>0] = Mxb[Mxb>0]+dhx, Myb[Myb>0]+dhy
        Mxy2 = Mxb**2 + Myb**2
        Zsh = Mxy2 / ( 2 * R )
        MbA, MtA = MbA-Zsh, MtA-Zsh
        MbB, MtB = MbB-Zsh, MtB-Zsh
        
    ma = np.column_stack((Me[IA].ravel(),Mw[IA].ravel(),
                          Ms[IA].ravel(),Mn[IA].ravel(),
                          MbA[IA].ravel(),MtA[IA].ravel()))
    
    mb = np.column_stack((Me[IB].ravel(),Mw[IB].ravel(),
                          Ms[IB].ravel(),Mn[IB].ravel(),
                          MbB[IB].ravel(),MtB[IB].ravel()))    
    
    m = np.vstack( ( ma, mb ) )
    d = np.hstack( ( DA[IA].ravel(),
                     DB[IB].ravel() ) )
    
    # Delete far field thin elements
    if R1 is not None :
        zero_h = np.abs( m[:,5] - m[:,4] ) < 0.5
        m = m[ ~zero_h ]
        d = d[ ~zero_h ]

    return m, d

# -----------------------------------------------------------------------------
def mesh_sb(Mx, My, Zc, hst, msk=None, R1=None, dc=2670, dw=1030, z_shift=True, R=R_wgs84):
    """
    Crate prism mesh_array valid for sea-bottom stations.
    hst = station depth (negative)
    R1 = limit between near and far field
    R = earth radius (spherical approximation)
    dc, dw = water and crustal densities
    """  
      
    dhx, dhy = np.mean( np.diff( Mx ) ) / 2, np.mean( np.diff( My, axis=0 ) ) / 2
    Mxc, Myc = Mx[ 1:, 1: ] - dhx, My[ 1:, 1: ] - dhy # center prism coordinates (array)  
    
    if R1!=None:
        Indx = ( Mxc < -R1 ) | ( Mxc > R1 ) | ( Myc < -R1 ) | ( Myc > R1 )
    else: 
        Indx = Zc == Zc # i.e. all array == True
    
#    Zc[ int(Zc.shape[0]/2)+1, int(Zc.shape[1]/2)+1 ] = hst
    
    Me, Mw = Mxc-dhx, Mxc+dhx
    Ms, Mn = Myc-dhy, Myc+dhy 

    # (A,B,C) compartment sorting ---------------------------------------------    
    if msk is not None : # with coastline mask array    
        IA = ( msk == 0 ) & Indx
        IB = ( Zc > hst ) & ( msk == 1 ) & Indx
        IC = ( Zc < hst ) & ( msk == 1 ) & Indx  
    if msk is None : # with coastline mask array    
        IA = ( Zc > 0 ) & Indx
        IB = ( Zc > hst ) & ( Zc < 0 ) & Indx
        IC = ( Zc < hst ) & ( Zc < 0 ) & Indx          
    # Bottom 
    MbA1, MbA2, MbB1, MbC1 = np.copy(Zc), np.copy(Zc), np.copy(Zc), np.copy(Zc)
    # Top
    MtA1, MtA2, MtB1, MtC1 = np.copy(Zc), np.copy(Zc), np.copy(Zc), np.copy(Zc)
    
    MbA1[IA],  MtA1[IA] = hst, 0    # compartment A (from hst to hdtm, where hdtm > 0)
    MbA2[IA],  MtA2[IA] = 0, MtA2[IA]
    MbB1[IB], MtB1[IB] = hst, MtB1[IB]  # compartment B1 (from hst to hdtm, where hst < hdtm < 0)
    MbC1[IC], MtC1[IC] = MbC1[IC], hst  # compartment C1 (from hdtm to hst, where hdtm < hst < 0)
    
    # desnsity array  
    DA1 = np.full( Zc.shape, dc-dw ) 
    DA2 = np.full( Zc.shape, dc ) 
    DB1 = np.full( Zc.shape, dc )
    DC1 = np.full( Zc.shape, -dc + dw )
    
    # z_shift (curvature correction of prisms) --------------------------------
    if z_shift==True:
        Mxb, Myb = np.copy( Mxc ), np.copy( Myc )
        Mxb[Mxb<0], Myb[Myb<0] = Mxb[Mxb<0] - dhx, Myb[Myb<0] - dhy
        Mxb[Mxb>0], Myb[Myb>0] = Mxb[Mxb>0] + dhx, Myb[Myb>0] + dhy
        Mxy2 = Mxb**2 + Myb**2
        Zsh = Mxy2 / ( 2 * R )
        MbA1, MtA1 = MbA1 - Zsh, MtA1 - Zsh
        MbA2, MtA2 = MbA2 - Zsh, MtA2 - Zsh
        MbB1, MtB1 = MbB1 - Zsh, MtB1 - Zsh
        MbC1, MtC1 = MbC1 - Zsh, MtC1 - Zsh
        
    ma1 = np.column_stack(( Me[IA].ravel(),  Mw[IA].ravel(),
                           Ms[IA].ravel(),  Mn[IA].ravel(),
                          MbA1[IA].ravel(), MtA1[IA].ravel() ))

    ma2 = np.column_stack(( Me[IA].ravel(),  Mw[IA].ravel(),
                           Ms[IA].ravel(),  Mn[IA].ravel(),
                          MbA2[IA].ravel(), MtA2[IA].ravel() ))
    
    mb1 = np.column_stack((  Me[IB].ravel(),   Mw[IB].ravel(),
                             Ms[IB].ravel(),   Mn[IB].ravel(),
                           MbB1[IB].ravel(), MtB1[IB].ravel() )) 
    
    mc1 = np.column_stack((  Me[IC].ravel(),   Mw[IC].ravel(),
                             Ms[IC].ravel(),   Mn[IC].ravel(),
                           MbC1[IC].ravel(), MtC1[IC].ravel() ))
    
    m = np.vstack( ( ma1, ma2, mb1, mc1 ) )
    d = np.hstack( ( DA1[IA].ravel(),
                     DA2[IA].ravel(),
                     DB1[IB].ravel(),
                     DC1[IC].ravel() ) )
    
    m = np.vstack( ( ma1, ma2, mb1, mc1 ) )
    d = np.hstack( ( DA1[IA].ravel(),
                     DA2[IA].ravel(),
                     DB1[IB].ravel(),
                     DC1[IC].ravel() ) )
    
    # Delete far field thin elements
    if R1 is not None :
        zero_h = np.abs( m[:,5] - m[:,4] ) < 0.5
        m = m[ ~zero_h ]
        d = d[ ~zero_h ]    
    
    return m, d 

# -----------------------------------------------------------------------------   
def st_loop( constant_arg, iterable_arg ) :  
    
    # Split constant arguments ------------------------------------------------
    R1, R2, R3, gs1, gs2, dtm1, dtm2, prjcode_g, tess, R, dc, dw, disable_checks, tot_len, \
    adjust_dtm, output_file, z_shift, prjcode_m = constant_arg
    
    # Split iterables arguments -----------------------------------------------     
    xg = iterable_arg[0]
    yg = iterable_arg[1]
    z = iterable_arg[2]
    st_num = iterable_arg[3] 
    st_type = iterable_arg[4]
    te = iterable_arg[5]
    te_near = iterable_arg[6]
    te_far = iterable_arg[7]
    zdtm = iterable_arg[8]
    st_num = int(st_num.item())
    R = utl.local_sph_raduis(yg)

    # Create mesh ( m1 = Near_Field, m2 = Far_Field ) -------------------------
    prjcode_m = f'+proj=ortho +lon_0={xg} +lat_0={yg} +ellps=sphere +R={R}'
    Mx, My, Zc, msk = mesh_xyz( xg, yg, R1, R2, gs1, gs2, dtm1, dtm2, st_num=st_num,
                                z=z, prjcode_in=prjcode_g, adjust_dtm=adjust_dtm, 
                                R=R, prjcode_m=None, plot=None )
                         
    # If the station is on the sea surface, zs is set to 0
    # i.e. computational point attached to the prism's top
    if st_type == 1:
        z = 0
        
    # Store mesh elements in two arry ( near-field( pr1 ), ( far-field( pr2 ) )
    
    if ( st_type == 0 ) or ( st_type == 1 ) :
        pr1 = mesh_ls( Mx[0], My[0], Zc[0], msk[0], R=R, dc=dc, dw=dw, z_shift=z_shift )
        if tess is False :
            pr2 = mesh_ls( Mx[1], My[1], Zc[1], msk[1], R1=R1, R=R, dc=dc, dw=dw, z_shift=z_shift )
        else :
            pr2 = mesh_ls( Mx[1], My[1], Zc[1], msk[1], R1=R1, R=R, dc=dc, dw=dw, z_shift=False )
    
    if st_type == 2 :
        pr1 = mesh_sb( Mx[0], My[0], Zc[0], hst=z, msk=msk[0], R=R, dc=dc, dw=dw, 
            z_shift=z_shift )
        if tess == False :
            pr2 = mesh_sb( Mx[1], My[1], Zc[1], hst=z, msk=msk[1], R1=R1, R=R, dc=dc, dw=dw, 
                z_shift=z_shift ) 
        else :
            pr2 = mesh_sb( Mx[1], My[1], Zc[1], hst=z, msk=msk[1], R1=R1, R=R, dc=dc, dw=dw, 
                z_shift=False ) 
       
    if R3 != None :
        pr2, pr3 = mesh_R3( pr2, R3 )
    
    # Coordinates of computational point (center of the mesh)
    st = [0, 0, z]
    
    # Calculate z_gravity of prisms at station point (st) ---------------------   
    
    if prjcode_m is None : 
        prjcode_m = f'+proj=ortho +lon_0=0 +lat_0=0 +ellps=sphere +R={R}'
    
    nag_near = gm.nagy( st, pr1[0], density=pr1[1] )
    te_near = nag_near.gz()

    if tess == True :
        
        pr2[0][:,0], pr2[0][:,2] = utl.prjxy( prjcode_m, prjcode_g, pr2[0][:,0], pr2[0][:,2] )
        pr2[0][:,1], pr2[0][:,3] = utl.prjxy( prjcode_m, prjcode_g, pr2[0][:,1], pr2[0][:,3] )  
        pr2[0][:,4], pr2[0][:,5] = pr2[0][:,4] + R, pr2[0][:,5] + R 
        te_far = hm.tesseroid_gravity( [ xg, yg, R+z ], pr2[0], density=pr2[1], field="g_z",
            disable_checks=True )
        
    else :

        nag_far = gm.nagy( st, pr2[0], density=pr2[1] )
        te_far = nag_far.gz()

    if R3 != None :
        pr3[0][2,:] = pr3[0][2,:] + R 
        pr3[0][0,:], pr3[0][1,:] = utl.prjxy( prjcode_m, prjcode_g, pr3[0][0,:], pr3[0][1,:] )

        te_far_p = hm.point_gravity( [0, 0, R+z], pr3[0], pr3[1], 'g_z', 
            coordinate_system='spherical')
        te_far = te_far + te_far_p

    # Sum near and far grav. effects
    te = te_near + te_far

    with open(output_file, 'a+') as f:
        
        line = "% 10d % 10f % 10f % 10.2f % 10.2f % 10.5f % 10.5f % 10.5f % 10d\n" % ( 
               st_num, np.round(xg, 6), np.round(yg, 6), np.round(z, 3), np.round(zdtm, 3),
               np.round(te_near, 6), np.round(te_far, 6), np.round(te, 6), st_type)
        
        f.write( line )
        
        f.seek(0)
        n_lines = sum(1 for line in f) - 6
        progress = str( round( 100 * n_lines / tot_len, 2 ) )
        print_line = line[:-1]+'  '+progress+' % \n'
        print( print_line )
        
        f.close() 

    pr1 = None
    pr2 = None
     
    # Return
    return te, te_near, te_far, z, st_num, pr1, pr2

# -----------------------------------------------------------------------------  
def te( x, y, dtm1, z='dtm', gs1=None, dtm2=None, R1=None, R2=None, gs2=None, 
        errg=0.5, herr=5, dc=2670, dw=1030, R=R_wgs84, prjcode_in=4326, tess=True,
        prjcode_m=None, st_type=0, plot=None, output_file='Te_correction', 
        disable_checks=True, cpu=1, ply_coast=None, new_dtm=False, adjust_dtm=False,
        R3=50000, z_shift=True, autoid_seasurf_st=False ):
    """
    Calculate topographic effect using prisms for near-field and prism/tesseroid 
    for far-field computations with Earth curvature correction.

    This function computes the gravitational effect of topography using a hybrid 
    approach: high-resolution prisms for the near-field and lower-resolution 
    prisms or tesseroids for the far-field. A z_shift correction accounts for 
    Earth's curvature when using prisms in Cartesian coordinates.

    Parameters
    ----------
    x, y : array_like
        Coordinates of computational points. Can be single values or arrays.
        Units depend on prjcode_in coordinate system.
    dtm1 : str or gdal.Dataset
        Digital Terrain Model for near-field computations. Can be file path 
        or GDAL dataset object.
    z : array_like or str, optional, default='dtm'
        Station heights. Options:
        - 'dtm': Use DTM values at station locations
        - Single value: Constant height for all stations
        - Array: Individual heights for each station
    gs1 : float, optional
        Grid step for near-field mesh [m]. If None, uses DTM resolution.
    dtm2 : str or gdal.Dataset, optional
        Digital Terrain Model for far-field computations. If None, uses dtm1.
    R1 : float, optional
        Radius of near-field domain [m]. If None, estimated from herr parameter.
    R2 : float, optional
        Radius of far-field domain [m]. If None, estimated empirically.
    gs2 : float, optional
        Grid step for far-field mesh [m]. If None, estimated empirically.
    errg : float, optional, default=0.5
        Accuracy requirement for gravity data [mGal]. Used for automatic 
        parameter estimation.
    herr : float, optional, default=5
        Accuracy of DTM data [m]. Used for R1 estimation.
    dc : float, optional, default=2670
        Crustal density [kg/m³].
    dw : float, optional, default=1030
        Water density [kg/m³].
    R : float, optional, default=R_wgs84
        Earth radius for spherical approximation [m].
    prjcode_in : int or str, optional, default=4326
        Input coordinate system code (PROJ format). Examples:
        - 4326 or 'epsg:4326' for WGS84 geographic
        - 32633 for UTM Zone 33N
    tess : bool, optional, default=True
        Use tesseroids for far-field instead of prisms. Recommended for 
        large-scale computations.
    prjcode_m : str, optional
        Metric projection code for internal computations. If None, uses 
        orthographic projection centered on data.
    st_type : int or array_like, optional, default=0
        Station type(s):
        - 0: Land stations
        - 1: Sea surface stations
        - 2: Sea bottom stations
    plot : int, optional
        Visualization options:
        - None: No plotting
        - 1: Show mesh structure
        - 2: Show topography
        - 12: Show both mesh and topography
    output_file : str, optional, default='Te_correction'
        Output directory/file name for results.
    disable_checks : bool, optional, default=True
        Disable harmonica input validation for faster computation.
    cpu : int, optional, default=1
        Number of CPU cores for parallel processing. If None, uses all available.
    ply_coast : str, optional
        Path to coastline polygon file for land/sea mask creation.
    new_dtm : bool, optional, default=False
        Force recreation of processed DTM files even if they exist.
    adjust_dtm : bool, optional, default=False
        Adjust DTM values at station locations to match z parameter.
    R3 : float, optional, default=50000
        Radius for point mass approximation beyond R2 [m]. Set to None to disable.
    z_shift : bool, optional, default=True
        Apply Earth curvature correction to prism coordinates.

    Returns
    -------
    te : numpy.ndarray
        Total topographic effect [mGal] for each station.
    te_near : numpy.ndarray
        Near-field topographic effect [mGal] for each station.
    te_far : numpy.ndarray
        Far-field topographic effect [mGal] for each station.
    R1 : float
        Final near-field radius used [m].
    R2 : float
        Final far-field radius used [m].
    gs1 : float
        Final near-field grid step used [m].
    gs2 : float
        Final far-field grid step used [m].
    pr1 : tuple
        Near-field prism data (coordinates, densities). Set to None after computation.
    pr2 : tuple
        Far-field prism/tesseroid data (coordinates, densities). Set to None after computation.

    Notes
    -----
    **Algorithm Overview:**
    1. Creates orthographic projection centered on data
    2. Generates near-field high-resolution prism mesh
    3. Generates far-field lower-resolution mesh (prisms or tesseroids)
    4. Applies density models based on station type and bathymetry
    5. Computes gravitational effects using Nagy's method (near) and harmonica (far)
    6. Applies Earth curvature corrections where appropriate
    
    **Coordinate Systems:**
    - Input coordinates are transformed to spherical (geographic) coordinates
    - Internal computations use orthographic projection for each station
    - Final results maintain input coordinate reference

    **Station Types:**
    - Land (0): Standard topographic correction
    - Sea surface (1): Accounts for water layer above seafloor
    - Sea bottom (2): Complex model with water column and seafloor effects

    **Performance:**
    - Supports multiprocessing for large datasets
    - Uses GDAL virtual file system for efficient DTM operations
    - Automatic parameter estimation reduces manual tuning

    **File Output:**
    Creates directory structure with processed DTMs and results file containing:
    - Station coordinates and heights
    - Near-field, far-field, and total effects
    - Processing metadata and timing information

    Examples
    --------
    Basic usage with automatic parameters:

    >>> x = [12.5, 12.6, 12.7]  # Longitude
    >>> y = [41.8, 41.9, 42.0]  # Latitude  
    >>> dtm_file = 'path/to/dtm.tif'
    >>> te_total, te_near, te_far, R1, R2, gs1, gs2, _, _ = te(x, y, dtm_file)

    Marine survey with custom parameters:

    >>> te_total, *_ = te(x, y, dtm_file, 
    ...                   z=-100,           # 100m water depth
    ...                   st_type=2,        # Sea bottom stations
    ...                   gs1=500,          # 500m near-field resolution
    ...                   R1=20000,         # 20km near-field radius
    ...                   cpu=4)            # Use 4 CPU cores

    References
    ----------
    - Nagy, D., Papp, G., & Benedek, J. (2000). The gravitational potential 
      and its derivatives for the prism. Journal of Geodesy, 74(7-8), 552-560.
    - Fatiando a Terra Project, Castro, Y. M., Esteban, F. D., 
      Li, L., Oliveira Jr, V. C., Pesce, A., Shea, N., Soler, S. R., Souza-Junior, 
      G. F., Tankersley, M., Uieda, L., & Uppal, I. (2024). 
      Harmonica v0.7.0: Forward modeling, inversion, and processing gravity 
      and magnetic data (0.7.0). Zenodo. https://doi.org/10.5281/zenodo.13308312

    See Also
    --------
    mesh_xyz : Create computation meshes
    mesh_ls : Generate land/sea prism arrays  
    mesh_sb : Generate sea-bottom prism arrays
    GrdStep2 : Estimate far-field grid step
    Radi2 : Estimate far-field radius
    """

    # Create new directory with computation's resaults ------------------------
    cwd = os.getcwd().split( os.sep )[-1]
    if cwd != output_file.split( os.sep )[-1] :
        ndir = str( np.copy( output_file ) )
        os.makedirs( ndir, exist_ok=True )
    else :
        ndir = os.getcwd()
    output_file = ndir + os.sep + output_file.split( os.sep )[-1]

    if os.path.isfile( output_file ) == True :
        os.remove( output_file )

    # Transform input data type -----------------------------------------------    
    x,y,z = np.asarray(x).ravel(), np.asarray(y).ravel(), np.asarray(z).ravel()
    st_type = np.asarray(st_type).ravel()
    
    # Stations  coordinates conversions ---------------------------------------
    # Convert to geographic coordinates (spherical approx.)
    prjcode_g = f'+proj=longlat +ellps=sphere +R={R}' 
    xg, yg = utl.prjxy( prjcode_in, prjcode_g, x, y )  # geo coordinates 
    xrmg ,yrmg = np.mean( xg ), np.mean( yg )  # geo coordinates of middle point
    # Convert to metric coordinates using orthographic projection
    if prjcode_m is None : 
        prjcode_m = f'+proj=ortho +lon_0={xrmg} +lat_0={yrmg} +ellps=sphere +R={R}'
    xm, ym = utl.prjxy( prjcode_g, prjcode_m, xg, yg ) # metric coordinates
    
    # First "large crop/resempling operation" on raster input files -----------
    # This will reduces the dtm model to the chosen limits and grid_step.
    # NB. A second "small crop/sempling operation" is done within the function 'mesh_xyz',
    # around each station point.
    # NB.NB. We decided to use Gdal functions because they are SUPER-FAST, 
    # when compared to scipy griddata functions. 
    # Gdal functions make use of '.vrt' file-formats to create intermidiate-step datasets.
    # The vrt format allowes superfast on-grid operations.
    # Therefore, our chice is motivated by the fact that Gdal is much more performing 
    # than any other pythonic method when dealing with large dtm arrays. 
    # More info https://gdal.org/user/virtual_file_systems.html.

    if type( dtm1 ) == str :
        gdal.Open( dtm1 )

    if type( dtm2 ) == str :
        gdal.Open( dtm2 )

    if dtm2 == None : 
        dtm2 = dtm1

    # Get dtm1 values at the station point ------------------------------------
    zdtm = rt.xy2rasterVal( dtm1, x, y, prjcode=prjcode_in  )[0]
    # Estimate R1 -------------------------------------------------------------
    if R1==None:
        R1 = Radi1( herr, R=R )

    # Define lim1 -------------------------------------------------------------
    minx, maxx, miny, maxy = np.min(xm), np.max(xm), np.min(ym), np.max(ym)
    lim1 = [ minx-R1-gs1, maxx+R1+gs1, miny-R1-gs1, maxy+R1+gs1 ]

    # Create dtm1 -------------------------------------------------------------
    print( "LOADING/CREATING DTM_1 ...")        
    if ( os.path.isfile( ndir + os.sep + 'dtm1' + os.sep + 'dtm1.tif' ) is False ) or \
       ( new_dtm is True ) :       
        if gs1 == None: # Take gs1 from maximum gridstep of dtm1
            res1 = rt.raster_res( dtm1 )
            gs1 = np.mean( res1 )  
        # Replace nan values with -99999.0 ---------------------------------------
        dtm1 = rt.set_raster_nodata( dtm1, -32768.0 )
        # Crop-Sample to lim1 -------------------------------------------------
        dtm_t1 = rt.raster_warp( 
            dtm1, 
            lim1, 
            xRes = gs1, 
            yRes = gs1, 
            out_prjcode = prjcode_m, 
            lim_prjcode = prjcode_m, 
            method = gdal.GRA_Average, 
            new_name = 'dtmt1', 
            new_path = ndir + os.sep + 'dtm1', 
            extension = 'tif',
            close = True )  

        # Adjust dtm1 values at station points --------------------------------
        if ( adjust_dtm is True ) :
            if ( type( z[0].astype(object) ) == str ) and ( z[0].astype(object) == 'dtm' ) :
                z = zdtm
            dtm_t1= rt.raster_edit( dtm_t1, x, y, z, prjcode=prjcode_in, band=1, close=True,
                                    path=ndir+os.sep+'dtm1', new_name='dtmt1', suffix='_e' ) 
              
        # Create mask for dtm_r1 from coastline polygon (if exist) ------------
        if ply_coast != None :    
            print('Add coast band')
            dtm1_new = rt.add_coastline_band( dtm_t1, ply_coast, 
                                              new_path=ndir+os.sep+'dtm1', new_name='dtm1' )
        else:
            dtm1_new = rt.raster_save( dtm_t1, new_path=ndir+os.sep+'dtm1', new_name='dtm1', 
                                       extension='tif', close=True)

        # Remove temporary rasters --------------------------------------------
        del dtm1, dtm_t1
        
        if os.path.exists( ndir + os.sep + 'dtm1' + os.sep + 'dtmt1.tif' ):
            os.remove( ndir + os.sep + 'dtm1' + os.sep + 'dtmt1.tif' ) 
            
        if os.path.exists( ndir + os.sep + 'dtm1' + os.sep + 'dtmt1_e.tif' ):
            os.remove( ndir + os.sep + 'dtm1' + os.sep + 'dtmt1_e.tif' )         
    else :
        dtm1_new = ndir + os.sep + 'dtm1' + os.sep + 'dtm1.tif' 
        
    # Estimate gs2 ------------------------------------------------------------
    print( "LOADING/CREATING DTM_2 ...")
    if gs2 == None : 
        topo_array = rt.raster2array( dtm1_new )[2]
        h = np.std( np.sqrt( topo_array**2 ) *2 ) + np.mean( np.sqrt( topo_array**2 ) )   
        print( 'Average Slab Thickness, Ring_1 =' + str( h ) )
        gs2, gs2_deg = GrdStep2( R1, h, errg, gs1 )[0]
        if gs2 < gs1 : 
            gs2 = gs1
        if gs2 > R1 : 
            gs2 = R1
        
    # Estimate R2 ---------------------------------------------------------
    if R2==None :  
        Rmax = 167*1e3
        limMax = [ minx-Rmax, maxx+Rmax, miny-Rmax, maxy+Rmax ] 
        dtmmax = rt.raster_warp( dtm2, 
                                 limMax, 
                                 xRes = gs2, 
                                 yRes = gs2, 
                                 out_prjcode = prjcode_m, 
                                 lim_prjcode = prjcode_m, 
                                 new_name = 'dtmmax', 
                                 method = gdal.GRA_Bilinear, 
                                 close = False)

        topo_array = dtmmax.GetRasterBand(1).ReadAsArray()
        h = np.mean( np.sqrt( topo_array**2 ) )
        print( 'Average Slab Thickness, Ring_2 =' + str( h ) )
        R2, R2_deg = Radi2( R1, h, errg, gs2 )[0]
        if R2 < R1 : 
            R2 = R1
        dtmmax = None   
    
    # Create dtm2 ------------------------------------------------------------- 
    if ( os.path.isfile( ndir + os.sep + 'dtm2' + os.sep + 'dtm2.tif' ) is False ) or \
       ( new_dtm is True ) : 

        # Replace nan values with -9999.0
        dtm2 = rt.set_raster_nodata( dtm2, -32768.0 )
        # Crop-Resample dtm2 based on R2 and gs2 ------------------------------
        lim2 = [ minx-R2-gs2, maxx+R2+gs2, miny-R2-gs2, maxy+R2+gs2 ]
        dtm_t2 = rt.raster_warp( dtm2, 
                                 lim = lim2, 
                                 xRes = gs2, 
                                 yRes = gs2, 
                                 out_prjcode = prjcode_m, 
                                 lim_prjcode = prjcode_m, 
                                 method = gdal.GRA_Average, 
                                 new_name ='dtmt2', 
                                 new_path = ndir+os.sep+'dtm2', 
                                 extension = 'tif',
                                 close = False ) 

        # Create mask for dtm_r1 from coastline polygon (if exist) ------------
        if ply_coast != None :    
            dtm2_new = rt.add_coastline_band( dtm_t2, ply_coast, 
                                              new_path=ndir+os.sep+'dtm2', new_name='dtm2' )
        
        else:
            dtm2_new = rt.raster_save( dtm_t2, new_path=ndir+os.sep+'dtm2', new_name='dtm2', 
                                       extension='tif', close=True)

        # Remove temporary rasters --------------------------------------------
        del dtm2, dtm_t2 
        os.remove( ndir + os.sep + 'dtm2' + os.sep + 'dtmt2.tif' )
        
    else:
        dtm2_new = ndir + os.sep + 'dtm2' + os.sep + 'dtm2.tif' 
                
    # Convert station attributes into iterable objects ------------------------
    if np.size( st_type ) < 2 :
       st_type = np.repeat( int( st_type ), np.size( xg ) )
    if ( np.size( xg ) > 1 ) and ( np.size( z ) == 1 ) and ( type( z ) in [float,int] ) :
       z = np.repeat( z, np.size( xg ) ) # if z is only one constant variable
    if ( type( z[0].astype(object) ) == str ) and ( z[0].astype(object) == 'dtm' ) :
       z = zdtm

    # Auto-identify sea-surface stations --------------------------------------
    if autoid_seasurf_st is True :
        dtm_att = rt.xy2rasterVal( dtm1_new, x, y, prjcode=prjcode_in, band=2 )[0]
        idx = ( dtm_att == 1 ) & ( st_type != 2 )
        st_type[ idx ] = 1  # sea-surface stations

    # Create empty arrays to allocate: 1)topo_effect_NearF, 2)topo_effect_FarF, 
    # 3)topo_effect_tot    
    te_near = np.zeros( np.size (xg ) ) #(1)
    te_far = np.zeros( np.size (xg ) ) #(2)
    te = np.zeros( np.size( xg ) ) #(3) 
    
    # Adjust mesh borders and step2 to prevent overlapping or gaps between near 
    # and far field (eventually this operation is done again within mesh_xyz function ... 
    # it doesn't change the resault using function adj_mesh multiple times) 
    gs1, R1, gs2, R2 = adj_mesh( gs1, R1, gs2, R2 ) 
    print( 'Computational Parameters, gs1 gs2 R1 R2 :' )
    print( round(gs1), round(gs2), round(R1), round(R2) )
    
    # Check mesh: plot dtbm mesh to control if the model is OkyDoky 
    if plot != None :
            mesh_xyz( xrmg, yrmg, R1, R2, gs1, gs2, dtm1_new, dtm2_new, prjcode_in=prjcode_g, plot=plot )           
    
    pcol = [ 'st_num', 'Lon', 'Lat', 'h_st', 'h_dtm', 'te_near', 'te_far', 'te_tot', 'st_type', 'prog.%' ]
    
    tot_len = np.size( x )
    st_num = np.arange( 0, np.size(xg), dtype=int )
    
    time_count = time.time()
    header=( '==================================================================================================\n'
             '==================================================================================================\n'     
            f'Computational parameters [m]:\n'
            f'R1 = {round(R1)} | R2 = {round(R2)} | gs1 = {round(gs1)} | gs2 = {round(gs2)}\n'
             '==================================================================================================\n'
            f'{pcol[0]:>10} {pcol[1]:>10} {pcol[2]:>10} {pcol[3]:>10} {pcol[4]:>10}'+
            f' {pcol[5]:>10} {pcol[6]:>10} {pcol[7]:>10} {pcol[8]:>10}') 
    
    with open(output_file, 'w') as f:
        f.write(header + '\n')
        f.close()
        
        # START ITERATION =====================================================
        # Iterative computation over each input grav. station (index st_num) --
        if (R3 is not None ) and ( R3 >= R2 ) : 
            R3 = None
        constant_arg = ( R1, R2, R3, gs1, gs2, 
                         dtm1_new, dtm2_new, prjcode_g, tess, R, dc, dw, 
                         disable_checks, tot_len, adjust_dtm, output_file, 
                         z_shift, prjcode_m )
        
        iterable_arg = np.column_stack( ( xg, yg, z, st_num, st_type, te, te_near, te_far, zdtm ) ) 
                                          
        patial = functools.partial( st_loop, constant_arg ) 

        if cpu is None :
            cpu = mpr.cpu_count()

        print( "LOOP STARTED ( #cpu:",cpu,") ...")
        print( header )
        if cpu > 1 :
            with mpr.Pool(cpu) as p :
                Te_list = list( zip( *p.map( patial, iterable_arg ) ) )
                
        if cpu == 1:
            Te_list = list( zip ( *map( patial, iterable_arg ) ) ) 
               

        te = np.array( ( Te_list[0] ) ) 
        te_near = np.array( ( Te_list[1] ) ) 
        te_far = np.array( ( Te_list[2] ) ) 
        z = np.array( ( Te_list[3] ) ) 
        st_num = np.array( ( Te_list[4] ) )
        pr1 = Te_list[5]
        pr2 = Te_list[6]
            
        # END ITERATION =======================================================   
        End = ( '==================================================================================================\n'   
                '==================================================================================================\n'
               f'Total_time: {time.time()-time_count} sec.' )
        
    with open(output_file, 'w') as f:  
        f.write(header + '\n')
        
        array = np.column_stack( ( st_num,
                                   np.round(xg, 6),
                                   np.round(yg, 6),
                                   np.round(z, 3),
                                   np.round(zdtm, 3),
                                   np.round(te_near, 5),
                                   np.round(te_far, 5),
                                   np.round(te, 5),
                                   st_type ) )
        
        # Sort array according to st_num ( it is necessary when run parallel)
        array = array[ np.argsort( array [ :, 0 ] ) ] 
        fmt = '% 10d % 10f % 10f % 10.2f % 10.2f % 10.4f % 10.4f % 10.4f % 10d'
        np.savetxt( f, array, fmt = fmt  )
        
        f.write(End)
        print( End )
        f.close() 
        
    return te, te_near, te_far, R1, R2, gs1, gs2, pr1, pr2 

# -----------------------------------------------------------------------------
def load_te( file, ds_type='dict' ) :
    """
    Import results
    """
    
    load_resoult = np.genfromtxt( file, skip_header=6, skip_footer=3 ) 
    
    array = load_resoult[ np.argsort(load_resoult[:,0]) ] 
    
    st_num, Lon, Lat, h_st, h_dtm, te_near, te_far, te_tot, st_type \
        = np.hsplit(array, array.shape[1])
        
    d = { 'st_num' : st_num.ravel(),
          'Lon' : Lon.ravel(), 
          'Lat' : Lat.ravel(),
          'h_st' : h_st.ravel(),
          'h_dtm' : h_dtm.ravel(),
          'te_near' : te_near.ravel(), 
          'te_far' : te_far.ravel(), 
          'te_tot' : te_tot.ravel(), 
          'st_type' : st_type.ravel()
        }
    
    # Return 
    if ds_type == 'dict' :
        
        return d
    
    if ds_type == 'array' :
        
        return array
    
    if ds_type == 'all' :
        
        return array, d

# -----------------------------------------------------------------------------
def run_te( x, y, dtm1, z='dtm', gs1=None, dtm2=None, R1=None, R2=None, gs2=None, 
            errg=0.5, herr=5, dc=2670, dw=1030, R=R_wgs84, prjcode_in=4326, tess=True,
            prjcode_m=None, st_type=0, plot=None, output_file='Te_correction', 
            disable_checks=True, cpu=None, ply_coast=None, new_dtm=False, R3=50000 ) :
    
    # Create new directory for computation's resaults -------------------------
    cwd = os.getcwd().split( os.sep )[-1]
    if cwd != output_file.split( os.sep )[-1] :
        ndir = str( np.copy( output_file ) )
        os.makedirs( ndir, exist_ok=True )
    else :
        ndir = os.getcwd()

    xyz = np.column_stack( ( x, y ) )
    
    if type( z ) != str:
        if np.size( z ) > 1 :
            xyz = np.column_stack( ( xyz, z ) )
        else:
            xyz = np.column_stack( ( xyz, np.repeat( z, np.size( x ) ) ) ) 
    else:
        xyz = np.column_stack( ( xyz, np.repeat( 9999, np.size( x ) ) ) )              

    if np.size( st_type ) > 1 :
        xyz = np.column_stack( ( xyz, st_type ) )
    else:
        xyz = np.column_stack( ( xyz, np.repeat( st_type, np.size( x ) ) ) )                
        
    np.savetxt( ndir + os.sep + 'xyz', xyz, fmt='%.6f %.6f %.2f %.0f' )
    
    if type( dtm1 ) != str : dtm1 = dtm1.GetDescription()
    if type( dtm1 ) != str : dtm1 = dtm1.GetDescription()
    if type( gs1 ) != str : gs1 = str( gs1 )
    if type( R1 ) != str : R1 = str( R1 )
    if type( gs2 ) != str : gs2 = str( gs2 )
    if type( R2 ) != str : R2 = str( R2 )
    if type( prjcode_in ) != str : prjcode_in = str( prjcode_in )
    if type( prjcode_m ) != str : prjcode_m = str( prjcode_m )  
    if type( output_file ) != str : output_file = str( output_file ) 
    if type( cpu ) != str : cpu = str( cpu ) 
    if type( ply_coast ) != str : ply_coast = str( ply_coast ) 
    if type( new_dtm ) != str : new_dtm = str( new_dtm ) 
    if type( R3 ) != str : R3 = str( R3 )  
      
    irs = prjcode_in
    pcs = prjcode_m
    out = output_file 
    
    cfn = os.path.abspath(__file__)
    
    run = f'python -W ignore' + \
          f' {cfn}' + \
          f' -xyz {ndir + os.sep + "xyz"}' + \
          f' -dtm1 {dtm1}' + \
          f' -gs1 {gs1}' + \
          f' -R1 {R1}' + \
          f' -dtm2 {dtm2}' + \
          f' -gs2 {gs2}' + \
          f' -R2 {R2}' + \
          f' -irs {irs}' + \
          f' -pcs {pcs}' + \
          f' -out {out}' + \
          f' -cpu {cpu}' + \
          f' -ply_coast {ply_coast}' + \
          f' -tess {tess}' + \
          f' -new_dtm {new_dtm}' + \
          f' -R3 {R3}'
              
    if platform.system() == 'Linux' :
        run_str = ndir + os.sep + 'run.sh'
    if platform.system() == 'Windows' :
        run_str = ndir + os.sep + 'run.bat'
        
    with open( run_str , 'w') as f :
              f.write(run)
              f.close()        
 
    os.system( run_str )
    
    out_file = ndir + os.sep + output_file.split( os.sep )[-1]
    
    result = load_te( out_file, ds_type='all' )
    
    with open( out_file, 'r' ) as f :
        print( f.read() )    
    
    return result
    
# -----------------------------------------------------------------------------
if __name__ == '__main__' :  
        
    p = argparse.ArgumentParser()
    p.add_argument( '-xyz', '--xyz', help='input file with x, y coordinates \
                    of computational points, z is optional', type=str )
    p.add_argument( '-dtm1', '--dtm1', type=str )
    p.add_argument( '-gs1', '--gs1', default=None, type=str )
    p.add_argument( '-R1', '--R1', default=None, type=str )
    p.add_argument( '-dtm2', '--dtm2', default=None, type=str )
    p.add_argument( '-gs2', '--gs2', default=None, type=str )
    p.add_argument( '-R2', '--R2', default=None, type=str )
    p.add_argument( '-R3', '--R3', default=None, type=str )
    p.add_argument( '-irs', '--irs', default=4326, type=str )
    p.add_argument( '-pcs', '--pcs', default=None, type=str )
    p.add_argument( '-del_xyz', '--del_xyz', default=str )
    p.add_argument( '-out', '--out', default='Te_correction', type=str )
    p.add_argument( '-cpu', '--cpu', default=None, type=str )
    p.add_argument( '-ply_coast', '--ply_coast', default=None, type=str )
    p.add_argument( '-tess', '--tess', default=True, type=str )
    p.add_argument( '-new_dtm', '--new_dtm', default=False, type=str )
                
    arg = p.parse_args()
    xyz = np.loadtxt( arg.xyz )

    if xyz.shape[1] == 2 :
        x, y = np.hsplit( xyz, 2 )
        z = 'dtm'
        st_type = 0
    if xyz.shape[1] == 3 :
        x, y, z = np.hsplit( xyz, 3 )
        st = 0
    if xyz.shape[1] == 4 :
        x, y, z, st_type = np.hsplit( xyz, 4 )   
    if np.any( ( z == 9999 ) ) :
        z = 'dtm'             
              
    arg.irs = eval( arg.irs )  
    arg.pcs = eval ( arg.pcs )   
    arg.cpu = eval ( arg.cpu )
    arg.R1 = eval ( arg.R1 )
    arg.R2 = eval ( arg.R2 )
    arg.gs1 = eval ( arg.gs1 )
    arg.gs2 = eval ( arg.gs2 )
    arg.tess = eval ( arg.tess )
    arg.ply_coast = eval ( arg.ply_coast )
    arg.new_dtm = eval ( arg.new_dtm )
    arg.R3 = eval ( arg.R3 )

    te( x, 
        y, 
        dtm1 = arg.dtm1, 
        z = z, 
        gs1 = arg.gs1, 
        dtm2 = arg.dtm2, 
        R1 = arg.R1, 
        R2 = arg.R2, 
        gs2 = arg.gs2, 
        prjcode_in = arg.irs,
        prjcode_m = arg.pcs, 
        st_type = st_type, 
        output_file = arg.out,
        cpu = arg.cpu,
        ply_coast = arg.ply_coast,
        tess = arg.tess,
        new_dtm = arg.new_dtm,
        R3 = arg.R3 )
    
    if arg.del_xyz == True:
        os.remove( arg.xyz )
        
    
        