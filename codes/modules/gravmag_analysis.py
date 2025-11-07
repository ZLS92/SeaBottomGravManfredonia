# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:31:52 2020

@author: lzampa
"""

# -----------------------------------------------------------------------------
# Import libraries

from collections import namedtuple
from . import utils as utl
from . import plot_tools as lz_plot
from . import raster_tools as rt
from . import shp_tools as shp

# import lszpy.te_harmonica as te_hm
from numba import jit, njit 
import numpy as np, scipy as scy, os
from scipy import signal



# -----------------------------------------------------------------------------
# Set the aliases for some libraries from the utils module
np = utl.np
os = utl.os
plt = utl.plt
copy = utl.copy
dt = utl.datetime
datetime = dt.datetime
timedelta = dt.timedelta


# -----------------------------------------------------------------------------
plta = utl.plta
pltr = rt.pltr

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
def AS( array, method='fft', padw=0, pmode='edge', remove='mean', order=[1],
        alpha=0, sx=1.0, sy=1.0, nodata=True, mask=None, 
        plot=False, vmin=None, vmax=None, filt=False, ftype='hanning', iter=1,
        radius=1, factor=2, method_dxy='fft', method_dz='fft' ) :
    
    """
    Analitical Signal (vector sum of gravity field first derivatives)
    """
    
    array = np.copy( array )
    nan = np.isnan( array )
    if remove == 'mean': 
        array = array - np.nanmean(array)
    if remove == 'trend': 
        array = polyfit2d(array=array, order=order)[0]
    
    ar_pad, original_shape_indx = utl.pad_array( array, padw, pmode, alpha=alpha,
                                            constant_values=0)
    
    Dx = dx(ar_pad, sx=sx, method=method_dxy )
    Dy = dy(ar_pad, sy=sy, method=method_dxy )
    Dz = dz(ar_pad, method=method_dz, sx=sx, sy=sy )
    AS = np.sqrt( Dx ** 2 + Dy ** 2 + Dz ** 2 )
    AS = utl.crop_pad( AS, original_shape_indx ) 
    
    if filt is True :
        Dz = utl.filt2d( AS, ftype=ftype, iter=iter, radius=radius, factor=factor )    
    
    if nodata is True :
        AS[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        AS[mask] = np.nan
        arraym = np.copy( array )
        arraym[mask] = np.nan
        
    if plot == True:
        utl.plta(arraym, sbplt=[1, 2, 1], tit='original')
        utl.plta(AS, sbplt=[1, 2, 2], tit='Analytical Signal')
    
    return AS

# -----------------------------------------------------------------------------
def dx( array, padw=0, method='fft', sx=1, n=1, pmode='edge', remove='mean', 
        order=[1], alpha=0, nodata=True, nanfill='gdal', mask=None, plot=False, 
        vmin=None, vmax=None):
    
    """
    Horizzontal derivative in x direction of 2D arrray (to n order)
    """
    
    nan = np.isnan( array )
    if remove is None: arrayr = array
    if remove == 'mean': arrayr = array - np.nanmean( array )
    if remove == 'trend': arrayr = polyfit2d( array=array, order=order )[0]
    if nanfill is not None: arrayr = utl.fillnan( arrayr, method=nanfill )
    
    ar_pad, original_shape_indx = utl.pad_array( arrayr, padw, pmode, alpha=alpha )
    Dx_pad = np.copy( ar_pad )
    
    if method == 'diff':
        for i in range(1, n + 1):
            md = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]]) 
            Dx_pad = utl.sp.ndimage.convolve( Dx_pad, md, mode='nearest' ) / ( 2 * sx )  
        Dx = utl.crop_pad( Dx_pad, original_shape_indx )
            
    if method == 'fft':
        K,fft2,Kx,_,_,_ = fft2d( Dx_pad, sx=sx, sy=sx )
        Dx_pad = np.real( np.fft.ifft2( ( ( 1j * Kx )**n ) * fft2 ) )     
        Dx = utl.crop_pad( Dx_pad, original_shape_indx ) 

    if nodata is True :
        Dx[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        Dx[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        utl.plta(arraym, sbplt=[1, 2, 1], tit='original')
        utl.plta(Dx, sbplt=[1, 2, 2], tit='X derivative')
        
    return Dx

# -----------------------------------------------------------------------------
def dy( array, padw=0, method='fft', n=1, sy=1, pmode='edge', remove='mean', 
        order=[1], alpha=0, nodata=True, nanfill='gdal', mask=None, 
        plot=False, vmin=None, vmax=None):
    
    """
    Horizzontal derivative in x direction of 2D arrray (to n order)
    """    
    
    nan = np.isnan( array )
    if remove is None: arrayr = array  
    if remove == 'mean': arrayr = array - np.nanmean(array)   
    if remove == 'trend': arrayr = polyfit2d(array=array, order=order)[0]
    if nanfill is not None: arrayr = utl.fillnan(arrayr, method=nanfill)
            
    ar_pad, original_shape_indx = utl.pad_array( arrayr, padw, pmode, alpha=alpha )
    Dy_pad = np.copy(ar_pad)
    
    if method == 'diff':
        for i in range(1, n + 1):
            md = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
            Dy_pad = utl.sp.ndimage.convolve(Dy_pad, md, mode='nearest') / ( 2 * sy )  
        Dy = utl.crop_pad( Dy_pad, original_shape_indx )   
            
    if method == 'fft':
        _,fft2,_,Ky,_,_ = fft2d( Dy_pad, sy=sy, sx=sy )
        Dy_pad = np.real( np.fft.ifft2( ( ( 1j * Ky )**n ) * fft2 ) )     
        Dy = utl.crop_pad( Dy_pad, original_shape_indx ) 
    
    if nodata is True :
        Dy[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        Dy[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        utl.plta(arraym, sbplt=[1, 2, 1], tit='original')
        utl.plta(Dy, sbplt=[1, 2, 2], tit='Y derivative')
        
    return Dy

# -----------------------------------------------------------------------------
def THD( array, padw=0, pmode='edge', sx=1.0, sy=1.0, remove='mean', order=[1], 
         n=1, alpha=0, nodata=True, mask=None, plot=False, vmin=None, vmax=None,
         method='fft', filt=False, ftype='hanning', iter=1, radius=1 ) :
    
    nan = np.isnan( array )
    
    array = np.copy(array)
    
    if remove is None: array = array
    if remove == 'mean': array = array - np.nanmean( array )        
    if remove == 'trend': array = polyfit2d( array=array, order=order )[0]        
    
    ar_pad, original_shape_indx = utl.pad_array( array, padw, pmode, alpha=alpha )    
    
    Dx = dx( ar_pad, method=method, sx=sx, n=n)
    Dy = dy( ar_pad, method=method, sy=sy, n=n)
    THD = np.sqrt( Dx ** 2 + Dy ** 2 )
    
    THD = utl.crop_pad( THD, original_shape_indx )   
    
    if filt is True :
        THD = utl.filt2d( THD, ftype=ftype, iter=iter, radius=radius )          
    
    if nodata is True :
        THD[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        THD[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plta(arraym, sbplt=[1, 2, 1], tit='original')
        plta(THD, vmin, vmax, sbplt=[1, 2, 2], tit='THD')
        
    return THD

# -----------------------------------------------------------------------------
def EHD( array, padw=0, pmode='edge', sx=1.0, sy=1.0, remove='mean', w=1, order=[1], 
         n=2, alpha=0, nodata=True, mask=None, plot=False, vmin=None, vmax=None,
         method_dxy='fft', filt=False, ftype='hanning', iter=1, radius=1 ):
         
    
    nan = np.isnan( array )
    if remove is None: arrayr = array
    if remove == 'mean': arrayr = array - np.nanmean(array)        
    if remove == 'trend': arrayr = polyfit2d(array=array, order=order)[0]        
    
    K, fft2, _, _, _, _ = fft2d( arrayr, sx, sy )
    U = np.real( np.fft.ifft2( fft2 / K ) )
    Taylor = np.copy( U )
    
    for i in range(1, n + 1):
        if w == 'fac': 
            w = 1 / np.math.factorial(i)
        Dxn = dx(U, method=method_dxy, padw=padw, alpha=alpha, sx=sx, n=i) * sx ** i * w
        Dyn = dy(U, method=method_dxy, padw=padw, alpha=alpha, sy=sy, n=i) * sy ** i * w
        Taylor = np.sqrt(Dxn ** 2 + Dyn ** 2) + Taylor

    Taylor_dx = dx(Taylor, padw=padw, alpha=alpha, sx=sx, n=2)
    Taylor_dy = dy(Taylor, padw=padw, alpha=alpha, sy=sy, n=2)
    EHD = np.sqrt(Taylor_dx ** 2 + Taylor_dy ** 2)
    
    if filt is True :
        EHD = utl.filt2d( EHD, ftype=ftype, iter=iter, radius=radius )         
    
    if nodata is True :
        EHD[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        EHD[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plta(arraym, sbplt=[1, 2, 1], tit='original')
        plta(EHD, vmin, vmax, sbplt=[1, 2, 2], tit='EHD')
        
    return EHD

# -----------------------------------------------------------------------------
def phi(array, method='fft', padw=0, pmode='linear_ramp', sx=1.0, sy=1.0, alpha=0, 
        remove='mean', order=[1], nodata=True, mask=None, radius=1, method_dxy='fft',
        method_dz='fft', plot=False, vmin=None, vmax=None, filt=False,
        ftype='hanning', iter=1 ):
    
    nan = np.isnan( array )
    if remove is None: arrayr = array
    if remove == 'mean': arrayr = array - np.nanmean(array)        
    if remove == 'trend': arrayr = polyfit2d(array=array, order=order)[0]        
    
    Dx = dx(arrayr, method=method_dxy, sx=sx, padw=padw, pmode=pmode, alpha=alpha)
    Dy = dy(arrayr, method=method_dxy, sy=sy, padw=padw, pmode=pmode, alpha=alpha)
    Dz = dz(arrayr, method=method_dz, sx=sx, sy=sy, padw=padw, pmode=pmode, alpha=alpha)
    p = np.sqrt(np.arctan( Dx / Dz ) ** 2 + np.arctan( Dy / Dz ) ** 2 )
    
    if filt is True :
        p = utl.filt2d( p, ftype=ftype, iter=iter, radius=radius )      
    
    if nodata is True :
        p[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        p[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plta(arraym, sbplt=[1, 2, 1], tit='original')
        plta(p, vmin, vmax, sbplt=[1, 2, 2], tit='Phase')
    return p

# -----------------------------------------------------------------------------
def tilt( array, padw=0, pmode='gdal', sx=1.0, sy=1.0, alpha=0, method_dz='fft',
          remove='mean', order=[1], nodata=True, mask=None, method_dxy='fft', radius=1, 
          plot=False, vmin=None, vmax=None, filt=False, ftype='hanning', iter=1 ):
    
    array = np.copy( array )
    nan = np.isnan( array )
    if remove == 'mean': 
        array = array - np.nanmean( array )        
    if remove == 'trend': 
        array = polyfit2d( array=array, order=order )[0]        
    
    ar_pad, original_shape_indx = utl.pad_array( array, padw, mode=pmode, alpha=alpha,
                                                constant_values=0)
    
    Dx = dx( ar_pad, method=method_dxy, sx=sx )
    Dy = dy( ar_pad, method=method_dxy, sy=sy )
    Dz = dz( ar_pad, method=method_dz, sx=sx, sy=sy )
    t = np.arctan2( Dz , np.sqrt( Dx**2 + Dy**2 ) )
    t = utl.crop_pad(t, original_shape_indx)
    
    if filt is True :
        t = utl.filt2d( t, ftype=ftype, iter=iter, radius=radius )    
        
    if nodata is True :
        t[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        t[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        utl.plta(arraym, sbplt=[1, 2, 1], tit='original')
        utl.plta(t, vmin, vmax, sbplt=[1, 2, 2], tit='Tilt')
        
    return t

# -----------------------------------------------------------------------------
def theta( array,  padw=0, pmode='gdal', sx=1.0, sy=1.0, alpha=0, method_dz='fft',
           remove='mean', nodata=True, order=[1], mask=None, method_dxy='fft',
           plot=False, vmin=None, vmax=None, filt=False, ftype='hanning', iter=1,
           radius=1 ):
    
    array = np.copy( array )
    nan = np.isnan( array )
    if remove == 'mean': 
        array = array - np.nanmean(array)        
    if remove == 'trend': 
        array = polyfit2d(array=array, order=order)[0]        
    
    ar_pad, original_shape_indx = utl.pad_array( array, padw, pmode, alpha=alpha,
                                        constant_values=0)
    
    Dx = dx( ar_pad, method=method_dxy, sx=sx )
    Dy = dy( ar_pad, method=method_dxy, sy=sy )
    Dz = dz(ar_pad, method=method_dz, sx=sx, sy=sy )
    theta = np.cos(np.arctan(Dz / np.sqrt(Dx ** 2 + Dy ** 2)))
    theta = utl.crop_pad(theta, original_shape_indx)

    if filt is True :
        theta = utl.filt2d( theta, ftype=ftype, iter=iter, radius=radius )

    if nodata is True :
        theta[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        theta[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        utl.plta(arraym, sbplt=[1, 2, 1], tit='original')
        utl.plta(theta, vmin, vmax, sbplt=[1, 2, 2], tit='Theta')
    return theta

# -----------------------------------------------------------------------------
def TDX( array, padw=0, pmode='edge', sx=1.0, sy=1.0, alpha=None, method_dz='fft',  
         remove='mean', order=[1], nodata=True, mask=None, method_dxy='fft',
         plot=False, vmin=None, vmax=None, filt=False, ftype='hanning', iter=1,
         radius=1 ):
    
    array = np.copy( array )
    nan = np.isnan( array )
    if remove == 'mean': 
        array = array - np.nanmean(array)        
    if remove == 'trend': 
        array = polyfit2d(array=array, order=order)[0]  
      
    ar_pad, original_shape_indx = utl.pad_array( array, padw, pmode, alpha=alpha,
                                        constant_values=0)   
    
    Dx = dx( ar_pad, method=method_dxy, sx=sx )
    Dy = dy( ar_pad, method=method_dxy, sy=sy )
    Dz = dz(ar_pad, method=method_dz, sx=sx, sy=sy )
    tdx = np.arctan(np.sqrt(Dx ** 2 + Dy ** 2) / Dz)
    tdx = utl.crop_pad( tdx, original_shape_indx )
    
    if filt is True :
        tdx = utl.filt2d( tdx, ftype=ftype, iter=iter, radius=radius )    
    
    if nodata is True :
        tdx[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        tdx[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plta(arraym, sbplt=[1, 2, 1], tit='original')
        plta(tdx, vmin, vmax, sbplt=[1, 2, 2], tit='TDX')
        
    return tdx

# -----------------------------------------------------------------------------
def Kxy( array, padw=0, pmode='edge', sx=1.0, sy=1.0, alpha=0, method_dz='fft',
         remove='mean', nodata=True, order=[1], mask=None, method_dxy='fft',
         plot=False, vmin=None, vmax=None, filt=False, ftype='hanning', iter=1,
         radius=1  ):
    
    nan = np.isnan( array )
    if remove is None: arrayr = array
    if remove == 'mean': arrayr = array - np.nanmean(array)        
    if remove == 'trend': arrayr = polyfit2d(array=array, order=order)[0]        
    
#    T = tilt( arrayr, padw=padw, pmode=pmode, alpha=alpha, sx=sx,sy=sy, 
#              method_dz=method_dz, method_dxy=method_dxy, )
#    Tx = dx( T, method=method_dxy, padw=padw, pmode=pmode, alpha=alpha, sx=sx )
#    Ty = dy( T, method=method_dxy, padw=padw, pmode=pmode, alpha=alpha, sy=sy )
    Dx = dx( arrayr, method=method_dxy, padw=padw, pmode=pmode, alpha=alpha, sx=sx )
    Dy = dy( arrayr, method=method_dxy, padw=padw, pmode=pmode, alpha=alpha, sy=sy )    
    Dz = dz( arrayr, method=method_dz, sx=sx, sy=sy, padw=padw, pmode=pmode, alpha=alpha)
    Dzx = dx( Dz, method=method_dxy, padw=padw, pmode=pmode, alpha=alpha, sx=sx )
    Dzy = dy( Dz, method=method_dxy, padw=padw, pmode=pmode, alpha=alpha, sy=sy )
    Dzz = dz( Dz, method=method_dz, sx=sx, sy=sy, padw=padw, pmode=pmode, alpha=alpha)
    
#    kxy = np.sqrt(Tx ** 2 + Ty ** 2)
    kxy = ( 1 / ( Dx**2 + Dy**2 + Dz**2 ) ) * ( Dx*Dzx + Dy*Dzy + Dz*Dzz )
    
    if filt is True :
        kxy = utl.filt2d( kxy, ftype=ftype, iter=iter, radius=radius )    
    
    if nodata is True :
        kxy[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        kxy[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plta(arraym, sbplt=[1, 2, 1], tit='Original')
        plta(kxy, vmin, vmax, sbplt=[1, 2, 2], tit='Wavenumber K')
        
    return kxy

# -----------------------------------------------------------------------------
def laplace( array, padw=0, sx=1, sy=1, pmode='linear_ramp', remove='mean', order=[1], 
             alpha=0, mask=None, nodata=True, plot=False, vmin=None, vmax=None, 
             filt=False, ftype='hanning', iter=1, radius=1 ):
         
    
    nan = np.isnan( array )
    if remove is None: arrayr = array
    if remove == 'mean': arrayr = array - np.nanmean(array)        
    if remove == 'trend': arrayr = polyfit2d(array=array, order=order)[0]        
    
    ar_pad, pad_shape = utl.pad_array(arrayr, padw, pmode, alpha=alpha)
    md = np.array( [ [0, 1, 0], [1, -4, 1], [0, 1, 0] ] )
    pad_laplace = signal.convolve2d(ar_pad, md, mode='same')
    lap = utl.crop_pad(pad_laplace, pad_shape)
    
    if filt is True :
        lap = utl.filt2d( lap, ftype=ftype, iter=iter, radius=radius )        
    
    if nodata is True :
        lap[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        lap[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plt.figure()
        plta(arraym, sbplt=[1, 2, 1], tit='original')
        plta(lap, vmin, vmax, sbplt=[1, 2, 2], tit='Laplace filt')
        
    return lap

# -----------------------------------------------------------------------------
def dz( array, method='fft', sx=1.0, sy=1.0, remove='mean', order=[1], padw=0,
        nodata=True, pmode='edge', alpha=0, mask=None, nanfill='gdal', 
        plot=False, vmin=None, vmax=None, filt=False, ftype='hanning', iter=1,
        radius=1, factor=2, ptype='percentage' ):
    
    nan = np.isnan( array )
    
    if remove is None: 
        arrayr = array
    if remove == 'mean': 
        arrayr = array - np.nanmean( array )     
    if remove == 'trend': 
        arrayr = polyfit2d( array=array, order=order )[0]  
    if nanfill is not None: 
        arrayr = utl.fillnan( arrayr, method=nanfill )

    if method == 'fft':
        ar_pad, pad_shape = utl.pad_array( arrayr, padw, pmode, alpha=alpha, ptype=ptype )
        K, fft2, _, _, _, _ = fft2d( ar_pad, sx, sy )
        Dz_pad = np.real(np.fft.ifft2( fft2 * K ) )
        Dz = utl.crop_pad(Dz_pad, pad_shape)
        
    if method == 'isvd':
        ar_pad, pad_shape = utl.pad_array(arrayr, padw, pmode, alpha=alpha, ptype=ptype )
        U = vi( ar_pad, sx=sx, sy=sy, padw=padw )
        Dxx = dx( U, n=2, sx=sx, method='diff' )
        Dyy = dy( U, n=2, sy=sy, method='diff' )
        Dz_pad = -( Dxx + Dyy )
        Dz = utl.crop_pad(Dz_pad, pad_shape)
        
    if filt is True :
        Dz = utl.filt2d( Dz, ftype=ftype, iter=iter, radius=radius, factor=factor )
        
    if nodata is True :
        Dz[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        Dz[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        utl.plta(arraym, sbplt=[1, 2, 1], tit='original')
        utl.plta(Dz, vmin, vmax, sbplt=[1, 2, 2], tit='Vertical gradient')
        
    return Dz

# -----------------------------------------------------------------------------
def vi( array, sx=1.0, sy=1.0, remove=None, order=[1], padw=0, nanfill='gdal',
        pmode='gdal', nodata=True, alpha=0, mask=None, plot=False, 
        vmin=None, vmax=None ):

    nan = np.isnan( array )
    
    if remove is None: 
        arrayr = array
    if remove == 'mean': 
        arrayr = array - np.nanmean(array)        
    if remove == 'trend': 
        arrayr = polyfit2d(array=array, order=order)[0]        
    if nanfill is not None: 
        arrayr = utl.fillnan(arrayr, method=nanfill)
        
    ar_pad, pad_shape = utl.pad_array( arrayr, padw, pmode, alpha=alpha )
    
    K, fft2, _, _, _, _ = fft2d( ar_pad, sx, sy )
    
    K[0,0] = min( ( K[0,1], K[1,1], K[1,0] ) ) / 2
    
    V_pad = np.real( np.fft.ifft2( fft2 / K ) )
    V = utl.crop_pad( V_pad, pad_shape )
    
    if nodata is True :
        V[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        V[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plta(arraym, sbplt=[1, 2, 1], tit='original')
        plta(V, vmin, vmax, sbplt=[1, 2, 2], tit='Vertical Integration')
        
    return V

# -----------------------------------------------------------------------------
def texture( array, method='std', padw=0, pmode='edge',
             remove=None, order=[1], nanfill='gdal', radius=2, iter=1,
             nodata=True, alpha=0, mask=None, plot=False, vmin=None, vmax=None) :
    
    nan = np.isnan( array )
    
    if remove is None: 
        arrayr = array
    if remove == 'mean': 
        arrayr = array - np.nanmean(array)        
    if remove == 'trend': 
        arrayr = polyfit2d(array=array, order=order)[0]        
        
    ar_pad, pad_shape = utl.pad_array(arrayr, padw, pmode, alpha=alpha)    

    if method == 'std' :
        win_shape = radius * 2 + 1, radius * 2 + 1 
        for i in range( iter ) :
            windows = utl.rolling_win_2d( ar_pad, win_shape )
            ar_pad = np.nanstd( windows, axis=1 ).reshape( ar_pad.shape )
        
    ar_txt = utl.crop_pad( ar_pad, pad_shape )   
    
    if nodata is True :
        ar_txt[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        ar_txt[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plta(arraym, sbplt=[1, 2, 1], tit='Original')
        plta(ar_txt, vmin, vmax, sbplt=[1, 2, 2], tit='Texture')
        
    return ar_txt
    
#------------------------------------------------------------------------------
def canny( array, xy=None, ltr=0.05, htr=0.09, 
           weak=25, strong=255, r=1, npoints=2, poly_deg=1, filt=False, sigma=2, 
           plot=False, color='r-', plot_steps=False):
    
    np.warnings.filterwarnings('ignore')
    if filt == True: 
        array = scy.ndimage.gaussian_filter(array, sigma=sigma)
    if xy == None:
        nx, ny = array.shape[1], array.shape[0]
        xi, yi = np.arange(0, nx), np.arange(0, ny)
        xx, yy = np.meshgrid(xi, yi)
    else: 
        xx, yy = xy[0], xy[1]
    if plot_steps == True: plta(array, sbplt=(1, 4, 1), tit='Original')

    def non_max_suppression(array, plot=False, vmin=None, vmax=None, sbplt=[1, 4, 2]):
        
        M, N = array.shape
        Z = np.zeros((M, N))
        rx = scy.ndimage.sobel(array, axis=0, mode='constant')
        ry = scy.ndimage.sobel(array, axis=1, mode='constant')
        sobel = np.hypot(rx, ry)
        angle = sobel * 180.0 / np.pi
        angle[(angle < 0)] += 180
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q = 255
                r = 255
                if not 0 <= angle[(i, j)] < 22.5:
                    if 157.5 <= angle[(i, j)] <= 180:
                        q = array[(i, j + 1)]
                        r = array[(i, j - 1)]
                if 22.5 <= angle[(i, j)] < 67.5:
                    q = array[(i + 1, j - 1)]
                    r = array[(i - 1, j + 1)]
                else:
                    if 67.5 <= angle[(i, j)] < 112.5:
                        q = array[(i + 1, j)]
                        r = array[(i - 1, j)]
                    else:
                        if 112.5 <= angle[(i, j)] < 157.5:
                            q = array[(i - 1, j - 1)]
                            r = array[(i + 1, j + 1)]
                if array[(i, j)] >= q and array[(i, j)] >= r:
                    Z[(i, j)] = array[(i, j)]
                else:
                    Z[(i, j)] = 0

        if plot == True:
            plta(Z, vmin, vmax, sbplt=sbplt, tit='NoMaxSuppression', cmap='Greys')
        return Z

    def threshold( array, ltr=ltr, htr=htr, plot=plot_steps, vmin=None, vmax=None,
                    sbplt=[1, 4, 3], weak=weak, strong=strong ):
        
        highThreshold = array.max() * htr
        lowThreshold = highThreshold * ltr
        M, N = array.shape
        res = np.zeros((M, N), dtype=(np.int32))
        weak = np.int32(weak)
        strong = np.int32(strong)
        strong_i, strong_j = np.where( array >= highThreshold )
        zeros_i, zeros_j = np.where( array < lowThreshold)
        weak_i, weak_j = np.where( ( array <= highThreshold ) & ( array >= lowThreshold ) )
        res[ ( strong_i, strong_j ) ] = strong
        res[ ( weak_i, weak_j ) ] = weak
        if plot == True:
            plta(res, vmin=(np.min(res)), vmax=(np.max(res)), sbplt=sbplt, tit='Threshold', cmap='Greys')
        return res

    def hysteresis(img, weak=weak, strong=strong, plot=False, vmin=None, vmax=None, sbplt=[1, 4, 4], cmap='Greys'):
        M, N = img.shape  
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e :
                        pass
        if plot==True: 
            utl.plta(img, vmin, vmax, tit='Hysteresis', sbplt=sbplt)
        return img

    def intersect(a, b):
        boolean = False
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                boolean = np.all(a[i, :] == b[j, :]) or boolean
        return boolean

    array = non_max_suppression(array, plot=plot_steps )
    array = threshold(array, plot=plot_steps, htr=htr, ltr=ltr, weak=weak, strong=strong )    
    array = hysteresis( array , weak, strong, plot=plot_steps )

    px = array >= strong
    xp = xx[ px ] 
    yp = yy[ px ]
    
    if plot == True :
        plt.figure()
        plt.scatter( xp, yp, c='k' )

    return xp, yp     

# -----------------------------------------------------------------------------
def parker( array, 
            drho, 
            n=5, 
            sx=1.0, 
            sy=1.0, 
            padw=0, 
            pmode='gdal', 
            mask=None, 
            nodata=True, 
            plot=False, 
            vmin=None, 
            vmax=None,
            obs_lev=None,
            ref_lev=None,
            neg2pos=False ) :
    """
    @author: L.S.Zampa 
    
    Rapid Calculation of Potential Anomalies using the 2D Fast Fourier Transform (FFT).
    NB. This computation is valid only on 2D-surfaces ("gridded layers")
    
    Ref.
    Parker 1972, https://doi.org/10.1111/j.1365-246X.1973.tb06513.x
    
    Parameters
    ----------
    array : NUMPY 2D-ARRAY
        Input 2D grid-interface in meters
    drho : FLOAT or INT
        Density difference in kg/m^3, between the matirial above the interface and below it.
        i.e., drho = rho_above - rho_below
    n : INT, optional
        Order of approximation of the Taylor series.
    sx : FLOAT or INT, optional
        grid-step in x direction
    sy : FLOAT or INT, optional
        grid-step in y direction
    padw : FLOAT or INT, optional
        percentage used to extend the grid, i.e. grid-pad (this may prevent borders artefacts to fill in) 
    pmode : STR, optional
        Type of alghorithm used for padding (see numpy.pad documentation). The default is 'edge'.
        Also fillnan function 
    mask : NUMPY 2D-ARRAY BOOLEAN, optional
        2D Boolean array used to mask undesiderd areas. The default is None.
    plot : BOOLEAN, optional
        If True, the result is shown in a Matplotlib figure. The default is False.
    vmin : TYPE, optional
        Min value of the colorscale in the output figure. The default is None.
    vmax : TYPE, optional
        Max value of the colorscale in the output figure. The default is None.

    Returns
    -------
    grav : NUMPY 2D-ARRAY
        Output 2D grid with the gravity effect generated by the input interface in mGal.
        
    """
    
    h_array = np.copy( array ).astype(float)
    nan = np.isnan( h_array )
        
    if neg2pos == True :
        h_array = h_array * -1    

    if ref_lev is None :
        ref_lev = np.nanmean( h_array )

    if obs_lev is None :
        obs_lev = np.nanmean( h_array )

    z0 = np.abs( np.abs(obs_lev) - np.abs( ref_lev ) )

    h_array = h_array - ref_lev

    ar_pad, original_shape_indx = utl.pad_array( h_array, padw, pmode )

    taylor_fft2 = np.zeros( ar_pad.shape )
    
    for ip in range(1, n + 1):
        pwr_array = ar_pad ** ip
        K, fft2, _, _, _, _ = fft2d( pwr_array, sx=sx, sy=sy )
        taylor_fft2 = taylor_fft2 +  ( K ** ( ip - 1 ) / np.math.factorial( ip ) ) * fft2

    fft_grav = -2 * np.pi * G * drho * np.exp( -K * z0 ) * taylor_fft2
    grav_pad = np.real( np.fft.ifft2( fft_grav ) ) * 1e5
    grav = utl.crop_pad( grav_pad, original_shape_indx )

    if nodata is True :
        grav[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        grav[mask] = np.nan
        arraym = np.copy( array )
        arraym[mask] = np.nan
        
    if plot == True:
        utl.plta( arraym, sbplt=[1, 2, 1], tit='discontinuity [m]')
        utl.plta( grav, vmin, vmax, sbplt=[1, 2, 2], tit='grav_effect [mGal]')
        
    return grav

# -----------------------------------------------------------------------------
def polyfit2d( array, order=[1], mask=None, sx=1.0, sy=1.0, lim=None, 
               padw=0, pmode='gdal', alpha=0, nanfill=None, nodata=True,
               plot=False, vmin=None, vmax=None ):
    
    """
    Polynomial fitting of 2D surfaces using least square method 
    """
    
    nans = np.isnan( array )
    
    if nanfill is not  None:
        array = utl.fillnan(array, method=nanfill )

    if padw != 0:
        array, original_shape_indx = utl.pad_array(array, padw, pmode )
        
    nx, ny = array.shape[1], array.shape[0]
    zz = np.copy(array)
    
    if lim is None:
        xi, yi = np.arange(0, nx * sx, sx), np.arange(0, ny * sy, sy)
        xx, yy = np.meshgrid(xi, yi)
        
    x, y, z = xx.ravel(), yy.ravel(), zz.ravel()
    nan = np.isnan(z)
    xn, yn, zn = x[(~nan)], y[(~nan)], z[(~nan)]
    xn, yn = (xn - np.mean(xn)) / np.std(xn), (yn - np.mean(yn)) / np.std(yn)
    
    if len(order) == 1:
        ox = order[0]
        oy = order[0]
        pl_order = order[0]
    if len(order) == 2:
        ox = order[0]
        oy = order[1]
        pl_order = None
    if len(order) == 3:
        ox = order[0]
        oy = order[1]
        pl_order = order[2]
        
    coeffs = np.ones((ox + 1, oy + 1))
    
    a = np.zeros((coeffs.size, xn.size))
    
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        
        if pl_order is not None and i + j > pl_order:
            arr = np.zeros_like(xn)
        else:
            arr = xn ** i * yn ** j
        a[index] = arr.flatten()

    M = np.linalg.lstsq( (a.T), zn, rcond=None )[0]
    pn = np.zeros( len( xn ) )
    
    for m, (j, i) in zip( M, np.ndindex(coeffs.shape) ):
        pn += m * xn ** i * yn ** j

    p = z
    p[~nan] = pn
    pp = p.reshape(zz.shape)
    rr = array - pp
    
    if padw != 0: 
        rr = utl.crop_pad(rr, original_shape_indx)
        pp = utl.crop_pad(pp, original_shape_indx)
        array = utl.crop_pad(array, original_shape_indx)
    
    if nodata is True :
        rr[nans] = np.nan
        pp[nans] = np.nan 
        array[nans] = np.nan  
    
    if mask is not None:
        rr[mask] = np.nan
        array[mask] = np.nan
        pp[mask] = np.nan
    
    rr = rr - np.nanmean( rr )
    
    if plot == True:
        utl.plta( array, sbplt=[1, 3, 1], tit='Original' )
        utl.plta( pp, sbplt=[1, 3, 2], tit='Trend', new_fig=False )
        utl.plta( rr, sbplt=[1, 3, 3], tit='Residual', vmin=vmin, vmax=vmax, new_fig=False )
    param = {'prm':M, 'des_mtx':a}
    
    return rr, pp, param 

# -----------------------------------------------------------------------------
def fft2d( array, sx=1.0, sy=1.0, 
           remove=None, order=1 ):
   
    pny, pnx = array.shape

    kx = 2 * np.pi * np.fft.fftfreq( pnx, sx ) # wave-number in x direction
    ky = 2 * np.pi * np.fft.fftfreq( pny, sy ) # wave-number in y direction
    
    if remove is None: 
        arrayr = array
    if remove == 'mean': 
        arrayr = array - np.nanmean( array ) 
    if remove == 'trend': 
        arrayr = polyfit2d( array=array, order=[order] )[0]
         
    KX, KY = np.meshgrid( kx, ky )
    K = np.sqrt( KX ** 2 + KY ** 2 ) # 2D-WaveNumber array 
    
    fft2 = np.fft.fft2( arrayr ) # 2D-Amplitude spectrum
    PDS = np.abs( fft2 ) ** 2 # 2D-Power density spectrum
    SPDS = np.fft.fftshift( PDS ) # 2D-Shifted-Power density spectrum
    
    return K, fft2, KX, KY, PDS, SPDS

# -----------------------------------------------------------------------------
def radial_fft( array, 
                sx=1, sy=1, 
                plot=False, 
                units='[units]', 
                remove='mean',
                padw=0, 
                pmode='linear_ramp', 
                fillnan=None, 
                order=[1], 
                alpha=None,
                maxSearchDist=None, 
                iter=None, 
                smooth=0, 
                tension=0.35,
                normalize=False,
                fit_line=True,
                **kwargs ) :

    if remove is None: 
        array = array
    if remove == 'mean': 
        array = array - np.nanmean( array ) 
    if remove == 'trend': 
        array = polyfit2d( array=array, order=order )[0]

    if fillnan is not None: 
        if fillnan is True :
            fillnan = 'gdal'
        array = utl.fillnan( array, method=fillnan, smooth=smooth,
                maxSearchDist=maxSearchDist, iter=iter, 
                edges=True, tension=tension )
    
    if padw != 0:
        ar_pad, _= utl.pad_array( array, 
                                padw, pmode, 
                                alpha=alpha, 
                                sqr_area=False, 
                                equal_shape=True)
    else:
        ar_pad = array

    gfft = fft2d( ar_pad, sx, sy )
    
    k = gfft[0]
    kx = gfft[2]
    ky = gfft[3]
    PDS = np.log( gfft[4] )
    SPDS = gfft[5]

    L_x = ar_pad.shape[1] * sx  # Dimensione della griglia in metri
    L_y = ar_pad.shape[0] * sy  # Dimensione della griglia in metri
    L_min = np.min( ( L_x, L_y ) )

    max_radius = min( kx.max(), ky.max() )
    ring_width = min(np.min(kx[kx > 0]), np.min(ky[ky > 0]))

    PDS_radial = []
    k_radial = []
    radius_i = 0
    
    while True:

        if radius_i == 0:
            inside = k <= 0.5 * ring_width
        else:
            inside = np.logical_and( k > (radius_i - 0.5) * ring_width, 
                                     k <= (radius_i + 0.5) * ring_width  )

        PDS_radial.append( PDS[inside].mean() )
        k_radial.append( radius_i * ring_width )

        radius_i += 1

        if radius_i * ring_width > max_radius:
            break

    for istart in range( len( PDS_radial ) ):
        if PDS_radial[istart+1] >= PDS_radial[istart] :
            continue
        else:
            break

    k_radial = np.array( k_radial[istart:] ) / (2*np.pi)
    PDS_radial = np.array( PDS_radial[istart:] )

    k_radial0 = k_radial == 0.0

    if  k_radial0.sum() > 0:
        k_radial = np.delete( k_radial, np.where( k_radial0 ) )
        PDS_radial = np.delete( PDS_radial, np.where( k_radial0 ) )

    if normalize == True:
        PDS_radial = PDS_radial / PDS_radial.max()

    if plot == True:

        if 'round' in kwargs:
            nround = kwargs['round']
            kwargs.pop('round')
        else :
            nround = 2

        if 'place_nodes' in kwargs and kwargs['place_nodes'] :
            for i, ni in enumerate( kwargs['place_nodes'][1] ):
                kwargs['place_nodes'][1][i] = 1 / kwargs['place_nodes'][1][i]

        if fit_line == True:
            segments = utl.fit_segmented_line( 
                k_radial, PDS_radial, plot=False, **kwargs )

        plt.scatter( k_radial, PDS_radial, color='k', 
                     marker='o', facecolors='none')

        depths = []
        ymin = plt.gca().get_ylim()[0]
        ymax = PDS_radial.max()

        if 'segments' in locals():
            for i, seg in enumerate( segments ):
                if i > 0:
                    seg0 = segments[i-1]
                    slope = ( seg0[1] - seg[1]) / (seg0[0] - seg[0] )
                    depth = np.round( np.abs( slope / 2 ), nround )
                    if nround <= 0 :
                        depth = int( depth )
                    depths.append( depth )
                    plt.plot( [seg0[0], seg[0]], [seg0[1], seg[1]], 
                            label=f"Depth: { depths[-1] }" )
                    if i < len(segments) - 1:
                        plt.vlines( seg[0], ymin, seg[1], 
                                    linestyles='dashed')
                        ptext = np.round( 1/seg[0], nround )
                        if nround <= 0 :
                            ptext = int( ptext )
                        plt.text( seg[0], ymin, 
                                f'{ptext}', 
                                ha='left', va='bottom' )

        # Label the axes
        plt.ylabel('ln( PSD )')

        # Get current x-ticks
        xticks = plt.gca().get_xticks()

        # Convert x-ticks from wavenumbers to wavelengths
        new_xticklabels = [ f"{ np.round( 1/ xtick, nround ) }" 
            if xtick != 0 else "∞" for xtick in xticks ]

        # Set new x-tick labels with wavelengths
        plt.gca().set_xticklabels( new_xticklabels )

        # Update the x-axis label to reflect wavelengths
        plt.xlabel( f'Wavelength {units}' )

        # Add a legend
        plt.legend()

        plt.tight_layout()

    return PDS_radial, k_radial, k

# -----------------------------------------------------------------------------
def btw_filt( array, lowcut=None, highcut=None, sx=1, sy=1, order=6, remove='mean', 
             fillnan=None, iter=1, maxSearchDist=None, padw=0, pmode='symmetric', 
             plot=False, vmin=None, vmax=None, nodata=True, mask=None, maxv=None,
             filt=False, ftype='hanning', radius=1, alpha=None, smooth=0, torder=1 ) :
    
    nan = np.isnan( array )
    arraym = np.copy( array )
    
    if remove == 'mean': array = array - np.nanmean( array ) 
    if remove == 'trend': array = polyfit2d( array=array, order=torder )[0]    

    if fillnan is not None:       
        array = utl.fillnan( array, method=fillnan, smooth=0,
                             maxSearchDist=maxSearchDist, iter=iter, edges=True )
    
    p1, r1,_ = radial_fft( array, sx=sx, sy=sy, plot=False, fillnan=fillnan, 
                         maxSearchDist=maxSearchDist, iter=iter,
                         padw=padw, pmode=pmode, alpha=alpha, smooth=smooth ) 

    ar_pad, original_shape_indx = utl.pad_array( array, padw, pmode, alpha=alpha )    

    fs = np.mean( ( 1/sx, 1/sy ) )
    nyq = 0.5 * fs
    
    if highcut is None : 
        b, a = utl.sp.signal.butter( order, lowcut/nyq, btype='highpass')
    elif lowcut is None :    
        b, a = utl.sp.signal.butter( order, highcut/nyq, btype='lowpass' )
    else :
        b, a = utl.sp.signal.butter( order, [ lowcut/nyq, highcut/nyq ], btype='band', analog=False )
        
    ar_pad_filt = utl.sp.signal.filtfilt( b, a, ar_pad )

    if filt is True :
        ar_pad_filt = utl.filt2d( ar_pad_filt, ftype=ftype, iter=iter, radius=radius )    

    p3, r3 = radial_fft( ar_pad_filt, dx=sx, dy=sy, plot=False )
    ar_filt = utl.crop_pad( ar_pad_filt, original_shape_indx )

    if nodata is True :
        ar_filt[nan] = np.nan
    
    if mask is not None:
        arraym[mask] = np.nan  
        ar_filt[mask] = np.nan
       
    r2, p2 = utl.sp.signal.freqz( b, a, worN=2000 )
    
    if plot is True : 
        plt.figure()
        plt.subplot( 2, 1, 1 )
        p1, p3 = np.log( p1 ), np.log( p3 ) 
        plt.plot( r1, p1, c='b' )
        plt.plot( r3, p3, c='g' )
        if maxv is None :
            maxv = p3.max()
        p2 = utl.normalize( abs(p2), b=maxv, a=p3.min() )
        r2 = (nyq / np.pi) * r2
        plt.plot( r2, p2, c='r', label="order = %d" % order )
        utl.plta( arraym, sbplt=[2,2,3], vmin=vmin, vmax=vmax )
        utl.plta( ar_filt, sbplt=[2,2,4], vmin=vmin, vmax=vmax )
    
    return [ ar_filt, ( r1, p1 ), ( r2, p2 ), ( r3, p3 ) ] 
    
# -----------------------------------------------------------------------------
def XYZ_crop2lim( X, Y, Z, lim ) :
    
    if len(Z.shape) == 2:
        
        xi = np.where((X[0, :] >= lim[0]) & (X[0, :] <= lim[1]))
        yi = np.where((Y[:, 0] >= lim[2]) & (Y[:, 0] <= lim[3]))
        
        Xc = X[np.min(yi):np.max(yi), np.min(xi):np.max(xi)]
        Yc = Y[np.min(yi):np.max(yi), np.min(xi):np.max(xi)]
        Zc = Z[np.min(yi):np.max(yi), np.min(xi):np.max(xi)]
        
        return (Xc, Yc, Zc)
    
    if len(Z.shape) == 1:
        
        xi = np.where((X >= lim[0]) & (X <= lim[1]))
        yi = np.where((Y >= lim[2]) & (Y <= lim[3]))
        
        xc = X[(xi, yi)]
        yc = Y[(xi, yi)]
        zc = Z[(xi, yi)]
        
        return (xc, yc, zc)

# -----------------------------------------------------------------------------
def xyz2grid( x, y, z, lim=None, extend=None, extend_method='percentage', 
              sqr_area=False, gstep=None, blkm=None, method='linear', prjcode_in=4326,
              prjcode_out=4326, filt=None, filt_radius=1, msk_radius=None, msk_shp=None,
              in_out='in', fill_nan=None, plot=False, vmin=None, vmax=None,
              pltxy=False, s=None, msk_arr=None, adjst_lim=False, iter=1, tension=0.35,
              filt_factor=2, filt_sigma=1, padw=0, pmode='gdal' ):
              
   
    if prjcode_in != prjcode_out:
        x, y = utl.prjxy( prjcode_in, prjcode_out, x, y )  
    
    if lim is None:
        lim = [np.min(x), np.max(x), np.min(y), np.max(y)]
        xl, yl, zl = x, y, z
    else:
        xl, yl, indx = utl.xy_in_lim( x, y, lim, extend=33 )
        zl = z[indx]
        
    if extend is not None:
        lim = utl.extend_lim(lim, extend, extend_method, sqr_area )

    if gstep == None:
        gstep = utl.min_dist( xl, yl )['mean']
    
    if blkm != None :    
        if blkm == True:
            xl, yl, zl = utl.block_m( xl, yl, zl, gstep, lim=lim )
        if type( blkm ) in ( int, float ) :
            xl, yl, zl = utl.block_m( xl, yl, zl, blkm, lim=lim )
    
    if adjst_lim is False :    
        xg = np.arange( lim[0], lim[1], gstep )
        yg = np.arange( lim[3], lim[2], -gstep )
    if adjst_lim is True :   
        xg = np.linspace( lim[0], lim[1], int( ( lim[1] - lim[0] ) / gstep ) )
        yg = np.linspace( lim[3], lim[2], int( ( lim[3] - lim[2] ) / gstep ) )
    xx, yy = np.meshgrid(xg, yg)
       

    if method == 'surface':
        xx, yy, zz = utl.gmt_surf( xl, yl, zl, lim=lim, gstep=gstep, tension_factor=tension )
    else:
        zz = utl.xyz2xy( ( xl, yl, zl ), (xx, yy), method=method, fillnan=False )

    if fill_nan is not None :
        if ( type(fill_nan) == bool ) and ( fill_nan is True ) :
            fill_nan = 'gdal'
        zz = utl.fillnan( zz, xy=( xx, yy ), method=fill_nan, iter=iter ) 
        
    if filt is not None :
        zz = utl.filt2d( zz, ftype=filt, iter=iter, radius=filt_radius, 
                         factor=filt_factor, sigma=filt_sigma, padw=padw, pmode=pmode )        
    
    if msk_radius is not None:
        zz = utl.mask2D( ( xl, yl, msk_radius ), (xx, yy, zz) )[0]
        
    if msk_shp is not None:
        zz = rt.mask_array( xx, yy, zz, msk_shp, prjcode=prjcode_out )

    if msk_arr is not None:
        zz[np.isnan( msk_arr ) ] = np.nan


    if plot == True:
        if pltxy == False:
            utl.plta( zz, vmin=vmin, vmax=vmax, cmap='rainbow' )
        if pltxy == True:
            limp = utl.xy2lim( xx, yy ) 
            utl.plta( zz, vmin=vmin, vmax=vmax, cmap='rainbow', lim=limp, points=[ xl, yl ] )
            
    return [ xx, yy, zz ], [ x, y, z ]

# -----------------------------------------------------------------------------
def upcont( array, h, sx=1, sy=1, padw=0, pmode='gdal', alpha=None, remove=None,
            nanfill='gdal', order=[1], mask=None, nodata=True, 
            plot=False, vmin=None, vmax=None):
    
    nan = np.isnan( array )
    if remove is None: 
        arrayr = array
    if remove == 'mean': 
        arrayr = array - np.nanmean(array)
    if remove == 'trend': 
        arrayr = polyfit2d(array=array, order=order)[0]
    if nanfill is not None: 
        arrayr = utl.fillnan(arrayr, method=nanfill)
        
    array_pad, pad_shape = utl.pad_array(arrayr, padw, pmode, alpha=alpha)

    K, fft2, _, _, _, _ = fft2d( array_pad, sx, sy )
    up_pad = np.real( np.fft.ifft2( fft2 * np.exp(-h * K) ) )

    up = utl.crop_pad( up_pad, pad_shape )
     
    if nodata is True :
        up[nan] = np.nan

    if mask is None:
        arraym = array
    else:
        up[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    diff = up - array

    if plot == True:
        utl.plta(arraym, vmin, vmax, sbplt=[1, 3, 1], tit='original')
        utl.plta(up, vmin, vmax, sbplt=[1, 3, 2], tit='upward')
        utl.plta(diff, sbplt=[1, 3, 3], tit='diff')
        utl.plt.tight_layout()
        
    return up

# -----------------------------------------------------------------------------
def edge2line( array, xx, yy, deg=30, dist1=2, dist2=1, poly_deg=1, n=3, num_points=2,
               val_th=None, polyfit=True, plot=False, dist_th=None, smooth_edges=None,
               save_raster=False, new_name='new_raster', path='/vsimem/', prjcode=4326, lim=None ):

    ar = np.copy( array ) 
    
    if lim is not None :
        xx, yy, ar = utl.XYZ_crop2lim( xx, yy, ar, lim )
    
    if val_th is None : 
        val_th = np.nanmean( ar ) 
        print('values treshold :', val_th )
    
    sx, sy, _ = utl.stepxy( xx, yy )
    
    smin = np.min( ( sx, sy ) )
    sxn = sx/smin
    syn = sy/smin  
    sd = np.sqrt((sx**2)+(sy**2))
    sdn = sd/smin        
    th = np.arctan(sy/sx)
    GN = np.zeros(ar.shape)
    MD = np.zeros(ar.shape)
    dist1 = sd*dist1
    dist2 = sd*dist2
    
    # -------------------------------------------------------------------------
    # Edge points generation 
    
    for i, j in np.ndindex(ar.shape):
        if i==0 or i==ar.shape[0]-1 or \
           j==0 or j==ar.shape[1]-1:
            continue
        N=0
        gmax = []
        xymax = []
        if ar[i-1, j-1] < ar[i, j] > ar[i+1, j+1]:
            N +=1
            a = 1/2 * ( ar[i-1, j-1] -2 * ar[i, j] + ar[i+1, j+1])
            b = 1/2 * ( ar[i+1, j+1] - ar[i-1, j-1] )
            xmax = -sdn*b/(2*a)
            xymax.append((xx[i-1, j-1]+xmax*smin*np.cos(th),
                          yy[i-1, j-1]-xmax*smin*np.sin(th)))
            gmax.append(a*(xmax**2)+b*xmax+ar[i, j])
        if ar[i-1, j] < ar[i, j] > ar[i+1, j]:
            N +=1
            a = 1/2*(ar[i-1, j] -2*ar[i, j] +ar[i+1, j])
            b = 1/2*(ar[i+1, j] -ar[i-1, j])
            xmax = -syn*b/(2*a)
            xymax.append((xx[i, j], yy[i+1, j]+xmax*smin))
            gmax.append(a*(xmax**2)+b*xmax+ar[i, j])
        if ar[i-1, j+1] < ar[i, j] > ar[i+1, j-1]:
            N +=1
            a = 1/2*(ar[i-1, j+1] -2*ar[i, j] +ar[i+1, j-1])
            b = 1/2*(ar[i+1, j-1] -ar[i-1, j+1])
            xmax = -sdn*b/(2*a)
            xymax.append((xx[i-1, j+1]+xmax*smin*np.cos(th),
                          yy[i-1, j+1]+xmax*smin*np.sin(th)))
            gmax.append(a*(xmax**2)+b*xmax+ar[i, j])
        if ar[i, j-1] < ar[i, j] > ar[i, j+1]:
            N +=1
            a = 1/2*(ar[i, j-1] -2*ar[i, j] +ar[i, j+1])
            b = 1/2*(ar[i, j+1] -ar[i, j-1])
            xmax = sxn*b/(2*a)
            xymax.append((xx[i, j-1]+xmax*smin, yy[i, j]))
            gmax.append(a*(xmax**2)+b*xmax+ar[i, j])
            
        if ( N < n ) or ( np.max( gmax ) < val_th ) : continue  
        
        GN[i,j] = np.max( gmax )
        MD[i,j] = N
        
    idx = GN > 0  
    points =  np.column_stack( ( xx[idx], yy[idx] ) )
#    GN[ GN > 0 ] = 1
    
    # -------------------------------------------------------------------------
    # Clustering scattered points
    
#    if ( dist_th is None ) :
#        dist_th = [ np.mean( ( sx, sy ) ) * 2, np.mean( ( sx, sy ) ) ] 
#    print( 'distance treshold :', dist_th )
#    
#    warnings.simplefilter('ignore', np.RankWarning)    
#    clusters = cluster.hierarchy.fclusterdata( points, dist_th[0], criterion="distance" )
#
#    srt = np.argsort( clusters )
#    points, clusters = points[srt], clusters[srt]
    
    # -------------------------------------------------------------------------
    # Clustering refinement 
    
#    unq = np.unique( clusters )
#    lines_pts = np.empty( ( 0, 3 ) ) 
#    cn = 0
#    for c in unq :
#        idx = clusters == c 
#        if np.sum( idx ) <= 1 : continue
#        xp = points[idx,0]
#        yp = points[idx,1]
#        pts = np.column_stack( ( xp, yp ) )
#        c = cluster.hierarchy.fclusterdata( pts, dist_th[1], criterion="distance" ) 
#        idxn = np.full( c.shape, True )
#        for ci in c :
#           if np.sum( c==ci ) <= 1 :
#               idxn[ c==ci ] = False
#        pts, c = pts[idxn], c[idxn]   
#        c += cn    
#        cn = int( c.max() )
#        lines_pts = np.vstack( ( lines_pts, np.column_stack( ( pts, c ) ) ) )

    # -------------------------------------------------------------------------
    # Lines generation
    
#    unq = np.unique( lines_pts[:,2] )
#    lines = np.zeros( lines_pts.shape )
#    ln = 0
#    for c in unq :
#        ln += 1
#        idx = lines_pts[:,2] == c 
#        ld = lines_pts[ idx ]
#        ld_x_range = ld[:,0].max() - ld[:,0].min()
#        ld_y_range = ld[:,1].max() - ld[:,1].min()
#        if ld_x_range >= ld_y_range :
#            sort = np.argsort( ld[:,0] )
#            ld = ld[ sort ]
#            spl = utl.sp.interpolate.interp1d( ld[:,0], ld[:,1])
##            coef = np.polyfit( ld[:,0], ld[:,1], 5 )
##            poly1d_fn = np.poly1d( coef ) 
#            lines[ idx, 0 ] = ld[:,0]
#            lines[ idx, 1 ] = spl( ld[:,0] ) 
#            lines[ idx, 2 ] = np.full( ld[:,1].shape, ln )
#            
#        if ld_x_range < ld_y_range :
#            sort = np.argsort( ld[:,1] )
#            ld = ld[ sort ] 
#            spl = utl.sp.interpolate.interp1d( ld[:,1], ld[:,0] )            
##            coef = np.polyfit( ld[:,1], ld[:,0], 5 )
##            poly1d_fn = np.poly1d( coef ) 
#            lines[ idx, 0 ] = spl(ld[:,1])
#            lines[ idx, 1 ] = ld[:,1]
#            lines[ idx, 2 ] = np.full( ld[:,1].shape, ln )        
        
    if save_raster is True :
        rt.array2raster( (xx, yy, MD), new_name=new_name, path=path, prjcode=prjcode,
                         eType=rt.gdal.GDT_Byte, options=['NBITS=1'], nodata=0 )
        
    if plot==True:
#        
        utl.plta( MD )
#        ax = plta(ar, lim=(xx, yy), points=(lines_pts[:,0], lines_pts[:,1] ) ) 
#        lin = np.unique( lines[:,2] )    
#        for l in lin :
#            i = lines[:,2] == l 
#            ax.plot( lines[i,0], lines[i,1], c='k' )    

    return MD, GN, points

# -----------------------------------------------------------------------------
def vel2den( vp, method='Gardner' ) :
    """
    Empirical formulas to convert seismic velocities ( Vp, [m/s] ) into densities [kg/m3] 
    
    Ref:
    - Gardner, G, L Gardner & A Gregory, 1974. 
      Formation velocity and density—the diagnostic basis for stratigraphic traps. 
      Geophysics 39, 770–780
    - Ludwig, W. J., J. E. Nafe, and C. L. Drake (1970). Seismic refraction, in
      The Sea, A. E. Maxwell (Editor), Vol. 4, Wiley-Interscience, New
      York, 53–84.
    - Brocher, T., 2005  Empirical Relations between Elastic Wavespeeds and 
      Density in the Earth's Crust. DOI:10.1785/0120050077
    """ 
    
    if type( vp ) in ( list, tuple ) :
        vp = np.array( np.copy( vp ) )
    
    if method == 'Gardner' :
        den = 310 * ( vp )**0.25

    if method == 'Ludwig' :
        vp_kms = vp / 1e3 
        den_gcm = 1.6612*(vp_kms) - 0.4721*(vp_kms)**2 + 0.0671*(vp_kms)**3 - \
                  0.0043*(vp_kms)**4 + 0.000106*(vp_kms**5)
        den = den_gcm * 1e3
        
    return den

# -----------------------------------------------------------------------------
def DeTrend_filt( array, 
                  radius=4,
                  remove=None, 
                  nanfill=None, 
                  pmode='linear_ramp', 
                  padw=0, 
                  alpha=None, 
                  zoom=None,
                  plot=False, 
                  vmin=None, 
                  vmax=None, 
                  ftype='mean',
                  iter=1, 
                  factor=2, 
                  filt=False, 
                  nodata=True, 
                  mask=None,
                  sliding_circle=True ) :
    """
    Applies detrending and filtering to a 2D array.

    Parameters:
    - array: 2D numpy array
        The input array to be detrended and filtered.
    - radius: int, optional (default=4)
        The radius of the moving window used for detrending.
    - pmode: str, optional (default='linear_ramp')
        The padding mode used when padding the array.
    - padw: int, optional (default=0)
        The width of the padding applied to the array.
    - alpha: float, optional (default=None)
        The alpha value used for padding.
    - plot: bool, optional (default=False)
        If True, plots the original array, trend, and filtered array.
    - vmin: float, optional (default=None)
        The minimum value for the color scale in the plots.
    - vmax: float, optional (default=None)
        The maximum value for the color scale in the plots.
    - ftype: str, optional (default='mean')
        The type of filtering to be applied.
    - iter: int, optional (default=1)
        The number of iterations for the filtering.
    - factor: int, optional (default=2)
        The factor used for the filtering.
    - filt: bool, optional (default=False)
        If True, applies filtering to the detrended array.
    - nodata: bool, optional (default=True)
        If True, sets the filtered values outside the array to NaN.
    - mask: numpy array, optional (default=None)
        A mask array where True values indicate locations to be set to NaN.

    Returns:
    - ar_filt: 2D numpy array
        The detrended and filtered array.
    - trend: 2D numpy array
        The trend component of the array.
    """

    array = np.copy( array )

    if zoom is not None:
        array = utl.resampling( array, zoom, spl_order=1 )

    nan = np.isnan( array )
    if remove == 'mean': 
        initial_mean = np.nanmean( array )
        array = array - initial_mean
    else:
        initial_mean = 0
    if nanfill is not None: 
        array = utl.fillnan( array, method=nanfill )

    nan = np.isnan(array)
    ar_pad, original_shape_indx = utl.pad_array( array, radius*2+padw, alpha=alpha, 
                                                 mode=pmode, ptype=None, plot=False)

    ar_filt = np.copy( array )
    trend = np.copy( array )

    # Create a grid of points
    X1, Y1 = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))

    # Create a circular mask
    if sliding_circle:
        wmsk = X1**2 + Y1**2 <= radius**2
        X1, Y1 = X1[wmsk], Y1[wmsk]
        wmsk = wmsk.flatten()
    else:
        wmsk = np.full(X1.shape, True).flatten()

    X1 = X1.flatten()
    Y1 = Y1.flatten()

    A = np.vstack([X1, Y1, np.ones(Y1.shape[0])]).T

    # Vectorized detrending
    for idx, ai in np.ndenumerate( array ):
    
        xi, yi = original_shape_indx[0] + idx[0], original_shape_indx[2] + idx[1]
        wi = ar_pad[xi-radius:xi+radius+1, yi-radius:yi+radius+1].ravel()[wmsk]
        C, _, _, _ = np.linalg.lstsq(A, wi, rcond=None)
        ti = C[0]*X1 + C[1]*Y1 + C[2]
        ri = wi - ti
        ar_filt[idx[0], idx[1]] = ri[len(ri)//2]
        trend[idx[0], idx[1]] = ti[len(ti)//2]

    if nodata:
        ar_filt[nan] = np.nan
    
    if zoom is not None:
        trend = utl.resampling( ar_pad, 1/zoom, spl_order=1 )
        ar_filt = utl.resampling( ar_filt, 1/zoom, spl_order=1 )

    if initial_mean != 0:
        trend += initial_mean

    if filt:
        ar_filt = utl.filt2d(ar_filt, ftype=ftype, 
                             iter=iter, radius=radius, factor=factor)

    if mask is not None:
        ar_filt[mask] = np.nan

    if plot:
        plta(array, sbplt=[1, 3, 1], tit='Original')
        plta(trend, vmin, vmax, sbplt=[1, 3, 2], tit='Trend', new_fig=False)
        plta(ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Filtered', new_fig=False)

    return ar_filt, trend

# -----------------------------------------------------------------------------
@njit( parallel=True )
def winfun( array, radius=4, pmode='linear_ramp', sx=1.0, sy=1.0, n=1,
            plot=False, vmin=None, vmax=None, ftype='mean', function='dz',
            iter=1, factor=2, filt=False, nodata=True, mask=None, method_dz='fft' ) :
    
    nan = np.isnan( array )
    
    ar_pad, original_shape_indx = utl.pad_array(array, radius+1, pmode, ptype=None )

    ar_filt = np.copy( array )
    
    for idx, ai in np.ndenumerate( array ) :
        xi, yi = original_shape_indx[0]+idx[0], original_shape_indx[2]++idx[1] 
        wi = ar_pad[ xi-radius:xi+radius, yi-radius:yi+radius ]
            
        if function == 'dz' : 
            pnx, pny = wi.shape
            wi = wi - wi.mean()
            kx = 2 * np.pi * np.fft.fftfreq( pnx, sx )
            ky = 2 * np.pi * np.fft.fftfreq( pny, sy ) 
            KX, KY = np.meshgrid( kx, ky )
            K = np.sqrt( KX ** 2 + KY ** 2 ) 
            fft2 = np.fft.fft2( wi ) 
            di = np.real(np.fft.ifft2( fft2 * K ) )
    
        ar_filt[ idx[0], idx[1] ] = di[ radius, radius ]
    
    if filt is True :
        ar_filt = utl.filt2d( ar_filt, ftype=ftype, iter=iter, radius=radius, 
                              factor=factor )
        
    if nodata is True :
        ar_filt[nan] = np.nan
    
    if mask is not None :        
        ar_filt[mask] = np.nan
    
    if plot == True:
        
        plta( array, sbplt=[1, 2, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 2, 2], tit='Filtered')
        
#        plta( ar_filt - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt 

# -----------------------------------------------------------------------------
def mlv_filter( array, filter_size, plot=False ) :

    array = np.copy( array )

    height, width = array.shape
    output_array = np.copy( array )
    
    for i in range(height):

        for j in range(width):
        
            neighborhood = array[i:i+filter_size, j:j+filter_size]
        
            variances = np.var(neighborhood, axis=(0,1))
        
            least_variance_index = np.argmin(variances)
        
            output_array[i, j] = np.median(neighborhood[least_variance_index])
    

    return output_array

# -----------------------------------------------------------------------------
def terracing( array, 
               padw=0, 
               pmode='linear_ramp', 
               remove='mean', 
               order=[1], 
               alpha=0, 
               mask=None, 
               nodata=True, 
               plot=False, 
               vmin=None, 
               vmax=None, 
               filt=True, 
               ftype='median', 
               iter=1, 
               fradius=1, 
               fiter=1,
               pfilt=True, 
               pfradius=1,
               pfiter=[ 1, 3 ],
               pftype=['gauss', 'hanning'],
               nanfill='gdal',
               zoom=None ) :  
    
    """
    Apply terracing operator to an input array.

    Parameters:
    - array: numpy.ndarray
        Input array.
    - padw: int, optional
        Width of padding. Default is 0.
    - pmode: str, optional
        Padding mode. Default is 'linear_ramp'.
    - remove: str or None, optional
        Method to remove trend from the array. Default is 'mean'.
    - order: list of int, optional
        Order of polynomial fit for trend removal. Default is [1].
    - alpha: float, optional
        Alpha value for padding. Default is 0.
    - mask: numpy.ndarray or None, optional
        Mask array. Default is None.
    - nodata: bool, optional
        Flag to indicate whether to assign NaN to nodata values. Default is True.
    - plot: bool, optional
        Flag to indicate whether to plot the original and terraced arrays. Default is False.
    - vmin: float or None, optional
        Minimum value for plotting. Default is None.
    - vmax: float or None, optional
        Maximum value for plotting. Default is None.
    - filt: bool, optional
        Flag to indicate whether to apply filtering to the terraced array. Default is False.
    - ftype: str, optional
        Type of filter to apply. Default is 'hanning'.
    - iter: int, optional
        Number of iterations for the terracing operator. Default is 1.
    - fradius: int, optional
        Radius for filtering. Default is 1.
    - fiter: int, optional
        Number of iterations for filtering. Default is 1.
    - pfilt: bool, optional
        Flag to indicate whether to apply additional filtering during iterations. Default is True.
    - pfsigma: float, optional
        Sigma value for additional filtering. Default is 0.5.
    - pfradius: int, optional
        Radius for additional filtering. Default is 2.
    - nanfill: str or None, optional
        Method to fill NaN values. Default is 'gdal'.
    - zoom: float or None, optional
        Zoom factor for resampling. Default is None.

    Returns:
    - terr: numpy.ndarray
        Terraced array.
    """

    array = np.copy( array )
    nan = np.isnan( array )
        
    if remove is None: 
        arrayr = array
    
    if remove == 'mean': 
        arrayr = array - np.nanmean( array )
    
    if remove == 'trend': 
        arrayr = polyfit2d( array=array, order=order )[0]
    
    if nanfill is not None: 
        arrayr = utl.fillnan(arrayr, method=nanfill)
    
    terr_pad, pad_shape = utl.pad_array( arrayr, padw, pmode, alpha=alpha )

    # Laplacian kernel 3x3
    lap_kernel = np.array( [ [  1,  1,  1 ],
                             [  1, -8,  1 ],
                             [  1,  1,  1 ] ] )

    for i in range( iter ) :

        if ( zoom is not None ) and ( zoom != 1 ) :
            terr_pad = utl.resampling( terr_pad, zoom, 
                                       prefilter=True, spl_order=1 )

        if pfilt is True :
            terr_pad = utl.filt2d( terr_pad, 
                                   ftype=pftype,
                                   radius=pfradius, 
                                   iter=pfiter )
        
        # Create shifted grids
        terr_pad_shift_lst = shift2Darray( terr_pad )[0]
        pad_lap_shift_lst = []
        pad_min_shift_lst = []
        pad_max_shift_lst = []

        # Crete staggred grids
        for ti in terr_pad_shift_lst :
            lap_i = utl.sp.ndimage.convolve( ti, lap_kernel )
            min_i = utl.sp.ndimage.minimum_filter( ti, 3, mode='nearest' )
            max_i = utl.sp.ndimage.maximum_filter( ti, 3, mode='nearest' )
            pad_lap_shift_lst.append( lap_i )
            pad_min_shift_lst.append( min_i )
            pad_max_shift_lst.append( max_i )
        
        # Average the staggered grids
        pad_lap = np.mean( np.stack( pad_lap_shift_lst ), axis=0 )
        pad_min = np.mean( np.stack( pad_min_shift_lst ), axis=0 )
        pad_max = np.mean( np.stack( pad_max_shift_lst ), axis=0 )

        idx_mi0 = pad_lap < 0
        idx_ma0 = pad_lap > 0

        terr_pad[ idx_mi0 ] = pad_max[ idx_mi0 ]
        terr_pad[ idx_ma0 ] = pad_min[ idx_ma0 ]
        
        if ( zoom is not None ) and\
           ( zoom != 1 ) :
            terr_pad = utl.resampling( terr_pad, 1/zoom, 
                                       prefilter=False, 
                                       spl_order=1 )

    terr = utl.crop_pad( terr_pad, pad_shape )

    if filt is True :
        terr = utl.filt2d( terr, 
                           ftype=ftype, 
                           iter=fiter, 
                           radius=fradius )
    
    if nodata is True :
        terr[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        terr[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plta( arraym, sbplt=[1, 2, 1], tit='original' )
        plta( terr, vmin, vmax, sbplt=[1, 2, 2], tit='Teracing operator')
        
    return terr

# -----------------------------------------------------------------------------
def conv( array, kernel='WE', padw=0, pmode='linear_ramp', remove='mean', order=[1], 
          alpha=0, mask=None, nodata=True, plot=False, vmin=None, vmax=None, 
          filt=False, ftype='hanning', iter=1, radius=1, iter_filt=1,  pre_filt=True ) :  

    array = np.copy( array )
    nan = np.isnan( array )
    
    if remove is None: array = array
    if remove == 'mean': array = array - np.nanmean( array )        
    if remove == 'trend': array = polyfit2d( array=array, order=order )[0]   

    if type( kernel ) == str :
        if kernel == 'NS' :
            kernel = [ [ -1, 2, -1 ],
                       [ -1, 2, -1 ],
                       [ -1, 2, -1 ], ]
            
        if kernel == 'WE' :
            kernel = [ [ -1, -1, -1 ],
                       [  2,  2,  2 ],
                       [ -1, -1, -1 ], ]   
            
        if kernel == 'SW_NE' :
            kernel = [ [ -1, -1,  2 ],
                       [ -1,  2, -1 ],
                       [  2, -1, -1 ], ]   

        if kernel == 'NW_SE' :
            kernel = [ [  2, -1, -1 ],
                       [ -1,  2, -1 ],
                       [ -1, -1,  2 ], ]               
            
    ar_pad, pad_shape = utl.pad_array( array, padw, pmode, alpha=alpha )
    
    ar_filt = utl.sp.ndimage.convolve( ar_pad, kernel, mode='nearest' )    
    
    ar_filt = utl.crop_pad( ar_filt, pad_shape )
    
    if filt is True :
        ar_filt = utl.filt2d( ar_filt, ftype=ftype, iter=iter_filt, radius=radius )        
    
    if nodata is True :
        ar_filt[nan] = np.nan
    
    if mask is None:
        arraym = array      
    else:
        ar_filt[mask] = np.nan
        arraym = np.copy(array)
        arraym[mask] = np.nan
        
    if plot == True:
        plta( arraym, sbplt=[1, 2, 1], tit='original' )
        plta( ar_filt, vmin, vmax, sbplt=[1, 2, 2], tit='Teracing operator')
        
    return ar_filt   

# -----------------------------------------------------------------------------
def flexure( array, padw=0, pmode='linear_ramp', sx=1.0, sy=1.0, rho_c=2670, rho_m=3300, 
             Te=None, D=1.11e21, E=100e11, v=0.25, g=9.81, alpha=0, ref_depth=30e3, 
             rho_w=1030, mask=None, nodata=True, plot=False, vmin=None, vmax=None ) :  

    array = np.copy( array )
    nan = np.isnan( array )       
    
    ar_pad, pad_shape = utl.pad_array( array, padw, pmode, alpha=alpha ) 
    
    if Te is not None :
        D = Te**3 * ( E / ( 12 * ( 1 - v**2 ) ) )
    
    K, fft2, _, _, _, _ = fft2d( ar_pad, sx, sy )
    flex_K = ( ( rho_c ) / ( rho_m - rho_c ) ) * ( ( ( D * K**4 ) / ( ( rho_m - rho_c ) * g ) ) + 1 )**(-1)
    flex_pad = np.real( np.fft.ifft2( fft2 * flex_K ) )    
    
    flexure = utl.crop_pad( flex_pad, pad_shape )
    
    if nodata is True :
        flexure[nan] = np.nan
    
    if mask is not None :
        flexure[mask] = np.nan
        array[mask] = np.nan
        
    if plot == True:
        plta( array, sbplt=[1, 2, 1], tit='Topo_Load' )
        plta( flexure, vmin, vmax, sbplt=[1, 2, 2], tit='Flexure')
        
    return flexure
    
# -----------------------------------------------------------------------------
def park_old_inv( grav_array, topo_array=0, rho=250, iter=5, n=5,
                  sx=1.0, sy=1.0, padw=0, pmode='gdal', z0=None,
                  nanfill=None, nodata=True, plot=False, vmin=None, vmax=None,
                  z_negative=True, mean_depth=None ) :
        
        
        if type( topo_array ) in ( int, float ) :
            topo_array = np.full( grav_array.shape, topo_array ) 

        grav_array = grav_array.copy().astype(float)
        topo_array = topo_array.copy().astype(float)

        nan_grav = np.isnan( grav_array )
        nan_topo = np.isnan( topo_array )


        if nanfill is not  None:
            grav_array = utl.fillnan( grav_array, method=nanfill )
            topo_array = utl.fillnan( topo_array, method=nanfill )

        grav_array = grav_array - np.nanmean( grav_array )
        grav_ar_pad, original_shape_indx = utl.pad_array( grav_array, padw, pmode )
        topo_ar_pad, original_shape_indx = utl.pad_array( topo_array, padw, pmode )
        topo_inv_pad = topo_ar_pad.copy()

        for i in range( iter ) :

            if z0 is None :
                z0 = np.abs( np.nanmean( topo_inv_pad ) )            

            taylor_fft_topo = np.zeros( topo_inv_pad.shape )
            
            for ip in range(1, n + 1):
                pwr_array = topo_inv_pad ** ip
                K, fft_topo, _, _, _, _ = fft2d( pwr_array, sx=sx, sy=sy )
                taylor_fft_topo = taylor_fft_topo +  ( K ** ( ip - 1 ) / np.math.factorial( ip ) ) * fft_topo

            K, fft_grav, _, _, _, _ = fft2d( grav_ar_pad/(1e5), sx=sx, sy=sy )
            fft_topo_inv = - ( ( fft_grav * np.exp( -K * z0 ) ) / ( 2 * np.pi * G * rho ) ) - taylor_fft_topo
            topo_inv_pad = np.real( np.fft.ifft2( fft_topo_inv ) ) 
            grav_inv_pad = parker( topo_inv_pad, rho=rho, sx=sx, sy=sy, n=n )

        topo_inv = utl.crop_pad( topo_inv_pad, original_shape_indx )
        grav_inv = utl.crop_pad( grav_inv_pad, original_shape_indx )

        if z_negative == True :
            topo_inv = 0 - topo_inv

        if plot == True :

            utl.plta( topo_array, sbplt=[1,3,1] )
            utl.plta( topo_inv, sbplt=[1,3,2] )
            utl.plta( grav_inv-grav_array, sbplt=[1,3,3] )
            plt.tight_layout()

        return topo_inv, grav_inv

# -----------------------------------------------------------------------------
def reduce_to_pole( array, sx, sy, inc, dec, sinc=None, sdec=None,
                    padw=0, pmode='gdal', alpha=None, nodata=True, mask=None,
                    remove='mean', order=[1], 
                    plot=True, vmin=None, vmax=None ) :

    """
    Reduce total field magnetic anomaly data to the pole.
    """

    array = np.copy( array )

    nan = np.isnan( array )

    if remove is None: 
        arrayr = array
    if remove == 'mean': 
        arrayr = array - np.nanmean( array )
    if remove == 'trend': 
        arrayr = polyfit2d(array=array, order=order)[0]

    ar_pad, pad_shape = utl.pad_array( arrayr, padw, pmode, alpha=alpha )

    d2r = np.pi / 180.
    vect = [np.cos(d2r * inc) * np.cos(d2r * dec),
            np.cos(d2r * inc) * np.sin(d2r * dec),
            np.sin(d2r * inc)]

    fx, fy, fz = np.transpose( [ 1 * i for i in vect ] )

    if sinc is None or sdec is None:
        mx, my, mz = fx, fy, fz

    else:
        # mx, my, mz = ang2vec(sinc, sdec)

        inc_rad = np.radians( inc )
        dec_rad = np.radians( dec )
        mx = np.cos( dec_rad ) * np.cos( inc_rad )
        my = np.sin( dec_rad ) * np.cos( inc_rad )
        mz = np.sin( inc_rad )

    pny, pnx = ar_pad.shape

    kx = 2 * np.pi * np.fft.fftfreq( pnx, sx ) # wave-number in x direction
    ky = 2 * np.pi * np.fft.fftfreq( pny, sy ) # wave-number in y direction
    kx, ky = np.meshgrid( kx, ky )
    kz_sqr = kx**2 + ky**2

    a1 = mz*fz - mx*fx
    a2 = mz*fz - my*fy
    a3 = -my*fx - mx*fy
    b1 = mx*fz + mz*fx
    b2 = my*fz + mz*fy

    with np.errstate( divide='ignore', invalid='ignore' ) :
        rtp = ( kz_sqr ) / ( a1*kx**2 + a2*ky**2 + a3*kx*ky +
                1j * np.sqrt( kz_sqr ) * ( b1*kx + b2*ky ) )

    rtp[0, 0] = 0
    ft_pole = rtp * np.fft.fft2( ar_pad )

    rtp_array_pad = np.real( np.fft.ifft2( ft_pole ) )

    rtp_array = utl.crop_pad( rtp_array_pad, pad_shape )

    if nodata is True :
        rtp_array[nan] = np.nan

    if mask :
        array[~mask] = np.nan
        rtp_array[~mask] = np.nan

    if plot == True:

        utl.plta( array, sbplt=[1, 3, 1], 
                  tit='original', new_fig=True )

        utl.plta( rtp_array, vmin, vmax, sbplt=[1, 3, 2], 
                  tit='RTP anomaly', new_fig=False )
        
        utl.plta( array - rtp_array, vmin, vmax, sbplt=[1, 3, 3], 
                  tit='Differences', new_fig=False )

    return rtp_array

# -----------------------------------------------
def shift2Darray( array, factor=1 ) :

    # List to store the original and shifted grids
    grids = [ array ]
    
    # Define the half cell size
    half_cell = 0.5 * factor
    
    # Define the possible shifts (in terms of half cell size)
    shifts = [
        (half_cell, 0), (-half_cell, 0),  # Right, Left
        (0, half_cell), (0, -half_cell),  # Down, Up
        (half_cell, half_cell), (half_cell, -half_cell),  # Down-Right, Up-Right
        (-half_cell, half_cell), (-half_cell, -half_cell)  # Down-Left, Up-Left
        ]
    
    # Lists to store shifted x and y coordinates
    x_shifted_coords = []
    y_shifted_coords = []
    
    # Apply each shift and interpolate the grid
    for shift_val in shifts:
        shifted_grid = utl.sp.ndimage.shift( array, 
                                             shift_val, 
                                             order=1,
                                             mode='nearest' )
        
        grids.append( shifted_grid )

        # Calculate shifted coordinates
        x_shift, y_shift = shift_val
        x_coords, y_coords = np.meshgrid( np.arange(array.shape[1] ), 
                                          np.arange(array.shape[0] ) )
        x_shifted = x_coords + x_shift
        y_shifted = y_coords + y_shift
        
        x_shifted_coords.append(x_shifted)
        y_shifted_coords.append(y_shifted)
    
    return grids, x_shifted_coords, y_shifted_coords


# -----------------------------------------------------------------------------

# def remove_small_imperfections(image, size_threshold):

