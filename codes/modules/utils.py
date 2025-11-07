# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:58:52 2020

@author: lzampa
"""

import os
import pyproj as prj
import numpy as np
import scipy as sp
import itertools
import random
import matplotlib.pyplot as plt
import time 
import copy
import datetime
from osgeo import gdal, osr, ogr
import shutil
import tempfile
import platform
import sys
from matplotlib import cm
from matplotlib import dates as mdates
from matplotlib.widgets import LassoSelector
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import shapely
from shapely.geometry import Polygon, LineString, Point 
from shapely.geometry import MultiPolygon, MultiPoint, MultiLineString
from scipy import signal 
import io 
import pdfkit


# -----------------------------------------------------------------------------
# Constants

G = 6.6743*1e-11 # [m3/(kg *s^2)]
M = 5.97*1e24 # [kg]
# GM_grs80 = 3986005 * 1e8 # [m3/s^2]
# GM_wgs84 = 3986004.41 * 1e8 # [m3/s^2]
a_wgs84 = 6378137 # [m]
c_wgs84 = 6356752 # [m]
R_wgs84 = ((a_wgs84**2)*c_wgs84)**(1/3) # [m]
J2_wgs84 = 1.081874*1e-3
w_wgs84 = 7.292115*1e-5 # [rad/sec]

# -----------------------------------------------------------------------------
# User color maps

# ---
# wysiwyg_gmt  (see GMT mapping-tools)
# Define the colors
wysiwyg_colors = [ (0.000000, (64/255, 0/255, 64/255)),
                   (0.052632, (64/255, 0/255, 192/255)),
                   (0.105263, (0/255, 64/255, 255/255)),
                   (0.157895, (0/255, 128/255, 255/255)),
                   (0.210526, (0/255, 160/255, 255/255)),
                   (0.263158, (64/255, 192/255, 255/255)),
                   (0.315789, (64/255, 224/255, 255/255)),
                   (0.368421, (64/255, 255/255, 255/255)),
                   (0.421053, (64/255, 255/255, 192/255)),
                   (0.473684, (64/255, 255/255, 64/255)),
                   (0.526316, (128/255, 255/255, 64/255)),
                   (0.578947, (192/255, 255/255, 64/255)),
                   (0.631579, (255/255, 255/255, 64/255)),
                   (0.684211, (255/255, 224/255, 64/255)),
                   (0.736842, (255/255, 160/255, 64/255)),
                   (0.789474, (255/255, 96/255, 64/255)),
                   (0.842105, (255/255, 32/255, 64/255)),
                   (0.894737, (255/255, 96/255, 192/255)),
                   (0.947368, (255/255, 160/255, 255/255)),
                   (1.000000, (255/255, 224/255, 255/255)) ]
# Check if the colormap is already registered
try:
    plt.cm.get_cmap('wysiwyg')
except ValueError:
    # If the colormap is not registered, register it
    wysiwyg = LinearSegmentedColormap.from_list("wysiwyg", wysiwyg_colors)
    plt.colormaps.register(name='wysiwyg', cmap=wysiwyg)
# ---

# ---
# cmy  (see Oasis Montaj)
# Define the colors
rgb_values = np.array([ [0, 0, 255],
                        [0, 85, 255],
                        [0, 127, 255],
                        [0, 170, 255],
                        [0, 212, 255],
                        [0, 233, 255],
                        [0, 255, 255],
                        [0, 255, 200],
                        [0, 255, 145],
                        [0, 255, 63],
                        [0, 255, 49],
                        [0, 255, 36],
                        [0, 255, 0],
                        [72, 255, 0],
                        [99, 255, 0],
                        [109, 255, 0],
                        [145, 255, 0],
                        [182, 255, 0],
                        [218, 255, 0],
                        [255, 255, 0],
                        [255, 233, 0],
                        [255, 212, 0],
                        [255, 191, 0],
                        [255, 180, 0],
                        [255, 170, 0],
                        [255, 148, 0],
                        [255, 137, 0],
                        [255, 127, 0],
                        [255, 106, 0],
                        [255, 85, 0],
                        [255, 53, 0],
                        [255, 21, 0],
                        [255, 0, 0],
                        [255, 0, 55],
                        [255, 0, 109],
                        [255, 0, 182],
                        [255, 11, 218],
                        [255, 121, 255],
                        [255, 159, 255] ] ) / 255
# Create a list of positions for the colors
positions = np.linspace(0, 1, len(rgb_values))
# Create a list of tuples where each tuple is a position, color pair
cmy_colors = [(position, color) for position, color in zip(positions, rgb_values)]
# Create the colormap
try:
    plt.cm.get_cmap('cmy')
except ValueError:
    # If the colormap is not registered, register it
    cmy = LinearSegmentedColormap.from_list("cmy", cmy_colors)
    plt.colormaps.register(name='cmy', cmap=cmy)
# ---

# -----------------------------------------------------------------------------
def modify_colormap(cmap, factor):

    # Get the colors from the colormap
    rgb_values = cmap(np.linspace(0, 1, cmap.N))

    # Create a list of positions for the colors
    positions = np.linspace(0, 1, len(rgb_values))

    # Shift positions to range -1 to 1
    shifted_positions = positions * 2 - 1

    # Apply a non-linear transformation to the positions
    modified_positions = np.sign(shifted_positions) * np.power(np.abs(shifted_positions), factor)

    # Shift positions back to range 0 to 1
    modified_positions = (modified_positions + 1) / 2

    # Create a list of tuples where each tuple is a position, color pair
    modified_colors = [(position, color) for position, color in zip(modified_positions, rgb_values)]

    # Create the modified colormap
    modified_cmap = LinearSegmentedColormap.from_list("modified_" + cmap.name, modified_colors)

    return modified_cmap

# -----------------------------------------------------------------------------
def cm2in( cm ) :
    
    inches = cm * 1/2.54 
    
    return inches 

# -----------------------------------------------------------------------------
def is_number(value):
    """
    Check if the given value can be converted to a float.
    Args:
        value: The value to check.
    Returns:
        bool: True if the value can be converted to a float, False otherwise.
    """
    
    try:
        
        float(value)
        
        return True
    
    except (ValueError, TypeError):
        
        return False

# -----------------------------------------------------------------------------
def tmi(t1=None, t2=None):
    """
    Print time interval from time t1 to t2
    --- If "t2 = None" , then, "t2 = now"
    """
    
    if ( t1 is None ) and ( t2 is None ) :
        return time.time()
    
    if t2 == None :
        print( time.time() - t1 )
    else:
        print( t2 - t1 )

# -----------------------------------------------------------------------------
def dms2dd( degrees, minutes=0.0, seconds=0.0 ):
    """
    Convert degrees, minutes, and seconds to decimal degrees.

    Parameters:
    degrees (int, float, list, tuple, numpy.ndarray): Degrees.
    minutes (int, float, list, tuple, numpy.ndarray): Minutes.
    seconds (int, float, list, tuple, numpy.ndarray): Seconds.

    Returns:
    numpy.ndarray: Decimal degrees.
    """
    # Convert inputs to numpy arrays if they are not already
    if not isinstance(degrees, np.ndarray):
        degrees = np.array(degrees)
    if not isinstance(minutes, np.ndarray):
        minutes = np.array(minutes)
    if not isinstance(seconds, np.ndarray):
        seconds = np.array(seconds)

    return degrees + minutes / 60 + seconds / 3600
    
# -----------------------------------------------------------------------------
def dd2dms( decimal_degrees ):
    """
    Convert decimal degrees to degrees, minutes, and seconds.

    Parameters:
    decimal_degrees (int, float, list, tuple, numpy.ndarray): Decimal degrees.

    Returns:
    tuple: Three numpy.ndarrays representing degrees, minutes, and seconds.
    """
    # Convert input to numpy array if it is not already
    if not isinstance( decimal_degrees, np.ndarray ):
        decimal_degrees = np.array(decimal_degrees)

    # Calculate degrees, minutes, and seconds
    degrees = np.floor(decimal_degrees)
    minutes = np.floor((decimal_degrees - degrees) * 60)
    seconds = (decimal_degrees - degrees - minutes / 60) * 3600

    return degrees, minutes, seconds
        
# -----------------------------------------------------------------------------
def stat( array, decimals=None, show=True, out=None, multilines=False, sep='' ):
    """
    This function prints the statistics of a numpy array.

    Parameters:
    - array (numpy array): The input array for which the statistics are to be calculated.
    - decimals (int, optional): The number of decimal places to round to. 
        If None, no rounding is applied. Default is None.
    - show (bool, optional): If True, the function will print the statistics. 
        If False, the function will not print anything. Default is True.
    - out (str, optional): If 'str', the function will return the statistics as a string. 
        If None, the function will return the statistics as a tuple. Default is None.
    - multilines (bool, optional): If True, the function will print each statistic on a new line. 
        If False, all statistics will be printed on the same line. Default is False.
    - sep (str, optional): The separator between statistics when printed on the same line. 
        Default is an empty string.

    Returns:
    If out == 'str', returns a string containing the statistics.
    If out == None, returns a tuple containing the statistics (Min, Max, Mean, Std).
    """

    Min, Max, Mean, Std = ( np.nanmin(array), np.nanmax(array),
                            np.nanmean(array), np.nanstd(array) )
    

    if decimals != None:
        Min, Max, Mean, Std = ( np.round(Min, decimals), np.round(Max, decimals),
                                np.round(Mean, decimals), np.round(Std, decimals) )   
    
    len_ = np.max( ( len(str(Min)), len(str(Max)), len(str(Mean)), len(str(Std)) ) )      
    
    string = f"Min:{Min}{sep} Max:{Max}{sep} Mean:{Mean}{sep} Std:{Std}"
   
    if multilines == True :
        string = f"Min  = {Min: >{len_}} \nMax  = {Max: >{len_}} \nMean = {Mean: >{len_}} \nStd  = {Std: >{len_}}"
    
    if show == True :
        print( string ) 
        
    if out == str :
        return string
    
    else :     
        return Min, Max, Mean, Std    

# -----------------------------------------------------------------------------
def extend_lim( lim, d, method='percentage', sqr_area=False, plot=False, round2=None,
                print_lim=False ):
    """
    - lim = [xmin, xmax, ymin, ymax]
    - two methods aveilable:
        1) distance --> extend for a distance in the same units of lim
        2) percentage --> extend for a percentage of the total xy lengths of the area
    - sqr_area = True --> force the output boundaries to have length_X = length_Y
    """
    
    xd, yd = lim[1] - lim[0], lim[3] - lim[2]
    if sqr_area == True:
        if xd > yd: lim2 = [lim[0], lim[1], lim[2] - (xd - yd) / 2, lim[3] + (xd - yd) / 2]
        if yd > xd: lim2 = [lim[0] - (yd - xd) / 2, lim[1] + (yd - xd) / 2, lim[2], lim[3]]
        xd, yd = lim2[1] - lim2[0], lim2[3] - lim2[2]
    else: lim2 = lim

    if np.size(d) == 1: dx, dy = d, d
    if np.size(d) == 2: dx, dy = d[0], d[1]
    if method == 'percentage':
        xp, yp = xd * dx / 100, yd * dy / 100
        lim_new = [lim2[0] - xp, lim2[1] + xp, lim2[2] - yp, lim2[3] + yp]

    if method == 'distance': lim_new = [lim2[0] - dx, lim2[1] + dx, lim2[2] - dy, lim2[3] + dy]
    
    if round2 is not None :
        lim_new = [ round(l, round2) for l in lim_new ]

    if plot == True:
        xplot1 = [lim[0], lim[0], lim[1], lim[1], lim[0]]
        yplot1 = [lim[2], lim[3], lim[3], lim[2], lim[2]]
        xplot2 = [lim_new[0], lim_new[0], lim_new[1], lim_new[1], lim_new[0]]
        yplot2 = [lim_new[2], lim_new[3], lim_new[3], lim_new[2], lim_new[2]]
        plt.plot(xplot1, yplot1, c='r')
        plt.plot(xplot2, yplot2, c='b')

    if print_lim == True :
        print( lim_new )
        
    return lim_new

# -----------------------------------------------------------------------------
def prj_( prjcode ):
    """
    Convert epsg or proj4 string to proj4 code object
    """
    if type( prjcode )==int:
        prj4_obj = prj.CRS( 'epsg:'+str( prjcode ) )
        
    if type( prjcode )==str:
        prj4_obj = prj.CRS( prjcode )
        
    return prj4_obj

# -----------------------------------------------------------------------------
def prj_units( prjcode ):
    """
    Return EPGS or Proj4 code units ( i.e., meters or degree )
    """
    
    prj4_obj = prj_( prjcode )
    units = prj4_obj.axis_info[0].unit_name
        
    return units

# -----------------------------------------------------------------------------
def prjxy( prjcode_in, prjcode_out, x, y, z=None ):
    """
    Transform coordinates from a reference system to another
    """
    
    x = np.copy( x )
    y = np.copy( y )
    
    if prjcode_in == prjcode_out :
        prj_coord = x, y
        
    else :
        prj_in = prj_( prjcode_in )
        prj_out = prj_( prjcode_out )
        trans = prj.Transformer.from_crs( prj_in, prj_out, always_xy=True )
        if z != None :
            prj_coord = trans.transform( x, y, z )
        if z == None :
            prj_coord = trans.transform( x, y )

    return prj_coord

# -----------------------------------------------------------------------------
def prj2epsg( prjcode ) :
    
    crs = prj.CRS( prjcode )
    epsg = crs.to_epsg()
    
    return epsg

# -----------------------------------------------------------------------------
def lim_sort( lim, p=False ):

    lim_out = [lim[0], lim[2], lim[1], lim[3]]

    if p == True :
        print( lim_out )

    return lim_out

# -----------------------------------------------------------------------------
def lim2points( lim ) :

    x = np.array( ( lim[0], lim[1], lim[1], lim[0] ) )
    y = np.array( ( lim[3], lim[3], lim[2], lim[2] ) )

    return x, y

# -----------------------------------------------------------------------------
def prj_lim(lim, prjcode_in, prjcode_out, sort='xlyl'):
    """
    Transform limits of an area (i.e. [LonMin, LonMax, LatMin, LatMax]) from a
    reference system to another
    """
    if sort=='xlyl':
        x,y = prjxy(prjcode_in, prjcode_out, (lim[0],lim[1]), (lim[2],lim[3]))
    if sort=='xyl':
        x,y = prjxy(prjcode_in, prjcode_out, (lim[0],lim[2]), (lim[1],lim[3]))

    limf = [float(x[0]), float(x[1]), float(y[0]), float(y[1])]

    return limf

# -----------------------------------------------------------------------------
def prj_centre( xy_centre, prj_type, ellps='WGS84' ):
    """
    Creates a PROJ string code, with a reference system centered (i.e tangent)
    to a defined point on Earth surface: xy_centre,
    i.e. [Lon_centre, Lat_centre] (Lat&Lon in geographic coordinates)
    """
    x_centre, y_centre = xy_centre
    if type(prj_type)==str:
       prj_cent_str = f'''+proj={prj_type} +lat_0={x_centre} +lon_0={y_centre} +ellps={ellps}'''
    if type(prj_type)==int:
       prj_cent_str = f'''+epsg={prj_type} +lat_0={x_centre} +lon_0={y_centre} +ellps={ellps}'''

    return prj_cent_str

# -----------------------------------------------------------------------------
def xy_in_lim( x, y, lim, extend=0, method='percentage', plot=False, s=1, plot_lim=None ):
    """
    Return all xy points within the given limits [xmin, xmax, ymin, ymax]
    """
    
    if extend != 0 :
        lim = extend_lim( lim, d=extend, method=method )
    
    idx = (x >= lim[0]) & (x <= lim[1]) & (y >= lim[2]) & (y <= lim[3])
    xl = x[ idx ]
    yl = y[ idx ]
    
    if plot is True :
        plt.scatter( x[~idx], y[~idx], c='b', s=s )
        plt.scatter( xl, yl, c='r', s=s )
        if plot_lim is not None : 
            lim_ext = extend_lim( lim, plot_lim )
            plt.xlim( lim_ext[0], lim_ext[1] )
            plt.ylim( lim_ext[2], lim_ext[3] )
        
    return xl, yl, idx

# -----------------------------------------------------------------------------
def find_by_dist( xy1, xy2, dist, prj_dist=None, prj_in=4326, plot=False ):
    
    """
    Find all points of a set (xy1) within a distance (dist)
    from another set of points (xy2)
    """

    set1 = np.column_stack(xy1)
    set2 = np.column_stack(xy2)

    if prj_dist is not None:
       set1[:,0], set1[:,1] = prjxy(prj_in, prj_dist, set1[:,0], set1[:,1])
       set2[:,0], set2[:,1] = prjxy(prj_in, prj_dist, set2[:,0], set2[:,1])

    f_dist = lambda i: np.any(np.sqrt((i[0]-set2[:,0])**2 + (i[1]-set2[:,1])**2)<=dist)
    set1_idx = np.array((list(map(f_dist, set1))))
    set1_new = set1[set1_idx]

    if plot==True:
        plt.scatter(set1[:,0],set1[:,1], c='g', label='set1')
        plt.scatter(set2[:,0],set2[:,1], c='b', label='set2')
        plt.scatter(set1_new[:,0],set1_new[:,1], marker='+', c='r',
                    label=f'''set1 points, {dist} distant from set2 points''')
        plt.legend()

    if prj_dist is not None:
        set1_new[:,0], set1_new[:,1] = prjxy(prj_dist, prj_in, set1_new[:,0], set1_new[:,1])

    return set1_idx, set1_new

# -----------------------------------------------------------------------------
def min_dist( x, y, prjcode_in=4326, prjcode_out=4326, 
              data_type='scattered_points', factor=1000 ) :
    """
    Find the minimum average distance among a set of scattered points.
    The result is a dictionary contatnjing also other statistics.
    """
    
    if prjcode_in != prjcode_out :
        x, y = prjxy( prjcode_in, prjcode_out, x, y )

    if data_type == 'grids' :
        dx, dy, mdxy = stepxy( x, y )
        d = {'dx':dx, 'dy':dy, 'mean':mdxy }        
        
    if  data_type == 'scattered_points' :   
        points_all = np.column_stack( ( x.ravel(), y.ravel() ) )
        points = np.unique( points_all, axis=0 )
        N = points.shape[0]
        md = np.zeros( N )

        # for i in range( N ):
        #     apoints = np.delete( points, i, 0 )
        #     d = np.sqrt( ( points[(i, 0)] - apoints[:, 0] ) ** 2 + \
        #                  ( points[(i, 1)] - apoints[:, 1] ) ** 2 )
        #     md[ i ] = np.min( d )

        # --------------------------------------
        kdtree = sp.spatial.cKDTree( points )
        distances, _ = kdtree.query( points, k=2 )
        md = np.min( distances[:, 1:], axis=1 )
        # --------------------------------------

        meand = np.mean(md)
        mind = np.min(md)
        maxd = np.max(md)
        stdd = np.std(md)
        moded = sp.stats.mode(md, nan_policy='omit', keepdims=True).mode[0]
        
        # moded = sp_stat.mode(md, nan_policy='omit')[0][0]
        d = { 'mean':meand,'val':md, 'min':mind, 'max':maxd, 'std':stdd, 'mode':moded }
    
    return d

# -----------------------------------------------------------------------------
def xy2XY( x, y, step=None, lim=None, dist='mean', method='distance', extend=False,
           plot=False, grid=False, rotate=True, treshold=1e-5, fix_point='first',
           s=None, return_step=False ) :

    if grid is True :
        
        if rotate == True :
            angle, x, y, fix_point = grid_rot_angle( x, y, treshold=treshold, fix_point=fix_point ) 
            
    if type( step ) in ( int, float ) :
        step = [ step, step ]
        
        if lim is None :
            lim = xy2lim( x, y, extend=extend, method=method )
            
        xm = np.arange( lim[0], lim[1]+step[0], step[0] )  
        ym = np.arange( lim[3], lim[2]-step[1], -step[1] ) 

    else :
        xm = np.unique( x )
        ym = np.unique( y )
    
    X, Y = np.meshgrid( xm, ym )
    
    if grid == True :
        if rotate == True :
            X, Y, _ = rotate_xy( X, Y, radians=-angle, fix_point=fix_point )
    
    if plot is True :
        plt.scatter( X.ravel(), Y.ravel(), c='r', s=s )
        
    if return_step == True :
        
        return X, Y, step
    
    else :
        
        return X, Y
        
# -----------------------------------------------------------------------------
def xyz2xy( xyz, xy, method='nearest',
            rescale=False, fillnan=True,
            plot=False, blkm='mean', smooth=0 ) :

    x, y, z = xyz

    nan = np.isnan( z )

    x1 = x[ ~nan ]
    y1 = y[ ~nan ]
    z1 = z[ ~nan ]
    
    if len( xy ) == 2 :
        x2, y2 = xy
    if len( xy ) > 2 : 
        x2, y2, z2 = xy
    
    if np.size( x1 ) < 4 :
        method = 'nearest'

    if np.size( x1 ) > 1 :
        min_dist_1 = min_dist( x1, y1, data_type='scattered_points' )['mean']
    else :
        min_dist_1 = 0.0
    if np.size( x2 ) > 1 :
        min_dist_2 = min_dist( x2, y2, data_type='scattered_points' )['mean']
    else : 
        min_dist_2 = 0.0

    if ( min_dist_1 < min_dist_2 ) and ( blkm is not None ) :
        if blkm is True :
            blkm = 'mean'
        x1, y1, z1 = block_m( x1, y1, z1, method=blkm, wind_size=min_dist_2 )

    if method == 'thin_plate':
        input_memory = ( x1.nbytes + y1.nbytes +\
                               z1.nbytes + x2.nbytes + y2.nbytes)
        num_points = len(x1)
        num_int_points = len(x2)
        dist_matrix_mem = num_points * num_int_points * 8
        weights_mem = num_points * 8  # Assuming 8 bytes per weight (float64)
        coeff_mem = num_points * 8 
        estimated_memory = input_memory + dist_matrix_mem + weights_mem + coeff_mem
        total_memory =  4 * 1024**3
        print( 'Estimated memory usage: ', estimated_memory )
        print( 'Total memory: ', total_memory )
        if estimated_memory < total_memory:
            print( 'ok0' )
            rbf = sp.interpolate.Rbf( x1, y1, z1, 
                                      function=method, 
                                      smooth=smooth )
            zn = rbf.ev(x2, y2)

        else :
            # Start block computing
            print( 'ok1' )
            xy1 = np.column_stack( ( x1, y1 ) )
            xy2 = np.column_stack( ( x2.ravel(), y2.ravel() ) )
            xyu, idx = np.unique( np.vstack( ( xy1, xy2 ) ), axis=0, return_index=True )
            X, Y = np.meshgrid( np.unique( xyu[:,0] ), np.unique( xyu[:,1] ) )
            Z = np.concatenate( ( z1, np.nan * np.ones( x2.size ) ) )[ idx ].reshape( X.shape )
            chunk_elements = int(total_memory / (Z.itemsize * 3)) 
            chunk_size = int(np.sqrt(chunk_elements))
            chunk_size =1000
            for i in range(0, Z.shape[0], chunk_size):
                for j in range( 0, Z.shape[1], chunk_size ):
                    chunk = Z[i:i+chunk_size, j:j+chunk_size]
                    X_chunk, Y_chunk = X[i:i+chunk_size, j:j+chunk_size],\
                                       Y[i:i+chunk_size, j:j+chunk_size]
                    valid_mask_chunk = ~np.isnan(chunk)
                    x_valid_chunk = X_chunk[valid_mask_chunk]
                    y_valid_chunk = Y_chunk[valid_mask_chunk]
                    z_valid_chunk = chunk[valid_mask_chunk]
                    if len(x_valid_chunk) > 0:
                        rbf = sp.interpolate.Rbf( x_valid_chunk, y_valid_chunk, z_valid_chunk, 
                                                  function=method, smooth=smooth)
                        filled_chunk = rbf(X_chunk, Y_chunk)
                        Z[i:i+chunk_size, j:j+chunk_size] = filled_chunk
            zn = sp.interpolate.griddata( (X.ravel(), Y.ravel()), Z.ravel(), (x2, y2), 'nearest' )

    elif method == 'nearest' or np.size(x1) < 4:
        zn = sp.interpolate.griddata( (x1, y1), z1, (x2, y2), 
                                      method='nearest', 
                                      rescale=rescale)
    else:
        try:    
            zn = sp.interpolate.griddata( (x1, y1), z1, (x2, y2), 
                                          method=method, rescale=rescale)
        except: 
            zn = sp.interpolate.griddata( (x1, y1), z1, (x2, y2), 
                                          method='nearest', rescale=rescale)

    if fillnan is True :
        
        if ( np.size( zn ) == 1 ) and ( np.any( np.isnan( zn ) ) ) :
            zn = sp.interpolate.griddata( ( x1, y1 ), z1, ( x2, y2 ), 
            method='nearest' ) 
            
        if ( np.size( zn ) > 1 ) and ( np.any( np.isnan( zn ) ) ) :
            idx = np.isnan( zn )
            zn[ idx ] = sp.interpolate.griddata( ( x1, y1 ), z1, ( x2[idx], y2[idx] ), 
                                                  method='nearest' )

    if plot is True :

        plt.figure()
        sts = stat( zn )
        plt.scatter( xy[0], xy[1], c=zn, vmin=sts[2]-sts[3]*2, 
                     vmax=sts[2]+sts[3]*2, cmap='rainbow' )
        plt.colorbar()

    return zn

# -----------------------------------------------------------------------------
def find_by_att( dat1, dat2, col_xy, delta_xy, col_att, delta_att, condition=True,
                 prj_in=4326, prj_dist=None, unique=True, plot=False ) :

    dat1c = np.copy(dat1)
    dat2c = np.copy(dat2)
    
    if prj_dist is not None:
        dat1c[:,col_xy[0]], dat1c[:,col_xy[1]] = prjxy(prj_in, prj_dist, dat1[:,col_xy[0]], dat1[:,col_xy[1]])
        dat2c[:,col_xy[0]], dat2c[:,col_xy[1]] = prjxy(prj_in, prj_dist, dat2[:,col_xy[0]], dat2[:,col_xy[1]])
        
    mask1 = np.full(dat1c.shape[0], False)
    idx21 = []
    uniq = False
    col = col_xy + col_att
    delta = delta_xy + delta_att
    
    for n, i in enumerate(dat2c):
        mask0 = np.full(dat1c.shape[0], True)
        
        for c, d in zip(col, delta):
            msk = (dat1c[:,c] >= i[c] - d ) & ( dat1c[:,c] <= i[c] + d )
            mask0 = mask0 & msk
            
        if np.any(mask0):
            idx21.append([n, mask0])
            if ( np.sum(idx21[-1][1]) > 1 ) and ( unique is True ):
               uniq= uniq | True
               dist = np.sqrt( ( dat1c[mask0,col_xy[0]]-i[1] )**2  + 
                               ( dat1c[mask0,col_xy[1]]-i[0] )**2  )
               idx210N = np.nonzero( mask0*1 )[0]
               min_dist = np.nanmin( dist )
               unid = idx210N[ dist == min_dist ]                   
               idx21[-1][1] = unid[0] 
            if ( np.sum(idx21[-1][1] ) == 1 ): 
                idx21[-1][1] = np.where(mask0)[0].item()
            if ( np.sum(idx21[-1][1]) > 1 ) and ( unique is False ):
                uniq = uniq | True
                idx21[-1][1] = np.where(mask0).tolist()
        mask1 =  mask1 | mask0


    if unique is True:
        idx21 = np.asarray(idx21)

    if condition==False: mask1 = ~mask1
    dat1_msk = dat1[mask1]

    print( 'All data: ', len(dat1) )
    print( 'True: ', len(dat1_msk) )
    print( 'False: ', len(dat1)-len(dat1_msk) )

    if plot==True:
        plt.scatter(dat1[:,col_xy[0]], dat1[:,col_xy[1]], c='g', label='set1')
        plt.scatter(dat2[:,col_xy[0]], dat2[:,col_xy[1]], c='b', label='set2')
        plt.scatter(dat1_msk[:,col_xy[0]], dat1_msk[:,col_xy[1]], marker='+', c='r',
                    label='set1 points, defined by condition')
        plt.legend()

    return mask1, dat1_msk, idx21

# -----------------------------------------------------------------------------
def ell_radius( lat, h=0, radians=False, a=a_wgs84, c=c_wgs84 ) :
    """
    This function calculates the radius of the Earth at a given latitude 
    based on the WGS84 ellipsoid model.

    Parameters:
    - lat (float or np.array): Latitude at which the radius is to be calculated. 
        It can be a single value or a numpy array of values. 
        The latitude should be in degrees by default. 
        If the latitude is in radians, the 'radians' parameter should be set to True.

    - radians (bool, optional): A flag to indicate if the input latitude is in radians. 
        Default is False, which means the function expects the latitude in degrees. 
        If this is set to True, the function will treat the input latitude as radians.

    The WGS84 ellipsoid model is defined by the semi-major axis (a_wgs84), 
    and the semi-minor axis (c_wgs84). 
    The function calculates the radius (R) at the given latitude, 
    using the formula for the radius of curvature in the plane of the meridian.

    Returns:
    - R (float or np.array): The radius of the Earth at the given latitude(s) 
        based on the WGS84 ellipsoid model. 
        The radius is returned in the same units as the semi-major and semi-minor axes.
    """

    if radians is False :
        lat = np.copy( np.deg2rad( lat ) )
        
    num = ( ( a**2 * np.cos( lat ) )**2 ) + ( ( c**2 * np.sin( lat ) )**2 )   
    den = ( ( a * np.cos( lat ) )**2 ) + ( ( c * np.sin( lat ) )**2 )  
    
    R = np.sqrt( num / den ) + h
    
    return R

# -----------------------------------------------------------------------------
def local_sph_raduis(lat):
    """
    Radius of the local sphere (based on latitude)

    Ref:
    - http://www2.ing.unipi.it/~a009220/lezioni/LI_ING_EDILE/AA1011/MATERIALE_DIDATTICO/APPUNTI/Geodesia.pdf

    """

    lat_r = np.radians( lat )
    
    e2 = ( a_wgs84**2 - c_wgs84**2 ) / a_wgs84**2
    
    N = a_wgs84 / np.sqrt( 1 - e2 * np.sin( lat_r )**2 )
    
    rho = a_wgs84 * ( 1 - e2 ) / np.sqrt( ( 1 - e2 * np.sin( lat_r )**2 )**3 )
    
    R = np.sqrt( rho * N )

    return R

# -----------------------------------------------------------------------------
def m2deg(md, n_digits=9, R=R_wgs84, lat=None):
    """
    Convert metric distances to decimal degrees (spherical approximation)

    lat = if not None, it calculate the distance using the Local Sphere Approximation,
            tangent to the wgs84 ellipsoid at the given latitude
    """

    if lat is None:   
        radi = R
    else:   
        radi = local_sph_raduis(lat)

    dd = np.round( md*360 / ( 2 * np.pi * radi ), n_digits)

    return dd

# -----------------------------------------------------------------------------
def deg2m(dd, lat=None, n_digits=3, R=R_wgs84):
    """
    Convert deg distances to meters (spherical approximation)

    lat = if not None, it calculate the distance using the Local Sphere Approximation,
            tangent to the wgs84 ellipsoid at the given latitude
    """

    if lat is None:   
        radi = R
    else:   
        radi = local_sph_raduis(lat)

    md = np.round( dd * 2 * np.pi * radi / 360, n_digits )

    return md

# -----------------------------------------------------------------------------
def pad_array( array, padw=25, mode='edge', alpha=None, 
               constant_values=np.nan,
               plot=False, vmin=None, vmax=None, iter=1, 
               ptype='percentage', sqr_area=False, 
               equal_shape=False, smooth=0 ):
    
    ny, nx = array.shape
    
    if type( padw ) in ( int , float ) :
        padw = ( padw, padw ) 
    
    if ptype == 'percentage' :
        padw = [ int( ny * padw[0] / 100 ), int( nx * padw[1] / 100 ) ]
        
    if sqr_area == True:
        if ny > nx : padw[1] = padw[1] + ny - nx
        if ny < nx : padw[0] = padw[0] + nx - ny

    if equal_shape:
        diff = abs(ny - nx)
        if ny > nx:
            padw[1] += diff // 2
            padw[0] += diff - diff // 2
        elif nx > ny:
            padw[0] += diff // 2
            padw[1] += diff - diff // 2
    
    if mode in ( 'surface', 'gdal', 'thin_plate' ): 
        pad_array = np.pad( array, pad_width=( ( padw[0], padw[0] ), 
                            ( padw[1], padw[1] ) ), 
                            mode='constant', constant_values=np.nan )
        pad_array = fillnan( pad_array, method=mode, 
                             iter=iter, smooth=smooth )

    elif mode == 'constant' : 
        pad_array = np.pad( array, pad_width=((padw[0], padw[0]), 
                            (padw[1], padw[1])), 
                            mode=mode, constant_values=constant_values )        
    else: 
        pad_array = np.pad( array, pad_width=((padw[0], padw[0]), 
                           (padw[1], padw[1])), 
                            mode=mode )
    
    pnx, pny = pad_array.shape
    y0, y1, x0, x1 = (padw[0], pnx - padw[0], padw[1], pny - padw[1])
    original_shape_indx = (y0, y1, x0, x1)
        
    if alpha is not None:
        pad_array = taper( pad_array, alpha=alpha )[0]
        
    if plot == True:
        plta( array, sbplt=[1,2,1], vmin=vmin, vmax=vmax )
        plta( pad_array, sbplt=[1,2,2], vmin=vmin, vmax=vmax )
        
    return pad_array, original_shape_indx

# -----------------------------------------------------------------------------
def pad_xx_yy( pad_arr, xx, yy, plot=False ):
    
    dx = abs( np.mean(np.diff( xx, axis=1 ) ) )
    dy = abs( np.mean(np.diff( yy, axis=0 ) ) )
    
    a, c = pad_arr[0], pad_arr[1]
    
    xmin1, xmax1 = np.min(xx), np.max(xx)
    ymin1, ymax1 = np.min(yy), np.max(yy)
    xmin2, xmax2 = xmin1 - dx * c[2], xmax1 + dx * (a.shape[1] - c[3])
    ymax2, ymin2 = ymax1 + dy * c[0], ymin1 - dy * (a.shape[0] - c[1])

    xv = np.linspace(xmin2, xmax2, a.shape[1])
    yv = np.linspace(ymax2, ymin2, a.shape[0])
    xxn, yyn = np.meshgrid( xv, yv )
    
    if plot == True:
        plt.figure()
        plt.scatter((xxn.flatten()), (yyn.flatten()), c='b')
        plt.scatter((xx.flatten()), (yy.flatten()), c='r')
        
    return xxn, yyn
    
# -----------------------------------------------------------------------------
def taper( array, plot=False, vmin=None, vmax=None ):
    
    nx = array.shape[1]
    ny = array.shape[0]

    # Create tapering windows using numpy.hanning
    tape_x = np.hanning(nx)
    tape_y = np.hanning(ny)
    
    t_xx, t_yy = np.meshgrid(tape_x, tape_y)
    taper_filt = t_xx * t_yy
    tarray = taper_filt * (array - np.nanmean(array))
    tarray = tarray + np.nanmean(array)
    
    if plot:
        plt.subplot(1, 3, 1)
        plt.imshow(array, vmin=vmin, vmax=vmax)
        plt.title('Original Array')
        
        plt.subplot(1, 3, 2)
        plt.imshow(taper_filt, vmin=vmin, vmax=vmax)
        plt.title('Taper Filter')
        
        plt.subplot(1, 3, 3)
        plt.imshow(tarray, vmin=vmin, vmax=vmax)
        plt.title('Tapered Array')
        
        plt.show()
    
    return tarray, taper_filt

# -----------------------------------------------------------------------------
def crop_pad(array, original_shape_idx, plot=False, vmin=None, vmax=None):
    
    array_crop = array[original_shape_idx[0]:original_shape_idx[1],
    original_shape_idx[2]:original_shape_idx[3]]
    
    if plot == True:
        
        plta( array, sbplt=[1,2,1], vmin=vmin, vmax=vmax )
        plta( array_crop, sbplt=[1,2,2], vmin=vmin, vmax=vmax )
        
    return array_crop

# -----------------------------------------------------------------------------
def xy2lim( x, y, prjcode_in=4326, prjcode_out=4326, extend=False,
            method='distance', sqr_area='False', 
            plot=False, round_val=None, p=False ):

    if prjcode_in != prjcode_out:
        x,y = prjxy(prjcode_in, prjcode_out, x, y)

    lim = [ np.min(x), np.max(x), np.min(y), np.max(y) ]

    if extend is not None or extend is not False:
        lim = extend_lim( lim, extend, method, sqr_area )

    if round_val is not None or round_val is not False:

        if round_val == True :
            lim = [np.floor(lim[0]), np.ceil(lim[1]), np.floor(lim[2]), np.ceil(lim[3])]

        if type( round_val ) in ( int, float ) :

            # Shift the decimal point, round, and shift back
            lim = [lim[0] * 10**round_val, lim[1] * 10**round_val, 
                   lim[2] * 10**round_val, lim[3] * 10**round_val]
            lim = [np.floor(lim[0]) / 10**round_val, np.ceil(lim[1]) / 10**round_val, 
                   np.floor(lim[2]) / 10**round_val, np.ceil(lim[3]) / 10**round_val]
    
    if p == True :
        print( "Xmin  Xmax Ymin Ymax :" )
        print( lim[0], lim[1], lim[2], lim[3] )

    if plot is True:

        plt.close("XY limits")
        plt.figure("XY limits")

        xplot = [lim[0], lim[0], lim[1], lim[1], lim[0]]
        yplot = [lim[2], lim[3], lim[3], lim[2], lim[2]]
        plt.scatter(x, y, marker='+')
        plt.plot(xplot, yplot, c='r')

        plt.show()

    return lim

# -----------------------------------------------------------------------------
def absolute_file_paths( directory ):

    path = os.path.abspath(directory)
    
    return [entry.path for entry in os.scandir(path) if entry.is_file()]

# -----------------------------------------------------------------------------
def array2step( arrays, axis=0 ) :

    if type( arrays ) != list :
        arrays = [arrays]
        
    if type( axis ) != list :
        axis = [axis]
        
    steps = []
    
    for i, a in zip( arrays, axis ):
        
        if len( i.shape ) == 1 :
            data = i.ravel()
            step = np.nanmean( np.diff( np.unique( data ) ) )
            
        if len( i.shape ) == 2 :    
            
            if a == 0 :
                data = i[ :, 0 ]
            if a == 1 :
                data = i[ 0, : ]            
            step = np.nanmean( np.diff( np.unique( data ) ) ) 
                
        if len( i.shape ) == 3 :    
            
            if a == 0 :
                data = i[ :, 0, 0 ]
            if a == 1 :
                data = i[ 0, :, 0 ]  
            if a == 2 :
                data = i[ 0, 0, : ] 
                                
            step = np.nanmean( np.diff( np.unique( data ) ) ) 
                
        steps.append( step )

    return steps

# -----------------------------------------------------------------------------
def plta( array, 
          vmin=None, 
          vmax=None, 
          tit=None, 
          lim=None, 
          stat=6, 
          sbplt=[],
          cmap='rainbow', 
          ax=None, 
          axis=False, 
          new_fig=True, 
          contours=[], 
          adjst_lim=True,
          flipud=False, 
          hillshade=False, ve=2, 
          aspect='auto', 
          blend_mode='overlay',
          mask=None, 
          points=None, 
          pc='k', ps=1, 
          clabel=None, 
          label_size='large',
          xlabel=None, 
          ylabel=None, 
          x_ax=True, 
          y_ax=True, 
          letter=None, 
          xlett=0, 
          ylett=0, 
          colorbar=True, 
          print_stat=True, 
          alpha=1, 
          lines=None,
          lc='k',
          lett_size='large', 
          lett_colour='k', 
          cc=None, 
          out_lim=None,
          cl=True, 
          resemp_fac=None,
          light_source=[ 315, 45 ], 
          spl_order=3, 
          place_colorbar=['right', 0.01, 0.07, 0.025, 0.80],
          place_clabel=None,
          cmap_fac=1, 
          figsize=None ) :
    
    """
    Plot 2D array and print statistics
    """
    print( place_colorbar)
    array = np.copy( array )
    
    if mask is not None :
        array[ mask ] = np.nan
        
    Min, Max, Mean, Std = np.nanmin(array), np.nanmax(array), \
                          np.nanmean(array), np.nanstd(array) 

    cmap = plt.cm.get_cmap( cmap )

    if cmap_fac != 1 :
        cmap = modify_colormap( cmap, cmap_fac )

    if stat != None:
        if type(tit) == int : stat = tit
        Min, Max, Mean, Std = np.round(Min, stat), np.round(Max, stat), \
                              np.round(Mean, stat), np.round(Std, stat) 
                              
        if print_stat == True :
        
            print(f"Min:{Min} Max:{Max} Mean:{Mean} Std:{Std}")

    if sbplt != [] :
        r, c, n = sbplt
    else :
        r, c, n = 1, 1, 1
        
    if new_fig :
        plt.figure( figsize=figsize )  
        
    if sbplt != []:
        plt.subplot( r, c, n )
        
    if vmin == None:
        vmin = Mean - 2 * Std
    if vmax == None:
        vmax = Mean + 2 * Std

    origin = 'upper'

    if lim != None :
        
        lim = copy.copy( lim )
        
        if len( lim ) == 2 :
            lim = xy2lim( x=lim[0], y=lim[1] )

        dx = np.abs( ( lim[1] - lim[0] ) / array.shape[1] ) 
        dy = np.abs( ( lim[3] - lim[2] ) / array.shape[0] ) 
        
        if adjst_lim == True :
            lim[1] = lim[1] + dx/2
            lim[3] = lim[3] + dy/2
            lim[0] = lim[1] - dx * array.shape[1] 
            lim[2] = lim[3] - dy * array.shape[0] 

    else :
        dx, dy = 1, 1
        
    if flipud == True :
        array = np.flipud( array )

    if resemp_fac is not None :
        array = resampling( array, resemp_fac, spl_order=spl_order, 
                            dimention='2D', mode='nearest' )

    if not ax :
        ax = plt.gca()

    if hillshade is False :
        ax.imshow( array, vmin=vmin, vmax=vmax, 
                   aspect=aspect, cmap=cmap, extent=lim,
                   alpha=alpha, origin=origin )
    
    if hillshade is True :
        # Normalize the array using vmin and vmax
        array = (array - vmin) / (vmax - vmin)
        cmap_instance = cm.get_cmap(cmap)  # Convert colormap name to colormap instance
        ls = LightSource( azdeg=light_source[0], altdeg=light_source[1] )
         # vmin and vmax are 0 and 1 after normalization
        rgb = ls.shade( array, cmap=cmap_instance, 
                        blend_mode=blend_mode, 
                        vert_exag=ve,
                        vmin=0, vmax=1, dx=1, dy=1 )  
         # vmin and vmax are 0 and 1 after normalization
        ax.imshow( rgb, extent=lim, alpha=alpha, 
                         aspect=aspect, vmin=0, vmax=1, 
                         cmap=cmap, origin=origin )  

        alpha2 = np.isnan( array )
        rgb[ alpha2 ] = np.nan

    plt.xlabel( xlabel, fontsize=label_size )
    plt.ylabel( ylabel, fontsize=label_size )
    
    if letter is not None :
        plt.annotate(letter, xy=(xlett,ylett), xycoords='axes fraction', size=lett_size,
                     c=lett_colour ) 

    if colorbar is True :  
        
        box = ax.get_position()

        left = box.x0 
        bottom = box.y0 
        width = box.width
        height = box.height 

        if len( place_colorbar ) == 4 :
            if place_colorbar[0] in ( 'bottom', 'top' ) :
                place_colorbar.insert( -1, 1 )
            if place_colorbar[0] in ( 'left', 'right' ) :
                place_colorbar.append( 1 )

        if place_colorbar[0] == 'bottom' :
            orientation = 'horizontal'
            left = box.x0 + place_colorbar[1]
            bottom += - place_colorbar[2]
            height = height * place_colorbar[4]
            width = width * place_colorbar[3]
            ax.set_position([box.x0 + width, box.y0 + height, 
                             box.width - width, box.height - height])
        elif place_colorbar[0] == 'top' :
            orientation = 'horizontal'
            bottom = box.y0 + box.height + place_colorbar[2]
            height = height * place_colorbar[4]
            ax.set_position([box.x0, box.y0, 
                             box.width, box.height - height])
        elif place_colorbar[0] == 'left' :
            orientation = 'vertical'
            left += - place_colorbar[1]
            width = width * place_colorbar[3]
            ax.set_position([box.x0 + width, box.y0, 
                             box.width - width, box.height])
        elif place_colorbar[0] == 'right' :
            orientation = 'vertical'
            left = box.x0 + box.width + place_colorbar[1]
            bottom = box.y0 + place_colorbar[2]
            height = height * place_colorbar[4]
            width = width * place_colorbar[3]
            ax.set_position([box.x0, box.y0, 
                             box.width - width, box.height])

        # Create the new axes
        cax = plt.gcf().add_axes([left, bottom, width, height])
        
        # Determine the orientation of the colorbar
        if place_colorbar[0] in ['bottom', 'top']:
            orientation = 'horizontal'
        else:
            orientation = 'vertical'

        # Create a ScalarMappable with the same colormap and normalization as array
        sm = cm.ScalarMappable( cmap=cmap, 
                                norm=plt.Normalize(vmin=vmin, vmax=vmax) )
        # Use the ScalarMappable to create the colorbar
        clb = plt.colorbar( sm, cax=cax, orientation=orientation )

        if clabel is not None :

            if not place_clabel :
                place_clabel = [-0.5, 1.05] 
            
            plt.text( place_clabel[0], place_clabel[1], clabel, 
                      fontsize = label_size, 
                      transform = clb.ax.transAxes )
        
    if points is not None :
        ax.scatter( points[0], points[1], c=pc, s=ps )

    if lines is not None :
        for line in lines :  
            ax.plot( line[0], line[1], c=lc)
        ax.set_xlim( lim[0], lim[1] )
        ax.set_ylim( lim[2], lim[3] )
    
    if contours != [] :
        # smoothed_array = sp.ndimage.gaussian_filter(np.ma.masked_invalid( np.flipud(array)), sigma=3)
        cont = ax.contour( array, 
                           levels=contours, extent=lim, colors=cc,
                           linestyles='solid', origin=origin)  
        
        if cl is True :
            plt.clabel( cont )
        
    if place_colorbar[0] == 'bottom':
        print('ok')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
    
    if place_colorbar[0] == 'left':
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    
    if axis is False:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    if x_ax is False:
        ax.axes.xaxis.set_visible(False)

    if y_ax is False:
        ax.axes.yaxis.set_visible(False)
        
    if type(tit) == str:
        ax.set_title(tit)
    if type(tit) == int:
        Min, Max, Mean, Std = np.round(Min, tit), np.round(Max, tit), \
                              np.round(Mean, tit), np.round(Std, tit)
        tit = f"[ Min={Min}  Max={Max}  Mean={Mean}  Std={Std} ]"
        ax.set_title(tit)
        
    if ( out_lim is not None ) and ( out_lim != [] ) :
        ax.set_xlim( [ out_lim[0], out_lim[1] ] )
        ax.set_ylim( [ out_lim[2], out_lim[3] ] )

    return ax
        
# -----------------------------------------------------------------------------
def map_profile( xy_start, xy_end, X, Y, Z=[], step=None, 
                 yaxis_sep=50, method='nearest',
                 colors=None,
                 legend_labels=None,
                 legend_label_size=None,
                 markers=None,
                 linestyle=None,
                 y_labels=None,
                 legend_ncol=1,
                 legend_loc='best',
                 x_label=None,
                 ax_label_size=None,
                 font='sans-serif',
                 y_ax_lim=None,
                 x_ax_lim=None,
                 plot=True,
                 subplot=(1, 1, 1),
                 text=None, text_size=14,
                 text_boxstyle='round',
                 text_facecolor='wheat',
                 text_alpha=0.5,
                 text_position=(0.95, 0.05),
                 xaxis_visible=True,
                 prjcode_in=4326,
                 smooth=1,
                 prjcode_out=None,
                 m2km=False,
                 deg2km=False,
                 prjcode_limits=None,
                 x_axis = True,
                 plot_map=False,
                 vmin=None,
                 vmax=None,
                 y_axis_type='multiple',
                 new_fig=True):

    from matplotlib import rcParams

    rcParams['font.family'] = font
    if prjcode_out is None:
        prjcode_out = prjcode_in

    if prjcode_limits is None:
        prjcode_limits = prjcode_in

    if prjcode_out != prjcode_in:
        X, Y = prjxy(prjcode_in, prjcode_out, X, Y)

    if prjcode_limits != prjcode_out:
        xy_start = prjxy(prjcode_limits, prjcode_out, xy_start[0], xy_start[1])
        xy_end = prjxy(prjcode_limits, prjcode_out, xy_end[0], xy_end[1])

    print( 'line step:', step )
    if step is None:
        step = stepxy( X, Y )[2]
        print(step)
    x0, x1 = xy_start[0], xy_end[0]
    y0, y1 = xy_start[1], xy_end[1]
    theta = np.arctan((y1 - y0) / ((x1 - x0)+1e-24) )
    length = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    lp = np.arange(0, length, step/smooth)
    xp = x0 + lp * np.cos(theta)
    yp = y0 + lp * np.sin(theta)
    
    if m2km is True :
        lp /= 1000
        
    if deg2km is True :
        lp = deg2m(lp)/1000  
        
    profiles = []
    for i, arr in enumerate(Z):
#        arr[np.isnan(arr)]=0
        lim = extend_lim( [ np.min((x0,x1)), np.max((x0,x1)), 
                            np.min((y0,y1)), np.max((y0,y1)) ], 2, sqr_area=True )
        xl, yl, idx = xy_in_lim( X, Y, lim )
        al = arr[idx]
        profiles.append( xyz2xy( (xl.ravel(), yl.ravel(), al.ravel()),
                                (xp, yp), method=method ) )
        
    if plot is True:
        if new_fig is True :
            plt.figure()
        if colors is None:
            colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(profiles))]
        if legend_labels is None:
            legend_labels = ['function_' + str(i) for i in range(len(profiles))]
        if markers == 'random':
            markers_rand = itertools.cycle( ( ',', '+', '.', '*', '<', '>', 'h', 'p', 's',
                                              '.', 'H', 'D' ) )
        if linestyle is None :
            linestyle = ['solid' for i in range( len( profiles) ) ]
            
        lns = []
        ax_orig = plt.subplot(subplot[0], subplot[1], subplot[2])
        
        for i, (arr, color, lnsty) in enumerate(zip(profiles, colors, linestyle)):         
            if ( i == 0 ) or ( y_axis_type == 'unique') :
                ax = ax_orig
            else:
                ax = ax_orig.twinx()
                ax.spines['right'].set_position(('outward', yaxis_sep * (i - 1)))
            if markers == 'random' :
                ln = ax.plot(lp, arr, color=color, marker=(next(markers_rand)), linestyle=lnsty)
            else:
                ln = ax.plot(lp, arr, color=color, linestyle=lnsty )
            if y_axis_type == 'unique' :    
                ax.tick_params(axis='y', colors='k')
            else :
                ax.tick_params(axis='y', colors=color )
            if y_labels is not None:
                plt.ylabel(y_labels, fontsize=ax_label_size)
            if x_label is not None:
                plt.xlabel(x_label, fontsize=ax_label_size)
            lns += ln
            if y_ax_lim is not None :
                ax.set_ylim( y_ax_lim[i] )      
            if x_ax_lim is not None :
                ax.set_xlim( x_ax_lim[i] )                          

        if text is not None:
            props = dict(boxstyle=text_boxstyle, facecolor=text_facecolor, alpha=text_alpha)
            ax_orig.annotate(text, xy=text_position, xycoords='axes fraction',
              fontsize=text_size,
              bbox=props)
            plt.tight_layout()

        if legend_labels is not None:
            ax_orig.legend(lns, legend_labels, fontsize=legend_label_size, 
                                 ncol=legend_ncol, loc=legend_loc, framealpha=1)
            ax_orig.axes.get_xaxis().set_visible(xaxis_visible)
    
        if x_axis is False :
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            
    if plot_map == True:
        plt.figure()
        if vmin == None:
            vmin = np.nanmean(Z[0]) - 2 * np.nanstd(Z[0])
        if vmax == None:
            vmax = np.nanmean(Z[0]) + 2 * np.nanstd(Z[0])
        plt.imshow(Z[0], vmin=vmin, vmax=vmax, cmap='rainbow',
                   extent=(X.min(), X.max(), Y.min(), Y.max()))
        plt.plot(xp, yp, color='k')            
    
    pr_list = []
    for i in profiles:
        pr_list.append(np.column_stack((xp, yp, lp, i)))

#    plt.tight_layout()

    return pr_list

# -----------------------------------------------------------------------------
def XYZ_crop2lim( X, Y, Z, lim ) :

    if len( Z.shape ) == 2:

        xi = np.where( ( X[0, :] >= lim[0]) & (X[0, :] <= lim[1] ) )
        yi = np.where( ( Y[:, 0] >= lim[2]) & (Y[:, 0] <= lim[3] ) )

        Xc = X[ np.min(yi):np.max(yi), np.min(xi):np.max(xi) ]
        Yc = Y[ np.min(yi):np.max(yi), np.min(xi):np.max(xi) ]
        Zc = Z[ np.min(yi):np.max(yi), np.min(xi):np.max(xi) ]
        

        return [ Xc, Yc, Zc ]

    if len( Z.shape ) == 1:

        xi = np.where( ( X >= lim[0] ) & ( X <= lim[1] ) )
        yi = np.where( ( Y >= lim[2] ) & ( Y <= lim[3] ) )

        xc = X[ ( xi, yi ) ]
        yc = Y[ ( xi, yi ) ]
        zc = Z[ ( xi, yi ) ]

        return [ xc, yc, zc ]

# -----------------------------------------------------------------------------
def isiterable(p_object):
    """
    Check if an object is iterable
    """

    try:
        it = iter(p_object)
    except TypeError:
        return False
    
    return True

# -----------------------------------------------------------------------------
def stepxy( xarray, yarray ) :
    
    " Distance between 2Darray grid data points "
    
    cx = int( xarray.shape[1]/2 )
    cy = int( yarray.shape[0]/2 )
    
    arx = xarray[ cy:cy+2, cx:cx+2 ]
    ary = yarray[ cy:cy+2, cx:cx+2 ]
    
    dx = np.sqrt( ( arx[0,1] - arx[0,0] )**2 + ( ary[0,1] - ary[0,0] )**2 )
    dy = np.sqrt( ( arx[0,0] - arx[1,0] )**2 + ( ary[0,0] - ary[1,0] )**2 )
    
    dmxy = np.mean( ( dx, dy ) )
        
    return dx, dy, dmxy

# -----------------------------------------------------------------------------
def gmt_surf( x, y, z, 
              grid_step=None, 
              lim=None, 
              blkm=True,
              pause=False, 
              filt='', 
              max_iterations=100, 
              tension_factor=0.35, 
              remove_files=True, 
              decimals=4,
              plot=False, vmin=None, vmax=None ): 
    
    """
    Generate a GMT surface plot using given x, y, and z data.
    To use this function, GMT 6 sotware MUST BE installed on your system.
    https://docs.generic-mapping-tools.org/latest/install.html

    Args:
        - x (array-like): The x-coordinates of the data points.
        - y (array-like): The y-coordinates of the data points.
        - z (array-like): The z-values of the data points.
        - gstep (float, optional): The grid step size. If not provided, it will be calculated based on the mean distance between data points.
        - lim (tuple, optional): The limits of the plot in the form of (x0, x1, y0, y1). If not provided, it will be calculated based on the minimum and maximum values of x and y.
        - grid_shape (list, optional): The shape of the grid in the form of [num_rows, num_columns]. If not provided, it will be calculated based on the lim and gstep values.
        - pause (bool, optional): Whether to pause after running the GMT commands. Default is False.
        - filt (str, optional): The filter to apply to the surface. Default is an empty string.
        - convergence_limit (str, optional): The convergence limit for the surface generation. Default is an empty string.
        - max_iterations (int, optional): The maximum number of iterations for the surface generation. Default is 100.
        - max_radius (float or str, optional): The maximum radius for the surface generation. Default is an empty string.
        - tension_factor (float, optional): The tension factor for the surface generation. Default is 0.35.
        - remove_files (bool, optional): Whether to remove intermediate files generated during the process. Default is True.
        - plot (bool, optional): Whether to plot the generated surface. Default is False.
        - vmin (float, optional): The minimum value for the color scale of the plot. Default is None.
        - vmax (float, optional): The maximum value for the color scale of the plot. Default is None.

    Returns:
        - list: A list containing the x-coordinates, y-coordinates, 
          and z-values of the generated surface.
    
    References :
        - GMT 6: Wessel, P., Luis, J. F., Uieda, L., Scharroo, 
          R., Wobbe, F., Smith, W. H. F., & Tian, D. (2019). 
          The Generic Mapping Tools version 6. Geochemistry, 
          Geophysics, Geosystems, 20, 5556-5564. 
          https://doi.org/10.1029/2019GC008515.
        
        - Smith, W. H. F, and P. Wessel, 1990, 
          Gridding with continuous curvature splines in tension, 
          Geophysics, 55, 293-305, 
          https://doi.org/10.1190/1.1442837.
    """

    # Remove NaN values from the data
    nan = np.isnan(z)
    x, y, z = x[(~nan)], y[(~nan)], z[(~nan) ]

    # If no limits are provided, calculate them based on the data
    if lim is None:
        lim = ( np.min(x), np.max(x), np.min(y), np.max(y) )
    x0, x1, y0, y1 = lim

    # Round the limits to avoid floating point precision issues
    x0, x1 = round(x0, decimals), round(x1, decimals)
    y0, y1 = round(y0, decimals), round(y1, decimals)

    # Calculate the mean distance between data points
    mdist = min_dist(x, y)['mean']
    if ( grid_step is None ) :
        grid_step = np.round( mdist, decimals )
        print( 'gstep : ', grid_step )
    
    grid_step = np.round( grid_step, decimals )

    # Calculate the dimensions of the grid
    x_dim = int(np.ceil((x1 - x0) / grid_step))
    y_dim = int(np.ceil((y1 - y0) / grid_step))

    while np.gcd(x_dim, y_dim) <= 1:
        # Increment x_dim and/or y_dim
        x_dim += 1
        y_dim += 1

    x1 = x0 + x_dim * grid_step
    y1 = y0 + y_dim * grid_step

    # Define the grid shape
    grid_shape = [x_dim, y_dim]

    # Define the grid step for the GMT surface command
    gs = f"-I{np.round(grid_step, decimals)}+e"

    if blkm is True :
        blkm = gs
    elif type( blkm ) in ( int, float ) :
        blkm = f"-I{blkm}"

    # Define the limits for the GMT commands
    Rlim = f"-R{np.round(x0,decimals)}/{np.round(x1,decimals)}/{np.round(y0,decimals)}/{np.round(y1,decimals)}"
    xyz_temp = np.column_stack((x, y, z))

    np.savetxt('xyz_temp', xyz_temp, fmt='%f')

    # Define the block mean command based on the block mean parameter
    # if blkm != '':
    blkm = f"gmt blockmean xyz_temp {blkm} {Rlim} > xyz_temp_blkm"

    # Define the pause command based on the pause parameter
    if pause == True:
        pause = 'pause'
    if pause == False:
        pause = ''

    # Define the input file name based on whether block mean is used
    if ( blkm is not None ) or ( blkm is not False ) : 
        in_file = 'xyz_temp_blkm'
    else :
        blkm = ''
        in_file = 'xyz_temp'

    # Define the GMT surface command
    surface = f"gmt surface {in_file} {gs} {Rlim} " \
              f"-Gsurf_temp.grd -T{tension_factor} -N{max_iterations}"

    if filt != '':
        filt = f"gmt grdfilter surf_temp.grd -Fg{filt} -Gsurf_temp.grd -Dp"

    # Define the script extension and prefix based on the operating system
    if platform.system() == 'Linux' :
        ext = '.sh'
        pre= './'
    else :
        ext = '.bat'
        pre = ''

    # Write the GMT commands to a script file
    gmt_script = 'run_temp' + ext
    with open( gmt_script, 'w') as rsh:
        gmt_code = blkm + '\n' +\
                   surface + '\n' +\
                   filt + '\n' +\
                   'gmt grd2xyz surf_temp.grd > surf_temp.xyz\n' +\
                   pause + '\n'
        rsh.write( gmt_code )
        rsh.close()
    rsh.close()

    # Run the script
    print( gmt_code )
    os.system( pre + gmt_script )
    xyz = np.loadtxt('surf_temp.xyz')

    # Remove intermediate files if requested
    if remove_files == True :
        os.remove( gmt_script )
        os.remove('xyz_temp')
        if os.path.exists("xyz_temp_blkm"):
          os.remove("xyz_temp_blkm")
        os.remove('surf_temp.grd')
        os.remove('surf_temp.xyz')
        os.remove('gmt.history')

    # Extract the x, y, and z values from the output
    x = np.unique(xyz[:, 0])
    y = np.unique(xyz[:, 1])
    zz = np.reshape(xyz[:, 2], (len(y), len(x)))
    xx = np.reshape(xyz[:, 0], (len(y), len(x)))
    yy = np.reshape(xyz[:, 1], (len(y), len(x)))
    
    # Plot the output if requested
    if plot == True :
        plt.close("GMT surface")
        plt.figure("GMT surface")
        plta( zz, vmin=vmin, vmax=vmax, 
            lim=(xx, yy), axis=True )

    return [xx, yy, zz]

# -----------------------------------------------------------------------------
def grid_fill( data, invalid=None ):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """

    if invalid is None: 
        invalid = np.isnan(data)

    ind = sp.ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    
    return data[ tuple(ind) ]

# -----------------------------------------------------------------------------
def mask_edges( mask, plot=False, c='k', s=None, plt_array=None, vmin=None, vmax=None ) :
    
    """
    Returne x-y coordinates of the edges of a "mask-array" ( boolean or 0/1 array )
    
    0 0 0 0 0 0 0     0 0 0 0 0 0 0     ( (1,1)
    0 1 1 1 1 1 0     0 1 1 1 1 1 0       (1,2)
    0 1 1 1 1 1 0     0 1 0 0 0 1 0       (1,3) 
    0 1 1 1 1 1 0 ==> 0 1 0 0 0 1 0 ==>   (1,4)
    0 1 1 1 1 1 0     0 1 0 0 0 1 0       (1,5)
    0 1 1 1 1 1 0     0 1 1 1 1 1 0       (2,1)
    0 0 0 0 0 0 0     0 0 0 0 0 0 0       (2,5)... )
    """
    
    if type( mask ) == bool :
        mask = mask*1
    
    fil = [[-1,-1,-1],
           [-1, 8,-1],
           [-1,-1,-1]]
    
    edges = np.where( sp.ndimage.convolve(mask*1, fil, mode='constant') > 1)
    
    x_ind, y_ind = edges[1], edges[0] 
        
    if plot is True :
        if plt_array is not None :
            plt.imshow( plt_array, vmin=vmin, vmax=vmax, cmap='rainbow')
            plt.colorbar()
        plt.scatter( x_ind, y_ind, c=c, s=s )
        
    return x_ind, y_ind    

# -----------------------------------------------------------------------------
def normalize( array, b=1, a=0 ) :
    
    Min = np.nanmin(  array )
    Max = np.nanmax( array )
    
    norm = ( b - a ) * ( ( array - Min ) / ( Max - Min ) ) + a
        
    return norm  

# -----------------------------------------------------------------------------
def fillnan( array, 
             xy=None, 
             method='nearest', 
             size=3, 
             iter=1,
             smooth=0, 
             maxSearchDist=None, 
             plot=False, 
             vmin=None, 
             vmax=None, 
             edges=False, 
             tension=0.35 ) :
    
    zz = np.copy( array )
    
    if np.all( np.isfinite( zz ) ):
        zzfn = zz
    
    if ( array.shape[1] > 1 ) and ( xy is None ):
        xi, yi = np.arange(0, zz.shape[1], 1), np.arange(0, zz.shape[0], 1)
        xx, yy = np.meshgrid( xi, yi )
        
    if xy is not None:
        xx, yy = xy[0], xy[1]
        
    zz_nan = np.isnan( zz )
    
    x, y, z = xx.flatten(), yy.flatten(), zz.flatten()
    nan = np.isnan(z)
    notnan = np.invert(nan)
    xn, yn = x[nan], y[nan]
    xi, yi, zi = x[notnan], y[notnan], z[notnan]
    
    if method == 'nearest':
        zfn = sp.interpolate.griddata( ( xi, yi), zi, (xn, yn), method='nearest')
        zn = np.copy( z )
        zn[nan] = zfn
        zzfn = zn.reshape( zz.shape )
        
    if method == 'nearest2D' :
        zzfn = grid_fill( array )
        
    if method == 'mean':
        zfn = np.nanmean( z )
        z[ np.isnan( z ) ] = zfn
        zzfn = z.reshape( zz.shape )
        
    if type( method ) in ('float', 'int'):
        zfn = method
        z[ np.isnan( z ) ] = zfn
        zzfn = z.reshape( np.shape( zz ) )
        
    if method == 'gdal':
        zz[ np.isnan( zz ) ] = 1e-24
        rx, ry, _ = stepxy( xx, yy )
        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create( '/vsimem/raster_filled.vrt', zz.shape[1], zz.shape[0],
                                 bands = 1, eType=gdal.GDT_Float32 )
    
        raster.SetGeoTransform( [ xx.min(), rx, 0.0, yy.max(), 0.0, -ry ] )
        raster.GetRasterBand( 1 ).SetNoDataValue( 1e-24 )
        
        out_band = raster.GetRasterBand( 1 )
        out_band.WriteArray( zz )
        
        if maxSearchDist is None: 
            maxSearchDist = int( np.max( zz.shape ) )
            
        gdal.FillNodata( targetBand = raster.GetRasterBand(1), 
                         maskBand=None, 
                         maxSearchDist=maxSearchDist,
                         smoothingIterations=iter )
        
        zzfn = raster.GetRasterBand( 1 ).ReadAsArray()
        zzfn[ zzfn == 1e-24 ] = np.nan
        raster = None

    if method == 'thin_plate':
        zfn = xyz2xy( ( xi, yi, zi ), (xn, yn), method='thin_plate', smooth=smooth )
        zfn = zfn( xn, yn )
        zn = np.copy( z )
        zn[nan] = zfn
        zzfn = zn.reshape( zz.shape )
        
    if method == 'surface':
        print( xx.shape )
        xxs, yys, zzs = gmt_surf( xx.ravel(), yy.ravel(), zz.ravel(),
                         tension_factor=tension )
        zzfn = xyz2xy( ( xxs, yys, zzs ), (xx, yy), method='nearest' )
        print( zzfn.shape )

    if edges is True :
        zzfn = filt_edges( zzfn, mask=~zz_nan*1, iter=iter, size=size, smooth=smooth )     
        
    if plot == True:
        plta( array, sbplt=[1, 3, 1], tit='original', vmin=vmin, vmax=vmax )
        plta( zzfn, sbplt=[1, 3, 2], tit='fill', vmin=vmin, vmax=vmax )
        plta( array - zzfn, sbplt=[1, 3, 3], tit='differences' )
        
    return zzfn

# -----------------------------------------------------------------------------
def geo_line_dist( x_line, y_line, prjcode_in=4326, prjcode_out=4326 ) :
    """
    Calculate the distance between consecutive points in a line.

    Parameters:
    -----------
    x_line (numpy.ndarray): Array of x-coordinates of the line.
    y_line (numpy.ndarray): Array of y-coordinates of the line.
    prjcode_in (int, optional): Input projection code. Defaults to 4326.
    prjcode_out (int, optional): Output projection code. Defaults to 4326.

    Returns:
    -----------
    numpy.ndarray: Array containing the relative distance, 
    cumulative distance, x-coordinates, and y-coordinates.
    """

    x_line = np.copy( x_line )
    y_line = np.copy( y_line )

    if prjcode_in != prjcode_out :
        x, y = prjxy( prjcode_in, prjcode_out, x_line, y_line )

    else :
        x, y = x_line, y_line  

    dx = np.diff(x)
    dy = np.diff(y)

    # Calculate relative distances
    rel_dist = np.hypot(dx, dy)
    rel_dist = np.insert(rel_dist, 0, 0)  # Add 0 at the beginning

    # Calculate cumulative distances
    cum_dist = np.cumsum(rel_dist)

    dist = np.column_stack( ( rel_dist, cum_dist, x, y ) )

    return dist

# -----------------------------------------------------------------------------
def sort_lines( xyzl, prjcode_in=None, prjcode_out=None, add_dist=True, 
                add_original_ord=True, line_c=3, x_c=0, y_c=1, 
                order_c=None, plot=False ) :

    """
    Sorts the input point cloud by line number and distance along the line.

    Parameters:
    -----------
    xyzl : numpy.ndarray
        Input point cloud with columns for x, y, z, and line number.
    prjcode_in : int or str, optional
        Input projection code. Default is None.
    prjcode_out : int or str, optional
        Output projection code. Default is None.
    add_dist : bool, optional
        If True, adds a column for distance along the line. Default is True.
    add_original_ord : bool, optional
        If True, adds a column for the original order of the points. Default is True.
    line_c : int, optional
        Column index for the line number. Default is 3.
    x_c : int, optional
        Column index for the x coordinate. Default is 0.
    y_c : int, optional
        Column index for the y coordinate. Default is 1.
    order_c : int, optional
        Column index for the order of the points. Default is None.
    plot : bool, optional
        If True, plots the sorted lines. Default is False.

    Returns:
    --------
    new_xyzl : numpy.ndarray
        Sorted point cloud with columns for x, y, z, line number, 
        distance along the line, and original order.
    """

    # Copy the input point cloud to avoid modifying the original
    xyzl = np.copy( xyzl )

    # Get the unique line numbers
    lines = np.unique( xyzl[:, line_c] )

    if order_c is None:

        # Add an index column to the input point cloud
        xyzl = np.column_stack( ( xyzl, np.arange(xyzl.shape[0] ) ) )

        # Initialize the output point cloud
        new_xyzl = np.empty( ( 0, xyzl.shape[1] + 1 ) )

        order_c = new_xyzl.shape[1] - 2

        # Loop through each line and add the distance column
        for i, l in enumerate(lines) :

            idx = xyzl[:, line_c] == l

            line = xyzl[idx]
            minx = np.min( line[:, x_c] )
            miny = np.min( line[:, y_c] )
            maxx = np.max( line[:, x_c] )
            maxy = np.max( line[:, y_c] )
            diffx = maxx - minx
            diffy = maxy - miny
            if diffx > diffy:
                idxs = np.argsort( line[:, x_c] )
            else:
                idxs = np.argsort( line[:, y_c] )
            line = line[idxs]

            # Calculate the distance of each point in the line
            dist = geo_line_dist( line[:, x_c], line[:, y_c],
                                prjcode_in=prjcode_in, prjcode_out=prjcode_out )[:, 1]

            # Add the distance column to the line and stack it to the output point cloud
            new_line = np.column_stack( ( line, dist ) )
            new_xyzl = np.vstack( ( new_xyzl, new_line ) )

        # Sort the output point cloud by distance and line number
        idx_s = np.lexsort( ( new_xyzl[:, -1], new_xyzl[:, line_c] ) )
        new_xyzl = new_xyzl[ idx_s ]

        # Remove the distance column if not specified
        if add_dist != True:
            new_xyzl = new_xyzl[:, :-1]

        # Remove the column with the original indices
        if add_original_ord != True:
            new_xyzl = np.delete(new_xyzl, order_c, axis=1)

    else:
        idx_s = np.argsort(  xyzl[:, order_c] )
        new_xyzl = xyzl[ idx_s ]

    # Plot the sorted lines if specified
    lines = np.unique(new_xyzl[:, line_c])

    if plot is True:
        plt.close('Sorted lines')
        plt.figure('Sorted lines')
        for l in lines:
            idx = new_xyzl[:, line_c] == l
            line = new_xyzl[idx]
            plt.plot(line[:, x_c], line[:, y_c], c='k')
        plt.scatter(new_xyzl[:, x_c], new_xyzl[:, y_c], c='r', s=1)

        plt.show()

    # Return the sorted point cloud and the indices of the original point cloud
    return new_xyzl

# -----------------------------------------------------------------------------
def join_lines(xyzl, dist_th=10000, angle_th=30, line_c=3, x_c=0, y_c=1, new_line_c=None):
    """
    Joins a set of lines into larger segments based on 
    the distance between the end of one line and the start of the next,
    and the angle between the lines.
    
    Parameters:
    -----------
        - xyzl : numpy.ndarray
            A numpy array containing the x, y, z coordinates of 
            the points and the line number they belong to.
        - dist_th : float, optional
            The maximum distance between 
            the end of one line and the start of 
            the next to consider them part of the same segment. Default is 1000.
        - angle_th : float, optional
            The maximum angle between the end of one line and 
            the start of the next to consider them part of the same segment. Default is 30 degrees.
        - line_c : int, optional
            The index of the column containing the line number in the input array. Default is 3.
        - x_c : int, optional
            The index of the column containing the x coordinate in the input array. Default is 0.
        - y_c : int, optional
            The index of the column containing the y coordinate in the input array. Default is 1.
        - new_line_c : int, optional
            The index of the column containing the new line number in the output array. 
            If None, the original line numbers are used. Default is None.
        
    Returns:
    --------
    xyzl_new : numpy.ndarray
        A numpy array containing the x, y, z coordinates of the points and the new line number they belong to.
    """

    # Make a copy of the input array
    xyzl = np.copy(xyzl)

    # Set the new line number column
    if new_line_c is None:
        new_line_c = line_c
    else:
        xyzl = np.column_stack((xyzl, np.ones(xyzl.shape[0]) * np.nan))

    # Initialize the output array
    i0 = xyzl[:, line_c] == xyzl[0, line_c]
    xyzl_new = xyzl[i0]
    xyzl_new[:, new_line_c] = 0

    # Get the unique line numbers
    lines = np.unique(xyzl[:, line_c])

    # Initialize the new line number
    new_line_num = 0

    # Function to calculate the angle between two vectors
    def calculate_angle(v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_v1, unit_v2)
        angle = np.arccos(dot_product)
        return np.degrees(angle)

    # Loop through each line
    for i in range(len(lines) - 1):

        # Get the current line and the next line
        line1 = xyzl[xyzl[:, line_c] == lines[i]]
        line2 = xyzl[xyzl[:, line_c] == lines[i + 1]]

        # Calculate the distance between the last point of the current line and the first point of the next line
        dist = np.sqrt((line1[-1, x_c] - line2[0, x_c])**2 + (line1[-1, y_c] - line2[0, y_c])**2)

        # Initialize angle to a low value
        angle = 0

        # Calculate the angle if both lines have more than one point
        if len(line1) > 1 and len(line2) > 1:
            v1 = line1[-1, [x_c, y_c]] - line1[-2, [x_c, y_c]]
            v2 = line2[1, [x_c, y_c]] - line2[0, [x_c, y_c]]
            angle = calculate_angle(v1, v2)

        # If the distance is less than the threshold and the angle is less than the threshold, join the line numbers
        if dist <= dist_th and (len(line1) == 1 or len(line2) == 1 or angle <= angle_th):
            # Set the new line number for the points in the current line
            line2[:, new_line_c] = new_line_num

        # If the distance is greater than the threshold or the angle is greater than the threshold, start a new line
        else:
            # Set the new line number for the points in the next line
            line2[:, new_line_c] = new_line_num + 1
            # Increment the new line number
            new_line_num += 1

        # Add the current line to the output array
        xyzl_new = np.vstack((xyzl_new, line2))

    return xyzl_new

# -----------------------------------------------------------------------------
def split_lines( xyzl, 
                 angle_th=45, 
                 step=250, 
                 dist_th=1000, 
                 n_points=0,
                 join_th=None, 
                 prjcode_in=None, 
                 prjcode_out=None, 
                 line_c=3, 
                 x_c=0, 
                 y_c=1, 
                 order_c='same', 
                 plot=False, 
                 new_line_c=None,  
                 new_xy=False, 
                 iter=1, 
                 size=None ) :

    """
    Splits a set of lines into smaller segments based on the angle between neighboring points and the distance between points.

    Parameters:
    -----------
    xyzl : numpy.ndarray
        A numpy array containing the x, y, z coordinates of the points and the line number they belong to.
    angle_th : float, optional
        The maximum angle between neighboring points to consider them part of the same segment. Default is 45.
    step : float, optional
        The distance between neighboring points in the resampled line. Default is 250.
    dist_th : float, optional
        The maximum distance between neighboring points to consider them part of the same segment. Default is 1000.
    n_points : int, optional
        The minimum number of points required for a segment to be included in the output. Default is 0.
    prjcode_in : int, optional
        The EPSG code of the input coordinate reference system. Default is None.
    prjcode_out : int, optional
        The EPSG code of the output coordinate reference system. Default is None.
    line_c : int, optional
        The index of the column containing the line number in the input array. Default is 3.
    x_c : int, optional
        The index of the column containing the x coordinate in the input array. Default is 0.
    y_c : int, optional
        The index of the column containing the y coordinate in the input array. Default is 1.
    order_c : str or None, optional
        The order in which to sort the lines before processing. Can be 'same' to keep the original order. Default is 'same'.
    plot : bool, optional
        Whether to plot the output. Default is False.

    Returns:
    --------
    xyzl_new : numpy.ndarray
        A numpy array containing the x, y, z coordinates of the points and the new line number they belong to.
    """

    # Make a copy of the input array
    xyzl = np.copy( xyzl )

    # If the input and output coordinate reference systems are different, project the coordinates
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = prjxy( prjcode_in, prjcode_out, 
                                          xyzl[:,x_c], xyzl[:,y_c] )

    # Sort the lines if necessary
    if order_c != 'same' :
        xyzl = sort_lines( xyzl, add_dist=False, line_c=line_c, x_c=x_c, y_c=y_c,
                           order_c=order_c )

    # If the new_line_c is None or False use the column indicated by line_c
    if ( ( new_line_c is None ) or ( new_line_c is False ) ) :
        if ( ( line_c is not None ) or ( line_c is not False ) ) :
            new_line_c = line_c
        else : 
            raise ValueError( 'The new line number column  (line_c) must be specified'+\
                              ' or the new_line_c must be set True' )

    # Else create a new column for the new line number
    else :
        new_line_c = xyzl.shape[1] 
        xyzl = np.column_stack( ( xyzl, np.ones( xyzl.shape[0] ) ) )
        if line_c == None :
            line_c = new_line_c

    # Itereate the process of line splitting multiple times 
    # to ensure that all lines are split
    # The number of iterations is set by the iter parameter

    xyli = np.copy( xyzl )

    for it in range(iter):

        # Initialize the output array
        xyzl_new = np.empty((0, xyli.shape[1]))
        nl = 0

        # Get the unique line numbers
        lines = np.unique(xyli[:, line_c])

        # Initialize a set to store the split lines
        split_lines_set = set()

        # Loop through each line and split it into smaller segments
        for il, l in enumerate(lines):

            idx = xyli[:, line_c] == l
            line = xyli[idx]

            if line.shape[0] < n_points:
                continue

            # Create a hashable representation of the line
            line_repr = (tuple(line[0]), tuple(line[-1]))

            # Check if the line has already been split
            if line_repr in split_lines_set:
                continue

            # If not, add it to the set and proceed with the splitting process
            split_lines_set.add(line_repr)

            line_dist = geo_line_dist(line[:, x_c], line[:, y_c])
            # Mean of relative distances between points along the line
            mean_line_step = np.mean(line_dist[1:, 0])

            line_resamp = resamp_lines(np.column_stack((line[:, x_c], line[:, y_c])),
                                    step=step, order_c='same')

            split_dist_lst = [[]]
            angle_segments0 = [[]]
            angle_segments1 = []
            new_split = False

            for ip, p in enumerate(line_resamp):

                angle_segments0[-1].append(line_resamp[ip])
                split_dist_lst[-1].append(step * ip)

                # If the point is not the last in the line, calculate the angle
                if (ip != 0) and (ip < line_resamp.shape[0] - 1) and (angle_segments0[-1] != []):

                    vector_1 = line_resamp[ip - 1] - p
                    vector_2 = line_resamp[ip + 1] - p

                    unit_vector_1 = vector_1 / (np.linalg.norm(vector_1) + 0.000001)
                    unit_vector_2 = vector_2 / (np.linalg.norm(vector_2) + 0.000001)
                    dot_product = np.dot(unit_vector_1, unit_vector_2)

                    dot_product = np.clip(dot_product, -0.9999, 0.9999)

                    angle = np.degrees(np.arccos(dot_product))
                    angle_diff = np.abs(180 - min(angle, 360 - angle))

                    # If the angle difference is greater than the threshold,
                    # create a new split line
                    if (angle_diff > angle_th) and (new_split is False):

                        angle_segments0.append([])
                        split_dist_lst.append([])
                        new_split = True

                    else:

                        new_split = False

            for i, as0 in enumerate(angle_segments0):

                as0 = np.array(as0)

                idx = (line_dist[:, 1] >= split_dist_lst[i][0] - step / 2) & \
                    (line_dist[:, 1] <= split_dist_lst[i][-1] + step / 2)

                angle_segments1.append(line[idx])

            for ls in angle_segments1:

                if ls.shape[0] < n_points:
                    continue

                rel_dist = geo_line_dist(ls[:, x_c], ls[:, y_c])[:, 0]
                indices = np.where(rel_dist > dist_th)[0]

                # Split arr into multiple arrays at the indices
                split_arrs = np.split(ls, indices)
                
                # Check for small segments and merge them if needed
                final_segments = []
                for i, arr in enumerate(split_arrs):
                    if arr.shape[0] < n_points:
                        if i > 0:
                            # Merge with the previous segment
                            final_segments[-1] = np.vstack((final_segments[-1], arr))
                        elif i < len(split_arrs) - 1:
                            # Merge with the next segment
                            split_arrs[i + 1] = np.vstack((arr, split_arrs[i + 1]))
                        continue
                    final_segments.append(arr)

                # After merging, add the segments to the final output
                for seg in final_segments:
                    seg[:, new_line_c] = nl
                    xyzl_new = np.vstack((xyzl_new, seg))
                    nl += 1

        xyli = xyzl_new


    # Join the lines to ensure that all lines are split
    if join_th is not None :
        xyzl_new = join_lines( xyzl_new, 
                               dist_th=join_th, 
                               line_c=new_line_c, 
                               x_c=x_c, y_c=y_c, 
                               angle_th=angle_th )


    # Plot the split lines if specified
    if plot is not None:

        plt.close('Splitted lines')
        plt.figure('Splitted lines')
        
        lineso = np.unique( xyzl[ :, line_c ])
        linesn = np.unique( xyzl_new[ :, new_line_c ])

        for l in linesn :

            idx = xyzl_new[ : , new_line_c ] == l
            line = xyzl_new[ idx ]
            plt.plot( line[:,x_c], line[:,y_c], 
                      linestyle='dashed', alpha=0.5, 
                      linewidth=size )
            plt.text( line[0,x_c], line[0,y_c], str(int(l)) )
            
        plt.scatter( xyzl[ :, x_c ], xyzl[ :, y_c ], c='r', s=size )
        plt.scatter( xyzl_new[ :, x_c ], xyzl_new[ :, y_c ], c='k', s=size, marker='x' )


        plt.tight_layout()

        if ( new_xy is False ) and ( prjcode_out != prjcode_in ) :
            xyzl_new[:,x_c], xyzl_new[:,y_c] = prjxy( prjcode_out, prjcode_in, 
                                            xyzl_new[:,x_c], xyzl_new[:,y_c] )

    return xyzl_new
        
# -----------------------------------------------------------------------------
def pad_lines( xyzl, pad_dist, pad_idx=-1, prjcode_in=None,
               prjcode_out=None, x_c=0, y_c=1, z_c=2, line_c=3, plot=False, s=1, 
               radius=0, order_c='same', dist=None ) :
    
    xyzl = np.copy( xyzl )
    
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = prjxy( prjcode_in, prjcode_out, 
                                          xyzl[:,x_c], xyzl[:,y_c])

    if order_c != 'same' :
        xyzli = sort_lines( xyzl, add_dist=False, x_c=x_c, y_c=y_c, 
                            line_c=line_c, order_c=order_c )
    else :
        xyzli = xyzl
    
    pad_xyzl = np.zeros( ( 0, xyzli.shape[1] ) ) 
    
    lines_id = np.unique( xyzli[:,line_c] )
    
    for i, l in enumerate( lines_id ) :
        
        line = xyzli[ xyzli[ :, line_c ] == l ] 
        
        len_line = line.shape[0]
        
        if len_line > 1 :
            
            if dist is None :    
                dist = min_dist( line[ :, x_c], line[ :, y_c] )['mean']
            
            if radius is None :
                radius = dist * 2
                
            n = int( int( pad_dist / dist ) + ( pad_dist % dist > 0 ) ) # round Up number of padding points
            new_line = np.copy( line )
            ramp_x = np.arange( 1, n+1 )
#            ramp = ( ramp_x/n )**4 - 2 *( ramp_x/n )**2 + 1
            r=0.1
            if n <= line.shape[0] :
                c = int( n/2 )
            else :
                c = int( line.shape[0] )
            ramp = ( -1 * ( np.e**(c*r) ) + 1 * ( np.e**(r*ramp_x) ) ) / ( ( np.e**(c*r) + ( np.e**(r*ramp_x) ) ) )
                            
            tg_l= []
            
            for i in range( 0, line.shape[0]-1 ) :
                if ( line[ i, y_c ] == line[ i+1, y_c ] ) and  \
                   ( line[ i, x_c ] == line[ i+1, x_c ] ) :
                    continue
                else :
                    num_i = line[ i, y_c ] - line[ i+1, y_c ] 
                    den_i = line[ i, x_c ] - line[ i+1, x_c ] 
                    if den_i == 0 :
                        tg_l.append( np.inf )
                    else :
                        tg_l.append( num_i / den_i ) 
  
            tg_l = np.abs( np.mean(tg_l) )
            
            for i in range( 1, n ) :
                
                signx = np.sign( line[ 0, x_c ] - line[ -1, x_c ] )
                signy = np.sign( line[ 0, y_c ] - line[ -1, y_c ] )
                row_i = np.copy( line[ 0, : ] )
                row_i[ x_c ] = line[ 0, x_c ] + signx * dist * i * np.cos( np.arctan( tg_l ) ) 
                row_i[ y_c ] = line[ 0, y_c ] + signy * dist * i * np.sin( np.arctan( tg_l ) )
                row_i[ -1 ] = pad_idx
                row_i[ z_c ] = line[ 0, z_c ]
                    
                row_f = np.copy( line[ -1, : ] )
                row_f[ x_c ] = line[ -1, x_c ] - signx * dist * i * np.cos( np.arctan( tg_l ) ) 
                row_f[ y_c ] = line[ -1, y_c ] - signy * dist * i * np.sin( np.arctan( tg_l ) )
                row_f[ -1 ] = pad_idx 
                
                row_f[ z_c ] = line[ -1, z_c ]
                new_line = np.vstack( ( row_i, new_line, row_f ) )
                
            if radius != 0 : 
                nearest_i = neighboring_points( ( line[ :, x_c ], line[ :, y_c ] ), 
                              ( line[ 0, x_c ], line[ 0, y_c ] ), radius )[1]
                rampi = normalize( ramp, line[ 0, z_c ], 
                                   np.nanmean( line[ nearest_i, z_c ] ) )
                new_line[0:n,z_c] = rampi
                nearest_f = neighboring_points( ( line[ :, x_c ], line[ :, y_c ] ), 
                              ( line[ -1, x_c ], line[ -1, y_c ] ), radius )[1]
                rampf = normalize( ramp, line[ -1, z_c ], 
                                   np.nanmean( line[ nearest_f, z_c ] ) )
                new_line[-n:,z_c] = np.flipud( rampf )                  
        
        else :
            new_line = line
        
        pad_xyzl = np.vstack( ( pad_xyzl, new_line ) )
        
    idx_original = pad_xyzl[ :, -1 ]
    pad_xyzl = np.delete( pad_xyzl, -1, 1 )
    
    if plot == True :
        
        plt.figure()
        
        add_idx = idx_original == pad_idx
        
        # plt.scatter( pad_xyzl[~add_idx, x_c], pad_xyzl[~add_idx, y_c], c='k', s=s, label='Original points')
        # plt.scatter( pad_xyzl[add_idx, x_c], pad_xyzl[add_idx, y_c], c='r', s=s, label='Added points')
        # plt.legend()
        # plt.gca().set_aspect('equal')

        plt.subplot( 1, 2, 1 )
        plt.scatter( xyzl[:, x_c], xyzl[:, y_c], c=xyzl[:, z_c], vmin=-20, vmax=10, s=1 )
        plt.subplot( 1, 2, 2 )
        plt.scatter( pad_xyzl[:, 0], pad_xyzl[:, 1], c=pad_xyzl[:, 2], vmin=-20, vmax=10, s=1 )

        plt.show()

    return pad_xyzl, idx_original  

# -----------------------------------------------------------------------------
def cross_over_points( xyzl, method='nearest', s=2, cmap='rainbow', plot=False, 
                       linewidths=0.05, vmin=None, vmax=None, colorbar=True,
                       prjcode_in=4326, prjcode_out=4326, extend_perc=50,
                       x_c=0, y_c=1, z_c=2, line_c=3, new_xy=False, absolute=True ) :
    
    xyzl = np.copy( xyzl )    
    if prjcode_in != prjcode_out :
        xyzl[:,x_c], xyzl[:,y_c] = prjxy( prjcode_in, prjcode_out, 
                                          xyzl[:,x_c], xyzl[:,y_c] )
        
    lines = np.unique( xyzl[:,line_c] )
    
    cross_lines = []
    cross_points = []
    cross_val = []
    
    for i1, l1 in enumerate( lines ) :
    
        if i1 is len(lines)-1 : 
            break
    
        li = xyzl[ xyzl[ :, line_c ] == l1 ]
        if li.shape[0] <= 1 : continue
    
        col_xy = [ x_c, y_c ]
        lsi = LineString( li[ :, col_xy ] )
            
        limi = xy2lim( li[:,x_c], li[:,y_c], extend=extend_perc, method='percentage' )    
        
        _,_,idx = xy_in_lim( xyzl[:,x_c], xyzl[:,y_c], limi )
        possible_cross_lines = np.unique( xyzl[idx, line_c] )
        possible_cross_lines = possible_cross_lines[ possible_cross_lines != l1 ]
        
        if ( np.size( possible_cross_lines ) == 0 ) : continue
        
        for i2, l2 in enumerate( possible_cross_lines ) :
            
            lii = xyzl[ xyzl[ :, line_c ] == l2 ] 
            if lii.shape[0] == 1 : continue
        
            lsii = LineString( lii[ :, col_xy ] )
            
            if lsi.intersects( lsii ) :
                
                i = lsi.intersection(lsii)
                
#                if  isiterable( inter ) is False : 
#                    i = [inter]
                
#                for i in inter :
                if type( i ) == shapely.geometry.point.Point :
                    if ( i.x, i.y ) not in cross_points :

                        cval_i = xyz2xy( ( li[:,x_c], li[:,y_c], li[:,z_c] ), 
                                         ( i.x, i.y ), method=method  ) 
                        cval_ii = xyz2xy( ( lii[:,x_c], lii[:,y_c], lii[:,z_c] ), 
                                          ( i.x, i.y ), method=method )
                        c_diff = cval_i - cval_ii
                        if absolute == True :
                            c_diff = np.abs( c_diff ) 
                    
                        cross_points.append( ( i.x, i.y ) )
                        cross_lines.append( ( l1, l2 ) )
                        cross_val.append( ( cval_i, cval_ii, c_diff ) )
                else :
                    continue
            
    cross_points = np.array( cross_points )
    cross_lines = np.array( cross_lines )
    cross_val = np.array( cross_val ) 
    

    cross_over_array = np.column_stack( ( cross_points, cross_lines, cross_val ) )
    
    if plot == True :
        
        plt.scatter( xyzl[:,x_c], xyzl[:,y_c], s=s, marker='_', c='k', 
                     linewidths=linewidths, alpha=0.5 )
        
        plt.scatter( cross_over_array[:,0], cross_over_array[:,1], s=s*10, 
                     c=cross_over_array[:,6], cmap=cmap, vmin=vmin, vmax=vmax )
        
        if colorbar is True :
            plt.colorbar()
        
        Min, Max, Mean, Std = stat( cross_over_array[:,6], decimals=2 )
        
        tit = f'Cross-Over Error : \n' + f' Min={Min}  Max={Max}  Mean={Mean}  Std={Std} '
        plt.title( tit )
        
    if new_xy is False :
        cross_over_array[:,0], cross_over_array[:,1] = prjxy( prjcode_out, prjcode_in, 
                cross_over_array[:,0], cross_over_array[:,1] )         
        
    return cross_over_array

# -----------------------------------------------------------------------------
def block_m( x, y, z=None, wind_size=None, method='mean', data_type='vector', lim=None,  
             prjcode_in=4326, prjcode_out=4326, nan=False, adjst_lim=True, 
             plot=False, s1=None, s2=None, xy_method='mean' ) :
    """
    Compute the block mean or median for a given 2D spatial dataset.

    Parameters:
    x (array-like): x-coordinates of the data points.
    y (array-like): y-coordinates of the data points.
    z (array-like): values of the data points.
    wind_size (float): size of the block window.
    method (str): statistic to compute (default is ‘mean’)
            ‘mean’ : compute the mean of values for points within each bin. Empty bins will be represented by NaN.
            ‘std’ : compute the standard deviation within each bin. This is implicitly calculated with ddof=0.
            ‘median’ : compute the median of values for points within each bin. Empty bins will be represented by NaN.
            ‘count’ : compute the count of points within each bin. This is identical to an unweighted histogram. values array is not referenced.
            ‘sum’ : compute the sum of values for points within each bin. This is identical to a weighted histogram.
            ‘min’ : compute the minimum of values for points within each bin. Empty bins will be represented by NaN.
            ‘max’ : compute the maximum of values for point within each bin. Empty bins will be represented by NaN.
            function : a user-defined function which takes a 1D array of values, 
                       and outputs a single numerical statistic. 
                       This function will be called on the values in each bin. 
                       Empty bins will be represented by function([]), 
                       or NaN if this returns an error.
    xy_method ('str'):'mean'.
    data_type (str): type of the output data. Can be 'vector' or 'grid'.
    lim (list): limits of the data points. If None, the limits are computed from the data.
    prjcode_in (int): input projection code.
    prjcode_out (int): output projection code.
    nan (bool): whether to remove NaN values from the output.
    adjst_lim (bool): whether to adjust the limits to the block size.
    plot (bool): whether to plot the data points and the block means.
    s1 (float): size of the data points in the plot.
    s2 (float): size of the block means in the plot.

    Returns:
    If data_type is 'vector', returns xb, yb, zb.
    If data_type is 'grid', returns Xb, Yb, Zb.
    """

    if prjcode_in != prjcode_out:
        x, y = prjxy(prjcode_in, prjcode_out, x, y)
    
    if lim == None:
        lim = [np.min(x), np.max(x), np.min(y), np.max(y)] 

    if wind_size == None:
        wind_size = min_dist( x, y )['mean'] * 2
    
    if z is None:
        z = np.ones_like( x ) 

    x_blk = np.linspace( lim[0]-wind_size/2, lim[1]+wind_size/2, 
                         num=int((lim[1]-lim[0])/wind_size)+1)
    y_blk = np.linspace( lim[2]-wind_size/2, lim[3]+wind_size/2, 
                         num=int((lim[3]-lim[2])/wind_size)+1)

    # Compute the block mean or median for z
    if method == 'mean':
        statistic = 'mean'
    elif method == 'median':
        statistic = np.median

    # Compute the block mean for x and y
    Xb, _, _, _ = sp.stats.binned_statistic_2d(x, y, x, statistic=xy_method, bins=[x_blk, y_blk])
    Yb, _, _, _ = sp.stats.binned_statistic_2d(x, y, y, statistic=xy_method, bins=[x_blk, y_blk])
    Zb, _, _, _ = sp.stats.binned_statistic_2d(x, y, z, statistic=method, bins=[x_blk, y_blk])
        
    if nan == False :
        xb = Xb[ ( ~np.isnan( Zb ) ) ].ravel()
        yb = Yb[ ( ~np.isnan( Zb ) ) ].ravel()
        zb = Zb[ ( ~np.isnan( Zb ) ) ].ravel()
    
    if plot == True:
        plt.figure()
        plt.scatter(x, y, c='b', s=s1)
        plt.scatter(xb, yb, c='r', s=s2)  

    if data_type == 'vector':
        return xb, yb, zb
        
    if data_type == 'grid':
        return Xb, Yb, Zb

# -----------------------------------------------------------------------------
def intersect_circle_line(circle_center, circle_radius, line_path):
    """
    Find intersections between a circle and a line segment.

    Args:
        circle_center (tuple or list): Coordinates of the circle's center (x, y).
        circle_radius (float): Radius of the circle.
        line_path (list of tuples): List of points defining the line segment.

    Returns:
        list of tuples: Intersection points (x, y).
    """
    intersections = []

    line_path = sorted( line_path, key=lambda p: (p[0], p[1]) )

    for i in range(len(line_path) - 1):
        p1 = np.array(line_path[i])
        p2 = np.array(line_path[i + 1])

        # Calculate the vector from p1 to p2
        v = p2 - p1

        # Calculate the vector from the circle center to p1
        w = p1 - circle_center

        # Coefficients for the quadratic equation
        a = np.dot(v, v)
        b = 2 * np.dot(w, v)
        c = np.dot(w, w) - circle_radius**2

        # Calculate the discriminant
        discriminant = b**2 - 4 * a * c

        if discriminant >= 0:
            t1 = (-b + np.sqrt(discriminant)) / (2 * a)
            t2 = (-b - np.sqrt(discriminant)) / (2 * a)

            if 0 <= t1 <= 1:
                intersections.append(p1 + t1 * v)
            if 0 <= t2 <= 1:
                intersections.append(p1 + t2 * v)

    return intersections

# -----------------------------------------------------------------------------
def resamp_lines( xyzl, step, prjcode_in=4326, prjcode_out=4326, 
                  plot=False, x_c=0, y_c=1, z_c=None, 
                  line_c=None, size=None, order_c='same', new_xy=False, 
                  lines=[], original_step=None, aspect='auto' ):
    """
    Resample lines to a given step size.

    Args:
        xyzl (numpy.ndarray): Array of shape (n, 4) containing x, y, z, and line number.
        step (float): Step size for resampling.
        prjcode_in (int, optional): Input projection code. Defaults to 4326.
        prjcode_out (int, optional): Output projection code. Defaults to 4326.
        cmap (str, optional): Colormap for plotting. Defaults to 'rainbow'.
        plot (bool, optional): Whether to plot the resampled lines. Defaults to False.
        x_c (int, optional): Index of x column in xyzl. Defaults to 0.
        y_c (int, optional): Index of y column in xyzl. Defaults to 1.
        z_c (int, optional): Index of z column in xyzl. Defaults to None.
        line_c (int, optional): Index of line number column in xyzl. Defaults to None.
        size (int, optional): Marker size for plotting. Defaults to None.
        order_c (str, optional): Order of lines. Defaults to None.
        new_xy (bool, optional): Whether to return new x and y coordinates. 
                                 Defaults to False.
        lines (list, optional): List of line numbers to resample. Defaults to [].
        original_step (float, optional): Original step size. Defaults to None.
        aspect (str, optional): Aspect ratio for plotting. Defaults to 'auto'.

    Returns:
        numpy.ndarray: Array of shape (m, 4) containing resampled x, y, z, 
        and line number.
    """

    # Convert xyzl to numpy array if it's a list or tuple
    if type(xyzl) in (list, tuple):
        xyzl = np.column_stack(xyzl)

    # Make a copy of xyzl
    xyzl = np.copy(xyzl)

    # Add a column of ones to xyzl if z_c is None
    z_c_i = True
    if z_c is None:
        xyzl = np.column_stack((xyzl, np.ones(xyzl.shape[0])))
        z_c = xyzl.shape[1] - 1
        z_c_i = False

    # Add a column of ones to xyzl if line_c is None
    line_c_i = True
    if line_c is None:
        xyzl = np.column_stack((xyzl, np.ones(xyzl.shape[0])))
        line_c = xyzl.shape[1] - 1
        line_c_i = False

    # Convert coordinates to output projection if prjcode_in != prjcode_out
    if prjcode_in != prjcode_out:
        xyzl[:, x_c], xyzl[:, y_c] = prjxy( prjcode_in, prjcode_out, 
                                            xyzl[:, x_c], xyzl[:, y_c])

    # Sort lines if order_c != 'same'
    
    if order_c != 'same':
        xyzl = sort_lines( xyzl, x_c=x_c, y_c=y_c, 
                           line_c=line_c, order_c=order_c, 
                           add_dist=False )

    # Create an empty array for the resampled lines
    xyzl_new = np.empty((0, 4,))

    # If lines argument is an empty list, 
    # set it equal to the unique line numbers 
    # found in the line column of the xyzl array
    if lines == []:
        lines = np.unique(xyzl[:, line_c])

    # Loop over unique line numbers in xyzl
    for l in np.unique(xyzl[:, line_c]):

        # Resample line if l is in lines
        if l in lines:

            # Get the line array from the complete xyzl array
            li = xyzl[xyzl[:, line_c] == l]
            # Get the path array of the selected line 
            # taking only the x and y columns
            path = np.column_stack( ( li[:, x_c], li[:, y_c], li[:, z_c] ) )

            # Get the distance between points on the line
            old_distances = geo_line_dist( path[:, 0], path[:, 1] )
            # Get relative and cumulative distances of points along the line
            rel_old_distances, cum_old_distances = old_distances[:, 0], old_distances[:, 1]

            # Set original_step to the mean of rel_dist_1[1:] 
            # if original_step is None,
            # starting from the second element ( the first element is 0 )
            if original_step is None:
                original_step = np.nanmean(rel_old_distances[1:])

            # Apply uniform filter if step > original_step for anti-alias
            if step > original_step:

                finer_sampling = np.arange(0, cum_old_distances[-1], original_step/2 )
                f_z_finer = np.interp( finer_sampling, cum_old_distances, path[:, 2] )
                kernel_size = int( np.ceil( step / original_step ) )
                f_z = sp.ndimage.uniform_filter( f_z_finer, size=kernel_size, mode='nearest' )
                f_z = np.interp( cum_old_distances, finer_sampling, f_z )

            else:
                f_z = li[:, z_c]

            new_cum_distances = np.arange(0, cum_old_distances[-1]+step, step)

            # Create interpolation functions for x and y
            new_x = np.interp( new_cum_distances, cum_old_distances, path[:,0] )
            new_y = np.interp( new_cum_distances, cum_old_distances, path[:,1] )

            new_path = np.column_stack( ( new_x, new_y ) )

            # If new_path has only one point, set it to the mean of the original path
            if new_path.shape[0] == 1:
                new_path[0, 0] = np.nanmean(path[:, 0])
                new_path[0, 1] = np.nanmean(path[:, 1])
                new_z = np.nanmean( f_z )
            else :
                # Interpolate z values of the resampled line
                new_z = np.interp( new_cum_distances, cum_old_distances, f_z )

            # Set line number of the resampled line to l
            new_l = np.full(new_path.shape[0], l)

            # Create an array of shape (n, 4) containing resampled x, y, z, and line number
            xyzl_i = np.column_stack((new_path[:, 0], new_path[:, 1], new_z, new_l))

        # Add the original line to xyzl_new if l is not in lines
        else:
            xyzl_i = np.column_stack(( xyzl[:, x_c], xyzl[:, y_c], 
                                       xyzl[:, z_c], xyzl[:, line_c]))

        # Add xyzl_i to xyzl_new
        xyzl_new = np.vstack((xyzl_new, xyzl_i))

    if line_c_i is False :
        xyzl_new = np.delete( xyzl_new, line_c, 1 )

    if z_c_i is False :
        xyzl_new = np.delete( xyzl_new, z_c, 1 )

    # Plot the resampled lines if plot is True
    if plot:

        plt.close('Reampled line')
        plt.figure('Reampled line')
        line, = plt.plot( path[:, 0], path[:, 1], c='b', marker='o', 
                          label='Old_line', markersize=size,)
        plt.plot( new_path[:, 0], new_path[:, 1], c='r', 
                  markersize=line.get_markersize() * 2,
                  alpha=0.7, marker='x', label='New_line')
        plt.gca().set_aspect(aspect)
        plt.title('Resampled Line ')
        plt.legend()

        # plt.subplot( 1, 2, 1 )
        # plt.scatter( xyzl[:, x_c], xyzl[:, y_c], c=xyzl[:, z_c], vmin=-20, vmax=10, s=1 )
        # plt.subplot( 1, 2, 2 )
        # plt.scatter( xyzl_new[:, 0], xyzl_new[:, 1], c=xyzl_new[:, 2], vmin=-20, vmax=10, s=1 )


    # Convert coordinates to input projection if prjcode_in != prjcode_out and new_xy is True
    if (prjcode_in != prjcode_out) and (new_xy is True):
        xyzl_new[:, 0], xyzl_new[:, 1] = prjxy( prjcode_out, prjcode_in, 
                                                xyzl_new[:, 0], xyzl_new[:, 1] )

    # Return xyzl_new
    return xyzl_new

# -----------------------------------------------------------------------------
def lines_samp_dist( xyzl, prjcode_in=4326, prjcode_out=4326, kind='mean',
                     x_c=0, y_c=1, line_c=3, s=1, round_dist=6, print_dist=True,
                     deg_to_m=False ) : 

    if prjcode_in != prjcode_out:
        xyzl[:,x_c], xyzl[:,y_c] = prjxy( prjcode_in, prjcode_out, 
                                          xyzl[:,x_c], xyzl[:,y_c] )   

    lines = np.unique( xyzl[ :, line_c ]  )    

    lines_dist = []
    
    for l in lines :
        
        li = xyzl[ xyzl[ :, line_c ] == l ]
        lines_dist.append( min_dist( li[:,x_c], li[:,y_c] )[kind] )
        
    minimum_dist = round( np.nanmean( lines_dist ), round_dist )
    
    if deg_to_m is True :
        print( 'Minimum sampling distance :', deg2m( minimum_dist ) )

    else :
        print( 'Minimum sampling distance :', minimum_dist )

    return minimum_dist

# -----------------------------------------------------------------------------
def array2csv( array, headers=[], sep=',', fmt='% 15.6f', 
               file='new_dataset.csv', nan=None, nan_c=2, nround=6 ) :
    
    # Ensure the directory exists
    path = os.path.dirname(file)
    print(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

    # Handle NaN replacement
    if nan is not None:
        inan = np.isnan(array[:, nan_c])
        if (type(nan) == bool) and (nan == False):
            array = array[~inan]
        if type(nan) in (float, int):
            array[inan, nan_c] = nan

    # Round float values to nround decimals
    if 'nround' in locals() or 'nround' in globals():
        nround_val = nround
    else:
        nround_val = 6  # Default value if not provided

    # Create a copy to avoid modifying the original array
    array_to_save = np.copy(array)
    for i in range(array_to_save.shape[0]):
        for j in range(array_to_save.shape[1]):
            val = array_to_save[i, j]
            if is_number(val):
                array_to_save[i, j] = np.round(float(val), nround_val)

    hd = ''
    fmt_str = ''

    if headers != []:
        for i, h in enumerate(headers):
            hd = hd + h + sep
        hd = hd[:-1]

    if fmt == '%s':
        array_to_save_str = np.array([[str(val) if not is_number(val) else f"{float(val):.{nround_val}f}" for val in row] for row in array_to_save], dtype=str)
        np.savetxt(file, array_to_save_str, header=hd, fmt='%s', comments='', delimiter=sep)
    
    else:
        np.savetxt(file, array_to_save, header=hd, fmt=fmt, comments='', delimiter=sep)

    abs_path = os.path.abspath(file)

    return abs_path

# -----------------------------------------------------------------------------
def read_csv( file, sep=' ', 
              header=None, 
              skiprows=[], 
              skipcols=[], 
              force_str=False ):
    """
    Reads a CSV file and returns the data as a dictionary, 
    array, header, format, comments, and raw data.

    Parameters:
    - file (str): The path to the CSV file.
    - sep (str): The delimiter used in the CSV file. Default is a space.
    - header (int or None): The row index of the header. Default is None.
    - skiprows (list): A list of row indices to skip. Default is an empty list.
    - skipcols (list): A list of column indices to skip. Default is an empty list.
    - force_str (bool): If True, all elements will be read as strings. Default is False.

    Returns:
    - dictionary (dict): A dictionary containing the data from the CSV file.
    - array (ndarray): A NumPy array containing the data from the CSV file.
    - hd (list): A list of column names.
    - data (list): A list of lists representing the raw data from the CSV file.
    """
    
    # Initialize an empty dictionary to store the data
    dictionary = {}
    # Initialize an empty list to store the data
    data = []
    # Initialize the header to None
    hd = None

    if type( skiprows ) in [ int, float ] :
        skiprows = [ int(sr) for sr in range(skiprows) ]

    # Open the CSV file
    with open(file, "r", encoding="utf8", errors='ignore') as f:
        # Read all lines from the file
        lines = f.readlines()

    # Loop over each line in the file
    for i, line in enumerate(lines):
        # If the current line is in the list of lines to skip, 
        # continue to the next line
        if i in skiprows:
            continue

        # Split the line into data using the separator
        line_data = line.split(sep)
        # Strip whitespace from each data item 
        # and ignore items in the skipcols list
        line_data = [ item.strip() for idx, item in enumerate(line_data) 
                      if idx not in skipcols and item.strip() != '' ]

        # If the current line is the header line, 
        # store the data as the header and continue to the next line
        if i == header:
            hd = line_data
            continue

        # If the header is still None, generate a default header
        if hd is None:
            hd = ['c' + str(n) for n in range(len(line_data))]

        # If the data list is empty, initialize it with empty lists for each column
        if data == []:
            data = [ [] for _ in range(len(hd)) ]

        # Loop over each value in the line data
        for c, val in enumerate(line_data):
            # If the value is an empty string, replace it with NaN
            if val == '':
                val = np.nan
            # If force_str is False, try to convert the value to a float
            elif not force_str:
                try:
                    val = float(val)
                # If the conversion fails, keep the value as a string
                except ValueError:
                    pass
            # Append the value to the corresponding column in the data list
            data[c].append(val)

    # Loop over each column in the header
    for i, h in enumerate(hd):
        # Store the column data in the dictionary
        dictionary[h] = np.array(data[i])

    # Convert the dictionary values to a NumPy array and transpose it
    array = np.array(list(dictionary.values())).T

    # Return the dictionary, array, header, and raw data
    return dictionary, array, hd, data

# -----------------------------------------------------------------------------
def d( csv_file, sep=',', header=None, skiprows=[], encoding="utf8", 
              n_col=None, adjust_last_col_fmt=True ) :
    
    d = read_csv( csv_file, sep=sep, header=header, skiprows=skiprows, 
                  encoding=encoding, n_col=n_col, 
                  adjust_last_col_fmt=adjust_last_col_fmt )[0]

    return d

# -----------------------------------------------------------------------------
def csv2array( csv_file, sep=',', header=None, skiprows=[], encoding="utf8", 
               n_col=None, adjust_last_col_fmt=True ) :
    
    a = read_csv( csv_file, sep=sep, header=header, skiprows=skiprows, 
                  encoding=encoding, n_col=n_col, 
                  adjust_last_col_fmt=adjust_last_col_fmt )[1]

    return a
   
# ----------------------------------------------------------------------------- 
def dict2array( dictionary, exclude=[], flen=None ) :
    
    headers = []
        
    field_len = [] 

    for k in dictionary :
        dictionary[ k ] = np.array( dictionary[ k ] ).ravel()
        field_len.append( np.size( dictionary[ k ] ) )
        
    if flen is None :
        flen = np.max( field_len )
        
    array = np.zeros( flen )

    for i, k in enumerate( dictionary ) :

        if np.size(dictionary[k]) != flen :
            if type(dictionary[k][0]) is float :
                pad_value = dictionary[k][0] if len(dictionary[k]) == 1 else np.nan
            elif type(dictionary[k][0]) is int :
                pad_value = dictionary[k][0] if len(dictionary[k]) == 1 else 0
            elif type(dictionary[k][0]) is str :
                pad_value = dictionary[k][0] if len(dictionary[k]) == 1 else ''
            else :
                pad_value = 0
            dictionary[k] = np.pad( dictionary[k], (0, flen - len(dictionary[k])), 
                                    constant_values=pad_value )

        if k not in exclude :
            carray = np.asarray( dictionary[ k ] )
            if type( carray[0] ) == np.datetime64 :
                carray = np.array( [ str( c ) for c in carray ] )
            array = np.column_stack( (  array, carray ) )
            headers.append( k )
            
    array = array[:, 1:]
    
    return array, headers
            
# -----------------------------------------------------------------------------            
def dict2csv( dictionary, sep=',', fmt='% 15.6f', file='new_dataset.csv', **kwargs ) :
    
    array, headers = dict2array( dictionary )
    
    abs_path = array2csv( array, headers=headers, sep=sep, fmt=fmt, 
                          file=file, **kwargs )
    
    return abs_path

# -----------------------------------------------------------------------------            
def addc2csv( csv_file, new_c=[], new_hd=[], new_fmt=[], sep=',', rep_fmt=None,
              file=None, rep_headers=None, header=None, skiprows=[], 
              encoding="utf8", n_col=None, adjust_last_col_fmt=True ) :

    if file == None :
        file = csv_file

    csvf = read_csv( csv_file, sep=sep, header=header, skiprows=skiprows, 
                     encoding=encoding, n_col=n_col, 
                     adjust_last_col_fmt=adjust_last_col_fmt )
    
    array = csvf[1]
    headers = csvf[2]
    fmt = csvf[3]
    n_c = array.shape[1]     
    new_array = np.copy( array )
    if new_hd == [] :
       new_hd = [ 'c_'+str(n_c+i) for i in range( len( new_c ) ) ] 
    if new_fmt == [] :
       new_fmt = [ '% 15.6f' for i in range( len( new_c ) ) ]   
    if type( new_hd ) == str :
        new_hd = [ new_hd ] * len( new_c )    
    if type( new_fmt ) == str :
        new_fmt = [ new_fmt ] * len( new_c )     
   
    for i, col in enumerate( new_c ) :
        new_array = np.column_stack( ( new_array, col ) )
        if headers != None :
            headers.append( new_hd[i] )
        fmt.append( new_fmt[i] )

    if rep_fmt != None :
        fmt = rep_fmt
        
    if rep_headers != None :
        headers = rep_headers

    abs_path = array2csv( new_array, headers=headers, sep=sep, fmt=fmt, 
                          file=file )
    
    return abs_path
   
# ----------------------------------------------------------------------------- 
class SelectFromCollection(object):
    # Select indices from a matplotlib collection using `LassoSelector`.
    #
    # Selected indices are saved in the `ind` attribute. This tool fades out the
    # points that are not part of the selection (i.e., reduces their alpha
    # values). If your collection has alpha < 1, this tool will permanently
    # alter the alpha values.
    #
    # Note that this tool selects collection objects based on their *origins*
    # (i.e., `offsets`).
    #
    # Parameters
    # ----------
    # ax : :class:`~matplotlib.axes.Axes`
    #     Axes to interact with.
    #
    # collection : :class:`matplotlib.collections.Collection` subclass
    #     Collection you want to select from.
    #
    # alpha_other : 0 <= float <= 1
    #     To highlight a selection, this tool sets all selected points to an
    #     alpha value of 1 and non-selected points to `alpha_other`.
    #

    def __init__(self, ax, collection, alpha_other=0.3, facecolors=None):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        if facecolors is not None: self.fc = facecolors

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.colledefeection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        
#------------------------------------------------------------------------------
def merge2Darrays( array_list, 
                   res_list, 
                   sigmab=1, 
                   spl_order=[], 
                   mean_shift=False, 
                   plot=False, 
                   vmin=None, 
                   vmax=None, 
                   buffer=None, 
                   plot_diff=False, 
                   s=None, 
                   iter=None ):
    """
    Merge multiple 2D arrays into a single array.

    Parameters:
    - array_list (list): List of 2D arrays to be merged.
    - res_list (list): List of resolutions corresponding to each array in array_list.
    - sigmab (int, optional): Buffer size for smoothing. Default is 1.
    - spl_order (list, optional): List of spline orders for each array in array_list. Default is an empty list.
    - mean_shift (bool, optional): Flag to enable mean shift. Default is False.
    - plot (bool, optional): Flag to enable plotting. Default is False.
    - vmin (float, optional): Minimum value for plotting. Default is None.
    - vmax (float, optional): Maximum value for plotting. Default is None.
    - buffer (int, optional): Buffer size for smoothing. Default is None.
    - plot_diff (bool, optional): Flag to enable plotting of differences. Default is False.
    - s (float, optional): Size of markers for plotting. Default is None.
    - iter (int, optional): Number of iterations for smoothing. Default is 1.

    Returns:
    - Z1 (ndarray): Merged 2D array.

    """

    Z1 =  np.copy( array_list[0] )
    shape1 = Z1.shape
    if spl_order == [] :
        spl_order = [ 1 for i in array_list ]
    
    for i, array in enumerate( array_list ) :
        
        if i == 0 : continue
        
        Z2 = np.array( array )
        shape2 = Z2.shape
        shape_ratio = shape1[0]/shape2[0], shape1[1]/shape2[1]
        res_ratio = round( res_list[i] / res_list[i-1] ) 

        if buffer is None :
            buffer = int( res_ratio )

        if sigmab is None :
            sigmab = int( buffer )
            
        buffer = buffer * sigmab

        if (shape_ratio[0] != 1.0) and (shape_ratio[1] != 1.0) :
            Z21 = sp.ndimage.zoom( Z2, shape_ratio, order=1 )

        else: 
            Z21 = np.copy( Z2 ) 
        mask1 = np.isfinite( Z1 ) 
        mask2 = np.isfinite( Z21 )
        mask1n2 = mask1 & ~mask2
        
        if mean_shift == True:
            Z1 = Z1 + np.nanmean( Z21[mask1] - Z1[mask1] )        
        # ---
        # If the i-th raster has the original resolution lower than the final chosen resolution,
        # it will be smoothed with a rolling average, 
        # The kernel size of the convolution is equal to the ratio betwee original and final resolution (+1 if it's even).
        # This will reduce aliasing artefacts in the low resolution area  
        
        if np.mean(shape_ratio) > 1 :
            Z21 = grid_fill( Z21 )
            if np.mean(shape_ratio) % 2 == 0 :
                Z21 = sp.ndimage.uniform_filter( Z21 , np.mean(shape_ratio)+1 )
            else :
                Z21 = sp.ndimage.uniform_filter( Z21 , np.mean(shape_ratio) )
            Z21[ ~mask2 ] = np.nan 
        
        D_21 = Z1 - Z21  
        mask1a2 = np.isfinite( D_21 )  
    
        # If there are no common pixel between the two arrays, 
        # they will be merged as they are withut blending the edges 
        # ---
        if np.all( mask1a2 == False  ):
            Z1[mask2] = Z21[mask2]
            continue

        Z1[ ~mask1 ] = 0
        Z21[ ~mask2 ] = 0
        D_21[ ~mask1a2 ] = np.nan
        D_21[ mask1n2 ] = Z1[ mask1n2 ]
        mask3 = ~sp.ndimage.uniform_filter( ~mask1*1, buffer*2+1 ).astype(bool)
        weight = sp.ndimage.gaussian_filter( mask3*1.0, buffer*2 ) 
        weight[ mask1n2 ] = 1
        D_21[ ~mask2  ] = 0

        if iter is None :
            iter = int( buffer )

        D_21fn = grid_fill( D_21 )
        
        FG = np.copy( D_21fn )

        for i in range( iter ) :
            
            FG1 = sp.ndimage.gaussian_filter( FG, 2 )
            WF = np.abs( sp.ndimage.gaussian_filter( ~mask1*1.0, 2 ) - 1 )
            DFG =  FG - FG1
            FG = FG1 + DFG * WF
            
        FG = FG * weight

        if plot_diff == True :
            plta( weight, vmin=vmin, vmax=vmax, sbplt=[1,2,1] )
            mask_edges( mask1, plot=True, c='k', s=s )
            plta( FG, vmin=vmin, vmax=vmax, sbplt=[1,2,2] )
            mask_edges( mask1, plot=True, c='k', s=s )

        FG[ mask1n2 ]  = Z1[ mask1n2 ] 
        
        Z2n = Z21 + FG
        Z2n[ ~( mask1 | mask2 ) ] = np.nan
        Z1 = Z2n
        
        mask5 = ~sp.ndimage.uniform_filter( ~mask1*1, int(buffer) ).astype(bool)
        mask6 = sp.ndimage.uniform_filter( mask1*1, int(buffer) ).astype(bool)
        mask7 = ( mask5 ^ mask6 ) & mask2
        WF = np.abs( sp.ndimage.gaussian_filter( mask7*1.0, 3 ) - 1 )
        fw = np.ones( ( 3, 3) ) / 9
        Zf = grid_fill( Z1, invalid=None )
        
        for i in range( buffer ) :
            Z1F = signal.convolve2d( Zf, fw, mode='same' ) 
            DZ =  Zf - Z1F
            Zf = Z1F + DZ * WF  
        Zf[ ~( mask1 | mask2 ) ] = np.nan
        Z1 = Zf
        
    if plot is True :
        plta( Z1, vmin=vmin, vmax=vmax)

    return Z1

# -----------------------------------------------------------------------------
def rolling_win_2d( arr, win_shape ) :
    
    if ( win_shape[0] % 2 == 0 ) or ( win_shape[1] % 2 == 0 ) :
        raise NameError('Both values in win_shape must be odd integers !')
    
    r_extra = np.floor(win_shape[0] / 2).astype(int)
    c_extra = np.floor(win_shape[1] / 2).astype(int)
    a = np.empty((arr.shape[0] + 2 * r_extra, arr.shape[1] + 2 * c_extra))
    a[:] = np.nan
    a[r_extra:-r_extra, c_extra:-c_extra] = arr    
    
    s = (a.shape[0] - win_shape[0] + 1,) + (a.shape[1] - win_shape[1] + 1,) + win_shape
    strides = a.strides + a.strides
    
    windows = np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)
    windows = windows.reshape( arr.size, win_shape[0] * win_shape[1] )
    
    return windows    

# -----------------------------------------------------------------------------
def median_filt( array, radius=1, padw=0, pmode='linear_ramp', 
                 plot=False, vmin=None, vmax=None, iter=1 ):
    
    ar_pad, original_shape_indx = pad_array(array, padw, pmode)
    fw = np.ones( ( radius * 2 + 1, radius * 2 + 1 ) )
    
    for i in range( iter ) : 
        ar_pad = sp.ndimage.median_filter( ar_pad, footprint=fw, 
                                               mode='nearest' )
    
    ar_filt = crop_pad( ar_pad, original_shape_indx )
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt

# -----------------------------------------------------------------------------
def mean_filt( array, radius=1, padw=0, pmode='linear_ramp', 
               plot=False, vmin=None, vmax=None, iter=1 ) :
    
    ar_pad, original_shape_indx = pad_array(array, padw, pmode)
    fw = np.ones( ( radius*2 + 1, radius*2 + 1) ) / ( radius * 2 + 1 ) ** 2
    
    for i in range( iter ) : 
        ar_pad = signal.convolve2d( ar_pad, fw, mode='same' )
    
    ar_filt = crop_pad( ar_pad, original_shape_indx )
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt

#------------------------------------------------------------------------------
def hanning_filt( array, padw=0, pmode='linear_ramp', 
                  plot=False, vmin=None, vmax=None, iter=1 ) :
    
    ar_pad, original_shape_indx = pad_array(array, padw, pmode)
    fw = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]) / 6
    
    for i in range(iter):
        ar_pad = sp.ndimage.convolve( ar_pad, fw, mode='nearest' )
    
    ar_filt = crop_pad(ar_pad, original_shape_indx)
    
    if plot:
        plta(array, sbplt=[1, 3, 1], tit='Original')
        plta(ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta(array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout()
        
    return ar_filt

# -----------------------------------------------------------------------------
def gauss_filt( array, radius=None, sigma=1, padw=0, pmode='linear_ramp', 
                alpha=None, iter=1, plot=False, vmin=None, vmax=None,
                truncate=4.0 ):

    ar_pad, original_shape_indx = pad_array( array, 
                                             padw=padw, 
                                             mode=pmode, 
                                             alpha=alpha )

    # N.B. radius = round(truncate * sigma)

    if radius is not None :
        sigma = radius / truncate

    for i in range( iter ) : 
        ar_pad = sp.ndimage.gaussian_filter( ar_pad, 
                                             sigma,
                                              truncate=truncate )
    
    ar_filt = crop_pad( ar_pad, original_shape_indx )
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt

# -----------------------------------------------------------------------------
def resampling_filt( array, factor, padw=0, pmode='linear_ramp', 
                     alpha=None, spl_order=2, 
                     plot=False, vmin=None, vmax=None ) :
    
    if padw == 0 :
        padw = [ 0, 0 ]
    if array.shape[0] % 2 == 0  :
        padw[0] += 1 
    if array.shape[1] % 2 == 0  :
        padw[1] += 1 

    ar_pad, original_shape_indx = pad_array( array, padw=padw, mode=pmode, alpha=alpha, ptype='' )
    
    # Down_Sampling
    ar_pad_dw = resampling( ar_pad, 1/factor, spl_order=spl_order )
    
    # Upsampling
    ar_pad_up = resampling( ar_pad_dw, factor, spl_order=spl_order )   
    
    ar_filt = crop_pad( ar_pad_up, original_shape_indx )
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt    

# -----------------------------------------------------------------------------
def std_filt( array, padw=0, pmode='gdal', radius=1, n=1, 
              alpha=None, plot=False, vmin=None, vmax=None ) :

    ar_pad, original_shape_indx = pad_array(array, padw=padw, mode=pmode, alpha=alpha)
    
    win_shape = radius * 2 + 1, radius * 2 + 1 
    windows = rolling_win_2d( ar_pad, win_shape )
    w_mean = np.nanmean( windows, axis=1 )
    w_std = np.nanstd( windows, axis=1 )
    w_filt = w_mean + np.sign( w_mean ) * w_std
    ar_pad_filt = w_filt.reshape( ar_pad.shape )
    
    ar_filt = crop_pad( ar_pad_filt, original_shape_indx )    
    
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
        
    return ar_filt        

#------------------------------------------------------------------------------
def filt2d( array, 
            radius=1, 
            padw=0, 
            pmode='linear_ramp', 
            plot=False, 
            vmin=None, 
            vmax=None, 
            iter=1, 
            ftype='mean', 
            sigma=1, 
            factor=2, 
            mask=None,
            fill=None ):

    if type( ftype ) is str :
        ftype = [ ftype ]
    
    if  type( iter ) is int :
        iter = [ iter for ft in ftype ]
        
    if fill != None :
        nan = np.isnan( array )
        array = fillnan( array, method=fill )
    
    ar_filt = np.copy( array )

    for i, f in enumerate( ftype ) :
        
        if ftype[i] == 'mean' :
            ar_filt = mean_filt( ar_filt, 
                                 radius=radius, 
                                 padw=padw, 
                                 pmode=pmode, 
                                 iter=iter[i] )  
            
        if ftype[i] == 'hanning' :
            ar_filt  = hanning_filt( ar_filt, 
                                     padw=padw, 
                                     pmode=pmode, 
                                     iter=iter[i] )
            
        if ftype[i] == 'median' :
            ar_filt  = median_filt( ar_filt, 
                                    radius=radius, 
                                    padw=padw, 
                                    pmode=pmode, 
                                    iter=iter[i] )
        if ftype[i] == 'gauss' :
            ar_filt  = gauss_filt( ar_filt, 
                                   radius=radius, 
                                   padw=padw, 
                                   pmode=pmode, 
                                   iter=iter[i], 
                                   sigma=sigma ) 
            
        if ftype[i] == 'resamplig' :
            ar_filt  = resampling_filt( ar_filt, factor=factor, padw=padw, pmode=pmode ) 
            
    if fill != None :
          ar_filt[ nan ] = np.nan   
            
    if mask :
            ar_filt[mask] = np.nan
      
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered',
              new_fig=False)
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences',
              new_fig=False )
        plt.tight_layout() 
        
    return ar_filt 

# -----------------------------------------------------------------------------
def filt_edges( array, mask=None, size=3, iter=1, plot=False, vmin=None, vmax=None,
                smooth=0 ) :
    
    if mask is None :
        mask = np.isfinite( array )
    
    edge_mask1 = sp.ndimage.uniform_filter( mask*1, size=size ) 
    edge_mask2 = np.abs( sp.ndimage.uniform_filter( 
                         np.abs(mask-1)*1, size=size ) -1 )
    
    edge_mask = edge_mask2 - edge_mask1
    
    edge_mask = sp.ndimage.uniform_filter( edge_mask*1.0, size=size )
    edge_mask[ edge_mask<1e-10 ] = 0
    edge_mask[ edge_mask>1 ] = 1
    
    ar_filt = np.copy( array )
    
    for i in range( iter ) :
        FG = sp.ndimage.uniform_filter( ar_filt, size )
        DFG =  ar_filt - FG
        ar_filt = ar_filt - DFG * edge_mask 
        
    if smooth != 0 :
        mean = np.nanmean( ar_filt )
        edge_mask3, original_shape_indx = pad_array( mask, padw=[2,2], 
                            ptype='', mode='constant', constant_values=0 ) 
        edge_mask3 = edge_mask3 * 2.0
        edge_mask3[:,[0,-1]] = 1.0
        edge_mask3[[0,-1],:] = 1.0   
        edge_mask3[(edge_mask3!=1.0)&( edge_mask3!=2.0)] = 1.5        
        SE = sp.ndimage.gaussian_filter( edge_mask3, smooth ) - 1
        SE = crop_pad( SE, original_shape_indx )
        ar_filt = ( ar_filt - mean ) * SE + mean
        
    if plot == True:
        
        plta( array, sbplt=[1, 3, 1], tit='Original')
        plta( ar_filt, vmin, vmax, sbplt=[1, 3, 2], tit='Filtered')
        plta( array - ar_filt, vmin, vmax, sbplt=[1, 3, 3], tit='Differences')
        plt.tight_layout() 
    
    return ar_filt   

# -----------------------------------------------------------------------------
def lim2grid( lim, step=None, xyz=None, plot=False, vmin=None, vmax=None, prjcode_in=4326,
              prjcode_out=4326, method='linear', blkm=False, filt=False, radius=None,
              nan=True, padw=0 ) :
    """
    Convert a given bounding box into a grid of points.

    Parameters:
    - lim (tuple): The bounding box limits in the form (xmin, xmax, ymin, ymax).
    - step (float): The grid step size. If None, it will be automatically calculated based on the input data.
    - xyz (tuple or None): The input data points. If None, only the grid points will be returned.
    - plot (bool): Whether to plot the resulting grid.
    - vmin (float): The minimum value for the plot color scale.
    - vmax (float): The maximum value for the plot color scale.
    - prjcode_in (int): The input projection code.
    - prjcode_out (int): The output projection code.
    - method (str): The method used for interpolation.
    - blkm (bool): Whether to perform block mean interpolation.
    - filt (bool): Whether to apply a filter to the interpolated data.
    - radius (int): The radius of the filter.
    - nan (bool): Whether to remove NaN values from the interpolated data.
    - padw (int): The padding width for the filter.

    Returns:
    - If xyz is None or len(xyz) == 2:
        - X (ndarray): The X coordinates of the grid points.
        - Y (ndarray): The Y coordinates of the grid points.
    - If xyz is not None and len(xyz) > 2:
        - X (ndarray): The X coordinates of the grid points.
        - Y (ndarray): The Y coordinates of the grid points.
        - Z (ndarray): The interpolated values at the grid points.
    """
    
    if xyz is None:

        if step is None:
            raise ValueError("If xyz is None, step must not be None.")
        
        xg = np.linspace( lim[0], lim[1], int( (lim[1]-lim[0])/step ) ) 
        yg = np.linspace( lim[3], lim[2], int( (lim[3]-lim[2])/step ) )
        X, Y = np.meshgrid( xg, yg )
        
        return X, Y

    if len( xyz ) <= 2 :
        x, y = xyz
    else :
        x, y, z = xyz
    
    if prjcode_in != prjcode_out :
        x, y = prjxy( prjcode_in, prjcode_out,  x, y )
        
    x, y, idx = xy_in_lim( x, y, lim )
    if len( xyz ) > 2 :
        z = z[idx]

    if step == None :
        step = min_dist( x, y )['mean']
        
    if ( blkm == True ) and  ( len( xyz ) > 2 ) :
        x, y, z = block_m( x, y, z, step*2, method='mean', data_type='vector', lim=lim )  
        
    xg = np.linspace( lim[0], lim[1], int( (lim[1]-lim[0])/step ) ) 
    yg = np.linspace( lim[3], lim[2], int( (lim[3]-lim[2])/step ) )
    X, Y = np.meshgrid( xg, yg )
    
    if len( xyz ) > 2 :
        Z = xyz2xy( ( x, y, z ), ( X, Y ), method=method, fillnan=False )
        
        if filt == True :
            if radius == None :
                radius = int( min_dist( x, y )['mean'] / step )
            Z = filt2d( Z, radius=radius, ftype='mean', padw=padw  )    
        
    if plot == True :
        plta( Z )
        
    if len( xyz ) <= 2 :
        
        return X, Y
    
    if len( xyz ) > 2 :
        
        if nan == False :
            X = X[ ~np.isnan( Z ) ]
            Y = Y[ ~np.isnan( Z ) ]
            Z = Z[ ~np.isnan( Z ) ]
            
        return X, Y, Z
    
# -----------------------------------------------------------------------------
def mask2D( xyr=None, xyzgrid=None, array=None, array_and_lim=None, mask=None,
            plot=False, vmin=None, vmax=None,
            convexhull=False ):

    if xyr is not None:
        x, y, r = xyr
        
    if xyzgrid is not None:
        xx, yy, zz = xyzgrid[0], xyzgrid[1], xyzgrid[2]
        xa, ya, za = xx.ravel(), yy.ravel(), zz.ravel()
        
    if array is not None :
        zz = np.copy( array )
        x, y = np.arange( zz.shape[1] ), np.arange( zz.shape[0] )
        xx, yy = np.meshgrid( x, y )
        xa, ya, za = xx.ravel(), yy.ravel(), zz.ravel()
        
    if array_and_lim is not None: 
        zz = array_and_lim[0]
        lim = array_and_lim[1]
        nx, ny = zz.shape[1], zz.shape[0]
        xi, yi = np.linspace(lim[0], lim[1], nx), np.linspace(lim[3], lim[2], ny)
        xx, yy = np.meshgrid(xi, yi)
        xa, ya, za = xx.ravel(), yy.ravel(), zz.ravel()
        
    if ( xyr is not None ) and ( convexhull is False ) :
        zm = np.empty( np.shape( za ) )
        zm[:] = np.nan
        for p in zip( x, y ):
            d = np.sqrt((p[0] - xa[:]) ** 2 + (p[1] - ya[:]) ** 2)
            zm[d <= r] = za[(d <= r)]

        ZM = zm.reshape( np.shape( zz ) )

    if ( xyr is not None ) and ( convexhull is True ) :
       mhull = xy_in_hull( xa, ya, ( x, y ), buffer=r, plot=False )[0]  
       Mhull = mhull.reshape( np.shape( zz ) ) 
       ZM = np.copy( zz )
       ZM[ ~Mhull ] = np.nan
        
    if xyr is None:
        ZM = zz

    if mask is not None:
        ZM[ np.isnan( mask ) ] = np.nan
        
    isfinit = np.isfinite( ZM )
    
    if plot == True:
        plta(zz, vmin, vmax, tit='original', sbplt=[1, 2, 1])
        plta(ZM, vmin, vmax, tit='mask', sbplt=[1, 2, 2])
        
    return ZM, isfinit

#------------------------------------------------------------------------------
def resampling( array, 
                factor, 
                spl_order=1, 
                dimention='2D', 
                mode='nearest', 
                plot=False, 
                vmin=None, vmax=None, 
                nan=True,
                prefilter=True ) :

    if type( array ) in ( list, tuple ) :
        
        if len( array ) == 3 :
            ax, ay, az = array[0], array[1], array[2]
        if len( array ) == 2 :  
            ax = np.copy( array )            
        if len( array ) == 1 :  
            az = np.copy( array )
        IsXy = True    

    else :
        az = np.copy( array )
        ax, ay = None, None
        IsXy = False
        
    if type( factor ) in ( list, tuple ) :
        factor = factor[0]/array.shape[0], factor[1]/array.shape[1]
        
    if ( np.isnan( az ).any() ) and ( spl_order > 1 ) :
        inan = np.isnan( az )
        az = fillnan( az, method='nearest' )
        IsNan = True
    else :
        IsNan = False
        
    azr = sp.ndimage.zoom( az, factor, order=spl_order, 
                           mode=mode, prefilter=prefilter ) 
    
    if IsNan == True :
        az [ inan ] = np.nan
        inanr = sp.ndimage.zoom( inan*1, factor, order=0, 
                                 mode=mode, prefilter=prefilter ).astype(bool)
        azr[ inanr ] = np.nan
    
    if plot == True :
        
        if dimention == '2D' :
            plta( az, sbplt=[1,2,1], tit='Original', vmin=vmin, vmax=vmax )
            plta( azr, sbplt=[1,2,2], tit='Resampled', vmin=vmin, vmax=vmax )
            
        if dimention == '1D' :
            plt.figure()
            plt.plot( az, c='k', label='Original' )
            plt.plot( azr, c='b', label='Resampled' )
            
    if IsXy == True :    
        axr = sp.ndimage.zoom( ax, factor, order=1 )  
        ayr = sp.ndimage.zoom( ay, factor, order=1 ) 
        
        if nan == False :
            axr = axr[ ~inanr ]
            ayr = axr[ ~inanr ]
            azr = axr[ ~inanr ]
        
        return [ axr, ayr, azr ]
    
    else :
        
        return azr
    
# -----------------------------------------------------------------------------
def neighboring_points( points1, 
                        points2, 
                        radius,
                        method='circle',
                        idx1=None,
                        idx2=None, 
                        plot=False, s=None):
    """
    Find points in points1 that are within a specified radius of any point in points2.

    Args:
        points1 (tuple/list of arrays): A tuple containing arrays of coordinates for points1.
        points2 (tuple/list of arrays): A tuple containing arrays of coordinates for points2.
        radius (float): The radius within which points are considered neighbors.
        method (str, optional): The method for finding neighbors, it can be 'circle' or 'box'. Default is 'circle'.
        plot (bool, optional): Whether to create a scatter plot to visualize the selected points. Default is False.
        s (int, optional): Marker size for plotting. Default is None.

    Returns:
        tuple: A tuple containing arrays of coordinates for the selected points and a boolean array indicating selection.
    
    Example:
        X, Y = np.meshgrid(np.arange(0, 30, 1), np.arange(0, 30, 1))  # np is: "import numpy as np"
        plt.figure()  # plt is: "import matplotlib.pyplot as plt"
        idx = neighboring_points((10, 10), (X, Y), 5, plot=True, method='circle')
    """

    # Convert input arrays to NumPy arrays
    points1 = np.array( points1 )
    points2 = np.array( points2 )

    # Select points based on the provided indices
    if idx1 is not None:
        points1 = points1[ :, idx1 ]
    if idx2 is not None:
        points2 = points2[ :, idx2 ]

    # Ensure points1 and points2 have the same number of dimensions
    assert points1.shape[0] == points2.shape[0], "points1 and points2 must have the same number of dimensions"

    # Store the original shape of points1
    shape = points1.shape[1:]

    # Finding neighboring points based on the chosen method
    if method == 'circle':
        # Create a meshgrid of differences between points1 and points2 for each dimension
        differences = [np.subtract.outer(points2[i].ravel(), points1[i].ravel()) for i in range(points1.shape[0])]

        # Calculate distances using vectorized operations
        distances = np.sqrt(sum(diff**2 for diff in differences))

        # Find indices where distances are less than or equal to the radius
        is_neighbor = distances < radius

        # Use advanced indexing to select the points that meet the condition
        selected_points = [points1[i].ravel()[is_neighbor.any(axis=0)] for i in range(points1.shape[0])]

        # Reshape the boolean array to match the shape of points1
        idx = is_neighbor.any(axis=0).reshape(shape)

    elif method == 'box':
        # Check if points2 are within a bounding box defined by points1 and the radius
        idx = np.all([(points1[i] > np.nanmin(points2[i]) - radius) & 
                      (points1[i] < np.nanmax(points2[i]) + radius) for i in range(points1.shape[0])], axis=0)

        selected_points = [points1[i][idx] for i in range(points1.shape[0])]

    if idx1 is not None:
        idx_new = np.full( np.size( idx1 ), False )
        idx_new[ idx1 ] = idx
        idx = idx_new

    if plot and points1.shape[0] > 1:
        # Plot the original points (points2) and the selected points
        plt.scatter(points2[0], points2[1], c='b', s=s, label='points2')
        plt.scatter(selected_points[0], selected_points[1], c='r', s=s, label='Selected Points')
        plt.scatter(points1[0], points1[1], c='k', s=s, label='points1', marker='x')
        plt.legend()
        plt.title('Selected Points')
        plt.show()

    return tuple(selected_points), idx

# -----------------------------------------------------------------------------
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = sp.spatial.Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = MultiLineString(edge_points)
    triangles = list( shapely.ops.polygonize(m) )
    
    return shapely.ops.cascaded_union(triangles), edge_points
    
# -----------------------------------------------------------------------------
def xy2convexhull( x, y, plot=False, close=True, size=0.5, color='b' ) :

    points = np.column_stack( ( x.ravel(), y.ravel() ) ) 
    
    hull = sp.spatial.ConvexHull( points )    
    
    xh, yh = points[hull.vertices,0], points[hull.vertices,1]
    
    if close is True :
        xh, yh = np.append( xh, xh[0] ), np.append( yh, yh[0] ) 
    
    if plot is True :
        
        plt.scatter( x, y, s=size, c=color  )
        plt.plot(xh, yh, 'r--', lw=2)
        
    return [ hull, xh, yh ]    

# -----------------------------------------------------------------------------
def xy_in_hull( x, y, hull, buffer=0, plot=False ) :

    original_shape = np.copy( x.shape )
    
    x, y = x.ravel(), y.ravel()
    
    if type( hull ) in ( list, tuple ) :
        xp, yp = np.asarray( hull[0] ), np.asarray( hull[1] )
        hull,_,_ = xy2convexhull( xp, yp )
            
    xy = np.column_stack( ( x, y ) )    
    
    in_hull = np.all( np.add(np.dot(xy, hull.equations[:,:-1].T ),
                      hull.equations[:,-1] ) <= buffer, axis=1 )
    
    xh, yh = xp[hull.vertices], yp[hull.vertices]
    xh, yh = np.append( xh, xh[0] ), np.append( yh, yh[0] ) 

    x, y = x.reshape( original_shape ), y.reshape( original_shape )
    in_hull = in_hull.reshape( original_shape ) 
    
    if plot == True :
        plt.plot( xh, yh, 'r--', lw=2 )
        plt.scatter( x[in_hull], y[in_hull] )
    
    return in_hull, x[in_hull], y[in_hull] 


# -----------------------------------------------------------------------------
def mask2hull( mask, nan=None, plot=False, close=True, size=0.5, color='b' ) :
    
    if type( mask ) in ( list, tuple ) :
        x, y, mask = mask
    else:
        x = np.arange( mask.shape[1] )
        y = np.arange( mask.shape[0] )
        x, y = np.meshgrid( x, y )
        
    if mask.dtype != 'bool' :
        if nan is not None :
            mask = mask != nan
        else :
            mask = np.isfinite( mask )

    x = x[ mask ]
    y = np.flipud(y)[ mask ]  
    
    hull_list = xy2convexhull( x, y, plot=plot, close=close, size=size, color=color )
    
    return hull_list
    
# -----------------------------------------------------------------------------
def filt_voids( xyzgrid, xyr, step=None, plot=False, vmin=None, vmax=None, 
                method='mean' ) :  
    
    _, isf = mask2D( xyr, xyzgrid, convexhull=True ) 
#    _, isf2 = mask2D( xyr, xyzgrid ) 
    array = np.copy( xyzgrid[2] )
    m = np.argwhere( isf )
#    plta( m*1.0, lim=(xyzgrid[0],xyzgrid[1]), points=(xyr[0],xyr[1]) )
    
    xx, yy, zz = xyzgrid
    xp, yp, r = np.copy( xyr )
    x, y = np.concatenate( ( xp, xx[~isf] ) ), np.concatenate( ( yp, yy[~isf] ) )
    
    if method == 'mean' :
        func = np.nanmean
    if method == 'median' :
        func = np.nanmedian
        
    if step is None : 
        step = min_dist( x, y )['mean']
    
    for i,j in m :
        isin = np.full( x.shape, False )
        inc = 0
        while  np.sum( isin ) < 3 :
            radius = r + step * inc    
            isin = ( x > xx[i,j] - radius ) & ( x < xx[i,j] + radius ) & \
                   ( y > yy[i,j] - radius ) & ( y < yy[i,j] + radius )
            inc += 1
            
        if inc == 0 : 
            continue
        
        win = ( xx > xx[i,j] - radius ) & ( xx < xx[i,j] + radius ) & \
              ( yy > yy[i,j] - radius ) & ( yy < yy[i,j] + radius )
    
        
        array[ i,j ] = func( array[ win ] )
        
    
    if plot == True :
        plta( xyzgrid[2], lim=(xyzgrid[0],xyzgrid[1]), points=(xyr[0],xyr[1]), 
              vmin=vmin, vmax=vmax, tit='Original', sbplt=[1,3,1] ) 
        plta( array, lim=(xyzgrid[0],xyzgrid[1]), points=(xyr[0],xyr[1]), 
              vmin=vmin, vmax=vmax, tit='Filtered', sbplt=[1,3,2] )
        plta( xyzgrid[2]-array, lim=(xyzgrid[0],xyzgrid[1]), points=(xyr[0],xyr[1]), 
              vmin=None, vmax=None, tit='Differences', sbplt=[1,3,3] )        
        
    return array

# -----------------------------------------------------------------------------
def mosaic_array( array_list, plot=False, vmin=None, vmax=None, ref_shape=0,
                 spl_order=1) :
    
    shape = array_list[ ref_shape ].shape
    
    array_mos = np.copy( array_list[ ref_shape ] )
    
    new_list = []
    
    for i, a in enumerate( array_list ) :
        shape_ratio = shape[0]/a.shape[0], shape[1]/a.shape[1]
        ar = sp.ndimage.zoom( a, shape_ratio, order=spl_order )
        new_list.append( ar )
        isn = np.isnan( array_mos ) 
        array_mos[isn] = ar[isn]
        
    if plot == True :
        plta( array_mos, vmin=vmin, vmax=vmax)    
        
    return array_mos, new_list

# -----------------------------------------------------------------------------
def del_in_lim( x, y, array, lim, remove='in' ) :
    
    ar = np.copy( array )
    idx = xy_in_lim( x, y, lim )[2]
    
    if remove == 'in' :
        ar = ar[ ~idx ] 
        x = x[ ~idx ]
        y = y[ ~idx ]
    
    if remove == 'out' :
        ar = ar[ idx ]    
        x = x[ idx ]
        y = y[ idx ]        
        
    return [x,y,ar]
    
# -----------------------------------------------------------------------------
def XY2bbox( X, Y, sx=0, sy=None ) :
    
    if sy is None : 
        sy = sx
    
    minbbX = np.nanmin( X ) - sx/2
    maxbbX = np.nanmax( X ) + sx/2
    
    minbbY = np.nanmin( Y ) - sy/2
    maxbbY = np.nanmax( Y ) + sy/2    
    
    bbox = [ minbbX, maxbbX, minbbY, maxbbY ]
    
    return bbox

# -----------------------------------------------------------------------------
def julian2date( jday, year, array=True, array_type=dict ) :
    
    if ( type(jday) is int ) and ( type(year) is int ) :
        date = datetime.datetime(int(year), 1, 1)+datetime.timedelta(days=int(jday) -1) 
        
    else :
        date = []
        for y, d in zip( year, jday ) :
            date.append( datetime.datetime(int(round(y)), 1, 1)+\
                         datetime.timedelta(days=int(round(d)) -1) )
            
        if array is True :
            if array_type == 'datetime64[D]' :
                date = np.array( date, dtype='datetime64[D]' )
            if array_type == str :
                date = np.array( date, dtype='datetime64[D]' )
                date = date.astype( str )
            if array_type == dict :
                date = np.array( [ [ x.year, x.month, x.day ] for x in date ] )
                date = { 'year':date[:,0], 'month':date[:,1], 'day':date[:,2] }
            if array_type == 'array' :
                date = np.array( [ [ x.year, x.month, x.day ] for x in date ] )

    return date

# -----------------------------------------------------------------------------
def date2julian( yy, mm, dd ) :
    
    if type( yy ) in ( float, int ) :
        yy = np.array( [ yy ] )
        mm = np.array( [ mm ] )
        dd = np.array( [ dd ] )
        
    jday = np.zeros( yy.shape )    
    for i,_ in enumerate( yy ):
        
        day = datetime.date( yy[i], mm[i], dd[i] ) 
        day = day.toordinal() + 1721424.5
        jday[i] = day
    
    return jday

# -----------------------------------------------------------------------------
def date2datetime( date, time, fmt='%d-%m-%y %H:%M:%S', array=True,
                   array_type='datetime64[ms]' ) :
    
    if ( type( date ) is str ) and ( type( time ) is str ) :
        date_time_str = date + ' ' + time 
        date_time = datetime.strptime( date_time_str, fmt )
        
    else :   
        date_time = []
        for d, t in zip( date, time ) :
            date_time_str = d + ' ' + t 
            date_time.append( datetime.datetime.strptime( date_time_str, fmt ) ) 
            
        if array is True :
            # a = pd.to_datetime( date_time )
            np_date_time = np.array( date_time, dtype=array_type )   
            
        if array is True :
            if array_type == 'datetime64[D]' :
                date = np.array( date, dtype='datetime64[D]' )
            if array_type == str :
                date = np.array( date, dtype='datetime64[D]' )
                date = date.astype( str )
            if array_type == dict :
                date = np.array( [ [ x.year, x.month, x.day ] for x in date ] )
                date = { 'year':date[:,0], 'month':date[:,1], 'day':date[:,2] }
            if array_type == 'array' :
                date = np.array( [ [ x.year, x.month, x.day ] for x in date ] )
    
    return np_date_time

# -----------------------------------------------------------------------------
def print_table( table, 
                 headers =None, 
                 space = 12, 
                 decimals = 2, 
                 rows=[], 
                 cols=[],
                 return_array  =False, 
                 idx = None, 
                 title = None,
                 row_index = True,
                 col_index = True, 
                 center_title = False,
                 return_str = False,
                 colsxrow = 100,
                 path_name = None, 
                 printf = True, 
                 reshape = None, 
                 print_tot_rows = True ):
    """
    Prints a table of values with optional headers, formatting, and row/column selection.

    Args:
        - table (list, tuple, dict, numpy.ndarray): The table of values to be printed.
        - headers (list, optional): The headers for each column. Defaults to None.
        - space (int, optional): The width of each column. Defaults to 12.
        - decimals (int, float, list, optional): The number of decimal places to display for each column.
            If a single value is provided, it will be applied to all columns. If a list is provided,
            each value will be applied to the corresponding column. Defaults to 2.
        - rows (int, list, optional): The indices of the rows to be included in the table. Defaults to [].
        - cols (int, list, optional): The indices of the columns to be included in the table. Defaults to [].
        - idx (boolean array, optional): The indices of the rows to be included in the table. Defaults to None.
        - title (str, optional): The title to be printed above the table. Defaults to None.
        - row_index (bool, optional): Whether to include a column of row numbers. Defaults to True.
        - col_index (bool, optional): Whether to include a row of column numbers. Defaults to True.
        - center_title (bool, optional): Whether to center the title above the table. Defaults to False.
        - return_str (bool, optional): Whether to return the table as a string. Defaults to False.
        - return_array (bool, optional): Whether to return the table as a numpy array. Defaults to False.
        - colsxrow (int, optional): The maximum number of columns to be printed per row. Defaults to 6.
        - path_name (str, optional): The path and name of the file to save the table. Defaults to None.
        - printf (bool, optional): Whether to print the table. Defaults to True.
        - reshape
        - print_tot_rows (bool, optional): Whether to print the total number of rows at the end of the table. Defaults to True.

    Returns:
        - None: If return_str is False and return_array is False.
        - str: If return_str is True.
        - numpy.ndarray: If return_array is True.
    """

    # Initialize an empty string to hold the output
    output = ""

    table = copy.deepcopy( table )

    # Check if the input table is a numpy array
    if type(table) == np.ndarray:
        # Create a copy of the table
        table = table

    if cols is None :
        cols = []

    # Check if the input table is a dictionary
    if type(table) == dict:
        table = copy.copy(table)
        # If there are keys to exclude, create a copy of the table and remove those keys
        if cols != []:
            for i, c in enumerate(cols):
                if type(c) == str:
                    cols[i] = list(table.keys()).index(c) 
        for k in table.keys():
            if len( k ) > space:
                space = len( k ) + 2
        # Convert the dictionary to an array
        table, headers = dict2array( table )

    # Check if the input table is a list or a tuple
    if type(table) in (list, tuple):
        # Convert each element of the table to a numpy array and stack them column-wise
        for i, t in enumerate(table):
            if type(t) in ( list, tuple, int, float ):
                ti = np.array(t)
            if i == 0:
                array = ti
            else:
                array = np.column_stack((array, ti))
        # Replace the original table with the newly created 2D array
        table = array.copy()

    # If the table is a 1D array and headers are not provided, reshape it into a 2D array
    if ( len(table.shape) == 1 or (len(table.shape) == 2 and table.shape[0] == 1) ) and ( not headers ):
        table = table.reshape(-1, 1)

    # If the table is a 1D array and headers are provided, reshape it into a 2D array 
    # with one row and as many columns as the length of the headers
    if ( len(table.shape) == 1 or (len(table.shape) == 2 and table.shape[0] == 1) ) and headers :
        table = table.reshape(1, len(headers) )

    # If reshape is provided, reshape the table accordingly
    if reshape is not None :
        table = table.reshape( reshape )

    # If decimals is a single number, create a list of that number repeated for each column of the table
    if type(decimals) in (int, float):
        decimals = [decimals] * len(table[0])

    # Create a boolean array indicating whether each element of the table is an integer
    is_int = np.array([[isinstance(val, (int, float)) and val % 1 == 0 for val in row] for row in table])

    # If idx is not None, select the specified rows from the table
    if idx is not None :
        table = table[idx]

    if ( rows is not None ) and ( rows != [] ) and ( rows is not False ) :
        # If rows or cols is a single number, create a list of numbers up to that number
        if isinstance(rows, int):
            rows = list(range(rows))
        if isinstance(rows, np.ndarray):
            if rows.dtype == bool:
                rows = np.where(rows)[0].tolist()

    if ( cols is not None ) and ( cols != [] ) :

        if headers is not None:
            headers = [ headers[i] for i in cols ]
            
        table = table[:, cols]

    # If rows or cols is not specified, 
    # create a list of numbers up to the number of rows or columns in the table    
    if not rows:
        if isinstance(table[0], str):
            rows = list(range(len(table))) 
        else:
            rows = list(range(len(table[:,0])))
        
    if not cols:
        if isinstance(table[0], str):
            cols = list(range(len(table))) 
        else:
            cols = list(range(len(table[0])))  

    # If row_index is True, add a column of row numbers to the table
    len_row_index = len( str( len( rows ) ) ) 

    if row_index:
        starting_space_len = len_row_index + 1
        starting_space_str = " " * ( starting_space_len )
    else:
        starting_space_len = 0
        starting_space_str = ""

    # Print the table headers
    if headers is None : 
        if col_index:
            output += starting_space_str
            for j in cols:
                output += f"% {space}d" % j
            output += "\n"
        output += starting_space_str
        for j in range(len(table[0])):
            output += "-"*space
        output += "\n"

    else :
        output += starting_space_str
        for j in headers:
            output += f"% {space}s" % j
        output += "\n"
        if col_index:
            output += starting_space_str
            for j in cols:
                output += f"% {space}d" % j
            output += "\n"
        output += starting_space_str
        for j in range( len( table[0] ) ):
            output += "-"*space
        output += "\n"     

    # Print the contents of the table
    for i in range( table.shape[0] ):

        if i not in rows:
            continue

        if row_index:
            output += f"%{len_row_index}d|" % (i) # Row nums
        
        for j in range( table.shape[1] ):

            if is_number( table[i][j] ):

                num_i = float( table[i][j] )
                
                if is_int[i][j]:
                    
                    num_i = int( num_i )
                    
                    ft = 'd'

                else:

                    ft = f".{decimals[j]}f"
                
                output += f"% {space}{ft}" % ( num_i )
            
            else:
                # Truncate the string if it's longer than `space`
                str_val = str(table[i][j])
                if len(str_val) > space:
                    str_val = str_val[:space-1]

                output += f"% {space}s" % (str_val)

        output += "\n"

    if not colsxrow:
        colsxrow = table.shape[1]

    if table.shape[1] > colsxrow:

        sections = range(0, table.shape[1], colsxrow)
        out_section = ""

        # Split the string into lines
        lines = output.split('\n')

        for i, section in enumerate(sections):

            if i == 0 :
                for line in lines:
                    chunck = line[ : space *colsxrow + starting_space_len ] + '\n'
                    out_section += chunck

            else:
                for line in lines:
                    chunck = line[ section * space + starting_space_len : 
                                   section * space + space * colsxrow+starting_space_len ] +'\n'
                    if row_index:
                        chunck = line[ : starting_space_len ] + chunck
                    out_section += chunck

        output = out_section


    # If a title is provided, print it centered above the table
    if title is not None:
        if center_title:
            title_str = "{:^{}}\n".format(title, space * len(table[0]))
        else:
            title_str = title + "\n"
        output = title_str + output

    if printf is True :

        print(output)

        if print_tot_rows is True:

            print( "\nTotal rows: " + str(table.shape[0]) + '\n' )

    if ( path_name is not None ) and ( path_name is not False ) :
        with open( path_name, 'w' ) as f:
            f.write( output )
        f.close()

    # If return_str is True, return the output string
    if return_str is True :
        return output

    # If return_str is True, return the table
    if return_array is True :
        return table


# -----------------------------------------------------------------------------
def combine64(years, months=1, days=1, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, 
              nanoseconds=None ):

    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1

    types = ('<M8[Y]', '<m8[M]', '<m8[D]',  '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')

    if seconds is not None:
        if np.any(np.mod(seconds, 1) != 0):
            nanoseconds = seconds.copy() * 1e9
            seconds = None

    vals = ( years, months, days, hours, minutes, seconds,
             milliseconds, microseconds, nanoseconds )

    datetime_type = np.sum( np.asarray(v, dtype=t) for t, v in zip(types, vals)
                            if v is not None)

    return datetime_type

# -----------------------------------------------------------------------------
def read_file( file, rows=None, printf=False ) :
    
    f = open( file, 'r' )
    
    lines = [line.rstrip() for line in f]
    
    if type( rows ) == int :
        rows = list( range( rows ) )
        
    if ( rows == [] ) or ( rows is None ) :
        rows = list( range( len( lines ) ) )
        
    lines = [ lines[index] for index in rows ]
    
    if printf is True :
        lines_str = '\n'.join( lines )
        print( lines_str )
    
    f.close()
    
    return lines

# -----------------------------------------------------------------------------
def open_dir( path ) :
    
    if os.name == 'nt' :
        os.system( 'start ' + path )
        
    if os.name == 'posix' :
        os.system( 'nautilus ' + path + '&' )
        
# -----------------------------------------------------------------------------
def del_files( path, string ) : 
    
    for file in os.listdir( path ) :
         
        if string in file :
            
            if os.path.isfile( path + os.sep + file ) :
            
                os.remove( path + os.sep + file )
                
            if os.path.isdir( path + os.sep + file ) :
            
                os.rmdir( path + os.sep + file )
            
# -----------------------------------------------------------------------------
def ls( path ) : 
    
    for file in os.listdir( path ) :
        
        print( file )    
        
# -----------------------------------------------------------------------------
def xyz2grid( x, y, z, lim=None, extend=None, extend_method='percentage', 
              sqr_area=False, gstep=None, blkm=None, method='linear', prjcode_in=4326,
              prjcode_out=4326, filt=None, filt_radius=1, msk_radius=None, msk_shp=None,
              in_out='in', fill_nan=None, plot=False, vmin=None, vmax=None, adjust_lim=True,
              pltxy=False, s=None, msk_arr=None, adjst_lim=False, iter=1, tension_factor=0.35,
              filt_factor=2, filt_sigma=1, padw=0, pmode='gdal', fillnan_mask=None ):

    if prjcode_in != prjcode_out:
        x, y = prjxy( prjcode_in, prjcode_out, x, y )

    if lim is None:
        lim = [ np.min(x), np.max(x), np.min(y), np.max(y) ]
        xl, yl, zl = x, y, z
    else:
        xl, yl, indx = xy_in_lim( x, y, lim, extend=33 )
        zl = z[indx]

    if extend is not None:
        lim = extend_lim(lim, extend, extend_method, sqr_area )

    if gstep == None:
        gstep = min_dist( xl, yl )['mean']

    if adjust_lim is True :
        lim[0], lim[2] = lim[0] - gstep/2, lim[2] - gstep/2
        lim[1], lim[3] = lim[1] + gstep/2, lim[3] + gstep/2

    if blkm is not None :    
        if blkm == True:
            xl, yl, zl = block_m( xl, yl, zl, gstep, lim=lim )
        if type( blkm ) in ( int, float ) :
            xl, yl, zl = block_m( xl, yl, zl, blkm, lim=lim )
    
    if adjst_lim is False :    
        xg = np.arange( lim[0]-gstep/2, lim[1]+gstep/2, gstep )
        yg = np.arange( lim[3]-gstep/2, lim[2]+gstep/2, -gstep )

    if adjst_lim is True :   
        xg = np.linspace( lim[0], lim[1], int( ( lim[1] - lim[0] ) / gstep ) )
        yg = np.linspace( lim[3], lim[2], int( ( lim[3] - lim[2] ) / gstep ) )

    xx, yy = np.meshgrid(xg, yg)
       
#    points = np.column_stack((xl, yl))
    if method == 'surface':
        xxs, yys, zzs = gmt_surf( xl, yl, zl, lim=extend_lim(lim, 10), grid_step=gstep, 
                                  tension_factor=tension_factor )
        zz = xyz2xy( ( xxs, yys, zzs ), (xx, yy), method='linear', fillnan=False )
    else:
#        zz = scy.interpolate.griddata(points, zl, (xx, yy), method=method)
        zz = xyz2xy( ( xl, yl, zl ), (xx, yy), method=method, fillnan=False )

    if fill_nan is not None :
        if fillnan_mask is not None :
            zz = mask2D( ( xl, yl, fillnan_mask ), (xx, yy, zz) )[0]
        if ( type(fill_nan) == bool ) and ( fill_nan is True ) :
            fill_nan = 'gdal'
        zz = fillnan( zz, xy=( xx, yy ), method=fill_nan, iter=iter ) 
        
    if filt is not None :
        zz = filt2d( zz, ftype=filt, iter=iter, radius=filt_radius, 
                         factor=filt_factor, sigma=filt_sigma, padw=padw, pmode=pmode )        
    
    if msk_radius is not None:
        zz = mask2D( ( xl, yl, msk_radius ), (xx, yy, zz) )[0]
        
    # if msk_shp is not None:
    #     zz = rt.mask_array( xx, yy, zz, msk_shp, prjcode=prjcode_out )

    if msk_arr is not None:
        zz[np.isnan( msk_arr ) ] = np.nan         

    if plot == True:

        if pltxy == False:
            plta( zz, vmin=vmin, vmax=vmax, cmap='rainbow', lim=(xx,yy) )
        if pltxy == True:
            plta( zz, vmin=vmin, vmax=vmax, cmap='rainbow', lim=(xx,yy), points=[ xl, yl ] )
            
    return [ xx, yy, zz ], [ x, y, z ]
    
# -----------------------------------------------------------------------------
def relative_max_min_2D( array, n=2 ) :
    
    ar = np.copy( array ) 
    MD = np.zeros(ar.shape)
    
    xu = np.arange( 0, ar.shape[1] )
    yu = np.arange( 0, ar.shape[0] )
    
    xx, yy = np.meshgrid( xu, yu )    

    # -------------------------------------------------------------------------
    # Edge points generation 
    
    for i, j in np.ndindex(ar.shape):
        
        if i==0 or i==ar.shape[0]-1 or \
           j==0 or j==ar.shape[1]-1:
            continue
        
        Np=0
        Nn=0
        
        if ar[i-1, j-1] < ar[i, j] > ar[i+1, j+1]:
            Np +=1
        if ar[i-1, j] < ar[i, j] > ar[i+1, j]:
            Np +=1
        if ar[i-1, j+1] < ar[i, j] > ar[i+1, j-1]:
            Np +=1
        if ar[i, j-1] < ar[i, j] > ar[i, j+1]:
            Np +=1
        
        if ar[i-1, j-1] > ar[i, j] < ar[i+1, j+1]:
            Nn +=1
        if ar[i-1, j] > ar[i, j] < ar[i+1, j]:
            Nn +=1
        if ar[i-1, j+1] > ar[i, j] < ar[i+1, j-1]:
            Nn +=1
        if ar[i, j-1] > ar[i, j] < ar[i, j+1]:
            Nn +=1            
            
        if Np > n :
            MD[i,j] = 1
        if Nn > n :
            MD[i,j] = -1
            
    xmin, ymin = xx[ MD==-1 ], yy[ MD==-1 ]
    xmax, ymax = xx[ MD==1 ], yy[ MD==1 ]
    ax = plta( ar )
    ax.scatter( xmin, ymin )
    ax.scatter( xmax, ymax )
        
    return MD


# -----------------------------------------------------------------------------
def rotate_xy( x, y, radians, shift=True, fix_point='first', plot=False ) :

    if shift is not None :
        
        if shift in ( tuple, list ) :    
            x = x - shift[0]
            y = y - shift[1]
            
        if shift is True :
            xu = np.unique( x )
            idx = x == xu[0]
            
            if type( fix_point ) == str :
                
                if fix_point == 'mean' : 
                    shift = np.mean(x), np.mean(y)
                    
                if fix_point == 'first' :
                    shift = x[ idx ][0], y[ idx ][0]
                    
            if type( fix_point ) in ( tuple, list ) :
                shift = fix_point

            x = x - shift[0]
            y = y - shift[1]
    
    else :
        
        shift = 0, 0
    
    fix_point = shift
    
    xr = x * np.cos( radians ) + y * np.sin( radians ) + shift[0]
    yr = -x * np.sin( radians ) + y * np.cos( radians ) + shift[1]
    
    if plot is True :
        
        plt.scatter( x, y, c='b' )
        plt.scatter( xr, yr, c='b' )

    return xr, yr, fix_point

# -----------------------------------------------------------------------------
def grid_rot_angle( x, y, plot=False, treshold=1e-5, 
                    fix_point='first', decimals=None ) :
    
    xu = np.unique( x )
    
    idx0 = x == xu[0]
    idx1 = x == xu[1]
    
    x0 = x[ idx0 ][0]
    y0 = y[ idx0 ][0]
    
    x1 = x[ idx1 ][0]
    y1 = y[ idx1 ][0]
    
    angle = np.abs( np.arctan( ( y1-y0 ) / ( x1-x0 ) ) )
    
    xr, yr, fix_point = rotate_xy( x, y, angle, fix_point=fix_point )
    
    diffx = np.abs( xr[ idx0 ][0] - xr[ idx1 ][0] )
    diffy = np.abs( yr[ idx0 ][0] - yr[ idx1 ][0] )
    
    if ( diffx > treshold ) and ( diffy > treshold ) :
    
        xr, yr, fix_point = rotate_xy( x, y, (np.pi/2)-angle, fix_point=fix_point )   
        
        diffx = np.abs( xr[ idx0 ][0] - xr[ idx1 ][0] )
        diffy = np.abs( yr[ idx0 ][0] - yr[ idx1 ][0] )
        
        if ( diffx > treshold ) and ( diffy > treshold ) :
            
            print( "SOMETHING IS WRONG WITH THE POINTS ROTATION!!!")
            
            plot = True 
            
    if decimals is None :
        decimals = int( np.abs( np.floor( np.log10( treshold ) ) ) ) - 1
    
    xr = np.round( xr, decimals )
    yr = np.round( yr, decimals )
    xru = np.unique( xr )
    yru = np.unique( yr )
    print(yr )

    Xr, Yr = np.meshgrid( xru, yru )
    shape = Xr.shape
    print( shape )

    xr = Xr.ravel()
    yr = Yr.ravel()

    if plot == True :

        plt.close('Roteded points')
        plt.figure("Roteded points")
        plt.scatter( x.ravel(), y.ravel(), c='b', label='original' )        
        plt.scatter( xr.ravel(), yr.ravel(), c='r', label='rotated' )
        plt.legend()
        plt.show()
        
    return angle, xr, yr, fix_point, shape
    
# -----------------------------------------------------------------------------
def mesh3Dmodel( x=None, y=None, z=None, lim=None, step=None, plot=False, 
                 sbplt=111, out_type='dictionary', size=None, centre_xy=False, 
                 xm=None, ym=None, treshold=1e-5, fix_point='first' ) :

    if step is None :
        x, y = np.copy( x ), np.copy( y )
        angle, x, y, fix_point = grid_rot_angle( x, y, treshold=treshold, fix_point=fix_point ) 
        xu = np.unique( x )
        yu = np.unique( y )
        zu = np.unique( z )

    if type(step) in (int, float) :
        step = step, step, step             

    if ( lim is not None ) and ( step is not None ) and ( centre_xy is True ) :
        if xm is None :
            xm = np.round( np.mean( ( lim[1], lim[0] ) ) )
        xr = np.max( (int((xm-lim[0])/step[0]), int((lim[1]-xm)/step[0]) ) )
        lim[0] = xm - step[0] * xr
        lim[1] = xm + step[0] * xr
        if ym is None :
            ym = np.round( np.mean( ( lim[2], lim[3] ) ) )
        yr = np.max( (int((ym-lim[2])/step[1]), int((lim[3]-ym)/step[1]) ) )
        lim[2] = ym - step[1] * yr
        lim[3] = ym + step[1] * yr

    if x is not None :
        xu = np.unique( x )
    else :
        xu = np.arange( lim[0], lim[1]+step[0], step[0] )

    if y is not None :
        yu = np.unique( y )
    else :
        yu = np.arange( lim[2], lim[3]+step[1], step[1] )

    if z is not None :
        zu = np.unique( z )
    else :
        zu = np.arange( lim[5], lim[4]-step[2], -step[2] )

    if lim is not None :
        xu = xu[ (xu>=lim[0]) & (xu<=lim[1]) ]
        yu = yu[ (yu>=lim[2]) & (yu<=lim[3]) ]
        zu = zu[ (zu>=lim[4]) & (zu<=lim[5]) ]

    X, Y, Z = np.meshgrid( xu, yu, zu )
    shape = X.shape

    if step is None :
        angle, X, Y, fix_point = grid_rot_angle( X, Y, treshold=treshold, fix_point=fix_point ) 

    if plot is True :
        if size is None :
            size = 1
        ax = plt.subplot(sbplt, projection='3d')
        ax.scatter( X[:,int(shape[1]/2),:].ravel(), 
                    Y[:,int(shape[1]/2),:].ravel(), 
                    Z[:,int(shape[1]/2),:].ravel(), s=size, c='b' )
        ax.scatter( X[int(shape[0]/2),:,:].ravel(), 
                    Y[int(shape[0]/2),:,:].ravel(), 
                    Z[int(shape[0]/2),:,:].ravel(), s=size, c='b' )

    if out_type == '3darray' :
        return X, Y, Z
    if out_type == '1darray' :
        return X.ravel(), Y.ravel(), Z.ravel(), shape
    if out_type == 'dictionary' :
        return { 'x':X.ravel(), 'y':Y.ravel(), 'z':Z.ravel() }, shape    

# -----------------------------------------------------------------------------
def despike_1D( array, treshold, radius=2, pmode='median', plot=False, kind='cubic' ) :

    signal = np.copy( array )

    psignal = np.pad( signal, radius, pmode)

    inan = np.full( psignal.shape, False ) 
    for i, e in enumerate( psignal ) :
        if ( i < radius ) or ( i > len( signal )-1 ) :
            continue

        meanw = np.mean( psignal[i-radius:i+radius] )
        if ( e > meanw + treshold ) or ( e < meanw - treshold ) :
            inan[ i ] = True 

    x = np.arange(1, len(psignal)+1, 1)
    f = sp.interpolate.interp1d( x[~inan], psignal[~inan], kind=kind )

    psignal[ inan ] = f( x[ inan ] )  

    new_signal = psignal[ radius : -radius ]

    if plot == True :

        plt.plot( signal, c='b', label='original' )
        plt.plot( new_signal, c='r', label='despiked' )
        plt.legend()

    return new_signal 


# -----------------------------------------------------------------------------
def xyzp2array3d( x, y, z, p, xi=None, yi=None, zi=None, lim=None, step=None, plot=False, 
                  sbplt=111, size=None, centre_xy=False, method='nearest' ) :
    
    """
    Convert coordinates (x, y, z) and a function (p) into a 3D numpy array.
    
    Args:
        x (array-like): x-coordinates.
        y (array-like): y-coordinates.
        z (array-like): z-coordinates.
        p (function or list or tuple): Function or list/tuple of functions.
        xi (array-like, optional): x-coordinates for the grid. Defaults to None.
        yi (array-like, optional): y-coordinates for the grid. Defaults to None.
        zi (array-like, optional): z-coordinates for the grid. Defaults to None.
        lim (array-like, optional): Limits of the grid. Defaults to None.
        step (float, optional): Step size for the grid. Defaults to None.
        plot (bool, optional): Whether to plot the 3D array. Defaults to False.
        sbplt (int, optional): Subplot specification for the plot. Defaults to 111.
        size (float or tuple, optional): Size of the plot. Defaults to None.
        centre_xy (bool, optional): Whether to center the x and y coordinates. Defaults to False.
        xm (float, optional): Midpoint of x. Defaults to None.
        ym (float, optional): Midpoint of y. Defaults to None.
        method (str, optional): Interpolation method. Defaults to 'nearest'.
    
    Returns:
        tuple: 3D numpy array (X, Y, Z) and the interpolated values (P).
    """
                  
    if xi is None :
        xi = x
    if yi is None :
        yi = y
    if zi is None :
        zi = z

    X, Y, Z = mesh3Dmodel( x=xi, y=yi, z=zi, lim=lim, step=step, out_type='3darray', 
                 centre_xy=centre_xy )
    
    if type(p) not in ( list, tuple ) :
        p = [ p ]

    P = []

    for i in p :
        Pi = sp.interpolate.griddata( ( x, y, z), i ( X, Y, Z ), method=method ) 
        P.append( Pi )

    if len( P ) == 1 :
        P = P[ 0 ]

    shape = X.shape

    if plot is True :
        if size is None :
            size = 1
        ax = plt.subplot(sbplt, projection='3d')
        ax.scatter( X[:,int(shape[1]/2),:].ravel(), 
                    Y[:,int(shape[1]/2),:].ravel(), 
                    Z[:,int(shape[1]/2),:].ravel(), s=size, c=P[ 0 ][:,int(shape[1]/2),:].ravel() )
        ax.scatter( X[int(shape[0]/2),:,:].ravel(), 
                    Y[int(shape[0]/2),:,:].ravel(), 
                    Z[int(shape[0]/2),:,:].ravel(), s=size, c=P[ 0 ][int(shape[0]/2),:,:].ravel() )
        
    if len( P ) == 1 :
        P = P[ 0 ]

    return X, Y, Z, P

# -----------------------------------------------------------------------------
def write_xyzp2vtk( x, y, z, p, path, name, nan2num=0 ):
    
    file_vtk = name + '.vtk'
    os.makedirs(path, exist_ok=True)  
    pf_new = os.path.join(path, file_vtk)

    idx = np.lexsort( ( x,y,z ) ) 
    p = p[ idx ]

    x_coordinates = np.unique( x )
    y_coordinates = np.unique( y )
    z_coordinates = np.unique( z )

    x_dim = np.size( x_coordinates )
    y_dim = np.size( y_coordinates )
    z_dim = np.size( z_coordinates )

    if nan2num is not None :
        inan = np.isnan( p )
        p[ inan ] = nan2num

    with open(pf_new, 'w') as fid:
        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('Gravity-Model %s\n' % name)
        fid.write('ASCII\n')
        fid.write('DATASET RECTILINEAR_GRID\n')
        fid.write('DIMENSIONS %d\t%d\t%d\n' % (x_dim, y_dim, z_dim))
        fid.write('X_COORDINATES %d double\n' % x_dim)
        fid.write(' '.join(map(str, x_coordinates)) + '\n')
        fid.write('Y_COORDINATES %d double\n' % y_dim)
        fid.write(' '.join(map(str, y_coordinates)) + '\n')
        fid.write('Z_COORDINATES %d double\n' % z_dim)
        fid.write(' '.join(map(str, z_coordinates)) + '\n')
        fid.write('POINT_DATA %d\n' % (x_dim * y_dim * z_dim))
        fid.write('SCALARS Density double\n')
        fid.write('LOOKUP_TABLE default\n')
        for ind_p in range( len( p ) ):
            fid.write('%f \n' % p[ind_p] )

    fid.close()  
     # File is automatically closed after the 'with' block

# -----------------------------------------------------------------------------
def dayyear2daymonth(day_numbers, years):
    """
    Convert day numbers to month numbers for a given year or years.

    Parameters:
    day_numbers (int or float or numpy.ndarray): Day numbers to convert to month numbers.
    years (int or float or numpy.ndarray): Year or years for which to convert day numbers to month numbers.

    Returns:
    numpy.ndarray: Month numbers corresponding to the input day numbers and years.
    """

    # If day_numbers is a single number, convert it to a numpy array
    if type(day_numbers) in (int, float):
        day_numbers = np.array([day_numbers])

    # If years is a single number, convert it to a numpy array
    if type(years) in (int, float):
        day_numbers = np.array([years])

    # Vectorize the datetime.datetime function
    datetime_vec = np.vectorize(datetime.datetime)

    # Create a date object for the first day of each year
    dates = datetime_vec(years, 1, 1)

    # Convert day_numbers to a list
    day_numbers_list = day_numbers.tolist()

    # Add day_numbers - 1 days to the dates
    dates = np.array([start + datetime.timedelta(days=int(day)) for start, day in zip(dates, day_numbers)])

    # Get the month numbers
    months = np.vectorize(lambda date: date.month)(dates)

    # Get the day of the month
    days = np.vectorize(lambda date: date.day)(dates)

    return days, months

# -----------------------------------------------------------------------------
def datetime2time(  dates, unit='s', ref_time='1970-01-01T00:00:00' ) :
    """
    Convert a numpy array of datetime64 objects to time since the Unix epoch.

    Parameters:
    dates (numpy.ndarray): Array of datetime64 objects.
    unit (str): Unit of time. Can be 's' (seconds), 'm' (minutes), 'h' (hours), 'D' (days), 'M' (months), or 'Y' (years).

    Returns:
    numpy.ndarray: Array of time since the Unix epoch in the specified units.
    """
    if unit == 'yy':
        # Convert datetime64 objects to days since the Unix epoch, then to years
        time = (dates - np.datetime64(ref_time)) / np.timedelta64(1, 'D')
        time /= 365.25 # Average length of a year in the Gregorian calendar
    elif unit == 'mm':
        # Convert datetime64 objects to days since the Unix epoch, then to months
        time = (dates - np.datetime64(ref_time)) / np.timedelta64(1, 'D')
        time /= 30.44 # Average length of a month in the Gregorian calendar
    elif unit == 'dd':
        # Convert datetime64 objects to days since the Unix epoch in days
        time = (dates - np.datetime64(ref_time)) / np.timedelta64(1, 'D')
    else:
        # Convert datetime64 objects to time since the Unix epoch in the specified units
        time = (dates - np.datetime64(ref_time)) / np.timedelta64(1, unit)

    return time

# -----------------------------------------------------------------------------
def get_sampling_rate(datetime_array, units='s'):
    """
    Calculate the sampling rate from a numpy datetime array.

    Parameters:
    datetime_array (numpy.ndarray): Array of datetime64 objects.
    unit (str): Unit of time for the sampling rate. Can be 'ns' (nanoseconds), 'ms' (milliseconds), 's' (seconds), 'm' (minutes), 'h' (hours), 'D' (days), 'M' (months), or 'Y' (years).

    Returns:
    float: Sampling rate in the specified unit.
    """
    # Calculate the time differences
    time_diffs = np.diff(datetime_array)

    # Find the most common difference
    most_common_diff = np.median(time_diffs)

    # Convert the most common difference to the specified unit and return the inverse
    units_to_seconds = {'ns': 1e-9, 'ms': 1e-3, 's': 1, 'm': 60, 'h': 3600, 'D': 86400, 'M': 2.628e+6, 'Y': 3.154e+7}
    most_common_diff_in_seconds = most_common_diff.astype('timedelta64[s]').item().total_seconds()
    most_common_diff_in_unit = most_common_diff_in_seconds / units_to_seconds[ units ]
    sampling_rate = 1 / most_common_diff_in_unit

    return sampling_rate

# -----------------------------------------------------------------------------
def normalize_moving_window( arr, window_size, a=0, b=1, smoothing_factor=1e-8 ):
    # Sostituisci i NaN con il valore di riempimento
    arr_filled = fillnan(arr)
    # Calcola la media mobile
    mean = sp.ndimage.uniform_filter(arr_filled, window_size)
    # Calcola la deviazione standard mobile
    std = np.sqrt(sp.ndimage.uniform_filter((arr_filled - mean)**2, window_size))
    # Aggiungi il fattore di smoothing alla deviazione standard per evitare la divisione per zero
    std += smoothing_factor
    # Normalizza l'array
    normalized_arr = (arr_filled - mean) / std
    # Scala l'array nell'intervallo [a, b]
    normalized_arr = a + (normalized_arr - np.nanmin(normalized_arr)) * (b - a) / (np.nanmax(normalized_arr) - np.nanmin(normalized_arr))
    # Ripristina i NaN
    normalized_arr = np.where(np.isnan(arr), np.nan, normalized_arr)
    
    return normalized_arr

# -----------------------------------------------------------------------------
def reshapexyz_1D_2D( x, y, z=None, flipud=False, fliplr=False, 
                      resamp=None, spl_order=1, mode='nearest',
                      plot=False, vmin=None, vmax=None):
    """
    Reshape 1D arrays `x`, `y`, and `z` (optional) into 2D arrays `X`, `Y`, and `Z`.
    
    Parameters:
        x (array-like): 1D array of x-coordinates.
        y (array-like): 1D array of y-coordinates.
        z (array-like, optional): 1D array of z-coordinates. Default is None.
        flipud (bool, optional): Whether to flip the resulting arrays vertically. Default is False.
        fliplr (bool, optional): Whether to flip the resulting arrays horizontally. Default is False.
        plot (bool, optional): Whether to plot the resulting 2D mesh. Default is False.
        vmin (float, optional): Minimum value for color mapping in the plot. Default is None.
        vmax (float, optional): Maximum value for color mapping in the plot. Default is None.
    
    Returns:
        X (ndarray): 2D array of reshaped x-coordinates.
        Y (ndarray): 2D array of reshaped y-coordinates.
        Z (ndarray, optional): 2D array of reshaped z-coordinates. Only returned if `z` is not None.
    """
    
    xa = np.array(x)
    ya = np.array(y)
    if z is not None:
        za = np.array(z)

    nx = np.unique(xa).size
    ny = np.unique(ya).size
    
    X = xa.reshape(nx, ny)
    Y = ya.resndimagehape(nx, ny)
    if z is not None:
        Z = za.reshape(nx, ny)

    if flipud is True:
        X = np.flipud(X)
        Y = np.flipud(Y)
        if z is not None:
            Z = np.flipud(Z)

    if fliplr is True:
        X = np.fliplr(X)
        Y = np.fliplr(Y)
        if z is not None:
            Z = np.fliplr(Z)

    if resamp is not None:
        X, Y, Z = resampling( array=[ X,Y,Z ], 
                              factor=resamp, 
                              spl_order=spl_order, 
                              mode=mode )

    if plot is True:
        plt.close("2D MESH")
        plt.figure("2D MESH")
        plt.scatter(X.ravel(), Y.ravel(), c=Z.ravel(), vmin=vmin, vmax=vmax)
        plt.show()

    return X, Y, Z

# -----------------------------------------------------------------------------
def fit_segmented_line( x, y, 
                        num_segments=None, 
                        threshold=1e-2, 
                        null_intercept=False,
                        keep_nodes=[], 
                        move_nodes=[],
                        place_nodes=[], 
                        plot=False ):
    """
    Fits a segmented line to the given data points.

    Parameters:

        - x (array-like): The x-coordinates of the data points.
      
        - y (array-like): The y-coordinates of the data points.
      
        - num_segments (int, optional): The number of line segments to fit. 
            If not specified, the algorithm will automatically determine the number of segments.
        
        - threshold (float, optional): The convergence threshold for the algorithm. 
            The algorithm stops when the difference in standard deviation between 
                iterations is below this threshold.
        
        - null_intercept (bool, optional): Whether to include a null intercept in the line fitting. 
            If True, the line will pass through the origin (0, 0).
        
        - plot (bool, optional): Whether to plot the fitted line segments.

    Returns:

        - array-like: An array of shape (num_segments, 2) 
            containing the x and y coordinates 
            of the fitted line segments.
    """

    if null_intercept :
        x = np.vstack([0] + x.tolist()).ravel()
        y = np.vstack([0] + y.tolist()).ravel()

    coefficients = np.polyfit( x, y, 1 ) 
    std = np.std( y )
    diff = std.copy()
    x_pol = np.linspace( 0, x.max(), 100 )
    y_pol = np.polyval( coefficients, x_pol )
    y_fit = np.polyval( coefficients, x )

    i = 0
    while diff > threshold :
        i += 1
        Mx = np.ones((x.shape[0], i+1 ))
        for j in range( i+1 ):
            Mx[:, j] = x.ravel()**j
        coefficients, _, _, _ = np.linalg.lstsq( Mx, y, rcond=None )
        y_fit = np.polyval( np.flip( coefficients ), x )
        res = y - y_fit
        stdi = np.std( res )
        diff = std - stdi
        std = stdi

    y_pol = np.polyval( np.flip( coefficients ), x_pol )

    # Calculate the first and second derivatives of y_pol
    y_pol_fd = np.diff( y_pol )[:-1]
    x_pol_fd = x_pol[1:-1] - ( x_pol[1] - x_pol[0] ) / 2
    y_pol_sd = np.diff( y_pol_fd )[:-1]
    x_pol_sd = x_pol_fd[1:-1] - ( x_pol_fd[1] - x_pol_fd[0] ) / 2

    y_pol_int = np.interp( x_pol_sd, x_pol_fd, y_pol_fd )

    # Calculate the curvature radius
    curvature = np.abs( y_pol_sd ) / ( 1 + y_pol_int**2 )**1.5
    # curvature = np.concatenate(([0,0], curvature, [0,0]))

    # Calculate the differences between consecutive elements
    diff = np.diff(curvature)

    # Find where the differences change sign
    sign_changes = np.diff(np.sign(diff))

    # The relative maxima are where the sign changes from positive to negative
    maxima_indices = np.where(sign_changes == -2)[0] + 1

    # Get the x_pol_sd values at the maxima
    curv_rel_max_x = x_pol_sd[maxima_indices]
    curv_rel_max = curvature[maxima_indices]

    if num_segments is None :
        num_segments = len( curv_rel_max ) + 1

    # Select changing points
    isort = np.flip( np.argsort( curv_rel_max ) )
    xes = np.sort( curv_rel_max_x[isort][: num_segments-1] )
    yei = np.interp( xes, x_pol, y_pol )
    xe = np.concatenate( ( [x_pol[0]], xes, [x_pol[-1]] ) ) 
    ye = np.concatenate( ( [y_pol[0]], yei, [y_pol[-1]] ) ) 
    segments = np.column_stack( (xe, ye) )

    if move_nodes :
        for i, mn in enumerate( move_nodes ) :
            ni = mn[0]
            shift = mn[1]
            print( segments[ni,0] )
            segments[ni,0] = segments[ni,0] + shift 
            print( segments[ni,0] )

    if keep_nodes :
        if 0 not in keep_nodes :
            keep_nodes.insert(0, 0)
        if np.size(xe)-1 not in keep_nodes :
            keep_nodes.append( np.size(xe)-1 )
        segments = segments[ keep_nodes ]

    if place_nodes :
        for i, pn in enumerate( np.column_stack( place_nodes ) ) :
            ni = int( pn[0] )
            xn = pn[1]
            yn = np.polyval( np.flip( coefficients ), xn )
            if i >= segments.shape[0]-1 :
                segments = np.vstack( (segments, [xn, yn]) )
            else :
                segments[ni,0], segments[ni,1] = xn, yn

    # Objetive function
    objf = lambda x, y, segments_x, segments_y :\
        np.sum( ( y - np.interp( x, segments_x, segments_y ) )**2 )

    # Gradient descent optimization
    def gradient_descent( objf, 
                          x, y, segments_x, segments_y, 
                          learning_rate=0.01, 
                          max_iterations=10000, 
                          tolerance=1e-6 ):
        
        """Gradient descent optimization."""
        
        residuals = []
        i = 0
        while True:
            gradients = np.zeros_like(segments_y)

            for j in range(len(segments_y)):
                segments_y[j] += learning_rate
                loss_plus = objf(x, y, segments_x, segments_y)
                segments_y[j] -= 2 * learning_rate
                loss_minus = objf(x, y, segments_x, segments_y)
                gradients[j] = (loss_plus - loss_minus) / (2 * learning_rate)
                segments_y[j] += learning_rate
            
            segments_y -= learning_rate * gradients
            
            residuals.append(objf(x, y, segments_x, segments_y))
            
            if i > 0 and abs(residuals[-1] - residuals[-2]) < tolerance:
                print(f'Converged after {i} iterations.')
                break
            i += 1
            if i >= max_iterations:
                print('Maximum number of iterations reached.')
                break

        return segments_y


    # Initial guess for the segmented line
    initial_guess = segments[:,1]

    # Perform gradient descent optimization
    optimized_segments_y = gradient_descent( objf, x, y, segments[:,0], initial_guess )


    # Combine segments_x with optimized_segments_y
    fitted_segments = np.column_stack((segments[:,0], optimized_segments_y))

    if plot == True :

        plt.figure()
        
        # Create a color map
        colors = cm.rainbow(np.linspace(0, 1, len(fitted_segments)-1))

        ymin = np.min( [ np.min(y), np.min(fitted_segments[:,1] ) ] )
        ymin = ymin - 0.1 * np.abs(ymin)
        for i in range(len(fitted_segments)-1):
            seg = fitted_segments[i:i+2]

            # Calculate the slope of the line
            slope = (seg[1,1] - seg[0,1]) / (seg[1,0] - seg[0,0])

            # Plot the line segment with a unique color
            plt.plot(seg[:,0], seg[:,1], color=colors[i], label=f'Slope: {slope:.2f}')

            # Add a vertical line from the end of the segment to the x-axis, except for the last segment
            if i < len(fitted_segments)-2:
                plt.vlines(seg[1,0], ymin, seg[1,1], linestyles='dashed')

                plt.text( seg[1,0], ymin, f'{seg[1,0]:.2f}', ha='left', va='bottom' )

        # Add a legenid
        plt.legend()

        # Plot the scatter plot with the grayscale color map
        plt.scatter(x, y, color='k', marker='o', facecolors='none')

        ymax = np.max( plt.gca().get_ylim() )
        plt.gca().set_ylim([ymin, ymax])

    return fitted_segments

# -----------------------------------------------------------------------------
def refine_edges(data, radius=1):
    # Create a structuring element, which is a square here
    struct_elem = sp.ndimage.generate_binary_structure(2, 1)
    
    # Perform erosion followed by dilation (opening operation)
    eroded = sp.ndimage.binary_erosion(data, structure=struct_elem, iterations=radius)
    refined = sp.ndimage.binary_dilation(eroded, structure=struct_elem, iterations=radius)
    
    return refined

# -----------------------------------------------------------------------------
def pad_model_3d( model, pad=1, step=100, 
                  plot=False, aspect='equal', vmin=None,
                  vmax=None ) :
    
    if type( pad ) == int : 
        pad = [ pad, pad, pad ]
    
    if type( step ) in ( int, float ) : 
        step = [ step, step, step ]
        
    xn = np.unique( model['x'] )
    yn = np.unique( model['y'] )
    zn = np.unique( model['z'] )

    xpad0 = np.linspace( xn[0], xn[0]-step[0]*pad[0], pad[0]+1 ) 
    xpad1 = np.linspace( xn[-1], xn[-1]+step[0]*pad[0], pad[0]+1 ) 
    xpad = np.concatenate( ( np.flip(xpad0[1:]), xn, xpad1[1:] ) )
    
    ypad0 = np.linspace( yn[0], yn[0]-step[1]*pad[1], pad[1]+1 ) 
    ypad1 = np.linspace( yn[-1], yn[-1]+step[1]*pad[1], pad[1]+1 ) 
    ypad = np.concatenate( ( np.flip(ypad0[1:]), yn, ypad1[1:] ) )
    
    zpad0 = np.linspace( zn[0], zn[0]-step[2]*pad[2], pad[2]+1 ) 
    zpad1 = np.linspace(zn[-1], zn[-1]+step[2]*pad[2], pad[2]+1 ) 
    zpad = np.concatenate( ( np.flip(zpad0[1:]), zn, zpad1[1:] ) )
    
    X, Y, Z = np.meshgrid( xpad, ypad, zpad )

    for i in model.keys() :
        if i in ('x', 'y', 'z') :
            continue
        model[i] = sp.interpolate.griddata( ( model['x'], model['y'], model['z'] ),
                                    model[i], ( X.ravel(), Y.ravel(), Z.ravel() ),
                                    method='nearest')
    
    # New model coordinates
    model['x'], model['y'], model['z'] = X.ravel(), Y.ravel(), Z.ravel()
    
        
    model['pad'] = ~( np.isin( model['x'], xn, assume_unique=False ) & \
                      np.isin( model['y'], yn, assume_unique=False ) & \
                      np.isin( model['z'], zn, assume_unique=False ) )
    
    return model

# -----------------------------------------------------------------------------
def costum_cmap(cmap_name, range_starts=[0], range_ends=[1], num_colors=256):
    """
    Create a custom colormap by modifying parts of an existing colormap.
    The colors at range_starts in the original colormap are mapped to range_ends in the new colormap.
    Colors below the first range_start are stretched to the first range_end,
    and colors from the last range_start onwards are compressed to the last range_end.

    Parameters:
        cmap_name: str, name of the original colormap.
        range_starts: list of floats, points in the original colormap to map to range_ends in the new colormap.
        range_ends: list of floats, target positions in the new colormap where range_starts will be mapped.
        num_colors: int, number of colors in the resulting colormap.

    Returns:
        new_cmap: LinearSegmentedColormap, the new custom colormap.
    """
    if len(range_starts) != len(range_ends):
        raise ValueError("range_starts and range_ends must have the same length")

    # Get the original colormap
    original_cmap = plt.get_cmap(cmap_name)
    
    # Create the new positions for the colormap based on the transformation
    new_positions = np.linspace(0, 1, num_colors)

    def transform(x):
        for start, end in zip(range_starts, range_ends):
            if x < end:
                # Stretch the colors below the current range_start to fill 0 to the current range_end
                return (x / end) * start
        # Compress the colors from the last range_start onwards to fill the last range_end to 1
        return range_starts[-1] + ((x - range_ends[-1]) / (1 - range_ends[-1])) * (1 - range_starts[-1])

    transformed_positions = np.vectorize(transform)(new_positions)
    new_colors = original_cmap(transformed_positions)
    
    new_cmap = LinearSegmentedColormap.from_list(f'custom_{cmap_name}', new_colors)
    
    return new_cmap

# -----------------------------------------------------------------------------
def shrink_cmap(cmap_name, center_start=0.4, center_end=0.6, num_colors=256):
    """
    Create a custom colormap by shrinking the central values in favor of the extremes.
    The colors between center_start and center_end in the original colormap are compressed,
    and the colors outside this range are expanded.

    Parameters:
        cmap_name: str, name of the original colormap.
        center_start: float, start of the central range to be compressed.
        center_end: float, end of the central range to be compressed.
        num_colors: int, number of colors in the resulting colormap.

    Returns:
        new_cmap: LinearSegmentedColormap, the new custom colormap.
    """
    if center_start >= center_end:
        raise ValueError("center_start must be less than center_end")

    # Get the original colormap
    original_cmap = plt.get_cmap(cmap_name)
    
    # Create the new positions for the colormap based on the transformation
    new_positions = np.linspace(0, 1, num_colors)

    def transform(x):
        if x < center_start:
            # Expand the colors below center_start
            return x * (center_start / center_end)
        elif x > center_end:
            # Expand the colors above center_end
            return center_start + ((x - center_end) / (1 - center_end)) * (1 - center_start)
        else:
            # Compress the colors between center_start and center_end
            return center_start + ((x - center_start) / (center_end - center_start)) * (center_end - center_start)

    transformed_positions = np.vectorize(transform)(new_positions)
    new_colors = original_cmap(transformed_positions)
    
    new_cmap = LinearSegmentedColormap.from_list(f'shrink_{cmap_name}', new_colors)
    
    return new_cmap

# -----------------------------------------------------------------------------
def create_log_file( main_function, 
                     main_args=(),
                     file_name='log_file', 
                     add2name='_log',
                     convert_to_pdf=True ):
    """"
    Executes a given function and captures its stdout output, 
    saving it to an HTML log file.
    This function redirects the standard output to capture all print statements and 
    matplotlib figures generated during the execution of the provided main_function. 
    The captured output is then formatted into an HTML file, which includes the 
    execution date and time, and saved with the specified file name.

    Args:
        - main_function (function): The main function to execute and capture output from.

        - file_name (str, optional): The base name of the log file to create. 
            Defaults to 'log_file'. If the provided file name 
            does not have an '.html' extension, it will be replaced with '.html'.
        
        - add2name (str, optional): The string to append to the file name.
        
        - convert_to_pdf (bool, optional): 
            Whether to convert the HTML log file to a PDF file.

    Returns:
        - str: The name of the created HTML log file.
    """

    base_name, ext = os.path.splitext( file_name )
    base_name = base_name + add2name

    # Replace the extension if it is different from .html
    if ext != '.html':
        out_file_name = base_name + '.html'
    else:
        out_file_name = base_name + ext

    class DualOutput:
        def __init__(self):
            self.terminal = sys.stdout
            self.buffer = io.StringIO()

        def write(self, message):
            self.terminal.write(message)
            self.buffer.write(message)

        def flush(self):
            self.terminal.flush()

        def add_figure(self, filename):
            # Control image width via inline CSS
            self.buffer.write(f'<img src="{filename}" alt="{filename}" style="max-width:640px; width:100%;">\n')

    dual_output = DualOutput()
    original_stdout = sys.stdout  # Keep track of the original stdout
    sys.stdout = dual_output

    original_savefig = plt.savefig
    def savefig_wrapper(filename, *args, **kwargs):
        dual_output.add_figure(filename)
        original_savefig(filename, *args, **kwargs)

    plt.savefig = savefig_wrapper

    if type( main_args ) not in ( tuple, list ):
        main_args = ( main_args, )
    if len( main_args )== 0:
        main_args = ()

    try:
        if main_args:
            main_function(*main_args)  # Execute the main function with arguments
        else:
            main_function()  # Execute the main function without arguments
    finally:
        output = dual_output.buffer.getvalue()
        sys.stdout = original_stdout  # Reset stdout
        plt.savefig = original_savefig
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content = f"""
        <html>
        <head><title>{file_name} Output</title></head>
        <body>
        <h1>{file_name} Output</h1>
        <p>Generated on: {current_datetime}</p>
        <pre>{output}</pre>
        </body>
        </html>
        """
        with open(out_file_name, "w") as file:
            file.write(html_content)

    if convert_to_pdf:
        pdf_file_name = base_name + '.pdf'

        try:
            pdfkit.from_file(out_file_name, pdf_file_name)
            out_file_name = pdf_file_name
        
        except Exception as e:
            print(f"Error converting HTML to PDF: {e}")

    return out_file_name
