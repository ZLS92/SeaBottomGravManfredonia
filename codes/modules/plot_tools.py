# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:32:47 2020

@author: lzampa
"""

# -----------------------------------------------------------------------------
# Import libraries

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.colors as LinearSegmentedColormap
from . import utils as utl
from . import raster_tools as rt
from . import shp_tools as shp

# -----------------------------------------------------------------------------
# Set the aliases for some libraries from the utils module

pltr = rt.pltr
plta = utl.plta
np = utl.np
os = utl.os

# -----------------------------------------------------------------------------
# Alias for the current directory (main dir.)
mdir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
def remove_plot( ax, label=[] ):

    if type( label ) not in ( list, tuple ) :
        label = [ label ]
    for ch in ax.get_children():
        if ch.get_label() in label : 
            ch.remove()
            
# -----------------------------------------------------------------------------
def slice_model_3d( NpArr3d, stride=1, nan2val=0, xyz_lim=None, raster_map=None,
                    vmin=None, vmax=None, prjcode=4326, units='units', labelpad=7,
                    X=None, Y=None, Z=None, array_type='ndarray', mcf=1, hspace=0.5, 
                    wspace=0.5 ):
                    
    
    if array_type == 'ndarray':
        
        lenx = NpArr3d.shape[0]
        leny = NpArr3d.shape[1]
        lenz = NpArr3d.shape[2]
    
        if xyz_lim==None: 
            xyz_lim = [0, lenx, 0, leny, -lenz, 0]
            
        xu = np.linspace(xyz_lim[0], xyz_lim[1], lenx)
        yu = np.linspace(xyz_lim[2], xyz_lim[3], leny)
        zu = np.linspace(xyz_lim[5], xyz_lim[4], lenz)
    
        X, Y, Z = np.meshgrid(xu, yu, zu, indexing='ij')
        
    if array_type == 'xyz':
        if type( NpArr3d ) in ( tuple, list ) :
            merge_colums = np.zeros( [np.size( NpArr3d[0] ), len( NpArr3d )])
            for n, i in enumerate( NpArr3d ) :
                if type( i ) in ( list, tuple ) :
                    i = np.array( i )
                merge_colums[:,n] = i
            NpArr3d = np.copy( merge_colums )
            
        xu = np.unique( NpArr3d[:,0] )
        yu = np.unique( NpArr3d[:,1] )
        zu = np.unique( NpArr3d[:,2] )
    
        X, Y, Z = np.meshgrid( xu, yu, zu, indexing='ij' )
        NpArr3d= NpArr3d[np.lexsort(( NpArr3d[:,2], NpArr3d[:,1], NpArr3d[:,0]))]
        NpArr3d = NpArr3d[:,3].reshape( Z.shape )
        lenx = NpArr3d.shape[0]
        leny = NpArr3d.shape[1]
        lenz = NpArr3d.shape[2]  
        
        xyz_lim = [ X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max() ]
    
    mxi = int(lenx/2)
    myi = int(leny/2)
    mzi = int(lenz/2)
    
    Min, Max, Mean, Std = utl.stat( NpArr3d, show=False )
    if vmin==None: vmin=Mean-2*Std
    if vmax==None: vmax=Mean+2*Std
    where_are_NaNs = np.isnan(NpArr3d)
    NpArr3d[where_are_NaNs] = nan2val    
    
    fig = plt.figure()
    gs = fig.add_gridspec(18, 3, hspace=hspace, wspace=wspace )
    
        
    mnl = gs[0:-3, 0:]   
    
    ax = fig.add_subplot(mnl, projection='3d')
    ax.set_xlim3d(np.min(xu), np.max(xu))
    ax.set_ylim3d(np.min(yu), np.max(yu))
    ax.set_zlim3d(np.min(zu), np.max(zu))
    
    norm = plt.Normalize(vmin, vmax)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    cbar = plt.colorbar(m, pad=0.1)
    cbar.set_label(f'{units}', labelpad=labelpad)
    
    ax.set_ylabel('Yaxes', labelpad=labelpad)
    ax.set_xlabel('Xaxes', labelpad=labelpad)
    ax.set_zlabel('Zaxes', labelpad=labelpad)
    
    axsx = fig.add_subplot(gs[-3, 0:])
    axsy = fig.add_subplot(gs[-2, 0:])
    axsz = fig.add_subplot(gs[-1, 0:])
    
    sx = Slider(axsx, 'Xslice', valmin=0, valmax=lenx, valstep=1, valinit=mxi, valfmt='%d')
    sy = Slider(axsy, 'Yslice', valmin=0, valmax=leny, valstep=1, valinit=myi, valfmt='%d')
    sz = Slider(axsz, 'Zslice', valmin=0, valmax=lenz, valstep=1, valinit=mzi, valfmt='%d')
    axsx.xaxis.set_visible(True)
    
    if raster_map is not None: 
        
        for i,_ in enumerate( xyz_lim ) :
            xyz_lim[i] *=mcf
        r_m_crop = rt.raster_warp( raster_map, lim=xyz_lim[0:4], 
                                   lim_prjcode=prjcode, out_prjcode=prjcode, close=False )
        gs_map = 4
        mapax = fig.add_subplot(gs[0:gs_map, 0])
        if r_m_crop.RasterCount>1:
            rgb_array, metadata = rt.RGBraster2array(r_m_crop)
            plot_RgbArray(rgb_array, extent=metadata['extent'], ax=mapax)   
        if r_m_crop.RasterCount==1:
            rt.pltr(r_m_crop, axes=True)   
    
        
    def map_slice(Xs, Ys, nm, mapax):
        xl = [np.min(Xs)*mcf, np.max(Xs)*mcf]
        yl = [np.min(Ys)*mcf, np.max(Ys)*mcf]
        remove_plot(mapax, nm)
        mapax.plot(xl,yl, label=nm, c='r')
    
    def remove_plot(ax, label):
        for ch in ax.get_children():
            if ch.get_label() == label: ch.remove()
    
    def updatex(valx):
        valx = int(valx)
        remove_plot(ax, 'sx')        
        psx = ax.plot_surface(X[valx,:,:], Y[valx,:,:], Z[valx,:,:], rstride=stride, cstride=stride,
                        alpha=1, facecolors=m.to_rgba(NpArr3d[valx,:,:]), cmap='jet', label='sx', 
                        antialiased=True, zorder=0.1)
        psx.set_edgecolor('k')  
        psx.set_linewidth(0.3) 
        if raster_map is not None: 
            map_slice(X[valx,:,:], Y[valx,:,:], 'sx', mapax)
            
    def updatey(valy):
        valy = int(valy)
        remove_plot(ax, 'sy') 
        psy = ax.plot_surface(X[:,valy,:], Y[:,valy,:], Z[:,valy,:], rstride=stride, cstride=stride,
                        facecolors=m.to_rgba(NpArr3d[:,valy,:]), cmap='jet', label='sy', 
                        antialiased=True, zorder=0.5)
        psy.set_edgecolor('k')  
        psy.set_linewidth(0.3) 
        psy.set_alpha(1)
        if raster_map is not None: 
            map_slice(X[:,valy,:], Y[:,valy,:], 'sy', mapax)
        
    def updatez(valz):
        valz = int(valz)
        remove_plot(ax, 'sz')    
        psz = ax.plot_surface(X[:,:,valz], Y[:,:,valz], Z[:,:,valz], rstride=stride, cstride=stride,
                        alpha=1, facecolors=m.to_rgba(NpArr3d[:,:,valz]), cmap='jet', label='sz',
                        antialiased=False)     
        psz.set_edgecolor('k') 
        psz.set_linewidth(0.3) 
         
    
    sx.on_changed(updatex)
    sy.on_changed(updatey)
    sz.on_changed(updatez)
    

    return fig, ax, sx, sy, sz

# -----------------------------------------------------------------------------
def plot_layers_2d( lines, lx, top=None, bottom=None, hatch=None, colors='random', 
                    fill=True, alpha=1, legend_labels=None, legend_position=(0.01, 0.01), 
                    legend_label_size=None, ylim=None, xlim=None, x_label=None, 
                    y_label=None, ax_label_size=None, font='serif', figsize=(10, 5), 
                    text=None, text_size=14, text_boxstyle='round', text_facecolor='wheat', 
                    text_alpha=0.5, text_position=(0.95, 0.05), legend_ncol=1, 
                    legend_loc='best', subplot=(1, 1, 1) ):
    
    import random
    from matplotlib import rcParams
    
    rcParams['font.family'] = font
    if top is not None:
        line_top = np.repeat(top, len(lx))
        lines.insert(0, line_top)
    if bottom is not None:
        line_bottom = np.repeat(bottom, len(lx))
        lines.append(line_bottom)
    xy_ply = []
    for i, line in enumerate(lines):
        if i == len(lines) - 1: break   
        line_stack1 = np.column_stack((lx, line))
        line_stack2 = np.column_stack((lx, lines[(i + 1)]))
        xy_ply.append(np.vstack((line_stack1, np.flipud(line_stack2))))

    if hatch == 'random':
        hatch = [''.join([random.choice(['---', '+++', 'ccc', '///', '...', '+++', 'ddd']) for j in range(2)]) for i in range(40)]
        hatch = np.unique(hatch)
    if hatch is None: 
        hatch = [None for j in range(len(lines)-1)]
        
    if colors == 'random':
        colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(40)]
        colors = np.unique(colors)
    if legend_labels is None:
        legend_labels = ['layer_' + str(i) for i in range(len(xy_ply))]
    ptc = []
    ax = plt.subplot(subplot[0], subplot[1], subplot[2])
    for i, (ply, ht, cl) in enumerate(zip(xy_ply, hatch, colors)):
        plt.plot(lx, (lines[i]), color=cl)
        poly = plt.Polygon((ply.tolist()), color=cl, fill=fill, alpha=alpha, hatch=ht,
          linewidth=None)
        ax.add_patch(poly)
        ptc.append(poly)

    if xlim is None:
        plt.xlim(np.min(lx), np.max(lx))
    if ylim is None:
        plt.ylim(np.min(lines[(-1)]), np.max(lines[0]))
    if x_label is not None:
        plt.xlabel(x_label, fontsize=ax_label_size)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=ax_label_size)
    if text is not None:
        props = dict(boxstyle=text_boxstyle, facecolor=text_facecolor, alpha=text_alpha)
        ax.annotate(text, xy=text_position, xycoords='axes fraction', fontsize=text_size, bbox=props)
    plt.tight_layout()
    
    if legend_labels is not False:
        ax.legend(ptc, legend_labels, fontsize=legend_label_size, ncol=legend_ncol, loc=legend_loc,
          bbox_to_anchor=legend_position)
        
    return xy_ply

# -----------------------------------------------------------------------------
def imsplt( img, m, n, p, aspect='auto' ):
    
    """
    Plot image files inyo subplots
    img = file path/name
    m = numbers of rows
    n = numbers of columns 
    p = position of the image 
    """
    
    plt.subplot(m,n,p)
    img = mpimg.imread(img)
    impl=plt.imshow(img, aspect=aspect )
    impl.axes.get_xaxis().set_visible(False)
    impl.axes.get_yaxis().set_visible(False)
    ax=impl.axes;
    ax.axis('off')

# -----------------------------------------------------------------------------
def figsplt( im_list, m=1, n=1, x_size=20, y_size=20, dpi=None, path_nm=None, 
             text_size=12, xn=0.1, yn=0.1, pad=1, h_pad=None, w_pad=None, tight=True,
             titles=None, alphabet=True, letters=[], bbox_inches=None, aspect='auto' ):
    
    sp = 0
    fig = plt.figure(figsize=(x_size*0.393701,y_size*0.393701))
    alph = ['a','b','c','d','e','f','g','h','i','l','m','n','o','p','q','r']
    if m!=1 or n!=1:
        for i, im in enumerate(im_list):
            sp += 1      
            imsplt(im,m,n,sp)  
            if (alphabet is True) and (letters==[]):
                plt.annotate(alph[i]+'.', xy=(xn,yn), xycoords='axes  fraction', size=text_size) 
            if (alphabet is True) and (letters!=[]):  
                plt.annotate(letters[i]+'.', xy=(xn,yn), xycoords='axes  fraction', size=text_size) 
            if titles is not None:
                plt.title(titles[i])
 
    else: imsplt(im_list[0],m,n,1, aspect=aspect )
    
    if tight is True:
        fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)                 

    if path_nm!=None:
        plt.savefig(path_nm, dpi=dpi, bbox_inches=bbox_inches )  
        
# -----------------------------------------------------------------------------
def colorbar( mappable, label="" ):
    
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    return fig.colorbar(mappable, cax=cax, label=label)

# -----------------------------------------------------------------------------
def geo_line( x, y, z, plt_type='1d', plot=True, prjcode_in=4326, prjcode_out=4326, c='b', 
              plot_points=True, marker='+', marker_color='k', deg2m=False, order=None,
              s=1.5, cmap='rainbow', new_fig=True, colorbar=True, vmin=None, vmax=None ) :
    
    dist = utl.geo_line_dist( x, y, prjcode_in=prjcode_in, prjcode_out=prjcode_out )

    dxyz = np.column_stack( ( dist[:,1], z ) )
    
    if deg2m != False :
        dxyz[:,0] = utl.deg2m( dxyz[:,0] )

    if type( deg2m ) in ( float, int ) :    
        dxyz[:,0] = dxyz[:,0] * deg2m
        
    if plot == True :
        
        if new_fig == True :
            plt.figure()
            
        if plt_type == '1d' :
            
            plt.plot( dxyz[:,0], dxyz[:,-1], c=c )
            
            if plot_points == True :
                plt.scatter( dxyz[:,0], dxyz[:,-1], s=s, c=marker_color, alpha=1, 
                             marker=marker )
                    
        if plt_type == '1.5d' :
            
            plt.scatter( dxyz[:,1], dxyz[:,2], s=s, c=dxyz[:,-1], cmap=cmap, vmin=vmin, 
                         vmax=vmax ) 
            if colorbar == True :
                plt.colorbar( ax=plt.gcf().axes )
            
        if plt_type == '2d' :    
            ax = plt.gca(projection='3d')
            ax.plot(dxyz[:,1], dxyz[:,2], dxyz[:,-1])
        
    return dxyz

# -----------------------------------------------------------------------------
class LineSlice:
    '''Allow user to drag a line on a pcolor/pcolormesh plot, and plot the Z values from that line on a separate axis.

    Example
    -------
    fig, (ax1, ax2) = plt.subplots( nrows=2 )    # one figure, two axes
    img = ax1.pcolormesh( x, y, Z )     # pcolormesh on the 1st axis
    lntr = LineSlice( img, ax2 )        # Connect the handler, plot LineSlice onto 2nd axis

    Arguments
    ---------
    img: the pcolormesh plot to extract data from and that the User's clicks will be recorded for.
    ax2: the axis on which to plot the data values from the dragged line.


    '''
    def __init__(self, img, ax):
        '''
        img: the pcolormesh instance to get data from/that user should click on
        ax: the axis to plot the line slice on
        '''
        self.img = img
        self.ax = ax
        self.data = img.get_array()

        # register the event handlers:
        self.cidclick = img.figure.canvas.mpl_connect('button_press_event', self)
        self.cidrelease = img.figure.canvas.mpl_connect('button_release_event', self)

        self.markers, self.arrow = None, None   # the lineslice indicators on the pcolormesh plot
        self.line = None    # the lineslice values plotted in a line
    #end __init__

    def __call__(self, event):
        '''Matplotlib will run this function whenever the user triggers an event on our figure'''
        if event.inaxes != self.img.axes: return     # exit if clicks weren't within the `img` axes
        if self.img.figure.canvas.manager.toolbar._active is not None: return   # exit if pyplot toolbar (zooming etc.) is active

        if event.name == 'button_press_event':
            self.p1 = (event.xdata, event.ydata)    # save 1st point
        elif event.name == 'button_release_event':
            self.p2 = (event.xdata, event.ydata)    # save 2nd point
            self.drawLineSlice()    # draw the Line Slice position & data
    #end __call__

    def drawLineSlice( self ):
        ''' Draw the region along which the Line Slice will be extracted, onto the original self.img pcolormesh plot.  Also update the self.axis plot to show the line slice data.'''
        '''Uses code from these hints:
        http://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
        http://stackoverflow.com/questions/34840366/matplotlib-pcolor-get-array-returns-flattened-array-how-to-get-2d-data-ba
        '''

        x0,y0 = self.p1[0], self.p1[1]  # get user's selected coordinates
        x1,y1 = self.p2[0], self.p2[1]
        length = int( np.hypot(x1-x0, y1-y0) )
        x, y = np.linspace(x0, x1, length),   np.linspace(y0, y1, length)

        # Extract the values along the line with nearest-neighbor pixel value:
        # get temp. data from the pcolor plot
        zi = self.data[x.astype(np.int), y.astype(np.int)]
        # Extract the values along the line, using cubic interpolation:
        #import scipy.ndimage
        #zi = scipy.ndimage.map_coordinates(self.data, np.vstack((x,y)))

        # if plots exist, delete them:
        if self.markers != None:
            if isinstance(self.markers, list):
                self.markers[0].remove()
            else:
                self.markers.remove()
        if self.arrow != None:
            self.arrow.remove()

        # plot the endpoints
        self.markers = self.img.axes.plot([x0, x1], [y0, y1], 'wo')
        # plot an arrow:
        self.arrow = self.img.axes.annotate("",
                    xy=(x0, y0),    # start point
                    xycoords='data',
                    xytext=(x1, y1),    # end point
                    textcoords='data',
                    arrowprops=dict(
                        arrowstyle="<-",
                        connectionstyle="arc3",
                        color='white',
                        alpha=0.7,
                        linewidth=3
                        ),

                    )

        # plot the data along the line on provided `ax`:
        if self.line != None:
            self.line[0].remove()   # delete the plot
        self.line = self.ax.plot(zi)
    #end drawLineSlice()
        
# -----------------------------------------------------------------------------
def plot_RgbArray(rgb_array, extent, colorlimit=(1,255), title='', ax=None, axis=False,
                  aspect='auto'):

    """
    Authors: Bridget Hass
    Ref: https://www.neonscience.org/resources/learning-hub/tutorials/plot-neon-rgb-py
    Last Updated: Oct 7, 2020

    --------
    plot_band_array reads in and plots a single band or an rgb band combination of a reflectance array
    --------
    Parameters
    --------
        rgb_array: flightline array of reflectance values, created from h5refl2array function
        refl_extent: extent of reflectance data to be plotted (xMin, xMax, yMin, yMax) - use metadata['extent'] from h5refl2array function
        colorlimit: range of values to plot (min,max). Best to look at the histogram of reflectance values before plotting to determine colorlimit.
        ax: optional, default = current axis
        title: string, optional; plot title
    --------
    Returns
        plots array of single band or RGB if given a 3-band

    """

    if ax is None:
        print('ok')
        plt.imshow(rgb_array, extent=extent, clim=colorlimit, aspect=aspect )
        plt.title(title)
        axi = plt.gca()
        axi.ticklabel_format(useOffset=False, style='plain') #do not use scientific notation #
        plt.setp(axi.get_xticklabels(),rotation=90) #rotate x tick labels 90 degrees
        if axis == False :
           plt.xticks([], [])
           plt.yticks([], [])
    if ax is not None:
        ax.imshow(rgb_array, extent=extent, clim=colorlimit, aspect=aspect)
        ax.set_title(title)
        ax.ticklabel_format(useOffset=False, style='plain') #do not use scientific notation #
        plt.setp(ax.get_xticklabels(),rotation=90) #rotate x tick labels 90 degrees
        if axis == False:
           ax.axes.xaxis.set_ticks([])
           ax.axes.yaxis.set_ticks([])
        
# -----------------------------------------------------------------------------    
def plot_lines( xyzl, x_c=0, y_c=1, z_c=2, line_c=3, deg2m=False, plot_points=True,
                marker='+', marker_color='k', s=1.5, x_units='',  y_units='', 
                legend=[], c='b', cross_points=True, order_c='same', lines=[] ) :

    if order_c != 'same' :
        xyzl = utl.sort_lines( xyzl, x_c=x_c, y_c=y_c, line_c=line_c, order_c=order_c )[0]
    
    if type( z_c ) in ( int, float ) :
        z_c = [ z_c ]    
    
    if lines == [] :
        lines = np.unique( xyzl[:,line_c] )    
    else :
        lines = np.array( lines )
    N = np.size( lines )
    
    fig = plt.figure()
    
    grid = plt.GridSpec( 12, 4, wspace=0.5, hspace=0.5 )
    
    # ------------------------------------------------------------------------- 
    # Position map 
    
    ax1 = plt.subplot( grid[:4, :2] )
    
    for i, l in enumerate(lines) :
        
        idx = xyzl[ : , line_c ] == l
        line = xyzl[ idx ]
        
        _ = ax1.plot( line[:,x_c], line[:,y_c], c='k' )
        ax1.text( line[0,x_c], line[0,y_c], int( l ) )
            
        if i == 0 :    
           _ = ax1.plot( line[:,x_c], line[:,y_c], c='r', label='ml' )
           
        ax1.axis('off')  

#    ax1.set_aspect('equal') 
           
    # -------------------------------------------------------------------------       
    # Line profile  
    
    ax2 = plt.subplot( grid[ 4:, : ] )
    
    line = xyzl[ xyzl[ : , 3 ] == xyzl[ 0 , 3 ] ]
    
    if type(c) == str :
        c = [ c for i in z_c ]
        
    for i, z in enumerate( z_c ) :    
        dxyz = geo_line( line[:,x_c], line[:,y_c], line[:,z], deg2m=deg2m, plot=False ) 
    
        ax2.plot( dxyz[:,0], dxyz[:,-1], c=c[i], label='pl' )
        
        if plot_points == True :
            ax2.scatter( dxyz[:,0], dxyz[:,-1], s=1, c=marker_color, alpha=1, 
                         marker=marker, label='sp' ) 
    
    if legend != [] :
        ax2.legend( legend )
        
    ax2.set_xlabel( ' Distance '+ x_units )
    ax2.set_ylabel( ' Amplitude '+ y_units )    

    # -------------------------------------------------------------------------       
    # Cross points

    if lines.size <=1 : 
        cross_points = False

    if cross_points == True :
        
        cross_points_list = []
        
        for i, z in enumerate( z_c ) :
            
            cross_points_list.append( utl.cross_over_points( xyzl, x_c=x_c, 
                                                            y_c=y_c, z_c=z, 
                                                            line_c=line_c, 
                                                            method='linear' ) )
            
        for i, cop in enumerate( cross_points_list ) :
    
            mean_err = np.nanmean( cop[:,6] ) * 2
            
            cop1 = cop[ cop[ :, 2 ] == xyzl[ 0 , 3 ] ]  
            cop2 = cop[ cop[ :, 3 ] == xyzl[ 0 , 3 ] ] 
            line = xyzl[ xyzl[ : , 3 ] == xyzl[ 0 , 3 ] ]
            x0, y0 = line[ 0, x_c ], line[ 0, y_c ]
            
            if np.size( cop1 ) != 0 :
                dcop1 = np.sqrt( (cop1[ :, 0 ]-x0)**2 + (cop1[ :, 1 ]-y0)**2 )
                if deg2m != False : dcop1 = utl.deg2m( dcop1 )
                if type( deg2m ) in ( float, int ) : dcop1 *= deg2m
        #        ax2.scatter( dcop1, cop1[:,5], s=2, c='r', alpha=1, marker='o', label='cp1' )
                err = np.repeat( mean_err, np.size( dcop1 ) )
                ax2.errorbar( dcop1, cop1[:,4], yerr=err, fmt='none', ecolor=c[i] ) 
                for x,y in zip(dcop1,cop1): ax2.text( x, y[5], int(y[3]) ) 
                
            if np.size( cop2 ) != 0 :    
                dcop2 = np.sqrt( (cop2[ :, 0 ]-x0)**2 + (cop2[ :, 1 ]-y0)**2 )       
                if deg2m != False : dcop2 = utl.deg2m( dcop2 )
                if type( deg2m ) in ( float, int ) : dcop2 *= deg2m        
        #        ax2.scatter( dcop2, cop2[:,4], s=2, c='r', alpha=1, marker='o', label='cp2' ) 
                err = np.repeat( mean_err, np.size( dcop2 ) )
                ax2.errorbar( dcop2, cop2[:,4], yerr=err, fmt='none', ecolor=c[i] ) 
                for x,y in zip(dcop2,cop2): ax2.text( x, y[4], int(y[2]) )     
            
    # ------------------------------------------------------------------------- 
    # Slider 
    axcolor = 'lightgoldenrodyellow'
    axs = plt.subplot( grid[1:2, 2:], facecolor=axcolor )  
    
    sl = Slider(axs, 'Line', 0, N-1, valinit=0, valstep=1, valfmt='%d' ) 
    
    # ------------------------------------------------------------------------- 
    # Update
    
    def update_line( val ):
        val = int( val )
        # Remove --------------------------------------------------------------
        remove_plot(ax1, 'ml')
        ax2.cla()
        # Select line ---------------------------------------------------------    
        line_idx = lines[ val ]
        s_line = xyzl[ xyzl[:,line_c] == line_idx ]
        sl.valtext.set_text('{}'.format( int( line_idx ) ) )
        # Upgrade ax1 ---------------------------------------------------------
        ax1.plot( s_line[:,x_c], s_line[:,y_c], c='r', label='ml' )
        # Upgrade ax2 ---------------------------------------------------------
        
        list_min = []
        list_max = []
        
        for i, z in enumerate( z_c ) : 
            
            dxyz = geo_line( s_line[:,x_c], s_line[:,y_c], s_line[:,z], deg2m=deg2m, plot=False ) 
            ax2.plot( dxyz[:,0], dxyz[:,-1], c=c[i], label='pl'  )
            if plot_points == True :
                ax2.scatter( dxyz[:,0], dxyz[:,-1], s=1, c=marker_color, alpha=1, 
                             marker=marker, label='sp' ) 
            list_min.append( dxyz[:,-1].min() )
            list_max.append( dxyz[:,-1].max() ) 
        
        if legend != [] :
            ax2.legend( legend )
            
        # Upgrade cross points (ax2) ------------------------------------------  
        if cross_points == True :
            
            for i, cop in enumerate( cross_points_list ) :      
                
                cop1 = cop[ cop[ :, 2 ] == line_idx ]  
                cop2 = cop[ cop[ :, 3 ] == line_idx ] 
                x0, y0 = s_line[ 0, x_c ], s_line[ 0, y_c ]
                if np.size( cop1 ) != 0 :
                    dcop1 = np.sqrt( (cop1[ :, 0 ]-x0)**2 + (cop1[ :, 1 ]-y0)**2 ) 
                    if deg2m != False : dcop1 = utl.deg2m( dcop1 )
                    if type( deg2m ) in ( float, int ) : dcop1 *= deg2m               
        #            ax2.scatter( dcop1, cop1[:,5], s=s*5, c='r', alpha=1, marker='o', label='cp1' ) 
                    err = np.repeat( mean_err, np.size( dcop1 ) )
                    ax2.errorbar( dcop1, cop1[:,4], yerr=err, fmt='none', ecolor=c[i] ) 
                    for x,y in zip(dcop1,cop1): ax2.text( x, y[5], int(y[3]) ) 
                    list_min.append( cop1[:,5].min() )
                    list_max.append( cop1[:,5].max() )  
                if np.size( cop2 ) != 0 :    
                    dcop2 = np.sqrt( (cop2[ :, 0 ]-x0)**2 + (cop2[ :, 1 ]-y0)**2 )  
                    if deg2m != False : dcop2 = utl.deg2m( dcop2 )
                    if type( deg2m ) in ( float, int ) : dcop2 *= deg2m               
        #            ax2.scatter( dcop2, cop2[:,4], s=s*5, c='r', alpha=1,marker='o', label='cp2' )
                    err = np.repeat( mean_err, np.size( dcop2 ) )
                    ax2.errorbar( dcop2, cop2[:,4], yerr=err, fmt='none', ecolor=c[i] )
                    for x,y in zip(dcop2,cop2): ax2.text( x, y[4], int(y[2]) )  
                    list_min.append( cop2[:,4].min() )
                    list_max.append( cop2[:,4].max() )     
                    
        # Limits (ax2) --------------------------------------------------------   
        xmin, xmax = dxyz[:,0].min(), dxyz[:,0].max()
        ymin = np.nanmin( list_min ) 
        ymax = np.nanmax( list_max ) 
        add_x = ( ( xmax - xmin ) / 100 ) * 5
        add_y = ( ( ymax - ymin ) / 100 ) * 10
        xmin, xmax = xmin - add_x, xmax + add_x
        ymin, ymax = ymin - add_y, ymax + add_y
        ax2.set_xlim( xmin, xmax )    
        ax2.set_ylim( ymin, ymax )  
        ax2.set_xlabel( ' Distance '+ x_units )
        ax2.set_ylabel( ' Amplitude '+ y_units ) 

    sl.on_changed( update_line )
    
#    fig.tight_layout()    
    
    return fig, ax1, ax2, sl


# -----------------------------------------------------------------------------
def slice_model_2d( NpArr3d, nan2val=0, xyz_lim=None, raster_map=None,
                    vmin=None, vmax=None, prjcode=4326, units='units', labelpad=0,
                    X=None, Y=None, Z=None, array_type='ndarray', mcf=1, cmap='jet',
                    aspect='auto', cunits='dist', suptitle='', hspace=0.25, wspace=0.25,
                    map_points=None, mpoints_size=1, mpoints_marker='.', mpoints_color='b',
                    map_lim=None, extend_lim=5, map_axis=False, points=None,
                    points_size=1, points_color='b', points_marker='.', 
                    points_label='points', mpoints_label='mpoints', linewidth=0.1,
                    nodes=False, shading='auto', MxRes=None, MyRes=None ):
                    
    if array_type == 'ndarray':
        
        lenx = NpArr3d.shape[0]
        leny = NpArr3d.shape[1]
        lenz = NpArr3d.shape[2]
    
        if xyz_lim is None: 
            xyz_lim = [0, lenx, 0, leny, -lenz, 0]
            
        xu = np.linspace(xyz_lim[0], xyz_lim[1], lenx)
        yu = np.linspace(xyz_lim[2], xyz_lim[3], leny)
        zu = np.linspace(xyz_lim[5], xyz_lim[4], lenz)
    
        X, Y, Z = np.meshgrid(xu, yu, zu, indexing='ij')
        
    if array_type == 'xyz':
        if type( NpArr3d ) in ( tuple, list ) :
            merge_colums = np.zeros( [np.size( NpArr3d[0] ), len( NpArr3d )])
            for n, i in enumerate( NpArr3d ) :
                if type( i ) in ( list, tuple ) :
                    i = np.array( i )
                merge_colums[:,n] = i
            NpArr3d = np.copy( merge_colums )
            
        if xyz_lim is not None : 
            
            idx = (NpArr3d[:,0] >= xyz_lim[0]) & (NpArr3d[:,0] <= xyz_lim[1]) &\
                  (NpArr3d[:,1] >= xyz_lim[2]) & (NpArr3d[:,1] <= xyz_lim[3]) &\
                  (NpArr3d[:,2] >= xyz_lim[4]) & (NpArr3d[:,2] <= xyz_lim[5])
                  
            NpArr3d = NpArr3d[idx,:]      
            
        xu = np.unique( NpArr3d[:,0] )
        yu = np.unique( NpArr3d[:,1] )
        zu = np.unique( NpArr3d[:,2] )
        
        X, Y, Z = np.meshgrid( xu, yu, zu, indexing='ij' )
        
        NpArr3d= NpArr3d[np.lexsort(( NpArr3d[:,2], NpArr3d[:,1], NpArr3d[:,0]))]
        NpArr3d = NpArr3d[:,3].reshape( Z.shape )
        lenx = NpArr3d.shape[0]
        leny = NpArr3d.shape[1]
        lenz = NpArr3d.shape[2]  
        
    if xyz_lim is None : 
        xyz_lim = [ X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max() ]
            
    valx = int(lenx/2)
    valy = int(leny/2)
    valz = int(lenz/2)    
    
    Min, Max, Mean, Std = utl.stat( NpArr3d, show=False )
    if vmin==None: vmin=Mean-2*Std
    if vmax==None: vmax=Mean+2*Std
    
    where_are_NaNs = np.isnan( NpArr3d )
    NpArr3d[ where_are_NaNs ] = nan2val   
    
    fig = plt.figure()
    gs = fig.add_gridspec(12, 3, hspace=hspace, wspace=wspace)  
    
    ax1 = fig.add_subplot( gs[0:-7, 1:2] )
    ax2 = fig.add_subplot( gs[0:-7, 2:] )
    ax3 = fig.add_subplot( gs[-5:, 2:] )
    
    axsx = fig.add_subplot( gs[-5:-4, 0:2] )
    axsy = fig.add_subplot( gs[-4:-3, 0:2] )
    axsz = fig.add_subplot( gs[-3:-2, 0:2] )
    axcm = fig.add_subplot( gs[-2:-1, 0:2] )    
    
    norm = plt.Normalize(vmin, vmax)
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap )
    m.set_array([])
    cbar = fig.colorbar(m, ax=axcm, cax=axcm, orientation='horizontal')
    cbar.set_label(f'{units}', rotation=0, labelpad=labelpad)
    cbar.ax.xaxis.set_label_position('bottom')
    
    if nodes is True : 
        xp = xu[:-1] + ( np.diff( xu )/2 )
        yp = yu[:-1] + ( np.diff( yu )/2 )
        zp = zu[:-1] + ( np.diff( zu )/2 )
        
        Yp, Xp, Zp = np.meshgrid( yp, xp, zp )
        
        NpArr3d = utl.sp.interpolate.griddata( (X.ravel(), Y.ravel(), Z.ravel()), 
            NpArr3d.ravel(),(Xp.ravel(), Yp.ravel(), Zp.ravel()), method='linear' )
        NpArr3d = NpArr3d.reshape( Xp.shape )
        nsl = -1
        
    else :
        Yp, Xp, Zp = Y, X, Z
        nsl = 0
    
    sx = Slider(axsx, 'Xslice', valmin=0, valmax=lenx-nsl, valstep=1, valinit=valx, valfmt='%d')
    sy = Slider(axsy, 'Yslice', valmin=0, valmax=leny-nsl, valstep=1, valinit=valy, valfmt='%d')
    sz = Slider(axsz, 'Zslice', valmin=0, valmax=lenz-nsl, valstep=1, valinit=valz, valfmt='%d')
    axsx.xaxis.set_visible(True)
    
    if suptitle is not None :
        plt.suptitle( suptitle )
    
    if raster_map is not None: 
        
        if map_lim is None :
            
            map_lim = [ i*mcf for i in xyz_lim ][0:4]
            
        map_lim = utl.extend_lim( map_lim, extend_lim )
        
        r_m_crop = rt.raster_warp( raster_map, lim=map_lim, 
                                   lim_prjcode=prjcode, out_prjcode=prjcode, close=False,
                                   xRes=MxRes, yRes=MyRes )
        
        mapax = fig.add_subplot(gs[0:-6, 0])
        if r_m_crop.RasterCount>1:
            rgb_array, metadata = rt.RGBraster2array(r_m_crop)
            plot_RgbArray( rgb_array, extent=metadata['extent'], ax=mapax, aspect=aspect,
                           axis=map_axis)   
        if r_m_crop.RasterCount==1:
            rt.pltr( r_m_crop, axis=True )   
        
        if map_points is not None :
            mapax.scatter( map_points[0], map_points[1], c=mpoints_color, marker=mpoints_marker,
                           s=mpoints_size, label=mpoints_label )
            
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/mcf))
        mapax.xaxis.set_major_formatter(ticks_x) 

        ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y/mcf))
        mapax.yaxis.set_major_formatter(ticks_y)     
        
        if points is not None :
            mapax.scatter( points[0]*mcf, points[1]*mcf, s=points_size, c=points_color, 
                         marker=points_marker, label=points_label )
            # mapax.legend()
        
    def map_slice(Xs, Ys, nm, mapax):
        xl = [np.min(Xs)*mcf, np.max(Xs)*mcf]
        yl = [np.min(Ys)*mcf, np.max(Ys)*mcf]
        remove_plot(mapax, nm)
        mapax.plot(xl,yl, label=nm, c='r')
    
    def remove_plot(ax, label):
        for ch in ax.get_children():
            if ch.get_label() == label: ch.remove()

    
    def updatex(valx):
        valx = int(valx)
        remove_plot(ax1, 'sx')
        remove_plot( ax2, 'linex' )
        remove_plot( ax3, 'linezx' )        
        ax1.pcolormesh( Y[valx,:,:], Z[valx,:,:], NpArr3d[ valx, :, : ], vmin=vmin, vmax=vmax, 
                        cmap=cmap, edgecolors='k', linewidth=linewidth, shading='auto', snap=True ) 
        ax1.set_aspect( aspect )
        # ax1.invert_xaxis()
        ax2.vlines( Xp[valx,:,:][0,0], ax1.get_ylim()[0], ax1.get_ylim()[1], color='r', label='linex' )
        ax3.vlines( Xp[valx,:,:][0,0], ax1.get_xlim()[0], ax1.get_xlim()[1], color='r', label='linezx' )  
        ax1.set_xlabel( 'Y '+cunits )
        ax1.set_ylabel( 'Z '+cunits )         
        if points is not None :
            idx = ( points[0] > X[valx-1,:,:][0,0] ) & ( points[0] < X[valx+1,:,:][0,0] ) 
            ax1.scatter( points[1][idx], points[2][idx], s=points_size, c=points_color, 
                          marker=points_marker, label=points_label )
        if raster_map is not None: 
            map_slice(X[valx,:,:], Y[valx,:,:], 'sx', mapax)
        ax1.set_title( 'X Slice')
            
    def updatey( valy ) :
        valy = int( valy )
        remove_plot(ax2, 'sy')
        remove_plot( ax1, 'liney' )
        remove_plot( ax3, 'linezy' )
        ax2.pcolormesh( X[:,valy,:], Z[:,valy,:], NpArr3d[ :, valy, : ], vmin=vmin, vmax=vmax, 
                        cmap=cmap, edgecolors='k', linewidth=linewidth, shading='auto', snap=True )    
        ax1.vlines( Yp[:,valy,:][0,0], ax2.get_ylim()[0], ax2.get_ylim()[1], color='r', label='liney' )
        ax3.hlines( Yp[:,valy,:][0,0], ax2.get_xlim()[0], ax2.get_xlim()[1], color='r', label='linezy' )
        ax2.set_aspect( aspect )
        ax2.set_xlabel( 'X '+cunits )
        ax2.set_ylabel( 'Z '+cunits ) 
        if points is not None :
            idx = ( points[1] > Y[:,valy-1,:][0,0] ) & ( points[1] < Y[:,valy+1,:][0,0] ) 
            ax2.scatter( points[0][idx], points[2][idx], s=points_size, c=points_color, 
                          marker=points_marker, label=points_label )        
        if raster_map is not None: 
            map_slice(X[:,valy,:], Y[:,valy,:], 'sy', mapax)
        ax2.set_title( 'Y slice')
        
    def updatez(valz):
        valz = int(valz)
        remove_plot(ax3, 'sz')    
        ax3.pcolormesh( X[:,:,valz], Y[:,:,valz], NpArr3d[ :, :, valz ], vmin=vmin, vmax=vmax, 
                              cmap=cmap, edgecolors='k', linewidth=linewidth, shading=shading, snap=True )  
        ax3.set_aspect( aspect )    
        ax3.set_title( 'Z slice ' +f'[ { Z[:,:,valz][0,0] } ]')
        ax3.set_xlabel( 'X '+cunits )
        ax3.set_ylabel( 'Y '+cunits )
        if points is not None :
            idx = ( points[2] > Z[:,:,valz-1][0,0] ) & ( points[2] < Z[:,:,valz+1][0,0] ) 
            ax3.scatter( points[0][idx], points[1][idx], s=points_size, c=points_color, 
                         marker=points_marker, label=points_label )        
         
    
    sx.on_changed(updatex)
    sy.on_changed(updatey)
    sz.on_changed(updatez)
    
    
    # fig.tight_layout()

    return fig, ax1, sx, sy, sz, ( X, Y, Z, NpArr3d )

# -----------------------------------------------------------------------------
def hist( data, w=None, edgecolor='w', color='b', xlabel='variable', ylabel='$counts$',
          sbplt=[1,1,1], label='stat', decimals=2, sep=' ;', title=None, hold=False, 
          alpha=1, leg_label=None ) :
    
    
    if label == 'stat' :
        leg_label = 'stat'
        label = utl.stat( data, decimals=decimals, show=False, multilines=True, out=str  )
        
    if title == 'stat' :
        title = utl.stat( data, decimals=decimals, show=False, multilines=False, sep=sep )        
    
    m,n,p = sbplt
    if hold is False :
        plt.subplot( m,n,p )
    if w is None :
        w = utl.stat( data, show=False )[3]/2
    bins = np.arange(np.nanmin(data), np.nanmax(data) + w, w )
    plt.hist( data, bins, edgecolor=edgecolor, color=color, label=label, alpha=alpha )
    
    if leg_label == 'stat' :
        plt.legend(handlelength=0, handletextpad=0)
    else :
        plt.legend()  
         
    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    
    if title is not None :
        plt.title( title )
        
        
# -----------------------------------------------------------------------------
def static_slices( NpArr3d, 
                   xyz_start, 
                   xyz_end, 
                   nan2val=0,
                   vmin=None, 
                   vmax=None, 
                   array_type='xyz',
                   xyz_lim=None, 
                   stepxy=None, 
                   stepz=None, 
                   zoom=None,
                   xlabel=None, 
                   ylabel=None, 
                   method='nearest', 
                   linewidth=0,
                   filter=None, 
                   filter_sigma=1, 
                   cbar_orientation='vertical',
                   zorder=3,
                   cmap='viridis',
                   plot=True ) :

    if array_type == 'ndarray':

        lenx = NpArr3d.shape[0]
        leny = NpArr3d.shape[1]
        lenz = NpArr3d.shape[2]

        if xyz_lim is None: 
            xyz_lim = [0, lenx, 0, leny, -lenz, 0]

        xu = np.linspace(xyz_lim[0], xyz_lim[1], lenx)
        yu = np.linspace(xyz_lim[2], xyz_lim[3], leny)
        zu = np.linspace(xyz_lim[5], xyz_lim[4], lenz)

        X, Y, Z = np.meshgrid(xu, yu, zu, indexing='ij')

    if array_type == 'xyz':

        for n, i in enumerate( NpArr3d ) :
            NpArr3d[n] = NpArr3d[n].ravel()

        if type( NpArr3d ) in ( tuple, list ) :
            merge_colums = np.zeros( [np.size( NpArr3d[0] ), len( NpArr3d )])
            for n, i in enumerate( NpArr3d ) :
                if type( i ) in ( list, tuple ) :
                    i = np.array( i )
                merge_colums[:,n] = i
            NpArr3d = np.copy( merge_colums )

        if xyz_lim is not None :
            idx = (NpArr3d[:,0] >= xyz_lim[0]) & (NpArr3d[:,0] <= xyz_lim[1]) & \
                (NpArr3d[:,1] >= xyz_lim[2]) & (NpArr3d[:,1] <= xyz_lim[3]) & \
                (NpArr3d[:,2] >= xyz_lim[4]) & (NpArr3d[:,2] <= xyz_lim[5]) 
        
            NpArr3d = NpArr3d[idx,:]
            
        xu = np.unique( NpArr3d[:,0] )
        yu = np.unique( NpArr3d[:,1] )
        zu = np.unique( NpArr3d[:,2] ) 

        X, Y, Z = np.meshgrid( xu, yu, zu, indexing='ij' )
        NpArr3d= NpArr3d[np.lexsort(( NpArr3d[:,2], NpArr3d[:,1], NpArr3d[:,0]))]
        NpArr3d = NpArr3d[:,3].reshape( Z.shape )
        lenx = NpArr3d.shape[0]
        leny = NpArr3d.shape[1]
        lenz = NpArr3d.shape[2]  
        
        xyz_lim = [ X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max() ]

    Min, Max, Mean, Std = utl.stat( NpArr3d  )
    
    if vmin==None: vmin=Mean-2*Std
    if vmax==None: vmax=Mean+2*Std
    
    where_are_NaNs = np.isnan( NpArr3d )
    NpArr3d[ where_are_NaNs ] = nan2val 

    # Create 2d slice matrix 
    if len( xyz_start ) <= 2 :
        xyz_start.append( Z.min() )

    if len( xyz_end ) <= 2 :
        xyz_end.append( Z.max() )

    if stepxy is None :
        stepx = np.min( np.diff( np.unique( X ) ) )
        stepy = np.min( np.diff( np.unique( Y ) ) )
        stepxy = np.min( [ stepx, stepy ] ) 

    if xyz_start[-1] != xyz_end[-1] :

        x0, x1 = xyz_start[0], xyz_end[0]
        y0, y1 = xyz_start[1], xyz_end[1]
        theta = np.arctan((y1 - y0) / ((x1 - x0)+1e-24) )
        length = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
        l = np.arange( 0, length, stepxy )
        x = x0 + l * np.cos(theta)
        y = y0 + l * np.sin(theta)

        if stepz is None :
            z = zu[ ( zu >= xyz_start[-1] ) & ( zu <= xyz_end[-1] ) ]
        else :   
            z = np.linspace( xyz_start[-1], xyz_end[-1], int( ( xyz_end[-1]-xyz_start[-1] )/stepz ) )

        zs = np.full( x.shape, z[0] )
        xs = np.copy( x )
        ys = np.copy( y )
        ls = np.copy( l )
        
        for i, zi in enumerate( z ) :

            if i == 0 : continue 

            xs = np.column_stack( ( xs, x ) )
            ys = np.column_stack( ( ys, y ) )
            zs = np.column_stack( ( zs, np.full( x.shape, z[i] ) ) )
            ls = np.column_stack( ( ls, l ) )

        ps = utl.sp.interpolate.griddata( 
            ( X.ravel(), Y.ravel(), Z.ravel() ), 
            NpArr3d.ravel(), ( xs, ys, zs ),
            method=method )

        print( np.unique(zs), 'ok1' )
        print( np.unique(ys), 'ok2' )
        print( np.unique(xs), 'ok3' )

        xs = np.rot90( ls.reshape( ( len(l), len(z) ) ) )
        ys = np.rot90( zs.reshape( ( len(l), len(z) ) ) ) 
        ps = np.rot90( ps.reshape( ( len(l), len(z) ) ) ) 

    else :

        x0, x1 = xyz_start[0], xyz_end[0]
        y0, y1 = xyz_start[1], xyz_end[1]

        x = np.arange( x0, x1+stepxy, stepxy )
        y = np.arange( y0, y1+stepxy, stepxy )

        xs, ys = np.meshgrid( x, y )
        zs = np.full( xs.shape, xyz_start[2] )

        ps = utl.sp.interpolate.griddata( 
            ( X.ravel(), Y.ravel(), Z.ravel() ), 
            NpArr3d.ravel(), ( xs, ys, zs ),
            method=method )
        
        ys = np.flipud( ys ) 
        ps = np.flipud( ps ) 

    if filter is not None :
        if filter == 'gaussian' :
            ps = utl.sp.ndimage.gaussian_filter( ps, filter_sigma )

    if zoom is not None : 
        xs = utl.resampling( xs, zoom, spl_order=zorder )
        ys = utl.resampling( ys, zoom, spl_order=zorder )
        ps = utl.resampling( ps, zoom, spl_order=zorder )

    if plot is True :
        
        lim = utl.xy2lim( xs, ys )
        ax = plt.pcolormesh( xs, ys, ps, vmin=vmin, vmax=vmax, 
            cmap=cmap,  linewidth=linewidth, shading='auto', 
            snap=True, rasterized=True )

        plt.xlabel( xlabel )
        plt.ylabel( ylabel )
        plt.colorbar( ax, label='Amplitude', orientation=cbar_orientation )
    
    else :
        ax = None

    return xs, ys, ps, ax
        
# -----------------------------------------------------------------------------
def plot_lim( lim, c='r', ax=None ) :
    
    xplot1 = [lim[0], lim[0], lim[1], lim[1], lim[0]]
    yplot1 = [lim[2], lim[3], lim[3], lim[2], lim[2]]

    if ax is None :
        plt.plot( xplot1, yplot1, c=c )
        
    else :
        ax.plot( xplot1, yplot1, c=c )

# -----------------------------------------------------------------------------
def plot_datetime_data( datetime_array, fp=[], window=1, units='D', resamp_factor=1/24, 
                        sbplt_col=1, labels=[] ):
    """
    Plot data against datetime and divide the plots into subplots according to the window and units.

    Parameters:
    datetime_array (numpy.ndarray): Array of datetime64 objects.
    fp (list of numpy.ndarray): List of arrays of data to plot against datetime.
    window (int): Number of units for each subplot.
    units (str): Unit of time for each subplot. Can be 'h' (hours), 'D' (days), 'M' (months), or 'Y' (years).
    resamp_factor (float): Resampling factor.

    Returns:
    None
    """
    timedelta = utl.datetime.timedelta
    # Convert numpy datetime64 to datetime
    datetime_array = datetime_array.astype('datetime64[ns]')

    # Convert units to timedelta
    units_to_timedelta = {'s': timedelta(seconds=1), 'm': timedelta(minutes=1), 'h': timedelta(hours=1), 
                          'D': timedelta(days=1), 'M': timedelta(days=30), 'Y': timedelta(days=365)}
    timedelta_unit = units_to_timedelta[units]

    # Calculate the size of the moving window for the filter
    sampling_rate = utl.get_sampling_rate( datetime_array, units=units )
    filter_size = int( resamp_factor / ( 1/(sampling_rate) ) )

    # Calculate the number of subplots
    total_time_in_seconds = utl.datetime2time( datetime_array[-1] ) - utl.datetime2time( datetime_array[0] )
    window_in_seconds = (window * timedelta_unit).total_seconds()
    num_subplots = int( total_time_in_seconds // window_in_seconds ) + 1 
    sbplt_row = int( np.ceil( num_subplots / sbplt_col ) )

    # Create a new figure with the calculated number of subplots
    fig, axs = plt.subplots( sbplt_row, sbplt_col, sharex=False )

    if labels == [] :
        labels = [ None for i in range(len(fp) ) ]  

    axs = axs.reshape( ( sbplt_row, sbplt_col ) )
    i = 0
    # Plot data
    for k in range( sbplt_row ) :
        for j in range( sbplt_col ) :
        
            # Modify the start time to the start of the day
            start_time = datetime_array[0] + np.timedelta64(i * window * timedelta_unit)
            start_time = np.datetime64(start_time, units)
            end_time = start_time + np.timedelta64(window * timedelta_unit)
            mask = (datetime_array >= start_time) & (datetime_array < end_time)
            n = 0
            for data in fp:
                # Apply the uniform filter to the data
                filtered_data = utl.sp.ndimage.uniform_filter1d(data[mask], size=filter_size)
                axs[k,j].plot(datetime_array[mask], filtered_data, '-', label=labels[n])
                total_length = end_time - start_time
                ext = total_length * 5 / 100
                axs[k,j].set_xlim(start_time-ext, end_time+ext)
                axs[k,j].title.set_text(f'{start_time} to {end_time}')
                n += 1

            if units == 'D':
                axs[k,j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            elif units == 'h':
                axs[k,j].xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
            elif units == 'm':
                axs[k,j].xaxis.set_major_formatter(mdates.DateFormatter('%S'))
            elif units == 'M':
                axs[k,j].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
            elif units == 'Y':
                axs[k,j].xaxis.set_major_formatter(mdates.DateFormatter('%m'))

            i += 1

    if labels[-1] is not None:
        axs[0,0].legend()

    plt.tight_layout()

    # Rotate x-axis labels
    fig.autofmt_xdate()

    # Show the plot
    plt.show()

# -----------------------------------------------------------------------------
def ortho_slices( xyzd,
                  section=[0,None,None],
                  lim=None, 
                  vmin=None, 
                  vmax=None, 
                  figname='Model',
                  grid_points=False, 
                  grid_points_color='y',
                  grid_points_size=1,
                  cmap='jet',
                  xlabel='',
                  ylabel='',
                  colorbar_label='',
                  edgecolors='k',
                  printf=True,
                  aspect='auto',
                  figsize=None,
                  title='',
                  multiple_models=False,
                  suptitle=None,
                  hspace=None,
                  points=None,
                  lines=None,
                  units='',
                  pcolor='k',
                  psize=10,
                  lcolor='k',
                  lwidth=1, 
                  zoom=None,
                  adjust2lim=True,
                  pad=0 ) :
    """
    Plot orthogonal slices of a 3D model.

    Parameters:

    - x (array-like): x-coordinates of the model grid points. Default is None.
    - y (array-like): y-coordinates of the model grid points. Default is None.
    - z (array-like): z-coordinates of the model grid points. Default is None.
    - data (array-like): values of the model at each grid point.
    - section (list, optional): Indices of the slices to plot. Default is [0, None, None].
    - lim (list, optional): Limits of the model grid. Default is None.
    - vmin (float, optional): Minimum value for the color scale. Default is None.
    - vmax (float, optional): Maximum value for the color scale. Default is None.
    - figname (str, optional): Title of the figure. Default is 'Model'.
    - grid_points (bool, optional): Whether to plot the grid points. Default is False.
    - grid_points_color (str, optional): Color of the grid points. Default is 'y'.
    - grid_points_size (float, optional): Size of the grid points. Default is 1.
    - cmap (str, optional): Colormap for the color scale. Default is 'jet'.
    - xlabel (str, optional): Label for the x-axis. Default is ''.
    - ylabel (str, optional): Label for the y-axis. Default is ''.
    - colorbar_label (str, optional): Label for the colorbar. Default is ''.
    - edgecolors (str, optional): Color of the grid lines. Default is 'k'.
    - printf (bool, optional): Whether to print the grid dimensions. Default is True.
    - aspect (str, optional): Aspect ratio of the plot. Default is 'auto'.
    - figsize (tuple, optional): Size of the figure. Default is None.
    - title (str, optional): Title of the plot. Default is None.

    Returns:
    - X_sect (ndarray): x-coordinates of the selected slice.
    - Y_sect (ndarray): y-coordinates of the selected slice.
    - V_sect (ndarray): values of the selected slice.
    """

    if multiple_models is False :
        xyzd = [xyzd]

    if points is not None :
        if type( points ) in ( list, tuple ) :
            if ( np.size( points )  == 3 ) and\
               ( isinstance( points[0], np.ndarray ) ) :
                points = [ points ]

    plt.close( figname )
    plt.figure( num=figname, figsize=figsize )
    V_sect_list = []

    for i, xyzdi in enumerate( xyzd ) :

        x, y, z, data = xyzdi

        if x is not None :
            x = np.array( x ).ravel()
        if y is not None:
            y = np.array( y ).ravel()
        if z is not None:
            z = np.array( z ).ravel()

        data = np.array( data ).ravel()

        model = { 'x':x, 'y':y, 'z':z, 'p':data }

        if lim is not None :
            if len( lim ) < 5 :
                idx = ( model['x'] >= lim[0] ) & ( model['x'] <= lim[1] ) &\
                      ( model['y'] >= lim[2] ) & ( model['y'] <= lim[3] )
            else :
                idx = ( model['x'] >= lim[0] ) & ( model['x'] <= lim[1] ) &\
                      ( model['y'] >= lim[2] ) & ( model['y'] <= lim[3] ) &\
                      ( model['z'] >= lim[4] ) & ( model['z'] <= lim[5] )
        else :
            idx = np.ones( np.size( model['x'] ), dtype=bool )

        xp = np.unique( model['x'][idx] )
        yp = np.unique( model['y'][idx] )
        zp = np.unique( model['z'][idx] )[::-1]

        xstep = np.diff( xp )
        ystep = np.diff( yp )
        zstep = np.diff( zp )
        nx = np.size( xp )
        ny = np.size( yp )
        nz = np.size( zp )

        if printf is True :
            print( f'Grid_dim : ny={ny}, nx={nx}, nz={nz}' )

        if lim is None :
            lim = [ xp.min()-xstep[0]/2, xp.max()+xstep[-1]/2, 
                    yp.min()-ystep[0]/2, yp.max()+ystep[-1]/2, 
                    zp.min()-zstep[0]/2 , zp.max()+zstep[-1]/2 ]

        elif len( lim ) < 5 :
            lim = [ lim[0], lim[1], lim[2], lim[3], 
                    zp.min()-zstep[0]/2 , zp.max()+zstep[-1]/2  ]

        if pad > 0 :
            xp = np.insert( xp, 0, xp[0] - xstep[0]*pad )
            xp = np.append( xp, xp[-1] + xstep[-1]*pad )
            yp = np.insert( yp, 0, yp[0] - ystep[0]*pad )
            yp = np.append( yp, yp[-1] + ystep[-1]*pad )
            zp = np.insert( zp, 0, zp[0] - zstep[0]*pad )
            zp = np.append( zp, zp[-1] + zstep[-1]*pad )
            for si, seci in enumerate( section ) :
                if seci is not None :
                    section[si] = seci + pad

        if adjust2lim is True :
            if ( lim[0] <= xp.min() ) :
                xp = np.insert( xp, 0, xp[0]-xstep[0] )
            if ( lim[1] >= xp.max() ) :
                xp = np.append( xp, xp[-1]+xstep[-1] )
            if ( lim[2] <= yp.min() ) :
                yp = np.insert( yp, 0, yp[0]-ystep[0] )
            if ( lim[3] >= yp.max() ) :
                yp = np.append( yp, yp[-1]+ystep[-1] )
            if ( lim[4] <= zp.min() ) :
                zp = np.insert( zp, 0, zp[0]-zstep[0] )
            if ( lim[5] >= zp.max() ) :
                zp = np.append( zp, zp[-1]+zstep[-1] )

        xstep = np.insert(xstep, 0, xstep[0]) / 2
        ystep = np.insert(ystep, 0, ystep[0]) / 2
        zstep = np.insert(zstep, 0, zstep[0]) / 2

        Xp, Yp, Zp = np.meshgrid( xp, yp, zp )
        Vp = utl.sp.interpolate.griddata( ( model['x'], model['y'], model['z'] ),
                                            model['p'], ( Xp, Yp, Zp ), method='nearest' )

        if zoom is not None :

            Xz = np.round( utl.sp.ndimage.zoom( Xp, zoom, order=1 ), 6 )
            Yz = np.round( utl.sp.ndimage.zoom( Yp, zoom, order=1 ), 6 )
            Zz = np.round( utl.sp.ndimage.zoom( Zp, zoom, order=1 ), 6 )
            Vz = utl.sp.ndimage.zoom( Vp, zoom, order=1 )
            Nanz = np.isnan(Vz)
            edgecolors = 'none'

        pline = []

        if type( section ) == str :

            if ( section == 'midy' ) :
                section = [ None, int( ny / 2 ), None ]
            if ( section == 'midx' ) :
                section = [ int( nx / 2 ), None, None ]
            if ( section == 'midz' )  :
                section = [ None, None, int( int( nz / 2 ) / 2 ) ]
        
        # Z slices
        if (section[0] is None) and (section[1] is None) :
            X_sect = Xp[:,:,section[2]]
            Y_sect = Yp[:,:,section[2]]
            V_sect = Vp[:,:,section[2]]
            unit_val = np.unique( Zp[:,:,section[2]] )
            units = units + ' ' + 'Z'
            step = np.sort( ( zp[section[2]]-zstep[section[2]], 
                              zp[section[2]]+zstep[section[2]] ) )
            xs = model['x']
            ys = model['y']
            if xlabel == '' :
                xlabel = 'X'
            if ylabel == '' :
                ylabel = 'Y'
            if lines is not None :
                for line in lines :
                    pline.append( [ line[0], line[1] ] )
            if zoom is not None :
                V_sect = utl.sp.interpolate.griddata( ( Xz[~Nanz], 
                                                        Yz[~Nanz], 
                                                        Zz[~Nanz] ), 
                                                        Vz[~Nanz], 
                                                     ( Xz[:,:,1], Yz[:,:,1], 
                                                       np.full( Xz[:,:,1].shape, unit_val ) ),
                                                     method='nearest' )
                X_sect = Xz[:,:,1]
                Y_sect = Yz[:,:,1]

        # Y slices
        elif (section[0] is None) and (section[2] is None) :

            X_sect = Yp[:,section[1],:]
            Y_sect = Zp[:,section[1],:]
            V_sect = Vp[:,section[1],:]
            unit_val = np.unique( Xp[:,section[1],:] )
            units = units + ' ' + 'X'
            step = np.sort( ( xp[section[1]]-xstep[section[1]], 
                              xp[section[1]]+xstep[section[1]] ) )
            xs = model['y']
            ys = model['z']
            if xlabel == '' :
                xlabel = 'Y'
            if ylabel == '' :
                ylabel = 'Z'
            if lines is not None :
                for line in lines :
                    if len( line ) > 2 :
                        pline.append( [ line[1], line[2] ] )
            if zoom is not None :
                V_sect = utl.sp.interpolate.griddata( ( Xz[~Nanz], 
                                                       Yz[~Nanz], 
                                                       Zz[~Nanz] ), 
                                                       Vz[~Nanz], 
                                                     ( np.full( Yz[:,1,:].shape, unit_val ), 
                                                       Yz[:,1,:], Zz[:,1,:] ),
                                                     method='nearest' )
                X_sect = Yz[:,1,:]
                Y_sect = Zz[:,1,:]

        # X slices
        elif ( section[1]is None) and (section[2] is None ) :

            X_sect = Xp[section[0],:,:]
            Y_sect = Zp[section[0],:,:]
            V_sect = Vp[section[0],:,:]
            unit_val = float( np.unique( Yp[section[0],:,:] ) )
            units = units + ' ' + 'Y'
            step = np.sort( ( yp[section[0]]-ystep[section[0]], 
                              yp[section[0]]+ystep[section[0]] ) )
            xs = model['x']
            ys = model['z']
            if xlabel == '' :
                xlabel = 'X'
            if ylabel == '' :
                ylabel = 'Z'
            if lines is not None :
                for line in lines :
                    if len( line ) > 2 :
                        pline.append( [ line[0], line[2] ] )
            if zoom is not None :
                V_sect = utl.sp.interpolate.griddata( ( Xz[~Nanz], 
                                                       Yz[~Nanz], 
                                                       Zz[~Nanz] ), 
                                                       Vz[~Nanz], 
                                                     ( Xz[1,:,:], np.full( Xz[1,:,:].shape, unit_val ) , 
                                                       Zz[1,:,:] ),
                                                     method='nearest' )
                X_sect = Xz[1,:,:]
                Y_sect = Zz[1,:,:]

        else :
            raise ValueError( 'section list must have exactly two None values,'+\
                              'e.g., [0, None, None]')

        plt.subplot( len( xyzd ), 1, i+1 )

        if type( vmin ) in ( list, tuple ) :
            ivmin = vmin[i]
        else :
            ivmin = vmin

        if type( vmax ) in ( list, tuple ) :
            ivmax = vmax[i]
        else :
            ivmax = vmax

        pcmax = plt.pcolormesh( X_sect, Y_sect, V_sect, 
                                cmap = cmap,
                                edgecolors = edgecolors, 
                                linewidth=0.01, 
                                shading='auto',
                                vmin = ivmin, 
                                vmax = ivmax )

        plt.gca().set_aspect( aspect )

        plt.colorbar( pcmax, label=colorbar_label )
        plt.xlabel( xlabel )
        plt.ylabel( ylabel )

        if grid_points is True :
            plt.scatter( xs, ys, 
                         c=grid_points_color, 
                         s=grid_points_size, 
                         label='grid_points')
            plt.legend()

        if len( pline ) > 0 :
            for line in pline :
                plt.plot( line[0], line[1], c=lcolor, lw=lwidth )

        if points is not None :
            pti = points[i]
            if 'X' in units :
                idx = ( pti[0] >= step[0] ) & ( pti[0] <= step[1]  )
                plt.scatter( points[1][idx], points[2][idx], c=pcolor, s=psize, label='points' )
            if 'Y' in units :
                idx = ( pti[1] >= step[0] ) & ( pti[1] <= step[1])
                plt.scatter( points[0][idx], points[2][idx], c=pcolor, s=psize, label='points' )
            if 'Z' in units :
                idx = ( pti[2] >= step[0] ) & ( pti[2] <= step[1] )
                plt.scatter( points[0][idx], points[1][idx], c=pcolor, s=psize, label='points' )

        if lim is not None :
            if 'X' in units :
                plt.xlim( ( lim[2], lim[3] ) )
                plt.ylim( ( lim[4], lim[5] ) )
            if 'Y' in units :
                plt.xlim( ( lim[0], lim[1] ) )
                plt.ylim( ( lim[4], lim[5] ) )
            if 'Z' in units :
                plt.xlim( ( lim[0], lim[1] ) )
                plt.ylim( ( lim[2], lim[3] ) )
        
        if title is None :
            ititle = f'Model {i}'
            plt.title( ititle )

        if type( title ) in ( list, tuple ) :
            plt.title( title[i] )

        V_sect_list.append( V_sect )
        
    plt.subplots_adjust( hspace=hspace )

    if suptitle is None :
        suptitle = f'Slice at {section} i.e., {float( unit_val )} {units}'
    plt.suptitle( suptitle )

    if len( V_sect_list ) == 1 :
        V_sect = V_sect_list[0]
    else :
        V_sect = V_sect_list

    return X_sect, Y_sect, V_sect