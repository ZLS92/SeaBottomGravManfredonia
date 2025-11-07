#!/usr/bin/env python3
# =================================================================================
#%% Import modules to run this script from command line

import argparse
import os
from modules.utils import create_log_file

# =================================================================================
#%% Script content 
def main():

    # =============================================================================
    # In[0]: 
    # File Headers
    cell_sep = "\n# =============================================================================\n"
    print( cell_sep + "Cell 0: File header" )

    file_headers ="""
    \tCreated on Mon Nov 3 17:57:00 2025

    \t@author: Zampa Luigi Sante
    \t@email_1: lzampa@ogs.it
    \t@org: National Institute of Oceanography and Applied Geophysics - OGS
    """
    print( file_headers )

    # =============================================================================
    # In[1]: 
    # Import base modules
    print( cell_sep + "Cell 1: Import base modules")

    # Import base modules
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import copy
    import importlib

    # Print base modules
    recording = False
    with open(__file__, 'r') as file:
        for line in file:
            if "Import base" in line:
                recording = True
                continue
            if "Print base" in line:
                break
            if recording:
                print('\t' + line.strip())
    file.close()

    # Import reload function
    rld = importlib.reload
    # Set "s" variable as system separator
    s = os.sep
    # Print variables "s" and "rld"
    print( '\ts = os.sep\n\trld = importlib.reload' )

    # Close all figures
    plt.close('all')

    # Get current file name and path 
    filename = os.path.basename( __file__ ).split('.')[0]
    filepath = os.path.abspath( __file__ )

    # =============================================================================
    # In[2]: 
    # Import local modules
    print( cell_sep + "Cell 2: Import local modules")

    # Import local modules
    from modules import utils as utl
    from modules import gravmag_processing as gmp
    from modules import gravmag_analysis as gma
    from modules import raster_tools as rt
    from modules import TePrismTess as Te
    from modules import plot_tools as pt

    # Print local modules
    recording = False
    with open( filepath, 'r') as file:
        for line in file:
            if "Import local" in line:
                recording = True
                continue
            if "Print local" in line:
                break
            if recording:
                print('\t' + line.strip())
    file.close()

    # =============================================================================
    # In[3]: 
    # Set local dir. paths
    print( cell_sep + "Cell 3: Set local dir. paths\n")

    # Set project home dir.
    hdir = os.sep.join( filepath.split( os.sep )[:-2] )

    # Create figures dir if not exists
    os.makedirs( hdir +s+ 'figures', exist_ok=True )

    # Create local paths dictionary
    p = {
        "cwd" :         os.getcwd(),
        "home" :        hdir,
        "fig":          hdir +s+ 'figures',
        "data":         hdir +s+ 'data',
        }

    # Check if the paths exist, if not raise an error
    for key, value in p.items():
        if not os.path.exists(value):
            raise FileNotFoundError(f"Path '{value}' does not exist. Please check the directory structure.")

    # Print path dictionary 
    print( '\tp = {' )
    for key, value in sorted(p.items(), key=lambda x: x[0]):
        print( "\t'{}' : '{}'".format(key, value) )
    print( '\t}' )

    # =============================================================================
    # In[4]:
    # Set study area limits 
    print( cell_sep + "Cell 4: Set study area limits\n")

    # Set limits dictionary
    l = {} 
    
    #Set parameters (manual)
    l['y1'] = 41.85 # N # ~max(gdat['yg'])
    l['y0'] = 41.15 # S # ~min(gdat['yg'])
    l['x1'] = 16.85 # E # ~max(gdat['xg'])
    l['x0'] = 15.85 # W # ~min(gdat['xg'])

    l['g'] = [l['x0'],l['x1'],l['y0'],l['y1']]

    # prjm = Planar projection (mercator tangent to central lat lon of study area ) 
    prjm = 32633 # EPSG UTM33N
    # prjg = Geographic projection (WGS84)
    prjg = 4326  # EPSG WGS84

    # Convert limits to metric coordinates
    l['m'] = utl.prj_lim( l['g'], prjg, prjm )

    # Print limits dictionary
    print( '\tl = {' )
    for key, value in sorted(l.items(), key=lambda x: x[0]):
        print( "\t'{}' : {}".format(key, value) )
    print( '\t}' )

    # =============================================================================
    # In[5]:
    # Import and process gravity data
    print( cell_sep + "Cell 5: Import and process gravity data\n")

    # Set gravityt data dictionary
    gd = {'ogs60':{}, 'ss':{}}

    # Sea Sea Bottom data 1960 
    gd['ogs60']['or'] = np.genfromtxt(p['data']+s+'ogs'+s+'grav'+s+'grav_OGS60.csv', skip_header=1) # original data table
    gd['ogs60']['or'] = gd['ogs60']['or'][gd['ogs60']['or'][:,0]!= 1311] # delete station number 1311 (Bad data point!)
    gd['ogs60']['xg'], gd['ogs60']['yg'] = utl.prjxy(4265,4326,gd['ogs60']['or'][:,2], gd['ogs60']['or'][:,1]) 
    gd['ogs60']['hs'] = gd['ogs60']['or'][:,3] # station_height
    gd['ogs60']['gobs'] = gd['ogs60']['or'][:,4] # observed_gravity
    gd['ogs60']['yr'] = np.repeat(1960, np.size(gd['ogs60']['xg']))
    PotErr = -15.42 # Estimated Potsdam error

    # Sea Sea Bottom data 1983
    # gd['ogs83']['or'] = np.genfromtxt(p['ogs']+s+'grav'+s+'grav_OGS83.csv', skip_header=1) # original data table
    # gd['ogs83']['xg'], gd['ogs83']['yg'] = gd['ogs83']['or'][:,2], gd['ogs83']['or'][:,1]
    # gd['ogs83']['hs'] = -gd['ogs83']['or'][:,-1] # station_height
    # gd['ogs83']['gobs'] = gd['ogs83']['or'][:,-2] # observed_gravity
    # gd['ogs83']['perr'] = -14.6 # Potsdam error
    # gd['ogs83']['yr'] = np.repeat(1983, np.size(gd['ogs83']['xg']))

    # Satellite altimetry derived gravity (Sandwell & Smith, 2019)
    gd['ss']['or'] = np.genfromtxt( p['data']+s+'ss'+s+'grav_ss.txt', skip_header=0 )
    gd['ss']['xg'],gd['ss']['yg'] = gd['ss']['or'][:,0], gd['ss']['or'][:,1]
    gd['ss']['fa'] = gd['ss']['or'][:,2]
    gd['ss']['fa_err'] = gd['ss']['or'][:,3] 

    print( '\tImported gravity data files:' )
    print( f"\t- Sea Bottom 1960:\n\t{p['data']+s+'ogs'+s+'grav'+s+'grav_OGS60.csv'}" )
    print( f"\t- Satellite altimetry derived gravity:\n\t{p['data']+s+'ss'+s+'grav_ss.txt'}" )

    # ============================================================================
    # In[6]:
    # Transform grav coordinates from geographic to metric
    print( cell_sep + "Cell 6: Transform grav coordinates from geographic to metric\n")

    for i in gd:
        gd[i]['xm'],gd[i]['ym'] = utl.prjxy(prjg, prjm, gd[i]['xg'], gd[i]['yg'] )

    # =============================================================================
    # In[7]:
    # Load/create DTBM (Digital Terrein-Bathymetric Model)
    print( cell_sep + "Cell 7: Load/create DTBM (Digital Terrein-Bathymetric Model)\n")

    # Set dtbm dictionary
    dtbm = rt.gdal.Open( p['data']+s+'dtbm'+s+'ManfredoniaDBTM.tif' )

    # =============================================================================
    # In[8]:
    # Plot Data 
    print( cell_sep + "Cell 8: Plot Data\n")

    # Plot DTBM
    ax1 = rt.pltr( dtbm, cmap='terrain', axis=True, clabel='m' )
    pt.plot_lim( l['g'], ax=ax1 )
    plt.savefig( p['fig']+s+'dtbm_area.png', dpi=200, bbox_inches='tight', pad_inches=0.3  )

    # Plot Grav. data over DTBM
    ax2 = rt.pltr( dtbm, cmap='terrain', axis=True, lim=l['g'],
                 vmin=-150, vmax=500, clabel='m' )
    ax2.scatter( gd['ss']['xg'], gd['ss']['yg'], c='red', s=5, label='S&S', alpha=0.5)
    ax2.scatter( gd['ogs60']['xg'], gd['ogs60']['yg'], c='k', s=10, label='SeaBott60', alpha=0.7)
    ax2.legend()
    plt.savefig( p['fig']+s+'grav_data.png', dpi=200, bbox_inches='tight', pad_inches=0.3 )

    # =============================================================================
    # In[9]:
    # Fix sea bottm station depth with DTBM
    print( cell_sep + "Cell 9: Fix sea bottm station depth\n")

    # Extract raster values at grav. station locations
    for i in gd:
        gd[i]['hdtbm'] = rt.xy2rasterVal(dtbm, gd[i]['xg'], gd[i]['yg'])[0]

    # Validate OGS station depth based on the accuracy of the original grav statoin depth
    # If (h_obs-h_dtm)>2+5%h_dtm the bathymetric value is =h_dtm, else is =h_obs
    
    gd['ogs60']['hnew'] = utl.copy.deepcopy( gd['ogs60']['hs'] )
    idx = ( np.abs( gd['ogs60']['hdtbm'] - gd['ogs60']['hs'] ) ) >= \
        ( 2 + 5 * np.abs( gd['ogs60']['hdtbm'] ) / 100 ) 
    gd['ogs60']['hnew'][idx] = gd['ogs60']['hdtbm'][idx]

    # =============================================================================
    # In[10]:
    # Free water correction/anomaly for sea bottom data 1960 
    print( cell_sep + "Cell 10: Free water correction/anomaly for sea bottom data 1960 \n")

    # Normal gravity (with atmospheric correction)    
    gd['ogs60']['gth'] = gmp.gn_80(gd['ogs60']['yg'])-gmp.atm_c(0)

    # Free water correction
    gd['ogs60']['fwc'] = gmp.fw_c(gd['ogs60']['hnew'])

    # Free water anomaly + Potsdam error
    gd['ogs60']['fa'] = gd['ogs60']['gobs'] + PotErr -\
        ( gd['ogs60']['gth'] + gd['ogs60']['fwc'] ) 
    
    # =============================================================================
    # In[11]:
    # Topo. effect via Tesseroids prism model

    # Topo effect for sea bottom data 1960
    # Using tesseroids in the far field (ie., tess=True, from R1 to R2)
    # or prisms (ie., tess=False) will not change significantly the results,
    # differences are in the range of +-0.005 mGals
    SBTe = Te.te( 
        x = gd['ogs60']['xg'], 
        y = gd['ogs60']['yg'], 
        dtm1 = p['data']+s+'dtbm'+s+'ManfredoniaDBTM.tif',
        dtm2 = p['data']+s+'dtbm'+s+'GEBCOTopoBatItaly.tif',
        z = gd['ogs60']['hnew'], 
        R1 = 9000, 
        R2 = 180000, 
        gs1 = 100, 
        gs2 = 2500, 
        st_type = 2,
        new_dtm = True, 
        tess=True,
        ply_coast = p['data'] +s+ 'coastline' +s+ 'coastline_ply.shp',
        output_file = p['cwd']+s+'Te_corretions'+s+'TeSeaBottom'
        )
    gd['ogs60']['te'] = SBTe[0]
    
    # Topo effect for S&S data
    SSTe = Te.te( 
        x = gd['ss']['xg'], 
        y = gd['ss']['yg'], 
        dtm1 = p['data']+s+'dtbm'+s+'ManfredoniaDBTM.tif',
        dtm2 = p['data']+s+'dtbm'+s+'GEBCOTopoBatItaly.tif',
        R1 = 9000, 
        R2 = 180000, 
        gs1 = 100, 
        gs2 = 2500, 
        st_type = 0,
        new_dtm = True, 
        tess=True,
        autoid_seasurf_st=True,
        ply_coast = p['data'] +s+ 'coastline' +s+ 'coastline_ply.shp',
        output_file = p['cwd']+s+'Te_corretions'+s+'TeSS'
        )
    gd['ss']['te'] = SSTe[0]

    # =============================================================================
    # In[12]:
    # Bouger anomaly 
    print( cell_sep + "Cell 12: Bouger anomaly \n")

    # Bouger anomaly for sea bottom data 1960
    gd['ogs60']['ba'] = gd['ogs60']['fa'] - gd['ogs60']['te'].ravel()

    # Bouger anomaly for S&S data
    gd['ss']['ba'] = gd['ss']['fa'] - gd['ss']['te'].ravel()

    # =============================================================================
    # In[13]:
    # Grid and plot 
    print( cell_sep + "Grid and plot\n")

    # Grid sea bottom Bouger anomaly
    Xm, Ym, Bgsb = gma.xyz2grid( 
        gd['ogs60']['xm'], 
        gd['ogs60']['ym'], 
        gd['ogs60']['ba'], 
        lim = l['m'], 
        gstep = 1000, 
        method = 'cubic',
        plot = False,
        )[0]
    nan_mask = np.isnan(Bgsb)

    # Plot Sea Bottom Bouger anomaly
    utl.plta(  
        Bgsb, 
        lim=[ Xm, Ym ],
        cmap = 'rainbow', 
        axis=True,
        vmin = 80, 
        vmax = 40, 
        tit = 'Sea Bottom 1960 Bouger Anomaly (mGals)', 
        clabel = 'mGals', 
        )
    
    plt.savefig( p['fig']+s+'BougSeaBottom1960.png', dpi=200, bbox_inches='tight', pad_inches=0.3 )
    
    # Grid S&S Bouger anomaly
    Xm, Ym, Bgss= gma.xyz2grid( 
        gd['ss']['xm'], 
        gd['ss']['ym'], 
        gd['ss']['ba'], 
        lim = l['m'], 
        gstep = 1000, 
        method = 'cubic',
        plot = False,
        )[0]
    Bgss[nan_mask] = np.nan

    # Plot S&S Bouger anomaly
    utl.plta(  
        Bgss, 
        lim=[ Xm, Ym ],
        cmap = 'rainbow', 
        axis=True,
        vmin = 80, 
        vmax = 40, 
        tit = 'S&S Bouger Anomaly (mGals)', 
        clabel = 'mGals', 
        )
    plt.savefig( p['fig']+s+'BougSS.png', dpi=200, bbox_inches='tight', pad_inches=0.3 )


# =================================================================================
#%% To run this script from command line 
if __name__ == "__main__" : 

    # Description for the argument parser
    parser_description = "Run specific lines of the script."
    parser = argparse.ArgumentParser(description=parser_description)

   # Add -log_file argument
    help = "Boolean flag to create a log file (True, False, 1, 0)"
    parser.add_argument( '-log_file', 
        type = lambda x: (str(x).lower() in ['true', '1']), 
        default = True, help = help,)

    # Parse the command-line arguments
    args = parser.parse_args()

    # RUN THE MAIN FUNCTION

    # If log_file is True, create a log file
    if args.log_file:
        
        create_log_file( 
            main_function = main,
            file_name = os.path.basename(__file__), 
            convert_to_pdf = True 
            )

    # If log_file is False, do not create a log file
    else:
        main()



