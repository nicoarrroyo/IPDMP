""" Individual Project Machine Learning Software (IPMLS-25-02) """
""" Update Notes (from previous version IPMLS-24-12)
- earth engine
    - tested implementation
    - time-limited try-except for connecting
    - created separate file for IPMLS earth engine functions
- optimisations
    - landsat process reduced to a single function
        - ~ 8 second improvement for compression=1 (175 sec to 167 sec)
        - ~ 400 lines to ~ 160 (increased for other functionality)
    - index calculation reduced to a single function
    - minimising saved variables
    - note: university PC is much faster that personal pc for large tasks
        - 89 seconds uni vs 435 seconds personal
- output plots improved
    - gradient legend to show what the colours mean
    - inline plots are faster overall, but Tk plots allow zooming
    - added capability to save full resolution image files
- sentinel 2 (nothing yet)
- machine learning (nothing yet)
- cloud masking (nothing yet)
- compositing (nothing yet)
- separating general water and reservoir water (nothing yet)
"""
# =============================================================================
# - to install a conda library
#     - %UserProfile%\miniconda3\condabin\activate
#     - conda activate ee
#     - conda install WhateverLibraryYouWant
# =============================================================================
# %% Start
# %%% External Library Imports
import time
MAIN_START_TIME = time.monotonic()
from PIL import Image
import os
import numpy as np

# %%% Internal Function Imports
from image_functions import compress_image, plot_image, composite, cloud_mask
from calculation_functions import get_indices
from satellite_functions import get_landsat_bands
# %% General Landsat Function
do_l7 = False
do_l8 = False
do_l9 = True
save_images = False

# main parent path where all image files are stored
home = 'P:\\Individual Project'
compression = 30 # 1 for full-sized images, bigger integer for smaller images
dpi = 1000 # 3000 for full resolution, below 1000, images become fuzzy
plot_size = (3, 3) # larger plots increase detail and pixel count

print('compression factor:', compression)
print('dots per inch (dpi):', dpi)
print(f'computing indices for landsat 7: {do_l7}, '
      f'landsat 8: {do_l8}, '
      f'landsat 9: {do_l9}')
print(f'saving images: {save_images}')

def get_landsat(landsat_number, folder, do_landsat):
    print('===================')
    print(f'||LANDSAT {landsat_number} START||')
    print('===================')
    file_paths = []
    images = []
    
    # %%% Establishing Paths, Opening and Resizing Images, and Creating Image Arrays
    print('establishing paths, opening and resizing images, creating image arrays', 
          end='... ')
    start_time = time.monotonic()
    
    satellite = f'\\Landsat {landsat_number}\\'
    PATH = home + satellite + folder
    os.chdir(PATH)
    
    (landsat_and_sensor, processing_correction_level,
    wrs_path_row, acquisition_date,
    processing_date, collection_number,
    collection_category) = folder.split('_')
    
    if processing_correction_level[1] == '1':
        PREFIX = folder + '_B'
    else:
        if processing_correction_level[2] == 'S':
            PREFIX = folder + '_SR_B'
        else:
            PREFIX = folder + '_B'
    
    bands = get_landsat_bands(landsat_number)
    for band in bands:
        file_paths.append(PREFIX + band + '.TIF')
    
    for file_path in file_paths:
        images.append(Image.open(file_path))
    
    width, height = images[1].size
    
    images, image_arrays, size = compress_image(compression, width, height, images)

    time_taken = time.monotonic() - start_time
    print(f'complete! time taken: {round(time_taken, 2)} seconds')
    
    # %%% Calculating Water Indices
    print('populating water index arrays', end='... ')
    start_time = time.monotonic()
    
    # Precompute masks and metrics
    blue, green, nir, swir1, swir2 = image_arrays
    np.seterr(divide='ignore', invalid='ignore')

    minimum = -1
    maximum = 1
    
    ndwi, mndwi, awei_sh, awei_nsh = get_indices(blue, green, nir, swir1, swir2)
    
    ndwi = np.maximum(np.minimum(ndwi, maximum), minimum) # 'cleaning'
    mndwi = np.maximum(np.minimum(mndwi, maximum), minimum) # 'cleaning'
    awei_sh = np.maximum(np.minimum(awei_sh, maximum), minimum) # 'cleaning'
    awei_nsh = np.maximum(np.minimum(awei_nsh, maximum), minimum) # 'cleaning'
    
    indices = [ndwi, mndwi, awei_sh, awei_nsh]
    
    time_taken = time.monotonic() - start_time
    print(f'complete! time taken: {round(time_taken, 2)} seconds')
    
    # %%% Showing Images
    if do_landsat:
        print('displaying and saving water index images...')
        start_time = time.monotonic()
        plot_image(indices, landsat_number, plot_size, 
                   minimum, maximum, compression, dpi, save_images)
        time_taken = time.monotonic() - start_time
        print(f'complete! time taken: {round(time_taken, 2)} seconds')
    
    return indices
# %% General Sentinel Function
do_s2 = True
# %% Running Functions    
"""
Landsat 7 has only one Short-Wave Infrared (SWIR) band, which means that Autom-
ated Water Extraction Index (AWEI) cannot be properly calculated. 
The AWEI is calculated anyway, however please note that the SWIR2 band is repl-
aced with the Mid-Wave Infrared (MIR) band. 
"""
if do_l7:
    l7_indices = get_landsat(landsat_number=7, 
                             folder='LE07_L2SP_201023_20000619_20200918_02_T1', 
                             do_landsat=do_l7)

"""
Landsat 8 has no Mid-Wave Infrared (MIR) band. This may have effects on calcul-
ating the Modified Normalised Water Index (MNDWI). (must check)
"""
if do_l8:
    l8_indices = get_landsat(landsat_number=8, 
                             folder='LC08_L2SP_201024_20241120_20241203_02_T1', 
                             do_landsat=do_l8)

"""
Landsat 9 has the same band imagers as Landsat 8, meaning that it lacks the MIR
band. 
"""
if do_l9:
    l9_indices = get_landsat(landsat_number=9, 
                             folder='LC09_L2SP_201023_20241011_20241012_02_T1', 
                             do_landsat=do_l9)
# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME
print(f'total time taken for all processes: {round(TOTAL_TIME, 2)} seconds')
