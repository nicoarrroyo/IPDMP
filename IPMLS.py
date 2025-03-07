""" Individual Project Machine Learning Software (IPMLS-25-03) """
""" Update Notes (from previous version IPMLS-25-02)
- earth engine
- optimisations
    - low resolution option for sentinel 2
- output plots improved
- sentinel 2
    - functional index calculation, plot outputs, and plot saving
    - new general function for landsat and/or sentinel
- machine learning
- cloud masking
    - now functional and included before index calculation
- compositing
- separating general water and reservoir water
"""
# %% Start
# %%% External Library Imports
import time
MAIN_START_TIME = time.monotonic()
from PIL import Image
import os
import numpy as np
import threading

# %%% Internal Function Imports
from image_functions import compress_image, plot_image, upscale_image_array
from calculation_functions import get_indices
from satellite_functions import get_landsat_bands, get_sentinel_bands
from misc_functions import table_print, performance_estimate

# %%% Connect with Earth Engine project (ee)
gee_connect = False
if gee_connect:
    from earth_engine_functions import authenticate_and_initialise
    print("connecting to google earth engine", end="... ")
    start_time = time.monotonic()
    thread = threading.Thread(target=authenticate_and_initialise)
    thread.start()
    thread.join(timeout=5)
    
    if thread.is_alive():
        print("Operation timed out after 5 seconds") # prevents slow network connection
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")

# %%% General Image and Plot Properties
compression = 15 # 1 for full-sized images, bigger integer for smaller images
dpi = 3000 # 3000 for full resolution, below 1000, images become fuzzy
plot_size = (3, 3) # larger plots increase detail and pixel count
save_images = False
high_res_Sentinel = False # use finer 10m spatial resolution (slower)
# main parent path where all image files are stored
HOME = "C:\\Users\\nicol\\Documents\\UoM\\YEAR 3\\Individual Project\\Downloads"

# %% General Mega Giga Function
do_l7 = False
do_l8 = True
do_l9 = False

do_s2 = True
perf_est = performance_estimate(gee_connect, compression, dpi, plot_size, 
                                save_images, high_res_Sentinel, 
                                do_l7, do_l8, do_l9, do_s2)
if perf_est <= 0.25:
    est_duration = "low"
elif perf_est > 0.25 and perf_est <= 0.5:
    est_duration = "medium"
elif perf_est < 0.5 and perf_est <= 0.75:
    est_duration = "high"
else:
    est_duration = "very high"

def get_sat(sat_name, sat_number, folder, do_sat):
    sat_start_time = time.monotonic()
    if sat_name == "Landsat":
        Landsat = True
        Sentinel = False
        title_line = "==================="
    elif sat_name == "Sentinel":
        Landsat = False
        Sentinel = True
        title_line = "===================="
    else:
        print("Bad satellite name" + sat_name)
        return
    
    print(title_line)
    print(f"||{sat_name} {sat_number} Start||")
    print(title_line)
    table_print(compression=compression, DPI=dpi, plot_size=plot_size, 
                do_sat=do_sat, save_images=save_images, GEE=gee_connect, 
                high_res_Sentinel=high_res_Sentinel, sim_duration=est_duration)
    
    # %%% 1. Establishing Paths, Opening and Resizing Images, and Creating Image Arrays
    print("establishing paths, opening and resizing images, creating image arrays", 
          end="... ")
    start_time = time.monotonic()
    
    file_paths = []
    satellite = f"\\{sat_name} {sat_number}\\"
    # %%%% 1a. Landsat Case
    if Landsat:
        res = "30m"
        path = HOME + satellite + folder
        os.chdir(path)
        
        (landsat_and_sensor, processing_correction_level,
        wrs_path_row, acquisition_date,
        processing_date, collection_number,
        collection_category) = folder.split("_")
        
        if processing_correction_level[1] == "1":
            prefix = folder + "_B"
        else:
            if processing_correction_level[2] == "S":
                prefix = folder + "_SR_B"
            else:
                prefix = folder + "_B"
        
        bands = get_landsat_bands(sat_number)
        for band in bands:
            file_paths.append(prefix + band + ".TIF")
    # %%%% 1b. Sentinel Case
    elif Sentinel:
        path = HOME + satellite + folder + "\\GRANULE"
        
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if len(subdirs) == 1:
            path = (f"{path}\\{subdirs[0]}\\")
            os.chdir(path)
        else:
            print("Too many subdirectories in 'GRANULE':", len(subdirs))
            return
        
        if high_res_Sentinel:
            res = "10m20m"
            path_10 = (path + "IMG_DATA\\R10m\\") # finer resolution for bands 2, 3, 8
            path_20 = (path + "IMG_DATA\\R20m\\") # regular resolution for bands 11, 12
        else:
            res = "60m"
            path_60 = (path + "IMG_DATA\\R60m\\") # lower resolution for all bands
        
        (sentinel_name, instrument_and_product_level, datatake_start_sensing_time, 
         processing_baseline_number, relative_orbit_number, tile_number_field, 
         product_discriminator_and_format) = folder.split("_")
        prefix = (f"{tile_number_field}_{datatake_start_sensing_time}_B")
        bands = get_sentinel_bands(sat_number, high_res_Sentinel)
        
        for band in bands:
            if high_res_Sentinel:
                max_pixels = Image.MAX_IMAGE_PIXELS
                Image.MAX_IMAGE_PIXELS = None # DecompressionBombWarning Suppression
                if band == "02" or band == "03" or band == "08":
                    file_paths.append(path_10 + prefix + band + "_10m.jp2")
                else:
                    file_paths.append(path_20 + prefix + band + "_20m.jp2")
                Image.MAX_IMAGE_PIXELS = max_pixels
            else:
                file_paths.append(path_60 + prefix + band + "_60m.jp2")
    # %%% 1. Continued
    image_arrays, size = compress_image(compression, file_paths)
    
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 2. Masking Clouds
    print("masking clouds", end="... ")
    start_time = time.monotonic()
    # %%%% 2a. Landsat Case
    if Landsat:
        clouds_array, size = compress_image(compression, folder + "_QA_PIXEL.TIF")
        clouds_array = (clouds_array / 65536.0) * 100
        # FLAG div 2**16 because it is being shown not with the gradient plot 
        # but with regular imshow pltshow
        # * 100 to make it into a cloud probability percentage
    # %%%% 2b. Sentinel Case
    elif Sentinel:
        path = HOME + satellite + folder + "\\GRANULE\\" + subdirs[0] + "\\QI_DATA\\"
        if high_res_Sentinel:
            image_arrays[-1] = upscale_image_array(image_arrays[-1], factor=2)
            image_arrays[-2] = upscale_image_array(image_arrays[-2], factor=2)
            path = path + "MSK_CLDPRB_20m.jp2"
        else:
            path = path + "MSK_CLDPRB_60m.jp2"
        clouds_array, size = compress_image(compression, path)
    # %%% 2. Continued
    clouds_array = np.where(clouds_array > 50, 100, clouds_array) # if it's 
    # more likely to  be a cloud than not, make it 100% a cloud
    cloud_positions = np.argwhere(clouds_array == 100) # find the position of 
    # each cloud and store it in an array where [y, x]
    for image_array in image_arrays:
        image_array[cloud_positions[:, 0], cloud_positions[:, 1]] = 0.00001
    
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 3. Calculating Water Indices
    print("populating water index arrays", end="... ")
    start_time = time.monotonic()
    
    if Sentinel:
        globals()["im_arr_sen"] = image_arrays
    
    blue, green, nir, swir1, swir2 = image_arrays
    indices = get_indices(blue, green, nir, swir1, swir2)
    
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 4. Showing Images
    if do_sat:
        minimum = -1
        maximum = 1
        if save_images:
            print("displaying and saving water index images...")
        else:
            print("displaying water index images...")
        start_time = time.monotonic()
        plot_image(indices, sat_number, plot_size, minimum, maximum, 
                   compression, dpi, save_images, res)
        time_taken = time.monotonic() - start_time
        print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 5. Satellite Output
    time_taken = time.monotonic() - sat_start_time
    print(f"{sat_name} {sat_number} complete! "
          f"time taken: {round(time_taken, 2)} seconds")
    return indices
# %% Running Functions    
"""
Landsat 7 has only one Short-Wave Infrared (SWIR) band, which means that Autom-
ated Water Extraction Index (AWEI) cannot be properly calculated. 
The AWEI is calculated anyway, however please note that the SWIR2 band is repl-
aced with the Mid-Wave Infrared (MIR) band. 
"""
if do_l7:
    l7_indices = get_sat(sat_name="Landsat", sat_number=7, 
                             folder="LE07_L1TP_201023_20230820_20230915_02_T1", 
                             do_sat=do_l7)

"""
Landsat 8 has no Mid-Wave Infrared (MIR) band. MNDWI is calculated with SWIR2, 
which is the correct method. 
"""
if do_l8:
    l8_indices = get_sat(sat_name="Landsat", sat_number=8, 
                             folder="LC08_L2SP_201024_20241120_20241203_02_T1", 
                             do_sat=do_l8)

"""
Landsat 9 has the same band imagers as Landsat 8, meaning that it lacks the MIR
band. 
"""
if do_l9:
    l9_indices = get_sat(sat_name="Landsat", sat_number=9, 
                             folder="LC09_L1TP_201023_20241011_20241011_02_T1", 
                             do_sat=do_l9)

"""
Sentinel 2 has varying resolution bands, with Blue (2), Green (3), Red (4), and 
NIR (8) having 10m spatial resolution, while SWIR 1 (11) and SWIR 2 (12) have 
20m spatial resolution. There is no MIR band, so MNDWI is calculated correctly 
with the SWIR2 band. 
"""
if do_s2:
    s2_indices = get_sat(sat_name="Sentinel", sat_number=2, 
                              folder=("S2C_MSIL2A_20250301T111031_N0511_R137"
                                  "_T31UCU_20250301T152054.SAFE"), 
                                  do_sat=do_s2)
# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME
print(f"total time taken for all processes: {round(TOTAL_TIME, 2)} seconds")
