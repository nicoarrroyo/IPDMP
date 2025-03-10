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
import os

# %%% Internal Function Imports
from image_functions import compress_image, plot_indices, mask_sentinel
from calculation_functions import get_indices
from satellite_functions import get_sentinel_bands
from misc_functions import table_print

# %%% General Image and Plot Properties
compression = 1 # 1 for full-sized images, bigger integer for smaller images
dpi = 3000 # 3000 for full resolution, below 1000, images become fuzzy
plot_size = (3, 3) # larger plots increase detail and pixel count
save_images = False
high_res = False # use finer 10m spatial resolution (slower)
# main parent path where all image files are stored
uni_mode = False
if uni_mode:
    HOME = "C:\\Users\\c55626na\\OneDrive - The University of Manchester\\Individual Project"
else:
    HOME = "C:\\Users\\nicol\\Documents\\UoM\\YEAR 3\\Individual Project\\Downloads"

# %% General Mega Giga Function
do_s2 = True

def get_sat(sat_name, sat_number, folder):    
    print("====================")
    print(f"||{sat_name} {sat_number} Start||")
    print("====================")
    table_print(compression=compression, DPI=dpi, plot_size=plot_size, 
                save_images=save_images, high_res=high_res, uni_mode=uni_mode)
    
    # %%% 1. Establishing Paths, Opening and Resizing Images, and Creating Image Arrays
    print("establishing paths, opening and resizing images, creating image arrays", 
          end="... ")
    start_time = time.monotonic()
    
    file_paths = []
    satellite = f"\\{sat_name} {sat_number}\\"
    path = HOME + satellite + folder + "\\GRANULE"
    
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if len(subdirs) == 1:
        path = (f"{path}\\{subdirs[0]}\\")
        os.chdir(path)
    else:
        print("Too many subdirectories in 'GRANULE':", len(subdirs))
        return
    
    if high_res:
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
    bands = get_sentinel_bands(sat_number, high_res)
    
    for band in bands:
        if high_res:
            if band == "02" or band == "03" or band == "08":
                file_paths.append(path_10 + prefix + band + "_10m.jp2")
            else:
                file_paths.append(path_20 + prefix + band + "_20m.jp2")
        else:
            file_paths.append(path_60 + prefix + band + "_60m.jp2")
    
    image_arrays, size = compress_image(compression, file_paths)
    
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 2. Masking Clouds
    print("masking clouds", end="... ")
    start_time = time.monotonic()
    
    path = HOME + satellite + folder + "\\GRANULE\\" + subdirs[0] + "\\QI_DATA\\"
    image_arrays = mask_sentinel(path, high_res, image_arrays, compression)
    
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 3. Calculating Water Indices
    print("populating water index arrays", end="... ")
    start_time = time.monotonic()
    
    blue, green, nir, swir1, swir2 = image_arrays
    indices = get_indices(blue, green, nir, swir1, swir2)
    
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 4. Showing Indices
    minimum = -1
    maximum = 1
    if save_images:
        print("displaying and saving water index images...")
    else:
        print("displaying water index images...")
    start_time = time.monotonic()
    plot_indices(indices, sat_number, plot_size, minimum, maximum, 
               compression, dpi, save_images, res)
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 5. Slicing Images
    print("a")
    import numpy as np
    
    def split_array(array, n_chunks):
        rows = np.array_split(array, np.sqrt(n_chunks), axis=0) # split into rows
        split_arrays = [np.array_split(row_chunk, np.sqrt(n_chunks), 
                                       axis=1) for row_chunk in rows]
        chunks = [subarray for row_chunk in split_arrays for subarray in row_chunk]
        return chunks
    
    # split indices into chunks
    chunks = []
    for index in indices:
        chunks.append(split_array(array=index, n_chunks=500))
    
    # look for rgb image everywhere
    def find_rgb_file(path):
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path): # if item is a folder
                found_rgb, rgb_path = find_rgb_file(full_path)
                if found_rgb:  
                    return True, rgb_path
            else: # if item is a file
                if "RGB" in item and "10m" in item: # check for "RGB" in filename
                    return True, full_path
        return False, None
    
    path = HOME + satellite + folder
    found_rgb, full_path = find_rgb_file(path)
    
    # if rgb image found
    if found_rgb:
        print("found 10 m resolution RGB image", full_path)
    else: # elif rgb image not found
        print("did not find 10 m resolution RGB image")
    
    
    import matplotlib.pyplot as plt
    plt.imshow(chunks[0][0], interpolation="nearest", 
               cmap="viridis", vmin=minimum, vmax=maximum)
    plt.title("chunk0")
    plt.show()
    
    
    path = HOME + satellite + folder + "\\GRANULE\\" + subdirs[0] + "\\"
    path_10 = (path + "IMG_DATA\\R10m\\") # finer resolution for bands 2, 3, 8
    path_20 = (path + "IMG_DATA\\R20m\\") # regular resolution for bands 11, 12
    path_60 = (path + "IMG_DATA\\R60m\\")
    rgb = False
    if rgb:
        blue_path = path_60 + "T31UCU_20250301T111031_B02_60m.jp2"
        green_path = path_60 + "T31UCU_20250301T111031_B03_60m.jp2"
        red_path = path_60 + "T31UCU_20250301T111031_B04_60m.jp2"
        from image_functions import get_rgb
        get_rgb(blue_path, green_path, red_path)
    # %%% XX. Satellite Output
    return indices
# %% Running Functions
"""
Sentinel 2 has varying resolution bands, with Blue (2), Green (3), Red (4), and 
NIR (8) having 10m spatial resolution, while SWIR 1 (11) and SWIR 2 (12) have 
20m spatial resolution. There is no MIR band, so MNDWI is calculated correctly 
with the SWIR2 band. 
"""
if do_s2:
    s2_indices = get_sat(sat_name="Sentinel", sat_number=2, 
                              folder=("S2C_MSIL2A_20250301T111031_N0511_R137"
                                      "_T31UCU_20250301T152054.SAFE"))
# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME
print(f"total time taken for all processes: {round(TOTAL_TIME, 2)} seconds")
