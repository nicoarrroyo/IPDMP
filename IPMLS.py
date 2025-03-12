""" Individual Project Machine Learning Software (IPMLS-25-03) """
""" Update Notes (from previous version IPMLS-25-02)
- earth engine
    - gee removed entirely, switch to local files
- optimisations
    - low resolution option for sentinel 2
- output plots
    - minimums and interpolation removed
- sentinel 2
    - functional index calculation, plot outputs, and plot saving
    - new general function for landsat and/or sentinel
- machine learning
    - new section to allow user to manually label chunks of an image
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
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# %%% Internal Function Imports
from image_functions import compress_image, plot_indices, mask_sentinel
from image_functions import get_rgb, find_rgb_file
from calculation_functions import get_indices
from satellite_functions import get_sentinel_bands
from misc_functions import table_print, split_array

# %%% General Image and Plot Properties
compression = 1 # 1 for full-sized images, bigger integer for smaller images
dpi = 3000 # 3000 for full resolution, below 1000, images become fuzzy
n_chunks = 5000 # number of chunks into which images are split
save_images = False
high_res = False # use finer 10m spatial resolution (slower)
label_data = True
uni_mode = True
if uni_mode:
    plot_size = (5, 5) # larger plots increase detail and pixel count
    plot_size_chunk = (8, 6)
    HOME = "C:\\Users\\c55626na\\OneDrive - The University of Manchester\\Individual Project"
else:
    plot_size = (3, 3) # larger plots increase detail and pixel count
    plot_size_chunk = (5, 3)
    HOME = "C:\\Users\\nicol\\Documents\\UoM\\YEAR 3\\Individual Project\\Downloads"

responses = np.zeros((2, n_chunks))
responses[0] = np.arange(n_chunks)
response_time = 0
# %% General Mega Giga Function
do_s2 = True

def get_sat(sat_name, sat_number, folder):
    print("====================")
    print(f"||{sat_name} {sat_number} Start||")
    print("====================")
    table_print(compression=compression, DPI=dpi, plot_size=plot_size, n_chunks=n_chunks, 
                save_images=save_images, high_res=high_res, labelling=label_data, 
                uni_mode=uni_mode)
    
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
    prefix = (f"{tile_number_field}_{datatake_start_sensing_time}")
    bands = get_sentinel_bands(sat_number, high_res)
    
    for band in bands:
        if high_res:
            if band == "02" or band == "03" or band == "08":
                file_paths.append(path_10 + prefix + "_B" + band + "_10m.jp2")
            else:
                file_paths.append(path_20 + prefix + "_B" + band + "_20m.jp2")
        else:
            file_paths.append(path_60 + prefix + "_B" + band + "_60m.jp2")

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
    globals()["im_arrs"] = image_arrays
    indices = get_indices(blue, green, nir, swir1, swir2)
    
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 4. Showing Indices
    if save_images:
        print("saving and displaying water index images...")
    else:
        print("displaying water index images...")
    start_time = time.monotonic()
    plot_indices(indices, sat_number, plot_size, compression, 
                 dpi, save_images, res)
    time_taken = time.monotonic() - start_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 5. Data Labelling
    if label_data:
        start_time = time.monotonic()
        
        # %%%% 5.1 Searching for, Opening, and Converting RGB Image
        path = HOME + satellite + folder
        found_rgb, full_path = find_rgb_file(path)
        if found_rgb:
            print("opening 10m resolution RGB image", end="... ")
            with Image.open(full_path) as rgb_image:
                rgb_array = np.array(rgb_image)
        else:
            print("generating and saving a new 10m resolution RGB image", end="... ")
            
            path = HOME + satellite + folder + "\\GRANULE\\" + subdirs[0] + "\\"
            os.chdir(path)
            path_10 = (path + "IMG_DATA\\R10m\\")
            
            blue_path = path_10 + prefix + "02_10m.jp2"
            green_path = path_10 + prefix + "03_10m.jp2"
            red_path = path_10 + prefix + "04_10m.jp2"
            
            rgb_array = get_rgb(blue_path, green_path, red_path, 
                                save_image=True, res=10, show_image=False)
        
        tci_file_name = prefix + "_TCI_10m.jp2"
        tci_path = f"{path}\\GRANULE\\{subdirs[0]}\\IMG_DATA\\R10m\\"
        with Image.open(tci_path + tci_file_name) as tci_image:
            tci_array = np.array(tci_image)
        print("complete!")
        # %%%% 5.2 Creating Chunks from Satellite Imagery
        print("creating", n_chunks, "chunks from satellite imagery", end="... ")
        index_chunks = []
        for index in indices:
            index_chunks.append(split_array(array=index, n_chunks=n_chunks))
        rgb_chunks = split_array(array=rgb_array, n_chunks=n_chunks)
        tci_chunks = split_array(array=tci_array, n_chunks=n_chunks)
        print("complete!")
        
        # %%%% 5.3 Outputting Images for Labelling
        print("outputting images for labelling", end="... ")
        index_labels = ["NDWI", "MNDWI", "AWEI-SH", "AWEI-NSH"]
        break_flag = False
        
        for i in range(len(index_chunks[0])):
            if break_flag:
                break
            fig, axes = plt.subplots(1, len(indices), figsize=plot_size_chunk)
            for count, index_label in enumerate(index_labels):
                axes[count].imshow(index_chunks[count][i])
                axes[count].set_title(f"{index_label} Chunk {i}", fontsize=6)
                axes[count].axis("off")
            plt.tight_layout()
            plt.show()
            
            fig, axes = plt.subplots(1, 2, figsize=plot_size)
            axes[0].imshow(rgb_chunks[i])
            axes[0].set_title(f"RGB Chunk {i}", fontsize=6)
            axes[0].axis("off")
            axes[1].imshow(tci_chunks[i])
            axes[1].set_title(f"TCI Chunk {i}", fontsize=6)
            axes[1].axis("off")
            plt.tight_layout()
            plt.show()
            
            # %%%% 5.4 User Labelling
            global response_time
            response_time_start = time.monotonic()
            n_reservoirs = input("how many reservoirs? ")
            while True:
                try:
                    n_reservoirs = int(n_reservoirs)
                    responses[1][i] = n_reservoirs
                    print("generating next chunk...")
                    response_time += time.monotonic() - response_time_start
                    break
                except ValueError:
                    if "break" in n_reservoirs:
                        print("taking a break")
                        response_time += time.monotonic() - response_time_start
                        break_flag = True
                        break
                    print("error: non-integer response. type 'break' to save and quit")
                    n_reservoirs = input("how many reservoirs? ")
    else:
        print("not labelling data")
        return indices
    
    time_taken = time.monotonic() - start_time - response_time
    print(f"complete! time taken: {round(time_taken, 2)} seconds")
    print(f"responding time: {round(response_time, 2)} seconds")
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
TOTAL_TIME = time.monotonic() - MAIN_START_TIME - response_time
print(f"total time taken for all processes: {round(TOTAL_TIME, 2)} seconds")
