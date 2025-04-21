# %% 0. Start
# %%% 0.1 Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import os
import numpy as np
from tensorflow import keras
import sys
import re # for parsing filenames

# %%% 0.2 Import Internal Functions
from data_handling import change_to_folder
from image_handling import image_to_array, mask_sentinel, save_image_file
from misc import split_array, create_9_random_coords
from user_interfacing import start_spinner, end_spinner

# %%% 0.3 Directory Management
try: # personal pc mode
    HOME = os.path.join("C:\\", "Users", "nicol", "OneDrive - "
                        "The University of Manchester", "Individual Project")
    os.chdir(HOME)
except: # uni mode
    HOME = os.path.join("C:\\", "Users", "c55626na", "OneDrive - "
                        "The University of Manchester", "Individual Project")
    os.chdir(HOME)

# %% Big guy
def run_model(folder, n_chunks, model_name):
    # %%% 1. Load Sentinel 2 Image File
    print("==========")
    print("| STEP 1 |")
    print("==========")
    stop_event, thread = start_spinner(message="opening images and "
                                       "creating image arrays")
    start_time = time.monotonic()
    
    # %%%% 1.1 Establish Paths
    """Most Sentinel 2 files that come packaged in a satellite image folder 
    follow naming conventions that use information contained in the title of 
    the folder. This information can be used to easily navigate through the 
    folder's contents."""
    
    satellite = "Sentinel 2"
    path = os.path.join(HOME, satellite, folder)
    os.chdir(path)
    
    # %%%%% 1.1.1 Subfolder naming convention edge case
    """This folder has a strange naming convention that doesn't quite apply to 
    the other folders, so it's difficult to find a rule that would work for 
    any Sentinel 2 image. The easier way of finding this folder is by 
    searching for any available directories in the GRANULE folder, and if 
    there is more than one, then alert the user and exit, otherwise go into 
    that one directory because it will be the one we're looking for."""
    
    path = os.path.join(path, "GRANULE")
    subdirs = [d for d in os.listdir(path) 
               if os.path.isdir(os.path.join(path, d))]
    if len(subdirs) == 1:
        path = os.path.join(path, subdirs[0])
    else:
        print("Too many subdirectories in 'GRANULE':", len(subdirs))
        exit()
    
    # %%%%% 1.1.2 Resolution selection and file name deconstruction
    """Low resolution is beneficial for faster processing times, but is not 
    good for rigorous data generation. High resolution combines some 10m and 
    20m bands for the highest fidelity images, but processing these images is 
    much slower. The file paths pointing to the band image files are also 
    finalised in this section."""
    
    path_10 = os.path.join(path, "IMG_DATA", "R10m") # green and nir
    
    (sentinel_name, instrument_and_product_level, datatake_start_sensing_time, 
     processing_baseline_number, relative_orbit_number, tile_number_field, 
     product_discriminator_and_format) = folder.split("_")
    prefix = (f"{tile_number_field}_{datatake_start_sensing_time}")
    bands = ["03", "08"] # green and nir
    
    file_paths = []
    for band in bands:
        file_paths.append(os.path.join(path_10, f"{prefix}_B{band}_10m.jp2"))
    
    # %%%% 1.2 Open and Convert Band Images
    """This is the long operation. It is very costly to open the large images, 
    which is why the high-res option exists. When troubleshooting or just 
    trying out the program, it is easier and much faster (about 20x faster) to 
    just use the 60m resolution images. However, when doing any actual data 
    generation, the 60m resolution images are not sufficient."""
    
    image_arrays = image_to_array(file_paths)
    
    time_taken = time.monotonic() - start_time
    end_spinner(stop_event, thread)
    print(f"step 1 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 2. Mask Clouds From the Image
    print("==========")
    print("| STEP 2 |")
    print("==========")
    stop_event, thread = start_spinner(message="masking clouds")
    start_time = time.monotonic()
    
    path = os.path.join(path, "QI_DATA")
    image_arrays = mask_sentinel(path=path, 
                                 high_res=True, 
                                 image_arrays=image_arrays)
    
    time_taken = time.monotonic() - start_time
    end_spinner(stop_event, thread)
    print(f"step 2 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 3. Calculate NDWI
    print("==========")
    print("| STEP 3 |")
    print("==========")
    stop_event, thread = start_spinner(message="populating ndwi array")
    start_time = time.monotonic()
    
    # first convert to int... np.uint16 type is bad for algebraic operations!
    for i, image_array in enumerate(image_arrays):
        image_arrays[i] = image_array.astype(int)
    green, nir = image_arrays
    
    np.seterr(divide="ignore", invalid="ignore")
    ndwi = ((green - nir) / (green + nir))
    
    end_spinner(stop_event, thread)
    time_taken = time.monotonic() - start_time
    print(f"step 3 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 4. Open and Convert True Colour Image
    """nico!! remember to add a description!"""
    print("==========")
    print("| STEP 4 |")
    print("==========")
    stop_event, thread = start_spinner(message="opening 10m "
                                       "resolution true colour image")
    start_time = time.monotonic()
    
    path = os.path.join(HOME, satellite, folder)
    
    # tci_path = os.path.join(path, "GRANULE", subdirs[0], "IMG_DATA", "R10m")
    # tci_file_name = prefix + "_TCI_10m.jp2"
    # tci_array = image_to_array(os.path.join(tci_path, tci_file_name))
    end_spinner(stop_event, thread)
    time_taken = time.monotonic() - start_time
    print(f"step 4 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 5. Save Satellite Image Chunks
    """nico!! remember to add a description!"""
    print("==========")
    print("| STEP 5 |")
    print("==========")
    stop_event, thread = start_spinner(message=f"creating {n_chunks} chunks"
                                       " from satellite imagery")
    start_time = time.monotonic()
    
    # %%%% 5.1 Create Chunks
    ndwi_chunks = split_array(array=ndwi, n_chunks=n_chunks)
    # tci_chunks = split_array(array=tci_array, n_chunks=n_chunks)
    chunk_size = ndwi_chunks[0].shape
    
    max_multiplier = 0.4
    global_min = min(np.nanmin(chunk) for chunk in ndwi_chunks)
    global_max = max_multiplier*max(np.nanmax(chunk) for chunk in ndwi_chunks)
    
    end_spinner(stop_event, thread)
    
    # %%%% 5.2 Chunk Check!
    test_data_path = os.path.join(path, "test data", f"ndwi_{max_multiplier}")
    change_to_folder(test_data_path)
    
    # %%%%% 5.2.1 Check current folder contents
    existing_files = []
    try:
        all_items = os.listdir(test_data_path)
        # Filter for files only, ignore subdirectories
        for item in all_items:
            if os.path.isfile(os.path.join(test_data_path, item)):
                existing_files.append(item)
    except:
        pass
    
    # Proceed only if the directory was accessible
    if existing_files:
        print(f"Target directory contains {len(existing_files)} file(s).")
    
        # %%%%% 5.2.2 Check latest saved chunk
        max_chunk_index = -1
        start_chunk_index = 0
        
        filename_pattern = re.compile(r"ndwi chunk (\d+) minichunk \d+\.png")
        
        for filename in existing_files:
            match = filename_pattern.match(filename)
            if match:
                # Extract the captured chunk index (group 1) and convert to int
                chunk_index = int(match.group(1))
                max_chunk_index = max(max_chunk_index, chunk_index)
        
        if max_chunk_index != -1:
            print(f"Highest chunk index found in filenames: {max_chunk_index}")
            start_chunk_index = max_chunk_index + 1
        else:
            print("No files matching the expected 'ndwi chunk i "
                  "minichunk j.png' pattern found.")
        
        # %%%%% 5.2.3 Ask the user if they want to continue
        if start_chunk_index / n_chunks < 0.5:
            while True:
                user_input = input("Do you want to continue processing? "
                                   "this may add/overwrite "
                                   "files (y/n): ").strip().lower()
                if user_input in ["y", "yes"]:
                    print("WARNING: this step may take VERY VERY LONG!")
                    print("if you want to check progress, go to the download "
                          "folder.")
                    print("the console may freeze or crash, but progress "
                          "should continue.")
                    break
                elif user_input in ["n", "no"]:
                    print("Exiting script")
                    sys.exit(0) # Exit the script cleanly
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
    
    # %%%% 5.3 Create and Save Mini-chunks
    print("saving chunks as image files")
    for i, chunk in enumerate(ndwi_chunks):
        if existing_files and i < start_chunk_index:
            continue # Skip chunks already processed
        mini_chunk_coords = create_9_random_coords(ulx=0, uly=0, 
                                                   lrx=chunk_size[0],
                                                   lry=chunk_size[1])
        for j, coords in enumerate(mini_chunk_coords):
            image_name = (f"ndwi chunk {i} minichunk {j}.png")
            save_image_file(data=chunk, image_name=image_name, normalise=True, 
                            coordinates=coords, margin=0, 
                            g_max=global_max, g_min=global_min, 
                            dupe_check=False)
    
    time_taken = time.monotonic() - start_time
    print(f"step 5 complete! time taken: {round(time_taken, 2)} seconds")
    # %%% 6. Load Model
    print("==========")
    print("| STEP 6 |")
    print("==========")
    # stop_event, thread = start_spinner(message="loading keras model")
    model_path = os.path.join(HOME, "IPMLS", "saved_models", model_name)
    model = keras.models.load_model(model_path)

# %% Run the big guy
run_model(folder=("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_"
                  "20250301T152054.SAFE"), 
          n_chunks=5000, 
          model_name="ndwi model epochs-200.keras")

# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME
print(f"total processing time: {round(TOTAL_TIME, 2)} seconds", flush=True)
