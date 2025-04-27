# %% 0. Start
# %%% i. Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import os
import numpy as np
import sys
import re # for parsing filenames
import math
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

# %%% ii. Import Internal Functions
from image_handling import image_to_array, mask_sentinel, save_image_file
from misc import split_array, create_9_random_coords
from user_interfacing import start_spinner, end_spinner

# %%% iii. Directory Management
try: # personal pc mode
    HOME = os.path.join("C:\\", "Users", "nicol", "OneDrive - "
                        "The University of Manchester", "Individual Project")
    os.chdir(HOME)
except: # uni mode
    HOME = os.path.join("C:\\", "Users", "c55626na", "OneDrive - "
                        "The University of Manchester", "Individual Project")
    os.chdir(HOME)

class_names = ["reservoirs", "water bodies"]

# %% Big guy
def run_model(folder, n_chunks, model_name, max_multiplier):
    # %%% 0. Check for Pre-existing Files
    print("==========")
    print("| STEP 0 |")
    print("==========")
    stop_event, thread = start_spinner(message="checking for "
                                       "pre-existing files")
    satellite = "Sentinel 2"
    path = os.path.join(HOME, satellite, folder)
    os.chdir(path)
    
    # %%%% 0.1 Chunk Check!
    test_data_path = os.path.join(path, "test data", f"ndwi_{max_multiplier}")
    existing_files = []
    real_n_chunks = math.floor(math.sqrt(n_chunks)) ** 2 - 1
    generate_chunks = False
    
    # %%%%% 0.1.1 Extract current folder contents
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
    
        # %%%%% 0.1.2 Find latest saved chunk
        max_chunk_index = -1
        start_chunk_index = 0
        
        filename_pattern = re.compile(r"ndwi chunk (\d+) minichunk \d+\.png")
        
        for filename in existing_files:
            match = filename_pattern.match(filename)
            if match:
                # Extract the captured chunk index (group 1) and convert to int
                chunk_index = int(match.group(1))
                max_chunk_index = max(max_chunk_index, chunk_index)
        end_spinner(stop_event, thread)
        print(f"Target directory contains {len(existing_files)} file(s).")
        if max_chunk_index != -1:
            print(f"{max_chunk_index} chunks saved in 'ndwi_{max_multiplier}'.")
            start_chunk_index = max_chunk_index + 1
            chunks_rem = real_n_chunks - max_chunk_index
            percent_rem = round(100 * (chunks_rem / real_n_chunks), 2)
            print(f"{chunks_rem} chunks remaining")
            print(f"{percent_rem}% remaining")
        else:
            print("No files matching the 'ndwi chunk i minichunk j.png' "
                  f"pattern found in 'ndwi_{max_multiplier}'.")
        
        # %%%%% 0.1.3 Ask user if they want to continue
        while True:
            if chunks_rem > 0:
                user_input = input("do you want to recalculate NDWI and fill "
                                   f"in the remaining {chunks_rem} chunks? "
                                   "this may add/overwrite files "
                                   "(y/n): ").strip().lower()
                if user_input in ["y", "yes"]:
                    print("WARNING: creating and saving many images may take " 
                          "very long! The console may freeze of crash, but "
                          "progress should continue regardless")
                    print("to check progress, go to the download directory.")
                    generate_chunks = True
                    break
                elif user_input in ["n", "no"]:
                    generate_chunks = False
                    print("Exiting script")
                    sys.exit(0) # Exit the script cleanly
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
            else:
                generate_chunks = False
                break
    
    end_spinner(stop_event, thread)

    # %%% 1. Load Sentinel 2 Image File
    print("==========")
    print("| STEP 1 |")
    print("==========")
    if generate_chunks:
        stop_event, thread = start_spinner(message="opening images and "
                                           "creating image arrays")
        start_time = time.monotonic()
        
        # %%%% 1.1 Establish Paths
        """Most Sentinel 2 files that come packaged in a satellite image folder 
        follow naming conventions that use information contained in the title 
        of the folder. This information can be used to easily navigate through 
        the folder's contents."""
        
        path = os.path.join(HOME, satellite, folder)
        os.chdir(path)
        
        # %%%%% 1.1.1 Subfolder naming convention edge case
        """This folder has a strange naming convention that doesn't quite 
        apply to the other folders, so it's difficult to find a rule that would 
        work for any Sentinel 2 image. The easier way of finding this folder is 
        by searching for any available directories in the GRANULE folder, and 
        if there is more than one, then alert the user and exit, otherwise go 
        into that one directory because it will be the one we're looking for."""
        
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
        good for rigorous data generation. High resolution combines some 10m 
        and 20m bands for the highest fidelity images, but processing these 
        images is much slower. The file paths pointing to the band image files 
        are also finalised in this section."""
        
        path_10 = os.path.join(path, "IMG_DATA", "R10m") # green and nir
        
        (sentinel_name, instrument_and_product_level, 
         datatake_start_sensing_time, processing_baseline_number, 
         relative_orbit_number, tile_number_field, 
         product_discriminator_and_format) = folder.split("_")
        prefix = (f"{tile_number_field}_{datatake_start_sensing_time}")
        bands = ["03", "08"] # green and nir
        
        file_paths = []
        for band in bands:
            file_paths.append(os.path.join(path_10, 
                                           f"{prefix}_B{band}_10m.jp2"))
        
        # %%%% 1.2 Open and Convert Band Images
        """This is the long operation. It is very costly to open the large 
        images, which is why the high-res option exists. When troubleshooting 
        or just trying out the program, it is easier and much faster 
        (about 20x faster) to just use the 60m resolution images. However, 
        when doing any actual data generation, the 60m resolution images are 
        not sufficient."""
        
        image_arrays = image_to_array(file_paths)
        
        time_taken = time.monotonic() - start_time
        end_spinner(stop_event, thread)
        print(f"step 1 complete! time taken: {round(time_taken, 2)} seconds")
    else:
        print("chunk generation disabled, skipping this step")
    
    # %%% 2. Mask Clouds From the Image
    print("==========")
    print("| STEP 2 |")
    print("==========")
    if generate_chunks:
        stop_event, thread = start_spinner(message="masking clouds")
        start_time = time.monotonic()
        
        path = os.path.join(path, "QI_DATA")
        image_arrays = mask_sentinel(path=path, 
                                     high_res=True, 
                                     image_arrays=image_arrays)
        
        time_taken = time.monotonic() - start_time
        end_spinner(stop_event, thread)
        print(f"step 2 complete! time taken: {round(time_taken, 2)} seconds")
    else:
        print("chunk generation disabled, skipping this step")
    
    # %%% 3. Calculate NDWI
    print("==========")
    print("| STEP 3 |")
    print("==========")
    if generate_chunks:
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
    else:
        print("chunk generation disabled, skipping this step")
    
    # %%% 4. Open and Convert True Colour Image
    """nico!! remember to add a description!"""
    print("==========")
    print("| STEP 4 |")
    print("==========")
    if generate_chunks:
        stop_event, thread = start_spinner(message="opening 10m "
                                           "resolution true colour image")
        start_time = time.monotonic()
        
        path = os.path.join(HOME, satellite, folder)
        
        end_spinner(stop_event, thread)
        time_taken = time.monotonic() - start_time
        print(f"step 4 complete! time taken: {round(time_taken, 2)} seconds")
    else:
        print("chunk generation disabled, skipping this step")
    
    # %%% 5. Save Satellite Image Chunks
    """nico!! remember to add a description!"""
    print("==========")
    print("| STEP 5 |")
    print("==========")
    if generate_chunks:
        # %%%% 5.1 Create Chunks
        stop_event, thread = start_spinner(message=f"creating {n_chunks} "
                                           "chunks from satellite imagery")
        start_time = time.monotonic()
        
        ndwi_chunks = split_array(array=ndwi, n_chunks=n_chunks)
        chunk_size = ndwi_chunks[0].shape
        globals()["chunk_size"] = chunk_size
        global_min = min(np.nanmin(chunk) for chunk in ndwi_chunks)
        global_max = max_multiplier*max(np.nanmax(chunk) for \
                                        chunk in ndwi_chunks)
        
        end_spinner(stop_event, thread)
        
        # %%%% 5.2 Create and Save Mini-chunks
        print("saving chunks as image files")
        for i, chunk in enumerate(ndwi_chunks):
            if existing_files and i < start_chunk_index:
                continue # Skip chunks already processed
            mini_chunk_coords = create_9_random_coords(ulx=0, uly=0, 
                                                       lrx=chunk_size[0],
                                                       lry=chunk_size[1])
            for j, coords in enumerate(mini_chunk_coords):
                image_name = (f"ndwi chunk {i} minichunk {j}.png")
                save_image_file(data=chunk, image_name=image_name, 
                                normalise=True, coordinates=coords, margin=0, 
                                g_max=global_max, g_min=global_min, 
                                dupe_check=False)
        
        time_taken = time.monotonic() - start_time
        print(f"step 5 complete! time taken: {round(time_taken, 2)} seconds")
    else:
        print("chunk generation disabled, skipping this step")
    # %%% 6. Load Model
    print("==========")
    print("| STEP 6 |")
    print("==========")
    model_path = os.path.join(HOME, "IPMLS", "saved_models", model_name)
    model = keras.models.load_model(model_path)
    height = 157
    width = 157
    all_file_names = os.listdir(test_data_path)[:20] # first 20 images
    for file_name in all_file_names:
        file_path = os.path.join(test_data_path, file_name)
        
        img = tf.keras.utils.load_img(
            file_path, target_size=(height, width)
        )
        
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.title(f"{file_name}", fontsize=7)
        plt.axis("off")
        plt.show() 
        
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        # Make prediction
        predictions = model.predict(img_array)
        # Apply softmax to get probabilities because the model outputs logits
        score = tf.nn.softmax(predictions[0])
        
        predicted_class_index = np.argmax(score)
        predicted_class_name = class_names[predicted_class_index]
        confidence = 100 * np.max(score)
        
        print(
            "Prediction: This image most likely belongs to "
            f"'{predicted_class_name}' "
            f"with a {confidence:.2f}% confidence."
        )
        
# %% Run the big guy
# =============================================================================
# run_model(folder=("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_"
#                   "20250301T152054.SAFE"), 
#           n_chunks=5000, 
#           model_name="ndwi model epochs-200.keras", 
#           max_multiplier=0.4)
# =============================================================================
run_model(folder=("S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_"
                  "20250301T152054.SAFE"), 
          n_chunks=5000, 
          model_name="ndwi model epochs-30_20250421_231511.keras", 
          max_multiplier=0.4)

# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME
print(f"total processing time: {round(TOTAL_TIME, 2)} seconds", flush=True)
