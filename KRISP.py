# %% 0. Start
# %%% i. Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import os
import numpy as np
import re # "regular expressions" for parsing filenames
import sys
import math
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

# %%% ii. Import Internal Functions
from data_handling import change_to_folder, extract_chunk_details
from data_handling import sort_prediction_results, sort_file_names
from data_handling import check_positive_int

from image_handling import image_to_array, mask_sentinel, save_image_file
from misc import split_array
from user_interfacing import start_spinner, end_spinner

# %%% iii. Directory Management
try: # personal pc mode
    HOME = os.path.join("C:\\", "Users", "nicol", "Documents", "UoM", "YEAR 3", 
                        "Individual Project", "Downloads")
    os.chdir(HOME)
except: # uni mode
    HOME = os.path.join("C:\\", "Users", "c55626na", "OneDrive - "
                        "The University of Manchester", "Individual Project")
    os.chdir(HOME)

class_names = ["land", "reservoirs", "water bodies"]

# %% Big guy
def run_model(folder, n_chunks, model_name, max_multiplier, plot_examples, 
              start_chunk, n_chunk_preds):
    start_chunk = check_positive_int(
        var=start_chunk, 
        description="chunk to start on")
    n_chunk_preds = check_positive_int(
        var=n_chunk_preds, 
        description="number of chunks to make predictions on")
    
    n_files = n_chunk_preds * 25
    start_file = start_chunk * 25
    # %%% 0. Check for Pre-existing Files
    print("==========")
    print("| STEP 0 |")
    print("==========")
    stop_event, thread = start_spinner(message="checking for "
                                       "pre-existing files")
    start_time = time.monotonic()
    satellite = "Sentinel 2"
    path = os.path.join(HOME, satellite, folder)
    os.chdir(path)
    
    # %%%% 0.1 Chunk Check!
    test_data_path = os.path.join(path, "test data", f"ndwi_{max_multiplier}")
    existing_files = []
    real_n_chunks = math.floor(math.sqrt(n_chunks)) ** 2 - 1
    n_mini_chunks = 25
    mc_per_len = int(np.sqrt(n_mini_chunks)) # mini-chunks per length
    # important note! ensure this matches the IMG_HEIGHT division in trainer
    # as well as the BOX_SIZE division in data_handling
    generate_chunks = False
    
    # %%%%% 0.1.1 Extract current folder contents
    try:
        all_items = os.listdir(test_data_path)
        # Filter for files only, ignore subdirectories
        for item in all_items:
            if os.path.isfile(os.path.join(test_data_path, item)):
                existing_files.append(item)
        existing_files[1] # try to access any item
        # this will induce an intended error in the event that a directory 
        # does exist but it is unpopulated
    except:
        end_spinner(stop_event, thread)
        while True:
            print("test data directory does not exist")
            print("WARNING: creating and saving many images may take " 
                  "very long! The console may freeze of crash, but "
                  "progress should continue regardless")
            print("to check progress, go to the download directory.")
            user_input = input("do you want to recalculate NDWI and fill in "
                               "the remaining chunks? this may add/overwrite "
                               "files (y/n): ").strip().lower()
            if user_input in ["y", "yes"]:
                generate_chunks = True
                print("starting chunk generation process")
                cwd = os.getcwd()
                # create the directory
                change_to_folder(test_data_path)
                os.chdir(cwd)
                break
            elif user_input in ["n", "no"]:
                generate_chunks = False
                print("without chunks, the script cannot run")
                sys.exit(0)
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    # Proceed only if the directory was accessible
    if len(existing_files) > 0:
    
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
            chunks_rem = real_n_chunks - max_chunk_index
        
        # %%%%% 0.1.3 Ask user if they want to continue
        while True:
            if chunks_rem > 0:
                print("WARNING: creating and saving many images may take " 
                      "very long! The console may freeze of crash, but "
                      "progress should continue regardless")
                print("to check progress, go to the download directory.")
                user_input = input("do you want to recalculate NDWI and fill "
                                   f"in the remaining {chunks_rem} chunks? "
                                   "this may add/overwrite files "
                                   "(y/n): ").strip().lower()
                if user_input in ["y", "yes"]:
                    generate_chunks = True
                    break
                elif user_input in ["n", "no"]:
                    generate_chunks = False
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
            else:
                print("disabling chunk generations")
                generate_chunks = False
                break
    
    end_spinner(stop_event, thread)
    time_taken = time.monotonic() - start_time
    print(f"step 0 complete! time taken: {round(time_taken, 2)} seconds")

    # %%% 1. Load Sentinel 2 Image File
    if generate_chunks:
        print("==========")
        print("| STEP 1 |")
        print("==========")
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
            sys.exit()
        
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
        
        end_spinner(stop_event, thread)
        time_taken = time.monotonic() - start_time
        print(f"step 4 complete! time taken: {round(time_taken, 2)} seconds")
        
    # %%% 5. Save Satellite Image Chunks
        """nico!! remember to add a description!"""
        print("==========")
        print("| STEP 5 |")
        print("==========")
        # %%%% 5.1 Create Chunks
        stop_event, thread = start_spinner(message=f"creating {n_chunks} "
                                           "chunks from satellite imagery")
        start_time = time.monotonic()
        
        ndwi_chunks = split_array(array=ndwi, n_chunks=n_chunks)
        global_min = min(np.nanmin(chunk) for chunk in ndwi_chunks)
        global_max = max_multiplier*max(np.nanmax(chunk) for \
                                        chunk in ndwi_chunks)
        
        end_spinner(stop_event, thread)
        
        # %%%% 5.2 Create and Save Mini-Chunks
        print("saving chunks as image files")
        change_to_folder(test_data_path)
        
        total_minichunks_saved = 0 # Optional counter
        
        for i, chunk in enumerate(ndwi_chunks):
            if i > real_n_chunks:
                print("WARNING: Exceeded expected number of chunks "
                      f"({real_n_chunks}). Stopping.")
                break
            if not generate_chunks and i < start_chunk_index:
                continue # Skip chunks already processed
            
            chunk_height, chunk_width = chunk.shape
            mini_chunk_h = chunk_height / mc_per_len
            mini_chunk_w = chunk_width / mc_per_len
            
            uly_s = np.linspace(0, chunk_height - mini_chunk_h, mc_per_len)
            ulx_s = np.linspace(0, chunk_width - mini_chunk_w, mc_per_len)
            
            mc_idx = 0 # mini-chunk index
            for j, ulx in enumerate(ulx_s):
                for k, uly in enumerate(uly_s):
                    image_name = (f"ndwi chunk {i} minichunk {mc_idx}.png")
                    mini_chunk_coord = [
                        float(ulx),                 # ulx
                        float(uly),                 # uly
                        float(ulx + mini_chunk_w),  # lrx
                        float(uly + mini_chunk_h)   # lry
                    ]
                    save_image_file(data=chunk, 
                                    image_name=image_name, 
                                    normalise=True, 
                                    coordinates=mini_chunk_coord, 
                                    g_max=global_max, g_min=global_min, 
                                    dupe_check=False)
                    mc_idx += 1
                    total_minichunks_saved += 1
        
        time_taken = time.monotonic() - start_time
        print(f"step 5 complete! time taken: {round(time_taken, 2)} seconds")
    else:
        print("============")
        print("| STEP 1-5 |")
        print("============")
        print("chunk generation disabled, skipping steps 1-5")
    # %%% 6. Load and Deploy Model
    print("==========")
    print("| STEP 6 |")
    print("==========")
    # %%%% 6.1 Load Data into Memory
    stop_event, thread = start_spinner(message="sorting and loading data")
    start_time = time.monotonic()
    results_list = []
    
    height = int(157/mc_per_len)
    width = int(157/mc_per_len)
    
    model_path = os.path.join(HOME, "saved_models", model_name)
    model = keras.models.load_model(model_path)
    
    all_file_names = os.listdir(test_data_path)
    all_file_names = sort_file_names(all_file_names)
    
    selected_file_names = all_file_names[start_file:(start_file+n_files)]
    del all_file_names # save memory
    
    if len(selected_file_names) > 50:
        plot_examples = False # override decision - that would be too many plots
    end_spinner(stop_event, thread)
    
    # %%%% 6.2 Make Predictions on the Loaded Data
    if not plot_examples:
        stop_event, thread = start_spinner(message="making predictions on "
                                           f"{n_files} files "
                                           f"({n_chunk_preds} chunks)")
    
    for file_name in selected_file_names:
        file_path = os.path.join(test_data_path, file_name)
        
        img = tf.keras.utils.load_img(
            file_path, target_size=(height, width)
        )
        
        if plot_examples:
            plt.figure(figsize=(3, 3))
            plt.imshow(img)
            plt.title(f"{file_name}", fontsize=7)
            plt.axis("off")
            plt.show()
        
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        # Apply softmax to get probabilities because the model outputs logits
        score = tf.nn.softmax(predictions[0])
        
        predicted_class_index = np.argmax(score)
        predicted_class_name = class_names[predicted_class_index].upper()
        confidence = (100 * np.max(score)).astype(np.float32)
        chunk_num, minichunk_num = extract_chunk_details(file_name)
        result = [chunk_num, minichunk_num, predicted_class_name, confidence]
        results_list.append(result)
        
        if plot_examples:
            print("PREDICTED | CONFIDENCE")
            print(f"|{predicted_class_name}| |{confidence}%|")
    
    if not plot_examples:
        end_spinner(stop_event, thread)
    
    time_taken = time.monotonic() - start_time
    print(f"step 6 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 7. Return
    sorted_results_list = sort_prediction_results(results_list)
    return sorted_results_list

# %% Run the big guy
if __name__ == "__main__":
    results = run_model(
        folder=("S2C_MSIL2A_20250301T111031_N0511_R137_"
                "T31UCU_20250301T152054.SAFE"), 
        n_chunks=5000, # number of chunks to split the image into
        model_name="ndwi model epochs-1000.keras", 
        max_multiplier=0.41, # multiply max value of ndwi
        plot_examples=False, 
        start_chunk=0, 
        n_chunk_preds=5
        )
    
    # %% Final
    TOTAL_TIME = time.monotonic() - MAIN_START_TIME
    print(f"total processing time: {round(TOTAL_TIME, 2)} seconds", flush=True)
