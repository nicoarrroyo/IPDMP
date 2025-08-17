""" Navigable Automated Labelling Interface for Regions of Attention (NALIRA)

Description:
NALIRA processes Sentinel 2 imagery to generate labelled data for the 
Keras Reservoir Identification Sequential Platform (KRISP) as part of the 
overarching Individual Project Data to Model Pipeline (IPDMP). It extracts 
water body information from satellite imagery and provides a UI for data 
labelling. The purpose of NALIRA is to create training and test data for KRISP.

Workflow:
1. Data Ingestion:
    - Reads Sentinel 2 image folders and locates necessary image bands.

2. Known Feature Masking:
    - Sea, rivers and streams, large reservoirs, urban areas, areas with 
    large slopes. 

3. Cloud Masking: 
    - OmniCloudMask, using red and green Sentinel 2 bands

4. Compositing:
    - Compute necessary spectral indices
        - Normalized Difference Water Index (NDWI)
        - Normalized Difference Vegetation Index (NDVI)
        - Enhanced Vegetation Index (EVI)
    - A set of Spectral-Temporal Metrics (STMs) computed for all pixels. 
        - These metrics are based on the temporal median, and 25th and 75th 
        percentiles of NDWI, NDVI, and/or EVI. 
        - The NDVI and/or EVI can be used to differentiate vegetation water 
        content from surface water bodies. 
    - Optional data visualization at this stage. 

5. Training Data Polygons:
    - Preliminary data preparation steps, including ensuring file content 
    validity and presence, as well as locating and opening the necessary True 
    Colour Image (TCI) for data labelling. 
    - Provides a Tkinter GUI for manual region of interest (ROI)  labelling via 
    rectangle selection.
    - Uses chunk-based processing; saves the quantity of water reservoirs and 
    water bodies, labelled ROI coordinates, and chunk numbers to a CSV file. 

Outputs:
    - Labelled data in CSV format, with chunk IDs, counts of water bodies, and 
    their coordinates.
    - Python list containing each calculated water index.
"""
# %% Start
# %%% External Library Imports
import time
MAIN_START_TIME = time.monotonic()
import os
import numpy as np
import csv
from PIL import Image
from omnicloudmask import predict_from_array

# %%% Internal Function Imports
from data_handling import rewrite, blank_entry_check, check_file_permission
from data_handling import extract_coords, change_to_folder

from image_handling import image_to_array, plot_indices
from image_handling import plot_chunks, save_image_file

from misc import get_sentinel_bands, split_array, combine_sort_unique
from misc import confirm_continue_or_exit

from user_interfacing import table_print, prompt_roi

# %%% General Directory and Plot Properties
dpi = 3000 # 3000 for full resolution, below 1000, images become fuzzy
n_chunks = 5000 # number of chunks into which images are split
high_res = False # use finer 10m spatial resolution (slower)
cloud_masking = False
show_index_plots = False
save_images = False
label_data = False
data_file_name = "responses_" + str(n_chunks) + "_chunks.csv"

title_size = 8
label_size = 4
plot_size = (5, 5) # larger plots increase detail and pixel count
plot_size_chunks = (6, 6)

HOME = os.path.dirname(os.getcwd()) # HOME path is one level up from the cwd

# for the case where the current directory is considered differently
# HOME = os.getcwd() # for example, in the google cloud virtual machine

# %% General Mega Giga Function
response_time = 0.0

def get_sat(sat_name, sat_number, folder):
    print("====================")
    print(f"||{sat_name} {sat_number} Start||")
    print("====================")
    table_print(n_chunks=n_chunks, high_res=high_res, 
                show_plots=show_index_plots, save_images=save_images, 
                labelling=label_data, cloud_masking=cloud_masking)
    
    # %%% 1. Opening Images and Creating Image Arrays
    print("==========")
    print("| STEP 1 |")
    print("==========")
    print("opening images and creating image arrays")
    start_time = time.monotonic()
    
    # %%%% 1.1 Establishing Paths
    """Most Sentinel 2 files that come packaged in a satellite image folder 
    follow naming conventions that use information contained in the title of 
    the folder. This information can be used to easily navigate through the 
    folder's contents."""
    file_paths = []
    file_paths_clouds = []
    satellite = f"{sat_name} {sat_number}"
    folder_path = os.path.join(HOME, "Downloads", satellite, folder)
    
    # %%%%% 1.1.1 Subfolder naming convention edge case
    """This folder has a strange naming convention that doesn't quite apply to 
    the other folders, so it's difficult to find a rule that would work for 
    any Sentinel 2 image. The easier way of finding this folder is by 
    searching for any available directories in the GRANULE folder, and if 
    there is more than one, then alert the user and exit, otherwise go into 
    that one directory because it will be the one we're looking for."""
    images_path = os.path.join(folder_path, "GRANULE")
    subdirs = [d for d in os.listdir(images_path) 
               if os.path.isdir(os.path.join(images_path, d))]
    if len(subdirs) == 1:
        images_path = os.path.join(images_path, subdirs[0])
    else:
        print("Too many subdirectories in 'GRANULE':", len(subdirs))
        return
    
    # %%%%% 1.1.2 Resolution selection and file name deconstruction
    """Low resolution is beneficial for faster processing times, but is not 
    good for rigorous data generation. High resolution combines some 10m and 
    20m bands for the highest fidelity images, but processing these images is 
    much slower. The file paths pointing to the band image files are also 
    finalised in this section."""
    if high_res:
        res = "10m"
        path_10 = os.path.join(images_path, "IMG_DATA", "R10m")
        path_20 = os.path.join(images_path, "IMG_DATA", "R20m") # for cloud masking
    else:
        res = "60m"
        path_60 = os.path.join(images_path, "IMG_DATA", "R60m")
    
    (sentinel_name, instrument_and_product_level, datatake_start_sensing_time, 
     processing_baseline_number, relative_orbit_number, tile_number_field, 
     product_discriminator_and_format) = folder.split("_")
    prefix = (f"{tile_number_field}_{datatake_start_sensing_time}")
    bands = get_sentinel_bands(sat_number, high_res)
    
    for band in bands:
        if high_res:
            file_paths.append(os.path.join(path_10, 
                                           f"{prefix}_B{band}_10m.jp2"))
            file_paths_clouds.append(os.path.join(path_20, 
                                    f"{prefix}_B{band}_20m.jp2"))
        else:
            file_paths.append(os.path.join(path_60, 
                                               f"{prefix}_B{band}_60m.jp2"))
            file_paths_clouds.append(os.path.join(path_60, 
                                    f"{prefix}_B{band}_60m.jp2"))
    
    # %%%% 1.2 Opening and Converting Images
    """This is the long operation. It is very costly to open the large images, 
    which is why the high-res option exists. When troubleshooting or just 
    trying out the program, it is easier and much faster (about 20x faster) to 
    just use the 60m resolution images. However, when doing any actual data 
    generation, the 60m resolution images are not sufficient."""
    image_arrays = image_to_array(file_paths)
    if cloud_masking:
        image_arrays_clouds = image_to_array(file_paths_clouds)
    
    time_taken = time.monotonic() - start_time
    print(f"step 1 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 2. Masking Clouds
    print("==========")
    print("| STEP 2 |")
    print("==========")
    if cloud_masking:
        print("masking clouds")
        start_time = time.monotonic()

        input_array = np.stack((
            image_arrays_clouds[2], 
            image_arrays_clouds[0], 
            image_arrays_clouds[1]
            ))
        
        # for nvidia case
        # pred_mask = predict_from_array(input_array, mosaic_device="cuda")
        
        # for no nvidia case (inference on cpu)
        pred_mask = predict_from_array(input_array, mosaic_device="cpu")
        
        cloud_thick_positions = np.argwhere(pred_mask[0] == 1)
        cloud_thin_positions = np.argwhere(pred_mask[0] == 2)
        cloud_shadows_positions = np.argwhere(pred_mask[0] == 3)
        cloud_positions = [
            cloud_thick_positions, 
            cloud_thin_positions, 
            cloud_shadows_positions
            ]

        # CHANGE RECOMMENDATION: CALCULATE CLOUD POSITIONS HERE BUT THEN MASK 
        # OUT THE PIXELS AFTER INDEX CALCULATION
        
        for i, image_array in enumerate(image_arrays_clouds):
            image_array[cloud_positions[i][:, 0], 
            cloud_positions[i][:, 1]] = 0 # better than setting it to 0

        time_taken = time.monotonic() - start_time
        print(f"step 2 complete! time taken: {round(time_taken, 2)} seconds")
    else:
        print("skipping cloud masking")
    
    # %%% 3. Calculating Water Indices
    print("==========")
    print("| STEP 3 |")
    print("==========")
    print("populating water index arrays")
    start_time = time.monotonic()
    
    # first convert to int... np.uint16 type is bad for algebraic operations!
    for i, image_array in enumerate(image_arrays):
        image_arrays[i] = image_array.astype(np.float32)
    green, nir, red = image_arrays
    
    np.seterr(divide="ignore", invalid="ignore")
    ndwi = ((green - nir) / (green + nir))
    del green, nir
    
    time_taken = time.monotonic() - start_time
    print(f"step 3 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 4. Showing Indices
    print("==========")
    print("| STEP 4 |")
    print("==========")
    if show_index_plots:
        if save_images:
            print("saving and displaying water index images")
        else:
            print("displaying water index images")
        start_time = time.monotonic()
        plot_indices(ndwi, plot_size, dpi, save_images, res)
        time_taken = time.monotonic() - start_time
        print(f"step 4 complete! time taken: {round(time_taken, 2)} seconds")
    else:
        print("not displaying water index images")
    
    # %%% 5. Data Preparation
    print("==========")
    print("| STEP 5 |")
    print("==========")
    start_time = time.monotonic()
    global response_time
    print("data preparation start")
    
    # %%%% 5.1 Searching for, Opening, and Converting RGB Image
    """nico!! remember to add a description!"""
    print(f"opening {res} resolution true colour image")
    
    tci_path = os.path.join(images_path, "IMG_DATA", f"R{res}")
    tci_file_name = prefix + f"_TCI_{res}.jp2"
    tci_array = image_to_array(os.path.join(tci_path, tci_file_name))
    
    tci_60_path = os.path.join(folder_path, "GRANULE", subdirs[0], 
                               "IMG_DATA", "R60m")
    tci_60_file_name = prefix + "_TCI_60m.jp2"
    with Image.open(os.path.join(tci_60_path, tci_60_file_name)) as img:
        size = (img.width//10, img.height//10)
        tci_60_array = np.array(img.resize(size))
    
    # %%%% 5.2 Creating Chunks from Satellite Imagery
    """nico!! remember to add a description!"""
    print(f"creating {n_chunks} chunks from satellite imagery")
    index_chunks = split_array(array=ndwi, n_chunks=n_chunks)
    tci_chunks = split_array(array=tci_array, n_chunks=n_chunks)
    
    # %%%% 5.3 Preparing File for Labelling
    """nico!! remember to add a description!"""
    break_flag = False
    
    labelling_path = os.path.join(folder_path, "training data")
    
    lines = []
    header = ("chunk,reservoirs,water bodies,reservoir "
    "coordinates,,,,,water body coordinates\n")
    data_file = os.path.join(labelling_path, data_file_name)
    blank_entry_check(file=data_file) # remove all blank entries
    
    # %%%%% 5.3.1 File validity check
    """This section is about checking that the contents of the file are sound. 
    This means checking that, for example, the file exists, or that the entry 
    for chunk 241 is after the entry for chunk 240, or any other problem that 
    may arise. As with every data handling operation, file permission is 
    checked directly before this step, as well as any blank entries. Although 
    this may seem an excessive step to perform every time, it is necessary to 
    ensure that the file data is exactly as it should be. 
    All these checks are carried out in 'read-only' mode unless the user 
    specifies otherwise. This is to make sure that the data is not accidentally 
    overwritten at any point, again, unless the user is sure this is the 
    intended behaviour."""
    print("preparing file for labelling")
    while True:
        # file will always exist due to blank_entry_check call
        with open(data_file, "r") as file:
            lines = file.readlines()
        try:
            # Validate file data
            for i in range(1, len(lines) - 1):
                current_chunk = int(lines[i].split(",")[0])
                next_chunk = int(lines[i+1].split(",")[0])
                if next_chunk - current_chunk != 1:
                    print(f"error in line {i + 2}, "
                          f"expected chunk {current_chunk + 1}")
                    raise ValueError("File validity error")
            last_chunk = int(lines[-1].split(",")[0])
            break
        except (ValueError, IndexError) as e:
            print(f"error - file with invalid data: {e}")
            print("type 'quit' to exit, or 'new' for a fresh file")
            response_time_start = time.monotonic()
            ans = input("press enter to retry: ").strip().lower()
            response_time += time.monotonic() - response_time_start
            if ans.lower() == 'quit':
                print("quitting")
                return ndwi
            elif ans.lower() == 'new':
                print("creating new file...")
                with open(data_file, "w") as file:
                    file.write(header)
                    file.write("0, 1, 0\n") # dummy file to start up
                continue
    
    i = last_chunk + 1 # from this point on, "i" is off-limits as a counter
    
    # %%%%% 5.3.2 Data completion check
    """Once file validity has been verified, this step is for ensuring that 
    the data in the file is complete. While the previous step (5.3.1) was 
    mostly intended for checking that the chunks are in the correct order, 
    this step additionally checks that the chunks that are supposed to have 
    data, i.e. a chunk is noted as containing a reservoir, that there are 
    coordinates outlining that reservoir. If this is not the case, the 
    'data_correction' mode is activated, in which the user is prompted to 
    essentially fill in the coordinates that should exist in the place where 
    a chunk is supposed to contain some water body."""
    # find chunks with invalid or incomplete reservoir coordinate data
    print("")
    reservoir_rows = []
    body_rows = []
    invalid_rows = []
    data_correction = False
    
    with open(data_file, "r") as file:
        lines = file.readlines() # reread lines in case of changes
        globals()["lines"] = lines
    for j in range(1, len(lines)): # starting from the "headers" line
        # check for reservoirs without coordinates
        num_of_reservoirs = int(lines[j].split(",")[1])
        try: # try to access coordinates
            res_coord = lines[j].split(",")[3]
            if res_coord[0] == "[":
                res_has_coords = True
            else:
                res_has_coords = False
        except: # if unable to access, they do not exist
            res_has_coords = False
        if num_of_reservoirs != 0 and not res_has_coords:
            reservoir_rows.append(j-1)
            data_correction = True
        elif num_of_reservoirs == 0 and res_has_coords:
            invalid_rows.append(j-1)
            data_correction = True
        
        # check for non-reservoir water bodies without coordinates
        num_of_bodies = int(lines[j].split(",")[2])
        try: # try to access coordinates
            body_coord = lines[j].split(",")[8]
            if body_coord[0] == "[":
                body_has_coords = True
            else:
                body_has_coords = False
        except: # if unable to access, they do not exist
            body_has_coords = False
        if num_of_bodies != 0 and not body_has_coords:
            body_rows.append(j-1)
            data_correction = True
        elif num_of_bodies == 0 and body_has_coords:
            invalid_rows.append(j-1)
            data_correction = True
    invalid_rows = combine_sort_unique(reservoir_rows, body_rows, invalid_rows)
    
    if data_correction:
        print(f"found {len(invalid_rows)} chunks containing "
               "incomplete, missing, or incorrect coordinate data")
        i = invalid_rows[0]
        invalid_rows_index = 0
    time_taken = time.monotonic() - start_time - response_time
    print(f"step 5 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 6. Data Labelling
    print("==========")
    print("| STEP 6 |")
    print("==========")
    start_time = time.monotonic()
    if not label_data:
        print("not labelling data")
    else:
        print("data labelling start")
        
        # %%%% 6.1 Outputting Images
        print("outputting images...")
        
        while i < len(index_chunks):
            if break_flag:
                break
            plot_chunks(ndwi, index_chunks, plot_size_chunks, i, 
                        title_size, label_size, tci_chunks, tci_60_array)
            max_index = [0, 0]
            max_index[0] = round(np.nanmax(index_chunks[i]), 2)
            print(f"MAX ADJUSTED NDWI: {max_index[0]}", end=" | ")
            max_index[1] = round(np.nanmax(index_chunks[i]), 2)
            print(f"MAX NDWI: {max_index[1]}")
            
            # %%%% 6.2 User Labelling
            blank_entry_check(file=data_file)
            response_time_start = time.monotonic()
            if data_correction:
                print((
                    "this chunk "
                    f"({invalid_rows_index+1}/{len(invalid_rows)})"
                    " should contain "
                    f"{int(lines[i+1].split(',')[1])} reservoirs and "
                    f"{int(lines[i+1].split(',')[2])} non-reservoir "
                    "water bodies"
                    ))
            n_reservoirs = input("how many reservoirs? ").strip().lower()
            n_bodies = ""
            entry_list = []
            while True:
                blank_entry_check(file=data_file)
                back_flag = False
                try:
                    # %%%%% 6.2.1 Regular integer response
                    """nico!! remember to add a description!"""
                    # handle number of reservoirs entry
                    n_reservoirs = int(n_reservoirs)
                    entry_list = [i,n_reservoirs,""]
                    while n_reservoirs > 5: # NOTE add user input type check
                        print("maximum of 5 reservoirs")
                        n_reservoirs = input("how many "
                                             "reservoirs? ").strip().lower()
                    if n_reservoirs != 0:
                        print("please draw a square around the reservoir(s)", 
                              flush=True)
                        chunk_coords = prompt_roi(tci_chunks[i], n_reservoirs)
                        for coord in chunk_coords:
                            entry_list.append(coord)
                    while len(entry_list) < 8:
                        entry_list.append("")
                     
                    # handle number of non-reservoir water bodies entry
                    n_bodies = input("how many ""non-reservoir "
                                     "water bodies? ").strip().lower()
                    n_bodies = int(n_bodies)
                    entry_list[2] = n_bodies
                    if n_bodies != 0:
                        print("please draw a square around the water bodies", 
                              flush=True)
                        chunk_coords = prompt_roi(tci_chunks[i], n_bodies)
                        for coord in chunk_coords:
                            entry_list.append(coord)
                    response_time += time.monotonic() - response_time_start
                    i += 1
                    print("generating next chunk...", flush=True)
                    break # exit loop and continue to next chunk
                
                # handle non-integer responses
                except:
                    n_reservoirs = str(n_reservoirs)
                    n_bodies = str(n_bodies)
                    if "break" in n_bodies or "break" in n_reservoirs:
                        # %%%%% 6.2.2 Non-integer response: "break"
                        """nico!! remember to add a description!"""
                        print("taking a break")
                        response_time += time.monotonic() - response_time_start
                        break_flag = True
                        break
                    elif "back" in n_bodies or "back" in n_reservoirs:
                        # %%%%% 6.2.3 Non-integer response: "back"
                        """nico!! remember to add a description!"""
                        back_flag = True
                        if data_correction:
                            print("cannot use 'back' during data correction")
                            break
                        try:
                            n_backs = int(n_reservoirs.split(" ")[1])
                        except:
                            n_backs = 1
                        i -= n_backs
                        check_file_permission(file_name=data_file)
                        with open(data_file, mode="r") as re: # read
                            rows = list(csv.reader(re))
                        for j in range(n_backs):
                            rows.pop() # remove the last "n_backs" rows
                        with open(data_file, mode="w") as wr: # write
                            rewrite(write_file=wr, rows=rows)
                        break
                    else:
                        # %%%%% 6.2.4 Non-integer response: error
                        """nico!! remember to add a description!"""
                        print("error: non-integer response."
                              "\ntype 'break' to save and quit"
                              "\ntype 'back' to go to previous chunk")
                        n_reservoirs = input("how many "
                                             "reservoirs? ").strip().lower()
            
            # %%%% 6.3 Saving Results
            """nico!! remember to add a description!"""
            if break_flag:
                break
            elif not break_flag and not back_flag:
                check_file_permission(file_name=data_file)
                csv_entry = ""
                first_csv_entry = True
                for entry in entry_list:
                    if first_csv_entry:
                        csv_entry = f"{entry}"
                    elif not first_csv_entry:
                        csv_entry = f"{csv_entry},{entry}"
                    first_csv_entry = False
                if data_correction: # add coordinates to data
                    lines[i] = f"{csv_entry}\n"
                    with open(data_file, mode="w") as wr: # write
                        for j in range(len(lines)):
                            current_entry = lines[j]
                            wr.write(f"{current_entry}")
                    invalid_rows_index += 1
                    if invalid_rows_index >= len(invalid_rows):
                        i = last_chunk + 1
                        data_correction = False
                    else:
                        i = invalid_rows[invalid_rows_index]
                else: # convert entry_list to a string for csv
                    with open(data_file, mode="a") as ap: # append
                        ap.write(f"\n{csv_entry}")
        print(f"responding time: {round(response_time, 2)} seconds")            
        time_taken = time.monotonic() - start_time - response_time
        print(f"step 6 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 7. Data Segmentation
    print("==========")
    print("| STEP 7 |")
    print("==========")
    start_time = time.monotonic()
    
    # %%%% 7.1 Extract Reservoir and Water Body Coordinates
    """nico!! remember to add a description!"""
    if not high_res:
        print("high resolution setting must be activated for data segmentation")
        print("exiting program")
        return ndwi
    
    print("program is about to begin data segmentation")
    confirm_continue_or_exit()
    
    print("extracting coordinates")
    res_rows = []
    res_coords = []
    
    body_rows = []
    body_coords = []
    
    land_rows = []
    land_coords = []
    none_coord = "[50.0 50.0 55.0 55.0]" # replicate res and bod coords format
    # allows it to be passed as an argument of extract_coords
    
    sea_rows = []
    sea_coords = []
    
    with open(data_file, "r") as file:
        lines = file.readlines()
    
    for i in range(1, len(lines)):
        lines[i] = lines[i].split(",")
        if int(lines[i][1]) > 0: # if there is a reservoir
            res_rows.append(lines[i])
            if int(res_rows[-1][1]) > 1:
                for j in range(3, 3+int(res_rows[-1][1])):
                    res_coords.append((i, extract_coords(res_rows[-1][j], 
                                                         create_box_flag=True)))
            elif int(res_rows[-1][1]) == 1:
                res_coords.append((i, extract_coords(res_rows[-1][3], 
                                                     create_box_flag=True)))
        
        # if there is a water body
        if int(lines[i][2]) > 0:
            body_rows.append(lines[i])
            first_coords = extract_coords(body_rows[-1][8], 
                                          create_box_flag=False)
            # and the water body is not the sea
            if first_coords[0] != 0 and first_coords[-1] != 157:
                if int(body_rows[-1][2]) > 1:
                    for j in range(8, 8+int(body_rows[-1][2])):
                        this_coord = extract_coords(body_rows[-1][j], 
                                                    create_box_flag=True)
                        body_coords.append((i, this_coord))
                elif int(body_rows[-1][2]) == 1:
                    this_coord = extract_coords(body_rows[-1][8], 
                                                  create_box_flag=True)
                    body_coords.append((i, this_coord))
            else:# if it IS the sea, save a minichunk of it too
                sea_rows.append(lines[i])
                sea_coords.append((i, extract_coords(none_coord, 
                                                  create_box_flag=True)))
        
        # if it's just land, save a minichunk of it too
        if int(lines[i][1]) == 0 and int(lines[i][2]) == 0:
            land_rows.append(lines[i])
            land_coords.append((i, extract_coords(none_coord, 
                                                  create_box_flag=True)))
    
    globals()["lines"] = lines
    
    # %%%% 7.2 Isolate and Save an Image of Each Reservoir and Water Body
    """nico!! remember to add a description! 0.4*max to bring down the ceiling 
    of ndwi so that reservoir and water bodies are better highlighted"""
    ndwi_chunks = index_chunks[0]
    valid_chunks = [chunk
                    for chunk in ndwi_chunks
                    if not np.all(np.isnan(chunk))]
    if valid_chunks:
        global_min = min(np.nanmin(chunk) for chunk in valid_chunks)
        global_max = 0.4*max(np.nanmax(chunk) for chunk in valid_chunks)
    else:
        global_min = np.nan
        print("Warning: All NDWI chunks contained only NaN values.")
    
    # %%%%% 7.2.1 Create an image of each water reservoir and save it
    print("segmenting reservoir data")
    had_an_oopsie = False
    for i in range(len(res_coords)):
        chunk_n = (int(res_coords[i][0])-1)
        
        # NDWI data
        res_ndwi_path = os.path.join(
            labelling_path, "ndwi", "reservoirs"
            )
        change_to_folder(res_ndwi_path)
        image_name = f"ndwi chunk {chunk_n} reservoir {i+1}.png"
        try:
            save_image_file(data=ndwi_chunks[chunk_n], 
                            image_name=image_name, 
                            normalise=True, 
                            coordinates=res_coords[i][1], 
                            g_min=global_min, g_max=global_max, 
                            dupe_check=True)
            # TCI data
            res_tci_path = os.path.join(
                labelling_path, "tci", "reservoirs"
                )
            change_to_folder(res_tci_path)
            image_name = f"tci chunk {chunk_n} reservoir {i+1}.png"
            save_image_file(data=tci_chunks[chunk_n], 
                            image_name=image_name, 
                            normalise=False, 
                            coordinates=res_coords[i][1], 
                            g_min=global_min, g_max=global_max, 
                            dupe_check=True)
        except:
            had_an_oopsie = True
    
    if had_an_oopsie:
        print("error in reservoir data segmentation")
    
    # %%%%% 7.2.2 Create an image of each water body and save it
    print("segmenting water body data")
    had_an_oopsie = False
    for i in range(len(body_coords)):
        chunk_n = (int(body_coords[i][0])-1)
        
        # NDWI data
        body_ndwi_path = os.path.join(
            labelling_path, "ndwi", "water bodies"
            )
        change_to_folder(body_ndwi_path)
        image_name = f"ndwi chunk {chunk_n} water body {i+1}.png"
        try:
            save_image_file(data=ndwi_chunks[chunk_n], 
                            image_name=image_name, 
                            normalise=True, 
                            coordinates=body_coords[i][1], 
                            g_min=global_min, g_max=global_max, 
                            dupe_check=True)
            # TCI data
            body_tci_path = os.path.join(
                labelling_path, "tci", "water bodies"
                )
            change_to_folder(body_tci_path)
            image_name = f"tci chunk {chunk_n} water body {i+1}.png"
            save_image_file(data=tci_chunks[chunk_n], 
                            image_name=image_name, 
                            normalise=False, 
                            coordinates=body_coords[i][1], 
                            g_min=global_min, g_max=global_max, 
                            dupe_check=True)
        except:
            had_an_oopsie = True
    
    if had_an_oopsie:
        print("error in water body data segmentation")
    
    # %%%% 7.3 Isolate and Save an Image of Mini-Chunks of Land and Sea
    # %%%%# 7.3.1 Land
    print("segmenting land data")
    had_an_oopsie = False
    for i in range(len(land_coords)):
        chunk_n = (int(land_coords[i][0])-1)
        
        # NDWI data
        land_ndwi_path = os.path.join(
            labelling_path, "ndwi", "land"
            )
        change_to_folder(land_ndwi_path)
        image_name = f"ndwi chunk {chunk_n} land {i+1}.png"
        try:
            save_image_file(data=ndwi_chunks[chunk_n], 
                            image_name=image_name, 
                            normalise=True, 
                            coordinates=land_coords[i][1], 
                            g_min=global_min, g_max=global_max, 
                            dupe_check=True)
            # TCI data
            land_tci_path = os.path.join(
                labelling_path, "tci", "land"
                )
            change_to_folder(land_tci_path)
            image_name = f"tci chunk {chunk_n} land {i+1}.png"
            save_image_file(data=tci_chunks[chunk_n], 
                            image_name=image_name, 
                            normalise=False, 
                            coordinates=land_coords[i][1], 
                            g_min=global_min, g_max=global_max, 
                            dupe_check=True)
        except:
            had_an_oopsie = True
    
    if had_an_oopsie:
        print("error in land data segmentation")
    
    # %%%%# 7.3.2 Sea
    print("segmenting sea data")
    had_an_oopsie = False
    for i in range(len(sea_coords)):
        chunk_n = (int(sea_coords[i][0])-1)
        
        # NDWI data
        sea_ndwi_path = os.path.join(
            labelling_path, "ndwi", "sea"
            )
        change_to_folder(sea_ndwi_path)
        image_name = f"ndwi chunk {chunk_n} sea {i+1}.png"
        try:
            save_image_file(data=ndwi_chunks[chunk_n], 
                            image_name=image_name, 
                            normalise=True, 
                            coordinates=sea_coords[i][1], 
                            g_min=global_min, g_max=global_max, 
                            dupe_check=True)
            # TCI data
            sea_tci_path = os.path.join(
                labelling_path, "tci", "sea"
                )
            change_to_folder(sea_tci_path)
            image_name = f"tci chunk {chunk_n} sea {i+1}.png"
            save_image_file(data=tci_chunks[chunk_n], 
                            image_name=image_name, 
                            normalise=False, 
                            coordinates=sea_coords[i][1], 
                            g_min=global_min, g_max=global_max, 
                            dupe_check=True)
        except:
            had_an_oopsie = True
    
    if had_an_oopsie:
        print("error in sea data segmentation")
    
    time_taken = time.monotonic() - start_time
    print(f"step 7 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 8. Satellite Output
    return ndwi
# %% Running Functions
"""
Sentinel 2 has varying resolution bands, with Blue (2), Green (3), Red (4), and 
NIR (8) having 10m spatial resolution, while SWIR 1 (11) and SWIR 2 (12) have 
20m spatial resolution. There is no MIR band, so MNDWI is calculated correctly 
with the SWIR2 band. 
"""
ndwi = get_sat(sat_name="Sentinel", sat_number=2, 
                          folder=("S2C_MSIL2A_20250301T111031_N0511_R137_"
                                  "T31UCU_20250301T152054.SAFE"))

# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME - response_time
print(f"total processing time: {round(TOTAL_TIME, 2)} seconds", flush=True)
