""" Individual Project Data Generation Software (IPDGS)

Description:

IPDGS processes Sentinel-2 imagery to generate labelled data for the 
Individual Project Random Forest Model (IPRFM) as part of the overarching 
Individual Project Machine Learning Software (IPMLS). It extracts water body 
information from satellite imagery and provides a UI for data labelling. The 
purpose of IPDGS is to create training and test data for IPRFM.

Workflow:

1. Data Ingestion:
  - Reads Sentinel-2 image folders and locates necessary image bands.

2. Preprocessing:
  - Upscales lower-resolution bands if needed.
  - Applies cloud masking using Sentinel-2 cloud probability data.

3. Index Computation:
  - Calculates water indices (NDWI, MNDWI, AWEI-SH, AWEI-NSH).

4. Visualization (Optional):
  - Displays calculated water index images.
  - Offers image saving.

5. Labelling:
  - Provides a Tkinter GUI for manual region of interest (ROI) (water body) 
  labelling via rectangle selection.
  - Uses chunk-based processing; saves the quantity of water reservoirs and 
  water bodies, labelled ROI coordinates, and chunk numbers to a CSV file.

Output:

- Labelled data in CSV format, with chunk IDs, counts of water bodies, and 
their coordinates.
"""
# %% Start
# %%% External Library Imports
import time
MAIN_START_TIME = time.monotonic()
import os
import numpy as np
import csv
from PIL import Image

# %%% Internal Function Imports
from data_handling import rewrite, blank_entry_check, check_file_permission
from data_handling import extract_coords

from image_handling import image_to_array, mask_sentinel, plot_indices
from image_handling import plot_chunks

from misc import get_sentinel_bands, split_array, combine_sort_unique

from user_interfacing import table_print, start_spinner, end_spinner, prompt_roi

# %%% General Image and Plot Properties
dpi = 3000 # 3000 for full resolution, below 1000, images become fuzzy
n_chunks = 5000 # number of chunks into which images are split
data_file = "responses_" + str(n_chunks) + "_chunks.csv"
high_res = False # use finer 10m spatial resolution (slower)
show_index_plots = False
save_images = False
label_data = False

try: # personal pc mode
    title_size = 8
    label_size = 4
    HOME = ("C:\\Users\\nicol\\OneDrive - " # personal computer user name
            "The University of Manchester\\Individual Project\\Downloads")
    os.chdir(HOME)
    plot_size = (3, 3) # larger plots increase detail and pixel count
    plot_size_chunks = (6, 6)
except: # uni mode
    title_size = 15
    label_size = 8
    HOME = ("C:\\Users\\c55626na\\OneDrive - " # university computer user name
            "The University of Manchester\\Individual Project\\Downloads")
    os.chdir(HOME)
    plot_size = (5, 5) # larger plots increase detail and pixel count
    plot_size_chunks = (11, 11)

# %% General Mega Giga Function
response_time = 0.0

def get_sat(sat_name, sat_number, folder):
    print("====================")
    print(f"||{sat_name} {sat_number} Start||")
    print("====================")
    table_print(n_chunks=n_chunks, high_res=high_res, 
                show_plots=show_index_plots, save_images=save_images, 
                labelling=label_data)
    
    # %%% 1. Opening Images and Creating Image Arrays
    print("==========")
    print("| STEP 1 |")
    print("==========")
    stop_event, thread = start_spinner(message="opening images and "
                                       "creating image arrays")
    start_time = time.monotonic()
    
    file_paths = []
    satellite = f"\\{sat_name} {sat_number}\\"
    path = HOME + satellite + folder
    os.chdir(path)
    
    path = path + "\\GRANULE"
    subdirs = [d for d in os.listdir(path) 
               if os.path.isdir(os.path.join(path, d))]
    if len(subdirs) == 1:
        path = (f"{path}\\{subdirs[0]}")
    else:
        print("Too many subdirectories in 'GRANULE':", len(subdirs))
        return
    
    if high_res:
        res = "10m"
        path_10 = (f"{path}\\IMG_DATA\\R10m") # blue, green, nir
        path_20 = (f"{path}\\IMG_DATA\\R20m") # swir1 and swir2
    else:
        res = "60m"
        path_60 = (f"{path}\\IMG_DATA\\R60m") # all bands
    
    (sentinel_name, instrument_and_product_level, datatake_start_sensing_time, 
     processing_baseline_number, relative_orbit_number, tile_number_field, 
     product_discriminator_and_format) = folder.split("_")
    prefix = (f"{tile_number_field}_{datatake_start_sensing_time}")
    bands = get_sentinel_bands(sat_number, high_res)
    
    for band in bands:
        if high_res:
            if band == "02" or band == "03" or band == "08":
                file_paths.append(f"{path_10}\\{prefix}_B{band}_10m.jp2")
            else:
                file_paths.append(f"{path_20}\\{prefix}_B{band}_20m.jp2")
        else:
                file_paths.append(f"{path_60}\\{prefix}_B{band}_60m.jp2")
    
    image_arrays = image_to_array(file_paths) # this is the long operation
    
    time_taken = time.monotonic() - start_time
    end_spinner(stop_event, thread)
    print(f"step 1 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 2. Masking Clouds
    print("==========")
    print("| STEP 2 |")
    print("==========")
    stop_event, thread = start_spinner(message="masking clouds")
    start_time = time.monotonic()
    
    path = (HOME + satellite + folder + 
            "\\GRANULE\\" + subdirs[0] + "\\QI_DATA\\")
    image_arrays = mask_sentinel(path, high_res, image_arrays)
    
    time_taken = time.monotonic() - start_time
    end_spinner(stop_event, thread)
    print(f"step 2 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 3. Calculating Water Indices
    print("==========")
    print("| STEP 3 |")
    print("==========")
    stop_event, thread = start_spinner(message="populating water index arrays")
    start_time = time.monotonic()
    
    # first convert to int. np.uint16 type is bad for algebraic operations!
    for i, image_array in enumerate(image_arrays):
        image_arrays[i] = image_array.astype(int)
    blue, green, nir, swir1, swir2 = image_arrays
    
    np.seterr(divide="ignore", invalid="ignore")
    ndwi = ((green - nir) / (green + nir))
    mndwi = ((green - swir1) / (green + swir1))
    awei_sh = (green + 2.5 * blue - 1.5 * (nir + swir1) - 0.25 * swir2)
    awei_nsh = (4 * (green - swir1) - (0.25 * nir + 2.75 * swir2))
    
    indices = [ndwi, mndwi, awei_sh, awei_nsh]
    
    time_taken = time.monotonic() - start_time
    end_spinner(stop_event, thread)
    print(f"step 3 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 4. Showing Indices
    print("==========")
    print("| STEP 4 |")
    print("==========")
    if show_index_plots:
        if save_images:
            print("saving and displaying water index images...")
        else:
            print("displaying water index images...")
        start_time = time.monotonic()
        plot_indices(indices, sat_number, plot_size, dpi, save_images, res)
        time_taken = time.monotonic() - start_time
        print(f"step 4 complete! time taken: {round(time_taken, 2)} seconds")
    else:
        print("not displaying water index images")
    
    # %%% 5. Data Labelling
    global response_time
    print("==========")
    print("| STEP 5 |")
    print("==========")
    start_time = time.monotonic()
    if label_data:
        print("data labelling start")
        
        # %%%% 5.1 Searching for, Opening, and Converting RGB Image
        stop_event, thread = start_spinner(message=f"opening {res} "
                                           "resolution true colour image")
        path = HOME + satellite + folder
        
        tci_path = f"{path}\\GRANULE\\{subdirs[0]}\\IMG_DATA\\R{res}\\"
        tci_file_name = prefix + f"_TCI_{res}.jp2"
        tci_array = image_to_array(tci_path + tci_file_name)
        
        tci_60_path = f"{path}\\GRANULE\\{subdirs[0]}\\IMG_DATA\\R60m\\"
        tci_60_file_name = prefix + "_TCI_60m.jp2"
        c = 10 # compress 60m resolution TCI for faster plotting
        with Image.open(tci_60_path + tci_60_file_name) as img:
            size = (img.width//c, img.height//c)
            tci_60_array = np.array(img.resize(size))
        end_spinner(stop_event, thread)
        
        # %%%% 5.2 Creating Chunks from Satellite Imagery
        stop_event, thread = start_spinner(message=f"creating {n_chunks} chunks"
                                           " from satellite imagery")
        index_chunks = []
        for index in indices:
            index_chunks.append(split_array(array=index, n_chunks=n_chunks))
        tci_chunks = split_array(array=tci_array, n_chunks=n_chunks)
        end_spinner(stop_event, thread)
        
        # %%%% 5.3 Preparing File for Labelling
        stop_event, thread = start_spinner(message="preparing file for labelling")
        index_labels = ["NDWI", "MNDWI", "AWEI-SH", "AWEI-NSH"]
        break_flag = False
        
        path = HOME + satellite + folder
        labelling_path = path + "\\data\\data_labelling"
        if os.path.exists(labelling_path):
            os.chdir(labelling_path)
        else:
            os.makedirs(labelling_path)
        
        lines = []
        header = ("chunk,reservoirs,water bodies,reservoir "
        "coordinates,,,,,water body coordinates\n")
        blank_entry_check(file=data_file) # remove all blank entries
        
        while True:
            # file will always exist due to blank_entry_check call
            with open(data_file, "r") as file:
                lines = file.readlines()
            try:
                # Validate file data
                for i in range(1, len(lines) - 1):
                    current_chunk = int(lines[i].split(",")[0])
                    next_chunk = int(lines[i + 1].split(",")[0])
                    if next_chunk - current_chunk != 1:
                        print(f"Line {i + 2}, expected chunk {current_chunk + 1}")
                        raise ValueError("File validity error")
                last_chunk = int(lines[-1].split(",")[0])
                break
            except (ValueError, IndexError) as e:
                end_spinner(stop_event, thread)
                print(f"error - file with invalid data: {e}")
                print("type 'quit' to exit, or 'new' for a fresh file")
                response_time_start = time.monotonic()
                ans = input("press enter to retry: ")
                response_time += time.monotonic() - response_time_start
                if ans.lower() == 'quit':
                    return indices
                elif ans.lower() == 'new':
                    print("creating new file...")
                    with open(data_file, "w") as file:
                        file.write(header)
                        file.write("0, 1, 0\n") # dummy file to start up
                    continue
        end_spinner(stop_event, thread)
        
        i = last_chunk + 1 # from this point on, "i" is off-limits as a counter
        
        # find chunks with no reservoir coordinate data
        reservoir_rows = []
        body_rows = []
        data_correction = False
        
        with open(data_file, "r") as file:
            lines = file.readlines() # reread lines in case of changes
            globals()["lines"] = lines
        for j in range(1, len(lines)): # starting from the "headers" line
            # check for reservoirs without coordinates
            num_of_reservoirs = int(lines[j].split(",")[1])
            res_no_coords = False # check if reservoirs have coordinates
            try: # try to access coordinates
                res_coord = lines[j].split(",")[2+num_of_reservoirs]
                if res_coord[0] != "[":
                    res_no_coords = True
            except: # if unable to access, they do not exist
                res_no_coords = True
            if num_of_reservoirs != 0 and res_no_coords:
                reservoir_rows.append(j-1)
                data_correction = True
            
            # check for non-reservoir water bodies without coordinates
            num_of_bodies = int(lines[j].split(",")[2])
            body_no_coords = False # check if water bodies have coordinates
            try: # try to access coordinates
                body_coord = lines[j].split(",")[7+num_of_bodies]
                if body_coord[0] != "[":
                    body_no_coords = True
            except: # if unable to access, they do not exist
                body_no_coords = True
            if num_of_bodies != 0 and body_no_coords:
                body_rows.append(j-1)
                data_correction = True
        invalid_rows = combine_sort_unique(reservoir_rows, body_rows)
        
        if data_correction:
            print(f"found {len(invalid_rows)} chunks containing "
                   "incomplete or missing no coordinate data")
            i = invalid_rows[0]
            invalid_rows_index = 0
        
        # %%%% 5.4 Outputting Images
        print("outputting images...")
        
        while i < len(index_chunks[0]):
            if break_flag:
                break
            plot_chunks(ndwi, mndwi, index_chunks, plot_size_chunks, i, 
                        title_size, label_size, tci_chunks, tci_60_array)
            max_index = [0, 0]
            max_index[0] = round(np.nanmax(index_chunks[0][i]), 2)
            print(f"MAX {index_labels[0]}: {max_index[0]}", end=" | ")
            max_index[1] = round(np.nanmax(index_chunks[1][i]), 2)
            print(f"MAX {index_labels[1]}: {max_index[1]}")
            
            # %%%% 5.5 User Labelling
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
            n_reservoirs = input("how many reservoirs? ")
            n_bodies = ""
            entry_list = []
            while True:
                blank_entry_check(file=data_file)
                back_flag = False
                try:
                    # handle number of reservoirs entry
                    n_reservoirs = int(n_reservoirs)
                    entry_list = [i,n_reservoirs,""]
                    while n_reservoirs > 5: # NOTE add user input type check
                        print("maximum of 5 reservoirs")
                        n_reservoirs = input("how many reservoirs? ")
                    if n_reservoirs != 0:
                        print("please draw a square around the reservoir(s)", 
                              flush=True)
                        chunk_coords = prompt_roi(tci_chunks[i], n_reservoirs)
                        for coord in chunk_coords:
                            entry_list.append(coord)
                    while len(entry_list) < 8:
                        entry_list.append("")
                    
                    # handle number of non-reservoir water bodies entry
                    n_bodies = input("how many non-reservoir water bodies? ")
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
                        print("taking a break")
                        response_time += time.monotonic() - response_time_start
                        break_flag = True
                        break
                    elif "back" in n_bodies or "back" in n_reservoirs:
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
                        print("error: non-integer response."
                              "\ntype 'break' to save and quit"
                              "\ntype 'back' to go to previous chunk")
                        n_reservoirs = input("how many reservoirs? ")
            
            # save results to the responses csv file
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
    else:
        print("not labelling data")
    print(f"responding time: {round(response_time, 2)} seconds")            
    
    time_taken = time.monotonic() - start_time - response_time
    print(f"data labelling complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 6. Data Segmentation
    print("==========")
    print("| STEP 6 |")
    print("==========")
    #stop_event, thread = start_spinner(message="dividing data into classes")
    start_time = time.monotonic()
    
    path = HOME + satellite + folder
    labelling_path = path + "\\data\\data_labelling"
    if os.path.exists(labelling_path):
        os.chdir(labelling_path)
    else:
        os.makedirs(labelling_path)
    
    # find the index of every chunk with a reservoir or water body
    res_rows = []
    res_coords = []
    body_rows = []
    body_coords = []
    with open(data_file, "r") as file:
        lines = file.readlines()
    for i in range(1, len(lines)):
        lines[i] = lines[i].split(",")
        if int(lines[i][1]) > 0: # if there is a reservoir
            res_rows.append(lines[i])
            if int(res_rows[-1][1]) > 1:
                for j in range(3, 3+int(res_rows[-1][1])):
                    res_coords.append(extract_coords(res_rows[-1][j]))
            elif int(res_rows[-1][1]) == 1:
                res_coords.append(extract_coords(res_rows[-1][3]))
        if int(lines[i][2]) > 0: # if there is a water body
            body_rows.append(lines[i])
            if int(body_rows[-1][2]) > 1:
                for j in range(8, 8+int(body_rows[-1][2])):
                    body_coords.append(extract_coords(body_rows[-1][j]))
            elif int(body_rows[-1][2]) == 1:
                body_coords.append(extract_coords(body_rows[-1][3]))
# =============================================================================
#             for j in range(8, 7+int(body_rows[-1][2])):
#                 body_coords.append(extract_coords(body_rows[-1][j]))
# =============================================================================
    globals()["lines"] = lines
    globals()["res_rows"] = res_rows
    globals()["body_rows"] = body_rows
    globals()["res_coords"] = res_coords
    globals()["body_coords"] = body_coords
    
    # isolate each reservoir and water body in their own image
# =============================================================================
#     res_coords = []
#     bod_coords = []
#     for i in range(0, len(res_rows)):
#             for j in range(3, 2+int(res_rows[i])):
#                 print("i", i)
#                 print("j", j)
#                 res_coords.append(extract_coords(res_rows[0][j]))
#     for i in range(0, len(body_rows)):
#         for j in range(8, 7+int(body_rows[i])):
#             bod_coords.append(extract_coords(lines[body_rows[i]][j]))
#     globals()["res_coords"] = res_coords
#     globals()["bod_coords"] = bod_coords
# =============================================================================
    
    # save each image
    segmenting_path = path + "\\data\\data_segmenting"
    if os.path.exists(segmenting_path):
        os.chdir(segmenting_path)
    else:
        os.makedirs(segmenting_path)
    
    
    time_taken = time.monotonic() - start_time
    #end_spinner(stop_event, thread)
    print(f"step 6 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 7. Satellite Output
    return indices
# %% Running Functions
"""
Sentinel 2 has varying resolution bands, with Blue (2), Green (3), Red (4), and 
NIR (8) having 10m spatial resolution, while SWIR 1 (11) and SWIR 2 (12) have 
20m spatial resolution. There is no MIR band, so MNDWI is calculated correctly 
with the SWIR2 band. 
"""
s2_indices = get_sat(sat_name="Sentinel", sat_number=2, 
                          folder=("S2C_MSIL2A_20250301T111031_N0511_R137"
                                  "_T31UCU_20250301T152054.SAFE"))
stop_event, thread = start_spinner(message="splitting indices")
ndwi, mndwi, awei_sh, awei_nsh = s2_indices
end_spinner(stop_event, thread)

# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME - response_time
print(f"total processing time: {round(TOTAL_TIME, 2)} seconds", flush=True)
