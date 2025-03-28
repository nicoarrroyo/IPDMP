""" Individual Project Data Generation Software (IPDGS-25-03) """
""" Update Notes (from previous version IPMLS-25-02)
- name change
    - from Individual Project Machine Learning Software (IPMLS) to
        Individual Project Data Generation Software (IPDGS)
    - Individual Project Random Forest Model (IPRFM) made for model development
    - IPMLS is the combination of IPDGS and IPRFM
- earth engine
    - gee removed entirely, switch to local files
- optimisations
    - low resolution option for sentinel 2
- output plots
    - minimums and interpolation removed
- sentinel 2
    - functional index calculation, plot outputs, and plot saving
    - new general function for landsat and/or sentinel
        - now removed landsat, focusing entirely on sentinel
- machine learning
    - new section to allow user to manually label chunks of an image
    - also able to outline a rectangle containing the reservoir
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
import csv
# import threading

# %%% Internal Function Imports
from image_functions import image_to_array, plot_indices, mask_sentinel
from image_functions import prompt_roi
from calculation_functions import get_indices
from satellite_functions import get_sentinel_bands
from misc_functions import table_print, split_array, rewrite, blank_entry_check
from misc_functions import check_file_permission # , spinner

# %%% General Image and Plot Properties
dpi = 3000 # 3000 for full resolution, below 1000, images become fuzzy
n_chunks = 4999 # number of chunks into which images are split
save_images = False
high_res = False # use finer 10m spatial resolution (slower)
show_index_plots = True
label_data = True

drive_link = ("https://drive.google.com/drive/folders/1z4jRh6LlInO5ivhA9s"
              "9CN0U2azSXtrBm?usp=sharing")
try: # personal pc mode
    HOME = ("C:\\Users\\nicol\\Documents\\UoM\\YEAR 3\\"
            "Individual Project\\Downloads")
    os.chdir(HOME)
    plot_size = (3, 3) # larger plots increase detail and pixel count
    plot_size_chunks = (5, 5)
except: # uni mode
    HOME = ("C:\\Users\\c55626na\\OneDrive - "
            "The University of Manchester\\Individual Project")
    os.chdir(HOME)
    plot_size = (5, 5) # larger plots increase detail and pixel count
    plot_size_chunks = (7, 7)

# %% General Mega Giga Function
do_s2 = True
response_time = 0

def get_sat(sat_name, sat_number, folder):
    print("====================")
    print(f"||{sat_name} {sat_number} Start||")
    print("====================")
    table_print(n_chunks=n_chunks, save_images=save_images, high_res=high_res, 
                show_plots=show_index_plots, labelling=label_data)
    
    # %%% 1. Opening Images and Creating Image Arrays
    print("==========")
    print("| STEP 1 |")
    print("==========")
    print("opening images and creating image arrays")
    start_time = time.monotonic()
    
    print("establishing paths", end="... ")
    file_paths = []
    satellite = f"\\{sat_name} {sat_number}\\"
    path = HOME + satellite + folder + "\\GRANULE"
    
    subdirs = [d for d in os.listdir(path) 
               if os.path.isdir(os.path.join(path, d))]
    if len(subdirs) == 1:
        path = (f"{path}\\{subdirs[0]}\\")
        os.chdir(path)
    else:
        print("Too many subdirectories in 'GRANULE':", len(subdirs))
        return
    
    if high_res:
        res = "10m"
        path_10 = (path + "IMG_DATA\\R10m\\") # blue, green, nir
        path_20 = (path + "IMG_DATA\\R20m\\") # swir1 and swir2
    else:
        res = "60m"
        path_60 = (path + "IMG_DATA\\R60m\\") # all bands
    
    (sentinel_name, instrument_and_product_level, datatake_start_sensing_time, 
     processing_baseline_number, relative_orbit_number, tile_number_field, 
     product_discriminator_and_format) = folder.split("_")
    prefix = (f"{tile_number_field}_{datatake_start_sensing_time}")
    bands = get_sentinel_bands(sat_number, high_res)
    globals()["bands"] = bands
    for band in bands:
        if high_res:
            if band == "02" or band == "03" or band == "08":
                file_paths.append(path_10 + prefix + "_B" + band + "_10m.jp2")
            else:
                file_paths.append(path_20 + prefix + "_B" + band + "_20m.jp2")
        else:
            file_paths.append(path_60 + prefix + "_B" + band + "_60m.jp2")
    print("complete!")
    
    print("opening and converting images", end="... ")
    image_arrays = image_to_array(file_paths)
    print("complete!")
    
    time_taken = time.monotonic() - start_time
    print(f"step 1 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 2. Masking Clouds
    print("==========")
    print("| STEP 2 |")
    print("==========")
    print("masking clouds")
    start_time = time.monotonic()
    
    path = (HOME + satellite + folder + 
            "\\GRANULE\\" + subdirs[0] + "\\QI_DATA\\")
    image_arrays = mask_sentinel(path, high_res, image_arrays)
    
    time_taken = time.monotonic() - start_time
    print(f"step 2 complete! time taken: {round(time_taken, 2)} seconds")
    
    # %%% 3. Calculating Water Indices
    print("==========")
    print("| STEP 3 |")
    print("==========")
    print("populating water index arrays", end="")
    start_time = time.monotonic()
    
    blue, green, nir, swir1, swir2 = image_arrays
    indices = get_indices(blue, green, nir, swir1, swir2)
    
    time_taken = time.monotonic() - start_time
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
    print("==========")
    print("| STEP 5 |")
    print("==========")
    if label_data:
        print("data labelling start")
        start_time = time.monotonic()
        
        # %%%% 5.1 Searching for, Opening, and Converting RGB Image
        print("opening " + res + " resolution true colour image.", end="")
        path = HOME + satellite + folder
        
        tci_path = f"{path}\\GRANULE\\{subdirs[0]}\\IMG_DATA\\R{res}\\"
        tci_file_name = prefix + f"_TCI_{res}.jp2"
        tci_array = image_to_array(tci_path + tci_file_name)
        
        tci_60_path = f"{path}\\GRANULE\\{subdirs[0]}\\IMG_DATA\\R60m\\"
        tci_60_file_name = prefix + "_TCI_60m.jp2"
        c = 5 # compress 60m resolution TCI for faster plotting
        with Image.open(tci_60_path + tci_60_file_name) as img:
            size = (img.width//c, img.height//c)
            tci_60_array = np.array(img.resize(size))
            side_length = img.width//c
            print(".", end=" ")
        print("complete!")
        
        # %%%% 5.2 Creating Chunks from Satellite Imagery
        print("creating", n_chunks, "chunks from satellite imagery", end="... ")
        index_chunks = []
        for index in indices:
            index_chunks.append(split_array(array=index, n_chunks=n_chunks))
        tci_chunks = split_array(array=tci_array, n_chunks=n_chunks)
        chunk_length = side_length / np.sqrt(len(tci_chunks))
        print("complete!")
        
        # %%%% 5.3 Preparing File for Labelling
        print("preparing file for labelling", end="... ")
        index_labels = ["NDWI", "MNDWI", "AWEI-SH", "AWEI-NSH"]
        break_flag = False
        
        path = HOME + satellite + folder
        os.chdir(path)
        
        lines = []
        data_file = "responses_" + str(n_chunks) + "_chunks.csv"
        blank_entry_check(file=data_file) # remove all blank entries
# =============================================================================
#         try: # check if file exists
#             with open(data_file, "r") as re: # read
#                 lines = re.readlines()
#             try: # check if file has data in it
#                 for i in range(1, len(lines) - 1): # check file validity
#                     try:
#                         current_chunk = int(lines[i].split(",")[0])
#                         next_chunk = int(lines[i+1].split(",")[0])
#                     except:
#                         print("bad data in responses file")
#                         print(f"line {i+2}, chunk {(current_chunk+1)}")
#                         return indices
#                     chunk_diff = next_chunk - current_chunk
#                     if chunk_diff != 1:
#                         print("error in responses file")
#                         print(f"line {i+2}, chunk {(current_chunk+1)}")
#                         return indices # end program if file is invalid
#                 last_chunk = int(lines[-1].split(",")[0])
#             except: # otherwise start at first point
#                 print("no data found")
#                 while True:
#                     print("please check the file for errors")
#                     print("press enter to start a new file")
#                     check_file_permission(file_name=data_file)
#                     ans = input("type quit to exit ")
#                     if "quit" in ans:
#                         return indices
#                     print("new file")
#                     with open(data_file, mode="w") as create:
#                         create.write("chunk,reservoirs,coordinates")
#                         last_chunk = -1 # must be new sheet, no last chunk
#         except: # otherwise create a file
#             print("new file")
#             with open(data_file, mode="w") as create:
#                 create.write("chunk,reservoirs,coordinates") # input headers
#                 last_chunk = -1 # must be new sheet, no last chunk
# =============================================================================
        
# =============================================================================
#         while True:
#             # check if file exists
#             try:
#                 with open(data_file, "r") as re: # read file
#                     lines = re.readlines()
#                 # check if file has valid data
#                 try:
#                     for i in range(1, len(lines) - 1):
#                         current_chunk = int(lines[i].split(",")[0])
#                         next_chunk = int(lines[i+1].split(",")[0])
#                         chunk_diff = next_chunk - current_chunk
#                         if chunk_diff != 1:
#                             print(f"line {i+2}, chunk {(current_chunk+1)}")
#                             raise Exception("file validity error")
#                     last_chunk = int(lines[-1].split(",")[0])
#                     break
#                 # file has invalid data
#                 except:
#                     print("error - file with invalid data")
#                     print("press enter to try again, otherwise: ")
#                     ans = input("type 'quit' to exit, 'new' for a wiped file ")
#                     if "quit" in ans:
#                         return indices
#                     elif "new" in ans:
#                         raise Exception("creating new file")
#             # file doesn't exist
#             except:
#                 # create a new file
#                 print("new file")
#                 with open(data_file, mode="w") as create:
#                     create.write("chunk,reservoirs,coordinates") # input headers
#                     last_chunk = -1 # must be new sheet, no last chunk
#                 break
# =============================================================================
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
                print(f"error - file with invalid data: {e}")
                print("type 'quit' to exit, or 'new' for a fresh file")
                ans = input("press enter to retry ")
                if ans.lower() == 'quit':
                    return indices
                elif ans.lower() == 'new':
                    print("creating new file...")
                    with open(data_file, "w") as file:
                        file.write("chunk,reservoirs,coordinates\n")
                        file.write("0, 1") # dummy file to start up
                    continue
        
        check_file_permission(file_name=data_file) # check if file is open
        print("complete!")
        
        i = last_chunk + 1 # from this point on, "i" is off-limits as a counter
        first = True
        
        # find chunks with no reservoir coordinate data
        reservoir_rows = []
        reservoir_rows_index = 0
        data_correction = False
        for j in range(1, len(lines)): # starting from headers line
            num_of_reservoirs = int(lines[j].split(",")[1])
            
            no_coords = False # check if reservoirs have coordinates
            try: # try to access coordinates
                coords = lines[j].split(",")[1+num_of_reservoirs]
                if len(coords) < 5: # may be "\n" in the coords position
                    no_coords = True
            except: # if unable to access, they do not exist
                no_coords = True
            
            if num_of_reservoirs != 0 and no_coords:
                reservoir_rows.append(j-1)
                data_correction = True
        if data_correction:
            print(f"found {len(reservoir_rows)} chunks containing "
                   "reservoirs with no coordinate data")
            i = reservoir_rows[0]
        
        # %%%% 5.4 Outputting Images
        print("outputting images...")
        while i < len(index_chunks[0]):
            if break_flag:
                break
            
            # plot index chunks
            max_indices_chunk = np.zeros(len(indices))
            fig, axes = plt.subplots(1, len(indices), figsize=plot_size_chunks)
            for count, index_label in enumerate(index_labels):
                axes[count].imshow(index_chunks[count][i])
                axes[count].set_title(f"{index_label} Chunk {i}", fontsize=6)
                axes[count].axis("off")
            plt.tight_layout()
            plt.show()
            for count, max_index in enumerate(max_indices_chunk):
                max_index = round(np.amax(index_chunks[count][i]), 2)
                print(f"MAX {index_labels[count]}: {max_index}", end=" | ")
            
            # plot tci chunks
            fig, axes = plt.subplots(1, 2, figsize=plot_size_chunks)
            axes[0].imshow(tci_chunks[i])
            axes[0].set_title(f"TCI Chunk {i}", fontsize=10)
            axes[0].axis("off")
            
            axes[1].imshow(tci_60_array)
            axes[1].set_title(f"C{c} TCI 60m Resolution", fontsize=8)
            axes[1].axis("off")
            
            chunk_uly = chunk_length * (i // np.sqrt(len(tci_chunks)))
            chunk_ulx = (i * chunk_length) % side_length
            
            axes[1].plot(chunk_ulx, chunk_uly, marker=",", color="red")
            for k in range(int(chunk_length)): # make a square around the chunk
                axes[1].plot(chunk_ulx+k, chunk_uly, 
                             marker=",", color="red")
                axes[1].plot(chunk_ulx+k, chunk_uly+chunk_length, 
                             marker=",", color="red")
                axes[1].plot(chunk_ulx, chunk_uly+k, 
                             marker=",", color="red")
                axes[1].plot(chunk_ulx+chunk_length, chunk_uly+k, 
                             marker=",", color="red")
            axes[1].plot(chunk_ulx+chunk_length, chunk_uly+chunk_length, 
                         marker=",", color="red")
            
            plt.show()
            
            # %%%% 5.5 User Labelling
            global response_time
            response_time_start = time.monotonic()
            if data_correction:
                print("this chunk "
                      f"({reservoir_rows_index+1}/{len(reservoir_rows)})"
                      " should contain "
                      f"{int(lines[i+1].split(',')[1])} reservoir(s)")
            n_reservoirs = input("how many reservoirs? ")
            while True:
                blank_entry_check(file=data_file)
                
                try:
                    n_reservoirs = int(n_reservoirs)
                    entry = f"{i},{n_reservoirs}"
                    if n_reservoirs != 0:
                        print("please draw a square around the reservoir(s)")
                        raw_coords = prompt_roi(tci_chunks[i], n_reservoirs)
                        raw_coords = np.array(raw_coords)
                        chunk_coords = raw_coords * len(tci_chunks[0]) / 500
                        for coord in chunk_coords:
                            entry = f"{entry},{coord}"
                    
                    if data_correction: # add coordinates to data
                        lines[i+1] = f"{entry}\n"
                        check_file_permission(file_name=data_file)
                        with open(data_file, mode="w") as wr: # write
                            for j in range(len(lines)):
                                entry = lines[j]
                                wr.write(f"{entry}")
                        reservoir_rows_index += 1
                        if reservoir_rows_index >= len(reservoir_rows):
                            i = last_chunk + 1
                            data_correction = False
                            first = True
                            break
                        i = reservoir_rows[reservoir_rows_index]
                        response_time += time.monotonic() - response_time_start
                        break
                    
                    check_file_permission(file_name=data_file)
                    with open(data_file, mode="a") as ap: # append
                        if first:
                            ap.write(f"{entry}")
                            first = False
                        elif not first:
                            ap.write(f"\n{entry}")
                    print("generating next chunk...")
                    response_time += time.monotonic() - response_time_start
                    i += 1
                    break
                except:
                    if "break" in n_reservoirs:
                        print("taking a break")
                        response_time += time.monotonic() - response_time_start
                        break_flag = True
                        break
                    elif "back" in n_reservoirs:
                        if data_correction:
                            print("cannot use 'back' during data correction")
                            break
                        try:
                            n_backs = int(n_reservoirs.split(" ")[1])
                        except:
                            n_backs = 1
                        i -= n_backs
                        print("returning to chunk", i)
                        check_file_permission(file_name=data_file)
                        with open(data_file, mode="r") as re: # read
                            rows = list(csv.reader(re))
                        for j in range(n_backs):
                            rows.pop() # remove the last "n_backs" rows
                        with open(data_file, mode="w") as wr: # write
                            rewrite(write_file=wr, rows=rows)
                        break
                    print("error: non-integer response."
                          "\ntype 'break' to save and quit"
                          "\ntype 'back' to go to previous chunk")
                    n_reservoirs = input("how many reservoirs? ")
    else:
        print("not labelling data")
    print(f"responding time: {round(response_time, 2)} seconds")            
    
    time_taken = time.monotonic() - start_time - response_time
    print(f"data labelling complete! time taken: {round(time_taken, 2)} seconds")
    
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
    ndwi, mndwi, awei_sh, awei_nsh = s2_indices
# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME - response_time
print(f"total processing time: {round(TOTAL_TIME, 2)} seconds")
