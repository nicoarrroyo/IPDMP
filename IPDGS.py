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
import csv

# %%% Internal Function Imports
from image_functions import compress_image, plot_indices, mask_sentinel
from image_functions import prompt_roi
from calculation_functions import get_indices
from satellite_functions import get_sentinel_bands
from misc_functions import table_print, split_array

# %%% General Image and Plot Properties
compression = 1 # 1 for full-sized images, bigger integer for smaller images
dpi = 3000 # 3000 for full resolution, below 1000, images become fuzzy
n_chunks = 5000 # number of chunks into which images are split
save_images = False
high_res = False # use finer 10m spatial resolution (slower)
show_index_plots = False
label_data = True
uni_mode = True
if uni_mode:
    plot_size = (5, 5) # larger plots increase detail and pixel count
    plot_size_chunks = (7, 7)
    HOME = ("C:\\Users\\c55626na\\OneDrive - "
            "The University of Manchester\\Individual Project")
else:
    plot_size = (3, 3) # larger plots increase detail and pixel count
    plot_size_chunks = (5, 5)
    HOME = ("C:\\Users\\nicol\\Documents\\UoM\\YEAR 3\\"
            "Individual Project\\Downloads")

# %% General Mega Giga Function
do_s2 = True
response_time = 0

def get_sat(sat_name, sat_number, folder):
    print("====================")
    print(f"||{sat_name} {sat_number} Start||")
    print("====================")
    table_print(compression=compression, DPI=dpi, 
                n_chunks=n_chunks, save_images=save_images, high_res=high_res, 
                show_plots=show_index_plots, labelling=label_data, 
                uni_mode=uni_mode)
    
    # %%% 1. Opening Images and Creating Image Arrays
    print("opening images and creating image arrays", end="... ")
    start_time = time.monotonic()
    
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
        path_60 = (path + "IMG_DATA\\R60m\\")
    
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
    
    path = (HOME + satellite + folder + 
            "\\GRANULE\\" + subdirs[0] + "\\QI_DATA\\")
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
    if show_index_plots:
        if save_images:
            print("saving and displaying water index images...")
        else:
            print("displaying water index images...")
        start_time = time.monotonic()
        plot_indices(indices, sat_number, plot_size, compression, 
                     dpi, save_images, res)
        time_taken = time.monotonic() - start_time
        print("image display complete! "
              f"time taken: {round(time_taken, 2)} seconds")
    else:
        print("not displaying water index images")
    # %%% 5. Data Labelling
    if label_data:
        print("data labelling start")
        start_time = time.monotonic()
        
        # %%%% 5.1 Searching for, Opening, and Converting RGB Image
        print("opening " + res + " resolution true colour image", end="... ")
        path = HOME + satellite + folder
        
        tci_path = f"{path}\\GRANULE\\{subdirs[0]}\\IMG_DATA\\R{res}\\"
        tci_file_name = prefix + f"_TCI_{res}.jp2"
        with Image.open(tci_path + tci_file_name) as tci_image:
            tci_array = np.array(tci_image)
        
        tci_60_path = f"{path}\\GRANULE\\{subdirs[0]}\\IMG_DATA\\R60m\\"
        tci_60_file_name = prefix + "_TCI_60m.jp2"
        
        c = 5 # compression for this operation
        tci_60_array, size = compress_image(c, tci_60_path + tci_60_file_name)
        side_length = size[1]
        print("complete!")
        
        # %%%% 5.2 Creating Chunks from Satellite Imagery
        print("creating", n_chunks, "chunks from satellite imagery", end="... ")
        index_chunks = []
        for index in indices:
            index_chunks.append(split_array(array=index, n_chunks=n_chunks))
        tci_chunks = split_array(array=tci_array, n_chunks=n_chunks)
        chunk_length = side_length / np.sqrt(len(tci_chunks))
        print("complete!")
        
        # %%%% 5.3 Preparing File for Labelling and Outputting Images
        print("preparing file for labelling and outputting images...")
        index_labels = ["NDWI", "MNDWI", "AWEI-SH", "AWEI-NSH"]
        break_flag = False
        
        path = HOME + satellite + folder
        os.chdir(path)
        
        lines = []
        responses_file_name = "responses_" + str(n_chunks) + "_chunks.csv"
        try: # check if file exists
            with open(responses_file_name, "r") as re: # read-write file
                lines = re.readlines()
                try: # check if file has data in it
                    last_chunk = int(lines[-1].split(",")[0])
                except: # otherwise start at first point
                    with open(responses_file_name, mode="w") as create:
                        create.write("chunk,reservoirs") # input headers
                        last_chunk = -1 # must be new sheet, no last chunk
        except: # otherwise create a file
            with open(responses_file_name, mode="w") as create:
                create.write("chunk,reservoirs") # input headers
                last_chunk = -1 # must be new sheet, no last chunk
        
        while True:
            try: # check if file is open
                with open(responses_file_name, mode="a") as ap:
                    break
            except IOError:
                print("could not open file - please close the responses file")
                input("press enter to retry")
        
        i = last_chunk + 1
        rewriting = False
        coords = []
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
            
            # %%%% 5.4 User Labelling
            global response_time
            response_time_start = time.monotonic()
            n_reservoirs = input("how many reservoirs? ")
            while True:
                try:
                    n_reservoirs = int(n_reservoirs)
                    with open(responses_file_name, mode="a") as ap: # append
                        if not rewriting:
                            ap.write(f"\n{i},{n_reservoirs}")
                        else:
                            ap.write(f"{i},{n_reservoirs}")
                    rewriting = False
                    if n_reservoirs != 0:
                        #prompt_roi(tci_chunks[i])
                        coords.append(prompt_roi(tci_chunks[i]))
                        globals()["coords"] = coords
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
                        rewriting = True
                        try:
                            n_backs = int(n_reservoirs.split(" ")[1])
                        except:
                            n_backs = 1
                        i -= n_backs
                        print("returning to chunk", i)
                        with open(responses_file_name, mode="r") as re: # read
                            rows = list(csv.reader(re))
                        for j in range(n_backs):
                            rows.pop() # remove the last "n_backs" rows
                        with open(responses_file_name, mode="w") as wr: # write
                            for j in range(len(rows)):
                                wr.write(f"{rows[j][0]},{rows[j][1]}\n")
                        break
                    print("error: non-integer response."
                          "\ntype 'break' to save and quit"
                          "\ntype 'back' to go to previous chunk")
                    n_reservoirs = input("how many reservoirs? ")
    else:
        print("not labelling data")
        return indices
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
    print("splitting")
    ndwi, mndwi, awei_sh, awei_nsh = s2_indices
    print("done")
# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME - response_time
print(f"total processing time: {round(TOTAL_TIME, 2)} seconds")
