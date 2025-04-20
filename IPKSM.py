# %% 0. Start
# %%% 0.1 Import External Libraries
import time
MAIN_START_TIME = time.monotonic()
import os
import numpy as np
from tensorflow import keras

# %%% 0.2 Import Internal Functions
from data_handling import change_to_folder
from image_handling import image_to_array, mask_sentinel, save_image_file
from misc import split_array, create_9_random_coords
from user_interfacing import start_spinner, end_spinner

# %%% 0.3 Directory Management
try: # personal pc mode
    HOME = ("C:\\Users\\nicol\\OneDrive - " # personal computer user name
            "The University of Manchester\\Individual Project\\Downloads")
    os.chdir(HOME)
except: # uni mode
    HOME = ("C:\\Users\\c55626na\\OneDrive - " # university computer user name
            "The University of Manchester\\Individual Project\\Downloads")
    os.chdir(HOME)

folder = "S2C_MSIL2A_20250301T111031_N0511_R137_T31UCU_20250301T152054.SAFE"
n_chunks = 5000

# %% 1. Load Sentinel 2 Image File
print("==========")
print("| STEP 1 |")
print("==========")
stop_event, thread = start_spinner(message="opening images and "
                                   "creating image arrays")
start_time = time.monotonic()

# %%% 1.1 Establishing Paths
"""Most Sentinel 2 files that come packaged in a satellite image folder 
follow naming conventions that use information contained in the title of 
the folder. This information can be used to easily navigate through the 
folder's contents."""

satellite = "\\Sentinel 2\\"
path = HOME + satellite + folder
os.chdir(path)

# %%%% 1.1.1 Subfolder naming convention edge case
"""This folder has a strange naming convention that doesn't quite apply to 
the other folders, so it's difficult to find a rule that would work for 
any Sentinel 2 image. The easier way of finding this folder is by 
searching for any available directories in the GRANULE folder, and if 
there is more than one, then alert the user and exit, otherwise go into 
that one directory because it will be the one we're looking for."""

path = path + "\\GRANULE"
subdirs = [d for d in os.listdir(path) 
           if os.path.isdir(os.path.join(path, d))]
if len(subdirs) == 1:
    path = (f"{path}\\{subdirs[0]}")
else:
    print("Too many subdirectories in 'GRANULE':", len(subdirs))
    exit()

# %%%% 1.1.2 Resolution selection and file name deconstruction
"""Low resolution is beneficial for faster processing times, but is not 
good for rigorous data generation. High resolution combines some 10m and 
20m bands for the highest fidelity images, but processing these images is 
much slower. The file paths pointing to the band image files are also 
finalised in this section."""

path_10 = (f"{path}\\IMG_DATA\\R10m") # green and nir

(sentinel_name, instrument_and_product_level, datatake_start_sensing_time, 
 processing_baseline_number, relative_orbit_number, tile_number_field, 
 product_discriminator_and_format) = folder.split("_")
prefix = (f"{tile_number_field}_{datatake_start_sensing_time}")
bands = ["03", "08"] # green and nir

file_paths = []
for band in bands:
    file_paths.append(f"{path_10}\\{prefix}_B{band}_10m.jp2")

# %%% 1.2 Opening and Converting Images
"""This is the long operation. It is very costly to open the large images, 
which is why the high-res option exists. When troubleshooting or just 
trying out the program, it is easier and much faster (about 20x faster) to 
just use the 60m resolution images. However, when doing any actual data 
generation, the 60m resolution images are not sufficient."""

image_arrays = image_to_array(file_paths)

time_taken = time.monotonic() - start_time
end_spinner(stop_event, thread)
print(f"step 1 complete! time taken: {round(time_taken, 2)} seconds")

# %% 2. Mask Clouds From the Image
print("==========")
print("| STEP 2 |")
print("==========")
stop_event, thread = start_spinner(message="masking clouds")
start_time = time.monotonic()

path = (HOME + satellite + folder + 
        "\\GRANULE\\" + subdirs[0] + "\\QI_DATA\\")
image_arrays = mask_sentinel(path=path, 
                             high_res=True, 
                             image_arrays=image_arrays)

time_taken = time.monotonic() - start_time
end_spinner(stop_event, thread)
print(f"step 2 complete! time taken: {round(time_taken, 2)} seconds")

# %% 3. Calculate NDWI
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

time_taken = time.monotonic() - start_time
end_spinner(stop_event, thread)
print(f"step 3 complete! time taken: {round(time_taken, 2)} seconds")

# %% 4. Open and Convert True Colour Image
"""nico!! remember to add a description!"""
print("==========")
print("| STEP 4 |")
print("==========")
stop_event, thread = start_spinner(message="opening 10m "
                                   "resolution true colour image")
path = HOME + satellite + folder

tci_path = f"{path}\\GRANULE\\{subdirs[0]}\\IMG_DATA\\R10m\\"
tci_file_name = f"{prefix}_TCI_10m.jp2"
tci_array = image_to_array(tci_path + tci_file_name)
end_spinner(stop_event, thread)

# %% 5. Create Chunks from Satellite Imagery
"""nico!! remember to add a description!"""
print("==========")
print("| STEP 5 |")
print("==========")
stop_event, thread = start_spinner(message=f"creating {n_chunks} chunks"
                                   " from satellite imagery")
ndwi_chunks = split_array(array=ndwi, n_chunks=n_chunks)
tci_chunks = split_array(array=tci_array, n_chunks=n_chunks)
chunk_size = ndwi_chunks[0].shape

global_min = min(np.nanmin(chunk) for chunk in ndwi_chunks)
global_max = 0.6*max(np.nanmax(chunk) for chunk in ndwi_chunks)

end_spinner(stop_event, thread)

stop_event, thread = start_spinner(message="saving chunks as image files")
for i, chunk in enumerate(ndwi_chunks):
    test_data_path = f"{path}\\test data\\ndwi"
    change_to_folder(test_data_path)
    mini_chunk_coords = create_9_random_coords(0, 0, 
                                               chunk_size[0], chunk_size[1])
    for j, coords in enumerate(mini_chunk_coords):
        image_name = (f"ndwi chunk {i} minichunk {j}.png")
        save_image_file(data=chunk, image_name=image_name, normalise=True, 
                        coordinates=coords, margin=0, 
                        g_max=global_max, g_min=global_min)

end_spinner(stop_event, thread)
# =============================================================================
# for i in range(len(unlabelled_chunk_indices)):
#     chunk_n = unlabelled_chunk_indices[i]
#     unlabelled_coords = create_9_random_coords(1, 1, 156, 156)
#     
#     # NDWI data
#     unlabelled_ndwi_path = path + "\\data\\ndwi\\unlabelled"
#     change_to_folder(unlabelled_ndwi_path)
#     for j, unlabelled_coord in enumerate(unlabelled_coords):
#         image_name = (f"ndwi unlabelled chunk({chunk_n}) "
#                       f"minichunk({j}).png")
#         save_image_file(data=ndwi_chunks[chunk_n], 
#                         image_name=image_name, 
#                         normalise=True, 
#                         coordinates=unlabelled_coord, 
#                         margin=0, 
#                         norm=norm)
#     # TCI data
#     unlabelled_tci_path = path + "\\data\\tci\\unlabelled"
#     change_to_folder(unlabelled_tci_path)
#     for j, unlabelled_coord in enumerate(unlabelled_coords):
#         image_name = (f"tci unlabelled chunk({chunk_n}) "
#                       f"minichunk({j}).png")
#         save_image_file(data=tci_chunks[chunk_n], 
#                         image_name=image_name, 
#                         normalise=False, 
#                         coordinates=unlabelled_coord, 
#                         margin=0, 
#                         norm=norm)
# print("complete")
# =============================================================================

# %% 6. Load Model

# %% Final
TOTAL_TIME = time.monotonic() - MAIN_START_TIME
print(f"total processing time: {round(TOTAL_TIME, 2)} seconds", flush=True)
