import numpy as np
import random

def get_sentinel_bands(sentinel_n, high_res):
    """
    Sentinel 2 has thirteen spectral bands, five of which are of interest for 
    the calculation of water detection indices. The blue, green, and near-
    infrared (NIR) bands can achieve 10 metre spatial resolution, while both 
    the short-wave infrared (SWIR) 1 and 2 can achieve 20 metre spatial 
    resolution. All bands also have 60 metre resolution versions. 
    
    Most bands have the same numbering whether they are low or 
    high resolution, except for the NIR band, which is number "08" in high-res 
    and "8A" in low-res. Low-res is favourable for quicker runs, while 
    high-res is necessary for any actual data generation and handling. 
    
    Parameters
    ----------
    sentinel_n : int
        The number of Sentinel satellite. This function currently only has 
        information about Sentinel 2 bands. 
    high_res : bool
        Variable that decides whether to extract the 10 metre or 60 metre 
        version of the NIR band.
    
    Returns
    -------
    BLUE_BAND : string
        The number (as a string) of the blue band in Sentinel 2 data.
    GREEN_BAND : string
        The number (as a string) of the green band in Sentinel 2 data.
    NIR_BAND : string
        The number (as a string) of the NIR band in Sentinel 2 data.
    SWIR1_BAND : string
        The number (as a string) of the SWIR1 band in Sentinel 2 data.
    SWIR2_BAND : string
        The number (as a string) of the SWIR2 band in Sentinel 2 data.
    
    """
    if sentinel_n == 2:
        BLUE_BAND = '02'
        GREEN_BAND = '03'
        SWIR1_BAND = '11'
        SWIR2_BAND = '12'
        if high_res:
            NIR_BAND = '08'
        else:
            NIR_BAND = '8A'
        return BLUE_BAND, GREEN_BAND, NIR_BAND, SWIR1_BAND, SWIR2_BAND

def split_array(array, n_chunks):
    """
    Split any integer array into any number of chunks. 
    
    Parameters
    ----------
    array : numpy array
        A numpy array containing integers.
    n_chunks : int
        The number of chunks into which the array must be split.
    
    Returns
    -------
    chunks : list
        A list containing every chunk split off from the full array.
    
    """
    rows = np.array_split(array, np.sqrt(n_chunks), axis=0) # split into rows
    split_arrays = [np.array_split(row_chunk, np.sqrt(n_chunks), 
                                   axis=1) for row_chunk in rows]
    chunks = [subarray for row_chunk in split_arrays for subarray in row_chunk]
    return chunks

def combine_sort_unique(*arrays):
    """
    Combines two arrays, sorts the combined array in ascending order, and 
    eliminates duplicates.
    
    Parameters
    ----------
    *arrays : list
        A variable number of input arrays.
    
    Returns
    -------
    sorted_array : list
        The sorted array with unique elements.
        
    """
    combined_array = []
    for arr in arrays:
        combined_array.extend(arr)
    unique_array = list(set(combined_array))
    sorted_array = sorted(unique_array)
    return sorted_array

def create_random_coords(min_bound, max_bound):
    """
    Creates a four-element list where:
    - The first element is smaller than the third.
    - The second element is smaller than the fourth.
    - All elements are between min_bound and max_bound (inclusive).
    
    Returns
    -------
    list
        A list of four integers meeting the criteria.
    
    """
    
    # Generate the first element, ensure space for a larger third element
    first_element = random.randint(min_bound, max_bound-2)
    
    # Generate the third element, ensuring it's greater than the first
    third_element = random.randint(first_element + 1, max_bound)
    
    # Generate the second element, ensure space for a larger fourth element
    second_element = random.randint(min_bound, max_bound-2)
    
    # Generate the fourth element, ensuring it's greater than the second
    fourth_element = random.randint(second_element + 1, max_bound)
    
    return [first_element, second_element, third_element, fourth_element]

def create_9_random_coords(ulx, uly, lrx, lry):
    """
    Splits a box defined by [ulx, uly, lrx, lry] into 9 smaller boxes 
    with slight overlaps.
    
    Parameters
    ----------
    ulx : int
        Upper-left x-coordinate of the original box.
    uly : int
        Upper-left y-coordinate of the original box.
    lrx : int
        Lower-right x-coordinate of the original box.
    lry : int
        Lower-right y-coordinate of the original box.
    
    Returns
    -------
    sub_boxes : list
        A list of 9 lists, each representing a smaller box in the 
        format [ulx, uly, lrx, lry].
    
    """
    
    # Calculate the base width and height of the sub-boxes
    width = lrx - ulx
    height = lry - uly
    sub_width = width / 3
    sub_height = height / 3
    
    # List to store the coordinates of the 9 sub-boxes
    sub_boxes = []
    
    for i in range(3):
        for j in range(3):
            # Calculate the coordinates of the current sub-box with overlap
            overlap_x1 = random.randint(0, 5)
            overlap_y1 = random.randint(0, 5)
            overlap_x2 = random.randint(0, 5)
            overlap_y2 = random.randint(0, 5)
            
            sub_ulx = ulx + j * sub_width - overlap_x1
            sub_uly = uly + i * sub_height - overlap_y1
            sub_lrx = ulx + (j + 1) * sub_width + overlap_x2
            sub_lry = uly + (i + 1) * sub_height + overlap_y2
            
            # Ensure the sub-box coords are in the bounds of the original box
            sub_ulx = max(sub_ulx, ulx)
            sub_uly = max(sub_uly, uly)
            sub_lrx = min(sub_lrx, lrx)
            sub_lry = min(sub_lry, lry)
            
            #handle edge case
            if sub_lrx <= sub_ulx:
                sub_lrx = sub_ulx + 1
            if sub_lry <= sub_uly:
                sub_lry = sub_uly + 1
            
            sub_boxes.append([sub_ulx, sub_uly, sub_lrx, sub_lry])
    return sub_boxes

"""
This section is storage for functions that are not currently used in the IPDGS 
program but may be useful in future. 
"""
from PIL import Image
import os

def get_rgb(blue_path, green_path, red_path, save_image, res, show_image):
    """
    Search for or generate an RGB image composite from the red, green, and 
    blue wavelengths. This function has become somewhat outdated since I 
    discovered the true-colour image (TCI) in Sentinel 2 folders, which is 
    much more usable than the custom composite. 
    
    Parameters
    ----------
    blue_path : string
        File path to the blue file.
    green_path : string
        File path to the green file.
    red_path : string
        File path to the red file.
    save_image : bool
        Boolean variable to check if the user wants the image saved.
    res : string
        The resolution of the image array being passed and plotted. This can 
        be 10m, 20m, or 60m for Sentinel 2. 
    show_image : bool
        Boolean variable to check if the user wants the image outputted.
    
    Returns
    -------
    rgb_array : numpy array
        The RGB image array file that is generated or found.
    
    """
    bands = []
    print("creating RGB image", end="... ")
    for path in (blue_path, green_path, red_path):
        with Image.open(path) as img:
            arr = np.array(img, dtype=np.float32)
            bands.append(((arr / arr.max()) * 255).astype(np.uint8))
    rgb_array = np.stack(bands, axis=-1)
    rgb_image = Image.fromarray(rgb_array)
    print("complete!")
    if save_image:
        print("saving image", end="... ")
        rgb_image.save(f"{res}m_RGB.png")
        print("complete!")
    if show_image:
        print("displaying image", end="... ")
        rgb_image.show()
        print("complete!")
    return rgb_array

def find_rgb_file(path):
    """
    This searches a directory completely for a file that contains the phrase 
    "RGB", and "10m" for the resolution, and "bright" as the custom RGB 
    composite is too dark. The "bright" is added manually after aritifically 
    brightening the file in the GNU Image Manipulation Program. Again, this 
    function is quite outdated and is no longer used as the 
    true-colour image (TCI) is much more usable. 
    
    Parameters
    ----------
    path : string
        Any file path that can be searched fully to find a an RGB image file. 
    
    Returns
    -------
    bool
        A bool variable to indicate whether an RGB image file has been found. 
    rgb_path or full_path : string
        If the RGB image file is found, this is the path to that file. 
    
    """
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path): # if item is a folder
            found_rgb, rgb_path = find_rgb_file(full_path)
            if found_rgb:  
                return True, rgb_path
        else: # if item is a file
            if "RGB" in item and "10m" in item and "bright" in item:
                return True, full_path
    return False, None

def logical_checks(high_res, show_index_plots, save_images, label_data):
    # saving nothing
    if save_images and not show_index_plots:
        print("index plots will not be shown, and hence not saved")
        valid_answer = False
        while not valid_answer:
            answer = input("do you want to save index plots? ")
            if "yes" in answer or "no" in answer:
                valid_answer = True
            else:
                print("please only answer 'yes' or 'no'")
                answer = input("do you want to save index plots? ")
        if "yes" in answer:
            show_index_plots = True
            save_images = True
    
    # computing high-res but not outputting
    if high_res and not show_index_plots and not label_data:
        print("please note that high-res images will be used, but they will "
              "not be displayed in any way")
        valid_answer = False
        while not valid_answer:
            answer = input("do you want to switch to high-res mode? ")
            if "yes" in answer or "no" in answer:
                valid_answer = True
            else:
                print("please only answer 'yes' or 'no'")
                answer = input("do you want to switch to high-res mode? ")
        if "yes" in answer:
            print("ok")
    return high_res, show_index_plots, save_images, label_data

from matplotlib import pyplot as plt
from data_handling import check_duplicate_name
def save_image_file(data, image_name, normalise):
    if normalise:
        cmap = plt.get_cmap("viridis")
        
        valid_chunks = [chunk for chunk in data if not np.isnan(chunk).all()]
        global_min = min(np.nanmin(chunk) for chunk in valid_chunks)
        global_max = 0.8*max(np.nanmax(chunk) for chunk in valid_chunks)
        norm = plt.Normalize(global_min, global_max)
        
        data = cmap(norm(data))
        data = (255 * data).astype(np.uint8)
    
    # check for duplicate file name (prevent overwriting)
    matches = check_duplicate_name(search_dir=os.getcwd(), 
                                   file_name=image_name)
    while matches:
        print(f"found duplicate {image_name} file in:")
        for path in matches:
            print(" -", path)
        ans = input("would you like to overwrite? ")
        valid_ans = False
        while not valid_ans:
            if "yes" in ans:
                valid_ans = True
                matches = []
                Image.fromarray(data).save(image_name)
            if "no" in ans:
                valid_ans = True
                print("you can rename the file and retry or skip this file")
                input("type 'retry' to scan again, 'skip' to skip this file")
                valid_ans2 = False
                while not valid_ans2:
                    if "retry" in ans:
                        matches = check_duplicate_name(search_dir=os.getcwd(), 
                                                       file_name=image_name)
