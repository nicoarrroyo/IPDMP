import numpy as np
import random
import math
import sys

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
            overlap_x1 = random.randint(0, 2)
            overlap_y1 = random.randint(0, 2)
            overlap_x2 = random.randint(0, 2)
            overlap_y2 = random.randint(0, 2)
            
            sub_ulx = ulx + j * sub_width - overlap_x1
            sub_uly = uly + i * sub_height - overlap_y1
            sub_lrx = ulx + (j + 1) * sub_width + overlap_x2
            sub_lry = uly + (i + 1) * sub_height + overlap_y2
            
            # Ensure the sub-box coords are in the bounds of the original box
            sub_ulx = max(sub_ulx, ulx)
            sub_uly = max(sub_uly, uly)
            sub_lrx = min(sub_lrx, lrx)
            sub_lry = min(sub_lry, lry)
            
            # handle edge case
            if sub_lrx <= sub_ulx:
                sub_lrx = sub_ulx + 1
            if sub_lry <= sub_uly:
                sub_lry = sub_uly + 1
            
            sub_boxes.append([sub_ulx, sub_uly, sub_lrx, sub_lry])
    return sub_boxes

def convert_seconds_to_hms(total_seconds):
    """
    Converts a duration given in total seconds into hours, minutes,
    and remaining seconds.
    
    Args:
      total_seconds: The total duration in seconds (can be an integer or float).
                     Negative inputs are not recommended and may raise an error
                     or be treated as positive depending on desired behavior.
    
    Returns:
      A tuple containing three integers: (hours, minutes, seconds).
      Raises TypeError if input is not numeric.
      Raises ValueError if input is negative.
    """
    if not isinstance(total_seconds, (int, float)):
        raise TypeError("Input 'total_seconds' must be a number.")
    
    if total_seconds < 0:
        # Option 1: Raise an error for negative time
        raise ValueError("Input 'total_seconds' cannot be negative.")
        # Option 2: Handle negative time, e.g., by taking absolute value
        # print("Warning: Input duration is negative. Using absolute value.")
        # total_seconds = abs(total_seconds)
    
    # Use math.floor to handle potential float inputs correctly before 
    # integer division
    # Ensures we only deal with whole seconds for the division/modulo logic
    total_seconds_int = math.floor(total_seconds)
    
    # Use divmod which efficiently returns quotient and remainder
    # First, get total minutes and leftover seconds
    minutes_total, seconds = divmod(total_seconds_int, 60)
    
    # Next, get total hours and leftover minutes from the total minutes
    hours, minutes = divmod(minutes_total, 60)
    
    # Return the results as integers
    return int(hours), int(minutes), int(seconds)

def confirm_continue_or_exit():
    """
    Asks the user if they want to continue with the program.
    
    If the user enters 'y' or 'yes', the function returns and the script 
    continues. If the user enters 'n' or 'no', the function prints a message 
    and exits the script. It will keep asking until a valid 
    input ('y', 'yes', 'n', 'no') is given.
    """
    while True: # Loop until valid input is received
        response = input("Do you want to continue? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            print("off we go - continuing program")
            return # Exit the function and let the main script proceed
        
        elif response in ['n', 'no']:
            print("will not continue - exiting program")
            sys.exit() # Stop the script immediately
        else:
            # Ask again if the input was invalid
            print("invalid input. 'y' for yes or 'n' for no.")

"""
This section is storage for functions that are not currently used in the IPDMP 
program but may be useful in future. 
"""
from PIL import Image
import os

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
