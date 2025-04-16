import numpy as np

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

def combine_sort_unique(array1, array2):
    """
    Combines two arrays, sorts the combined array in ascending order, and 
    eliminates duplicates.
    
    Parameters
    ----------
    array1 : list
        The first input array.
    array2 : list
        The second input array
    
    Returns
    -------
    sorted_array : list
        The sorted array with unique elements.
        
    """
    combined_array = array1 + array2
    unique_array = list(set(combined_array))
    sorted_array = sorted(unique_array)
    return sorted_array

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
