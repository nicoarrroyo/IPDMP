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

import matplotlib.pyplot as plt
def plot_chunks(plot_size_chunks, index_chunks, tci_chunks, i, tci_60_array, c, 
                side_length):
    """
    Plots image chunks of calculated indices (NDWI, MNDWI, AWEI-SH, AWEI-NSH) 
    and True Color Images (TCI).
    
    This function visualizes the processed image data, displaying index chunks 
    and corresponding TCI chunks alongside a full-resolution TCI image with 
    the chunk's location highlighted.
    
    Parameters
    ----------
    plot_size_chunks : tuple of int
        The size of the plot figure (width, height) for the subplots.
    index_chunks : list of numpy.ndarray
        A list containing numpy arrays representing the calculated index 
        chunks (NDWI, MNDWI, AWEI-SH, AWEI-NSH). Each element of the list is a 
        3D numpy array, where the first dimension corresponds to the chunk 
        index, and the last two dimensions are the spatial dimensions of the 
        chunk.
    tci_chunks : numpy.ndarray
        A 3D numpy array representing the True Color Image (TCI) chunks. The 
        first dimension corresponds to thechunk index, and the last two 
        dimensions are the spatial dimensions of the chunk.
    i : int
        The index of the specific chunk to be plotted.
    tci_60_array : numpy.ndarray
        A numpy array representing the full-resolution (60m) True Color Image (TCI).
    c : int
        A counter or identifier used in the title of the full-resolution TCI plot.
    side_length : int
        The side length of the original full-resolution image, used to 
        calculate chunk positions.
        
    Returns
    -------
    None
        This function displays plots and prints maximum index values; it does 
        not return any values.
    """
    index_labels = ["NDWI", "MNDWI", "AWEI-SH", "AWEI-NSH"]
    # plot index chunks
    max_indices_chunk = np.zeros(len(index_labels))
    fig, axes = plt.subplots(1, len(index_labels), figsize=plot_size_chunks)
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
    
    chunk_length = side_length / np.sqrt(len(tci_chunks))
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
