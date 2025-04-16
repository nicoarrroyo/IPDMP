import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors

def image_to_array(file_path_s):
    """
    Convert an image or list of images to a numpy array. The image is opened 
    temporarily but not opened permanently. Note: the conversion of the image 
    to a numpy array forces its contents into the numpy.uint16 type, which 
    causes overflow errors, which then causes the index calculation to break. 
    To fix this, convert the uint16 arrays must be converted to integer type 
    when calculating the indices. 
    
    Parameters
    ----------
    file_path_s : list or string
        A list containing all the file paths 
        
    Returns
    -------
    image_arrays : list of numpy arrays
        A list containing some number of numpy arrays converted from images. 
        
    """
    if not isinstance(file_path_s, list):
        with Image.open(file_path_s) as img:
            image_array = np.array(img)
        return image_array
    else:
        image_arrays = []
        for file_path in file_path_s:
            with Image.open(file_path) as img:
                image_arrays.append(np.array(img))
        return image_arrays

def upscale_image_array(img_array, factor=2):
    """
    An image, for example the band image for SWIR1 or SWIR2 may be of lower 
    resolution that others to which it is being compared, for example the band 
    images for blue, green, and NIR, so it must be scaled up to match their 
    pixel-count. 
    
    Parameters
    ----------
    img_array : numpy array
        Numpy array containing data about an image. 
    factor : int, optional
        The default is 2. This upscales the image from 10m to 20m. 
    
    Returns
    -------
    img_array : numpy array
        The 20m resolution image array is upscaled to match the 10m reoslution.
    
    """
    return np.repeat(np.repeat(img_array, factor, axis=0), factor, axis=1)

def mask_sentinel(path, high_res, image_arrays):
    """
    Start by opening the cloud probability file from Sentinel 2 imagery data 
    and converting this image into an array. Turn every pixel that is more 
    than 50% likely to be a cloud into a 100% likelihood cloud, store the 
    positions of those clouds and "mask out" the corresponding pixels in the 
    band image arrays by setting those pixel values to not-a-number. This step 
    should be done before the calculation of the water indices so that the 
    index arrays are not calculating with cloud pixels. 
    
    Parameters
    ----------
    path : string
        The file path to the cloud probability file in Sentinel 2 imagery. 
    high_res : bool
        The True/False variable to check which resolution of cloud probability 
        file is needed. This resolution can be either 10m (which is when 
        high_res is set to true), 20m (also means high_res is set to True but 
        some images only have 20m resolution e.g. SWIR1 and SWIR2) or 60m 
        (which is the case when high_res is set to False). 
    image_arrays : list of numpy arrays
        A list containing some number of numpy arrays converted from images. 
    
    Returns
    -------
    image_arrays : list of numpy arrays
        A list containing some number of numpy arrays converted from images. 
        This list is also adjusted in that it contains the upscaled band images
        for SWIR1 and SWIR2. 
    
    """
    if high_res:
        image_arrays[-1] = upscale_image_array(image_arrays[-1], factor=2)
        image_arrays[-2] = upscale_image_array(image_arrays[-2], factor=2)
        path = path + "MSK_CLDPRB_20m.jp2"
        clouds_array = image_to_array(path)
        clouds_array = upscale_image_array(clouds_array, factor=2)
    else:
        path = path + "MSK_CLDPRB_60m.jp2"
        clouds_array = image_to_array(path)
    
    clouds_array = np.where(clouds_array > 50, 100, clouds_array)
    cloud_positions = np.argwhere(clouds_array == 100)
    
    for image_array in image_arrays:
        image_array[cloud_positions[:, 0], cloud_positions[:, 1]] = 0
    
    return image_arrays

def plot_indices(data, sat_n, size, dpi, save_image, res):
    """
    Take a list of indices and plot them for the user's viewing pleasure. 
    Other than being nice pictures to look at, there isn't that much use to 
    the images themselves, but the index arrays are used for labelling. 
    
    Parameters
    ----------
    data : list of numpy arrays
        A list containing some number of numpy arrays converted from images. 
        In this case, these arrays contain index values to be plotted. 
    sat_n : int
        The satellite number to be used as a part of the plot and file titles.
    size : tuple
        The required size of the image plots.
    dpi : int
        Dots-per-inch to which the image must be plotted. A higher value is 
        more intensive but provides clearer images. 
    save_image : bool
        Boolean variable to check if the user wants the image saved.
    res : string
        The resolution of the image array being passed and plotted. This can 
        be 10m, 20m, or 60m for Sentinel 2. 
    
    Returns
    -------
    None.
    
    """
    indices = ["NDWI", "MNDWI", "AWEI-SH", "AWEI-NSH"]
    for i, water_index in enumerate(data):
        plt.figure(figsize=(size))
        if sat_n != 2:
            sat_name = "Landsat"
            sat_letter = "L"
        else:
            sat_name = "Sentinel"
            sat_letter = "S"
        if i == 0:
            plt.title(f"{sat_name} {sat_n} {indices[i]} DPI{dpi} R{res}", 
                      fontsize=8)
            
            ax = plt.gca()
            plt.imshow(water_index)
            
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(left=False, bottom=False, 
                           labelleft=False, labelbottom=False)
        
        if save_image:
            print(f"saving {indices[i]} image", end="... ")
            plot_name = f"{sat_letter}{sat_n}_{indices[i]}_DPI{dpi}_R{res}.png"
            
            # check for file name already existing and increment file name
            base_name, extension = os.path.splitext(plot_name)
            counter = 1
            while os.path.exists(plot_name):
                plot_name = f"{base_name}_{counter}{extension}"
                counter += 1
            
            plt.savefig(plot_name, dpi=dpi, bbox_inches="tight")
            print(f"complete! saved as {plot_name}")
        
        print(f"displaying {indices[i]} image", end="... ")
        plt.show()
        print(f"{indices[i]} image display complete!")

def plot_chunks(ndwi, mndwi, index_chunks, plot_size_chunks, i, title_size, 
                label_size, tci_chunks, tci_60_array):
    """
    Plots image chunks of calculated indices (NDWI and MNDWI) and True Color 
    Images (TCI).
    
    This function visualizes the processed image data, displaying index chunks 
    and corresponding TCI chunks alongside a full-resolution TCI image with 
    the chunk's location highlighted.
    
    Parameters
    ----------
    ndwi : numpy.ndarray
        The numpy array with all calculated pixel values of NDWI in a given 
        satellite image.
    mndwi : numpy.ndarray
        The numpy array with all calculated pixel values of NDWI in a given 
        satellite image.
    index_chunks : list of numpy.ndarray
        A list containing numpy arrays representing the calculated index 
        chunks (NDWI, MNDWI, AWEI-SH, AWEI-NSH). Each element of the list is a 
        3D numpy array, where the first dimension corresponds to the chunk 
        index, and the last two dimensions are the spatial dimensions of the 
        chunk.
    plot_size_chunks : tuple of int
        The size of the plot figure (width, height) for the subplots.
    i : int
        The index of the specific chunk to be plotted.
    title_size : int
        The size of the title in each plot.
    label_size : int
        The size of the labels (i.e. the axes labels) in each plot.
    tci_chunks : numpy.ndarray
        A 3D numpy array representing the True Color Image (TCI) chunks. The 
        first dimension corresponds to thechunk index, and the last two 
        dimensions are the spatial dimensions of the chunk.
    tci_60_array : numpy.ndarray
        A numpy array representing the full-resolution (60m) True Color 
        Image (TCI).
        
    Returns
    -------
    None
        This function displays plots and prints maximum index values; it does 
        not return any values.
    
    """
    index_labels = ["NDWI", "MNDWI"]
    norm_ndwi = colors.Normalize(vmin=np.nanmin(ndwi), 
                                 vmax=np.nanmax(ndwi)*0.8)
    norm_mndwi = colors.Normalize(vmin=np.nanmin(mndwi), 
                                 vmax=np.nanmax(mndwi)*0.8)
    
    fig, axes = plt.subplots(2, 2, figsize=plot_size_chunks)
    # plot 1, top left: NDWI chunk (full resolution)
    axes[0][0].imshow(index_chunks[0][i], norm=norm_ndwi)
    axes[0][0].set_title(f"{index_labels[0]} Chunk {i}", 
                         fontsize=title_size)
    axes[0][0].tick_params(axis="both", labelsize=label_size)
    
    # plot 2, top right: MNDWI chunk (merged resolution)
    axes[0][1].imshow(index_chunks[1][i], norm=norm_mndwi)
    axes[0][1].set_title(f"{index_labels[1]} Chunk {i}", 
                         fontsize=title_size)
    axes[0][1].tick_params(axis="both", labelsize=label_size)
    
    # plot 3, bottom left: TCI chunk (full resolution)
    axes[1][0].imshow(tci_chunks[i])
    axes[1][0].set_title(f"TCI Chunk {i}", fontsize=title_size)
    axes[1][0].tick_params(axis="both", labelsize=label_size)
    
    # plot 4, bottom right: tracker TCI (60m resolution)
    axes[1][1].imshow(tci_60_array)
    axes[1][1].set_title("Tracker TCI", fontsize=title_size)
    axes[1][1].axis("on")
    
    # calculate chunk geometry
    chunks_per_side = int(np.sqrt(len(tci_chunks)))
    chunk_col = i % chunks_per_side
    chunk_row = i // chunks_per_side
    axes[1][1].text(0.5, 0.95, f"COL {chunk_col} ROW {chunk_row}", 
                    transform=axes[1][1].transAxes, ha="center", 
                    va="center", fontsize=label_size+1, color="yellow")
    
    # calculate dimensions in the 60m array
    side_length = tci_60_array.shape[0] # assuming square image
    chunk_length = side_length / chunks_per_side
    chunk_ulx = chunk_col * chunk_length
    chunk_uly = chunk_row * chunk_length
    
    # axes on TCI "tracker" image are "number of chunks"
    axes[1][1].set_xticks(np.linspace(0, side_length, 8))
    axes[1][1].set_yticks(np.linspace(0, side_length, 8))
    axes_tick_labels = np.linspace(0, chunks_per_side, 8).astype(int)
    axes[1][1].set_xticklabels(axes_tick_labels, fontsize=label_size)
    axes[1][1].set_yticklabels(axes_tick_labels, fontsize=label_size)
    axes[1][1].set_xlabel("Chunk Column", fontsize=label_size+1)
    axes[1][1].set_ylabel("Chunk Row", fontsize=label_size+1)
    
    # draw a red square around the current chunk
    tci_tracker_square = plt.Rectangle((chunk_ulx, chunk_uly), 
                                   chunk_length, chunk_length, 
                                   linewidth=1, edgecolor="r", 
                                   facecolor=None)
    axes[1][1].add_patch(tci_tracker_square)
    
    plt.tight_layout()
    plt.show()
