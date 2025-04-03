import numpy as np

def get_sentinel_bands(sentinel_n, high_res):
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
