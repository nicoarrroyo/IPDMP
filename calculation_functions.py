import numpy as np

def get_indices(blue, green, nir, swir1, swir2):
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.where((green + nir) == 0, -1, -(green - nir) / (green + nir))
    mndwi = np.where((green + swir2) == 0, -1, -(green - swir1) / (green + swir1))
    awei_sh = -((blue + (2.5 * green) - (1.5 * (nir + swir1)) - (0.25 * swir2))/(2**16))
    awei_nsh = -(((4 * (green - swir1)) - (0.25 * nir - 2.75 * swir2))/(2**16))
    return ndwi, mndwi, awei_sh, awei_nsh
