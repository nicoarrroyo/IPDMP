import numpy as np

def get_indices(blue, green, nir, swir1, swir2):
    np.seterr(divide="ignore", invalid="ignore")
    
    ndwi = ((green - nir) / (green + nir))
    mndwi = ((green - swir1) / (green + swir1))
    awei_sh = (green + 2.5 * blue - 1.5 * (nir + swir1) - 0.25 * swir2)
    awei_nsh = (4 * (green - swir1) - (0.25 * nir + 2.75 * swir2))
    indices = [ndwi, mndwi, awei_sh, awei_nsh]
    return indices
