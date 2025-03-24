import numpy as np

def get_indices(blue, green, nir, swir1, swir2):
    np.seterr(divide="ignore", invalid="ignore")
    
    ndwi = ((green - nir) / (green + nir)) # FLAG bad values
    print(".", end="")
    mndwi = ((green - swir1) / (green + swir1)) # FLAG bad values
    print(".", end="")
    awei_sh = (green + 2.5 * blue - 1.5 * (nir + swir1) - 0.25 * swir2)
    print(".", end="")
    awei_nsh = (4 * (green - swir1) - (0.25 * nir + 2.75 * swir2))
    print(".", end="")
    indices = [ndwi, mndwi, awei_sh, awei_nsh]
    print(" complete!")
    return indices
