def get_landsat_bands(landsat_n):
    if landsat_n == 7:
        BLUE_BAND = '1'
        GREEN_BAND = '2'
        NIR_BAND = '4'
        SWIR1_BAND = '5'
        MIR_BAND = '7'
        return BLUE_BAND, GREEN_BAND, NIR_BAND, SWIR1_BAND, MIR_BAND
    else:
        BLUE_BAND = '2'
        GREEN_BAND = '3'
        NIR_BAND = '5'
        SWIR1_BAND = '6'
        SWIR2_BAND = '7'
        return BLUE_BAND, GREEN_BAND, NIR_BAND, SWIR1_BAND, SWIR2_BAND

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
