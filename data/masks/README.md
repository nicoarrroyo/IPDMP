# Masks Directory
This directory is built to contain all the masking information requested by NALIRA (data-labelling portion of IPDMP). The example file names shown below are the exact files used in the development of the program. Currently, these are the only files compatible with NALIRA, however in future, this will be more flexible. 

## Boundaries
- **File Source**: [UK GOV](https://www.data.gov.uk/dataset/2e17269d-10b9-4e43-b67b-57f9b02bd0f8/countries-december-2021-boundaries-uk-buc)
- **File Type**: Shapefile (.shp) in a folder
- **Folder Name**:
To restrict study to one country, we can This has the significant added benefit of masking out the sea as well. The sea can make up a large part of a satellite image, and since often has clouds over it as well, small parts of the sea can often be confused with reservoirs. 

## Known Reservoirs
- **File Source**: [UK GOV](https://www.data.gov.uk/dataset/aa1e16e8-eded-4a60-8d1d-0df920c319b6/inventory-of-reservoirs-amounting-to-90-of-total-uk-storage)
- **File Type**: PLACEHOLDER_FILE_TYPE in a folder
- **Folder Name**: 

## Rivers and Streams
- **File Source**: [Ordnance Survey](https://www.ordnancesurvey.co.uk/products/os-open-rivers)
- **File Type**: PLACEHOLDER_FILE_TYPE in a folder
- **Folder Name**: 

## Urban Areas
- **File Source**: [UK CEH](https://catalogue.ceh.ac.uk/documents/5af9e97d-9f33-495d-8323-e57734388533?_gl=1*4k1tkl*_ga*NTgzNDcxODg0LjE3NTUwOTY2NjE.*_ga_27CMQ4NHKV*czE3NTUwOTY2NjEkbzEkZzAkdDE3NTUwOTY2NjEkajYwJGwwJGgw)
- **File Type**: PLACEHOLDER_FILE_TYPE in a folder
- **Folder Name**: 

## Terrain (NOT IMPLEMENTED YET)
- **File Source**: [UK GOV](https://environment.data.gov.uk/dataset/13787b9a-26a4-4775-8523-806d13af58fc) or [Ordnance Survey](https://www.ordnancesurvey.co.uk/products/os-terrain-50#get)
- **File Type**: PLACEHOLDER_FILE_TYPE in a subfolder in a folder
- **Folder Name**: NOT IMPLEMENTED YET

## Special Case (Clouds)
Cloud masking is properly implemented in IPDMP, however there is no need for the user to place any specific files in any specific directory. IPDMP uses the [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) Python library (can be installed and found hosted on GitHub), which is a deep learning solution to cloud and cloud shadow segmentation.
