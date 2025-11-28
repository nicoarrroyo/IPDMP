# Data Directory
This directory contains all input and output data for the machine learning pipeline. There should be three folders in this directory. These folders are the place where you (the user) will place the data that the program will then use to train the ROI recognition model, deploy the model, or aid in data labelling. This README explains the purpose and expected data format for each of these folders. 

## Directory Overview
```
data/
├── masks/                   # Mask shapefiles organized by category
│   ├── boundaries/          # Boundary masks
│   ├── known_reservoirs/    # Known reservoir masks
│   ├── rivers/              # River masks
│   ├── terrain/             # Terrain masks
│   └── urban_areas/         # Urban area masks
├── saved_models/            # Boundary shapefiles for labeling
├── sentinel_2/              # Preprocessed data (generated)
```

## `masks` (input data)
This folder contains several sub-folders. Each sub-folder will expect a type of file that the program will automatically search for and use for masking. For more information, visit the README file in the `masks` directory.
- **Format**: Shapefile (.shp + .shx + .dbf + .prj)
- **Naming**: Should match corresponding satellite image region
- **Example**: `region_2024_boundaries.shp`

## `saved_models` (output)
This is an output data folder, meaning you don't need to put anything in here yourself. Any models trained using the `KRISP_trainer` script will be automatically saved here. Then, when the model is deployed with the `KRISP-Y` script, it will be searched for here as well. Models will be saved as TensorFlow Keras models (e.g. `ndwi model epochs-100.keras`). For more information, visit the README file in the `saved_models` directory.

## `sentinel_2` (input)
This folder expects Sentinel 2 satellite image folders, which can be downloaded from the Copernicus Browser as .zip folders. Simply unzip the folder here and the program should find everything automatically. Please take care to not change any of the names of the files or sub-folders inside the image folder, as the program uses specific, standardised naming rules to find the necessary files. For more information, visit the README file in the `sentinel_2` directory.
- **File Format**: File: GeoTIFF (.tif)
- **Folder Format**: Standard Archive Format for Europe (SAFE) format specification (see ESA SentiWiki)
- **Naming**: Standardised ESA Sentinel folder naming scheme.
- **Example**: `S2B_MSIL2A_20240719T110619_N0510_R137_T31UCU_20240719T142134.SAFE`

## Notes
- hi :)
