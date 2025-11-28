# Data Directory
This directory contains all input and output data for the machine learning pipeline. There should be three folders in this directory. These folders are the place where you (the user) will place the data that the program will then use to train the ROI recognition model, deploy the model, or aid in data labelling. This README explains the purpose and expected data format for each of these folders. 

## `masks` (input)
This folder contains several sub-folders. Each sub-folder will expect a type of file that the program will automatically search for and use for masking. For more information, there is a README file in there as well.

## `saved_models` (output)


## `sentinel_2` (input)


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
├── labels/                  # Generated labels (generated)
└── models/                  # Trained model outputs (generated)

```

## Input Data Requirements

### `masks/`
Place your boundary/annotation shapefiles here.
- **Format**: Shapefile (.shp + .shx + .dbf + .prj)
- **Naming**: Should match corresponding satellite image region
- **Example**: `region_2024_boundaries.shp`

### `sentinel_2/`
Place your satellite imagery files here.
- **Format**: GeoTIFF (.tif)
- **Naming**: Standard Archive Format for Europe (SAFE) format specification (see ESA SentiWiki)
- **Example**: `region_2024_satellite.tif`

## Output Directories

The following directories will be populated automatically during processing:

### `processed/`
Preprocessed and tiled satellite imagery ready for training.

### `labels/`
Generated labels extracted from shapefiles and matched to image tiles.

### `models/`
Trained model checkpoints and final model weights.

## Quick Start

1. Add your satellite image(s) to `satellite_images/`
2. Add corresponding shapefile(s) to `shapefiles/`
3. Run the preprocessing script: `python scripts/preprocess.py`
4. Begin training: `python scripts/train.py`

## Notes

- Ensure shapefiles and satellite images use the same coordinate reference system (CRS)
- Large files (>100MB) should be tracked with Git LFS or excluded via `.gitignore`
