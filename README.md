# Individual Project Data Generation Software (IPDGS)

## Description

The Individual Project Data Generation Software (IPDGS) is a Python package designed for processing satellite imagery, specifically Sentinel-2 data, to calculate water indices and facilitate the generation of labelled data for machine learning applications. This tool enables users to:

* Calculate key water indices: NDWI, MNDWI, AWEI-SH, and AWEI-NSH.
* Perform cloud masking to improve the accuracy of water index calculations.
* Visually inspect and label regions of interest (ROIs) within satellite images.
* Process Sentinel-2 imagery at different resolutions.

## Features

* **Modular Design:** Well-organized modules for satellite band selection, index calculations, image manipulation, and more.
* **Water Index Calculation:** Efficient calculation of essential water indices using NumPy.
* **Cloud Masking:** Automated cloud masking to ensure accurate water analysis.
* **Interactive Labelling:** A graphical user interface (Tkinter) for easy region of interest (ROI) selection and labelling.
* **Sentinel-2 Support:** Specifically designed to process Sentinel-2 imagery with handling for different band resolutions.
* **Flexible Processing:** Options for high/low resolution processing and saving/displaying images.
* **Error Handling:** Robust error handling to manage file operations and user inputs.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install numpy matplotlib Pillow tkinter
    ```

    * Ensure you have Python 3.x installed.

## Usage

1.  **Prepare your Sentinel-2 data:**
    * Download Sentinel-2 imagery and organize it into the required directory structure.  The code expects a specific folder structure within the main directory (see `IPDGS.py` for details on the expected file paths).
2.  **Configure settings (Optional):**
    * Modify the variables at the beginning of `IPDGS.py` to adjust processing parameters such as:
        * `high_res`:  Process at 10m resolution (True) or 60m resolution (False).
        * `save_images`:  Save output images (True) or not (False).
        * `show_index_plots`: Display water index plots (True) or not (False).
        * `label_data`:  Enable the data labelling tool (True) or not (False).
        * `n_chunks`: Number of chunks to split the image into for labelling.
        * `HOME`:  The main directory where the Sentinel-2 data is located.  You will need to modify this variable to point to the correct path on your system.
3.  **Run the `IPDGS.py` script:**

    ```bash
    python IPDGS.py
    ```

4.  **Data Labelling (if enabled):**

    * If `label_data` is set to `True`, the script will guide you through the process of labelling regions of interest (reservoirs and other water bodies) in the imagery.
    * Follow the on-screen prompts to draw rectangles around the features.
    * The labelling results are saved to a CSV file (`responses_<n_chunks>_chunks.csv`).

## Code Structure

* `IPDGS.py`: The main script that orchestrates the image processing and labelling workflow.
* `calculation_functions.py`:  Contains functions for calculating water indices (NDWI, MNDWI, AWEI).
* `image_functions.py`:  Provides functions for image manipulation (opening, masking, plotting, ROI selection).
* `misc_functions.py`:  Includes miscellaneous utility functions (printing tables, array splitting, file handling, spinner).
* `satellite_functions.py`:  Handles satellite band selection for Sentinel-2.

## Contributing

1.  **Fork the repository.**
2.  **Create a new branch for your feature or bug fix.**
3.  **Make your changes and commit them.**
4.  **Push your changes to your fork.**
5.  **Submit a pull request.**

## License

[Specify the license you are using, e.g., MIT License]

## Acknowledgements

* [Any acknowledgements to libraries, datasets, or individuals]

## Contact

* [Your Name/Contact Information]
