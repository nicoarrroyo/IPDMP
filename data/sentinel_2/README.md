# Sentinel 2 Directory
This section will describe the Sentinel 2 folder naming structure and how / where to put your downloaded Sentinel 2 images for IPDMP to run as intended. Example folder / file names will be added throughout this description like so: 

> `EXAMPLE FOLDER`

or like so:

> `EXAMPLE_FILE.tif`

## Sentinel 2 Folder Naming
Upon downloading a selected image from the [Copernicus Browser](https://browser.dataspace.copernicus.eu/?zoom=5&lat=50.16282&lng=20.78613&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE) the file on your computer should be roughly 1 GB in size and it should be packaged as a `.zip` folder. This folder should be extracted and the target extraction location should be THIS directory (the one you're in right now!), which is where IPDMP will search for the images. Once this extraction is complete, a folder will be in this directory and it will look something like this:

> `S2C_MSIL2A_20250331T110651_N0511_R137_T31UCU_20250331T143812.SAFE`
