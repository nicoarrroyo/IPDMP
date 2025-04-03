import csv
import os

def rewrite(write_file, rows):
    """
    Used to remove any blank or blatantly invalid entries in a csv
    
    Parameters
    ----------
    write_file : file
        A pre-opened file to which the program rewrites any values.
    rows : list
        A list containing every row in a csv file.
    
    Returns
    -------
    None.
    
    """
    for j in range(len(rows)):
        entry = f"{rows[j][0]},{rows[j][1]}"
        for k in range(2, len(rows[j])): # add coordinates
            entry = f"{entry},{rows[j][k]}"
        write_file.write(f"{entry}\n")

def blank_entry_check(file):
    """
    Check a formatted csv file for blank entries when the expected format is 
    one entry after the other. Blank entries are removed.
    
    Parameters
    ----------
    file : string
        The file name of the file which is having its rows checked for blank 
        entries. This file is opened locally, then the rows are checked. 
    
    Returns
    -------
    None.
    
    """
    check_file_permission(file_name=file)
    cleaned_rows = []
    invalid_rows = []
    with open(file, mode="r") as re: # read the file once
        rows = csv.reader(re)
        for i, row in enumerate(rows):
            if row and row[0].isdigit() or row and "chunk" in row[0]:
                cleaned_rows.append(row) # only keep valid rows
            else:
                invalid_rows.append(i)
    
    if not invalid_rows:
        pass
    else:
        with open(file, mode="w", newline="") as wr: # write cleaned rows back
            csv_writer = csv.writer(wr)
            csv_writer.writerows(cleaned_rows)
        # print(f"{len(invalid_rows)} invalid entries were removed on", invalid_rows)

def check_file_permission(file_name):
    """
    A useful function to make sure a file is not being used / is open on the 
    computer already before accessing it. If it is open, the user can simply 
    close the file and press enter to retry. This function is necessary to 
    call several times throughout the IPDGS program as it avoids crashing the 
    program due to denied permission errors that occur as a result of trying 
    to open a file that's already in use. 
    
    Parameters
    ----------
    file_name : string
        The name of the file being checked.

    Returns
    -------
    None.

    """
    while True:
        try: # check if file is open
            with open(file_name, mode="a"):
                break
        except IOError:
            print("could not open file - please close the responses file")
            input("press enter to retry")

def find_rgb_file(path):
    """
    This searches a directory completely for a file that contains the phrase 
    "RGB", and "10m" for the resolution, and "bright" as the custom RGB 
    composite is too dark. The "bright" is added manually after aritifically 
    brightening the file in the GNU Image Manipulation Program. Again, this 
    function is quite outdated and is no longer used as the 
    true-colour image (TCI) is much more usable. 
    
    Parameters
    ----------
    path : string
        Any file path that can be searched fully to find a an RGB image file. 
    
    Returns
    -------
    bool
        A bool variable to indicate whether an RGB image file has been found. 
    rgb_path or full_path : string
        If the RGB image file is found, this is the path to that file. 
    
    """
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path): # if item is a folder
            found_rgb, rgb_path = find_rgb_file(full_path)
            if found_rgb:  
                return True, rgb_path
        else: # if item is a file
            if "RGB" in item and "10m" in item and "bright" in item:
                return True, full_path
    return False, None