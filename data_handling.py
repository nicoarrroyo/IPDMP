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

def extract_coords(coord_string):
    """
    Extracts coordinates from a string (including square brackets) and returns 
    them as a list of floats.
    
    Args:
      coord_string (str): A string containing coordinates within square 
      brackets, separated by spaces.
      
    Returns:
      list: A list of floats representing the coordinates.
    """
    try:
        # Remove square brackets, split the string into numeric strings
        coord_string = coord_string.replace('[', '').replace(']', '')
        coord_strings = coord_string.split()
        
        # Convert each numeric string to a float
        coordinates = [float(coord) for coord in coord_strings]
    except:
        coordinates = []
    return coordinates

def change_to_folder(folder_path):
    if os.path.exists(folder_path):
        os.chdir(folder_path)
    else:
        os.makedirs(folder_path)
        os.chdir(folder_path)
        path_name = folder_path.split("\\")
        print(f"created folder named '{path_name[-1]}' in '{path_name[-2]}'")

def check_duplicate_name(search_dir, file_name):
    duplicates = False
    for files in os.listdir(search_dir):
        if file_name in files:
            duplicates = True
    return duplicates
