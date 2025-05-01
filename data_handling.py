import csv
import os
import re

def rewrite(write_file, rows):
    """
    Used to rewrite the lines in a pre-opened file with a given list of "rows"
    
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
            if row and row[0].isdigit() or row and "chunk" in row[0] or i < 2:
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

def create_box(coords):
    """
    Creates a square bounding box containing the input coordinates,
    adjusted to stay within the 0-157 boundary.
    
    Args:
      coords: A list of four floats representing the input rectangle's
              coordinates in the format [ULX, ULY, LRX, LRY]
              (Upper Left X, Upper Left Y, Lower Right X, Lower Right Y).
    
    Returns:
      A list of four floats representing the coordinates of the  bounding box 
      in the format [ULX, ULY, LRX, LRY].
    """
    # --- Constants ---
    MIN_COORD = 0.0
    MAX_COORD = 157.0
    BOX_SIZE = MAX_COORD / 5
    
    # --- Input Validation ---
    if len(coords) != 4:
        raise ValueError("Input coords list must contain exactly four values.")
    ulx, uly, lrx, lry = coords
    if not all(isinstance(c, (int, float)) for c in coords):
        raise TypeError("All coordinates must be numbers.")
    if ulx >= lrx or uly >= lry:
        raise ValueError("Invalid coordinates: ULX must be < LRX and ULY must "
                         "be < LRY.")
    if not all(MIN_COORD <= c <= MAX_COORD for c in coords):
        print(f"Warning: Input coordinates {coords} contain values outside the "
              f"{MIN_COORD}-{MAX_COORD} range.")
        # Depending on requirements, you might raise an error here instead
        # Or clamp the input coordinates first
    
    # --- 1. Calculate Center of the input rectangle ---
    center_x = (ulx + lrx) / 2.0
    center_y = (uly + lry) / 2.0
    
    # --- 2. Create Initial Box centered around the input rectangle ---
    # Half the size of the desired box
    half_size = BOX_SIZE / 2.0
    box_ulx = center_x - half_size
    box_uly = center_y - half_size
    box_lrx = center_x + half_size
    box_lry = center_y + half_size
    
    # --- 3. Adjust Box to stay within bounds [MIN_COORD, MAX_COORD] ---
    # Adjust right edge if it exceeds MAX_COORD
    if box_lrx > MAX_COORD:
      offset = box_lrx - MAX_COORD
      box_lrx = MAX_COORD
      box_ulx -= offset # Shift left edge by the same amount
    
    # Adjust bottom edge if it exceeds MAX_COORD
    if box_lry > MAX_COORD:
      offset = box_lry - MAX_COORD
      box_lry = MAX_COORD
      box_uly -= offset # Shift top edge by the same amount
    
    # Adjust left edge if it's less than MIN_COORD
    # This needs to happen *after* adjusting the right edge in case shifting 
    # left pushed it below MIN_COORD
    if box_ulx < MIN_COORD:
      offset = MIN_COORD - box_ulx
      box_ulx = MIN_COORD
      box_lrx += offset # Shift right edge by the same amount
    
    # Adjust top edge if it's less than MIN_COORD
    # This needs to happen *after* adjusting the bottom edge
    if box_uly < MIN_COORD:
        offset = MIN_COORD - box_uly
        box_uly = MIN_COORD
        box_lry += offset # Shift bottom edge by the same amount
    
    # --- Ensure the box is exactly a quarter of MAX_COORD after adjustments ---
    final_box_ulx = max(MIN_COORD, box_ulx)
    final_box_uly = max(MIN_COORD, box_uly)
    final_box_lrx = final_box_ulx + BOX_SIZE
    final_box_lry = final_box_uly + BOX_SIZE
    
    # Final boundary check if enforcing size pushed it over the max boundary
    if final_box_lrx > MAX_COORD:
        final_box_lrx = MAX_COORD
        final_box_ulx = MAX_COORD - BOX_SIZE
    if final_box_lry > MAX_COORD:
        final_box_lry = MAX_COORD
        final_box_uly = MAX_COORD - BOX_SIZE
    
    # Return the final coordinates as floats
    return [float(final_box_ulx), float(final_box_uly), 
            float(final_box_lrx), float(final_box_lry)]

def extract_coords(coord_string, create_box_flag):
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
        if create_box_flag:
            coordinates = create_box(coordinates)
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

def extract_chunk_details(file_name):
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    
    # Define a pattern to find "chunk" followed by digits,
    # and "minichunk" followed by digits.
    # (\d+) creates capturing groups for the digits.
    pattern = re.compile(r"chunk\s+(\d+)\s+minichunk\s+(\d+)")
    
    # Search the base name for the pattern
    match = pattern.search(base_name)
    
    chunk_num = None
    minichunk_num = None
    
    if match:
        try:
            chunk_num = int(match.group(1))
            minichunk_num = int(match.group(2))
        except ValueError:
            print("Error: Could not convert extracted parts to integers "
                  f"in '{file_name}'")
            chunk_num = 0
            minichunk_num = 0
        except IndexError:
             print("Error: Could not find expected groups in regex match "
                   f"for '{file_name}'")
             chunk_num = 0
             minichunk_num = 0
    else:
        chunk_num = 0
        minichunk_num = 0
    return chunk_num, minichunk_num

def sort_prediction_results(data_list):
    """
    Sorts a list of lists based on the first two integer elements.
    
    The sorting is done primarily by the first element (ascending)
    and secondarily by the second element (ascending). Assumes the input
    is a list where each element is a list starting with two numbers.
    
    Args:
      data_list: A list of lists, e.g.,
                 [[0, 19, 'CLASS_A', 99.5],
                  [0, 2,  'CLASS_B', 88.0],
                  [1, 0,  'CLASS_A', 95.1], ...]
    
    Returns:
      A NEW list containing the elements of data_list sorted according
      to the specified criteria. Returns an empty list if input is not a list
      or if sorting fails.
    """
    if not isinstance(data_list, list):
        print("Error: Input must be a list.")
        return [] # Return empty list or raise error
    
    try:
        # Use the sorted() function to create and return a new sorted list.
        # The key is a lambda function that returns a tuple (item[0], item[1]).
        # Python sorts tuples element by element, so this achieves the desired
        # primary sort by item[0] and secondary sort by item[1].
        sorted_list = sorted(data_list, key=lambda item: (item[0], item[1]))
        return sorted_list
    except (IndexError, TypeError) as e:
        # Handle cases where inner items aren't lists or don't have 
        # enough elements
        print(f"Error during sorting: {e}")
        print("Please ensure input is a list of lists, each with at least two "
              "sortable elements.")
        return [] # Return empty list or original list, or raise error

def extract_chunk_minichunk_key(filename):
    """
    Helper function to extract (chunk, minichunk) numbers from a filename.
    Used as the key for sorting. Returns numbers or values that sort last 
    on error.
    """
    # Get just the filename part without extension, in case a full path is passed
    base_name = os.path.splitext(os.path.basename(str(filename)))[0]
    
    # Pattern to find "chunk [digits]" and "minichunk [digits]"
    pattern = re.compile(r"chunk\s+(\d+)\s+minichunk\s+(\d+)")
    match = pattern.search(base_name)
    
    if match:
        try:
            # Extract numbers as integers
            chunk_num = int(match.group(1))
            minichunk_num = int(match.group(2))
            # Return a tuple for sorting: (primary_key, secondary_key)
            return (chunk_num, minichunk_num)
        except (ValueError, IndexError):
            # Handle cases where captured groups aren't valid numbers or missing
            pass # Fall through to return the default 'sort last' key
    
    # If pattern doesn't match or conversion fails, return a tuple
    # that will sort these items after all valid items.
    # float('inf') is larger than any integer.
    return (float('inf'), float('inf'))

def sort_file_names(filename_list):
    """
    Sorts a list of filenames based on embedded chunk and minichunk numbers.
    
    Assumes filenames generally follow the pattern 
    '... chunk [num] minichunk [num] ...'.
    Sorts ascending primarily by chunk number, then secondarily by minichunk 
    number. Filenames not matching the pattern are sorted towards the end.
    
    Args:
      filename_list: A list of strings, where each string is a filename.
    
    Returns:
      A NEW list containing the filenames sorted according to the criteria.
      Returns an empty list if the input is not a list or sorting fails globally.
    """
    if not isinstance(filename_list, list):
        print("Error: Input must be a list of filenames.")
        return []
    
    try:
        # Use sorted() with the helper function as the key
        # This applies extract_chunk_minichunk_key to each filename
        # and sorts based on the (chunk_num, minichunk_num) tuple returned.
        sorted_list = sorted(filename_list, key=extract_chunk_minichunk_key)
        return sorted_list
    except Exception as e:
        # Catch any other unexpected errors during sorting
        print(f"An unexpected error occurred during sorting: {e}")
        return []

def check_positive_int(var, description):
    while not isinstance(var, int):
        print(f"\nerror: {description} ({var}) is not "
              "an integer.")
        user_input = input("please enter a new {description} (integer): ")
        try:
            new_var = int(user_input)
            if new_var <= 0:
                print("error: must be positive")
                continue
            return new_var
        except:
            print("error: must be integer")
    return var
