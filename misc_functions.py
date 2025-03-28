import numpy as np
import csv
import sys
import time
import threading

def table_print(**kwargs):
    if not kwargs:
        print("No data to display.")
        return
    
    # Compute max lengths efficiently
    max_var_length = max(map(len, kwargs.keys()), default=8)
    max_value_length = max(map(lambda v: len(str(v)), kwargs.values()), default=5)

    # Format the header and separator dynamically
    header = f"| {'Variable'.ljust(max_var_length)} | {'Value'.ljust(max_value_length)} |"
    separator = "-" * len(header)

    # Print table
    print(separator)
    print(header)
    print(separator)
    for key, value in kwargs.items():
        print(f"| {key.ljust(max_var_length)} | {str(value).ljust(max_value_length)} |")
    print(separator)

def split_array(array, n_chunks):
    rows = np.array_split(array, np.sqrt(n_chunks), axis=0) # split into rows
    split_arrays = [np.array_split(row_chunk, np.sqrt(n_chunks), 
                                   axis=1) for row_chunk in rows]
    chunks = [subarray for row_chunk in split_arrays for subarray in row_chunk]
    return chunks

def rewrite(write_file, rows):
    for j in range(len(rows)):
        entry = f"{rows[j][0]},{rows[j][1]}"
        for k in range(2, len(rows[j])): # add coordinates
            if len(rows[j][k]) > 5: # ensure entry is a coordinate
                entry = f"{entry},{rows[j][k]}"
        write_file.write(f"{entry}\n")

def blank_entry_check(file):
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
        print(f"{len(invalid_rows)} invalid entries were removed on", invalid_rows)

def check_file_permission(file_name):
    while True:
        try: # check if file is open
            with open(file_name, mode="a"):
                break
        except IOError:
            print("could not open file - please close the responses file")
            input("press enter to retry")

def spinner(stop_event, message):
    """
    A simple spinner that runs until stop_event is set.
    The spinner updates the ellipsis by overwriting the same line.
    """
    chase = ["   ", ".  ", ".. ", "...", " ..", "  .", "   ", 
             "  .", " ..", "...", ".. ", ".  "]
    wobble = [" \ ", " | ", " / ", " | "]
    woosh = ["|   ", ")   ", " )  ", "  ) ", "   )", "   |", 
             "   |", "  ( ", " (  ", "(   ", "|   "]
    ellipses = ["   ", ".  ", ".. ", "..."]
    dude = [" :D ", " :) ", " :\ ", " :( ", " :\ ", " :) "]
    
    frames = chase
    frames = wobble
    frames = woosh
    frames = ellipses
    frames = dude
    
    frames = wobble
    i = 0
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        sys.stdout.write("\r" + message + frame)
        sys.stdout.flush()
        time.sleep(0.2)
        i += 1
    # Clear the spinner message on stop
    sys.stdout.write("\r" + message + "... complete! \n")
    sys.stdout.flush()

def start_spinner(message):
    # Create an event for signaling the spinner to stop
    stop_event = threading.Event()
    
    # Use a thread to run the spinner concurrently
    thread = threading.Thread(target=spinner, args=(stop_event, message))
    thread.start()
    return stop_event, thread

def end_spinner(stop_event, thread):
    # Turn off the spinner once processing is done
    stop_event.set()
    thread.join()
