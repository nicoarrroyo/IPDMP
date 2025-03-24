import numpy as np
import csv

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

def performance_estimate(gee_connect, compression, dpi, plot_size, save_images, 
                         high_res_Sentinel, do_l7, do_l8, do_l9, do_s2):
    perf_params = [gee_connect, compression, dpi, plot_size, save_images,
                   high_res_Sentinel, do_l7, do_l8, do_l9, do_s2]
    
    perf_counter = 0
    for param in perf_params:
        if isinstance(param, bool) and param:
            perf_counter += 1
            if do_s2 and param == high_res_Sentinel:
                perf_counter += 4
            if param == save_images:
                perf_counter += 3
        
        if isinstance(param, int) and not isinstance(param, bool):
            if param == compression and param <= 3:
                perf_counter += 3
            else:
                perf_counter += 1
            if param == dpi and param >= 2000 and save_images:
                perf_counter += 1
        if isinstance(param, tuple):
            if param == plot_size and param > (3, 3):
                perf_counter += 2
    
    max_counter = 30 # incorrect - needs tweaking
    return perf_counter/max_counter

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
            entry = f"{entry},{rows[j][k]}"
        write_file.write(f"\n{entry}")

def blank_entry_check(file):
    k = 0
    popped = False
    with open(file, mode="r") as re: # read
        rows = list(csv.reader(re))
    while k < len(rows): # check for blank entries
        if len(rows[k]) < 2:
            rows.pop(k)
            print(f"eliminated blank entry on line {k} (chunk {k-2})")
            k -= 1
            popped = True
        k += 1
    if popped:
        with open(file, mode="w") as wr: # write
            for j in range(len(rows)):
                wr.write(f"{rows[j][0]},{rows[j][1]}\n")
        with open(file, mode="w") as wr: # write
            rewrite(write_file=wr, rows=rows)
