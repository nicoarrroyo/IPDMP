import numpy as np

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
