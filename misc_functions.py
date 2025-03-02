def table_print(**kwargs):
    if not kwargs:
        print("No data to display.")
        return
    
    # Compute max lengths efficiently
    max_var_length = max(map(len, kwargs.keys()), default=8)  # Default for "Variable"
    max_value_length = max(map(lambda v: len(str(v)), kwargs.values()), default=5)  # Default for "Value"

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

# =============================================================================
# def calculate_water_indices(image_arrays):
#     print("populating water index arrays", end="... ")
#     start_time = time.monotonic()
# 
#     blue, green, nir, swir1, swir2 = image_arrays
#     ndwi, mndwi, awei_sh, awei_nsh = get_indices(blue, green, nir, swir1, swir2)
#     indices = [ndwi, mndwi, awei_sh, awei_nsh]
# 
#     time_taken = time.monotonic() - start_time
#     print(f"complete! time taken: {round(time_taken, 2)} seconds")
#     return indices
# 
# def show_images(do_sat, indices, sat_number):
#     if do_sat:
#         minimum = -1
#         maximum = 1
#         if save_images:
#             print("displaying and saving water index images...")
#         else:
#             print("displaying water index images...")
#         start_time = time.monotonic()
#         plot_image(indices, sat_number, plot_size, 
#                    minimum, maximum, compression, dpi, save_images)
#         time_taken = time.monotonic() - start_time
#         print(f"complete! time taken: {round(time_taken, 2)} seconds")
# =============================================================================
