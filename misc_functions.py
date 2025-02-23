def table_print(compression, dpi, do_l7, do_l8, do_l9, save_images, plot_size, gee_connect):
    # Define the variable names and their symbols
    variables = [
        ("Image Compression", "c", compression), 
        ("Dots per Inch", "dpi", dpi), 
        ("Landsat 7", "do_l7", do_l7), 
        ("Landsat 8", "do_l8", do_l8), 
        ("Landsat 9", "do_l9", do_l9), 
        ("Image Saving", "save_images", save_images), 
        ("Plot Size", "plot_size", plot_size), 
        ("Google Earth Engine", "gee_connect", gee_connect)
    ]
    
    # Determine the maximum lengths for formatting
    max_var_length = max(len(var[0]) for var in variables)
    max_symbol_length = max(len(var[1]) for var in variables)
    max_value_length = max(len(str(var[2])) for var in variables)

    # Print the header
    header = f"| {'variable'.ljust(max_var_length)} | {'symbol'.ljust(max_symbol_length)} | {'value'.ljust(max_value_length)} |"
    separator = '|' + '-' * (max_var_length + 2) + '|' + '-' * (max_symbol_length + 2) + '|' + '-' * (max_value_length + 2) + '|'
    
    print(header)
    print(separator)
    
    # Print each variable row
    for var_name, symbol, value in variables:
        print(f"| {var_name.ljust(max_var_length)} | {symbol.ljust(max_symbol_length)} | {str(value).ljust(max_value_length)} |")
