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
