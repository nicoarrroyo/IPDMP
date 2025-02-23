def table_print(**kwargs):
    # Prepare the variables from the keyword arguments
    variables = [(key, value) for key, value in kwargs.items()]
    
    # Determine the maximum lengths for formatting
    max_var_length = max(len(var[0]) for var in variables)
    max_value_length = max(len(str(var[1])) for var in variables)

    # Print the header
    header = f"| {'Variable'.ljust(max_var_length)} | {'Value'.ljust(max_value_length)} |"
    separator = '|' + '-' * (max_var_length + 2) + '|' + '-' * (max_value_length + 2) + '|'
    
    print(header)
    print(separator)
    
    # Print each variable row
    for var_name, value in variables:
        print(f"| {var_name.ljust(max_var_length)} | {str(value).ljust(max_value_length)} |")
