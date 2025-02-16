def nice_formatting(header1, header2, unit1, unit2):
    gap = 2
    # Define column headers
    headers = [header1, header2]
    units = [unit1, unit2]
    # Print column headers and units
    header_line = "|".join(header.center(len(header) + gap) for header in headers)
    unit_line = "|".join(unit.center(len(headers[i]) + gap) for i, unit in enumerate(units))
    print(header_line)
    print(unit_line)
    return headers, units, gap