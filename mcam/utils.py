

def convert_wells_to_indices(wells):
    '''Converts a list of well names to indices'''
    indices = []
    if isinstance(wells, list):
        for well in wells:
            indices.append(convert_well_to_index(well))
    elif isinstance(wells, str):
        indices.append(convert_well_to_index(wells))

    return indices


def convert_well_to_index(well):
    '''Converts a well name to an index. First index is 1'''
    if not isinstance(well, str) or len(well) != 2:
        raise ValueError('well must be a 2 character string')

    row = well[0]
    col = int(well[1])

    if row == 'A':
        index = col
    elif row == 'B':
        index = col + 6
    elif row == 'C':
        index = col + 12
    elif row == 'D':
        index = col + 18
    else:
        raise ValueError('row must be A/B/C/D')

    return index
