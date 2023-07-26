

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


def add_bins(mcam, time_bin=1.0):
    '''Adds a binned_time column per given time in seconds'''

    for condition in mcam.dataframes:
        for metric in mcam.dataframes[condition]:
            for conc in mcam.concentrations:

                df = mcam.dataframes[condition][metric][conc]
                bin_count = time_bin

                df['binned_time'] = 0

                for i, row in df.iterrows():
                    if df.iloc[i, 0] < bin_count:
                        df.loc[i, 'binned_time'] = bin_count
                    else:
                        bin_count += time_bin
                        df.loc[i, 'binned_time'] = bin_count


def get_stimulus_starts_stops(mcam):
    '''Returns the start and stop indices of stimuli'''

    df = mcam.stim_df

    locomotor_start = df[df['stim_name'] == 'locomotor'].index[0]
    locomotor_stop = df[df['stim_name'] == 'locomotor'].index[-1]

    light_epoch1_start = df[df['stim_name'] == 'light_epoch'].index[0]
    light_epoch1_stop = df[(df['stim_name'] == 'light_epoch') & (df['stim_num'] == 5)].index[-1]

    light_epoch2_start = df[(df['stim_name'] == 'light_epoch') & (df['stim_num'] == 6)].index[0]
    light_epoch2_stop = df[(df['stim_name'] == 'light_epoch') & (df['stim_num'] == 11)].index[-1]

    light_epoch3_start = df[(df['stim_name'] == 'light_epoch') & (df['stim_num'] == 12)].index[0]
    light_epoch3_stop = df[(df['stim_name'] == 'light_epoch') & (df['stim_num'] == 17)].index[-1]

    dark_epoch1_start = df[df['stim_name'] == 'dark_epoch'].index[0]
    dark_epoch1_stop = df[(df['stim_name'] == 'dark_epoch') & (df['stim_num'] == 5)].index[-1]

    dark_epoch2_start = df[(df['stim_name'] == 'dark_epoch') & (df['stim_num'] == 6)].index[0]
    dark_epoch2_stop = df[(df['stim_name'] == 'dark_epoch') & (df['stim_num'] == 11)].index[-1]

    dark_epoch3_start = df[(df['stim_name'] == 'dark_epoch') & (df['stim_num'] == 12)].index[0]
    dark_epoch3_stop = df[(df['stim_name'] == 'dark_epoch') & (df['stim_num'] == 17)].index[-1]

    dark_flash1_start = df[df['stim_name'] == 'dark_flash'].index[0]
    dark_flash2_start = df[(df['stim_name'] == 'dark_flash') & (df['stim_num'] == 5)].index[0]

    light_flash1_start = df[df['stim_name'] == 'light_flash'].index[0]
    light_flash2_start = df[(df['stim_name'] == 'light_flash') & (df['stim_num'] == 5)].index[0]

    vibration_startle1_start = df[df['stim_name'] == 'vibration_startle'].index[0]
    vibration_startle2_start = df[(df['stim_name'] == 'vibration_startle') & (df['stim_num'] == 5)].index[0]
    vibration_startle2_stop = vibration_startle2_start + (light_epoch2_stop - vibration_startle1_start)

    starts_stops = {
        'locomotor': (locomotor_start, locomotor_stop),
        'light_epoch1': (light_epoch1_start, light_epoch1_stop),
        'light_epoch2': (light_epoch2_start, light_epoch2_stop),
        'light_epoch3': (light_epoch3_start, light_epoch3_stop),
        'dark_epoch1': (dark_epoch1_start, dark_epoch1_stop),
        'dark_epoch2': (dark_epoch2_start, dark_epoch2_stop),
        'dark_epoch3': (dark_epoch3_start, dark_epoch3_stop),
        'dark_flash1': (dark_flash1_start, light_epoch1_stop),
        'dark_flash2': (dark_flash2_start, light_epoch3_stop),
        'light_flash1': (light_flash1_start, dark_epoch1_stop),
        'light_flash2': (light_flash2_start, dark_epoch2_stop),
        'vibration_startle1': (vibration_startle1_start, light_epoch2_stop),
        'vibration_startle2': (vibration_startle2_start, vibration_startle2_stop)
    }

    return starts_stops
