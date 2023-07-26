import numpy as np
import os
import pandas as pd
from pathlib import Path
from utils import convert_wells_to_indices


class MCAM:
    def __init__(self, folder_path, prefix='EK', concentrations=[0], conc_orientation='cols', remove_wells=dict()):
        self.folder_path = folder_path
        self.concentrations = concentrations
        self.stim_df = None
        self.dataframes = dict()

        self.data_paths = dict()
        self.process_filestructure()

        if len(self.dataframes) == 0:
            self.combine_dataframes(prefix, concentrations, conc_orientation, remove_wells)

    def process_filestructure(self):
        '''Creates a data_paths attribute with the paths to different files'''
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if entry.name == 'stim_df.csv':
                    self.stim_df = pd.read_csv(entry.path, index_col=0)
                    self.data_paths['stim_df'] = Path(entry.path)
                elif entry.name == 'csv_files':
                    self.data_paths['csv_files'] = Path(entry.path)
                elif entry.name == 'figures':
                    self.data_paths['figures'] = Path(entry.path)

        if 'csv_files' in self.data_paths:
            with os.scandir(self.data_paths['csv_files']) as entries:
                for entry in entries:
                    *condition, metric, concentration = entry.name.split('_')
                    condition = '_'.join(condition)

                    if condition not in self.dataframes:
                        self.dataframes[condition] = dict()

                    if metric not in self.dataframes[condition]:
                        self.dataframes[condition][metric] = dict()

                    concentration = float(concentration[:concentration.rfind('.')])
                    self.dataframes[condition][metric][concentration] = pd.read_csv(entry.path, index_col=0)
                    print(f'found {entry.path}')

    def combine_dataframes(self, prefix, concentrations, conc_orientation, remove_wells):
        '''Processes individual dataframes and combines to one'''
        if conc_orientation != 'cols' and conc_orientation != 'rows':
            raise ValueError('conc_orientation can be either \'rows\' or \'cols\'')

        dist_traveled_cols = ['distance_traveled', 'speed'] * 24
        dist_traveled_cols = ['time'] + dist_traveled_cols

        tracking_cols = ['snout', 'snout', 'snout', 'L_eye', 'L_eye', 'L_eye', 'R_eye', 'R_eye', 'R_eye',
                        'center_y', 'center_x', 'center_likelihood', 'caudal_fin', 'caudal_fin', 'caudal_fin', 'mid_tail',
                        'mid_tail', 'mid_tail', 'between_center_and_mid', 'between_center_and_mid',
                        'between_center_and_mid', 'between_mid_and_caudal', 'between_mid_and_caudal',
                        'between_mid_and_caudal'] * 24
        tracking_cols = ['time'] + tracking_cols

        wells_per_conc = 24 / len(concentrations)
        if conc_orientation == 'cols':
            cols_per_conc = int(wells_per_conc / 4)
            well_inds = list()

            for i in range(cols_per_conc):
                well_inds.append(np.arange(i+1, 25, 6))

            well_inds = np.sort(np.concatenate(well_inds))

        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if os.path.isdir(entry.path) and entry.name.startswith(prefix):

                    remove = False
                    if entry.name in remove_wells:
                        remove = True

                    with os.scandir(entry.path) as subentries:
                        for subentry in subentries:
                            if os.path.isdir(subentry.path):

                                remove_well_inds = list()
                                if remove and subentry.name in remove_wells[entry.name]:
                                    remove_well_inds = convert_wells_to_indices(remove_wells[entry.name][subentry.name])

                                dist_traveled_path = os.path.join(subentry.path, 'results', 'distance_traveled_metrics.csv')
                                tracking_path = os.path.join(subentry.path, 'results', 'tracking_data.csv')

                                if self.stim_df is None:
                                    stimulus_path = os.path.join(subentry.path, 'results', 'stimulus_metadata.csv')
                                    self.create_stim_df(stimulus_path)

                                dist_traveled_metrics = pd.read_csv(dist_traveled_path, header=9).set_axis(dist_traveled_cols, axis='columns')
                                dist_traveled_df = dist_traveled_metrics[['time', 'distance_traveled']]
                                speed_df = dist_traveled_metrics[['time', 'speed']]

                                tracking_df = pd.read_csv(tracking_path, header=10).set_axis(tracking_cols, axis='columns')
                                tracking_df = tracking_df[['time', 'center_y', 'center_x']]

                                if subentry.name not in self.dataframes.keys():
                                    self.dataframes[subentry.name] = dict()
                                    self.dataframes[subentry.name]['distance'] = dict()
                                    self.dataframes[subentry.name]['speed'] = dict()
                                    self.dataframes[subentry.name]['tracking'] = dict()

                                    for i, conc in enumerate(concentrations):
                                        self.dataframes[subentry.name]['distance'][conc] = list()
                                        self.dataframes[subentry.name]['speed'][conc] = list()
                                        self.dataframes[subentry.name]['tracking'][conc] = list()

                                        if conc_orientation == 'cols':
                                            dist_ind_arr = np.array([ind for ind in np.r_[0, well_inds+i] if ind not in remove_well_inds])
                                            self.dataframes[subentry.name]['distance'][conc].append(dist_traveled_df.iloc[:, dist_ind_arr])
                                            self.dataframes[subentry.name]['speed'][conc].append(speed_df.iloc[:, dist_ind_arr])

                                            track_ind_arr = np.array([ind for ind in np.r_[0, well_inds+i, well_inds+i+24] if ind not in remove_well_inds])
                                            self.dataframes[subentry.name]['tracking'][conc].append(tracking_df.iloc[:, track_ind_arr])

                                        elif conc_orientation == 'rows':
                                            start_well = (i*wells_per_conc)+1
                                            end_well = start_well + wells_per_conc

                                            dist_ind_arr = np.array([ind for ind in np.r_[0, start_well:end_well] if ind not in remove_well_inds])
                                            self.dataframes[subentry.name]['distance'][conc].append(dist_traveled_df.iloc[:, dist_ind_arr])
                                            self.dataframes[subentry.name]['speed'][conc].append(speed_df.iloc[:, dist_ind_arr])

                                            track_ind_arr = np.array([ind for ind in np.r_[0, start_well:end_well, start_well+24:end_well+24] if ind not in remove_well_inds])
                                            self.dataframes[subentry.name]['tracking'][conc].append(tracking_df.iloc[:, track_ind_arr])

                                else:
                                    for i, conc in enumerate(concentrations):

                                        if conc_orientation == 'cols':
                                            dist_ind_arr = np.array([ind for ind in np.r_[well_inds+i] if ind not in remove_well_inds])
                                            self.dataframes[subentry.name]['distance'][conc].append(dist_traveled_df.iloc[:, dist_ind_arr])
                                            self.dataframes[subentry.name]['speed'][conc].append(speed_df.iloc[:, dist_ind_arr])

                                            track_ind_arr = np.array([ind for ind in np.r_[well_inds+i, well_inds+i+24] if ind not in remove_well_inds])
                                            self.dataframes[subentry.name]['tracking'][conc].append(tracking_df.iloc[:, track_ind_arr])

                                        elif conc_orientation == 'rows':
                                            start_well = (i*wells_per_conc)+1
                                            end_well = start_well + wells_per_conc

                                            dist_ind_arr = np.array([ind for ind in np.r_[start_well:end_well] if ind not in remove_well_inds])
                                            self.dataframes[subentry.name]['distance'][conc].append(dist_traveled_df.iloc[:, dist_ind_arr])
                                            self.dataframes[subentry.name]['speed'][conc].append(speed_df.iloc[:, dist_ind_arr])

                                            track_ind_arr = np.array([ind for ind in np.r_[start_well:end_well, start_well+24:end_well+24] if ind not in remove_well_inds])
                                            self.dataframes[subentry.name]['tracking'][conc].append(tracking_df.iloc[:, track_ind_arr])

                                print(f'processed {entry.name} - {subentry.name}')

        for condition in self.dataframes:
            for metric in self.dataframes[condition]:
                for concentration in self.dataframes[condition][metric]:
                    dfs = self.dataframes[condition][metric][concentration]
                    final_df = pd.concat(dfs, axis=1)

                    if metric == 'distance':
                        final_df['average_dist'] = final_df.iloc[:, 1:].mean(axis=1)
                        final_df['sem'] = final_df.iloc[:, 1:].sem(axis=1)
                    elif metric == 'speed':
                        final_df['average_speed'] = final_df.iloc[:, 1:].mean(axis=1)
                        final_df['sem'] = final_df.iloc[:, 1:].sem(axis=1)
                    elif metric == 'tracking':
                        n_fish = len(final_df['center_y'].columns)
                        final_df[['dist_from_center']+[f'dist_from_center.{i}' for i in range(1, n_fish)]] = 0

                        x = final_df['center_x'].copy()
                        y = final_df['center_y'].copy()

                        for i, row in final_df.iterrows():
                            for j in range(n_fish):
                                if j == 0:
                                    final_df.loc[i, 'dist_from_center'] = np.linalg.norm((x.iloc[i, j], y.iloc[i, j]))
                                else:
                                    final_df.loc[i, f'dist_from_center.{j}'] = np.linalg.norm((x.iloc[i, j], y.iloc[i, j]))

                        dist_cols = [col for col in final_df if col.startswith('dist_from_center')]
                        final_df['average_dist_from_center'] = final_df[dist_cols].mean(axis=1)
                        final_df['sem'] = final_df[dist_cols].sem(axis=1)

                    final_df = pd.concat([final_df, self.stim_df], axis=1)

                    csv_files_path = os.path.join(self.folder_path, 'csv_files')
                    if not os.path.exists(csv_files_path):
                        os.mkdir(csv_files_path)
                    self.data_paths['csv_files'] = Path(csv_files_path)

                    final_df.to_csv(os.path.join(csv_files_path, f'{condition}_{metric}_{concentration}.csv'))
                    print(f'saved {condition}_{metric}_{concentration}.csv')
                    self.dataframes[condition][metric][concentration] = final_df

    def create_stim_df(self, stimulus_path):
        '''Create a dataframe of stimulus metadata for each time point'''
        stim_df = pd.read_csv(stimulus_path, header=6)

        # Label the different stimuli
        stim_df['stim_name'] = ''
        for i, row in stim_df.iterrows():
            if row['time'] <= 300.0:
                stim_df.loc[i, 'stim_name'] = 'locomotor'
            elif row['vibration_frequency'] == 300:
                stim_df.loc[i, 'stim_name'] = 'vibration_startle'
            elif row['flash_lux'] == 10000:
                stim_df.loc[i, 'stim_name'] = 'light_flash'
            elif row['flash_lux'] == 5000:
                stim_df.loc[i, 'stim_name'] = 'light_epoch'
            else:
                stim_df.loc[i, 'stim_name'] = 'dark_epoch'

        # Distinguish dark flash from dark epoch
        row_inds = stim_df[stim_df.stim_name == 'dark_epoch'].index.values  # row indices of dark stimuli
        diffs = np.diff(row_inds)

        begin = False
        start = 0
        stop = 0

        for i, diff in enumerate(diffs):
            if diff == 1 and begin is False:
                # if it's the beginning of a new dark stimulus
                begin = True
                start = row_inds[i]
            elif diff == 1 and begin is True:
                # if the dark stimulus has already begun
                continue
            elif diff != 1:
                # if the dark stimulus ends
                begin = False
                stop = row_inds[i]
                if stim_df.loc[stop, 'time'] - stim_df.loc[start, 'time'] <= 1.05:
                    stim_df.loc[start:stop, 'stim_name'] = 'dark_flash'

        # Add stimulus numbers
        stim_df['stim_num'] = 0

        for stim in stim_df.stim_name.unique():
            stim_num = 0
            sub_df = stim_df[stim_df.stim_name == stim]
            prev_i = -1
            for i, row in sub_df.iterrows():
                if prev_i == -1:  # for the first row of sub_df
                    stim_df.loc[i, 'stim_num'] = stim_num
                    prev_i = i
                elif i-1 == prev_i:
                    stim_df.loc[i, 'stim_num'] = stim_num
                    prev_i = i
                else:
                    stim_num += 1
                    stim_df.loc[i, 'stim_num'] = stim_num
                    prev_i = i

        stim_df.to_csv(os.path.join(self.folder_path, 'stim_df.csv'))
        print('saved stim_df.csv')
        self.stim_df = stim_df
        self.data_paths['stim_df'] = Path(os.path.join(self.folder_path, 'stim_df.csv'))