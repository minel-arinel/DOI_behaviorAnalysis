from bouter import FreelySwimmingExperiment
from datetime import datetime as dt
import numpy as np
import os
import pandas as pd


class Fish:
    def __init__(self, folder_path, prefix='EK', calibration_params=dict(), overwrite=False):
        self.folder_path = folder_path

        bout_df_exists = False
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if entry.name.endswith('_bout_df.h5'):
                    self.bout_df = pd.read_hdf(entry.path)
                    bout_df_exists = True

        if not bout_df_exists or (bout_df_exists and overwrite):
            self.bout_dfs = list()

            self.process_experiment(prefix, calibration_params)
            self.bout_df = pd.concat(self.bout_dfs, ignore_index=True)
            self.bout_df.to_hdf(os.path.join(folder_path, str(self.fish_id)+'_bout_df.h5'), key='fish_bout_df', mode='w')

    def process_experiment(self, prefix, calibration_params):
        with os.scandir(self.folder_path) as conditions:
            for condition in conditions:
                if os.path.isdir(condition.path):
                    print(condition.path)
                    self.exp = FreelySwimmingExperiment(condition.path)

                    if len(calibration_params) != 0:
                        self.exp['stimulus']['calibration_params'] = calibration_params

                    self.parse_metadata(condition.path, prefix)
                    self.exp.reconstruct_missing_segments(continue_curvature=4)

                    self.stim_df = self.create_stimdf(condition.path)

                    try:
                        self.bout_df = self.exp.get_bout_properties()
                        self.bout_df['concentration'] = self.exp['general']['animal']['concentration']
                        self.bout_df['fish_id'] = self.exp['general']['animal']['id']
                        self.bout_df['condition'] = self.exp['general']['animal']['condition']
                        self.bout_df.dropna(how='any', inplace=True)
                        self.bout_df.reset_index(drop=True, inplace=True)

                        self.combine_boutdf_stimdf()

                        self.bout_dfs.append(self.bout_df)

                    except IndexError:
                        print('ERROR:' + condition.path)

    def parse_metadata(self, path, prefix):
        age_start = path.find(prefix)+3
        age_stop = path.find('dpf')
        self.age = path[age_start:age_stop]

        condition = path[path.rfind('/')+1:]

        id_stop = path.find(condition)-1
        id_start = path[:id_stop].rfind('/')+1
        self.fish_id = int(path[id_start:id_stop])

        conc_stop = path.find('ugml')
        conc_start = path[:conc_stop].rfind('_')+1
        self.concentration = float(path[conc_start:conc_stop])

        self.exp['general']['animal']['age'] = self.age
        self.exp['general']['animal']['id'] = self.fish_id
        self.exp['general']['animal']['concentration'] = self.concentration
        self.exp['general']['animal']['condition'] = condition

    @staticmethod
    def create_stimdf(path):
        with os.scandir(path) as files:
            for file in files:
                if file.name.endswith('.txt'):
                    stim_path = file.path

        with open(stim_path) as file:
            contents = file.read()

        parsed = contents.split('\n')
        stimulus_details = parsed[1:]
        times = [i[:i.find('{')] for i in stimulus_details]
        stimulus_dicts = [eval(i[i.find('{'):]) for i in stimulus_details if 'stationary_end' not in i]
        stim_df = pd.DataFrame(stimulus_dicts)

        ntime_array = []
        stim_id_array = []

        for i in range(len((times))):
            try:
                ntime_array.append(dt.strptime(times[i].split(' ')[1], '%H:%M:%S.%f:'))
            except ValueError:
                ntime_array.append(dt.strptime(times[i].split(' ')[1], '%H:%M:%S:'))

            stim_id_array.append(int(times[i].split(' ')[2]))

        stim_df.loc[:, 'stim_index'] = stim_id_array
        ntime_array = ntime_array[1:]

        t_counter = []
        for t in np.diff(ntime_array):
            t_counter.append(t.total_seconds())

        t_ascending = np.cumsum(t_counter)
        t_ascending = np.insert(t_ascending, 0, 0)

        final_stims = stim_df.iloc[1:]

        final_stims.loc[:, 't'] = t_ascending

        final_stims['stationary'] = False
        final_stims.loc[np.r_[final_stims.velocity == 0], 'stationary'] = True

        return final_stims

    def combine_boutdf_stimdf(self):
        self.bout_df['velocity'] = 0.0
        self.bout_df['angle'] = 0
        self.bout_df['stat_time'] = 0
        self.bout_df['stim_name'] = ''
        self.bout_df['duration'] = 0
        self.bout_df['stim_index'] = 0
        self.bout_df['stim_start_t'] = 0
        self.bout_df['stationary'] = False

        self.bout_df['delta_t'] = 0
        self.bout_df['delta_x'] = 0
        self.bout_df['delta_y'] = 0
        self.bout_df['delta_theta'] = 0
        self.bout_df['distance'] = 0
        self.bout_df['speed'] = 0
        self.bout_df['angular_velocity'] = 0

        for i, row in self.bout_df.iterrows():
            stim = self.stim_df[np.r_[self.stim_df.t <= row.t_start]].iloc[-1, :]
            self.bout_df.loc[i, 'velocity'] = stim['velocity']
            self.bout_df.loc[i, 'angle'] = stim['angle']
            self.bout_df.loc[i, 'stat_time'] = stim['stat_time']
            self.bout_df.loc[i, 'stim_name'] = stim['stim_name']
            self.bout_df.loc[i, 'duration'] = stim['duration']
            self.bout_df.loc[i, 'stim_index'] = stim['stim_index']
            self.bout_df.loc[i, 'stim_start_t'] = stim['t']
            self.bout_df.loc[i, 'stationary'] = stim['stationary']

            delta_t = row.t_end - row.t_start
            delta_x = row.x_end - row.x_start
            delta_y = row.y_end - row.y_start
            delta_theta = row.theta_end - row.theta_start
            distance = np.sqrt(delta_x**2 + delta_y**2)

            self.bout_df.loc[i, 'delta_t'] = delta_t
            self.bout_df.loc[i, 'delta_x'] = delta_x
            self.bout_df.loc[i, 'delta_y'] = delta_y
            self.bout_df.loc[i, 'delta_theta'] = delta_theta
            self.bout_df.loc[i, 'distance'] = distance
            self.bout_df.loc[i, 'speed'] = distance / delta_t
            self.bout_df.loc[i, 'angular_velocity'] = delta_theta / delta_t
