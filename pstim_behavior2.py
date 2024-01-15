
# This code is written by Matthew D. Loring.
# Changes to the code by Minel Arinel have been commented with the tag 'minel'

import os
import json

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
np.seterr(all="ignore")

from datetime import datetime as dt
from numba import jit
from pathlib import Path
from tqdm.auto import tqdm
from master_functions import mpWrapper, stimTimersMP, trialExcluderMP, cumThetaMP

goodFish = [13, 15, 17, 18, 2, 21, 23, 24, 26, 27, 28, 30, 31, 39, 40, 44, 46, 47, 50, 51, 52, 53, 54, 55, 56]
rig_dict = {
            '1': [6,13,21,27,33,39,45,51],
            '2': [5, 14, 28, 34, 40, 46, 52],
            '3': [3, 15, 23, 29, 35, 41, 47, 53],
            '4': [4, 16, 24, 30, 36, 42, 48, 54],
            '5': [1, 17, 25, 31, 37, 43, 49, 55],
            '6': [2, 18, 26, 32, 38, 44, 50, 56]
           }


def load_data(input_paths, onlygood=False):
    data = [[]]*len(input_paths)

    n = 0
    for _p in input_paths:
        bouts = []
        datas = []
        with os.scandir(_p) as entries:
            for entry in entries:
                fishid = entry.name
                try:
                    int(fishid)
                except ValueError:
                    continue
                if onlygood:
                    if int(fishid) in goodFish:
                        p = os.path.join(_p, fishid)
                        p_data = os.path.join(p, 'dataframe.h5')
                        p_bout = os.path.join(p, 'boutframe.h5')

                        try:
                            datas.append(pd.read_hdf(p_data))
                            bouts.append(pd.read_hdf(p_bout))
                        except FileNotFoundError:
                            pass
                else:
                    p = os.path.join(_p, fishid)
                    p_data = os.path.join(p, 'dataframe.h5')
                    p_bout = os.path.join(p, 'boutframe.h5')

                    try:
                        datas.append(pd.read_hdf(p_data))
                        bouts.append(pd.read_hdf(p_bout))
                    except FileNotFoundError:
                        pass

        data[n] = [datas, bouts]
        n += 1
    return data


def get_rig(val, my_dict=rig_dict):
    for key, value in my_dict.items():
        for values in value:
            if values == int(val):
                return key
    return np.nan

__spec__ = None

def preHist(dataframe):
    #minel - commented out because no need
    """if 'ID' not in dataframe.columns:
        dataframe.loc[:, 'ID'] = dataframe.fish_id.values"""

    dataframe = mpWrapper(stimTimersMP, dataframe)
    dataframe = mpWrapper(trialExcluderMP, dataframe, big=True)
    # dataframe = mpWrapper(cumThetaMP, dataframe) # minel - commented out because no need

    if dataframe is None: #minel - added an if statement for bad recordings with no bouts
        return None
    else:
        #boutframe = _extactor(dataframe) #minel - calculate boutframe in the jupyter notebook
        return dataframe #minel - so only returning the dataframe


def _extactor(df):
    dsts = []
    thetas = []
    stim_index = []
    stim_names = []
    ids = []
    time_tots = []
    bout_labels = []

    _boutdf = df[(df.bout.notna()) & (df.f0_x.notna())]

    for _id in _boutdf.fish_id.unique(): #minel - changed to fish_id
        boutdf = _boutdf[_boutdf.fish_id == _id] #minel - changed to fish_id
        for b in boutdf.bout.unique():
            _bout_data = boutdf[(boutdf.bout == b)]
            last_indx = _bout_data.f0_x.last_valid_index()
            last_indy = _bout_data.f0_y.last_valid_index()

            if last_indx and last_indy is not None:
                last_ind = min(last_indx, last_indy)
                sub = _bout_data.loc[:last_ind]

                if len(sub) >= 5:
                    _stim_index = _bout_data.stim_index.values[0]
                    _stim_name = _bout_data.stim_name.values[0]
                    ts = boutdf[(boutdf.stim_index == _stim_index)].stim_time.values
                    time_tots.append(ts[-1] - ts[0])

                    dst1 = [sub.f0_x.values[0], sub.f0_y.values[0]]
                    dst2 = [sub.f0_x.values[-1], sub.f0_y.values[-1]]
                    dst = np.linalg.norm([dst1, dst2])
                    dsts.append(dst)

                    thetas.append(sub.cum_theta.values[-1] - sub.cum_theta.values[0])

                    stim_index.append(_stim_index)
                    stim_names.append(_stim_name)
                    bout_labels.append(b)
                    ids.append(_id)

    extracted_df = pd.DataFrame({'stim_index': stim_index,
                                     'stim_name': stim_names,
                                     'bout_angle': thetas,
                                     'distance': dsts,
                                     't_tot': time_tots,
                                     'bout': bout_labels,
                                     'fish_id': ids}) #minel - changed to fish_id
    return extracted_df


def Hist2(boutframe, stims=['forward'], binsize=1.5):

    __sub = boutframe[boutframe.stim_name.isin(stims)]
    __sub = __sub[__sub.bout_angle.notna()]

    hists = []
    _bins = np.arange(-100, 100, binsize)

    for i in tqdm(__sub.ID.unique(), 'fish done', position=0, leave=True):
        _sub = __sub[__sub.ID == i]
        for stim in _sub.stim_index.unique():
            sub = _sub[_sub.stim_index == stim]
            if len(sub) <= 1:
                continue

            _hist, bins = np.histogram(sub.bout_angle.values, _bins)
            hist = (1000 * _hist) / sub.t_tot.values[0]

            hists.append(hist)
    finhists = np.average(hists, axis=0)
    finstds = np.std(hists, axis=0) / len(hists)
    return finhists, finstds, _bins


def Hist(dataframe, boutframe, stims=['forward'], binsize=1.5):

    __sub = boutframe[boutframe.stim_name.isin(stims)]
    __sub = __sub[__sub.bout_angle.notna()]
    _df = dataframe[dataframe.stim_name.isin(stims)]

    hists = []
    _bins = np.arange(-100, 100, binsize)

    for i in tqdm(__sub.ID.unique(), 'fish done', position=0, leave=True):
        _sub = __sub[__sub.ID == i]
        for stim in _sub.stim_index.unique():
            sub = _sub[_sub.stim_index == stim]
            if len(sub) <= 1:
                continue

            df_sub = _df[(_df.ID == sub.ID.values[0]) & (_df.stim_index == sub.stim_index.values[0])]
            stim_timer = df_sub.t.values[-1] - df_sub.t.values[0]

            _hist, bins = np.histogram(sub.bout_angle.values, _bins)
            hist = (1000 * _hist) / stim_timer
            hists.append(hist)
    finhists = np.average(hists, axis=0)
    finstds = np.std(hists, axis=0) / len(hists)
    return finhists, finstds, _bins


def main(parent_folder, rerun=False):
    n = False
    # iterate through folder to find the files inside
    with os.scandir(parent_folder) as entries:
        for entry in entries:
            if entry.is_file():
                '''if entry.name.endswith('bout_df.h5'): # minel - commented out so that it rewrites the files when the code is re-run
                    bout_df_path = os.path.join(parent_folder, entry.name)
                if entry.name.endswith('main_df.h5'):
                    n = True
                    main_df_path = os.path.join(parent_folder, entry.name)'''
                if entry.name.endswith('.txt'):
                    stim_path = os.path.join(parent_folder, entry.name)
                if entry.name.endswith('behavior_log.csv'):
                    behavior_path = os.path.join(parent_folder, entry.name)
                if entry.name.endswith('.json'):
                    metadata_path = os.path.join(parent_folder, entry.name)

    if n and not rerun:
        return pd.read_hdf(main_df_path)
    else:

        with open(metadata_path) as json_file:
            metadata = json.load(json_file)
        start_time = dt.strptime(metadata['general']['t_protocol_start'].split('T')[1], '%H:%M:%S.%f').time()

        with open(stim_path) as file:
            contents = file.read()

        parsed = contents.split('\n')
        fish_details = parsed[0]
        stimulus_details = parsed[1:]

        fish_id = int(fish_details.split('_')[0].split('fish')[1])

        times = [i[:i.find('{')] for i in stimulus_details]
        tex_freq = False
        if 'tex_freq' in stimulus_details[0]:
            tex_freq = True
            _stimulus_dicts = []
            tex_freqs = []
            for i in stimulus_details:
                _stimulus_dicts.append(i[i.find('{'):i.find('}') + 1])
                tex_freqs.append(i[i.find('freq: '):].split(' ')[-1])

            stimulus_dicts = [eval(i[i.find('{'):]) for i in _stimulus_dicts if 'stationary_end' not in i]
            freq_fixer = []
            for i in range(len(tex_freqs)):
                if tex_freqs[i] == '}':
                    freq_fixer.append(tex_freqs[i - 1])
                else:
                    freq_fixer.append(tex_freqs[i])
        else:
            stimulus_dicts = [eval(i[i.find('{'):]) for i in stimulus_details if 'stationary_end' not in i]

        # mostly a binocular gratings fix, need to stack the tuples into two separate columns
        for stim in range(len(stimulus_dicts)):
            for item in stimulus_dicts[stim].copy():
                try:
                    if len(stimulus_dicts[stim][item]) > 1 and type(stimulus_dicts[stim][item]) is not str:
                        for i in range(len(stimulus_dicts[stim][item])):
                            name = item + '_' + str(i)
                            stimulus_dicts[stim][name] = stimulus_dicts[stim][item][i]
                        stimulus_dicts[stim].pop(item)
                except:
                    pass

        stim_df = pd.DataFrame(stimulus_dicts)

        final_stims = stim_df
        if tex_freq:
            final_stims.loc[:, 'freq'] = freq_fixer

        # interpret the times and set up an array to measure elapsed times across experiment
        ntime_array = []
        stim_id_array = []

        for i in range(len((times))):
            try:
                ntime_array.append(dt.strptime(times[i].split(' ')[1], '%H:%M:%S.%f:'))
            except ValueError:
                ntime_array.append(dt.strptime(times[i].split(' ')[1], '%H:%M:%S:'))

            stim_id_array.append(int(times[i].split(' ')[2]))

        stim_df.loc[:, 'stim_index'] = stim_id_array
        if stim_df.stim_index.max() <= 0:
            print('no trials started')
            return

        ntime_array = ntime_array[1:]

        t_counter = []
        for t in np.diff(ntime_array):
            t_counter.append(t.total_seconds())

        t_ascending = np.cumsum(t_counter)
        t_ascending = np.insert(t_ascending, 0, 0)

        time_array = []
        rawt_array = []
        for i in range(len(times)):
            try:
                # time_array.append((ntime_array[i + 1] - ntime_array[i]).total_seconds())
                rawt_array.append(str(ntime_array[i])[11:])
            except:
                pass

        final_stims = final_stims.iloc[1:]

        final_stims.loc[:, 't'] = t_ascending
        final_stims.loc[:, 'raw_t'] = rawt_array

        stimuli = final_stims

        stim_changes = stimuli.stim_type
        #stim_changes = np.where(allstims[:-1] != allstims[1:])
        #stim_changes = stim_changes[0]

        # start and stop times for stimuli
        s = []
        #e = [] #minel - commented out because we don't need it
        # stimuli themselves
        stims = []
        moving = []
        # fill in the starts and stops
        #last = 0 #minel - commented out because we don't need it
        for i in range(len(stim_changes)): # minel - changed to range of length of stim_changes instead of just stim_changes
            #     if i == stim_changes[0]:
            #         print('centering', stim_df.loc[0].time, stim_df.loc[i].time)
            """if last is not None: #minel - commented out because we don't need the last variable
                # print('centering fish', last, stim_df.loc[i+1].time)
                s.append(last)
                e.append(stimuli.iloc[i + 1].t)
                stims.append(-1)
                moving.append(np.nan)"""
            if stimuli.iloc[i].stim_type == 'centering':
                # print('stim', stim_df.loc[i+1].stim_index, stim_df.loc[i+1].time, stim_df.loc[i+2].time)
                s.append(stimuli.iloc[i + 1].t)
                # try: #minel - commented out because we don't need the e list
                #     e.append(stimuli.iloc[i + 2].t)
                # except KeyError:
                #     e.append(len(stimuli))
                stims.append(stimuli.iloc[i + 1].stim_index) #minel - changed i+1 to i
                moving.append(0)
                last = None
            elif stimuli.iloc[i].stim_type != 'centering':
                # print('stim:', stim_df.loc[i].stim_index, stim_df.loc[i].time, stim_df.loc[i+1].time)
                s.append(stimuli.iloc[i].t)
                #e.append(stimuli.iloc[i + 1].t) #minel - commented out because we don't need the e list
                stims.append(stimuli.iloc[i].stim_index) #minel - changed i+1 to i
                moving.append(1)
                #last = stimuli.iloc[i + 1].t  #minel - commented out because we don't need the last variable
        last_t = (stimuli[-1:].t + stimuli[-1:].duration - stimuli[-1:].stat_time).values[0] # minel - added the final t of experiment

        behave_df = pd.read_csv(behavior_path,  sep=';', dtype=np.float32)
        
        # iterate through starts/stops to label stims
        for i in range(len(s)):
            try:
                row_s = behave_df[behave_df.t >= s[i]].index[0]
            except IndexError:
                break
            try:
                row_e = behave_df[behave_df.t >= s[i + 1]].index[0] - 1 # minel - changed from e[i] to s[i + 1], added -1 at the end because loc is inclusive of end index
            except IndexError:
                row_e = behave_df[behave_df.t >= last_t].index[0] - 1 # minel - so that remaining stytra tracking values are eliminated
            behave_df.loc[row_s:row_e, 'stim_index'] = stims[i]
            behave_df.loc[row_s:row_e, 'motion'] = moving[i]

        try:
            behave_df = behave_df[behave_df.stim_index.notna()]
        except AttributeError:
            behave_df.loc[:, 'stim_index'] = -1
            behave_df.loc[:, 'motion'] = 0

        # subset of stimuli df without all the duplicates and without most rows
        # may need velocities and things depending on protocol
        #minel - commented out due to angle_0 and angle_1 errors
        #smol_stim_df = stimuli.loc[:,
        #               ['stim_index', 'stim_type', 'angle', 'angle_0', 'angle_1', 'stim_name']].drop_duplicates(
        #    subset='stim_index', keep='first')
        smol_stim_df = stimuli.loc[:,
                       ['stim_index', 'stim_type', 'angle', 'stim_name', 'stat_time', 'duration']].drop_duplicates(
            subset='stim_index', keep='first')

        fulldf = pd.merge(behave_df, smol_stim_df, on='stim_index', how='left')

        fulldf.loc[:, 'stim_type'] = fulldf.stim_type.astype('category')
        fulldf.loc[:, 'stim_name'] = fulldf.stim_name.astype('category')
        fulldf.loc[:, 'fish_id'] = fish_id

        fulldf.fish_id = pd.Series(fulldf.fish_id.values, dtype='int16')
        fulldf.angle = pd.Series(fulldf.angle.values, dtype='int16')
        #minel - commented out due to angle_0 and angle_1 errors
        #fulldf.angle_0 = pd.Series(fulldf.angle_0.values, dtype='float16')
        #fulldf.angle_1 = pd.Series(fulldf.angle_1.values, dtype='float16')
        fulldf.motion = pd.Series(fulldf.motion.values, dtype='float16')

        fulldf.drop('Unnamed: 0', axis=1, inplace=True)

        # fulldf = fulldf[fulldf.f0_x.notna()] # minel - removed so that the actual duration can be calculated later

        fulldf = bout_applier(fulldf, metadata_path)

        loc_fish_details = fish_details.replace(':', '.')

        saving_path1 = str(parent_folder) + '/' + loc_fish_details + '_main_df' + '.h5'
        #saving_path1 = saving_path1.replace(' ', '_')
        print(saving_path1)

        fulldf.to_hdf(saving_path1, 'main_df', format='table', mode='w') # minel - added 'w' mode to not keep deleting the files when re-running

        return fulldf


def bout_applier(df, metadata_path):
    exp = StytraBouts(metadata_path)
    bouts, cont = extract_bouts(exp)
    for i in range(len(bouts)):
        s, e = bouts[i].iloc[[0, -1]].index
        df.loc[s:e, 'bout'] = i
    return df


class StytraBouts(dict):
    """
    Parameters
    ----------
    path :
    Returns
    -------
    """

    log_mapping = dict(
        stimulus_param_log=["dynamic_log", "stimulus_log", "stimulus_param_log"],
        estimator_log=["estimator_log"],
        behavior_log=["tracking_log", "log", "behavior_log"],
    )

    def __init__(self, path, session_id=None):
        # Prepare path:
        inpath = Path(path)

        if inpath.suffix == ".json":
            self.path = inpath.parent
            session_id = inpath.name.split("_")[0]

        else:
            self.path = Path(path)

            if session_id is None:
                meta_files = list(self.path.glob("*metadata.json"))

                # Load metadata:
                if len(meta_files) == 0:
                    raise FileNotFoundError("No metadata file in specified path!")
                elif len(meta_files) > 1:
                    raise FileNotFoundError(
                        "Multiple metadata files in specified path!"
                    )
                else:
                    session_id = str(meta_files[0].name).split("_")[0]

        self.session_id = session_id
        metadata_file = self.path / (session_id + "_metadata.json")

        source_metadata = json.load(open(metadata_file))

        # Temporary workaround:
        try:
            source_metadata["behavior"] = source_metadata.pop("tracking")
        except KeyError:
            pass

        super().__init__(**source_metadata)

        self._stimulus_param_log = None
        self._behavior_log = None
        self._estimator_log = None

    def _get_log(self, log_name):
        uname = "_" + log_name

        if getattr(self, uname) is None:
            for possible_name in self.log_mapping[log_name]:
                try:
                    logname = next(
                        self.path.glob(self.session_id + "_" + possible_name + ".*")
                    ).name
                    setattr(self, uname, self._load_log(logname))
                    break
                except StopIteration:
                    pass
            else:
                raise ValueError(log_name + " does not exist")

        return getattr(self, uname)

    @property
    def stimulus_param_log(self):
        return self._get_log("stimulus_param_log")

    @property
    def estimator_log(self):
        return self._get_log("estimator_log")

    @property
    def behavior_log(self):
        return self._get_log("behavior_log")

    def _load_log(self, data_name):
        """
        Parameters
        ----------
        data_name :
        Returns
        -------
        """

        file = self.path / data_name
        if file.suffix == ".csv":
            return pd.read_csv(str(file), delimiter=";").drop("Unnamed: 0", axis=1)
        elif file.suffix == ".h5" or file.suffix == ".hdf5":
            return pd.read_hdf(file)
        elif file.suffix == ".feather":
            return pd.read_feather(file)
        elif file.suffix == ".json":
            return pd.read_json(file)
        else:
            raise ValueError(
                str(data_name) + " format is not supported, trying to load " + str(file)
            )

    def stimulus_starts_ends(self):
        starts = np.array([stim["t_start"] for stim in self["stimulus"]["log"]])
        ends = np.array([stim["t_stop"] for stim in self["stimulus"]["log"]])
        return starts, ends

    @staticmethod
    def resample(df_in, resample_sec=0.005):
        """
        Parameters
        ----------
        df_in :
        resample_sec :
        Returns
        -------
        """
        df = df_in.copy()
        t_index = pd.to_timedelta(
            (df["t"].as_matrix() * 10e5).astype(np.uint64), unit="us"
        )
        df.set_index(t_index - t_index[0], inplace=True)
        df = df.resample("{}ms".format(int(resample_sec * 1000))).mean()
        df.index = df.index.total_seconds()
        return df.interpolate().drop("t", axis=1)


# Functions for bout analysis:
def _fish_renames(i_fish, n_segments):
    return dict(
        {
            "f{:d}_x".format(i_fish): "x",
            "f{:d}_vx".format(i_fish): "vx",
            "f{:d}_y".format(i_fish): "y",
            "f{:d}_vy".format(i_fish): "vy",
            "f{:d}_theta".format(i_fish): "theta",
            "f{:d}_vtheta".format(i_fish): "vtheta",
        },
        **{
            "f{:d}_theta_{:02d}".format(i_fish, i): "theta_{:02d}".format(i)
            for i in range(n_segments)
        }
    )


def _fish_column_names(i_fish, n_segments):
    return [
               "f{:d}_x".format(i_fish),
               "f{:d}_vx".format(i_fish),
               "f{:d}_y".format(i_fish),
               "f{:d}_vy".format(i_fish),
               "f{:d}_theta".format(i_fish),
               "f{:d}_vtheta".format(i_fish),
           ] + ["f{:d}_theta_{:02d}".format(i_fish, i) for i in range(n_segments)]


def _rename_fish(df, i_fish, n_segments):
    return df.filter(["t"] + _fish_column_names(i_fish, n_segments)).rename(
        columns=_fish_renames(i_fish, n_segments)
    )


def _extract_bout(df, s, e, n_segments, i_fish=0, scale=1.0):
    bout = _rename_fish(df.iloc[s:e], i_fish, n_segments)
    # scale to physical coordinates
    dt = (bout.t.values[-1] - bout.t.values[0]) / bout.shape[0]
    bout.iloc[:, 1:5] *= scale
    bout.iloc[:, 2:7:2] /= dt
    return bout


def extract_bouts(metadata, max_interpolate=2, window_size=7, recalculate_vel=False, scale=0.12, filter_nan=True, clip=False, **kwargs):
    """ Splits a dataframe with fish tracking into bouts
    :param metadata_file: the path of the metadata file
    :param max_interpolate: number of points to interpolate if surrounded by NaNs in trackign
    :param max_frames: the maximum numbers of frames to process, useful for debugging
    :param threshold: velocity threshold
    :param min_duration: minimal number of frames for a bout
    :param pad_before: number of frames that gets added before
    :param pad_after: number of frames added after
    :return: list of single bout dataframes
    """

    df = metadata.behavior_log
    # if clip:
    #     df = df[(200 < df.f0_x) & (df.f0_x < 950) & (200 < df.f0_y) & (df.f0_y < 950)]
    scale = scale or get_scale_mm(metadata)

    n_fish = get_n_fish(df)
    n_segments = get_n_segments(df)
    dfint = df.interpolate("linear", limit=max_interpolate, limit_area="inside")
    bouts = []
    continuous = []
    for i_fish in range(n_fish):
        if recalculate_vel:
            for thing in ["x", "y", "theta"]:
                dfint["f{}_v{}".format(i_fish, thing)] = np.r_[
                    np.diff(dfint["f{}_{}".format(i_fish, thing)]), 0
                ]

        vel = dfint["f{}_vx".format(i_fish)] ** 2 + dfint["f{}_vy".format(i_fish)] ** 2
        vel = vel.rolling(window=window_size, min_periods=1).median()
        bout_locations, continuity = extract_segments_above_thresh(vel.values, **kwargs)
        all_bouts_fish = [
            _extract_bout(dfint, s, e, n_segments, i_fish, scale)
            for s, e in bout_locations
        ]
        bouts.extend(all_bouts_fish)
        continuous.extend(continuity)

    return bouts, np.array(continuous)


@jit(nopython=True)
def extract_segments_above_thresh(vel, threshold=0.1, min_duration=20, pad_before=12, pad_after=25, skip_nan=True):
    """ Useful for extracing bouts from velocity or vigor
    :param vel:
    :param threshold:
    :param min_duration:
    :param pad_before:
    :param pad_after:
    :return:
    """
    bouts = []
    in_bout = False
    start = 0
    connected = []
    continuity = False
    i = pad_before + 1
    bout_ended = pad_before
    while i < vel.shape[0] - pad_after:
        if np.isnan(vel[i]):
            continuity = False
            if in_bout and skip_nan:
                in_bout = False

        elif i > bout_ended and vel[i - 1] < threshold < vel[i] and not in_bout:
            in_bout = True
            start = i - pad_before

        elif vel[i - 1] > threshold > vel[i] and in_bout:
            in_bout = False
            if i - start > min_duration:
                bouts.append((start, i + pad_after))
                bout_ended = i + pad_after
                if continuity:
                    connected.append(True)
                else:
                    connected.append(False)
            continuity = True

        i += 1

    return bouts, connected


def get_scale_mm(metadata):
    cal_params = metadata["stimulus"]["calibration_params"]
    proj_mat = np.array(cal_params["cam_to_proj"])
    return np.linalg.norm(np.array([1.0, 0.0]) @ proj_mat[:, :2]) * cal_params["mm_px"]


def get_n_segments(df, prefix=True):
    if prefix:

        def _tail_part(s):
            ps = s.split("_")
            if len(ps) == 3:
                return ps[2]
            else:
                return 0

    else:

        def _tail_part(s):
            ps = s.split("_")
            if len(ps) == 2:
                return ps[1]
            else:
                return 0

    tpfn = np.vectorize(_tail_part, otypes=[int])
    return np.max(tpfn(df.columns.values)) + 1


def get_n_fish(df):
    def _fish_part(s):
        ps = s.split("_")
        if len(ps) == 3:
            return ps[0][1:]
        else:
            return 0

    tpfn = np.vectorize(_fish_part, otypes=[int])
    return np.max(tpfn(df.columns.values)) + 1


def reduce_to_pi(ar):
    """Reduce angles to the -pi to pi range"""
    return np.mod(ar + np.pi, np.pi * 2) - np.pi


def angle_mean(angles, axis=1):
    """Correct calculation of a mean of an array of angles
    """
    return np.arctan2(np.sum(np.sin(angles), axis), np.sum(np.cos(angles), axis))


def rot_mat(theta):
    """The rotation matrix for an angle theta """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


@jit(nopython=True)
def smooth_tail_angles(tail_angles):
    """Smooths out the tau jumps in tail angles, so that the angle between
    tail segments is smoothly changing
    Parameters
    ----------
    tail_angles :
        return:
    Returns
    -------
    """

    tau = 2 * np.pi

    for i in range(1, tail_angles.shape[0]):
        previous = tail_angles[i - 1]
        dist = np.abs(previous - tail_angles[i])
        if np.abs(previous - (tail_angles[i] + tau)) < dist:
            tail_angles[i] += tau
        elif np.abs(previous - (tail_angles[i] - tau)) < dist:
            tail_angles[i] -= tau

    return tail_angles


def normalise_bout(bout):
    dir_init = angle_mean(bout.f0_theta.iloc[0:2], axis=0)
    coord = bout[["f0_x", "f0_y", "f0_theta"]].values
    coord[:, :2] = (coord[:, :2] - coord[:1, :2]) @ rot_mat(dir_init + np.pi)
    coord[:, 2] -= dir_init
    coord[:, 2] = reduce_to_pi(coord[:, 2])
    return coord
