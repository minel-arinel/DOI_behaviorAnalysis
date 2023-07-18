
# This code is written by Matthew D. Loring.
# Changes to the code by Minel Arinel have been commented with the tag 'minel'

import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import caiman as cm
except:
    print('no caiman available')

import os

try:
    from tqdm.auto import tqdm
except:
    print('no tqdm available')

from datetime import datetime as dt
from pathlib import Path
try:
    from pandastim.textures import RadialSinCube
except:
    print('no pandastim available')

__spec__ = None


## This takes in data from xx liu in excel and spits it into a subfolder with graphs by sheet

def excel_to_histograms(data_path):
    savingDir = Path(data_path).parents[0].joinpath(f"data_{str(dt.now().date()).replace('-', '_')}")
    try:
        os.mkdir(savingDir)
    except FileExistsError:
        print('aready created folder')
        pass

    file = pd.ExcelFile(data_path)

    for stim in file.sheet_names:

        data = file.parse([stim])[stim].loc[1:]

        header = data.loc[1]
        data.columns = header
        data = data[1:]
        df = data.loc[:, ~data.columns.isna()]

        _bins = np.arange(-100, 100, 1.5)
        _hist, bins = np.histogram(df["Turning angle (°)"], _bins)

        plt.plot(bins[1:], _hist)
        plt.title(stim)
        plt.savefig(savingDir.joinpath(f"{stim.replace('-', '_')}.jpg"))
        plt.show()


def radial_wrapper(slice_chunk, phase_change, output):
    winSize = (1024, 1024)
    outputs = []
    for item in slice_chunk:
        _phase = item * phase_change
        outputs.append(RadialSinCube(winSize, phase=_phase))
    output.put([slice_chunk[0], outputs])
    return


def return_rad_sin(n=10):
    slices = np.arange(0, 190)
    phase_change = 0.1

    all_slices = [slices[i:i + n] for i in range(0, len(slices), n)]

    data_out = mp.Queue()

    tasks = [mp.Process(target=radial_wrapper, args=(all_slices[i], phase_change, data_out,)) for i in
             range(len(all_slices))]
    [p.start() for p in tasks]
    results = [data_out.get() for p in tasks]
    [p.join() for p in tasks]
    results.sort(key = lambda results: results[0])
    results = [i[1:] for i in results]
    results = [item for sublist in results for item in sublist]
    results = [item for sublist in results for item in sublist]
    return results


def pseudo_random(df, col):
    randomized = False
    while not randomized:
        xlist = df.sample(frac=1).reset_index(drop=True) # where xlistbase is the original file read in
        # check for repeats
        for i in range(0, len(xlist)):
            try:
                if i == len(xlist) - 1:
                    randomized = True
                elif xlist[col][i] != xlist[col][i+1]:
                    continue
                elif xlist[col][i] == xlist[col][i+1]:
                    break
            except IndexError:
                pass
    return xlist


def stimTimersMP(df, outputs):
    # this takes in dataframes with a single fish ID
    df.loc[:,'stim_time'] = 0
    for ind in tqdm(df.stim_index.unique(), 'Fish trials', position=0, leave=True):
        sub = df[df.stim_index==ind]
        inds = sub.index
        alltimes = sub.t.values
        elapsed = alltimes[-1] - alltimes[0]
        stim_timer = np.linspace(0,elapsed,num=len(alltimes))
        df.loc[inds, 'stim_time'] = stim_timer
    outputs.put(df)
    return


def cumThetaMP(df, outputs):
    df.loc[:, 'cum_theta'] = 0

    for ind in tqdm(df.stim_index.unique(), 'Fish trials', position=0, leave=True):
        sub = df[df.stim_index==ind]
        inds = sub.index
        thetas = sub.f0_vtheta.values
        dsts = sub.f0_vx.values
        cum_theta = np.cumsum(thetas)
        cum_dst = np.cumsum(dsts)

        df.loc[inds, "cum_theta"] = cum_theta
        df.loc[inds, "cum_dst"] = cum_dst

    outputs.put(df)
    return


def mpWrapper(fxn, fullDataframe, big=False):
    # takes full dataframes and splits on ID
    dfs = []
    #minel - changed to fish_id from ID
    for i in fullDataframe.fish_id.unique():
        dfs.append(fullDataframe[fullDataframe.fish_id==i])

    output = mp.Queue()
    tasks = [mp.Process(target=fxn, args=(dfs[i],output,)) for i in range(len(dfs))]
    [p.start() for p in tasks]
    if not big:
        results = [output.get() for p in tasks]
        [p.join() for p in tasks]
        if len(results) == 0: #minel - added an if statement for bad recordings with no bouts
            return
        else:
            return pd.concat(results).sort_index()
    else:
        results = [output.get() for p in tasks]
        [p.join() for p in tasks]
        inds = []
        kept_ts = []
        for i in results:
            inds.append(i[0])
            kept_ts.append(i[1])

        for d in range(len(dfs)):
            i = dfs[d].fish_id.values[0] #minel - changed to fish)id from ID
            ind = np.where(inds == i)[0][0]

            dfs[d] = dfs[d][dfs[d].stim_index.isin(kept_ts[ind])]

        return pd.concat(dfs).sort_index()



def t_tot_finder(df, bouts, thresh=3, min_time=5):
    # feed only one ID
    # thresh in seconds, the maximum gap between start/stop
    # minimum trial time in seconds

    last_bout = df[df.bout == bouts.bout.values[-1]].stim_time.values[-1]
    first_bout = df[df.bout == bouts.bout.values[0]].stim_time.values[0]
    motion_on = df[df.motion == 1].stim_time.values[0]
    trial_end = df[df.f0_x.notna()].stim_time.values[-1]

    if trial_end - motion_on <= min_time:
        return np.nan
    if abs(first_bout - motion_on) <= thresh:
        start_offset = 0
    else:
        start_offset = first_bout - motion_on - thresh
    if abs(last_bout - trial_end) <= thresh:
        end_offset = 0
    else:
        end_offset = abs(last_bout - trial_end + thresh)

    offsets = start_offset + motion_on + end_offset
    fin_time = trial_end - offsets
    if fin_time <= min_time:
        return np.nan
    else:
        return fin_time


def histmaker(boutframe, dataframe, stims=['forward'], ):
    if 'stim_time' not in dataframe.columns:
        df = mpWrapper(stimTimersMP, dataframe)
    else:
        df = dataframe

    __sub = boutframe[boutframe.stim_name.isin(stims)]
    __sub = __sub[__sub.bout_angle.notna()]

    stim_times = []
    hists = []

    _bins = np.arange(-100, 100, 1.5)

    for i in tqdm(__sub.ID.unique(), 'fish done', position=0, leave=True):
        _sub = __sub[__sub.ID == i]
        for stim in _sub.stim_index.unique():
            sub = _sub[_sub.stim_index == stim]
            if len(sub) <= 1:
                continue
            df_sub = df[(df.ID == sub.ID.values[0]) & (df.stim_index == sub.stim_index.values[0])]
            try:
                stim_timer = (t_tot_finder(df_sub, sub))
            except IndexError:
                continue
            if stim_timer is np.nan:
                hists.append(np.zeros(len(_bins) - 1))
            else:
                _hist, bins = np.histogram(sub.bout_angle.values, _bins)
                hist = (1000 * _hist) / stim_timer
                hists.append(hist)

    finhists = np.average(hists, axis=0)
    finstds = np.std(hists, axis=0) / len(hists)
    finbins = _bins
    try:
        if np.isnan(finhists):
            finhists = np.zeros(len(_bins) - 1)
            finstds = np.zeros(len(_bins) - 1)
            return finhists, finstds, finbins
    except ValueError:
        return finhists, finstds, finbins


def tolerant_mean(arrs):
    # https://stackoverflow.com/questions/10058227/calculating-mean-of-arrays-with-different-lengths
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def trialExcluderMP(df, output):
    good_trials = []
    trials = df.stim_index.unique()
    for t in trials:
        subdf = df[(df.stim_index==t)&(df.f0_x.notna())]
        diffvals = np.diff(subdf.f0_x.values)
        if np.std(diffvals)>=15:
            continue
        else:
            good_trials.append(t)
    output.put([df.fish_id.values[0], good_trials])
    return


def raw_text_frametimes_to_df(time_path):
    with open(time_path) as file:
        contents = file.read()
    parsed = contents.split('\n')

    times = []
    for line in range(len(parsed) - 1):
        times.append(dt.strptime(parsed[line], '%H:%M:%S.%f').time())
    return pd.DataFrame(times)


def raw_text_logfile_to_df(log_path, frametimes=None):
    with open(log_path) as file:
        contents = file.read()
    split = contents.split('\n')

    movesteps = []
    times = []
    for line in range(len(split)):
        if 'piezo' in split[line] and 'connected' not in split[line] and 'stopped' not in split[line]:
            t = split[line].split(' ')[0][:-1]
            z = split[line].split(' ')[6]
            try:
                if isinstance(eval(z), float):
                    times.append(dt.strptime(t, '%H:%M:%S.%f').time())
                    movesteps.append(z)
            except NameError:
                continue
    else:
        # last line is blank and likes to error out
        pass
    log_steps = pd.DataFrame({'times': times, 'steps': movesteps})

    if frametimes is not None:
        log_steps = log_aligner(log_steps, frametimes)
    else:
        pass
    return log_steps


def log_aligner(logsteps, frametimes):
    # this just trims the dataframe to only include the steps included in our frametimes
    trimmed_logsteps = logsteps[(logsteps.times >= frametimes.iloc[0].values[0])&(logsteps.times <= frametimes.iloc[-1].values[0])]
    return trimmed_logsteps


def volumeSplitter(logPath, frametimePath, imgPath, leadingFrame=False, extraStep=False, intermediate_return=False):
    frametimes = raw_text_frametimes_to_df(frametimePath)
    logfile = raw_text_logfile_to_df(logPath, frametimes)
    img = cm.load(imgPath)

    if leadingFrame:
        img = img[1:]
        frametimes = frametimes.loc[1:]

    if intermediate_return:
        return frametimes, logfile, img

    if extraStep:
        n_imgs = logfile.steps.nunique() - 1
    else:
        n_imgs = logfile.steps.nunique()

    imgs = [[]] * n_imgs
    frametime_all = [[]] * n_imgs

    imgpaths = []
    frametime_paths = []

    root_path = Path(logPath).parents[0].joinpath('planes')

    try:
        os.mkdir(root_path)
    except FileExistsError:
        pass

    x=0
    for i in range(n_imgs):
        new_img = img[i::n_imgs]
        new_img_frametime = frametimes.iloc[1::n_imgs]

        imgs[i] = new_img
        frametime_all[i] = new_img_frametime

        new_img_path = root_path.joinpath(f'{x}.tif')
        new_framet_path = root_path.joinpath(f'{x}_frametimes.h5')

        imgpaths.append(new_img_path)
        frametime_paths.append(new_framet_path)

        new_img.save(new_img_path)
        new_img_frametime.to_hdf(new_framet_path, 'frametimes')

        print(f'saved {new_img_path}')
        print(f'saved {new_framet_path}')
        x += 1

    return [imgs, frametime_all], [imgpaths, frametime_paths]


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


def normalise_bout(bout):
    dir_init = angle_mean(bout.f0_theta.iloc[0:2], axis=0)
    coord = bout[["f0_x", "f0_y", "f0_theta"]].values
    coord[:, :2] = (coord[:, :2] - coord[:1, :2]) @ rot_mat(dir_init + np.pi)
    coord[:, 2] -= dir_init
    coord[:, 2] = reduce_to_pi(coord[:, 2])
    return coord


def bout_extactor(df):
    dsts = []
    thetas = []
    stim_index = []
    stim_names = []
    ids = []
    time_tots = []
    bout_labels = []

    _boutdf = df[df.bout.notna()]


    for id in _boutdf.fish_id.unique(): #minel - changed to fish_id
        boutdf = _boutdf[_boutdf.fish_id==id] #minel - changed to fish_id
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
                    _id = _bout_data.fish_id.values[0] #minel - changed to fish_id

                    ts = sub[(sub.stim_index == _stim_index)].stim_time.values
                    final = normalise_bout(sub)[-1]

                    time_tots.append(ts[-1] - ts[0])
                    dsts.append(np.linalg.norm(final[0:2]))
                    thetas.append(final[-1] * 180 / np.pi)
                    stim_index.append(_stim_index)
                    stim_names.append(_stim_name)
                    bout_labels.append(b)
                    ids.append(_id)



        else:
            continue

    extracted_df = pd.DataFrame({'stim_index': stim_index,
                                 'stim_name': stim_names,
                                 'bout_angle': thetas,
                                 'distance': dsts,
                                 't_tot': time_tots,
                                 'bout': bout_labels,
                                 'fish_id' : ids})
    return extracted_df