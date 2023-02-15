import sys
import os
from pstim_behavior2 import main, Hist, preHist, _extactor, Hist2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
from bioinfokit.analys import stat
import statsmodels.api as sm
from statsmodels.formula.api import ols
from random import choice

roi = (1216, 1216)  # roi of camera
x0 = roi[0] / 2  # x-coord of the center
y0 = roi[1] / 2  # y-coord of the center
radius = min(x0, y0) / np.sqrt(2)

# Eliminate bouts with durations shorter than 100 ms and longer than 1 second (control for tracking issues)
bout_duration_lowerthreshold = 0.1
bout_duration_upperthreshold = 1

# Eliminate fish with average bout count per min lower than 10 (control for non-moving fish)
# and higher than 120 (control for tracking issues)
bout_count_lowerthreshold = 5
bout_count_upperthreshold = 120

treatment_duration = 0  # The duration of drug treatment (in sec)
bin_duration = 60  # The duration of each bin for analysis (in sec)
omr_bin_angle = 3  # The bin width for bout angle histograms (in degrees)
bout_duration_bin = 0.02  # The duration of each bin for bout duration histograms (in sec)

conditions = ['baseline', 'drugtreated']


def create_newdf(parent_folder):
    dfs = []

    with os.scandir(parent_folder) as entries:
        for entry in entries:
            if entry.is_dir():
                fish_path = os.path.join(parent_folder, entry)
                for cond in conditions:
                    finalpath = os.path.join(fish_path, cond)
                    fulldf = main(finalpath)
                    if fulldf is not None:
                        finaldf = preHist(fulldf)
                        if finaldf is not None:
                            finaldf['condition'] = cond
                            dfs.append(finaldf)

    df = pd.concat(dfs, ignore_index=True)
    df_path = os.path.join(parent_folder, parent_folder[parent_folder.rfind('\\') + 1:] + '_finaldf.h5')
    df.to_hdf(df_path, 'finaldf', format='table', mode='w')
    print("Experiment dataframe is saved")
    return


def create_newboutdf(parent_folder):
    df_path = os.path.join(parent_folder, parent_folder[parent_folder.rfind('\\') + 1:] + '_finaldf.h5')
    df = pd.read_hdf(df_path)

    concentrations = []
    ages = []
    ids = []
    conds = []
    stim_names = []
    stim_indices = []
    bout_labels = []
    dsts = []
    thetas = []
    bout_starts = []
    bout_durations = []
    cum_bout_starts = []
    timebin_inds = []
    stim_durations = []
    stat_times = []
    dist_from_centers = []
    in_centers = []

    conc = parent_folder[parent_folder.rfind('DOI') + 4: parent_folder.rfind('ugml') + 4]
    age = int(parent_folder[:parent_folder.rfind('dpf')][parent_folder[:parent_folder.rfind('dpf')].rfind('_') + 1:])

    # durations calculated by the sum of each stim's duration
    baseline_duration = np.sum(
        [df[(df.condition == 'baseline') & (df.stim_index == _stim)].duration.values[0] for _stim in
         df[df.condition == 'baseline'].stim_index.unique()])
    drugtreated_duration = np.sum(
        [df[(df.condition == 'drugtreated') & (df.stim_index == _stim)].duration.values[0]
         for _stim in df[df.condition == 'drugtreated'].stim_index.unique()])
    total_duration = baseline_duration + treatment_duration + drugtreated_duration
    timebins = np.arange(0, total_duration + bin_duration + 1, bin_duration)

    for _id in df.fish_id.unique():
        cum_time = 0
        for _cond in conditions:
            conddf = df[(df.fish_id == _id) & (df.condition == _cond)]
            if _cond == 'drugtreated':
                cum_time += treatment_duration
            for stim in conddf.stim_index.unique():
                stimdf = conddf[conddf.stim_index == stim]
                stim_duration = stimdf.duration.values[0]
                boutdf = stimdf[stimdf.bout.notna()]
                added = False

                for b in boutdf.bout.unique():
                    _bout_data = boutdf[(boutdf.bout == b) & (boutdf.f0_x.notna())]
                    last_indx = _bout_data.f0_x.last_valid_index()
                    last_indy = _bout_data.f0_y.last_valid_index()
                    if last_indx and last_indy is not None:
                        last_ind = min(last_indx, last_indy)
                        sub = _bout_data.loc[:last_ind]
                        bout_duration = sub.stim_time.values[-1] - sub.stim_time.values[0]
                        bout_angle = (sub.f0_theta.values[-1] - sub.f0_theta.values[
                            0]) * 180 / np.pi  # minel - changed this from cum_theta to f0_theta because cum_theta was the cumulative of f0_vtheta
                        if len(sub) >= 5 and bout_duration_lowerthreshold <= bout_duration <= bout_duration_upperthreshold and -180 <= bout_angle <= 180:
                            ts = sub.stim_time.values
                            bout_starts.append(ts[0])
                            bout_durations.append(bout_duration)

                            dst1 = [sub.f0_x.values[0], sub.f0_y.values[0]]
                            dst2 = [sub.f0_x.values[-1], sub.f0_y.values[-1]]
                            dst = np.sqrt((dst1[0] - dst2[0]) ** 2 + (dst1[1] - dst2[1]) ** 2)
                            dsts.append(dst)
                            dstfromcenter = np.sqrt((x0 - dst2[0]) ** 2 + (y0 - dst2[1]) ** 2)
                            dist_from_centers.append(dstfromcenter)
                            thetas.append(bout_angle)
                            stim_names.append(_bout_data.stim_name.values[0])
                            stim_indices.append(stim)
                            conds.append(_cond)
                            bout_labels.append(b)
                            ids.append(_id)
                            concentrations.append(conc)
                            ages.append(age)
                            stim_durations.append(stim_duration)
                            stat_times.append(_bout_data.stat_time.values[0])
                            cum_bout_starts.append(ts[0] + cum_time)
                            timebin_inds.append(np.digitize(ts[0] + cum_time, timebins))

                            i = 0
                            center = False
                            while i < len(sub['f0_x'].values):
                                if np.sqrt((sub['f0_x'].values[i] - x0) ** 2 + (
                                        sub['f0_y'].values[i] - y0) ** 2) <= radius:
                                    center = True
                                    break
                                i += 1
                            in_centers.append(center)
                            added = True

                if not added:
                    concentrations.append(conc)
                    ages.append(age)
                    ids.append(_id)
                    conds.append(_cond)
                    stim_names.append(stimdf.stim_name.values[0])
                    stim_indices.append(stim)
                    bout_labels.append(None)
                    dsts.append(None)
                    thetas.append(None)
                    bout_starts.append(None)
                    bout_durations.append(None)
                    cum_bout_starts.append(None)
                    timebin_inds.append(None)
                    stim_durations.append(stim_duration)
                    stat_times.append(stimdf.stat_time.values[0])
                    dist_from_centers.append(min(x0, y0))
                    in_centers.append(False)

                cum_time += stim_duration
            print(f"Bouts done for fish {_id}, condition {_cond}")

    boutdf = pd.DataFrame({'concentration': concentrations, 'age': ages, 'fish_id': ids,
                           'condition': conds, 'stim_name': stim_names, 'stim_index': stim_indices, 'bout': bout_labels,
                           'distance': dsts, 'dist_from_center': dist_from_centers, 'in_center': in_centers,
                           'bout_angle': thetas, 'bout_duration': bout_durations, 'bout_start': bout_starts,
                           'cum_bout_start': cum_bout_starts, 'timebin_ind': timebin_inds,
                           'stim_duration': stim_durations,
                           'stat_time': stat_times})

    boutdf.in_center = boutdf.in_center.astype(bool)
    bout_path = os.path.join(parent_folder, parent_folder[parent_folder.rfind('\\') + 1:] + '_finalboutdf.h5')
    '''boutstore = pd.HDFStore(bout_path, 'w')
    boutstore.put('finalboutdf', boutdf, format='table')
    boutstore.get_storer('finalboutdf').attrs.metadata = {'baseline_duration': baseline_duration,
                                                          'drugtreated_duration': drugtreated_duration,
                                                          'total_duration': total_duration}
    boutstore.close()'''
    boutdf.to_hdf(bout_path, 'finalboutdf', format='table', mode='w')
    print("Bout dataframe is saved")
    return


def create_concdf(data_folder, DOI_conc, good_fish):
    # Update combined dfs

    good_dfs = []
    good_boutdfs = []

    if len(good_fish) == 0:
        return

    for fish in good_fish:
        for root, dirs, files in os.walk(data_folder):
            if str(fish) in dirs:
                good_dfs.extend([os.path.join(root, file) for file in os.listdir(root) if
                                 'finaldf.h5' in file and os.path.join(root, file) not in good_dfs])
                good_boutdfs.extend([os.path.join(root, file) for file in os.listdir(root) if
                                     'finalboutdf.h5' in file and os.path.join(root, file) not in good_boutdfs])

    good_df = pd.concat([pd.read_hdf(df) for df in good_dfs], ignore_index=True)
    good_df = good_df[good_df.fish_id.isin(good_fish)]
    good_df.to_hdf(os.path.join(data_folder, f'{DOI_conc}_DOI_selected_finaldf.h5'), 'finaldf', format='table',
                   mode='w')
    good_boutdf = pd.concat([pd.read_hdf(boutdf) for boutdf in good_boutdfs], ignore_index=True)
    good_boutdf = good_boutdf[good_boutdf.fish_id.isin(good_fish)]
    good_boutdf.to_hdf(os.path.join(data_folder, f'{DOI_conc}_DOI_selected_finalboutdf.h5'), 'finalboutdf',
                       format='table', mode='w')
    return


def create_alldf(data_folder):
    # Combine selected fish boutdfs from different concentrations into one df

    concdfs = []
    with os.scandir(data_folder) as entries:
        for entry in entries:
            if entry.name.endswith('_selected_finalboutdf.h5'):
                concdf = pd.read_hdf(os.path.join(data_folder, entry.name))
                concdfs.append(concdf)
    concdf = pd.concat(concdfs, ignore_index=True)
    concdf.to_hdf(os.path.join(data_folder, 'alldf.h5'), 'alldf', format='table', mode='w')
    return


def measurepertime(boutdf, measure, time):
    # Measure can be either 'distance', 'boutcount'

    measures = ['distance', 'boutcount']
    if measure not in measures:
        print('Measure not available. Accepted measures are: ', measures)
        return

    times = ['min', 'sec']
    if time not in times:
        print('Time not available. Accepted times are: ', times)
        return
    elif time == 'min':
        t = 60
    elif time == 'sec':
        t = 1

    labels = list(boutdf.fish_id.unique())
    labels.sort()
    _dict = {_id: {_cond: {_stim: 0 for _stim in boutdf.stim_name.unique()} for _cond in conditions} for _id in labels}

    for _id in labels:
        for _cond in conditions:
            for _stim in boutdf.stim_name.unique():
                # only include the trials with bouts
                subdf = boutdf[(boutdf.fish_id == _id) & (boutdf.condition == _cond) & (boutdf.stim_name == _stim) &
                               (boutdf.bout.notna())]
                stim_duration = np.sum(
                    [subdf[subdf.stim_index == s].stim_duration.values[0] for s in subdf.stim_index.unique()])
                if measure == 'distance':
                    val = np.nan_to_num(subdf.distance.sum() / stim_duration) * t
                elif measure == 'boutcount':
                    val = np.nan_to_num(subdf.bout.count() / stim_duration) * t
                _dict[_id][_cond][_stim] = val

    _df = pd.DataFrame.from_dict({(_id, _cond): _dict[_id][_cond]
                                  for _id in _dict.keys() for _cond in _dict[_id].keys()})
    _df.measure = measure
    return _df



def get_stimulus_names(df):
    """takes in a pandas dataframe and returns a list of the stimuli (rows of dataframe) presented to the fish"""
    data_top = df.head()
    return list(data_top.index)   # [habituation, forward, right, left, backwards]


def barplot_perfish(folder, df, level, DOI_conc=0):
    # Plot bar graphs of values in a df per fish
    # Measure can be 'distance' or 'boutcount'
    # Level can be 'exp' or 'conc'

    measures = ['distance', 'boutcount']
    if df.measure not in measures:
        print('Measure not available. Accepted measures are: ', measures)
        return

    fish_ids = list(df.columns.get_level_values(0).unique())
    labels = [f'{_cond}; {_stim}' for _cond in df.columns.get_level_values(1).unique() for _stim in
              df.index]  # labels of bars
    num_bars = len(labels)  # number of bars to plot per fish

    width = 0.35  # the width of the bars
    stepsize = (num_bars + 3) * width  # distance between the center of fish labels
    x = np.arange(len(fish_ids) * stepsize, step=stepsize)  # the label locations

    fig, ax = plt.subplots(figsize=(num_bars * 3 / 2, 5))
    ax.set_xticks(x)
    ax.set_xticklabels(fish_ids)
    ax.set_xlabel('Fish ID', fontsize='x-large')

    good_fish = []
    inactive_fish = []
    """
    criteria for good_fish:
        - neither baseline nor drugtreated should cross upper bound -----DONE-----
        - baseline habituation, baseline forward, and either baseline left or right over lower bound
        
    criteria for inactive_fish:
        - neither baseline nor drugtreated should cross upper bound
        - fails baseline criteria for good_fish
    """

    """get the names of the actual treatments instead of index=0"""

    stimulus_lst = get_stimulus_names(df) #makes a call to get a list of all the stimulus names

    for i, _id in enumerate(fish_ids):
        """TODO: make this part below into a separate function"""
        index = 0
        fish_criteria_dict = {
            "below-upper": True,
            "habituation": True,
            "forward": True,
            "right-left": False
        }
<<<<<<< Updated upstream

        values = []
        for _cond in df[_id].columns:
            values.extend(df[_id][_cond].values)
            if _cond == 'baseline' and df.measure == 'boutcount':  # added the check for drugtreated as well
                for val in values:
=======
        for condition in df[_id].columns:
            data_values = df[_id][condition].values()
            if condition == 'baseline' and df.measure == 'boutcount':
                for value in data_values:
>>>>>>> Stashed changes
                    stimulus = stimulus_lst[index]
                    if value >= bout_count_upperthreshold:
                        fish_criteria_dict["below-upper"] = False
                    if (stimulus == "habituation" or stimulus == "forward") and value <= bout_count_lowerthreshold:
                        fish_criteria_dict[stimulus] = False
                    elif (stimulus == "left" or stimulus == "right") and value > bout_count_lowerthreshold:
                        fish_criteria_dict["right-left"] = True
                    index += 1
            if condition == 'drugtreated' and df.measure == 'boutcount':  # added the check for drugtreated as well
                for value in data_values:
                    if value >= bout_count_upperthreshold:
                        fish_criteria_dict["below-upper"] = False
            criteria_value_lst = list(fish_criteria_dict.values())
            if all(criteria_value_lst):
                # if all the criteria is met, add fish to good_fish list
                good_fish.append(_id)
            elif False in criteria_value_lst and criteria_value_lst[0] is not False:
                # if not all criteria is met (excluding being over upper bound), add fish to inactive fish
                inactive_fish.append(_id)

        for j, value in enumerate(vals):
            _x = x[i] - (width / 2) * ((num_bars - 1) - 2 * j)  # x coordinates of individual bars for fish
            if i == 0:  # only add labels for the first fish so that they are not repeated in the legend for every fish
                ax.bar(_x, vals[j], width, label=labels[j])
            else:
                ax.bar(_x, vals[j], width)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if df.measure == 'distance':
        ax.set_ylabel('Bout Distance per min (px/min)', fontsize='x-large')
        save_name = '_boutdist_fish.png'
    elif df.measure == 'boutcount':
        ax.set_ylabel('Bout Count per min', fontsize='x-large')
        plt.axhline(bout_count_upperthreshold, color='red', ls='dotted')
        plt.axhline(bout_count_lowerthreshold, color='red', ls='dotted')
        save_name = '_boutcount_fish.png'

    if level == 'exp':
        plt.title(folder[folder.rfind('\\') + 1:], fontsize='x-large', pad=10)
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + save_name), dpi=300, transparent=False)
    elif level == 'conc':
        plt.title(DOI_conc, fontsize='x-large', pad=10)
        plt.savefig(os.path.join(folder, DOI_conc + save_name), dpi=300, transparent=False)
    else:
        print('Cannot save figure, level should be "exp" or "conc"')

    return good_fish, inactive_fish

    '''below_baseline_habit = np.minimum(np.array(baseline_habits), bout_count_upperthreshold)
        below_baseline_loco = np.minimum(np.array(baseline_locos), bout_count_upperthreshold)
        below_drug_habit = np.minimum(np.array(drug_habits), bout_count_upperthreshold)
        below_drug_loco = np.minimum(np.array(drug_locos), bout_count_upperthreshold)

        above_baseline_habit = np.maximum(np.array(baseline_habits) - bout_count_upperthreshold, 0)
        above_baseline_loco = np.maximum(np.array(baseline_locos) - bout_count_upperthreshold, 0)
        above_drug_habit = np.maximum(np.array(drug_habits) - bout_count_upperthreshold, 0)
        above_drug_loco = np.maximum(np.array(drug_locos) - bout_count_upperthreshold, 0)

        rects1 = ax.bar(x - 1.5 * width, below_baseline_habit, width, label='Baseline; Habituation')
        rects2 = ax.bar(x - width / 2, below_baseline_loco, width, label='Baseline; Locomotion')
        rects3 = ax.bar(x + width / 2, below_drug_habit, width, label='DOI treated; Habituation')
        rects4 = ax.bar(x + 1.5 * width, below_drug_loco, width, label='DOI treated; Locomotion')

        rects5 = ax.bar(x - 1.5 * width, above_baseline_habit, width, color='black', bottom=below_baseline_habit)
        rects6 = ax.bar(x - width / 2, above_baseline_loco, width, color='black', bottom=below_baseline_loco)
        rects7 = ax.bar(x + width / 2, above_drug_habit, width, color='black', bottom=below_drug_habit)
        rects8 = ax.bar(x + 1.5 * width, above_drug_loco, width, color='black', bottom=below_drug_loco)'''


def lineplot_perfish_incubation(folder, boutdf, level, measure, DOI_conc=0):
    # Plot line graph of measure over time per fish
    # Level can be 'exp' or 'conc'
    # Measure can be 'distance' or 'dist_from_center'

    measures = ['distance', 'dist_from_center']
    if measure not in measures:
        print('Measure not available. Accepted measures are: ', measures)
        return

    labels = list(boutdf.fish_id.unique())
    labels.sort()
    fig, ax = plt.subplots(figsize=(20, 5))

    baseline_duration = np.sum(
        [boutdf[(boutdf.condition == 'baseline') & (boutdf.stim_index == _stim)].stim_duration.values[0]
         for _stim in boutdf[boutdf.condition == 'baseline'].stim_index.unique()])
    drugtreated_duration = np.sum(
        [boutdf[(boutdf.condition == 'drugtreated') & (boutdf.stim_index == _stim)].stim_duration.values[0]
         for _stim in boutdf[boutdf.condition == 'drugtreated'].stim_index.unique()])
    total_duration = baseline_duration + treatment_duration + drugtreated_duration

    timebins = np.arange(0, total_duration + bin_duration + 1, bin_duration)

    maxval = 0
    for fish in labels:
        iddf = boutdf[boutdf['fish_id'] == fish]
        if measure == 'distance':
            bin_vals = np.nan_to_num([iddf[iddf['timebin_ind'] == i].distance.sum() for i in range(1, len(timebins))])
        elif measure == 'dist_from_center':
            bin_vals = np.nan_to_num(
                [iddf[(iddf['timebin_ind'] == i) & (iddf.bout.notna())].dist_from_center.mean() for i in
                 range(1, len(timebins))])
        if max(bin_vals) > maxval:
            maxval = max(bin_vals)

        # get the last timebin_ind of baseline and first timebin_ind of drugtreated to determine masked regions
        mask_start = np.digitize(baseline_duration, timebins) - 1
        mask_end = np.digitize(baseline_duration + treatment_duration, timebins) - 1
        masked_vals = np.ma.array(bin_vals)
        masked_vals[mask_start:mask_end] = np.ma.masked
        plt.plot(timebins[1:], masked_vals, label=f'Fish {fish}')

    if measure == 'distance':
        ax.set_ylabel('Bout Distance (px)', fontsize='x-large')
        save_name = '_distovertime_perfish.png'
    elif measure == 'dist_from_center':
        ax.set_ylabel('Average Distance from Center (px)', fontsize='x-large')
        save_name = '_distfromcenterovertime_perfish.png'

    ax.set_xlabel('Time (s)', fontsize='x-large')

    # plt.axvline(baseline_duration, linestyle='dashed', color='black', linewidth=2)
    plt.axvspan(baseline_duration, baseline_duration + treatment_duration + bin_duration, color='blue', alpha=0.1)
    plt.text(baseline_duration + bin_duration + treatment_duration / 2, maxval / 2, 'DRUG TREATMENT',
             verticalalignment='center',
             horizontalalignment='center', rotation='vertical', fontsize='xx-large', color='navy')
    # plt.axvspan(habit_duration + baseline_loco_duration, 2 * habit_duration + baseline_loco_duration, color='blue',
    # alpha=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if level == 'exp':
        plt.title(folder[folder.rfind('\\') + 1:], fontsize='x-large', pad=10)
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + save_name), dpi=300, transparent=False)
    elif level == 'conc':
        plt.title(DOI_conc, fontsize='x-large', pad=10)
        plt.savefig(os.path.join(folder, DOI_conc + save_name), dpi=300, transparent=False)
    else:
        print('Cannot save figure, level should be "exp" or "conc"')
    return


def lineplot_perfish(folder, boutdf, level, measure, DOI_conc=0):
    # Plot line graph of measure over time per fish
    # Level can be 'exp' or 'conc'
    # Measure can be 'distance' or 'dist_from_center'

    measures = ['distance', 'dist_from_center']
    if measure not in measures:
        print('Measure not available. Accepted measures are: ', measures)
        return

    labels = list(boutdf.fish_id.unique())
    labels.sort()
    fig, ax = plt.subplots(figsize=(20, 5))

    baseline_duration = np.sum(
        [boutdf[(boutdf.condition == 'baseline') & (boutdf.stim_index == _stim)].stim_duration.values[0]
         for _stim in boutdf[boutdf.condition == 'baseline'].stim_index.unique()])
    drugtreated_duration = np.sum(
        [boutdf[(boutdf.condition == 'drugtreated') & (boutdf.stim_index == _stim)].stim_duration.values[0]
         for _stim in boutdf[boutdf.condition == 'drugtreated'].stim_index.unique()])
    total_duration = baseline_duration + treatment_duration + drugtreated_duration

    timebins = np.arange(0, total_duration + bin_duration + 1, bin_duration)

    maxval = 0
    for fish in labels:
        iddf = boutdf[boutdf['fish_id'] == fish]
        if measure == 'distance':
            bin_vals = np.nan_to_num([iddf[iddf['timebin_ind'] == i].distance.sum() for i in range(1, len(timebins))])
        elif measure == 'dist_from_center':
            bin_vals = np.nan_to_num(
                [iddf[(iddf['timebin_ind'] == i) & (iddf.bout.notna())].dist_from_center.mean() for i in
                 range(1, len(timebins))])
        if max(bin_vals) > maxval:
            maxval = max(bin_vals)

        # get the last timebin_ind of baseline and first timebin_ind of drugtreated to determine masked regions
        mask_start = np.digitize(baseline_duration, timebins) - 1
        mask_end = np.digitize(baseline_duration + treatment_duration, timebins) - 1
        masked_vals = np.ma.array(bin_vals)
        masked_vals[mask_start:mask_end] = np.ma.masked
        plt.plot(timebins[1:], masked_vals, label=f'Fish {fish}')

        plt.axvline(x=baseline_duration, color='b', ls='--')

    if measure == 'distance':
        ax.set_ylabel('Bout Distance (px)', fontsize='x-large')
        save_name = '_distovertime_perfish.png'
    elif measure == 'dist_from_center':
        ax.set_ylabel('Average Distance from Center (px)', fontsize='x-large')
        save_name = '_distfromcenterovertime_perfish.png'

    ax.set_xlabel('Time (s)', fontsize='x-large')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if level == 'exp':
        plt.title(folder[folder.rfind('\\') + 1:], fontsize='x-large', pad=10)
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + save_name), dpi=300, transparent=False)
    elif level == 'conc':
        plt.title(DOI_conc, fontsize='x-large', pad=10)
        plt.savefig(os.path.join(folder, DOI_conc + save_name), dpi=300, transparent=False)
    else:
        print('Cannot save figure, level should be "exp" or "conc"')
    return


def histplot(folder, df, level, measure):
    # Plot histogram of measure
    # Includes stat_time

    # determine the number of bins by calculating the square root of the number of data points
    df = df[df.bout.notna()]
    bin_n = ceil(np.sqrt(len(df[measure])))
    fig, ax = plt.subplots(figsize=(10, 5))

    for _cond in conditions:
        subdf = df[df.condition == _cond]
        tot_stimduration = np.sum([subdf[subdf.stim_index == s].stim_duration.values[0] for s in
                                   subdf.stim_index.unique()])
        cnts, bins = np.histogram(np.nan_to_num(subdf[measure]), bins=bin_n)
        time_normalized_cnts = cnts / tot_stimduration * 1000
        bin_centers = bins[:-1] + np.diff(bins)[0] / 2  # find the center points of bins to have a line plot
        ax.plot(bin_centers, gaussian_filter1d(time_normalized_cnts, sigma=1), label=_cond)

    ax.legend()
    ax.set_ylabel('Frequency (mHz)', fontsize='x-large')
    ax.set_xlabel(measure, fontsize='x-large')

    if level == 'exp':
        plt.title(folder[folder.rfind('\\') + 1:], fontsize='x-large', pad=10)
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + '_' + measure + '_histogram'), dpi=300,
                    transparent=False)


def histplot_perfish(folder, df, level, measure):
    # Plot histogram of measure per fish
    # Does not include stat_time

    # determine the number of bins by calculating the square root of the number of data points
    df = df[df.bout.notna()]
    bin_n = ceil(np.sqrt(max([len(df[df.fish_id == _id][measure]) for _id in df.fish_id.unique()])))
    fig, axs = plt.subplots(nrows=len(df.stim_name.unique()), ncols=len(df.fish_id.unique()), sharex=True, sharey=True,
                            figsize=(10 * len(df.fish_id.unique()), 5 * len(df.stim_name.unique())))
    stims = df.stim_name.unique()

    for j, _id in enumerate(df.fish_id.unique()):
        for _cond in conditions:
            for i, _stim in enumerate(stims):
                subdf = df[(df.fish_id == _id) & (df.condition == _cond) & (df.stim_name == _stim) & (
                        df.bout_start >= df.stat_time)]
                tot_stimduration = np.sum([subdf[subdf.stim_index == s].stim_duration.values[0] -
                                           subdf[subdf.stim_index == s].stat_time.values[0] for s in
                                           subdf.stim_index.unique()])
                cnts, bins = np.histogram(np.nan_to_num(subdf[measure]), bins=bin_n)
                time_normalized_cnts = cnts / tot_stimduration * 1000
                bin_centers = bins[:-1] + np.diff(bins)[0] / 2  # find the center points of bins to have a line plot
                axs[i, j].plot(bin_centers, time_normalized_cnts, label=_cond)
                axs[i, j].legend()
                if measure == 'bout_angle':
                    axs[i, j].axvline(0, linestyle='dashed', color='black', linewidth=2)
                if j == 0:
                    axs[i, j].set_ylabel(_stim, fontsize='large')
                if i == 0:
                    axs[i, j].set_title(_id, fontsize='large')

    fig.supylabel('Frequency (mHz)', fontsize='x-large', x=0.095)
    fig.supxlabel(measure, fontsize='x-large', y=0.1)
    if level == 'exp':
        fig.suptitle(folder[folder.rfind('\\') + 1:], y=0.9, fontsize='x-large')
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + '_' + measure + '_histogram_perfish'),
                    dpi=300, transparent=False)


def scatterplot_perfish(folder, df, level, stim_name):
    # Scatter plot of x-y coordinates over a given stim_name per condition per fish
    # stim_index for the given stim is randomly selected

    _df = df[df.bout.notna()]
    fig, axs = plt.subplots(nrows=len(_df.condition.unique()), ncols=len(_df.fish_id.unique()),
                            figsize=(10 * len(_df.fish_id.unique()), 10 * len(_df.condition.unique())),
                            constrained_layout=True)
    for j, _id in enumerate(_df.fish_id.unique()):
        for i, _cond in enumerate(_df[_df.fish_id == _id].condition.unique()):
            subdf = _df[(_df.fish_id == _id) & (_df.condition == _cond) & (_df.stim_name == stim_name)]
            if len(subdf.stim_index.unique()) != 0:
                stim = choice(subdf.stim_index.unique())
                sub = subdf[subdf.stim_index == stim]
                im = axs[i, j].scatter(sub.f0_x, sub.f0_y, c=sub.stim_time, cmap='Purples')
                if i == 0 and j == len(_df.fish_id.unique()) - 1:
                    cbar = fig.colorbar(im, ax=axs[:, j], shrink=0.5)
                    cbar.set_label('stim_time', rotation=270, labelpad=20)
            # axs[i, j].set(adjustable='box', aspect='equal')
            axs[i, j].set_xlabel('f0_x')
            axs[i, j].set_ylabel('f0_y')
            if j == 0:
                axs[i, j].set_ylabel(_cond, fontsize='x-large')
            if i == 0:
                axs[i, j].set_title(_id, fontsize='x-large')

    fig.supylabel('Condition', fontsize='xx-large')
    fig.supxlabel('Fish ID', fontsize='xx-large')
    if level == 'exp':
        fig.suptitle(folder[folder.rfind('\\') + 1:] + '_' + stim_name, fontsize='xx-large')
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + '_' + stim_name + '_xycoords_perfish'),
                    dpi=300, transparent=False)


def lineplot_perconc(folder, boutdf, measure):
    # Plot line graph of average measure over time per concentration
    # Measure can be 'distance' or 'dist_from_center'

    measures = ['distance', 'dist_from_center']
    if measure not in measures:
        print('Measure not available. Accepted measures are: ', measures)
        return

    labels = list(boutdf.concentration.unique())
    labels.sort()
    fig, ax = plt.subplots(figsize=(20, 5))

    baseline_duration = np.sum(
        [boutdf[(boutdf.condition == 'baseline') & (boutdf.stim_index == _stim)].stim_duration.values[0]
         for _stim in boutdf[boutdf.condition == 'baseline'].stim_index.unique()])
    drugtreated_duration = np.sum(
        [boutdf[(boutdf.condition == 'drugtreated') & (boutdf.stim_index == _stim)].stim_duration.values[0]
         for _stim in boutdf[boutdf.condition == 'drugtreated'].stim_index.unique()])
    total_duration = baseline_duration + treatment_duration + drugtreated_duration
    timebins = np.arange(0, total_duration + bin_duration + 1, bin_duration)

    # get the last timebin_ind of baseline and first timebin_ind of drugtreated to determine masked regions
    mask_start = np.digitize(baseline_duration, timebins) - 1
    mask_end = np.digitize(baseline_duration + treatment_duration, timebins) - 1

    maxval = 0
    for conc in labels:
        concdf = boutdf[boutdf.concentration == conc]
        if measure == 'distance':
            bin_vals = [[concdf[(concdf.fish_id == fish) & (concdf['timebin_ind'] == i)].distance.sum() for fish in
                         concdf.fish_id.unique()] for i in range(1, len(timebins))]
        elif measure == 'dist_from_center':
            bin_vals = [
                [concdf[(concdf.fish_id == fish) & (concdf['timebin_ind'] == i)].dist_from_center.mean() for fish in
                 concdf.fish_id.unique()] for i in range(1, len(timebins))]
        bin_means = np.nanmean(bin_vals, axis=1)
        bin_stds = [np.nanstd(t, ddof=1) / np.sqrt(len(t)) for t in bin_vals]
        if max(bin_means) > maxval:
            maxval = max(bin_means)

        masked_vals = np.ma.array(bin_means)
        masked_vals[mask_start:mask_end] = np.ma.masked
        plt.plot(timebins[1:], masked_vals, label=f'{conc}, n={len(concdf.fish_id.unique())}')
        plt.fill_between(timebins[1:], masked_vals + bin_stds, masked_vals - bin_stds, alpha=0.3)

    if measure == 'distance':
        ax.set_ylabel('Bout Distance (px)', fontsize='x-large')
        save_name = 'distovertime_perconc.png'
    elif measure == 'dist_from_center':
        ax.set_ylabel('Average Distance from Center (px)', fontsize='x-large')
        save_name = 'distfromcenterovertime_perconc.png'

    ax.set_xlabel('Time (s)', fontsize='x-large')
    plt.axvspan(baseline_duration, baseline_duration + treatment_duration + bin_duration, color='blue', alpha=0.1)
    plt.text(baseline_duration + bin_duration + treatment_duration / 2,
             (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2 + ax.get_ylim()[0], 'DRUG TREATMENT', verticalalignment='center',
             horizontalalignment='center', rotation='vertical', fontsize='xx-large', color='navy')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig(os.path.join(folder, save_name), dpi=300, transparent=False)


def histplot_perconc(folder, df, measure):
    # Plot histogram of average measure per concentration
    # Does not include stat_time

    if measure == 'bout_duration':
        histplot_boutduration(folder, df)
        return

    df = df[df.bout.notna()]
    bins = np.arange(-180, 180 + 1, omr_bin_angle)
    bin_centers = bins[:-1] + np.diff(bins)[0] / 2
    fig, axs = plt.subplots(nrows=len(df.concentration.unique()), ncols=len(df.stim_name.unique()), sharex=True,
                            sharey=True,
                            constrained_layout=True,
                            figsize=(10 * len(df.stim_name.unique()), 5 * len(df.concentration.unique())))
    stims = df.stim_name.unique()

    for i, _conc in enumerate(df.concentration.unique()):
        for _cond in conditions:
            for j, _stim in enumerate(stims):
                bin_vals = []
                for _id in df[df.concentration == _conc].fish_id.unique():
                    subdf = df[(df.fish_id == _id) & (df.condition == _cond) & (df.stim_name == _stim) & (
                            df.bout_start >= df.stat_time)]
                    tot_stimduration = np.sum([subdf[subdf.stim_index == s].stim_duration.values[0] -
                                               subdf[subdf.stim_index == s].stat_time.values[0] for s in
                                               subdf.stim_index.unique()])
                    cnts, _ = np.histogram(subdf.bout_angle, bins)
                    bin_vals.append(cnts / tot_stimduration * 1000)
                bin_means = np.mean(bin_vals, axis=0)
                bin_stds = np.std(bin_vals, axis=0, ddof=1) / np.sqrt(np.shape(bin_vals)[0])
                '''new_vals = np.transpose(bin_vals)
                bin_stds = [stats.t.interval(0.95, np.shape(new_vals)[1]-1, loc=np.mean(new_vals[a]), scale=stats.sem(new_vals[a])) for a in range(np.shape(new_vals)[0])]
                cis = np.transpose(bin_stds)'''
                if len(df.concentration.unique()) == 1:
                    axs[j].plot(bin_centers, bin_means, label=_cond)
                    axs[j].fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, alpha=0.3)
                    axs[j].legend()
                    if measure == 'bout_angle':
                        axs[j].axvline(0, linestyle='dashed', color='black', linewidth=2)
                    if j == 0:
                        axs[j].set_ylabel(f'{_conc}, n={len(df[df.concentration == _conc].fish_id.unique())}',
                                          fontsize='large')
                    if i == 0:
                        axs[j].set_title(_stim, fontsize='large')
                else:
                    axs[i, j].plot(bin_centers, bin_means, label=_cond)
                    axs[i, j].fill_between(bin_centers, bin_means + bin_stds, bin_means - bin_stds, alpha=0.3)
                    axs[i, j].legend()
                    if measure == 'bout_angle':
                        axs[i, j].axvline(0, linestyle='dashed', color='black', linewidth=2)
                    if j == 0:
                        axs[i, j].set_ylabel(f'{_conc}, n={len(df[df.concentration == _conc].fish_id.unique())}',
                                             fontsize='large')
                    if i == 0:
                        axs[i, j].set_title(_stim, fontsize='large')

    fig.supylabel('Concentration', fontsize='x-large')
    fig.supxlabel(measure, fontsize='x-large')
    plt.savefig(os.path.join(folder, measure + '_histogram_perconc'), dpi=300, transparent=False)


def histplot_boutduration(folder, df):
    df = df[df.bout.notna()]
    bins = np.arange(0, 1 + bout_duration_bin, bout_duration_bin)
    bin_centers = bins[:-1] + np.diff(bins)[0] / 2
    fig, axs = plt.subplots(ncols=len(df.concentration.unique()), sharex=True, sharey=True, constrained_layout=True,
                            figsize=(10 * len(df.concentration.unique()), 5))

    for i, _conc in enumerate(df.concentration.unique()):
        for _cond in conditions:
            bin_vals = []
            for _id in df[df.concentration == _conc].fish_id.unique():
                subdf = df[(df.concentration == _conc) & (df.fish_id == _id) & (df.condition == _cond)]
                tot_stimduration = np.sum([subdf[subdf.stim_index == s].stim_duration.values[0] for s in
                                           subdf.stim_index.unique()])
                cnts, _ = np.histogram(subdf['bout_duration'], bins)
                bin_vals.append(cnts / tot_stimduration * 1000)
            bin_means = np.mean(bin_vals, axis=0)
            bin_stds = np.std(bin_vals, axis=0, ddof=1) / np.sqrt(np.shape(bin_vals)[0])
            if len(df.concentration.unique()) == 1:
                axs.plot(bin_centers, bin_means, label=_cond)
                axs.fill_between(bin_centers, bin_means + bin_stds, bin_means - bin_stds, alpha=0.3)
                axs.legend()
                axs.set_ylabel('Frequency (mHz)', fontsize='large')
                axs.set_title(f'{_conc}, n={len(df[df.concentration == _conc].fish_id.unique())}',
                              fontsize='large')
            else:
                axs[i].plot(bin_centers, bin_means, label=_cond)
                axs[i].fill_between(bin_centers, bin_means + bin_stds, bin_means - bin_stds, alpha=0.3)
                axs[i].legend()
                axs[i].set_title(f'{_conc}, n={len(df[df.concentration == _conc].fish_id.unique())}',
                                 fontsize='large')
                if i == 0:
                    axs[i].set_ylabel('Frequency (mHz)', fontsize='large')
    fig.supxlabel('Bout duration (s)', fontsize='x-large')
    plt.savefig(os.path.join(folder, 'bout_duration_histogram_perconc'), dpi=300, transparent=False)


def lineplot_perconc_baselinenorm(folder, boutdf, measure):
    # Plot line graph of average measure over time per concentration, normalized to baseline
    # Measure can be 'distance' or 'dist_from_center'

    measures = ['distance', 'dist_from_center']
    if measure not in measures:
        print('Measure not available. Accepted measures are: ', measures)
        return

    labels = list(boutdf.concentration.unique())
    labels.sort()
    fig, ax = plt.subplots(figsize=(20, 5))

    baseline_duration = np.sum(
        [boutdf[(boutdf.condition == 'baseline') & (boutdf.stim_index == _stim)].stim_duration.values[0]
         for _stim in boutdf[boutdf.condition == 'baseline'].stim_index.unique()])
    drugtreated_duration = np.sum(
        [boutdf[(boutdf.condition == 'drugtreated') & (boutdf.stim_index == _stim)].stim_duration.values[0]
         for _stim in boutdf[boutdf.condition == 'drugtreated'].stim_index.unique()])
    total_duration = baseline_duration + treatment_duration + drugtreated_duration
    timebins = np.arange(0, total_duration + bin_duration + 1, bin_duration)

    # get the last timebin_ind of baseline and first timebin_ind of drugtreated to determine masked regions
    mask_start = np.digitize(baseline_duration, timebins) - 1
    mask_end = np.digitize(baseline_duration + treatment_duration, timebins) - 1

    maxval = 0
    for conc in labels:
        concdf = boutdf[boutdf.concentration == conc]
        # baseline_avg = np.nan_to_num(concdf[concdf.condition == 'baseline'][measure]).sum()/len(timebins)
        if measure == 'distance':
            bin_vals = [[concdf[(concdf.fish_id == fish) & (concdf['timebin_ind'] == i)].distance.sum() for fish in
                         concdf.fish_id.unique()] for i in range(1, len(timebins))]
            baseline_avg = np.nanmean(np.ndarray.flatten(np.array(bin_vals[:mask_start])))
            norm_bin_vals = bin_vals / baseline_avg
        elif measure == 'dist_from_center':
            bin_vals = [
                [concdf[(concdf.fish_id == fish) & (concdf['timebin_ind'] == i)].dist_from_center.mean() for fish
                 in concdf.fish_id.unique()] for i in range(1, len(timebins))]
            baseline_avg = np.nanmean(np.ndarray.flatten(np.array(bin_vals[:mask_start])))
            norm_bin_vals = bin_vals / baseline_avg

        bin_means = np.nanmean(norm_bin_vals, axis=1)
        bin_stds = np.nanstd(norm_bin_vals, ddof=1, axis=1) / np.sqrt(np.shape(norm_bin_vals)[1])
        if max(bin_means) > maxval:
            maxval = max(bin_means)

        masked_vals = np.ma.array(bin_means)
        masked_vals[mask_start:mask_end] = np.ma.masked
        plt.plot(timebins[1:], masked_vals, label=f'{conc}, n={len(concdf.fish_id.unique())}')
        plt.fill_between(timebins[1:], masked_vals + bin_stds, masked_vals - bin_stds, alpha=0.3)

    if measure == 'distance':
        ax.set_ylabel('Baseline Normalized\nBout Distance (px)', fontsize='x-large')
        save_name = 'distovertime_perconc_baselinenorm.png'
    elif measure == 'dist_from_center':
        ax.set_ylabel('Baseline Normalized\nAverage Distance from Center (px)', fontsize='x-large')
        save_name = 'distfromcenterovertime_perconc_baselinenorm.png'

    ax.set_xlabel('Time (s)', fontsize='x-large')
    plt.axvspan(baseline_duration, baseline_duration + treatment_duration + bin_duration, color='blue', alpha=0.1)
    plt.text(baseline_duration + bin_duration + treatment_duration / 2,
             (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2 + ax.get_ylim()[0], 'DRUG TREATMENT', verticalalignment='center',
             horizontalalignment='center', rotation='vertical', fontsize='xx-large', color='navy')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig(os.path.join(folder, save_name), dpi=300, transparent=False)


def normalize_fish(df, measure, bin_stds, mask_start):
    # Normalize data to baseline average

    if measure == 'distance':
        bin_vals = [df[df.timebin_ind == i].distance.sum() for i in range(1, len(timebins))]
    elif measure == 'dist_from_center':
        bin_vals = [df[df.timebin_ind == i].dist_from_center.mean() for i in range(1, len(timebins))]

    baseline_mean = np.nanmean(bin_vals[:mask_start])
    norm_bin_vals = bin_vals / baseline_mean


def plot_distovertime(folder, boutdf, level, DOI_conc=0):
    # Plot total distance over time per fish
    # Level can be 'exp' or 'conc'

    fig, ax = plt.subplots(figsize=(20, 5))
    timebins = np.arange(0, 2 * habit_duration + baseline_loco_duration + drug_loco_duration + 1, 60)
    labels = list(boutdf.fish_id.unique())
    labels.sort()
    maxval = 0
    for fish in labels:
        iddf = boutdf[boutdf['fish_id'] == fish]
        bin_sums = [iddf[iddf['timebin_ind'] == i].distance.sum() for i in range(1, len(timebins))]
        if max(bin_sums) > maxval:
            maxval = max(bin_sums)
        plt.plot(timebins[1:], bin_sums, label=f'Fish {fish}')
    ax.set_ylabel('Bout Distance (px)', fontsize='x-large')
    ax.set_xlabel('Time (s)', fontsize='x-large')
    plt.axvspan(0, habit_duration, color='blue', alpha=0.2)
    plt.axvspan(habit_duration + baseline_loco_duration, 2 * habit_duration + baseline_loco_duration, color='blue',
                alpha=0.2)
    ax.legend(loc='upper right')
    if level == 'exp':
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + '_distovertime_perfish.png'), dpi=300,
                    transparent=False)
    elif level == 'conc':
        plt.savefig(os.path.join(folder, f'{DOI_conc}_distovertime_perfish.png'), dpi=300, transparent=False)
    else:
        print('Cannot save figure, level should be "exp" or "conc"')
    return


def plot_distovertime_normalized(folder, boutdf, level, DOI_conc=0):
    # Plot total distance over time per fish, normalized to average baseline locomotor activity
    # Level can be 'exp' or 'conc'

    fig, ax = plt.subplots(figsize=(20, 5))
    timebins = np.arange(0, 2 * habit_duration + baseline_loco_duration + drug_loco_duration + 1, 60)
    labels = list(boutdf.fish_id.unique())
    labels.sort()
    maxval = 0
    for fish in labels:
        inds = np.digitize(boutdf[boutdf['fish_id'] == fish].cum_bout_start, timebins)
        inds2 = np.digitize(boutdf[(boutdf['fish_id'] == fish) & (boutdf['condition'] == 'baseline') &
                                   (boutdf['stim_name'] == 'locomotion')].cum_bout_start, timebins)
        bin_means = [boutdf[boutdf['fish_id'] == fish][inds == i].distance.sum() for i in range(len(timebins))]
        baseline_vals = [boutdf[(boutdf['fish_id'] == fish) & (boutdf['condition'] == 'baseline') &
                                (boutdf['stim_name'] == 'locomotion')][inds2 == i].distance.sum() for i in
                         range(len(timebins))]
        baseline = np.nan_to_num(np.mean([baseline_vals[i] for i in np.nonzero(baseline_vals)[0]]))
        norm_bin_means = (bin_means - baseline) / baseline
        if max(norm_bin_means) > maxval:
            maxval = max(norm_bin_means)
        plt.plot(timebins, norm_bin_means, label=f'Fish {fish}')
    ax.set_ylabel('Normalized Bout Distance (ΔDist/Dist)', fontsize='x-large')
    ax.set_xlabel('Time (s)', fontsize='x-large')
    plt.axvspan(0, habit_duration, color='blue', alpha=0.2)
    plt.axvspan(habit_duration + baseline_loco_duration, 2 * habit_duration + baseline_loco_duration, color='blue',
                alpha=0.2)
    ax.legend(loc='upper right')
    if level == 'exp':
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + '_normdistovertime_perfish.png'), dpi=300,
                    transparent=False)
    elif level == 'conc':
        plt.savefig(os.path.join(folder, f'{DOI_conc}_normdistovertime_perfish.png'), dpi=300, transparent=False)
    else:
        print('Cannot save figure, level should be "exp" or "conc"')
    return


def plot_all_distovertime(folder, alldf):
    # Plot average distance over time per concentration

    fig, ax = plt.subplots(figsize=(20, 5))
    timebins = np.arange(0, 2 * habit_duration + baseline_loco_duration + drug_loco_duration + 1, 60)
    labels = [int(i) for i in alldf.concentration.unique()]
    labels.sort()

    for conc in labels:
        subdf = alldf[alldf.concentration == str(conc)]
        bin_sums = [[] for _ in range(len(timebins))]
        for fish in subdf.fish_id.unique():
            inds = np.digitize(subdf[subdf['fish_id'] == fish].cum_bout_start, timebins)
            bin_sum = [subdf[subdf['fish_id'] == fish][inds == i].distance.sum() for i in range(len(timebins))]
            for t in range(len(bin_sum)):
                bin_sums[t].append(bin_sum[t])
        bin_means = []
        for t in range(len(bin_sums)):
            bin_means.append(np.mean(bin_sums[t]))
        plt.plot(timebins, bin_means, label=f'{conc} ug/ml, n={len(subdf.fish_id.unique())}')

    ax.set_ylabel('Bout Distance (px)', fontsize='x-large')
    ax.set_xlabel('Time (s)', fontsize='x-large')
    plt.axvspan(0, habit_duration, color='blue', alpha=0.2)
    plt.axvspan(habit_duration + baseline_loco_duration, 2 * habit_duration + baseline_loco_duration, color='blue',
                alpha=0.2)
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(folder, 'distovertime_perconc.png'), dpi=300, transparent=False)


def plot_all_distovertime_normalized(folder, alldf):
    # Plot average distance over time per concentration, normalized to average baseline locomotor activity

    fig, ax = plt.subplots(figsize=(20, 5))
    timebins = np.arange(0, 2 * habit_duration + baseline_loco_duration + drug_loco_duration + 1, 60)
    labels = [int(i) for i in alldf.concentration.unique()]
    labels.sort()
    bin_sums = [[[] for _ in range(len(labels))] for _ in range(len(timebins))]
    for i, conc in enumerate(labels):
        subdf = alldf[alldf.concentration == str(conc)]
        for fish in subdf.fish_id.unique():
            inds = np.digitize(subdf[subdf['fish_id'] == fish].cum_bout_start, timebins)
            inds2 = np.digitize(subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                      (subdf['stim_name'] == 'locomotion')].cum_bout_start, timebins)
            bin_sum = [subdf[subdf['fish_id'] == fish][inds == i].distance.sum() for i in range(len(timebins))]
            baseline_vals = [subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                   (subdf['stim_name'] == 'locomotion')][inds2 == i].distance.sum() for i in
                             range(len(timebins))]
            baseline = np.nan_to_num(np.mean([baseline_vals[i] for i in np.nonzero(baseline_vals)[0]]))
            norm_bin_sums = (bin_sum - baseline) / baseline
            for t in range(len(bin_sum)):
                bin_sums[t][i].append(norm_bin_sums[t])
        bin_means = []
        for t in range(len(bin_sums)):
            bin_means.append(np.mean(bin_sums[t][i]))
        plt.plot(timebins, bin_means, label=f'{conc} ug/ml, n={len(subdf.fish_id.unique())}')

    ax.set_ylabel('Normalized Bout Distance (ΔDist/Dist)', fontsize='x-large')
    ax.set_xlabel('Time (s)', fontsize='x-large')
    plt.axvspan(0, habit_duration, color='blue', alpha=0.2)
    plt.axvspan(habit_duration + baseline_loco_duration, 2 * habit_duration + baseline_loco_duration, color='blue',
                alpha=0.2)
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(folder, 'normdistovertime_perconc.png'), dpi=300, transparent=False)
    return bin_sums


def bin_anova(bin_sums, labels):
    for t in range(len(bin_sums)):
        fvalue, pvalue = stats.f_oneway(bin_sums[t][0], bin_sums[t][1], bin_sums[t][2], bin_sums[t][3], bin_sums[t][4],
                                        bin_sums[t][5])  # CHANGE THIS
        if pvalue < 0.05:
            print(f'Time point: {t}, F value: {fvalue}, p value: {pvalue}')
            concs = []
            norm_dists = []
            for i, conc in enumerate(labels):
                concs.extend([conc] * len(bin_sums[t][i]))
                norm_dists.extend(bin_sums[t][i])
            t_df = {'concentration': concs, 'norm_dist': norm_dists}

            res = stat()
            res.tukey_hsd(df=pd.DataFrame(t_df), res_var='norm_dist', xfac_var='concentration',
                          anova_model='norm_dist ~ C(concentration)')
            print('Tukey\'s HSD test for multiple comparisons')
            print(res.tukey_summary)

            model = ols('norm_dist ~ C(concentration)', data=pd.DataFrame(t_df)).fit()
            w, pval = stats.shapiro(model.resid)
            print(f'Shapiro-Wilk test to check if normal dist: statistic: {w}, p value: {pval}')
            if pval < 0.05:
                print('Data is not from normal distribution')
            else:
                print('Data is from normal distribution')

            """sm.qqplot(res.anova_std_residuals, line='45')
            plt.xlabel("Theoretical Quantiles")
            plt.ylabel("Standardized Residuals")
            plt.show()

            plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
            plt.xlabel("Residuals")
            plt.ylabel('Frequency')
            plt.show()"""

            if pval >= 0.05:
                w, pval = stats.bartlett(bin_sums[t][0], bin_sums[t][1], bin_sums[t][2], bin_sums[t][3], bin_sums[t][4],
                                         bin_sums[t][5])  # CHANGE THIS
                print(
                    f'Since data is drawn from normal distribution, Bartlett\'s test to check if equal variances: statistic: {w}, '
                    f'p value: {pval}\nNon-significant p value means equal variances')
            else:
                res.levene(df=pd.DataFrame(t_df), res_var='norm_dist', xfac_var='concentration')
                print(
                    'Since data is not drawn from normal distribution, Levene\'s test to check if equal variances.\nNon-significant p value means equal variances')
                print(res.levene_summary)

    return


def plot_avgdistovertime(folder, boutdf, level, DOI_conc=0):
    # Plot average total distance over time
    # Level can be 'exp' or 'conc'

    fig, ax = plt.subplots(figsize=(20, 5))
    timebins = np.arange(0, 2 * habit_duration + baseline_loco_duration + drug_loco_duration + 1, 60)
    labels = list(boutdf.fish_id.unique())
    labels.sort()
    bin_sums = [[] for _ in range(len(timebins))]

    for fish in labels:
        inds = np.digitize(boutdf[boutdf['fish_id'] == fish].cum_bout_start, timebins)
        bin_sum = [boutdf[boutdf['fish_id'] == fish][inds == i].distance.sum() for i in range(len(timebins))]
        for t in range(len(bin_sum)):
            bin_sums[t].append(bin_sum[t])

    bin_means = []
    for t in range(len(bin_sums)):
        bin_means.append(np.mean(bin_sums[t]))

    plt.plot(timebins, bin_means)
    ax.set_ylabel('Average Bout Distance (px)', fontsize='x-large')
    ax.set_xlabel('Time (s)', fontsize='x-large')
    plt.axvspan(0, habit_duration, color='blue', alpha=0.2)
    plt.axvspan(habit_duration + baseline_loco_duration, 2 * habit_duration + baseline_loco_duration, color='blue',
                alpha=0.2)
    if level == 'exp':
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + '_avgdistovertime.png'), dpi=300,
                    transparent=False)
    elif level == 'conc':
        plt.savefig(os.path.join(folder, f'{DOI_conc}_avgdistovertime.png'), dpi=300, transparent=False)
    else:
        print('Cannot save figure, level should be "exp" or "conc"')
    return


def plot_avgdistovertime_normalized(folder, boutdf, level, DOI_conc=0):
    # Plot average total distance over time, normalized to average baseline locomotor activity
    # Level can be 'exp' or 'conc'

    fig, ax = plt.subplots(figsize=(20, 5))
    timebins = np.arange(0, 2 * habit_duration + baseline_loco_duration + drug_loco_duration + 1, 60)
    labels = list(boutdf.fish_id.unique())
    labels.sort()
    bin_sums = [[] for _ in range(len(timebins))]

    for fish in labels:
        inds = np.digitize(boutdf[boutdf['fish_id'] == fish].cum_bout_start, timebins)
        inds2 = np.digitize(boutdf[(boutdf['fish_id'] == fish) & (boutdf['condition'] == 'baseline') &
                                   (boutdf['stim_name'] == 'locomotion')].cum_bout_start, timebins)
        bin_sum = [boutdf[boutdf['fish_id'] == fish][inds == i].distance.sum() for i in range(len(timebins))]
        baseline_vals = [boutdf[(boutdf['fish_id'] == fish) & (boutdf['condition'] == 'baseline') &
                                (boutdf['stim_name'] == 'locomotion')][inds2 == i].distance.sum() for i in
                         range(len(timebins))]
        baseline = np.nan_to_num(np.mean([baseline_vals[i] for i in np.nonzero(baseline_vals)[0]]))
        norm_bin_sums = (bin_sum - baseline) / baseline
        for t in range(len(norm_bin_sums)):
            bin_sums[t].append(norm_bin_sums[t])

    bin_means = []
    for t in range(len(bin_sums)):
        bin_means.append(np.mean(bin_sums[t]))

    plt.plot(timebins, bin_means)
    ax.set_ylabel('Normalized Average Bout Distance (ΔDist/Dist)', fontsize='x-large')
    ax.set_xlabel('Time (s)', fontsize='x-large')
    plt.axvspan(0, habit_duration, color='blue', alpha=0.2)
    plt.axvspan(habit_duration + baseline_loco_duration, 2 * habit_duration + baseline_loco_duration, color='blue',
                alpha=0.2)
    if level == 'exp':
        plt.savefig(os.path.join(folder, folder[folder.rfind('\\') + 1:] + '_normavgdistovertime.png'), dpi=300,
                    transparent=False)
    elif level == 'conc':
        plt.savefig(os.path.join(folder, f'{DOI_conc}_normavgdistovertime.png'), dpi=300, transparent=False)
    else:
        print('Cannot save figure, level should be "exp" or "conc"')
    return


def plt_avgperconc(folder, alldf, measure):
    # Plot bar graphs of average measures per condition
    # Measure can be 'dist', 'boutcount', 'thigmotaxis_dist', or 'thigmotaxis_time'

    timebins = np.arange(0, 2 * habit_duration + baseline_loco_duration + drug_loco_duration + 1, 60)
    concs = [int(i) for i in alldf.concentration.unique()]
    concs.sort()
    labels = ['Baseline; Habituation', 'Baseline; Locomotion', 'DOI treated; Habituation', 'DOI treated; Locomotion']

    x = np.arange(len(labels) * 2, step=2)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for i, conc in enumerate(concs):
        subdf = alldf[alldf.concentration == str(conc)]
        baseline_habits = []
        baseline_locos = []
        drug_habits = []
        drug_locos = []
        for fish in subdf.fish_id.unique():
            if 'thigmotaxis' not in measure:
                inds1 = np.digitize(subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                          (subdf['stim_name'] == 'habituation')].cum_bout_start, timebins)
                inds2 = np.digitize(subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                          (subdf['stim_name'] == 'locomotion')].cum_bout_start, timebins)
                inds3 = np.digitize(subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                          (subdf['stim_name'] == 'habituation')].cum_bout_start, timebins)
                inds4 = np.digitize(subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                          (subdf['stim_name'] == 'locomotion')].cum_bout_start, timebins)
                if measure == 'dist':
                    baseline_habit = [subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                            (subdf['stim_name'] == 'habituation')][inds1 == i].distance.sum() for i in
                                      range(len(timebins))]
                    baseline_loco = [subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                           (subdf['stim_name'] == 'locomotion')][inds2 == i].distance.sum() for i in
                                     range(len(timebins))]
                    drug_habit = [subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                        (subdf['stim_name'] == 'habituation')][inds3 == i].distance.sum() for i in
                                  range(len(timebins))]
                    drug_loco = [subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                       (subdf['stim_name'] == 'locomotion')][inds4 == i].distance.sum() for i in
                                 range(len(timebins))]
                elif measure == 'boutcount':
                    baseline_habit = [subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                            (subdf['stim_name'] == 'habituation')][inds1 == i].bout.count() for i in
                                      range(len(timebins))]
                    baseline_loco = [subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                           (subdf['stim_name'] == 'locomotion')][inds2 == i].bout.count() for i in
                                     range(len(timebins))]
                    drug_habit = [subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                        (subdf['stim_name'] == 'habituation')][inds3 == i].bout.count() for i in
                                  range(len(timebins))]
                    drug_loco = [subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                       (subdf['stim_name'] == 'locomotion')][inds4 == i].bout.count() for i in
                                 range(len(timebins))]
                else:
                    print(
                        'Cannot calculate; measure should be "dist", "boutcount", "thigmotaxis_dist", or "thigmotaxis_time"')
                    return
                baseline_habits.append(
                    np.nan_to_num(np.mean([baseline_habit[i] for i in np.nonzero(baseline_habit)[0]])))
                baseline_locos.append(np.nan_to_num(np.mean([baseline_loco[i] for i in np.nonzero(baseline_loco)[0]])))
                drug_habits.append(np.nan_to_num(np.mean([drug_habit[i] for i in np.nonzero(drug_habit)[0]])))
                drug_locos.append(np.nan_to_num(np.mean([drug_loco[i] for i in np.nonzero(drug_loco)[0]])))

            elif 'thigmotaxis' in measure:
                if measure == 'thigmotaxis_dist':
                    baseline_habit = 100 * (subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                                  (subdf['stim_name'] == 'habituation') & (
                                                          subdf['in_center'] == False)].distance.sum()) / (subdf[(
                                                                                                                         subdf[
                                                                                                                             'fish_id'] == fish) & (
                                                                                                                         subdf[
                                                                                                                             'condition'] == 'baseline') & (
                                                                                                                         subdf[
                                                                                                                             'stim_name'] == 'habituation')].distance.sum())

                    baseline_loco = 100 * (subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                                 (subdf['stim_name'] == 'locomotion') & (
                                                         subdf['in_center'] == False)].distance.sum()) / (subdf[(
                                                                                                                        subdf[
                                                                                                                            'fish_id'] == fish) & (
                                                                                                                        subdf[
                                                                                                                            'condition'] == 'baseline') & (
                                                                                                                        subdf[
                                                                                                                            'stim_name'] == 'locomotion')].distance.sum())

                    drug_habit = 100 * (subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                              (subdf['stim_name'] == 'habituation') & (
                                                      subdf['in_center'] == False)].distance.sum()) / (subdf[(subdf[
                                                                                                                  'fish_id'] == fish) & (
                                                                                                                     subdf[
                                                                                                                         'condition'] == 'drugtreated') & (
                                                                                                                     subdf[
                                                                                                                         'stim_name'] == 'habituation')].distance.sum())

                    drug_loco = 100 * (subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                             (subdf['stim_name'] == 'locomotion') & (
                                                     subdf['in_center'] == False)].distance.sum()) / (subdf[(subdf[
                                                                                                                 'fish_id'] == fish) & (
                                                                                                                    subdf[
                                                                                                                        'condition'] == 'drugtreated') & (
                                                                                                                    subdf[
                                                                                                                        'stim_name'] == 'locomotion')].distance.sum())

                elif measure == 'thigmotaxis_time':
                    baseline_habit = 100 * (subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                                  (subdf['stim_name'] == 'habituation') & (
                                                          subdf['in_center'] == False)].bout_duration.sum()) / (
                                         subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') & (
                                                 subdf['stim_name'] == 'habituation')].bout_duration.sum())

                    baseline_loco = 100 * (subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') &
                                                 (subdf['stim_name'] == 'locomotion') & (
                                                         subdf['in_center'] == False)].bout_duration.sum()) / (
                                        subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'baseline') & (
                                                subdf['stim_name'] == 'locomotion')].bout_duration.sum())

                    drug_habit = 100 * (subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                              (subdf['stim_name'] == 'habituation') & (
                                                      subdf['in_center'] == False)].bout_duration.sum()) / (subdf[(
                                                                                                                          subdf[
                                                                                                                              'fish_id'] == fish) & (
                                                                                                                          subdf[
                                                                                                                              'condition'] == 'drugtreated') & (
                                                                                                                          subdf[
                                                                                                                              'stim_name'] == 'habituation')].bout_duration.sum())

                    drug_loco = 100 * (subdf[(subdf['fish_id'] == fish) & (subdf['condition'] == 'drugtreated') &
                                             (subdf['stim_name'] == 'locomotion') & (
                                                     subdf['in_center'] == False)].bout_duration.sum()) / (subdf[(
                                                                                                                         subdf[
                                                                                                                             'fish_id'] == fish) & (
                                                                                                                         subdf[
                                                                                                                             'condition'] == 'drugtreated') & (
                                                                                                                         subdf[
                                                                                                                             'stim_name'] == 'locomotion')].bout_duration.sum())
                else:
                    print(
                        'Cannot calculate; measure should be "dist", "boutcount", "thigmotaxis_dist", or "thigmotaxis_time"')
                    return
                baseline_habits.append(np.nan_to_num(baseline_habit))
                baseline_locos.append(np.nan_to_num(baseline_loco))
                drug_habits.append(np.nan_to_num(drug_habit))
                drug_locos.append(np.nan_to_num(drug_loco))

            else:
                print(
                    'Cannot calculate; measure should be "dist", "boutcount", "thigmotaxis_dist", or "thigmotaxis_time"')
                return

        conc_vals = [np.mean(baseline_habits), np.mean(baseline_locos), np.mean(drug_habits), np.mean(drug_locos)]
        rects = ax.bar(x - (width * ((len(concs) - 1) - (2 * i))) / 2, conc_vals, width,
                       label=f'{conc} ug/ml, n={len(subdf.fish_id.unique())}')

    ax.legend()

    if measure == 'dist':
        ax.set_ylabel('Average Distance per min (px)', fontsize='x-large')
        plt.savefig(os.path.join(folder, 'avgboutdist_conc.png'), dpi=300, transparent=False)
        return
    elif measure == 'boutcount':
        ax.set_ylabel('Average Bout Count per min', fontsize='x-large')
        plt.savefig(os.path.join(folder, 'avgboutcount_conc.png'), dpi=300, transparent=False)
        return
    elif measure == 'thigmotaxis_dist':
        ax.set_ylabel('Average %Thigmotaxis (distance moved)', fontsize='x-large')
        plt.savefig(os.path.join(folder, 'avgthigmotaxisdist_conc.png'), dpi=300, transparent=False)
        return
    elif measure == 'thigmotaxis_time':
        ax.set_ylabel('Average %Thigmotaxis (time spent)', fontsize='x-large')
        plt.savefig(os.path.join(folder, 'avgthigmotaxistime_conc.png'), dpi=300, transparent=False)
        return
    else:
        print('Cannot save figure; measure should be "dist", "boutcount", "thigmotaxis_dist", or "thigmotaxis_time"')
        return

# def plt_avgperconc_normalized(folder, alldf, measure):
