import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np


def save_figure(mcam, fig, fig_name):
    '''Saves given figure'''
    try:
        fig_path = os.path.join(mcam.data_paths['figures'], fig_name)
    except KeyError:
        mcam.data_paths['figures'] = Path(os.path.join(mcam.folder_path, 'figures'))
        os.mkdir(mcam.data_paths['figures'])
        fig_path = os.path.join(mcam.data_paths['figures'], fig_name)

    fig.savefig(fig_path)


def plot_distance_per_condition(mcam, conditions=list(), concentrations=list(), savefig=True, rolling_window=0):
    '''Plots distance over time of given conditions and concentrations'''

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    fig, axs = plt.subplots(len(conditions), figsize=(25, (10*len(conditions))), sharey=True, sharex=True)

    for i, condition in enumerate(conditions):
        dfs = mcam.dataframes[condition]['distance']

        for conc in concentrations:
            df = dfs[conc]

            x = df.iloc[:, 0]

            if rolling_window != 0:
                y = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).mean()
                err = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).sem()
            else:
                y = df['average_dist']
                err = df['sem']

            n_fish = len(df['distance_traveled'].columns)

            axs[i].plot(x, y, label=f'{str(conc)}, n={n_fish}')
            axs[i].fill_between(x, y-err, y+err, alpha=0.5)

        axs[i].set_title(condition, fontsize=18)
        axs[i].legend()

    fig.supxlabel('Time (s)', fontsize=18)
    fig.supylabel('Distance traveled (m)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, f'distance_per_condition_rolling{rolling_window}.pdf')


def plot_distance_per_concentration(mcam, conditions=list(), concentrations=list(), savefig=True, rolling_window=0):
    '''Plots distance over time of given conditions and concentrations'''

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    fig, axs = plt.subplots(len(concentrations), figsize=(25, (10*len(concentrations))), sharey=True, sharex=True)

    for i, conc in enumerate(concentrations):
        for condition in conditions:
            df = mcam.dataframes[condition]['distance'][conc]

            x = df.iloc[:, 0]

            if rolling_window != 0:
                y = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).mean()
                err = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).sem()
            else:
                y = df['average_dist']
                err = df['sem']

            n_fish = len(df['distance_traveled'].columns)

            axs[i].plot(x, y, label=f'{condition}, n={n_fish}')
            axs[i].fill_between(x, y-err, y+err, alpha=0.5)

        axs[i].set_title(str(conc), fontsize=18)
        axs[i].legend()

    fig.supxlabel('Time (s)', fontsize=18)
    fig.supylabel('Distance traveled (m)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, f'distance_per_concentration_rolling{rolling_window}.pdf')


def plot_survival(mcam, savefig=True):
    '''Plots the survival of fish in 24 hour recovery'''

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))

    x = mcam.concentrations
    final_ps = list()

    for conc in x:
        start_n = len(mcam.dataframes['baseline']['distance'][conc]['distance_traveled'].columns)
        final_n = len(mcam.dataframes['24hour_recovery']['distance'][conc]['distance_traveled'].columns)

        start_p = start_n/start_n * 100
        final_p = final_n/start_n * 100

        if final_p not in final_ps:
            axs[1].plot(('baseline', '24 hour recovery'), (start_p, final_p), label=str(conc), alpha=0.7)
        else:
            count = final_ps.count(final_p)
            dx, dy = (count * 2/72.), (count * -2/72.)
            offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            transform = axs[1].transData + offset
            axs[1].plot(('baseline', '24 hour recovery'), (start_p, final_p), label=str(conc), alpha=0.7, transform=transform)

        final_ps.append(final_p)

    axs[1].legend()
    axs[1].set_ylim(bottom=0, top=110)

    axs[0].plot(x, final_ps)
    axs[0].set_xscale('log')
    axs[0].set_xlabel('DOI concentration (µg/ml)', fontsize=18)
    axs[0].set_ylabel('Survival (%)', fontsize=18)
    axs[0].set_ylim(bottom=0, top=110)

    if savefig:
        save_figure(mcam, fig, f'percent_survival.pdf')


def plot_stimulus_distance_per_condition(mcam, stim_name, stim_num, gap=0, conditions=list(), concentrations=list(), savefig=True):
    '''Plots distance traveled over time for a given stimulus name and number'''

    if not isinstance(stim_num, list):
        raise TypeError('stim_num must be a list')

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    fig, axs = plt.subplots(len(conditions), figsize=(25, (10*len(conditions))), sharey=True, sharex=True)

    for i, condition in enumerate(conditions):
        dfs = mcam.dataframes[condition]['distance']

        for conc in concentrations:
            df = dfs[conc]

            stim_start = df[(df.stim_name == stim_name) & (df.stim_num == stim_num[0])].index.values[0]
            stim_stop = df[(df.stim_name == stim_name) & (df.stim_num == stim_num[-1])].index.values[-1]

            stim_subdf = df.loc[stim_start-gap:stim_stop+gap, :]

            x = stim_subdf.iloc[:, 0] - stim_subdf.iloc[0, 0]
            y = stim_subdf['average_dist'].copy()
            err = stim_subdf['sem'].copy()

            stimulus = y.copy()
            stimulus[stim_subdf.stim_name != stim_name] = np.nan

            stimulus_err = err.copy()
            stimulus_err[stim_subdf.stim_name != stim_name] = np.nan

            n_fish = len(df['distance_traveled'].columns)

            axs[i].plot(x, y, color='lightgray')
            axs[i].fill_between(x, y-err, y+err, color='lightgray', alpha=0.5)

            axs[i].plot(x, stimulus, label=f'{str(conc)}, n={n_fish}')
            axs[i].fill_between(x, stimulus-stimulus_err, stimulus+stimulus_err, alpha=0.5)

            axs[i].set_xticks(np.arange(min(x), max(x)+1, 2.0))

        axs[i].legend()
        axs[i].set_title(condition, fontsize=18)

    fig.supxlabel('Time (s)', fontsize=18)
    fig.supylabel('Distance traveled (m)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, f'{stim_name}_{stim_num[0]}_{stim_num[-1]}_distance_per_condition.pdf')


def plot_stimulus_distance_per_concentration(mcam, stim_name, stim_num, gap=0, conditions=list(), concentrations=list(), savefig=True):
    '''Plots distance traveled over time for a given stimulus name and number'''

    if not isinstance(stim_num, list):
        raise TypeError('stim_num must be a list')

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    fig, axs = plt.subplots(len(concentrations), figsize=(25, (10*len(concentrations))), sharey=True, sharex=True)

    for i, conc in enumerate(concentrations):
        for condition in conditions:
            df = mcam.dataframes[condition]['distance'][conc]

            stim_start = df[(df.stim_name == stim_name) & (df.stim_num == stim_num[0])].index.values[0]
            stim_stop = df[(df.stim_name == stim_name) & (df.stim_num == stim_num[-1])].index.values[-1]

            stim_subdf = df.loc[stim_start-gap:stim_stop+gap, :]

            x = stim_subdf.iloc[:, 0] - stim_subdf.iloc[0, 0]
            y = stim_subdf['average_dist'].copy()
            err = stim_subdf['sem'].copy()

            stimulus = y.copy()
            stimulus[stim_subdf.stim_name != stim_name] = np.nan

            stimulus_err = err.copy()
            stimulus_err[stim_subdf.stim_name != stim_name] = np.nan

            n_fish = len(df['distance_traveled'].columns)

            axs[i].plot(x, y, color='lightgray')
            axs[i].fill_between(x, y-err, y+err, color='lightgray', alpha=0.5)

            axs[i].plot(x, stimulus, label=f'{condition}, n={n_fish}')
            axs[i].fill_between(x, stimulus-stimulus_err, stimulus+stimulus_err, alpha=0.5)

            axs[i].set_xticks(np.arange(min(x), max(x)+1, 2.0))

        axs[i].legend()
        axs[i].set_title(str(conc), fontsize=18)

    fig.supxlabel('Time (s)', fontsize=18)
    fig.supylabel('Distance traveled (m)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, f'{stim_name}_{stim_num[0]}_{stim_num[-1]}_distance_per_concentration.pdf')


def plot_thigmotaxis_per_condition(mcam, conditions=list(), concentrations=list(), savefig=True, rolling_window=0):
    '''Plots distance from center over time of given conditions and concentrations'''

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    fig, axs = plt.subplots(len(conditions), figsize=(25, (10*len(conditions))), sharey=True, sharex=True)

    for i, condition in enumerate(conditions):
        dfs = mcam.dataframes[condition]['tracking']

        for conc in concentrations:
            df = dfs[conc]

            x = df.iloc[:, 0]

            if rolling_window != 0:
                y = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).mean()
                err = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).sem()
            else:
                y = df['average_dist']
                err = df['sem']

            n_fish = len(df['distance_traveled'].columns)

            axs[i].plot(x, y, label=f'{str(conc)}, n={n_fish}')
            axs[i].fill_between(x, y-err, y+err, alpha=0.5)

        axs[i].set_title(condition, fontsize=18)
        axs[i].legend()

    fig.supxlabel('Time (s)', fontsize=18)
    fig.supylabel('Distance traveled (m)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, f'distance_per_condition_rolling{rolling_window}.pdf')
