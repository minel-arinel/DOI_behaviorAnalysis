import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from scipy.stats import sem
from utils import add_bins, get_stimulus_starts_stops


def save_figure(mcam, fig, fig_name):
    '''Saves given figure'''
    try:
        fig_path = os.path.join(mcam.data_paths['figures'], fig_name)
    except KeyError:
        mcam.data_paths['figures'] = Path(os.path.join(mcam.folder_path, 'figures'))
        os.mkdir(mcam.data_paths['figures'])
        fig_path = os.path.join(mcam.data_paths['figures'], fig_name)

    fig.savefig(fig_path)


def color_palette():
    colors = {
        'copper': '#c84e00',
        'persimmon': '#e89923',
        'dandelion': '#ffd960',
        'piedmont': '#a1b70d',
        'eno': '#339898',
        'magnolia': '#1d6363',
        'prussian_blue': '#005587',
        'shale_blue': '#0577B1',
        'ironweed': '993399',
    }
    return colors


def plot_stimulus_range(mcam, ax, stimuli):
    '''Plots the timing for the given stimuli on the axis'''
    if not isinstance(stimuli, dict):
        raise TypeError('stimuli must be a dictionary with the key: value pair \'stim_name\': [stim_nums]')

    df = mcam.stim_df

    for stim in stimuli:
        for num in stimuli[stim]:
            subdf = df[(df.stim_name == stim) & (df.stim_num == num)]
            start = subdf.iloc[0, 0]
            end = subdf.iloc[-1, 0]

            ax.axvspan(start, end, alpha=0.2)


def plot_baseline_activity(mcam, savefig=True):
    '''Plots a histogram of total baseline distance per fish'''

    baseline_dists = list()

    for conc in mcam.dataframes['baseline']['distance']:
        df = mcam.dataframes['baseline']['distance'][conc]

        cols = [col for col in df if col.startswith('distance_traveled')]
        baseline_dists.append(df[cols].sum())

    baseline_dists = np.concatenate(baseline_dists)

    fig = plt.figure(figsize=(10, 5))
    plt.hist(baseline_dists, bins=40)
    plt.xlabel('Total distance traveled (m)')
    plt.ylabel('Count')

    if savefig:
        save_figure(mcam, fig, f'baseline_activity.pdf')


def plot_distance_per_condition(mcam, conditions=list(), concentrations=list(), savefig=True, rolling_window=0, time_bin=0, force_add_bins=False):
    '''Plots distance over time of given conditions and concentrations'''

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    if force_add_bins:
        add_bins(mcam, time_bin=time_bin)

    fig, axs = plt.subplots(len(conditions), figsize=(25, (10*len(conditions))), sharey=True, sharex=True)

    for i, condition in enumerate(conditions):
        dfs = mcam.dataframes[condition]['distance']

        for conc in concentrations:

            df = dfs[conc]

            cols = [col for col in df if col.startswith('distance_traveled')]
            n_fish = len(cols)

            if rolling_window != 0:
                x = df.iloc[:, 0]
                y = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).mean()
                err = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).sem()

            elif time_bin != 0:
                if 'binned_time' not in df.columns:
                    add_bins(mcam, time_bin=time_bin)

                grouped_df = df.groupby(['binned_time']).sum()
                x = grouped_df.index.values
                y = grouped_df[cols].mean(axis=1)
                err = grouped_df[cols].sem(axis=1)

            else:
                x = df.iloc[:, 0]
                y = df['average_dist']
                err = df['sem']

            axs[i].plot(x, y, label=f'{str(conc)}, n={n_fish}')
            axs[i].fill_between(x, y-err, y+err, alpha=0.5)

        axs[i].set_title(condition, fontsize=18)
        axs[i].legend()

        plot_stimulus_range(mcam, axs[i], stimuli={'dark_epoch': [0, 6, 12]})

    fig.supxlabel('Time (s)', fontsize=18)
    fig.supylabel('Distance traveled (m)', fontsize=18)

    if savefig:
        if rolling_window != 0:
            save_figure(mcam, fig, f'distance_per_condition_rolling{rolling_window}.pdf')
        elif time_bin != 0:
            save_figure(mcam, fig, f'distance_per_condition_timebin{time_bin}s.pdf')


def plot_distance_per_concentration(mcam, conditions=list(), concentrations=list(), savefig=True, rolling_window=0, time_bin=0, force_add_bins=False):
    '''Plots distance over time of given conditions and concentrations'''

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    if force_add_bins:
        add_bins(mcam, time_bin=time_bin)

    fig, axs = plt.subplots(len(concentrations), figsize=(25, (10*len(concentrations))), sharey=True, sharex=True)

    for i, conc in enumerate(concentrations):
        for condition in conditions:
            df = mcam.dataframes[condition]['distance'][conc]

            cols = [col for col in df if col.startswith('distance_traveled')]
            n_fish = len(cols)

            if rolling_window != 0:
                x = df.iloc[:, 0]
                y = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).mean()
                err = df['average_dist'].rolling(rolling_window, center=True, min_periods=0).sem()

            elif time_bin != 0:
                if 'binned_time' not in df.columns:
                    add_bins(mcam, time_bin=time_bin)

                grouped_df = df.groupby(['binned_time']).sum()
                x = grouped_df.index.values
                y = grouped_df[cols].mean(axis=1)
                err = grouped_df[cols].sem(axis=1)

            else:
                x = df.iloc[:, 0]
                y = df['average_dist']
                err = df['sem']

            axs[i].plot(x, y, label=f'{condition}, n={n_fish}')
            axs[i].fill_between(x, y-err, y+err, alpha=0.5)

        axs[i].set_title(str(conc), fontsize=18)
        axs[i].legend()

        plot_stimulus_range(mcam, axs[i], stimuli={'dark_epoch': [0, 6, 12]})

    fig.supxlabel('Time (s)', fontsize=18)
    fig.supylabel('Distance traveled (m)', fontsize=18)

    if savefig:
        if rolling_window != 0:
            save_figure(mcam, fig, f'distance_per_concentration_rolling{rolling_window}.pdf')
        elif time_bin != 0:
            save_figure(mcam, fig, f'distance_per_concentration_timebin{time_bin}s.pdf')


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
                y = df['average_dist_from_center'].rolling(rolling_window, center=True, min_periods=0).mean()
                err = df['average_dist_from_center'].rolling(rolling_window, center=True, min_periods=0).sem()
            else:
                y = df['average_dist_from_center']
                err = df['sem']

            cols = [col for col in df if col.startswith('center_y')]
            n_fish = len(cols)

            axs[i].plot(x, y, label=f'{str(conc)}, n={n_fish}')
            axs[i].fill_between(x, y-err, y+err, alpha=0.5)

        axs[i].set_title(condition, fontsize=18)
        axs[i].legend()

        plot_stimulus_range(mcam, axs[i], stimuli={'dark_epoch': [0, 6, 12]})

    fig.supxlabel('Time (s)', fontsize=18)
    fig.supylabel('Distance from center (m)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, f'distance_from_center_per_condition_rolling{rolling_window}.pdf')


def plot_thigmotaxis_per_concentration(mcam, conditions=list(), concentrations=list(), savefig=True, rolling_window=0):
    '''Plots distance from center over time of given conditions and concentrations'''

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    fig, axs = plt.subplots(len(concentrations), figsize=(25, (10*len(concentrations))), sharey=True, sharex=True)

    for i, conc in enumerate(concentrations):
        for condition in conditions:
            df = mcam.dataframes[condition]['tracking'][conc]

            x = df.iloc[:, 0]

            if rolling_window != 0:
                y = df['average_dist_from_center'].rolling(rolling_window, center=True, min_periods=0).mean()
                err = df['average_dist_from_center'].rolling(rolling_window, center=True, min_periods=0).sem()
            else:
                y = df['average_dist_from_center']
                err = df['sem']

            cols = [col for col in df if col.startswith('center_y')]
            n_fish = len(cols)

            axs[i].plot(x, y, label=f'{condition}, n={n_fish}')
            axs[i].fill_between(x, y-err, y+err, alpha=0.5)

        axs[i].set_title(str(conc), fontsize=18)
        axs[i].legend()

        plot_stimulus_range(mcam, axs[i], stimuli={'dark_epoch': [0, 6, 12]})

    fig.supxlabel('Time (s)', fontsize=18)
    fig.supylabel('Distance from center (m)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, f'distance_from_center_per_concentration_rolling{rolling_window}.pdf')


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Taken from matplotlib.org
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    Taken from matplotlib.org
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_percent_thigmotaxis_distance(mcam, well_diameter=15, conditions=list(), concentrations=list(), savefig=True):
    '''Plots percentage of distance moved in the outer zone of the well over the total distance
    well_diameter: the diameter of the bottom of each well in mm'''

    well_area = np.pi * (well_diameter/2)**2
    inner_area = well_area / 2
    inner_radius = np.sqrt(inner_area/np.pi)

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    thigmo_heatmap = []

    for conc in concentrations:
        thigmo_heatmap.append(list())

        for condition in conditions:

            distance_df = mcam.dataframes[condition]['distance'][conc]
            dist_cols = [col for col in distance_df if col.startswith('distance_traveled')]
            distance_df = distance_df[dist_cols]

            tracking_df = mcam.dataframes[condition]['tracking'][conc]
            center_cols = [col for col in tracking_df if col.startswith('dist_from_center')]
            tracking_df = tracking_df[center_cols]

            percentages = []

            for fish in range(len(center_cols)):
                tracking_col = tracking_df.iloc[:, fish]
                distance_col = distance_df.iloc[:, fish]
                total_outer_distance = distance_col[np.where(tracking_col*1000 > inner_radius)[0]].sum()
                total_distance = distance_col.sum()
                percent_thigmotaxis = (total_outer_distance / total_distance) * 100
                percentages.append(percent_thigmotaxis)

            thigmo_heatmap[-1].append(np.array(percentages).mean())

    thigmo_heatmap = np.array(thigmo_heatmap)
    change_from_baseline = np.array([row-row[0] for row in thigmo_heatmap])

    im, cbar = heatmap(thigmo_heatmap, concentrations, conditions, ax=axs[0], cmap='Blues', cbarlabel='%thigmotaxis (distance moved)')
    annotate_heatmap(im, valfmt="{x:.1f}%")

    im2, cbar = heatmap(change_from_baseline, concentrations, conditions, ax=axs[1], cmap='Blues', cbarlabel='change from baseline %thigmotaxis')
    annotate_heatmap(im2, valfmt="{x:.1f}%")

    fig.tight_layout()

    if savefig:
        save_figure(mcam, fig, f'percent_thigmotaxis_distance.pdf')


def plot_percent_thigmotaxis_time(mcam, well_diameter=15, conditions=list(), concentrations=list(), savefig=True):
    '''Plots percentage of distance moved in the outer zone of the well over the total distance
    well_diameter: the diameter of the bottom of each well in mm'''

    well_area = np.pi * (well_diameter/2)**2
    inner_area = well_area / 2
    inner_radius = np.sqrt(inner_area/np.pi)

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    thigmo_heatmap = []

    for conc in concentrations:
        thigmo_heatmap.append(list())

        for condition in conditions:
            tracking_df = mcam.dataframes[condition]['tracking'][conc]
            delta_t = np.diff(tracking_df.iloc[:, 0]).mean()

            cols = [col for col in tracking_df if col.startswith('dist_from_center')]

            percentages = []

            for fish in range(len(cols)):
                tracking_col = tracking_df[cols].iloc[:, fish]
                total_outer_time = len(np.where(tracking_col*1000 > inner_radius)[0]) * delta_t

                percent_thigmotaxis = (total_outer_time / tracking_df.iloc[-1, 0]) * 100
                percentages.append(percent_thigmotaxis)

            thigmo_heatmap[-1].append(np.array(percentages).mean())

    thigmo_heatmap = np.array(thigmo_heatmap)
    change_from_baseline = np.array([row-row[0] for row in thigmo_heatmap])

    im, cbar = heatmap(thigmo_heatmap, concentrations, conditions, ax=axs[0], cmap='Blues', cbarlabel='%thigmotaxis (time spent)')
    annotate_heatmap(im, valfmt="{x:.1f}%")

    im2, cbar = heatmap(change_from_baseline, concentrations, conditions, ax=axs[1], cmap='Blues', cbarlabel='change from baseline %thigmotaxis')
    annotate_heatmap(im2, valfmt="{x:.1f}%")

    fig.tight_layout()

    if savefig:
        save_figure(mcam, fig, f'percent_thigmotaxis_time.pdf')


def plot_percent_thigmotaxis_distance_line(mcam, well_diameter=15, conditions=list(), concentrations=list(), savefig=True):
    '''Plots percentage of distance moved in the outer zone of the well over the total distance per condition
    well_diameter: the diameter of the bottom of each well in mm'''

    well_area = np.pi * (well_diameter/2)**2
    inner_area = well_area / 2
    inner_radius = np.sqrt(inner_area/np.pi)

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    fig, ax = plt.subplots(figsize=(10, 10))

    for conc in concentrations:

        condition_means = []
        condition_sems = []
        for condition in conditions:
            distance_df = mcam.dataframes[condition]['distance'][conc]
            dist_cols = [col for col in distance_df if col.startswith('distance_traveled')]
            distance_df = distance_df[dist_cols]

            tracking_df = mcam.dataframes[condition]['tracking'][conc]
            center_cols = [col for col in tracking_df if col.startswith('dist_from_center')]
            tracking_df = tracking_df[center_cols]

            percentages = []

            for fish in range(len(center_cols)):
                tracking_col = tracking_df.iloc[:, fish]
                distance_col = distance_df.iloc[:, fish]
                total_outer_distance = distance_col[np.where(tracking_col*1000 > inner_radius)[0]].sum()
                total_distance = distance_col.sum()
                percent_thigmotaxis = (total_outer_distance / total_distance) * 100
                percentages.append(percent_thigmotaxis)

            percentages = np.array(percentages)
            condition_means.append(percentages.mean())
            condition_sems.append(sem(percentages))

        condition_means = np.array(condition_means)
        condition_sems = np.array(condition_sems)

        ax.errorbar(conditions, condition_means, yerr=condition_sems, label=str(conc))

    ax.legend()
    ax.set_ylim([0, 100])
    ax.set_ylabel('% Thigmotaxis (distance)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, f'percent_thigmotaxis_distance_line.pdf')


def photomotor_aggregates(mcam, conditions=list(), concentrations=list(), savefig=True):
    '''Aggregates the distance data per 5 minute epoch'''

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    x = np.arange(len(concentrations))  # the label locations
    width = 0.25  # the width of the bars

    starts_stops = get_stimulus_starts_stops(mcam)

    fig, axs = plt.subplots(7, 1, figsize=(25, 10*len(starts_stops)), sharey=True)

    for i, stim in enumerate(starts_stops):
        multiplier = 0

        for condition in conditions:
            aggregates = list()
            sems = list()

            offset = width * multiplier

            for j, conc in enumerate(concentrations):
                df = mcam.dataframes[condition]['distance'][conc]
                cols = [col for col in df if col.startswith('distance_traveled')]
                stimdf = df[cols].iloc[starts_stops[stim][0]:starts_stops[stim][1], :]
                aggregates.append(stimdf.sum().mean())
                sems.append(stimdf.sum().sem())

                axs[i].scatter(np.repeat(x[j] + offset, stimdf.sum().size), stimdf.sum(), color='lightgray', alpha=0.5, zorder=10)

            axs[i].bar(x + offset, aggregates, width, yerr=sems, label=condition)

            multiplier += 1

        axs[i].legend()
        axs[i].set_xticks(x + width, concentrations)
        axs[i].set_title(stim, fontsize=18)
        axs[i].set_ylabel('Total distance traveled (m)', fontsize=18)
        axs[i].set_xlabel('DOI concentration (µg/ml)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, 'photomotor_aggregates.pdf')


def photomotor_aggregates_stim_repetition(mcam, conditions=list(), concentrations=list(), savefig=True):
    '''
    Aggregates the distance data per 5 minute epoch and plots line for each fish.
    Each fish is colored by concentration.
    x-axis is condition, with each stimulus repetition grouped together.
    '''

    if len(conditions) == 0:
        conditions = list(mcam.dataframes.keys())

    if len(concentrations) == 0:
        concentrations = mcam.concentrations

    x = np.arange(len(conditions))  # the label locations
    width = 0.25  # the width of the bars

    starts_stops = get_stimulus_starts_stops(mcam)
    stimuli = ['light_epoch', 'dark_epoch']

    fig, axs = plt.subplots(len(stimuli), 1, figsize=(25, 10*len(stimuli)), sharey=True)
    colors = color_palette()
    conc_colors = {
        0: 'eno',
        0.05: 'shale_blue',
        0.5: 'prussian_blue',
        2.5: 'piedmont',
        5: 'persimmon',
        50: 'copper'
    }

    for i, stim in enumerate(stimuli):
        repetitions = [key for key in starts_stops if key.startswith(stim)]

        for j, condition in enumerate(conditions):
            for conc in concentrations:
                df = mcam.dataframes[condition]['distance'][conc]

                all_fish = list()

                for fish in df:
                    if fish.startswith('distance_traveled'):
                        fishdf = df.loc[:, fish]
                        fish_dists = list()
                        offsets = list()

                        multiplier = 0

                        for rep in repetitions:
                            stimdf = fishdf.iloc[starts_stops[rep][0]:starts_stops[rep][1]]
                            fish_dists.append(stimdf.sum())
                            offsets.append(width * multiplier)
                            multiplier += 1

                        all_fish.append(fish_dists)
                        axs[i].plot(np.array(offsets) + x[j], fish_dists, color=colors[conc_colors[conc]], alpha=0.25)

                means = np.array(all_fish).mean(axis=0)
                axs[i].plot(np.array(offsets) + x[j], means, color=colors[conc_colors[conc]], label=conc, linewidth=4)

        handles, labels = axs[i].get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        axs[i].legend(*zip(*unique))

        axs[i].set_title(stim, fontsize=18)
        axs[i].set_xticks(x + width, conditions)
        axs[i].set_ylabel('Total distance traveled (m)', fontsize=18)

    if savefig:
        save_figure(mcam, fig, 'epoch_repetitions.pdf')
