import matplotlib.pyplot as plt
from mcam import MCAM


def plot_distance_per_condition(mcam, conditions, concentrations):
    '''Plots distance over time of given conditions and concentrations'''

    fig, axs = plt.subplots(1, len(conditions),
                            figsize=(12*len(conditions), 10),
                            sharey=True, sharex=True)

    for i, condition in enumerate(conditions):
        dfs = mcam.dataframes[condition]['distance']

        for conc in concentrations:
            df = dfs[conc]