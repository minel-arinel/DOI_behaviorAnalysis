from fish import Fish
import os
import pandas as pd


class Tank:
    def __init__(self, folder_path, prefix='EK', **kwargs):
        self.folder_path = folder_path
        self.bout_dfs = list()

        self.process_fish(prefix, **kwargs)
        self.bout_df = pd.concat(self.bout_dfs, ignore_index=True)
        self.bout_df.to_hdf(os.path.join(folder_path, 'bout_df.h5'), key='bout_df', mode='w')

    def process_fish(self, prefix, **kwargs):
        with os.scandir(self.folder_path) as entries:  # For each experiment folder
            for entry in entries:
                if os.path.isdir(entry.path) and entry.name.startswith(prefix):

                    with os.scandir(entry.path) as id_entries:  # For each fish for that experiment
                        for id_entry in id_entries:
                            if os.path.isdir(id_entry.path):
                                fish = Fish(id_entry.path, prefix=prefix, **kwargs)
                                self.bout_dfs.append(fish.bout_df)