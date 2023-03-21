"""
Created on 3/21/23

@author: aoshotse
"""

import pandas as pd

class omrFish:
    # TODO: figure out how to make this work with actual data (how to load in id and all other relevant info).
    # could use the path to parse out the id, age, and concentration (refer to fishflux)
    def __init__(self, fish_path, id, active):
        self.stimuli = "stim"
        self.active = active
        self.fish_path = fish_path
        self.id = id
        self.load_fish_details()

    "EK_6dpf_DOI_0ugml_20230206"

    "[EK, 6dpf, DOI, 0ugml, 20230206]"

    def load_fish_details(self):
        string_path = str(self.fish_path)
        string_list = string_path.split("_")
        self.concentration = int(string_list[-2][0])
        self.age = int(string_list[1][0])

    def load_fish_to_dataframe(self, df_file):
        # load in the dataframe
        df = pd.read_csv(df_file)

        fish_dict = {
            'id': self.id,
            'age': self.age,
            'concentration': self.concentration,
            'stimuli': self.stimuli,
            'active': self.active
        }
        df = df.append(fish_dict, ignore_index=True)
        df.to_csv(df_file, index=False)