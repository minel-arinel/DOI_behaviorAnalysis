"""
Created on 3/21/23

@author: aoshotse
"""

import pandas as pd


class omrFish:
    # TODO: figure out how to make this work with actual data (how to load in id and all other relevant info).
    # could use the path to parse out the id, age, and concentration (refer to fishflux)
    def __init__(self, fish_path, id, active):
        self.active = active
        self.fish_path = fish_path
        self.id = id
        self.load_fish_details()

    def load_fish_details(self):
        string_path = str(self.fish_path)
        string_list = string_path.split("_")
        conc = string_list[-2]
        index = conc.rfind("ugml")
        self.concentration = int(conc[:index])
        self.age = int(string_list[1][0])
        if (self.concentration == 0):
            self.stimuli = "eggH20"
        else:
            self.stimuli = "DOI"

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
        # converting the dictionary into a pandas dataframe
        new_data = pd.DataFrame(fish_dict, index=[0])
        df = pd.concat([df, new_data], axis=0)
        df.to_csv(df_file, index=False)

