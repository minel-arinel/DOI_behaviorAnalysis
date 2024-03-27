"""
Created on 11/26/23
@author: aloyeoshotse
"""
import pandas as pd
import os

"""
Questions: 
"""

"""    
COMPLETED: 
    - adding specific to startle_response_speed and startle_response_distance
    - making the df being overwritten w/ new y value
    - change the fish_id when the dose is changed, but keep consistent when treatment is different (baseline vs 
            drugtreated vs 24 hr) --> can make a dataframe keeping track of this or a dictionary
    - Make % thigomotaxis time in outer circle function --> 
            - specific --> time in thigomotaxis (epoch_time_of_thigmotaxis) / total epoch duration (before startle --> 
                    end of epoch - start of epoch) * 100
            - not specific --> average the specific ones 
    - Make % thigmotaxis distance in outer circle -->
        - specific --> (total thig distance in that epoch / total distance in that epoch)  * 100
     - automate making the csv file --> all of the possibilities should be ran in this method
            - figure out how to decide between photomotor, startle, and thigmotaxis
            - add a feature that determines fish id by column or row
    - Make a note that the mean is distance / frame (average per frame) 
    - 24 HR RECOVERY
        - Wells are A1-A6...D1-D6
            - each column gets a different concentration 
            - first experiment had no fish removed
    - update the function "update_data_file" to filter removed fish for the
        24 hr recovery period (could also put it in populate_df_dictionary()" before
        the update_data_file function is called (line 212)
    - columns to skip / fish_id to remove:
        0.0 conc --> 5, 11, 12 // (5, 11, 12)
        .05 conc --> 17 // (37)
        0.5 conc --> 17 // (57)
        2.5 conc --> 8 // (68)
        5.0 conc --> 6, 8, 13, 14, 20 // (86, 88, 93, 94, 100)
        50 conc --> 

TODO:
    - add something to account for column and rows --> take
    - get on the server
    - download df to lab computers
    - once df is ready, we need a checkpoint to confirm it
        - write 1 function 
        - dark epoch (each epoch) -->  total distance moved
            - all concentrations
            - baseline, drugtreated, and recovery 
            - get average of all fish
    - add function to handle duplicated row and to replace values (pandas.duplicated())
    """
"""

Volinanserin Notes:

- remove recovery column
- df for each separate experiment 

- MCAM/20231024_NaumannLab_Volinanserin_Experiments
    -  5 experiments
    - Given by rows
    - 4 concentrations
    - 0 micrograms/mL
    - 5
    - 10
    - 20
    - test 4 concentration on 50 ug/mL DOI
    - baseline, then add volinanserin, 20 min later add 50 ug/mL DOI, and 10 min later 
    - The control (no volinanserin, and only DOI) of this should match the the DOI experiments at 50ug/mL
    - Trying to see what concentration of volinanserin works best with 50ug/mL
    
- MCAM/20231128_NaumannLab_DOI_Volinanserin_Experiments
    - 5 experiments 
    - different concentrations of DOI (same as DOI experiments)
    - all 20 ug/ML of volinanserin
    - trying to see what concentration will the dosage of volinanserin remove the affect of DOI (how it interacrs with
        the DOI)
    - The volinanserin increased the activity in all DOI concentrations
    - fish_id is the same
    
    
- MCAM/20231201_NaumannLab_Volinanserin_Eggwater_Experiments
    - 1 single experiment
    - wanted to confirm what volinanserin by itself is doing
    - 2 concentration of volinanserin
        - 0 and 20
    - volinanserin increases locomotion
    - These are column based (3 columns will get the same concentration)
    
- RNA in situ hybridization --> takes a mRNA, create an opposite primer with a fluorescent tag
    
Notes:
    1) 
        have a dictionary for the final df we want to make: --> make this using parallel lists that much up the fishes with their 
        data
        column 1: y -> float, this should have the numeric value of what you calculated
            column 2 (metadata): metric -> str, this should be a unique name for your individual metric (e.g., “sum_distance_dark_epoch_all”)
            column 3: fish -> int, a fish ID for each individual fish that you will calculate these for
            column 4 (metadata): baseline -> int, True if during baseline condition, False if otherwise
            column 5 (metadata): recovery -> int, True if during recovery condition, False if otherwise
        column 6: drug -> int, True if under the influence of DOI, False if otherwise (i.e., False if baseline or recovery)
            column 7 (metadata): dose -> float, the concentration of DOI tested, 0 if negative control
    
    2) across baseline, treated, and recovery the fish is the same in the same order within the same concentration
        when a new concentration is introduced, we need to add new fish ID
    
    3) 24hr drug treated is same as baseline 
    
    4) leave recovery for last because some fish died (24 hr recovery should have less fish)
    
    5) Add the different individual epochs to metric name
    
    6) well diameter = 15mm
    
"""

epoch_time_ranges = {
    # 1 is light epoch and 0 is dark epoch
    # accounts for and removes the 15 seconds of stimuli
    0: [(600, 885), (1200, 1485), (1800, 2085)],
    1: [(300, 585), (900, 1185), (1500, 1785)]
}

startle_time_ranges = {
    "light": [(884, 900), (1484, 1500)],
    "dark": [(584, 600), (1784, 1800)],
    "vibration": [(1184, 1200), (2084, 2110)]
}

fish_nums_columns = {
    0.0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    0.05: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    0.5: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    2.5: [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    5.0: [81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
    50.0: [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
}

recovery_fish_nums_columns = {
    0.0: [1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20],
    0.05: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40],
    0.5: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60],
    2.5: [61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    5.0: [81, 82, 83, 84, 85, 87, 89, 90, 91, 92, 95, 96, 97, 98, 99],
    50.0: [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
}

thigmotaxic_distance = 0.0053

csv_file_path = "csv_files/baseline"


def run_all_metric_functions(path):
    global csv_file_path
    behavior = ["photomotor", "startle", "thigmotaxis"]
    period = ["light", "dark", "vibration"]
    metric = ["max", "mean", "sum"]
    specific = [True, False]
    csv_file_path = path

    for csv_file in os.listdir(csv_file_path):
        print(csv_file)
        for i in range(len(behavior)):
            for x in range(len(period)):
                for y in range(len(metric)):
                    for j in range(len(specific)):
                        if behavior[i] != "startle" and period[x] == "vibration":
                            # Startle are the only behavior that has vibration stimuli
                            pass
                        elif "speed" in csv_file and metric[y] == "sum":
                            # We do not calculate sum of speed
                            pass
                        elif behavior[i] != "thigmotaxis" and "tracking" in csv_file:
                            # Only calculate thigmotaxis if it is a tracking file
                            pass
                        elif behavior[i] == "thigmotaxis" and metric[y] != "mean":
                            # Thigmotaxis functions should only run once
                            pass
                        else:
                            process_file(csv_file, behavior[i], period[x], metric[y], specific[j])
        print(csv_file + " - completed!")


def process_file(file_name, behavior, period, metric="", specific=False):
    file_name = os.path.join(csv_file_path, file_name)
    df = pd.read_csv(file_name, low_memory=False)
    metadata = extract_metadata(file_name)
    tracking = metadata[0][1]  # will either be distance or speed
    data = []

    if "photomotor" in behavior:
        if period == "light":
            light = 1
        elif period == "dark":
            light = 0
        else:
            print("For the period parameter where behavior is 'photomotor', input either 'light' (for light epoch) or "
                  "'dark' (for dark epoch)")
            return
        data.extend(list(handle_photomotor_behavior(df, light, tracking, metric, specific)))

    elif "startle" in behavior:
        if period == "light" or period == "dark" or period == "vibration":
            data.extend(handle_startle_response(df, period, tracking, metric, specific))
        else:
            print("For the period parameter where behavior is 'startle', input either 'light' (for light flash), "
                  "'dark' (for dark flash), or 'vibration' (for vibration startle)")
            return

    elif "thigmotaxis" in behavior:
        if period == "light":
            light = 1
        elif period == "dark":
            light = 0
        else:
            print("For the period parameter where behavior is 'thigmotaxis', input either 'light' (for light epoch) or "
                  "'dark' (for dark epoch)")
            return
        df2 = retrieve_distance_csv(file_name)
        data.extend(handle_thigmotaxis(df, df2, light, specific))

    else:
        print("Input either 'photomotor', 'startle', or 'thigmotaxis' for 'behavior'.")
        return

    for metric_data in data:
        populate_df_dictionary(metadata, metric_data, specific)


def extract_metadata(file_name):
    file_name = file_name.split("/")[-1] if "/" in file_name else file_name.split("\\")[-1]
    info = [("metric", file_name.split("_")[1] if "recovery" not in file_name else file_name.split("_")[2]),
            ("baseline", 1 if "baseline" in file_name else 0), ("recovery", 1 if "recovery" in file_name else 0),
            ("dose", file_name.split("_")[-1][:-4])]
    return info


def retrieve_distance_csv(filename):
    filename = filename.split("/")[-1] if "\\" not in filename else filename.split("\\")[-1]
    lst = filename.split("_")
    if "recovery" not in lst:
        lst[1] = "distance"
    else:
        lst[2] = "distance"
    distance_file_name = "_".join(lst) if ".csv" in lst[-1] else "_".join(lst) + ".csv"
    return pd.read_csv(os.path.join(os.getcwd(), csv_file_path, distance_file_name))


def populate_df_dictionary(metadata, metric_data, specific):

    data_dict = {
        "fish": 0,
        "metric": "",
        "y": 0.0,
        "baseline": 0,
        "recovery": 0,
        "drug": 0,
        "dose": 0.0
    }

    for col_name, value in metadata:
        data_dict[col_name] = value

    if specific:
        metric_name = metric_data[0].split(";")
    else:
        metric_name = [metric_data[0]]

    if data_dict["baseline"] == 1:
        data_dict["condition"] = "baseline"
    elif data_dict["recovery"] == 1:
        data_dict["condition"] = "recovery"
    elif data_dict["baseline"] == 0 and data_dict["recovery"] == 0:
        data_dict["condition"] = "drugtreated"

    data_dict.pop('recovery')
    data_dict.pop('baseline')
    data_dict.pop('drug')

    values = metric_data[1]
    for fish_ids, fish_metrics in values:
        for x in range(len(fish_ids)):
            for y in range(len(metric_name)):
                data_dict['metric'] = metric_name[y]
                data_dict["fish"] = fish_ids[x]
                data_dict["y"] = round(fish_metrics[x][y], 8)
                update_data_file(data_dict, "MCAM_fish_metrics.csv")


def adjust_fish_id(dictionary):
    global fish_nums_columns
    global recovery_fish_nums_columns

    concentration = float(dictionary["dose"])
    index = dictionary["fish"] - 1
    condition = dictionary["condition"]

    return fish_nums_columns[concentration][index] if condition != "recovery" else recovery_fish_nums_columns[concentration][index]


def update_data_file(dictionary, file_name):
    global fish_nums_columns
    file_name = os.path.join(csv_file_path, file_name)
    file_path = os.path.abspath(file_name)
    file_exists = os.path.isfile(file_path)
    dictionary["fish"] = adjust_fish_id(dictionary)
    df = pd.DataFrame([dictionary])
    df['dose'] = df['dose'].astype(float)

    if file_exists:
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(file_path, index=False)
        return combined_df

    df.to_csv(file_path, index=False)
    return df


def handle_photomotor_behavior(df, light, tracking, metric, specific):
    metric_name_dict = {}
    light_name = "light" if light == 1 else "dark"
    metric_name = ""
    if specific:
        name = f"{metric}_{tracking}_{light_name}_epoch_"
        for x in range(1, 4):
            metric_name += name + str(x) + ";" if x != 3 else name + str(x)
    else:
        metric_name = f"{metric}_{tracking}_{light_name}_epoch_all"

    if tracking == "distance":
        metric_name_dict[metric_name] = [epoch_distance(df, light, metric, specific)]
        return metric_name_dict.items()
    elif tracking == "speed":
        metric_name_dict[metric_name] = [epoch_speed(df, light, metric, specific)]
        return metric_name_dict.items()


def handle_startle_response(df, startle_response, tracking, metric, specific):
    metric_name_dict = {}
    startle_response_name = ""
    diff_first_last_rep_name = ""
    average_latency_name = ""
    if specific:
        name_startle = f"{metric}_{tracking}_{startle_response}_startle_rep_"
        name_diff = f"diff_{metric}_{tracking}_{startle_response}_startle_rep_"
        name_avg_lat = f"avg_latency_{metric}_{tracking}_{startle_response}_startle_rep_"
        for x in range(1, 3):
            startle_response_name += name_startle + str(x) + ";" if x != 2 else name_startle + str(x)
            diff_first_last_rep_name += name_diff + str(x) + ";" if x != 2 else name_diff + str(x)
            average_latency_name += name_avg_lat + str(x) + ";" if x != 2 else name_avg_lat + str(x)
    else:
        startle_response_name = f"{metric}_{tracking}_{startle_response}_startle_all"
        diff_first_last_rep_name = f"diff_{metric}_{tracking}_{startle_response}_startle_all"
        average_latency_name = f"avg_latency_{metric}_{tracking}_{startle_response}_startle_all"

    if tracking == "distance":
        metric_name_dict[startle_response_name] = [startle_response_distance(df, startle_response, metric, specific)]
        metric_name_dict[diff_first_last_rep_name] = [diff_first_last_startle_rep_distance(df, startle_response, metric,
                                                                                           specific)]
        metric_name_dict[average_latency_name] = [average_latency_startle_distance(df, startle_response, specific)]
        return metric_name_dict.items()
    elif tracking == "speed":
        metric_name_dict[startle_response_name] = [startle_response_speed(df, startle_response, metric, specific)]
        metric_name_dict[diff_first_last_rep_name] = [diff_first_last_startle_rep_speed(df, startle_response, metric,
                                                                                        specific)]
        metric_name_dict[average_latency_name] = [average_latency_startle_speed(df, startle_response, specific)]
        return metric_name_dict.items()


def handle_thigmotaxis(df, df2, light, specific):
    metric_name_dict = {}
    light_name = "light" if light == 1 else "dark"
    mean_dist_name = ""
    thig_time_name = ""
    total_dist_name = ""
    per_thig_time_name = ""
    per_thig_dist_name = ""

    if specific:
        name_mean_dist = f"thigmotaxis_mean_distance_{light_name}_epoch_"
        name_time = f"thigmotaxis_total_time_{light_name}_epoch_"
        name_total_dist = f"thigmotaxis_total_distance_{light_name}_epoch_"
        name_per_thig_time = f"thigmotaxis_percent_time_{light_name}_epoch_"
        name_per_thig_dist = f"thigmotaxis_percent_distance_{light_name}_epoch_"

        for x in range(1, 4):
            mean_dist_name += name_mean_dist + str(x) + ";" if x != 3 else name_mean_dist + str(x)
            thig_time_name += name_time + str(x) + ";" if x != 3 else name_time + str(x)
            total_dist_name += name_total_dist + str(x) + ";" if x != 3 else name_total_dist + str(x)
            per_thig_time_name += name_per_thig_time + str(x) + ";" if x != 3 else name_per_thig_time + str(x)
            per_thig_dist_name += name_per_thig_dist + str(x) + ";" if x != 3 else name_per_thig_dist + str(x)

    else:
        mean_dist_name = f"thigmotaxis_mean_distance_{light_name}_epoch_all"
        thig_time_name = f"thigmotaxis_total_time_{light_name}_epoch_all"
        total_dist_name = f"thigmotaxis_total_distance_{light_name}_epoch_all"
        per_thig_time_name = f"thigmotaxis_percent_time_{light_name}_epoch_all"
        per_thig_dist_name = f"thigmotaxis_percent_distance_{light_name}_epoch_all"

    metric_name_dict[mean_dist_name] = [epoch_mean_dist_thigmotaxis(df, light, specific)]
    metric_name_dict[thig_time_name] = [epoch_time_of_thigmotaxis(df, light, specific)]
    metric_name_dict[total_dist_name] = [epoch_total_distance_thigmotaxis(df, df2, light, specific)]
    metric_name_dict[per_thig_time_name] = [percent_thig_time(df, light, specific)]
    metric_name_dict[per_thig_dist_name] = [percent_thig_distance(df, df2, light, specific)]
    return metric_name_dict.items()


def calculate_metric(filtered_data, metric):
    totals = []
    if metric == "sum":
        totals.append(filtered_data.sum())
    elif metric == "mean":
        n = len(filtered_data)
        totals.append(filtered_data.sum() / n if n != 0 else 0)
    elif metric == "max":
        totals.append(max(filtered_data) if len(filtered_data) > 0 else 0)
    return totals


def epoch_distance(df, light, metric, specific):
    global epoch_time_ranges
    distance_df = df.filter(regex='distance_traveled|time|stim_name')
    stim_name = "light_epoch" if light == 1 else "dark_epoch"
    fish_ids = [fish_id for fish_id, column in enumerate(distance_df) if "distance_traveled" in column]
    metric_lst = []

    for column in distance_df:
        if "distance_traveled" in column:
            totals = []
            for start, end in epoch_time_ranges[light]:
                filtered_time = distance_df.loc[distance_df['time'].between(start, end, inclusive='left')]
                stim_filtered = filtered_time[filtered_time['stim_name'] == stim_name][column]
                totals.extend(calculate_metric(stim_filtered, metric))
            if specific:
                metric_lst.append(totals)
            else:
                aggregated_metric = max(totals) if metric == "max" else sum(totals) if metric == "sum" else sum(
                    totals) / len(totals)
                metric_lst.append([aggregated_metric])
    return fish_ids, metric_lst


def epoch_speed(df, light, metric, specific):
    if metric == "sum":
        return "Sum is not a valid metric for speed. Input either mean or max."

    global epoch_time_ranges
    speed_df = df.filter(regex='^(?!.*average).*speed|time|stim_name')
    stim_name = "light_epoch" if light == 1 else "dark_epoch"
    fish_ids = [fish_id for fish_id, column in enumerate(speed_df) if "speed" in column]
    metric_lst = []

    for column in speed_df:
        if 'speed' in column:
            totals = []
            for start, end in epoch_time_ranges[light]:
                filtered_time = speed_df.loc[speed_df['time'].between(start, end, inclusive='left')]
                stim_filtered = filtered_time[filtered_time['stim_name'] == stim_name][column]
                totals.extend(calculate_metric(stim_filtered, metric))
            if specific:
                metric_lst.append(totals)
            else:
                aggregated_metric = max(totals) if metric == "max" else sum(totals) / len(totals)
                metric_lst.append([aggregated_metric])
    return fish_ids, metric_lst


def startle_response_distance(df, startle_response, metric, specific):
    global startle_time_ranges
    distance_df = df.filter(regex='distance_traveled|time|stim_name')
    fish_ids = [fish_id for fish_id, column in enumerate(distance_df) if "distance_traveled" in column]
    metric_lst = []

    if startle_response == "light":
        stim_name = "light_flash"
    elif startle_response == "dark":
        stim_name = "dark_flash"
    else:
        stim_name = "vibration_startle"

    for column in distance_df:
        if "distance_traveled" in column:
            totals = []
            for start, end in startle_time_ranges[startle_response]:
                filtered_time = distance_df.loc[distance_df['time'].between(start, end, inclusive='left')]
                stim_filtered = filtered_time[filtered_time['stim_name'] == stim_name][column]
                totals.extend(calculate_metric(stim_filtered, metric))
            if specific:
                metric_lst.append(totals)
            else:
                aggregated_metric = max(totals) if metric == "max" else sum(totals) if metric == "sum" else sum(
                    totals) / len(totals)
                metric_lst.append([aggregated_metric])
    return fish_ids, metric_lst


def startle_response_speed(df, startle_response, metric, specific):
    if metric == "sum":
        return "Sum is not a valid metric for speed. Input either mean or max."

    global startle_time_ranges
    speed_df = df.filter(regex='^(?!.*average).*speed|time|stim_name')
    fish_ids = [fish_id for fish_id, column in enumerate(speed_df) if "speed" in column]
    metric_lst = []

    if startle_response == "light":
        stim_name = "light_flash"
    elif startle_response == "dark":
        stim_name = "dark_flash"
    else:
        stim_name = "vibration_startle"

    for column in speed_df:
        if 'speed' in column:
            totals = []
            for start, end in startle_time_ranges[startle_response]:
                filtered_time = speed_df.loc[speed_df['time'].between(start, end, inclusive='left')]
                stim_filtered = filtered_time[filtered_time['stim_name'] == stim_name][column]
                totals.extend(calculate_metric(stim_filtered, metric))
            if specific:
                metric_lst.append(totals)
            else:
                aggregated_metric = max(totals) if metric == "max" else sum(totals) / len(totals)
                metric_lst.append([aggregated_metric])
    return fish_ids, metric_lst


def diff_first_last_startle_rep_distance(df, startle_response, metric, specific):
    global startle_time_ranges
    distance_df = df.filter(regex='distance_traveled|time|stim_name|stim_num')
    fish_ids = [fish_id for fish_id, column in enumerate(distance_df) if "distance_traveled" in column]
    metric_lst = []

    if startle_response == "light":
        stim_name = "light_flash"
    elif startle_response == "dark":
        stim_name = "dark_flash"
    else:
        stim_name = "vibration_startle"

    for column in distance_df:
        if "distance_traveled" in column:
            stim_num = 0
            stim_num_lst = []
            for start, end in startle_time_ranges[startle_response]:
                totals = []
                for x in range(2):
                    filtered_time = distance_df.loc[distance_df['time'].between(start, end, inclusive='left')]
                    stim_filtered = filtered_time[(filtered_time['stim_name'] == stim_name) &
                                                  (filtered_time['stim_num'] == stim_num)][column]
                    totals.extend(calculate_metric(stim_filtered, metric))
                    stim_num += 4 if stim_num == 0 or stim_num == 5 else 1
                stim_num_lst.append(totals[0] - totals[1])
            if specific:
                metric_lst.append(stim_num_lst)
            else:
                average_diff = (stim_num_lst[0] + stim_num_lst[1]) / len(stim_num_lst)
                metric_lst.append([average_diff])
    return fish_ids, metric_lst


def diff_first_last_startle_rep_speed(df, startle_response, metric, specific):
    if metric == "sum":
        return "Sum is not a valid metric for speed. Input either mean or max."

    global startle_time_ranges
    speed_df = df.filter(regex='^(?!.*average).*speed|time|stim_name|stim_num')
    fish_ids = [fish_id for fish_id, column in enumerate(speed_df) if "speed" in column]
    metric_lst = []

    if startle_response == "light":
        stim_name = "light_flash"
    elif startle_response == "dark":
        stim_name = "dark_flash"
    else:
        stim_name = "vibration_startle"

    for column in speed_df:
        if 'speed' in column:
            stim_num = 0
            stim_num_lst = []
            for start, end in startle_time_ranges[startle_response]:
                totals = []
                for x in range(2):
                    filtered_time = speed_df.loc[speed_df['time'].between(start, end, inclusive='left')]
                    stim_filtered = filtered_time[(filtered_time['stim_name'] == stim_name) &
                                                  (filtered_time['stim_num'] == stim_num)][column]
                    totals.extend(calculate_metric(stim_filtered, metric))
                    stim_num += 4 if stim_num == 0 or stim_num == 5 else 1
                stim_num_lst.append(totals[0] - totals[1])
            if specific:
                metric_lst.append(stim_num_lst)
            else:
                average_diff = (stim_num_lst[0] + stim_num_lst[1]) / len(stim_num_lst)
                metric_lst.append([average_diff])
    return fish_ids, metric_lst


def average_latency_startle_distance(df, startle_response, specific):
    global startle_time_ranges
    distance_df = df.filter(regex='distance_traveled|time|stim_name|stim_num')
    fish_ids = [fish_id for fish_id, column in enumerate(distance_df) if "distance_traveled" in column]
    metric_lst = []

    if startle_response == "light":
        stim_name = "light_flash"
    elif startle_response == "dark":
        stim_name = "dark_flash"
    else:
        stim_name = "vibration_startle"

    for column in distance_df:
        if "distance_traveled" in column:
            stim_num = 0
            latency_lst = []
            for start, end in startle_time_ranges[startle_response]:
                times = []
                for x in range(5):
                    filtered_time = distance_df.loc[distance_df['time'].between(start, end, inclusive='left')]
                    stim_filtered = filtered_time[(filtered_time['stim_name'] == stim_name) &
                                                  (filtered_time['stim_num'] == stim_num)][column]
                    initial_time = distance_df._get_value(stim_filtered.index[0], 'time')
                    peak_time = filtered_time.loc[filtered_time[column] == stim_filtered.max(), 'time'].iloc[0]
                    times.append(peak_time - initial_time)
                    stim_num += 1
                avg_latency = sum(times) / len(times)
                latency_lst.append(avg_latency)
            if specific:
                metric_lst.append(latency_lst)
            else:
                avg_rep_latency = sum(latency_lst) / len(latency_lst)
                metric_lst.append([avg_rep_latency])
    return fish_ids, metric_lst


def average_latency_startle_speed(df, startle_response, specific):
    global startle_time_ranges
    speed_df = df.filter(regex='^(?!.*average).*speed|time|stim_name|stim_num')
    fish_ids = [fish_id for fish_id, column in enumerate(speed_df) if "speed" in column]
    metric_lst = []

    if startle_response == "light":
        stim_name = "light_flash"
    elif startle_response == "dark":
        stim_name = "dark_flash"
    else:
        stim_name = "vibration_startle"

    for column in speed_df:
        if 'speed' in column:
            stim_num = 0
            latency_lst = []
            for start, end in startle_time_ranges[startle_response]:
                times = []
                for x in range(5):
                    filtered_time = speed_df.loc[speed_df['time'].between(start, end, inclusive='left')]
                    stim_filtered = filtered_time[(filtered_time['stim_name'] == stim_name) &
                                                  (filtered_time['stim_num'] == stim_num)][column]
                    initial_time = speed_df._get_value(stim_filtered.index[0], 'time')
                    peak_time = filtered_time.loc[filtered_time[column] == stim_filtered.max(), 'time'].iloc[0]
                    times.append(peak_time - initial_time)
                    stim_num += 1
                avg_latency = sum(times) / len(times)
                latency_lst.append(avg_latency)
            if specific:
                metric_lst.append(latency_lst)
            else:
                avg_rep_latency = sum(latency_lst) / len(latency_lst)
                metric_lst.append([avg_rep_latency])
    return fish_ids, metric_lst


def epoch_mean_dist_thigmotaxis(df, light, specific):
    global epoch_time_ranges
    tracking_df = df.filter(regex='^(?!.*average).*dist_from_center|time|stim_name')
    stim_name = "light_epoch" if light == 1 else "dark_epoch"
    fish_ids = [fish_id for fish_id, column in enumerate(tracking_df) if "dist_from_center" in column]
    metric_lst = []

    for column in tracking_df:
        if "dist_from_center" in column:
            totals = []
            for start, end in epoch_time_ranges[light]:
                filtered_time = tracking_df.loc[tracking_df['time'].between(start, end, inclusive='left')]
                stim_filtered = filtered_time[filtered_time['stim_name'] == stim_name][column]
                totals.extend(calculate_metric(stim_filtered, "mean"))
            if specific:
                metric_lst.append(totals)
            else:
                aggregated_metric = sum(totals) / len(totals)
                metric_lst.append([aggregated_metric])
    return fish_ids, metric_lst


def percent_thig_time(df, light, specific):
    global epoch_time_ranges
    global thigmotaxic_distance
    tracking_df = df.filter(regex='^(?!.*average).*dist_from_center|time|stim_name')
    stim_name = "light_epoch" if light == 1 else "dark_epoch"
    fish_ids = [fish_id for fish_id, column in enumerate(tracking_df) if "dist_from_center" in column]
    metric_lst = []
    time_per_index = 0.033328974

    for column in tracking_df:
        if "dist_from_center" in column:
            totals = []
            for start, end in epoch_time_ranges[light]:
                filtered_time = tracking_df.loc[tracking_df['time'].between(start, end, inclusive='left')]
                stim_filtered = filtered_time[filtered_time['stim_name'] == stim_name]
                thig_filtered = stim_filtered[stim_filtered[column] > thigmotaxic_distance]["time"]
                total_time_thigmotaxis = len(thig_filtered) * time_per_index
                per_thig_time = (total_time_thigmotaxis / (end - start)) * 100
                totals.append(per_thig_time)
            if specific:
                metric_lst.append(totals)
            else:
                aggregated_metric = sum(totals) / len(totals)
                metric_lst.append([aggregated_metric])
    return fish_ids, metric_lst


def epoch_time_of_thigmotaxis(df, light, specific):
    global epoch_time_ranges
    global thigmotaxic_distance
    tracking_df = df.filter(regex='^(?!.*average).*dist_from_center|time|stim_name')
    stim_name = "light_epoch" if light == 1 else "dark_epoch"
    fish_ids = [fish_id for fish_id, column in enumerate(tracking_df) if "dist_from_center" in column]
    metric_lst = []
    time_per_index = 0.033328974

    for column in tracking_df:
        if "dist_from_center" in column:
            totals = []
            for start, end in epoch_time_ranges[light]:
                filtered_time = tracking_df.loc[tracking_df['time'].between(start, end, inclusive='left')]
                stim_filtered = filtered_time[filtered_time['stim_name'] == stim_name]
                thig_filtered = stim_filtered[stim_filtered[column] > thigmotaxic_distance]["time"]
                total_time_thigmotaxis = len(thig_filtered) * time_per_index
                totals.append(total_time_thigmotaxis)
            if specific:
                metric_lst.append(totals)
            else:
                aggregated_metric = sum(totals)
                metric_lst.append([aggregated_metric])
    return fish_ids, metric_lst


def percent_thig_distance(df, df2, light, specific):
    global epoch_time_ranges
    global thigmotaxic_distance
    tracking_df = df.filter(regex='^(?!.*average).*dist_from_center|time|stim_name')
    distance_df = df2.filter(regex='distance_traveled|time|stim_name')
    stim_name = "light_epoch" if light == 1 else "dark_epoch"
    fish_ids = [fish_id for fish_id, column in enumerate(tracking_df) if "dist_from_center" in column]
    metric_lst = []
    distance_column = 1

    for column in tracking_df:
        if "dist_from_center" in column:
            totals = []
            for start, end in epoch_time_ranges[light]:
                filtered_time = tracking_df.loc[tracking_df['time'].between(start, end, inclusive='left')]
                stim_filtered = filtered_time[filtered_time['stim_name'] == stim_name]
                thig_filtered = stim_filtered[stim_filtered[column] > thigmotaxic_distance]["time"]
                thig_filtered_ind_lst = list(thig_filtered.index.values)

                filtered_dist_time = distance_df.loc[distance_df['time'].between(start, end, inclusive='left')]
                col_name = distance_df.columns[distance_column]
                stim_dist_filtered = filtered_dist_time[filtered_dist_time['stim_name'] == stim_name][col_name]

                filtered_thig_distances = distance_df.iloc[thig_filtered_ind_lst, distance_column]
                dist_thig = calculate_metric(filtered_thig_distances, "sum")
                total_distance = calculate_metric(stim_dist_filtered, "sum")

                per_thig_dist = [(dist_thig[0] / total_distance[0]) * 100]
                totals.extend(per_thig_dist)

            if specific:
                metric_lst.append(totals)
            else:
                aggregated_metric = sum(totals) / len(totals)
                metric_lst.append([aggregated_metric])
            distance_column += 1
    return fish_ids, metric_lst


def epoch_total_distance_thigmotaxis(df, df2, light, specific):
    global epoch_time_ranges
    global thigmotaxic_distance
    tracking_df = df.filter(regex='^(?!.*average).*dist_from_center|time|stim_name')
    distance_df = df2.filter(regex='distance_traveled|time')
    stim_name = "light_epoch" if light == 1 else "dark_epoch"
    fish_ids = [fish_id for fish_id, column in enumerate(tracking_df) if "dist_from_center" in column]
    metric_lst = []
    distance_column = 1

    for column in tracking_df:
        if "dist_from_center" in column:
            totals = []
            for start, end in epoch_time_ranges[light]:
                filtered_time = tracking_df.loc[tracking_df['time'].between(start, end, inclusive='left')]
                stim_filtered = filtered_time[filtered_time['stim_name'] == stim_name]
                thig_filtered = stim_filtered[stim_filtered[column] > thigmotaxic_distance]["time"]
                thig_filtered_ind_lst = list(thig_filtered.index.values)
                filtered_thig_distances = distance_df.iloc[thig_filtered_ind_lst, distance_column]
                totals.extend(calculate_metric(filtered_thig_distances, "sum"))
            if specific:
                metric_lst.append(totals)
            else:
                aggregated_metric = sum(totals)
                metric_lst.append([aggregated_metric])
            distance_column += 1
    return fish_ids, metric_lst


# def tot_dist_dark_epoch(filename):
#     treatment_dict = {
#         "baseline": [],
#         "drugtreated": [],
#         "24hour_recovery": []
#     }
#     """for each treatment, each DOI doses, and each epoch get the average total distance
#         traveled in dark epochs."""
#     df = pd.read_csv(filename)
#     dark_distance_df = df[(df["metric"].str.startswith("sum_distance_dark_epoch_")) &
#                           (~df["metric"].str.endswith("all"))]
#     dose_lst = dark_distance_df["dose"].unique()
#
#     baseline_df = dark_distance_df[dark_distance_df["baseline"] == 1]
#     drugtreated_df = dark_distance_df[dark_distance_df["drug"] == 1]
#     recovery_df = dark_distance_df[dark_distance_df["recovery"] == 1]
#
#     for dose in dose_lst:
#         baseline_dose_df = baseline_df[baseline_df["dose"] == dose]
#         baseline_epoch1 = baseline_dose_df[baseline_dose_df["metric"].str.endswith("1")]["y"]
#         baseline_epoch2 = baseline_dose_df[baseline_dose_df["metric"].str.endswith("2")]["y"]
#         baseline_epoch3 = baseline_dose_df[baseline_dose_df["metric"].str.endswith("3")]["y"]
#
#         drug_dose_df = drugtreated_df[drugtreated_df["dose"] == dose]
#         drug_epoch1 = drug_dose_df[drug_dose_df["metric"].str.endswith("1")]["y"]
#         drug_epoch2 = drug_dose_df[drug_dose_df["metric"].str.endswith("2")]["y"]
#         drug_epoch3 = drug_dose_df[drug_dose_df["metric"].str.endswith("3")]["y"]
#
#         recovery_dose_df = recovery_df[recovery_df["dose"] == dose]
#         recovery_epoch1 = recovery_dose_df[recovery_dose_df["metric"].str.endswith("1")]["y"]
#         recovery_epoch2 = recovery_dose_df[recovery_dose_df["metric"].str.endswith("2")]["y"]
#         recovery_epoch3 = recovery_dose_df[recovery_dose_df["metric"].str.endswith("3")]["y"]
#
#         treatment_dict["baseline"].append((dose, [calculate_metric(baseline_epoch1, "mean")[0],
#                                                   calculate_metric(baseline_epoch2, "mean")[0],
#                                                   calculate_metric(baseline_epoch3, "mean")[0]]))
#         treatment_dict["drugtreated"].append((dose, [calculate_metric(drug_epoch1, "mean")[0],
#                                                      calculate_metric(drug_epoch2, "mean")[0],
#                                                      calculate_metric(drug_epoch3, "mean")[0]]))
#         treatment_dict["24hour_recovery"].append((dose, [calculate_metric(recovery_epoch1, "mean")[0],
#                                                          calculate_metric(recovery_epoch2, "mean")[0],
#                                                          calculate_metric(recovery_epoch3, "mean")[0]]))
#
#     return treatment_dict


def combine_all_df():
    baseline_df = pd.read_csv("csv_files/baseline/MCAM_fish_metrics.csv")
    drug_df = pd.read_csv("csv_files/drugtreated/MCAM_fish_metrics.csv")
    recovery_df = pd.read_csv("csv_files/recovery/MCAM_fish_metrics.csv")
    all_dfs = [baseline_df, drug_df, recovery_df]
    final_df = pd.concat(all_dfs)
    final_df.to_csv("MCAM_fish_metrics.csv", index=False)
    return final_df



if __name__ == '__main__':
    csv_file_path_lst = ["csv_files/baseline", "csv_files/drugtreated", "csv_files/recovery"]
    for path in csv_file_path_lst:
        run_all_metric_functions(path)
    combine_all_df()
