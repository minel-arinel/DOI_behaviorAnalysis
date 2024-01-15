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
TODO LATER:
    - Make % thigomotaxis time in outer circle function --> 
        - specific --> time in thigomotaxis (epoch_time_of_thigmotaxis) / total epoch duration (before startle --> 
                end of epoch - start of epoch) * 100
        - not specific --> average the specific ones 
    - Make % thigmo distance in outer circle -->
        - specific --> (total thig distance in that epoch / total distance in that epoch)  * 100
    - Add specific feature to startle_response_distance and startle_response_speed
    - automate making the csv file --> all of the possibilities should be ran in this method
        - figure out how to decide between photomotor, startle, and thigmotaxis
        - add a feature that determines fish id by column or row
    - Make a note that the mean is distance / frame (average per frame)
    - Change update_data_file to be able to override rows (do not check if the y-value is the same)
        - change the fish_id when the dose is changed, but keep consistent when treatment is different (baseline vs 
            drugtreated vs 24 hr) --> can make a dataframe keeping track of this or a dictionary 
    
    
    24 HR RECOVERY
        - Wells are A1-A6...D1-D6
            - each column gets a different concentration 
            - first experiement had no fish removed
        
        
    """
"""
Notes
    1) 
        have a dictionary for the final df we want to make: --> make this using parallel lists that much up the fishes with their data
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

data_dict = {
    "fish": 0,
    "metric": "",
    "y": 0.0,
    "baseline": 0,
    "recovery": 0,
    "drug": 0,
    "dose": 0.0
}

thigmotaxic_distance = 0.0053

csv_file_path = "/Users/aloyeoshotse/AJO/NAUMANN_LAB/DOI_behaviorAnalysis/csv_files"


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
    file_name = file_name.split("/")[-1]
    info = [("metric", file_name.split("_")[1]), ("baseline", 1 if "baseline" in file_name else 0),
            ("recovery", 1 if "recovery" in file_name else 0), ("dose", file_name.split("_")[-1][:-4])]
    return info


def retrieve_distance_csv(filename):
    filename = filename.split("/")[-1]
    lst = filename.split("_")
    lst[1] = "distance"
    distance_file_name = os.path.join(csv_file_path, "_".join(lst))
    return pd.read_csv(distance_file_name)


def populate_df_dictionary(metadata, metric_data, specific):
    global data_dict

    for col_name, value in metadata:
        data_dict[col_name] = value

    if specific:
        metric_name = metric_data[0].split(";")
    else:
        metric_name = [metric_data[0]]
    values = metric_data[1]
    for fish_ids, fish_metrics in values:
        for x in range(len(fish_ids)):
            for y in range(len(metric_name)):
                data_dict['metric'] = metric_name[y]
                data_dict["fish"] = fish_ids[x]
                data_dict["y"] = round(fish_metrics[x][y], 8)
                data_dict["drug"] = 0 if data_dict["baseline"] == 1 or data_dict["recovery"] == 1 or data_dict[
                    "dose"] == 0.0 else 1
                update_data_file(data_dict, "/Users/aloyeoshotse/AJO/NAUMANN_LAB/DOI_behaviorAnalysis",
                                 "MCAM_fish_metrics.csv")


def update_data_file(dictionary, directory, file_name):
    file_path = os.path.join(directory, file_name)
    file_exists = os.path.isfile(file_path)
    df = pd.DataFrame([dictionary])
    df['dose'] = df['dose'].astype(float)

    if file_exists:
        existing_df = pd.read_csv(file_path)

        for index, row in existing_df.iterrows():
            duplicate_data = []
            single_row = pd.DataFrame([existing_df.iloc[index]])
            for col in single_row.columns:
                if df[col].iloc[0] == single_row[col].iloc[0]:
                    duplicate_data.append(True)
                else:
                    duplicate_data.append(False)
            if all(duplicate_data):
                print("The data in the following row already exists in the DataFrame.")
                print(f"{df}\n")
                return existing_df

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
    startle_response_name = f"{metric}_{tracking}_{startle_response}_startle"
    diff_first_last_rep_name = ""
    average_latency_name = ""
    if specific:
        name_diff = f"diff_{metric}_{tracking}_{startle_response}_startle_rep_"
        name_avg_lat = f"avg_latency_{metric}_{tracking}_{startle_response}_startle_rep_"
        for x in range(1, 3):
            diff_first_last_rep_name += name_diff + str(x) + ";" if x != 2 else name_diff + str(x)
            average_latency_name += name_avg_lat + str(x) + ";" if x != 2 else name_avg_lat + str(x)
    else:
        diff_first_last_rep_name = f"diff_{metric}_{tracking}_{startle_response}_startle_all"
        average_latency_name = f"avg_latency_{metric}_{tracking}_{startle_response}_startle_all"

    if tracking == "distance":
        metric_name_dict[startle_response_name] = [startle_response_distance(df, startle_response, metric)]
        metric_name_dict[diff_first_last_rep_name] = [diff_first_last_startle_rep_distance(df, startle_response, metric,
                                                                                           specific)]
        metric_name_dict[average_latency_name] = [average_latency_startle_distance(df, startle_response, specific)]
        return metric_name_dict.items()
    elif tracking == "speed":
        metric_name_dict[startle_response_name] = [startle_response_speed(df, startle_response, metric)]
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
    if specific:
        name_mean_dist = f"thigmotaxis_mean_distance_{light_name}_epoch_"
        name_time = f"thigmotaxis_total_time_{light_name}_epoch_"
        name_total_dist = f"thigmotaxis_total_distance_{light_name}_epoch_"
        for x in range(1, 4):
            mean_dist_name += name_mean_dist + str(x) + ";" if x != 3 else name_mean_dist + str(x)
            thig_time_name += name_time + str(x) + ";" if x != 3 else name_time + str(x)
            total_dist_name += name_total_dist + str(x) + ";" if x != 3 else name_total_dist + str(x)
    else:
        mean_dist_name = f"thigmotaxis_mean_distance_{light_name}_epoch_all"
        thig_time_name = f"thigmotaxis_total_time_{light_name}_epoch_all"
        total_dist_name = f"thigmotaxis_total_distance_{light_name}_epoch_all"

    metric_name_dict[mean_dist_name] = [epoch_mean_dist_thigmotaxis(df, light, specific)]
    metric_name_dict[thig_time_name] = [epoch_time_of_thigmotaxis(df, light, specific)]
    metric_name_dict[total_dist_name] = [epoch_total_distance_thigmotaxis(df, df2, light, specific)]
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


def startle_response_distance(df, startle_response, metric):
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
            metric_lst.append(
                [max(totals)] if metric == "max" else [sum(totals)] if metric == "sum" else [sum(totals) / len(totals)])
    return fish_ids, metric_lst


def startle_response_speed(df, startle_response, metric):
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
            metric_lst.append([max(totals)] if metric == "max" else [sum(totals) / len(totals)])
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


if __name__ == '__main__':
    # process_file("baseline_distance_0.05.csv", "photomotor", "dark", "sum", True)
    # process_file("baseline_speed_2.5.csv", "photomotor", "dark", "max", True)
    process_file("baseline_speed_0.05.csv", "startle", "dark", "mean")
    # process_file("baseline_distance_0.05.csv", "startle", "vibration", "sum")
    # process_file("baseline_tracking_50.csv", "thigmotaxis", "light", "mean", True)
