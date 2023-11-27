"""
Created on 11/26/23
@author: aloyeoshotse
"""

''' 
For MCAM paradigms:
    
      baseline light epochs: 600sec-900sec, 1200sec-1500sec, and 1800sec-2100sec
      baseline dark epochs: 900sec-1200sec, 1500sec-1800sec, and 2100sec-2410sec

      treatment light epochs: 300sec-600sec, 900sec-1200sec, and 1500sec-1800sec
      treatment dark epochs: 600sec-900sec, 1200sec-1500sec, and 1800sec-2110sec
'''

time_ranges = {
        # baseline
        (True, True): [(600, 899.99), (1200, 1499.99), (1800, 2099.99)],
        (False, True): [(900, 1199.99), (1500, 1799.99), (2100, 2410)],
        # treated
        (True, False): [(300, 599.99), (900, 1199.99), (1500, 1799.99)],
        (False, False): [(600, 899.99), (1200, 1499.99), (1800, 2110)],
    }

def total_epoch_distance(df, light, baseline):
    """df is the datafram with the MCAM data
        light is a boolean deciding whether to look at light epoch or dark epoch
        baseline is a boolean deciding whether to look at baseline or treated
    """
    global time_ranges

    # Filter columns where 'time' is between specified ranges
    total_distance = sum(
        df[(df['time'].between(start, end, inclusive=True))]
        ['tracking_metrics']['distance_traveled'].sum()
        for start, end in time_ranges[(light, baseline)]
        if 'tracking_metrics' in df.columns
    )

    return total_distance

