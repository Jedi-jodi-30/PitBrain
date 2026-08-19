from basic_imports import *
import pandas as pd

session = get_session(2023, 'Monza', 'R',check_cache=True)
clean_laps = get_clean_laps(session)

#find median race pace per driver
median_pace_drivers = clean_laps.groupby('Driver')['LapTimeSeconds'].median().sort_values()
median_pace_drivers_df = pd.DataFrame(median_pace_drivers).reset_index()

#find the fastest lap of the race
fastest_lap = clean_laps.loc[clean_laps['LapTimeSeconds'].idxmin()]
fastest_lap_time = fastest_lap['LapTimeSeconds']
fastest_lap_driver = fastest_lap['Driver']
fastest_lap_compound = fastest_lap['Compound']
fastest_lap_lap_number = fastest_lap['LapNumber']
print(f"The fastest lap of the race was set by {fastest_lap['Driver']} with a time of {fastest_lap['LapTimeSeconds']:.3f} seconds.")

#find the fastest lap of the race for each driver
fastest_laps_per_driver = clean_laps.loc[clean_laps.groupby('Driver')['LapTimeSeconds'].idxmin()]
fastest_laps_per_driver_df = pd.DataFrame(fastest_laps_per_driver).reset_index()

#find mean lap time for each driver
mean_lap_time_per_driver = clean_laps.groupby('Driver')['LapTimeSeconds'].mean().sort_values()
mean_lap_time_per_driver_df = pd.DataFrame(mean_lap_time_per_driver).reset_index()

#pace delta to the field's fastest median pace, for each driver
field_min_median_pace = median_pace_drivers_df['LapTimeSeconds'].min()
pace_delta_to_fastest_lap_df = median_pace_drivers_df.copy()
pace_delta_to_fastest_lap_df['PaceDelta'] = pace_delta_to_fastest_lap_df['LapTimeSeconds'] - field_min_median_pace

#rolling pace (smoothed) per driver, window of 3 laps
rolling_window = 3
rolling_pace = clean_laps.sort_values(['Driver', 'LapNumber']).copy()
rolling_pace['RollingPace'] = rolling_pace.groupby('Driver')['LapTimeSeconds'].transform(
    lambda x: x.rolling(rolling_window, min_periods=1).mean()
)
rolling_pace_df = pd.DataFrame(rolling_pace).reset_index()

