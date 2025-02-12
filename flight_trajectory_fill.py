import pandas as pd
from pycontrails import Flight, MetDataset
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import constants
from matplotlib import pyplot as plt
from pycontrails.datalib.ecmwf import ERA5, ERA5ModelLevel
from emission_index import p3t3_nox
from emission_index import p3t3_nvpm_meem, p3t3_nvpm_meem_mass, thrust_setting
import pickle
from pycontrails import Flight
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import HistogramMatching, ExponentialBoostHumidityScaling
from pycontrails.models.ps_model import PSFlight
from pycontrails.models.emissions import Emissions
from pycontrails.datalib import ecmwf
from pycontrails.core.cache import DiskCacheStore
from pathlib import Path
import copy

import warnings
flight = 'sin_maa'
aircraft = 'A20N_full'

df = pd.read_csv(f"flight_trajectories/{flight}.csv")
df = df.rename(columns={'geoaltitude': 'altitude', 'groundspeed': 'groundspeed', 'timestamp':'time'})
callsign = df['callsign'].dropna().unique()[0]
df['altitude'] = df['altitude']*0.3048 #foot to meters
df['groundspeed'] = df['groundspeed']*0.514444444
attrs = {
    "flight_id" : f"{callsign}",
    "aircraft_type": f"{aircraft}",
    "engine_uid": "01P22PW163"
}


from geopy.distance import geodesic



def load_and_preprocess(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Convert timestamp to datetime format
    df["time"] = pd.to_datetime(df["timestamp"])

    # Convert geoaltitude from feet to meters
    df["geoaltitude"] = df["geoaltitude"] * 0.3048

    return df


def remove_static_rows(df):
    # Identify and remove rows where latitude and longitude remain the same as the previous row
    df["prev_latitude"] = df["latitude"].shift(1)
    df["prev_longitude"] = df["longitude"].shift(1)
    df = df[~((df["latitude"] == df["prev_latitude"]) & (df["longitude"] == df["prev_longitude"]))]
    df = df.drop(columns=["prev_latitude", "prev_longitude"]).reset_index(drop=True)
    return df


def create_full_timeline(df):
    # Create a full second-by-second timeline
    start_time = df["time"].min()
    end_time = df["time"].max()
    full_time_range = pd.date_range(start=start_time, end=end_time, freq="1S")
    return pd.DataFrame({"time": full_time_range})


def merge_and_interpolate(df_full_time, df_cleaned):
    # Merge flight data into the full timeline
    df_merged = df_full_time.merge(df_cleaned, on="time", how="left")

    # Interpolate missing values for latitude, longitude, and geoaltitude
    df_merged["latitude"] = df_merged["latitude"].interpolate(method="linear")
    df_merged["longitude"] = df_merged["longitude"].interpolate(method="linear")
    df_merged["geoaltitude"] = df_merged["geoaltitude"].interpolate(method="linear")

    return df_merged


def remove_outliers(df, column, threshold):
    df = df.copy()
    df["diff"] = df[column].diff().abs()
    df.loc[df["diff"] > threshold, column] = np.nan
    df[column] = df[column].interpolate(method="linear")
    df = df.drop(columns=["diff"])
    return df


def compute_segment_lengths(df):
    segment_lengths = []
    timestamps = []
    for i in range(1, len(df)):
        p1 = df.iloc[i - 1]
        p2 = df.iloc[i]
        horizontal_distance = geodesic((p1["latitude"], p1["longitude"]), (p2["latitude"], p2["longitude"])).meters
        vertical_distance = abs(p2["geoaltitude"] - p1["geoaltitude"]) if not pd.isna(
            p1["geoaltitude"]) and not pd.isna(p2["geoaltitude"]) else 0
        total_distance = np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)
        segment_lengths.append(total_distance)
        timestamps.append(p2["time"])
    return timestamps, segment_lengths


def resample_to_60s(df):
    return df.set_index("time").resample("60S").first().reset_index()


def plot_data(df, timestamps, segment_lengths):
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["geoaltitude"], marker="o", linestyle="-", alpha=0.7,
             label="Altitude Over Time (60s Resampled)")
    plt.title("Altitude vs. Time After 60s Resampling")
    plt.xlabel("Time")
    plt.ylabel("Altitude (meters)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    # plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, segment_lengths, marker="o", linestyle="-", alpha=0.7,
             label="Segment Length (3D) - 60s Resampled")
    plt.axhline(y=np.mean(segment_lengths), color='r', linestyle="--",
                label=f"Mean Segment Length: {np.mean(segment_lengths):.2f} m")
    plt.legend()
    plt.title("Segment Length Over Time After 60s Resampling")
    plt.xlabel("Timestamp")
    plt.ylabel("Segment Length (meters)")
    plt.grid(True)
    plt.xticks(rotation=45)
    # plt.show()

def process_flight(trajectory, flight_results):
    # Construct file name from trajectory details
    file_name = f"{trajectory['departure_airport'].lower()}_{trajectory['arrival_airport'].lower()}.csv"

# Read and preprocess the flight data
df = pd.read_csv(f"flight_trajectories/{flight}.csv")
df = df.rename(columns={'groundspeed': 'groundspeed', 'timestamp': 'time'})
df['geoaltitude'] = df['geoaltitude'] * 0.3048  # foot to meters
df['groundspeed'] = df['groundspeed'] * 0.514444444
df['time'] = pd.to_datetime(df['time'])
df = df.dropna(subset=['latitude', 'longitude', 'geoaltitude'])
df_pycontrails = df.copy()
df_pycontrails['altitude'] = df_pycontrails['geoaltitude']

df_cleaned = remove_static_rows(df)
df_full_time = create_full_timeline(df_cleaned)
df_interpolated = merge_and_interpolate(df_full_time, df_cleaned)
df_interpolated = remove_outliers(df_interpolated, "geoaltitude", 200)
df_interpolated = remove_outliers(df_interpolated, "latitude", 0.01)
df_interpolated = remove_outliers(df_interpolated, "longitude", 0.01)
df_resampled_60 = resample_to_60s(df_interpolated)
timestamps, segment_lengths = compute_segment_lengths(df_resampled_60)
plot_data(df_resampled_60, timestamps, segment_lengths)


# df= df.dropna(subset=['latitude', 'longitude', 'altitude'])
fl = Flight(df_pycontrails, attrs=attrs)

fl.plot_profile(kind="scatter", s=5, figsize=(10, 6))


fl.plot(kind="scatter", s=5, figsize=(10, 6))

fl_segment = fl.segment_length()
plt.figure()
plt.plot(fl.dataframe['time'], fl_segment, marker='o', linestyle='-', label='Segment Length')
plt.title('Segment Length Before Resample')
plt.xlabel('Time in Minutes')
plt.ylabel('Segment Length (m)')
# plt.legend()
plt.grid(True)
# plt.show()

# fl = Flight(df_pycontrails, attrs=attrs)
df_cleaned_p = remove_static_rows(df_pycontrails)
# df_full_time_p = create_full_timeline(df_cleaned_p)
# df_interpolated_p = merge_and_interpolate(df_full_time_p, df_cleaned_p)

fl = Flight(df_cleaned_p, attrs=attrs)
fl = fl.resample_and_fill(freq="60s", drop=False, fill_method='geodesic', geodesic_threshold=1e3)

fl.plot_profile(kind="scatter", s=5, figsize=(10, 6))

fl.plot(kind="scatter", s=5, figsize=(10, 6))

print('flight length', fl.length)
# print('segment_lengths', fl.segment_length()[:-1].max())
fl_segment = fl.segment_length()
plt.figure()
plt.plot(fl.dataframe['time'], fl_segment,  marker='o', linestyle='-', label='Segment Length')
plt.title('segment length after resample')
plt.xlabel('Time in minutes')
plt.ylabel('segment length m')
plt.legend()
plt.grid(True)
plt.show()


# """------RETRIEVE METEOROLOGIC DATA----------------------------------------------"""
#
# # time_bounds = ("2024-06-07 9:00", "2024-06-08 02:00")
#
# pressure_levels_10 = np.arange(150, 400, 10)  # 150 to 400 with steps of 10
# pressure_levels_50 = np.arange(400, 1001, 50)  # 400 to 1000 with steps of 50
# pressure_levels_model = np.concatenate((pressure_levels_10, pressure_levels_50))
#
# if flight == 'malaga':
#     local_cache_dir = Path("F:/era5model/malaga")
#     variables_model = ("t", "q", "u", "v", "w", "ciwc", "vo", "clwc")
# else:
#     local_cache_dir = Path("F:/era5model/flights")
#     variables_model = ("t", "q", "u", "v", "w", "ciwc")
#
# local_cachestore = DiskCacheStore(cache_dir=local_cache_dir)
#
# era5ml = ERA5ModelLevel(
#                 time=time_bounds,
#                 variables=variables_model,
#                 model_levels=range(67, 133),
#                 pressure_levels=pressure_levels_model,
#                 cachestore=local_cachestore
#             )
# # era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,))
#
# # download data from ERA5 (or open from cache)
# met = era5ml.open_metdataset()
# # Extract min/max longitude and latitude from the dataframe
# west = fl.dataframe["longitude"].min() - 50  # Subtract 1 degree for west buffer
# east = fl.dataframe["longitude"].max() + 50 # Add 1 degree for east buffer
# south = fl.dataframe["latitude"].min() - 50  # Subtract 1 degree for south buffer
# north = fl.dataframe["latitude"].max() + 50  # Add 1 degree for north buffer
#
# # Define the bounding box with altitude range
# bbox = (west, south, 150, east, north, 1000)  # (west, south, min-level, east, north, max-level)
# met = met.downselect(bbox=bbox)
# met_ps = copy.deepcopy(met)#era5ml.open_metdataset() # meteorology
# met_emi = copy.deepcopy(met)
# # rad = era5sl.open_metdataset() # radiation
#
#
# # fl_test = copy.deepcopy(fl)
# # print(fl_test.intersect_met(met_ps['specific_humidity']))
# """-----RUN AIRCRAFT PERFORMANCE MODEL--------------------------------------------"""
#
# perf = PSFlight(
#     met=met_ps,
#     fill_low_altitude_with_isa_temperature=True,  # Estimate temperature using ISA
#     fill_low_altitude_with_zero_wind=True
# )
# fp = perf.eval(fl)
# df_p = fp.dataframe
# df_p.update(df_p.select_dtypes(include=[np.number]).interpolate(method='linear', limit_area='inside'))
# fp = Flight(df_p, attrs=attrs)