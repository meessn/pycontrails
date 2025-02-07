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
flight = 'bos_fll'
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



# Convert time to datetime format and sort
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values(by="time").reset_index(drop=True)

# Identify missing data
df["missing"] = df["latitude"].isna()

# Find start and end of missing segments
gap_starts = df.index[df["missing"] & ~df["missing"].shift(1, fill_value=False)]
gap_ends = df.index[df["missing"] & ~df["missing"].shift(-1, fill_value=False)]

# Ensure valid pairs
if len(gap_starts) > len(gap_ends):
    gap_starts = gap_starts[:-1]

df_filled = df.copy()

# Interpolating missing flight path using velocity-based approach
for start, end in zip(gap_starts, gap_ends):
    prev_point = df.iloc[start - 1]
    next_point = df.iloc[end + 1]

    prev_time = prev_point["time"]
    next_time = next_point["time"]

    prev_lat, prev_lon, prev_alt = prev_point["latitude"], prev_point["longitude"], prev_point["altitude"]
    next_lat, next_lon, next_alt = next_point["latitude"], next_point["longitude"], next_point["altitude"]

    time_delta = (next_time - prev_time).total_seconds()

    if time_delta > 0:
        # Compute 3D distance
        horizontal_distance = geodesic((prev_lat, prev_lon), (next_lat, next_lon)).meters
        vertical_distance = abs(next_alt - prev_alt) if not pd.isna(next_alt) else 0
        total_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)

        # Compute velocity (m/s)
        velocity = total_distance / time_delta

        # Generate missing times
        missing_times = pd.date_range(start=prev_time, end=next_time, freq="60S")[1:-1]

        # Compute linear interpolation steps
        lat_step = (next_lat - prev_lat) / time_delta
        lon_step = (next_lon - prev_lon) / time_delta
        alt_step = (next_alt - prev_alt) / time_delta if not pd.isna(prev_alt) else 0

        # Fill missing values
        for i, t in enumerate(missing_times, start=1):
            df_filled.loc[len(df_filled)] = {
                "time": t,
                "icao24": prev_point["icao24"],
                "callsign": prev_point["callsign"],
                "latitude": prev_lat + lat_step * (i * 60),
                "longitude": prev_lon + lon_step * (i * 60),
                "altitude": prev_alt + alt_step * (i * 60) if not pd.isna(prev_alt) else None,
                "groundspeed": None,
                "missing": False
            }

# Sort by time again
df_filled = df_filled.sort_values(by="time").reset_index(drop=True)

# Final interpolation for any remaining missing values
df_filled["latitude"] = df_filled["latitude"].interpolate(method="linear")
df_filled["longitude"] = df_filled["longitude"].interpolate(method="linear")
df_filled["altitude"] = df_filled["altitude"].interpolate(method="linear")

# Compute 3D segment length
segment_lengths = []
times = []

for i in range(1, len(df_filled)):
    p1 = df_filled.iloc[i - 1]
    p2 = df_filled.iloc[i]

    horizontal_distance = geodesic((p1["latitude"], p1["longitude"]), (p2["latitude"], p2["longitude"])).meters
    vertical_distance = abs(p2["altitude"] - p1["altitude"]) if not pd.isna(p1["altitude"]) and not pd.isna(p2["altitude"]) else 0
    total_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)

    segment_lengths.append(total_distance)
    times.append(p2["time"])

# Plot corrected flight path
plt.figure(figsize=(12, 6))
plt.plot(df["time"], df["latitude"], 'o', markersize=2, alpha=0.5, label="Original Data")
plt.plot(df_filled["time"], df_filled["latitude"], '-', markersize=2, alpha=0.7, label="Final Interpolation Fix")
plt.legend()
plt.title("Corrected Flight Path After Velocity-Based Interpolation")
plt.xlabel("Timestamp")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# Plot 3D segment length
plt.figure(figsize=(12, 6))
plt.plot(times, segment_lengths, marker="o", linestyle="-", alpha=0.7, label="Segment Length (3D)")
plt.axhline(y=np.mean(segment_lengths), color='r', linestyle="--", label=f"Mean Segment Length: {np.mean(segment_lengths):.2f} m")
plt.legend()
plt.title("Segment Length Over Time (3D)")
plt.xlabel("Timestamp")
plt.ylabel("Segment Length (meters)")
plt.grid(True)
plt.show()
# df= df.dropna(subset=['latitude', 'longitude', 'altitude'])
# fl = Flight(df, attrs=attrs)
#
# fl.plot_profile(kind="scatter", s=5, figsize=(10, 6))
#
#
# fl.plot(kind="scatter", s=5, figsize=(10, 6))
#
# fl_segment = fl.segment_length()
# plt.figure()
# plt.plot(fl.dataframe['time'], fl_segment, marker='o', linestyle='-', label='Segment Length')
# plt.title('Segment Length Before Resample')
# plt.xlabel('Time in Minutes')
# plt.ylabel('Segment Length (m)')
# # plt.legend()
# plt.grid(True)
# # plt.show()
#
#
#
#
# fl = fl.resample_and_fill(freq="60s", drop=False, fill_method='geodesic', geodesic_threshold=1e3)
#
# fl.plot_profile(kind="scatter", s=5, figsize=(10, 6))
#
# fl.plot(kind="scatter", s=5, figsize=(10, 6))
#
# print('flight length', fl.length)
# # print('segment_lengths', fl.segment_length()[:-1].max())
# fl_segment = fl.segment_length()
# plt.figure()
# plt.plot(fl.dataframe['time'], fl_segment,  marker='o', linestyle='-', label='Segment Length')
# plt.title('segment length after resample')
# plt.xlabel('Time in minutes')
# plt.ylabel('segment length m')
# plt.legend()
# plt.grid(True)
# plt.show()


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