import os
import pandas as pd
from pycontrails import Flight, MetDataset
from pycontrails.datalib.ecmwf import ERA5, ERA5ModelLevel
from pycontrails.core.cache import DiskCacheStore
from pathlib import Path
import numpy as np
from pycontrails.models.cocip import Cocip
from pycontrails.datalib import ecmwf
from datetime import datetime, timedelta
# # Path to the specific flight directory
# # lhr_ist_dir = "flight_trajectories/processed_flights/lhr_ist"
# directories = ["flight_trajectories/processed_flights/sin_maa"]
#
# # Helper function to parse and adjust time bounds
# def calculate_time_bounds(flight_data):
#     """
#     Calculate time bounds for meteorological data:
#     - From the start time of the flight
#     - To the end time of the flight + 12 hours
#     """
#     # Extract flight times
#     start_time = flight_data["time"].min()
#     end_time = flight_data["time"].max()
#
#
#     # Adjust bounds
#     adjusted_start = start_time
#     adjusted_end = end_time + timedelta(hours=12)
#
#     return adjusted_start.strftime("%Y-%m-%d %H:%M"), adjusted_end.strftime("%Y-%m-%d %H:%M")
#
# for flight_dir in directories:
#     print(f"Processing directory: {flight_dir}")
#     # Loop through files in the lhr_ist directory
#     for file in os.listdir(flight_dir):
#         if file.endswith(".csv"):
#             # Extract the full path to the flight file
#             flight_path = os.path.join(flight_dir, file)
#
#             # Extract flight identifier (e.g., "lhr_ist_2023-02-06_daytime")
#             flight_identifier = os.path.splitext(file)[0]
#
#             print(f"Processing flight: {flight_identifier}")
#
#             """------READ FLIGHT CSV AND PREPARE FORMAT---------------------------------------"""
#             df = pd.read_csv(flight_path)  # Read the flight data
#
#             # Ensure the time column is parsed as datetime
#             df["time"] = pd.to_datetime(df["time"])
#
#             fl = Flight(df)  # Initialize the flight object
#
#             """------CALCULATE TIME BOUNDS FOR METEOROLOGIC DATA-----------------------------"""
#             time_bounds = calculate_time_bounds(df)  # Dynamically calculate time bounds
#             print(f"Time bounds for {flight_identifier}: {time_bounds}")
#
#             """------RETRIEVE METEOROLOGIC DATA----------------------------------------------"""
#             pressure_levels = (
#             1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175)  # hPa
#
#             era5pl = ERA5(
#                 time=time_bounds,
#                 variables=Cocip.met_variables + Cocip.optional_met_variables + (ecmwf.PotentialVorticity,) + (
#                 ecmwf.RelativeHumidity,),
#                 pressure_levels=pressure_levels,
#             )
#             era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,))
#
#             # Download data from ERA5 (or open from cache)
#             met = era5pl.open_metdataset()  # Meteorology
#             rad = era5sl.open_metdataset()  # Radiation
#
#             print(f"Finished processing flight: {flight_identifier}")

# time_bounds = ("2023-02-05 14:00", "2023-02-07 11:00")
# time_bounds = ("2023-05-04 14:00", "2023-05-06 11:00")
# time_bounds = ("2023-08-05 14:00", "2023-08-07 11:00")
# time_bounds = ("2023-11-05 14:00", "2023-08-07 11:00")

"""FLIGHT PARAMETERS"""
engine_model = 'GTF'        # GTF , GTF2035, GTF1990, GTF2000
water_injection = [0, 0, 0]     # WAR climb cruise approach/descent
SAF = 0                   # 0, 20, 100 unit = %
flight = 'malaga'
aircraft = 'A20N_full'        # A20N ps model, A20N_wf is change in Thrust and t/o and idle fuel flows
                            # A20N_wf_opr is with changed nominal opr and bpr
                            # A20N_full has also the eta 1 and 2 and psi_0

"""------READ FLIGHT CSV AND PREPARE FORMAT---------------------------------------"""
df = pd.read_csv(f"{flight}_flight.csv")
df = df.rename(columns={'geoaltitude': 'altitude', 'groundspeed': 'groundspeed', 'timestamp':'time'})
df = df.dropna(subset=['callsign'])
df = df.dropna(subset=['altitude'])
df = df.drop(['Unnamed: 0', 'icao24', 'callsign'], axis=1)

# df = df[df['altitude'] > 1900]
column_order = ['longitude', 'latitude', 'altitude', 'groundspeed', 'time']
df = df[column_order]
df['altitude'] = df['altitude']*0.3048 #foot to meters
df['groundspeed'] = df['groundspeed']*0.514444444

if engine_model == 'GTF' or engine_model == 'GTF2035':
    engine_uid = '01P22PW163'
elif engine_model == 'GTF1990':
    engine_uid = '1CM009'
elif engine_model == 'GTF2000':
    engine_uid = '3CM026'
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}. ")

attrs = {
    "flight_id" : "34610D",
    "aircraft_type": f"{aircraft}",
    "engine_uid": f"{engine_uid}"
}

fl = Flight(df, attrs=attrs)

"""------ERA5model--------------------------------"""
time_bounds_list = [
    ("2023-02-05 14:00", "2023-02-07 11:00"),
    ("2023-05-04 14:00", "2023-05-06 11:00"),
    ("2023-08-05 14:00", "2023-08-07 11:00"),
    ("2023-11-05 14:00", "2023-11-07 11:00")  # Fixed the incorrect end date
]

time_step = timedelta(hours=6)  # Subdivide into 6-hour chunks

pressure_levels_10 = np.arange(150, 400, 10)  # 150 to 400 with steps of 10
pressure_levels_50 = np.arange(400, 1001, 50)  # 400 to 1000 with steps of 50
pressure_levels_model = np.concatenate((pressure_levels_10, pressure_levels_50))

local_cache_dir = Path("F:/era5model/flights")
local_cachestore = DiskCacheStore(cache_dir=local_cache_dir)

for start_time_str, end_time_str in time_bounds_list:
    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M")

    current_time = start_time
    while current_time < end_time:
        next_time = min(current_time + time_step, end_time)
        time_bounds = (current_time.strftime("%Y-%m-%d %H:%M"), next_time.strftime("%Y-%m-%d %H:%M"))
        print(f"Processing time bounds: {time_bounds}")

        try:
            era5ml = ERA5ModelLevel(
                time=time_bounds,
                variables=("t", "q", "u", "v", "w", "ciwc"),
                model_levels=range(67, 133),
                pressure_levels=pressure_levels_model,
                cachestore=local_cachestore
            )

            met = era5ml.open_metdataset()
            print(f"Successfully processed {time_bounds}")

        except Exception as e:
            print(f"Error processing {time_bounds}: {e}")

        # Move to the next time chunk
        current_time = next_time

print("All processing complete.")

"""ERA5"""

# time_bounds_list = [
#     ("2023-02-05 14:00", "2023-02-07 11:00"),
#     ("2023-05-04 14:00", "2023-05-06 11:00"),
#     ("2023-08-05 14:00", "2023-08-07 11:00"),
#     ("2023-11-05 14:00", "2023-11-07 11:00")  # Fixed the incorrect end date
# ]
#
# time_step = timedelta(hours=6)  # Subdivide into 6-hour chunks
#
# pressure_levels = (
#             1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175)  # hPa
#
# local_cache_dir = Path("F:/era5pressure/Cache")
# local_cachestore = DiskCacheStore(cache_dir=local_cache_dir)
#
# for start_time_str, end_time_str in time_bounds_list:
#     start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M")
#     end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M")
#
#     current_time = start_time
#     while current_time < end_time:
#         next_time = min(current_time + time_step, end_time)
#         time_bounds = (current_time.strftime("%Y-%m-%d %H:%M"), next_time.strftime("%Y-%m-%d %H:%M"))
#         print(f"Processing time bounds: {time_bounds}")
#
#         try:
#             era5pl = ERA5(
#                             time=time_bounds,
#                             variables=Cocip.met_variables + Cocip.optional_met_variables + (ecmwf.PotentialVorticity,) + (
#                             ecmwf.RelativeHumidity,),
#                             pressure_levels=pressure_levels,
#                             cachestore=local_cachestore
#                         )
#             era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,), cachestore=local_cachestore)
#
#             # Download data from ERA5 (or open from cache)
#             met = era5pl.open_metdataset()  # Meteorology
#             rad = era5sl.open_metdataset()  # Radiation
#
#         except Exception as e:
#             print(f"Error processing {time_bounds}: {e}")
#
#         # Move to the next time chunk
#         current_time = next_time
#
# print("All processing complete.")

