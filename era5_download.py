import os
import pandas as pd
from pycontrails import Flight, MetDataset
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.datalib import ecmwf
from datetime import datetime, timedelta
# Path to the specific flight directory
# lhr_ist_dir = "flight_trajectories/processed_flights/lhr_ist"
directories = ["flight_trajectories/processed_flights/sin_maa"]

# Helper function to parse and adjust time bounds
def calculate_time_bounds(flight_data):
    """
    Calculate time bounds for meteorological data:
    - From the start time of the flight
    - To the end time of the flight + 12 hours
    """
    # Extract flight times
    start_time = flight_data["time"].min()
    end_time = flight_data["time"].max()


    # Adjust bounds
    adjusted_start = start_time
    adjusted_end = end_time + timedelta(hours=12)

    return adjusted_start.strftime("%Y-%m-%d %H:%M"), adjusted_end.strftime("%Y-%m-%d %H:%M")

for flight_dir in directories:
    print(f"Processing directory: {flight_dir}")
    # Loop through files in the lhr_ist directory
    for file in os.listdir(flight_dir):
        if file.endswith(".csv"):
            # Extract the full path to the flight file
            flight_path = os.path.join(flight_dir, file)

            # Extract flight identifier (e.g., "lhr_ist_2023-02-06_daytime")
            flight_identifier = os.path.splitext(file)[0]

            print(f"Processing flight: {flight_identifier}")

            """------READ FLIGHT CSV AND PREPARE FORMAT---------------------------------------"""
            df = pd.read_csv(flight_path)  # Read the flight data

            # Ensure the time column is parsed as datetime
            df["time"] = pd.to_datetime(df["time"])

            fl = Flight(df)  # Initialize the flight object

            """------CALCULATE TIME BOUNDS FOR METEOROLOGIC DATA-----------------------------"""
            time_bounds = calculate_time_bounds(df)  # Dynamically calculate time bounds
            print(f"Time bounds for {flight_identifier}: {time_bounds}")

            """------RETRIEVE METEOROLOGIC DATA----------------------------------------------"""
            pressure_levels = (
            1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175)  # hPa

            era5pl = ERA5(
                time=time_bounds,
                variables=Cocip.met_variables + Cocip.optional_met_variables + (ecmwf.PotentialVorticity,) + (
                ecmwf.RelativeHumidity,),
                pressure_levels=pressure_levels,
            )
            era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,))

            # Download data from ERA5 (or open from cache)
            met = era5pl.open_metdataset()  # Meteorology
            rad = era5sl.open_metdataset()  # Radiation

            print(f"Finished processing flight: {flight_identifier}")