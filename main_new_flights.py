from main_emissions import run_emissions
from main_climate import run_climate
from main_emissions_verification import run_emissions_verification
import os

# Updated root directory for new trajectories
root_dir = "flight_trajectories/extra_trajectories_yin_2/processed_flights"

# Set fixed simulation configuration
engine_models = {
    "GTF": True,  # Only this one
    "GTF1990": False,
    "GTF2000": False,
    "GTF2035": False,
    "GTF2035_wi": False
}

saf_dict = {"SAF20": False, "SAF100": False}
prediction = "mees"
weather_model = "era5model"
accuracy = None

# Use only daytime Feb 6 flight
target_date = "2023-02-06"
target_period = "daytime"
time_bounds = ("2023-02-05 14:00", "2023-02-07 11:00")

# Collect all folders from new data
for trajectory in os.listdir(root_dir):
    trajectory_path = os.path.join(root_dir, trajectory)
    if not os.path.isdir(trajectory_path):
        continue

    flight_files = [f for f in os.listdir(trajectory_path) if f.endswith(".csv")]

    for flight_file in flight_files:
        for engine_model, do_run in engine_models.items():
            if not do_run:
                continue

            water_injection = [0, 0, 0]
            SAF = 0

            flight_path = os.path.join(trajectory_path, flight_file)
            print(f"Running emissions for {flight_file}, Engine: {engine_model}")
            run_emissions(
                trajectory=trajectory,
                flight_path=flight_path,
                engine_model=engine_model,
                water_injection=water_injection,
                SAF=SAF,
                aircraft="A20N_full",
                time_bounds=time_bounds
            )

            # print(f"Running climate model for {flight_file}, Engine: {engine_model}")
            # run_climate(
            #     trajectory=trajectory,
            #     flight_path=flight_path,
            #     engine_model=engine_model,
            #     water_injection=water_injection,
            #     SAF=SAF,
            #     aircraft="A20N_full",
            #     time_bounds=time_bounds,
            #     prediction=prediction,
            #     diurnal="day",
            #     weather_model=weather_model,
            #     accuracy=accuracy
            # )