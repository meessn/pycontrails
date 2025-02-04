from main_emissions import run_emissions
import os




# Root directory containing flight trajectories
root_dir = "flight_trajectories/processed_flights"

# Select which trajectories to simulate
flight_trajectories_to_simulate = {
    "bos_fll": True,
    "cts_tpe": False,
    "dus_tos": False,
    "gru_lim": False,
    "hel_kef": False,
    "lhr_ist": False,
    "sfo_dfw": False,
    "sin_maa": False
}

# Time bounds for different flight dates
time_bounds_dict = {
    "2023-02-06": ("2023-02-05 14:00", "2023-02-07 11:00"),
    "2023-05-05": ("2023-05-04 14:00", "2023-05-06 11:00"),
    "2023-08-06": ("2023-08-05 14:00", "2023-08-07 11:00"),
    "2023-11-06": ("2023-11-05 14:00", "2023-11-07 11:00")
}

# Engine models to run
engine_models = {
    "GTF1990": True,
    "GTF2000": True,
    "GTF": True,
    "GTF2035": True,
    "GTF2035_wi_gass_on_design": True
}

# SAF values based on engine model
saf_dict = {
    "SAF20": True,
    "SAF100": True
}

# Loop through flight directories
for trajectory, should_simulate in flight_trajectories_to_simulate.items():
    if not should_simulate:
        continue

    trajectory_path = os.path.join(root_dir, trajectory)

    if not os.path.exists(trajectory_path):
        print(f"Warning: {trajectory_path} does not exist. Skipping.")
        continue

    flight_files = [f for f in os.listdir(trajectory_path) if f.endswith(".csv")]

    for flight_file in flight_files:
        file_parts = flight_file.split("_")
        flight_date = file_parts[2]  # Extracts YYYY-MM-DD

        if flight_date not in time_bounds_dict:
            print(f"Skipping {flight_file} (no matching time bounds).")
            continue

        time_bounds = time_bounds_dict[flight_date]
        flight_path = os.path.join(trajectory_path, flight_file)

        # Run for each engine model
        for engine_model, run_engine in engine_models.items():
            if not run_engine:
                continue

            # Determine SAF values
            saf_values = [0]
            if engine_model in ("GTF2035", "GTF2035_wi_gass_on_design"):
                if saf_dict["SAF20"]:
                    saf_values.append(20)
                if saf_dict["SAF100"]:
                    saf_values.append(100)

            # Determine water injection values
            water_injection = [0, 0, 0]
            if engine_model == "GTF2035_wi_gass_on_design":
                water_injection = [15, 15, 15]

            for SAF in saf_values:
                run_emissions(trajectory, flight_path, engine_model, water_injection, SAF, aircraft="A20N_full", time_bounds=time_bounds)