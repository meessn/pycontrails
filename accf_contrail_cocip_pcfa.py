import pandas as pd
import os
import numpy as np

# CONFIGURATION
trajectories_to_analyze = {
    "sfo_dfw": True,
    "malaga": False,
    "bos_fll": True,
    "cts_tpe": True,
    "dus_tos": True,
    "gru_lim": True,
    "hel_kef": True,
    "lhr_ist": True,
    "sin_maa": True
}

seasons_to_analyze = {
    "2023-02-06": True,
    "2023-05-05": True,
    "2023-08-06": True,
    "2023-11-06": True
}

diurnal_to_analyze = {
    "daytime": True,
    "nighttime": True
}

engine_models_to_analyze = {
    "GTF1990": True,
    "GTF2000": True,
    "GTF": True,
    "GTF2035": True,
    "GTF2035_wi": True
}

saf_levels_to_analyze = [0, 20, 100]
water_injection_levels = ["0", "15"]

# PATH SETUP
base_path = 'main_results_figures/results'

# # ANALYSIS
# results = []
# altitude_ranges = {}
#
# def sign_classification(x):
#     if x > 0:
#         return 'positive'
#     elif x < 0:
#         return 'negative'
#     else:
#         return 'zero'


def accf_contrail(diurnal, air_temperature, olr, efficacy):
    # Replace this with your real logic later
    if diurnal == 'daytime':
        result = 1e-10* (-1.7 - 0.0088*olr)*0.0151
    elif diurnal == 'nighttime':
        if air_temperature > 201:
            result = 1e-10* (0.0073*10**(0.0107*air_temperature)-1.03)*0.0151
        else:
            result = 0
    else:
        print('diurnal not specified or incorrect')

    if efficacy == True:
        result *= 0.42
    return result

for trajectory, trajectory_enabled in trajectories_to_analyze.items():
    if not trajectory_enabled:
        continue

    trajectory_path = os.path.join(base_path, trajectory)
    if not os.path.exists(trajectory_path):
        print(f"Trajectory folder not found: {trajectory_path}")
        continue

    min_altitude_trajectory = float('-inf')
    max_altitude_trajectory = float('inf')

    for folder in os.listdir(trajectory_path):
        for season, season_enabled in seasons_to_analyze.items():
            if not season_enabled or season not in folder:
                continue

            for diurnal, diurnal_enabled in diurnal_to_analyze.items():
                if not diurnal_enabled or diurnal not in folder:
                    continue

                climate_path = os.path.join(trajectory_path, folder, 'climate/mees/era5model')
                if not os.path.exists(climate_path):
                    print(f"climate folder not found: {climate_path}")
                    continue

                dfs = {}

                for engine, engine_enabled in engine_models_to_analyze.items():
                    if not engine_enabled:
                        continue

                    for saf in ([0] if engine in ["GTF1990", "GTF2000", "GTF"] else saf_levels_to_analyze):
                        for water_injection in (["15"] if engine == "GTF2035_wi" else ["0"]):
                            pattern = f"{engine}_SAF_{saf}_A20N_full_WAR_{water_injection}_climate.csv"
                            files = [f for f in os.listdir(climate_path) if f == pattern]

                            if not files:
                                print(f"No files found for {engine} SAF {saf} WAR {water_injection} in {folder}")
                                continue

                            file_path = os.path.join(climate_path, files[0])
                            df = pd.read_csv(file_path)





                            # Check if 'cocip_persistent_1' column exists
                            if 'cocip_persistent_1' in df.columns:
                                # Fill NaNs with 0 for logical condition check
                                condition = df['cocip_persistent_1'].fillna(0.0) == 1.0

                                # Apply logic: use accf_contrail if persistent == 1.0, else 0
                                df['accf_sac_accf_contrail_cocip'] = np.where(
                                    condition,
                                    df.apply(lambda row: accf_contrail(
                                        diurnal,
                                        row['air_temperature'],
                                        row['accf_sac_olr'],
                                        True
                                    ), axis=1),
                                    0.0
                                )
                            else:
                                # If column doesn't exist, set to 0
                                df['accf_sac_accf_contrail_cocip'] = 0.0

                            df.to_csv(file_path, index=False)



