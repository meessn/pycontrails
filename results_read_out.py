import pandas as pd
import os

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
water_injection_levels = ["0_0_0", "15_15_15"]

# PATH SETUP
base_path = 'main_results_figures/results'

# ANALYSIS
results = []
altitude_ranges = {}

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

                emissions_path = os.path.join(trajectory_path, folder, 'emissions')
                if not os.path.exists(emissions_path):
                    print(f"Emissions folder not found: {emissions_path}")
                    continue

                dfs = {}

                for engine, engine_enabled in engine_models_to_analyze.items():
                    if not engine_enabled:
                        continue

                    for saf in ([0] if engine in ["GTF1990", "GTF2000", "GTF"] else saf_levels_to_analyze):
                        for water_injection in (["15_15_15"] if engine == "GTF2035_wi" else ["0_0_0"]):
                            pattern = f"{engine}_SAF_{saf}_A20N_full_WAR_{water_injection}.csv"
                            files = [f for f in os.listdir(emissions_path) if f == pattern]

                            if not files:
                                print(f"No files found for {engine} SAF {saf} WAR {water_injection} in {folder}")
                                continue

                            file_path = os.path.join(emissions_path, files[0])
                            df = pd.read_csv(file_path)

                            # Store dataframe with key
                            dfs[(engine, saf, water_injection)] = df

                if not dfs:
                    continue

                # Trim to common altitude range across engines for this specific folder
                min_altitude = max(df['altitude'].min() for df in dfs.values())
                max_altitude = min(df['altitude'].max() for df in dfs.values())

                # Update trajectory-wide min and max altitude
                min_altitude_trajectory = max(min_altitude_trajectory, min_altitude)
                max_altitude_trajectory = min(max_altitude_trajectory, max_altitude)

    altitude_ranges[trajectory] = (min_altitude_trajectory, max_altitude_trajectory)

for trajectory, trajectory_enabled in trajectories_to_analyze.items():
    if not trajectory_enabled:
        continue

    trajectory_path = os.path.join(base_path, trajectory)
    if not os.path.exists(trajectory_path):
        continue

    min_altitude_trajectory, max_altitude_trajectory = altitude_ranges[trajectory]

    for folder in os.listdir(trajectory_path):
        for season, season_enabled in seasons_to_analyze.items():
            if not season_enabled or season not in folder:
                continue

            for diurnal, diurnal_enabled in diurnal_to_analyze.items():
                if not diurnal_enabled or diurnal not in folder:
                    continue

                emissions_path = os.path.join(trajectory_path, folder, 'emissions')
                if not os.path.exists(emissions_path):
                    continue

                dfs = {}

                for engine, engine_enabled in engine_models_to_analyze.items():
                    if not engine_enabled:
                        continue

                    for saf in ([0] if engine in ["GTF1990", "GTF2000", "GTF"] else saf_levels_to_analyze):
                        for water_injection in (["15_15_15"] if engine == "GTF2035_wi" else ["0_0_0"]):
                            pattern = f"{engine}_SAF_{saf}_A20N_full_WAR_{water_injection}.csv"
                            files = [f for f in os.listdir(emissions_path) if f == pattern]

                            if not files:
                                continue

                            file_path = os.path.join(emissions_path, files[0])
                            df = pd.read_csv(file_path)

                            dfs[(engine, saf, water_injection)] = df

                if not dfs:
                    continue

                for key, df in dfs.items():
                    untrimmed_min_altitude = df['altitude'].min()
                    untrimmed_max_altitude = df['altitude'].max()

                    trimmed_df = df[(df['altitude'] >= min_altitude_trajectory) & (df['altitude'] <= max_altitude_trajectory)].copy()
                    trimmed_df['time'] = pd.to_datetime(trimmed_df['time'])
                    time_diff = trimmed_df['time'].diff().dt.total_seconds()
                    gaps = time_diff[time_diff > 60]
                    if not gaps.empty:
                        print(f"WARNING: Gaps detected in {trajectory} {season} {diurnal} {key}:\n{gaps}")

                    fuel_sum = trimmed_df['fuel_flow'].sum()
                    emissions_sum = trimmed_df['emissions'].sum() if 'emissions' in trimmed_df.columns else None
                    climate_impact_sum = trimmed_df['climate_impact'].sum() if 'climate_impact' in trimmed_df.columns else None


                    results.append({
                                            'trajectory': trajectory,
                                            'season': season,
                                            'diurnal': diurnal,
                                            'engine': key[0],
                                            'saf_level': key[1],
                                            'water_injection': key[2],
                                            'untrimmed_min_altitude': untrimmed_min_altitude,
                                            'untrimmed_max_altitude': untrimmed_max_altitude,
                                            'trimmed_min_altitude': min_altitude_trajectory,
                                            'trimmed_max_altitude': max_altitude_trajectory,
                                            'fuel_sum': fuel_sum,
                                            'emissions_sum': emissions_sum,
                                            'climate_impact_sum': climate_impact_sum
                                        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display or export results
print(results_df)