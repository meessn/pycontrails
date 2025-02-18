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
water_injection_levels = ["0", "15"]

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

                climate_path = os.path.join(trajectory_path, folder, 'climate/mees/era5model')
                if not os.path.exists(climate_path):
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
                                continue

                            file_path = os.path.join(climate_path, files[0])
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

                    dt = (trimmed_df['time'].iloc[1] - trimmed_df['time'].iloc[0]).total_seconds()
                    # FOR BOTH ENGINES!!!!!
                    fuel_sum = trimmed_df['fuel_flow'].sum()
                    fuel_kg_sum = (trimmed_df['fuel_flow']*dt).sum()
                    energy_sum = (trimmed_df['fuel_flow'] * trimmed_df['LHV'] * dt).sum() #kJ
                    ei_co2_conservative_sum = trimmed_df['ei_co2_conservative'].sum()
                    ei_co2_optimistic_sum = trimmed_df['ei_co2_optimistic'].sum()
                    co2_conservative_sum = (trimmed_df['fuel_flow'] * trimmed_df['ei_co2_conservative'] * dt).sum()
                    co2_optimistic_sum = (trimmed_df['fuel_flow'] * trimmed_df['ei_co2_optimistic'] * dt).sum()
                    ei_nox_sum = trimmed_df['ei_nox'].sum()
                    nox_sum = (trimmed_df['ei_nox']*trimmed_df['fuel_flow']*dt).sum()
                    ei_nvpm_mass_sum = trimmed_df['nvpm_ei_m'].sum()
                    nvpm_mass_sum = (trimmed_df['nvpm_ei_m'] * trimmed_df['fuel_flow'] * dt).sum()
                    ei_nvpm_num_sum = trimmed_df['nvpm_ei_n'].sum()
                    nvpm_num_sum = (trimmed_df['nvpm_ei_n'] * trimmed_df['fuel_flow'] * dt).sum()
                    #climate impact add PMO
                    nox_impact_sum = (trimmed_df['fuel_flow']*dt*(trimmed_df['accf_sac_aCCF_O3']+trimmed_df['accf_sac_aCCF_CH4']*1.29)*trimmed_df['ei_nox']).sum()
                    co2_impact_cons_sum = (trimmed_df['fuel_flow']*dt*trimmed_df['accf_sac_aCCF_CO2']*trimmed_df['ei_co2_conservative']).sum()
                    co2_impact_opti_sum = (trimmed_df['fuel_flow'] * dt * trimmed_df['accf_sac_aCCF_CO2'] * trimmed_df[
                        'ei_co2_optimistic']).sum()
                    h2o_impact_sum = (trimmed_df['fuel_flow']*dt*trimmed_df['accf_sac_aCCF_H2O']*trimmed_df['ei_co2_conservative']).sum()
                    # trimmed_df['cocip_atr20'] = trimmed_df['cocip_atr20'].fillna(0)
                    contrail_atr20_cocip = trimmed_df['cocip_atr20'].fillna(0).sum() if 'cocip_atr20' in trimmed_df.columns else 0
                    contrail_atr20_accf = trimmed_df['accf_sac_contrails_atr20'].sum()
                    climate_total_cons = nox_impact_sum + h2o_impact_sum + contrail_atr20_cocip + co2_impact_cons_sum
                    climate_total_opti = nox_impact_sum + h2o_impact_sum + contrail_atr20_cocip + co2_impact_opti_sum

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
                                            'fuel_kg_sum': fuel_kg_sum,
                                            'energy_sum': energy_sum,
                                            'ei_co2_conservative_sum': ei_co2_conservative_sum,
                                            'ei_co2_optimistic_sum': ei_co2_optimistic_sum,
                                            'co2_conservative_sum': co2_conservative_sum,
                                            'co2_optimistic_sum': co2_optimistic_sum,
                                            'ei_nox_sum': ei_nox_sum,
                                            'nox_sum': nox_sum,
                                            'ei_nvpm_mass_sum': ei_nvpm_mass_sum,
                                            'nvpm_mass_sum': nvpm_mass_sum,
                                            'ei_nvpm_num_sum': ei_nvpm_num_sum,
                                            'nvpm_num_sum': nvpm_num_sum,
                                            # HIER KOMEN NOG KLIMAAT RESULTATEN!!!!!
                                            # Climate impact variables
                                            'nox_impact_sum': nox_impact_sum,
                                            'co2_impact_cons_sum': co2_impact_cons_sum,
                                            'co2_impact_opti_sum': co2_impact_opti_sum,
                                            'h2o_impact_sum': h2o_impact_sum,
                                            'contrail_atr20_cocip_sum': contrail_atr20_cocip,
                                            'contrail_atr20_accf_sum': contrail_atr20_accf,
                                            'climate_total_cons_sum': climate_total_cons,
                                            'climate_total_opti_sum': climate_total_opti

                                        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display or export results
print(results_df)
results_df.to_csv('results_main_simulations.csv', index=False)

climate_columns = [
    'contrail_atr20_cocip_sum',
    'contrail_atr20_accf_sum',
    'climate_total_cons_sum',
    'climate_total_opti_sum'
]

# Check the signs for each row
signs_df = results_df[climate_columns].applymap(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'zero')

# Display the first few rows to get a sense of it
print(signs_df.value_counts())
print(signs_df.head())