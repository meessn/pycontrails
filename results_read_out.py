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

saf_levels_to_analyze = [0, 20, 60, 100]
water_injection_levels = ["0", "7_5" ,"15"]

# PATH SETUP
base_path = 'main_results_figures/results'

# ANALYSIS
results = []
altitude_ranges = {}
cocip_atr20_zero_count = 0
def sign_classification(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

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
                        for water_injection in (["7_5", "15"] if engine == "GTF2035_wi" else ["0"]):
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
                        for water_injection in (["7_5", "15"] if engine == "GTF2035_wi" else ["0"]):
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

                    saf_level = key[1]

                    if saf_level == 0:
                        trimmed_df['ei_co2_conservative'] = 3.825
                        trimmed_df['ei_co2_optimistic'] = 3.825
                        trimmed_df['ei_h2o'] = 1.237  # create new column if needed
                    elif saf_level == 20:
                        trimmed_df['ei_co2_conservative'] = 3.75
                        trimmed_df['ei_co2_optimistic'] = 3.1059
                        trimmed_df['ei_h2o'] = 1.264
                    elif saf_level == 60:
                        trimmed_df['ei_co2_conservative'] = 3.596
                        trimmed_df['ei_co2_optimistic'] = 1.6677
                        trimmed_df['ei_h2o'] = 1.3177
                    elif saf_level == 100:
                        trimmed_df['ei_co2_conservative'] = 3.4425
                        trimmed_df['ei_co2_optimistic'] = 0.2295
                        trimmed_df['ei_h2o'] = 1.370



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
                    ei_h2o_sum = trimmed_df['ei_h2o'].sum()
                    h2o_sum = (trimmed_df['ei_h2o'] * trimmed_df['fuel_flow'] * dt).sum()
                    ei_nvpm_mass_sum = trimmed_df['nvpm_ei_m'].sum()
                    nvpm_mass_sum = (trimmed_df['nvpm_ei_m'] * trimmed_df['fuel_flow'] * dt).sum()
                    ei_nvpm_num_sum = trimmed_df['nvpm_ei_n'].sum()
                    nvpm_num_sum = (trimmed_df['nvpm_ei_n'] * trimmed_df['fuel_flow'] * dt).sum()

                    if key[0] in ["GTF2000", "GTF", "GTF2035", "GTF2035_wi"]:
                        # Grab baseline for same (trajectory, season, diurnal)
                        baseline_key = ("GTF1990", 0, "0")
                        baseline_df = dfs.get(baseline_key)

                        if baseline_df is not None:
                            baseline_trimmed_df = baseline_df[
                                (baseline_df['altitude'] >= min_altitude_trajectory) &
                                (baseline_df['altitude'] <= max_altitude_trajectory)
                                ].copy()

                            baseline_trimmed_df.reset_index(drop=True, inplace=True)
                            trimmed_df.reset_index(drop=True, inplace=True)

                            # Ensure same length
                            min_len = min(len(trimmed_df), len(baseline_trimmed_df))
                            trimmed_df = trimmed_df.iloc[:min_len].copy()
                            baseline_trimmed_df = baseline_trimmed_df.iloc[:min_len].copy()
                            assert (trimmed_df['time'].iloc[:min_len].reset_index(drop=True) ==
                                    baseline_trimmed_df['time'].iloc[:min_len].reset_index(drop=True)).all(), \
                                f"Timestamps are not aligned for {trajectory}, {season}, {diurnal}"
                            # Compute dt per row
                            dt = 60

                            # Compute nvpm_num for current and baseline
                            nvpm_num = trimmed_df['nvpm_ei_n'] * trimmed_df['fuel_flow'] * dt
                            nvpm_num_baseline = baseline_trimmed_df['nvpm_ei_n'] * baseline_trimmed_df['fuel_flow'] * dt

                            # Avoid divide-by-zero
                            # nvpm_num_baseline = nvpm_num_baseline.replace(0, np.nan)
                            # Compute nvpm_num for current engine
                            # nvpm_num = trimmed_df['nvpm_ei_n']

                            # New constant baseline nvpm value
                            # nvpm_num_baseline_constant = 1e15
                            # Compute delta_pn
                            # Step 1: delta_pn (no changes)
                            delta_pn = nvpm_num / nvpm_num_baseline
                            delta_pn[trimmed_df['altitude'] <= 9160] = 1.0  # No correction at low altitudes
                            delta_pn = delta_pn.clip(lower=0.1)

                            # Step 2: Correction mask (all altitudes where valid)
                            mask = (delta_pn >= 0.1) & (delta_pn <= 1.0)

                            # Step 3: Compute ΔRF correction for all valid altitudes (your request)
                            delta_rf_contr = np.ones_like(delta_pn)
                            delta_rf_contr[mask] = np.arctan(1.9 * delta_pn[mask] ** 0.74) / np.arctan(1.9)

                            # Step 4: Cruise + high altitude mask (for reporting only)
                            cruise_mask = trimmed_df['flight_phase'].str.lower() == 'cruise'
                            high_alt_mask = trimmed_df['altitude'] > 9160
                            combined_mask = mask & cruise_mask & high_alt_mask

                            # Step 5: Compute stats ONLY where cruise + high alt + Δpn is valid
                            if combined_mask.any():
                                mean_delta_pn = delta_pn[combined_mask].mean()
                                mean_delta_rf = delta_rf_contr[combined_mask].mean()
                                mean_eta_current = trimmed_df.loc[combined_mask, 'engine_efficiency'].mean()
                                mean_eta_baseline = baseline_trimmed_df.loc[combined_mask, 'engine_efficiency'].mean()
                                mean_nvpm_current = trimmed_df.loc[combined_mask, 'nvpm_ei_n'].mean()
                                mean_nvpm_baseline = baseline_trimmed_df.loc[combined_mask, 'nvpm_ei_n'].mean()

                                print(
                                    f"[INFO] ΔRF correction stats for {key[0]} SAF {key[1]} | {trajectory}, {season}, {diurnal}")
                                print(f"  → Mean ΔRF factor: {mean_delta_rf:.4f}")
                                print(f"  → Mean Δpn factor: {mean_delta_pn:.4f}")
                                print(f"  → Mean η (current engine): {mean_eta_current:.4f}")
                                print(f"  → Mean η (baseline GTF1990): {mean_eta_baseline:.4f}")
                                print(f"  → Mean nvpm (current engine): {mean_nvpm_current:.2e}")
                                print(f"  → Mean nvpm (baseline GTF1990): {mean_nvpm_baseline:.2e}")

                            # Step 6: Apply correction globally (all altitudes, as long as Δpn is valid)
                            trimmed_df['accf_sac_aCCF_Cont_nvpm'] = trimmed_df['accf_sac_aCCF_Cont'] * delta_rf_contr
                    else:
                        trimmed_df['accf_sac_aCCF_Cont_nvpm'] = trimmed_df['accf_sac_aCCF_Cont']


                    """CLIMATE ACCF ALTITUDE FILTER!!"""
                    climate_df = trimmed_df[trimmed_df['altitude'] > 9160]
                    nox_impact_sum = (climate_df['fuel_flow']*dt*(climate_df['accf_sac_aCCF_O3']+climate_df['accf_sac_aCCF_CH4']*1.29)*climate_df['ei_nox']).sum()
                    co2_impact_cons_sum = (climate_df['fuel_flow']*dt*climate_df['accf_sac_aCCF_CO2']*(climate_df['ei_co2_conservative']/3.825)).sum()
                    co2_impact_opti_sum = (climate_df['fuel_flow'] * dt * climate_df['accf_sac_aCCF_CO2'] * (climate_df[
                        'ei_co2_optimistic']/3.825)).sum()
                    h2o_impact_sum = (climate_df['fuel_flow']*dt*climate_df['accf_sac_aCCF_H2O']*(climate_df['ei_h2o']/1.237)).sum()
                    """KEER EFFICACY NOG VOOR ATR20!!!! """
                    if 'cocip_atr20' in climate_df.columns:
                        cocip_sum = climate_df['cocip_atr20'].fillna(0).sum()
                        if cocip_sum == 0:
                            cocip_atr20_zero_count += 1
                        contrail_atr20_cocip = cocip_sum * 0.42
                    else:
                        contrail_atr20_cocip = 0
                    """SAC ACCF KEER SEGMENT LENGHT!! en dan sum"""
                    contrail_atr20_accf = (climate_df['accf_sac_aCCF_Cont']*climate_df['accf_sac_segment_length_km']).sum()
                    # contrail_atr20_accf_cocip_pcfa = (climate_df['accf_sac_accf_contrail_cocip']*climate_df['accf_sac_segment_length_km']).sum()
                    contrail_atr20_accf_cocip_pcfa = (climate_df['accf_sac_aCCF_Cont_nvpm'] * climate_df[
                        'accf_sac_segment_length_km']).sum()
                    climate_non_co2_cocip = nox_impact_sum + h2o_impact_sum + contrail_atr20_cocip
                    climate_total_cons_cocip = nox_impact_sum + h2o_impact_sum + contrail_atr20_cocip + co2_impact_cons_sum
                    climate_total_opti_cocip = nox_impact_sum + h2o_impact_sum + contrail_atr20_cocip + co2_impact_opti_sum

                    climate_non_co2_accf = nox_impact_sum + h2o_impact_sum + contrail_atr20_accf
                    climate_total_cons_accf = nox_impact_sum + h2o_impact_sum + contrail_atr20_accf + co2_impact_cons_sum
                    climate_total_opti_accf = nox_impact_sum + h2o_impact_sum + contrail_atr20_accf + co2_impact_opti_sum

                    climate_non_co2_accf_cocip_pcfa = nox_impact_sum + h2o_impact_sum + contrail_atr20_accf_cocip_pcfa
                    climate_total_cons_accf_cocip_pcfa = nox_impact_sum + h2o_impact_sum + contrail_atr20_accf_cocip_pcfa + co2_impact_cons_sum
                    climate_total_opti_accf_cocip_pcfa = nox_impact_sum + h2o_impact_sum + contrail_atr20_accf_cocip_pcfa + co2_impact_opti_sum

                    accf_sac_pcfa_sum = trimmed_df['accf_sac_pcfa'].sum()
                    accf_sac_issr_sum = trimmed_df['accf_sac_issr'].sum()
                    accf_sac_sac_sum = trimmed_df['accf_sac_sac'].sum()
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
                                            'ei_h2o_sum': ei_h2o_sum,
                                            'h2o_sum': h2o_sum,
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
                                            'contrail_atr20_accf_cocip_pcfa_sum': contrail_atr20_accf_cocip_pcfa,
                                            'climate_non_co2_cocip': climate_non_co2_cocip,
                                            'climate_total_cons_cocip': climate_total_cons_cocip,
                                            'climate_total_opti_cocip': climate_total_opti_cocip,
                                            'climate_non_co2_accf': climate_non_co2_accf,
                                            'climate_total_cons_accf': climate_total_cons_accf,
                                            'climate_total_opti_accf': climate_total_opti_accf,
                                            'climate_non_co2_accf_cocip_pcfa': climate_non_co2_accf_cocip_pcfa,
                                            'climate_total_cons_accf_cocip_pcfa': climate_total_cons_accf_cocip_pcfa,
                                            'climate_total_opti_accf_cocip_pcfa': climate_total_opti_accf_cocip_pcfa,
                                            'accf_sac_pcfa_sum': accf_sac_pcfa_sum,
                                            'accf_sac_issr_sum': accf_sac_issr_sum,
                                            'accf_sac_sac_sum': accf_sac_sac_sum

                                        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
accf_totals_by_engine_saf = results_df.groupby(['engine', 'saf_level'])[
    ['accf_sac_pcfa_sum', 'accf_sac_issr_sum', 'accf_sac_sac_sum']
].sum().reset_index()

# Display the results
print(accf_totals_by_engine_saf[['engine',  'accf_sac_pcfa_sum', 'accf_sac_issr_sum',  'accf_sac_sac_sum']])
# # Create a column for special cases
# results_df['is_special_case'] = (
#     (results_df['climate_total_cons_sum'] < 0) & (results_df['climate_total_opti_sum'] < 0) |
#     (results_df['climate_total_cons_sum'] > 0) & (results_df['climate_total_opti_sum'] < 0) |
#     (results_df['climate_non_co2'] < 0)
# )
#
# def get_sign_change_mask(baseline_engine, engines_to_compare=None):
#     baseline_df = results_df[results_df['engine'] == baseline_engine][['trajectory', 'season', 'diurnal', 'contrail_atr20_cocip_sum']]
#     merged_df = results_df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))
#
#     # Optional filtering for GTF baseline
#     if engines_to_compare is not None:
#         merged_df = merged_df[merged_df['engine'].isin(engines_to_compare)]
#
#     sign_change_mask = (
#         (np.sign(merged_df['contrail_atr20_cocip_sum']) != np.sign(merged_df['contrail_atr20_cocip_sum_baseline'])) &
#         (merged_df['contrail_atr20_cocip_sum'] != 0) & (merged_df['contrail_atr20_cocip_sum_baseline'] != 0)
#     )
#
#     # Return boolean mask for sign changes
#     mask_indices = merged_df.loc[sign_change_mask].index
#     return results_df.index.isin(mask_indices)
#
# # Get masks for sign changes
# sign_change_gtf1990_mask = get_sign_change_mask('GTF1990')
# sign_change_gtf_mask = get_sign_change_mask('GTF', engines_to_compare=['GTF2035', 'GTF2035_wi'])
#
# # Mark these as special cases
# results_df.loc[sign_change_gtf1990_mask, 'is_special_case'] = True
# results_df.loc[sign_change_gtf_mask, 'is_special_case'] = True
#
# # # Find all (trajectory, season, diurnal) combinations where any special case occurs
# # special_case_combinations = results_df.loc[
# #     results_df['is_special_case'], ['trajectory', 'season', 'diurnal']
# # ].drop_duplicates()
# #
# # # Create a multi-index set of these combinations for fast lookup
# # special_case_combinations_set = set(
# #     special_case_combinations.itertuples(index=False, name=None)
# # )
#
# # Find all (trajectory, season, diurnal) where GTF1990 has no contrail
# no_contrail_baseline_combinations = results_df[
#     (results_df['engine'] == 'GTF1990') & (results_df['contrail_atr20_cocip_sum'] == 0)
# ][['trajectory', 'season', 'diurnal']].drop_duplicates()
#
# # Convert to set for fast lookup
# no_contrail_baseline_combinations_set = set(
#     no_contrail_baseline_combinations.itertuples(index=False, name=None)
# )
#
# # Flag all engines for this (trajectory, season, diurnal) as special case if any other engine forms contrails
# for comb in no_contrail_baseline_combinations_set:
#     engines_in_combination = results_df[
#         (results_df['trajectory'] == comb[0]) &
#         (results_df['season'] == comb[1]) &
#         (results_df['diurnal'] == comb[2])
#     ]
#
#     # Check if any other engine forms a contrail (positive or negative)
#     if (engines_in_combination['contrail_atr20_cocip_sum'] != 0).any():
#         results_df.loc[
#             (results_df['trajectory'] == comb[0]) &
#             (results_df['season'] == comb[1]) &
#             (results_df['diurnal'] == comb[2]),
#             'is_special_case'
#         ] = True
#
#         print(f"CFM1990 zero contrail, other not: {comb[0]}, season: {comb[1]}, diurnal: {comb[2]}")
#
# # FINAL: Refresh the set after ALL special case markings are complete!
# special_case_combinations_set = set(
#     results_df.loc[
#         results_df['is_special_case'], ['trajectory', 'season', 'diurnal']
#     ].itertuples(index=False, name=None)
# )
#
#
#
#
# # Apply the final contrail category assignment, considering special cases across all engines for a combination
# def classify_contrail_category(row):
#     if (row['trajectory'], row['season'], row['diurnal']) in special_case_combinations_set:
#         return 'special case'
#     elif row['contrail_atr20_cocip_sum'] == 0:
#         return 'no contrail'
#     elif row['contrail_atr20_cocip_sum'] > 0:
#         return 'warming'
#     elif row['contrail_atr20_cocip_sum'] < 0:
#         return 'cooling'
#     else:
#         return 'unknown'
#
# results_df['contrail_category'] = results_df.apply(classify_contrail_category, axis=1)
#
#
#
#
# category_counts = results_df['contrail_category'].value_counts()
# print("Contrail Category Counts:")
# print(category_counts)

# Save results to CSV
results_df.to_csv('results_main_simulations_saf_war.csv', index=False)
print(f"Number of DataFrames where 'cocip_atr20' exists but sum is zero: {cocip_atr20_zero_count}")

GTF1990_2000_no_contrail_other_has_count = 0

# Filter for all GTF203_wi and GTF203_wi rows with zero contrail
GTF1990_baseline_rows = results_df[
    (results_df['engine'].isin(['GTF1990'])) &
    (results_df['contrail_atr20_cocip_sum'] == 0)
]

for idx, row in GTF1990_baseline_rows.iterrows():
    # Get all other engines for same flight scenario
    matching_rows = results_df[
        (results_df['trajectory'] == row['trajectory']) &
        (results_df['season'] == row['season']) &
        (results_df['diurnal'] == row['diurnal']) &
        (~results_df['engine'].isin(['GTF1990'])) &
        (results_df['contrail_atr20_cocip_sum'] != 0)
    ]

    # Each matching row is a case where another engine forms a contrail
    GTF1990_2000_no_contrail_other_has_count += len(matching_rows)

print(f"GTF1990/2000 had zero contrail while other engines had non-zero contrails {GTF1990_2000_no_contrail_other_has_count} times.")

# Set to store unique combinations
distinct_combinations = set()

# Loop as before
for idx, row in GTF1990_baseline_rows.iterrows():
    matching_rows = results_df[
        (results_df['trajectory'] == row['trajectory']) &
        (results_df['season'] == row['season']) &
        (results_df['diurnal'] == row['diurnal']) &
        (~results_df['engine'].isin(['GTF1990'])) &
        (results_df['contrail_atr20_cocip_sum'] != 0)
    ]
    if not matching_rows.empty:
        combo = (row['trajectory'], row['season'], row['diurnal'])
        distinct_combinations.add(combo)

print(f"Number of distinct trajectory + season + diurnal combinations (CoCiP): {len(distinct_combinations)}")
for combo in sorted(distinct_combinations):
    print(combo)


GTF1990_2000_no_contrail_other_has_count_accf = 0

# Filter for all GTF203_wi and GTF203_wi rows with zero contrail
GTF1990_baseline_rows_accf = results_df[
    (results_df['engine'].isin(['GTF1990'])) &
    (results_df['contrail_atr20_accf_cocip_pcfa_sum'] == 0)
]

for idx, row in GTF1990_baseline_rows_accf.iterrows():
    # Get all other engines for same flight scenario
    matching_rows_accf = results_df[
        (results_df['trajectory'] == row['trajectory']) &
        (results_df['season'] == row['season']) &
        (results_df['diurnal'] == row['diurnal']) &
        (~results_df['engine'].isin(['GTF1990'])) &
        (results_df['contrail_atr20_accf_cocip_pcfa_sum'] != 0)
    ]

    # Each matching row is a case where another engine forms a contrail
    GTF1990_2000_no_contrail_other_has_count_accf += len(matching_rows_accf)

print(f"GTF1990/2000 had zero contrail while other engines had non-zero contrails (accf) {GTF1990_2000_no_contrail_other_has_count_accf} times.")

distinct_combinations_accf = set()

for idx, row in GTF1990_baseline_rows_accf.iterrows():
    matching_rows_accf = results_df[
        (results_df['trajectory'] == row['trajectory']) &
        (results_df['season'] == row['season']) &
        (results_df['diurnal'] == row['diurnal']) &
        (~results_df['engine'].isin(['GTF1990'])) &
        (results_df['contrail_atr20_accf_cocip_pcfa_sum'] != 0)
    ]
    if not matching_rows_accf.empty:
        combo = (row['trajectory'], row['season'], row['diurnal'])
        distinct_combinations_accf.add(combo)

print(f"Number of distinct trajectory + season + diurnal combinations (aCCF): {len(distinct_combinations_accf)}")
for combo in sorted(distinct_combinations_accf):
    print(combo)


#
#
# climate_columns = [
#     'contrail_atr20_cocip_sum',
#     'contrail_atr20_accf_sum',
#     'climate_non_co2',
#     'climate_total_cons_sum',
#     'climate_total_opti_sum'
# ]
#
# # Check the signs for each row
# signs_df = results_df[climate_columns].apply(lambda col: col.map(sign_classification))
# print(sign_classification(-2.729908e-11))
# print(results_df.loc[530, climate_columns].apply(sign_classification))
# # Display the first few rows to get a sense of it
# print(signs_df.value_counts())
# # print(signs_df.head())
# cocip_nonzero_when_accf_zero = results_df[
#     (results_df['contrail_atr20_accf_sum'] == 0) &
#     (results_df['contrail_atr20_cocip_sum'] != 0)
# ]
# print('nonzero cocip when accf zero', cocip_nonzero_when_accf_zero[['contrail_atr20_cocip_sum', 'contrail_atr20_accf_sum', 'trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection']])
#
cocip_zero_count = (results_df['contrail_atr20_cocip_sum'] == 0).sum()
print(f"Number of flights where contrail_atr20_cocip_sum is zero: {cocip_zero_count}")

accf_zero_count = (results_df['contrail_atr20_accf_sum'] == 0).sum()
print(f"Number of flights where contrail_atr20_accf_sum is zero: {accf_zero_count}")

accf_cocip_pcfa_zero_count = (results_df['contrail_atr20_accf_cocip_pcfa_sum'] == 0).sum()
print(f"Number of flights where contrail_atr20_accf_cocip_pcfa_sum is zero: {accf_cocip_pcfa_zero_count}")
#
cocip_warming_count = (results_df['contrail_atr20_cocip_sum'] > 0).sum()
print(f"Number of flights where contrail_atr20_cocip_sum is warming: {cocip_warming_count}")
#
cocip_cooling_count = (results_df['contrail_atr20_cocip_sum'] < 0).sum()
print(f"Number of flights where contrail_atr20_cocip_sum is cooling: {cocip_cooling_count}")

accf_cocip_pcfa_warming_count = (results_df['contrail_atr20_accf_cocip_pcfa_sum'] > 0).sum()
print(f"Number of flights where contrail_atr20_accf_cocip_pcfa_sum is warming: {accf_cocip_pcfa_warming_count}")
#
accf_cocip_pcfa_cooling_count = (results_df['contrail_atr20_accf_cocip_pcfa_sum'] < 0).sum()
print(f"Number of flights where contrail_atr20_accf_cocip_pcfa_sum is cooling: {accf_cocip_pcfa_cooling_count}")
#
nighttime_cooling_cocip_count = results_df[
    (results_df['diurnal'] == 'nighttime') & (results_df['contrail_atr20_cocip_sum'] < 0)
].shape[0]
#
print(f"Number of nighttime flights with cooling (negative) cocip contrails: {nighttime_cooling_cocip_count}")
#
daytime_cooling_cocip_count = results_df[
    (results_df['diurnal'] == 'daytime') & (results_df['contrail_atr20_cocip_sum'] != 0)
].shape[0]
#
print(f"Number of daytime flights with cocip contrails: {daytime_cooling_cocip_count}")
#
# daytime_cooling_cocip_count_1990 = results_df[
#     (results_df['diurnal'] == 'daytime') & (results_df['contrail_atr20_cocip_sum'] != 0) & (results_df['engine'] == 'GTF1990')
# ].shape[0]
#
# print(f"Number of daytime flights with cocip contrails and 1990: {daytime_cooling_cocip_count_1990}")
#
# negative_non_co2_count = (results_df['climate_non_co2'] < 0).sum()
# print(f"Number of flights where climate_non_co2 is negative: {negative_non_co2_count}")
#
# # Extract signs
sign_cocip = np.sign(results_df['contrail_atr20_cocip_sum'])
sign_accf = np.sign(results_df['contrail_atr20_accf_sum'])
#
# # Mask for flights where signs differ, and both are non-zero
different_signs_mask = (sign_cocip != sign_accf) & (sign_cocip != 0) & (sign_accf != 0)
#
# Count the number of such flights
different_signs_count = different_signs_mask.sum()
#
print(f"Flights with different signs for ACCF and CoCiP contrails: {different_signs_count}")
#
different_signs_df = results_df[different_signs_mask]

# Check if any of these flights were during nighttime
nighttime_different_signs_df = different_signs_df[different_signs_df['diurnal'] == 'nighttime']

# Display results
if not nighttime_different_signs_df.empty:
    print(f"{len(nighttime_different_signs_df)} flights with different ACCF vs CoCiP signs occurred during nighttime:")
    print(nighttime_different_signs_df[['trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection']])
else:
    print("All flights with different ACCF vs CoCiP signs occurred during daytime.")


"""cocip vs accf_cocip_pcfa"""
sign_cocip = np.sign(results_df['contrail_atr20_cocip_sum'])
sign_accf_cocip_pcfa = np.sign(results_df['contrail_atr20_accf_cocip_pcfa_sum'])
#
# # Mask for flights where signs differ, and both are non-zero
different_signs_mask = (sign_cocip != sign_accf_cocip_pcfa) & (sign_cocip != 0) & (sign_accf_cocip_pcfa != 0)
#
# Count the number of such flights
different_signs_count = different_signs_mask.sum()
#
print(f"Flights with different signs for accf_cocip_pcfa and CoCiP contrails: {different_signs_count}")
#
different_signs_df = results_df[different_signs_mask]

# Check if any of these flights were during nighttime
nighttime_different_signs_df = different_signs_df[different_signs_df['diurnal'] == 'nighttime']

# Display results
if not nighttime_different_signs_df.empty:
    print(f"{len(nighttime_different_signs_df)} flights with different accf_cocip_pcfa vs CoCiP signs occurred during nighttime:")
    print(nighttime_different_signs_df[['trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection']])
else:
    print("All flights with different accf_cocip_pcfa vs CoCiP signs occurred during daytime.")


""""cooling/warming"""
# Group by trajectory, season, diurnal
grouped = results_df.groupby(['trajectory', 'season', 'diurnal'])

cooling_groups = []
warming_groups = []

for name, group in grouped:
    values = group['contrail_atr20_cocip_sum']

    if (values < 0).all():
        cooling_groups.append((name, group[['engine', 'saf_level', 'water_injection', 'contrail_atr20_cocip_sum']]))
    elif (values > 0).all():
        warming_groups.append((name, group[['engine', 'saf_level', 'water_injection', 'contrail_atr20_cocip_sum']]))

# Display results
print("\n=== All Cooling Contrail Combinations ===")
for (traj_seas_diur, configs) in cooling_groups:
    print(f"\n{traj_seas_diur}:")
    print(configs.to_string(index=False))

print("\n=== All Warming Contrail Combinations ===")
for (traj_seas_diur, configs) in warming_groups:
    print(f"\n{traj_seas_diur}:")
    print(configs.to_string(index=False))

# Filter to relevant engines
gtf2035_df = results_df[results_df['engine'] == 'GTF2035']
gtf1990_df = results_df[results_df['engine'] == 'GTF1990']

# Merge on scenario identifiers
merged_df = pd.merge(
    gtf2035_df,
    gtf1990_df,
    on=['trajectory', 'season', 'diurnal'],
    suffixes=('_gtf2035', '_gtf1990')
)

# Compare NOx climate impact
nox_higher_mask = merged_df['nox_impact_sum_gtf2035'] > merged_df['nox_impact_sum_gtf1990']

# Extract combinations
higher_nox_combinations = merged_df[nox_higher_mask][
    ['trajectory', 'season', 'diurnal', 'nox_impact_sum_gtf2035', 'nox_impact_sum_gtf1990']]

# Display result
print("Flight scenarios where GTF2035 has higher NOx climate impact than GTF1990:")
print(higher_nox_combinations.to_string(index=False))

#
# # Find the flight with positive cons and negative opti
# weird_flight_1 = results_df[
#     (results_df['climate_total_cons_sum'] > 0) &
#     (results_df['climate_total_opti_sum'] < 0)
# ]
# # Show all columns for the specific flight cases
# pd.set_option('display.max_columns', None)  # Show all columns
#
# print("Flight with positive cons and negative opti:")
# print(weird_flight_1) # sin_maa  2023-05-05  daytime  GTF2035_wi        100              15
# print("total climate impact negative (both cons and pos)", results_df[
#     (results_df['climate_total_cons_sum'] < 0) &
#     (results_df['climate_total_opti_sum'] < 0)
# ])
#
# # Filter for GTF engine and rows where all relevant values are non-zero
# gtf_df = results_df[
#     (results_df['engine'] == 'GTF') &
#     (results_df['contrail_atr20_cocip_sum'] != 0) &
#     (results_df['contrail_atr20_accf_sum'] != 0) &
#     (results_df['nox_impact_sum'] != 0)
# ].copy()
#
# # Compute absolute values
# gtf_df['abs_cocip'] = gtf_df['contrail_atr20_cocip_sum'].abs()
# gtf_df['abs_accf'] = gtf_df['contrail_atr20_accf_sum'].abs()
# gtf_df['abs_nox'] = gtf_df['nox_impact_sum'].abs()
#
# # Split by diurnal
# night_df = gtf_df[gtf_df['diurnal'] == 'nighttime']
# day_df = gtf_df[gtf_df['diurnal'] == 'daytime']
#
# # Compute means of absolute values
# night_mean_cocip = night_df['abs_cocip'].mean()
# night_mean_accf = night_df['abs_accf'].mean()
# night_mean_nox = night_df['abs_nox'].mean()
#
# day_mean_cocip = day_df['abs_cocip'].mean()
# day_mean_accf = day_df['abs_accf'].mean()
# day_mean_nox = day_df['abs_nox'].mean()
#
# # Print results
# print("== Mean ABS Values (GTF engine only) ==")
# print(f"Nighttime — CoCiP: {night_mean_cocip:.6e}, ACCF: {night_mean_accf:.6e}, NOx Impact: {night_mean_nox:.6e}")
# print(f"Daytime   — CoCiP: {day_mean_cocip:.6e}, ACCF: {day_mean_accf:.6e}, NOx Impact: {day_mean_nox:.6e}")
#
# impact_columns = [
#     'nox_impact_sum',
#     'co2_impact_cons_sum',
#     'co2_impact_opti_sum',
#     'h2o_impact_sum'
# ]
#
# # Check if any values are negative
# negative_values = results_df[impact_columns].lt(0).sum()
#
# # Print the results nicely
# for col, count in negative_values.items():
#     if count > 0:
#         print(f"{col} has {count} negative values")
#     else:
#         print(f"{col} has NO negative values")
#
#
#
# # Check if cocip and accf have different signs (ignoring zeros)
# sign_cocip = np.sign(results_df['contrail_atr20_cocip_sum'])
# sign_accf = np.sign(results_df['contrail_atr20_accf_sum'])
#
# # Identify where signs are different (ignores zeros by design)
# different_signs_mask = (sign_cocip != sign_accf) & (sign_cocip != 0) & (sign_accf != 0)
#
# # Count the number of such cases
# different_signs_count = different_signs_mask.sum()
#
# print(f"Number of flights where cocip and accf have different signs: {different_signs_count}")
# #
# # If you want to display these rows
# different_signs_rows = results_df[different_signs_mask]
# print(different_signs_rows[['trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection', 'contrail_atr20_cocip_sum', 'contrail_atr20_accf_sum']])

import pandas as pd
import os

# Constants
target_engine = "GTF"
target_season = "2023-02-06"
target_diurnal = "daytime"
meters_to_feet = 3.28084

# List to store rows
selected_rows = []

# Loop through relevant trajectories
for trajectory, enabled in trajectories_to_analyze.items():
    if not enabled:
        continue

    trajectory_path = os.path.join(base_path, trajectory)
    if not os.path.exists(trajectory_path):
        continue

    for folder in os.listdir(trajectory_path):
        if target_season not in folder or target_diurnal not in folder:
            continue

        climate_path = os.path.join(trajectory_path, folder, 'climate/mees/era5model')
        if not os.path.exists(climate_path):
            continue

        pattern = f"{target_engine}_SAF_0_A20N_full_WAR_0_climate.csv"
        file_path = os.path.join(climate_path, pattern)

        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df['altitude_ft'] = df['altitude'] * meters_to_feet

        if 'flight_phase' not in df.columns:
            print(f"No flight_phase column in: {file_path}")
            continue

        cruise_indices = df.index[df['flight_phase'] == 'cruise'].tolist()
        if not cruise_indices:
            print(f"No cruise phase found for {trajectory}")
            continue

        toc_index = max(0, cruise_indices[0] - 2)
        toc_row = df.loc[toc_index]

        middle_cruise_index = cruise_indices[len(cruise_indices) // 2]
        middle_cruise_row = df.loc[middle_cruise_index]

        climb_df = df[df['flight_phase'] == 'climb']
        start_climb_row = climb_df.iloc[(climb_df['altitude'] - 920).abs().argmin()]

        # Get descent rows sorted by altitude closeness to 920m
        descent_df = df[df['flight_phase'] == 'descent'].copy()
        descent_df['altitude_diff'] = (descent_df['altitude'] - 920).abs()
        descent_df_sorted = descent_df.sort_values('altitude_diff')

        required_columns = ['thrust_gsp', 'nvpm_ei_n', 'ei_nox', 'nvpm_ei_m', 'fuel_flow_gsp']

        # Find first valid descent row
        for _, row in descent_df_sorted.iterrows():
            if all(pd.notnull(row[col]) and row[col] != 0 for col in required_columns) and row['thrust_gsp'] >= 0.0:
                start_descent_row = row
                break
        else:
            print(f"[WARNING] No valid descent row with thrust ≥ 0.0 and non-zero engine metrics for {trajectory}")
            start_descent_row = descent_df_sorted.iloc[0]  # fallback

        # Calculate total distance
        total_distance_km = df['accf_sac_segment_length_km'].sum()

        # Get first and last non-zero altitude
        nonzero_altitudes = df[df['altitude'] > 0]
        first_altitude = nonzero_altitudes.iloc[0]['altitude'] if not nonzero_altitudes.empty else None
        last_altitude = nonzero_altitudes.iloc[-1]['altitude'] if not nonzero_altitudes.empty else None

        def collect_row(label, row):
            return {
                'trajectory': trajectory,
                'phase': label,
                'air_temperature': row['air_temperature'],
                'thrust_gsp': row['thrust_gsp'],
                'nvpm_ei_n': row['nvpm_ei_n'],
                'ei_nox (g/kg)': row['ei_nox'] * 1000,
                'nvpm_ei_m (mg/kg)': row['nvpm_ei_m'] * 1e6,
                'fuel_flow_gsp': row['fuel_flow_gsp'],
                'altitude_m': row['altitude'],
                'altitude_ft': row['altitude_ft'],
                'total_distance_km': total_distance_km,
                'first_altitude_m': first_altitude,
                'last_altitude_m': last_altitude
            }

        selected_rows.append(collect_row("top_of_climb", toc_row))
        selected_rows.append(collect_row("middle_cruise", middle_cruise_row))
        selected_rows.append(collect_row("start_of_climb", start_climb_row))
        selected_rows.append(collect_row("start_of_descent", start_descent_row))


# Convert to DataFrame
selected_df = pd.DataFrame(selected_rows)

# Mapping from trajectory name to (Start, End)
trajectory_city_map = {
    'hel_kef': ('Helsinki, Finland', 'Reykjavik, Iceland'),
    'dus_tos': ('Dusseldorf, Germany', 'Tromso, Norway'),
    'lhr_ist': ('London, UK', 'Istanbul, Turkey'),
    'cts_tpe': ('Sapporo, Japan', 'Taipei, Taiwan'),
    'bos_fll': ('Boston, USA', 'Miami, USA'),
    'sfo_dfw': ('San Francisco, USA', 'Dallas, USA'),
    'sin_maa': ('Singapore, Singapore', 'Chennai, India'),
    'gru_lim': ('Sao Paulo, Brazil', 'Lima, Peru')
}

# Apply mapping to new columns
selected_df['start_city'] = selected_df['trajectory'].map(lambda x: trajectory_city_map.get(x, ('', ''))[0])
selected_df['end_city'] = selected_df['trajectory'].map(lambda x: trajectory_city_map.get(x, ('', ''))[1])

# Format numbers with comma as decimal and semicolon delimiter
selected_df.to_csv(
    'results_report/document_yin/gtf_daytime_feb06_selected_points.csv',
    # sep=';',
    index=False,
    float_format='%.6f',
    # decimal=','
)

print("Saved: gtf_daytime_feb06_selected_points.csv")
