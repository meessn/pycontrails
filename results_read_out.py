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

                    saf_level = key[1]

                    if saf_level == 0:
                        trimmed_df['ei_co2_conservative'] = 3.825
                        trimmed_df['ei_co2_optimistic'] = 3.825
                        trimmed_df['ei_h2o'] = 1.237  # create new column if needed
                    elif saf_level == 20:
                        trimmed_df['ei_co2_conservative'] = 3.75
                        trimmed_df['ei_co2_optimistic'] = 3.1059
                        trimmed_df['ei_h2o'] = 1.264
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
                    ei_nvpm_mass_sum = trimmed_df['nvpm_ei_m'].sum()
                    nvpm_mass_sum = (trimmed_df['nvpm_ei_m'] * trimmed_df['fuel_flow'] * dt).sum()
                    ei_nvpm_num_sum = trimmed_df['nvpm_ei_n'].sum()
                    nvpm_num_sum = (trimmed_df['nvpm_ei_n'] * trimmed_df['fuel_flow'] * dt).sum()

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
                    contrail_atr20_accf_cocip_pcfa = (climate_df['accf_sac_aCCF_Cont'] * climate_df[
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
results_df.to_csv('results_main_simulations.csv', index=False)
print(f"Number of DataFrames where 'cocip_atr20' exists but sum is zero: {cocip_atr20_zero_count}")

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
