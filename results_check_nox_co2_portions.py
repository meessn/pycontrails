import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.optimize import curve_fit

# # CONFIGURATION
# trajectories_to_analyze = {
#     "sfo_dfw": True,
#     "malaga": False,
#     "bos_fll": True,
#     "cts_tpe": True,
#     "dus_tos": True,
#     "gru_lim": True,
#     "hel_kef": True,
#     "lhr_ist": True,
#     "sin_maa": True
# }
#
# seasons_to_analyze = {
#     "2023-02-06": True,
#     "2023-05-05": True,
#     "2023-08-06": True,
#     "2023-11-06": True
# }
#
# diurnal_to_analyze = {
#     "daytime": True,
#     "nighttime": True
# }
#
# engine_models_to_analyze = {
#     "GTF1990": True,
#     "GTF2000": True,
#     "GTF": True,
#     "GTF2035": True,
#     "GTF2035_wi": True
# }
#
# saf_levels_to_analyze = [0, 20, 100]
# water_injection_levels = ["0", "15"]
#
# base_path = 'main_results_figures/results'
#
# # First pass: Determine common altitude range & waypoints per trajectory
# altitude_ranges = {}
# trajectory_waypoints = {}
#
# for trajectory, trajectory_enabled in trajectories_to_analyze.items():
#     if not trajectory_enabled:
#         continue
#
#     trajectory_path = os.path.join(base_path, trajectory)
#     if not os.path.exists(trajectory_path):
#         print(f"Trajectory folder not found: {trajectory_path}")
#         continue
#
#     min_altitude_trajectory = float('-inf')
#     max_altitude_trajectory = float('inf')
#     all_indices = []  # Collect waypoints across all seasons & diurnal conditions
#
#     for folder in os.listdir(trajectory_path):
#         for season, season_enabled in seasons_to_analyze.items():
#             if not season_enabled or season not in folder:
#                 continue
#
#             for diurnal, diurnal_enabled in diurnal_to_analyze.items():
#                 if not diurnal_enabled or diurnal not in folder:
#                     continue
#
#                 climate_path = os.path.join(trajectory_path, folder, 'climate/mees/era5model')
#                 if not os.path.exists(climate_path):
#                     print(f"Climate folder not found: {climate_path}")
#                     continue
#
#                 dfs = {}
#
#                 for engine, engine_enabled in engine_models_to_analyze.items():
#                     if not engine_enabled:
#                         continue
#
#                     for saf in ([0] if engine in ["GTF1990", "GTF2000", "GTF"] else saf_levels_to_analyze):
#                         for water_injection in (["15"] if engine == "GTF2035_wi" else ["0"]):
#                             pattern = f"{engine}_SAF_{saf}_A20N_full_WAR_{water_injection}_climate.csv"
#                             file_path = os.path.join(climate_path, pattern)
#
#                             if not os.path.exists(file_path):
#                                 continue
#
#                             df = pd.read_csv(file_path)
#                             dfs[(engine, saf, water_injection)] = df
#
#                 if not dfs:
#                     continue
#
#                 # Find common altitude range
#                 min_altitude = max(df['altitude'].min() for df in dfs.values())
#                 max_altitude = min(df['altitude'].max() for df in dfs.values())
#
#                 min_altitude_trajectory = max(min_altitude_trajectory, min_altitude)
#                 max_altitude_trajectory = min(max_altitude_trajectory, max_altitude)
#
#                 # Collect waypoints for common indices calculation (only within the altitude range)
#                 for df in dfs.values():
#                     df_filtered = df[(df['altitude'] >= min_altitude_trajectory) & (df['altitude'] <= max_altitude_trajectory)]
#                     all_indices.append(set(df_filtered['index']))
#
#     # Determine final common waypoints for this trajectory
#     if all_indices:
#         common_indices = set.intersection(*all_indices)
#         print(f"{trajectory}: Found {len(common_indices)} common waypoints across all seasons & diurnal conditions.")
#     else:
#         print(f"{trajectory}: No common waypoints found across all conditions.")
#         continue  # Skip this trajectory
#
#     # Store results for second pass
#     altitude_ranges[trajectory] = (min_altitude_trajectory, max_altitude_trajectory)
#     trajectory_waypoints[trajectory] = common_indices  # Store waypoints per trajectory
#
# # Second pass: Load and apply the common altitude & waypoints
# dataframes = []
#
# for trajectory, trajectory_enabled in trajectories_to_analyze.items():
#     if not trajectory_enabled:
#         continue
#
#     trajectory_path = os.path.join(base_path, trajectory)
#     if not os.path.exists(trajectory_path):
#         continue
#
#     min_altitude_trajectory, max_altitude_trajectory = altitude_ranges[trajectory]
#     common_indices = trajectory_waypoints[trajectory]
#
#     for folder in os.listdir(trajectory_path):
#         for season, season_enabled in seasons_to_analyze.items():
#             if not season_enabled or season not in folder:
#                 continue
#
#             for diurnal, diurnal_enabled in diurnal_to_analyze.items():
#                 if not diurnal_enabled or diurnal not in folder:
#                     continue
#
#                 climate_path = os.path.join(trajectory_path, folder, 'climate/mees/era5model')
#                 if not os.path.exists(climate_path):
#                     continue
#
#                 dfs = {}
#
#                 for engine, engine_enabled in engine_models_to_analyze.items():
#                     if not engine_enabled:
#                         continue
#
#                     for saf in ([0] if engine in ["GTF1990", "GTF2000", "GTF"] else saf_levels_to_analyze):
#                         for water_injection in (["15"] if engine == "GTF2035_wi" else ["0"]):
#                             pattern = f"{engine}_SAF_{saf}_A20N_full_WAR_{water_injection}_climate.csv"
#                             file_path = os.path.join(climate_path, pattern)
#
#                             if not os.path.exists(file_path):
#                                 continue
#
#                             df = pd.read_csv(file_path)
#
#                             # Apply trimming based on the precomputed ranges
#                             df = df[(df['altitude'] >= min_altitude_trajectory) & (df['altitude'] <= max_altitude_trajectory) & (df['index'].isin(common_indices))]
#
#                             if saf == 0:
#                                 df['ei_co2_conservative'] = 3.825
#                                 df['ei_co2_optimistic'] = 3.825
#                                 df['ei_h2o'] = 1.237
#                             elif saf == 20:
#                                 df['ei_co2_conservative'] = 3.75
#                                 df['ei_co2_optimistic'] = 3.1059
#                                 df['ei_h2o'] = 1.264
#                             elif saf == 100:
#                                 df['ei_co2_conservative'] = 3.4425
#                                 df['ei_co2_optimistic'] = 0.2295
#                                 df['ei_h2o'] = 1.370
#
#                             df['trajectory'] = trajectory
#                             df['season'] = season
#                             df['diurnal'] = diurnal
#                             df['engine'] = engine
#                             df['saf_level'] = saf
#                             df['water_injection'] = water_injection
#
#                             # Print final length per engine
#                             print(f"Final Trim {trajectory} - {season} - {diurnal} - Engine: {engine}, SAF: {saf}, WAR: {water_injection} → Length: {len(df)}")
#
#                             dfs[(engine, saf, water_injection)] = df
#
#                 if not dfs:
#                     continue
#
#                 # Verify if all engines for this flight setup have the same length
#                 lengths = [len(df) for df in dfs.values()]
#                 unique_lengths = set(lengths)
#
#                 if len(unique_lengths) == 1:
#                     print(f"FINAL CHECK: {trajectory} - All flights now have the same length ({unique_lengths.pop()} rows)")
#
#                     # Append the filtered data
#                     for df in dfs.values():
#
#                         #Ensure 'cocip_atr20' exists in df, fillwith 0 if missing
#                         if 'cocip_atr20' not in df.columns:
#                             df['cocip_atr20'] = 0
#
#                         selected_columns = [
#                             'index', 'time','altitude', 'fuel_flow', 'ei_nox', 'nvpm_ei_n',
#                             'thrust_setting_meem', 'TT3', 'PT3', 'FAR', 'specific_humidity_gsp',
#                             'flight_phase', 'trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection',
#                             'accf_sac_aCCF_O3', 'accf_sac_aCCF_CH4', 'accf_sac_aCCF_CO2', 'ei_co2_conservative',
#                             'ei_co2_optimistic', 'ei_h2o', 'cocip_atr20', 'accf_sac_contrails_atr20', 'accf_sac_aCCF_H2O'
#                         ]
#                         dataframes.append(df[selected_columns].copy())
#
#                 else:
#                     print(f"{trajectory}: Length mismatch across flights! Unique lengths: {unique_lengths}")
#                     print("Skipping this dataset to maintain consistency.")
#
#                 print("\n")
#
# # CONCATENATE ALL FLIGHTS INTO ONE DATAFRAME
# final_df = pd.concat(dataframes, ignore_index=True)
# dt = 60
# final_df['nox'] = final_df['ei_nox']*final_df['fuel_flow']*dt #unit is kg (kg/kg fuel * kg fuel/s * s )
# final_df['nvpm_n'] = final_df['nvpm_ei_n']*final_df['fuel_flow']*dt #unit is # (#/kg fuel * kg fuel/s * s )
# print(f"Collected {len(final_df)} rows from {len(dataframes)} flight data files.")
#
# # Add calculations per waypoint to final_df
# final_df['nox_impact'] = final_df['fuel_flow'] * dt * (final_df['accf_sac_aCCF_O3'] + final_df['accf_sac_aCCF_CH4'] * 1.29) * final_df['ei_nox']
#
# KEROSENE_EI_CO2 = 3.825
# KEROSENE_EI_H2O = 1.237
#
# final_df['co2_impact_cons'] = (
#     final_df['fuel_flow'] * dt *
#     final_df['accf_sac_aCCF_CO2'] *
#     (final_df['ei_co2_conservative'] / KEROSENE_EI_CO2)
# )
#
# final_df['co2_impact_opti'] = (
#     final_df['fuel_flow'] * dt *
#     final_df['accf_sac_aCCF_CO2'] *
#     (final_df['ei_co2_optimistic'] / KEROSENE_EI_CO2)
# )
#
# final_df['h2o_impact'] = (
#     final_df['fuel_flow'] * dt *
#     final_df['accf_sac_aCCF_H2O'] *
#     (final_df['ei_h2o'] / KEROSENE_EI_H2O)
# )
# final_df['contrail_atr20_cocip'] = final_df['cocip_atr20'].fillna(0) if 'cocip_atr20' in final_df.columns else 0
# final_df['contrail_atr20_accf'] = final_df['accf_sac_contrails_atr20']
#
# # Calculate climate impact per waypoint
# final_df['climate_non_co2'] = final_df['nox_impact'] + final_df['h2o_impact'] + final_df['contrail_atr20_cocip']
# final_df['climate_total_cons'] = final_df['climate_non_co2'] + final_df['co2_impact_cons']
# final_df['climate_total_opti'] = final_df['climate_non_co2'] + final_df['co2_impact_opti']
# # print(final_df['fuel_flow'])
# final_df.to_csv('results_check_nox_co2_portions.csv', index=False)

final_df = pd.read_csv('results_check_nox_co2_portions.csv')

# Map engine names for display in the plot
engine_display_names = {
    'GTF1990': 'CFM1990/2008',  # Combined label
    'GTF2000': 'CFM1990/2008',  # Combined label
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

# Use Matplotlib colors for consistency
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF1990': {'marker': '^', 'color': 'tab:blue'},
    'GTF2000': {'marker': '^', 'color': 'tab:blue'},
    'GTF': {'marker': 'o', 'color': 'tab:green'},
    'GTF2035': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}  # Next matplotlib color
}

# X-axis variables and their units for labeling (not titles)
x_vars = {
    'TT3': ('TT3', 'TT3 (K)'),
    'PT3': ('PT3', 'PT3 (bar)'),
    'FAR': ('FAR', 'FAR (-)'),
    'specific_humidity_gsp': ('Specific Humidity', 'Specific Humidity [kg/kg]')
}

# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# axs = axs.flatten()
#
# legend_handles = {}
# seen_cfm = False

# for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
#     ax = axs[i]
#
#     for engine, style in engine_groups.items():
#         subset = final_df[final_df['engine'] == engine]
#         if not subset.empty:
#             ax.scatter(subset[x_var], subset['ei_nox'] * 1000,
#                        label=engine_display_names[engine], marker=style['marker'],
#                        color=style['color'], alpha=0.3, s=10)
#
#             # Collect unique legend handles - combine CFM1990/2000 into one
#             if engine in ['GTF1990', 'GTF2000']:
#                 if not seen_cfm:
#                     legend_handles['CFM1990/2000'] = mlines.Line2D(
#                         [], [], color=style['color'], marker=style['marker'], linestyle='None', markersize=8, label='CFM1990/2000'
#                     )
#                     seen_cfm = True
#             else:
#                 if engine_display_names[engine] not in legend_handles:
#                     legend_handles[engine_display_names[engine]] = mlines.Line2D(
#                         [], [], color=style['color'], marker=style['marker'], linestyle='None', markersize=8, label=engine_display_names[engine]
#                     )
#
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(f'$EI_{{\\mathrm{{NOx}}}}$ (g/ kg Fuel)')
#     ax.set_title(f'$EI_{{\\mathrm{{NOx}}}}$ vs {title_label}')
#
# # Place legend **in the top-left corner** of the **first plot (TT3 vs EI NOx)**
# axs[0].legend(handles=legend_handles.values(), loc='upper left', title="Engine")
#
# plt.tight_layout()
# plt.savefig('results_report/emissions/nox_emissions_tt3_pt3_far_scatter.png', format='png')
# #plt.show()

altitude_plots = {
    'ei_nox': ('$EI_{\\mathrm{NOx}}$', 'kg/kg fuel'),
    'nox': ('NOx', 'kg'),
    'fuel_flow': ('Fuel Flow', 'kg/s'),
    'nvpm_ei_n': ('$EI_{\\mathrm{nvPM}}$', 'particles/kg fuel'),
    'nvpm_n': ('nvPM Number', 'particles'),
    'nox_impact': ('NOx Impact', 'P-ATR20 (K)'),
    'co2_impact_cons': ('CO₂ Impact (Conservative)', 'P-ATR20 (K)')
}

selected_engine = "GTF"
subset = final_df[final_df['engine'] == selected_engine]
style = engine_groups[selected_engine]
label = engine_display_names[selected_engine]

# Individual plots
for var, (title, unit) in altitude_plots.items():
    fig, ax = plt.subplots(figsize=(8, 6))

    if not subset.empty:
        ax.scatter(subset['altitude'], subset[var],
                   label=label,
                   color=style['color'], marker=style['marker'],
                   alpha=0.3, s=10)

    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel(f'{title} ({unit})')
    ax.set_title(f'{title} vs Altitude ({label})')
    ax.grid(True)
    ax.legend(title="Engine", loc='upper right')

plt.show()

# Combined plot: NOx and CO₂ impact vs altitude
fig, ax = plt.subplots(figsize=(8, 6))

if not subset.empty:
    ax.scatter(subset['altitude'], subset['nox_impact'],
               label='NOx Impact', color='tab:blue', alpha=0.4, s=10)

    ax.scatter(subset['altitude'], subset['co2_impact_cons'],
               label='CO₂ Impact (Cons)', color='tab:orange', alpha=0.4, s=10)

ax.set_xlabel('Altitude (m)')
ax.set_ylabel('Climate Impact (unitless)')
ax.set_title(f'NOx and CO₂ Impact vs Altitude ({label})')
ax.grid(True)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
