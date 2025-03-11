import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.optimize import curve_fit

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

# base_path = 'main_results_figures/results'
base_path = 'main_r_f_24_02/results'
# First pass: Determine common altitude range & waypoints per trajectory
altitude_ranges = {}
trajectory_waypoints = {}

for trajectory, trajectory_enabled in trajectories_to_analyze.items():
    if not trajectory_enabled:
        continue

    trajectory_path = os.path.join(base_path, trajectory)
    if not os.path.exists(trajectory_path):
        print(f"Trajectory folder not found: {trajectory_path}")
        continue

    min_altitude_trajectory = float('-inf')
    max_altitude_trajectory = float('inf')
    all_indices = []  # Collect waypoints across all seasons & diurnal conditions

    for folder in os.listdir(trajectory_path):
        for season, season_enabled in seasons_to_analyze.items():
            if not season_enabled or season not in folder:
                continue

            for diurnal, diurnal_enabled in diurnal_to_analyze.items():
                if not diurnal_enabled or diurnal not in folder:
                    continue

                climate_path = os.path.join(trajectory_path, folder, 'climate/mees/era5model')
                if not os.path.exists(climate_path):
                    print(f"Climate folder not found: {climate_path}")
                    continue

                dfs = {}

                for engine, engine_enabled in engine_models_to_analyze.items():
                    if not engine_enabled:
                        continue

                    for saf in ([0] if engine in ["GTF1990", "GTF2000", "GTF"] else saf_levels_to_analyze):
                        for water_injection in (["15"] if engine == "GTF2035_wi" else ["0"]):
                            pattern = f"{engine}_SAF_{saf}_A20N_full_WAR_{water_injection}_climate.csv"
                            file_path = os.path.join(climate_path, pattern)

                            if not os.path.exists(file_path):
                                continue

                            df = pd.read_csv(file_path)
                            dfs[(engine, saf, water_injection)] = df

                if not dfs:
                    continue

                # Find common altitude range
                min_altitude = max(df['altitude'].min() for df in dfs.values())
                max_altitude = min(df['altitude'].max() for df in dfs.values())

                min_altitude_trajectory = max(min_altitude_trajectory, min_altitude)
                max_altitude_trajectory = min(max_altitude_trajectory, max_altitude)

                # Collect waypoints for common indices calculation (only within the altitude range)
                for df in dfs.values():
                    df_filtered = df[(df['altitude'] >= min_altitude_trajectory) & (df['altitude'] <= max_altitude_trajectory)]
                    all_indices.append(set(df_filtered['index']))

    # Determine final common waypoints for this trajectory
    if all_indices:
        common_indices = set.intersection(*all_indices)
        print(f"{trajectory}: Found {len(common_indices)} common waypoints across all seasons & diurnal conditions.")
    else:
        print(f"{trajectory}: No common waypoints found across all conditions.")
        continue  # Skip this trajectory

    # Store results for second pass
    altitude_ranges[trajectory] = (min_altitude_trajectory, max_altitude_trajectory)
    trajectory_waypoints[trajectory] = common_indices  # Store waypoints per trajectory

# Second pass: Load and apply the common altitude & waypoints
dataframes = []

for trajectory, trajectory_enabled in trajectories_to_analyze.items():
    if not trajectory_enabled:
        continue

    trajectory_path = os.path.join(base_path, trajectory)
    if not os.path.exists(trajectory_path):
        continue

    min_altitude_trajectory, max_altitude_trajectory = altitude_ranges[trajectory]
    common_indices = trajectory_waypoints[trajectory]

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
                            file_path = os.path.join(climate_path, pattern)

                            if not os.path.exists(file_path):
                                continue

                            df = pd.read_csv(file_path)

                            # Apply trimming based on the precomputed ranges
                            df = df[(df['altitude'] >= min_altitude_trajectory) & (df['altitude'] <= max_altitude_trajectory) & (df['index'].isin(common_indices))]

                            df['trajectory'] = trajectory
                            df['season'] = season
                            df['diurnal'] = diurnal
                            df['engine'] = engine
                            df['saf_level'] = saf
                            df['water_injection'] = water_injection

                            # Print final length per engine
                            print(f"Final Trim {trajectory} - {season} - {diurnal} - Engine: {engine}, SAF: {saf}, WAR: {water_injection} → Length: {len(df)}")

                            dfs[(engine, saf, water_injection)] = df

                if not dfs:
                    continue

                # Verify if all engines for this flight setup have the same length
                lengths = [len(df) for df in dfs.values()]
                unique_lengths = set(lengths)

                if len(unique_lengths) == 1:
                    print(f"FINAL CHECK: {trajectory} - All flights now have the same length ({unique_lengths.pop()} rows)")

                    # Append the filtered data
                    for df in dfs.values():
                        selected_columns = [
                            'index', 'time', 'fuel_flow', 'fuel_flow_per_engine', 'ei_nox', 'nvpm_ei_n',
                            'thrust_setting_meem', 'TT3', 'PT3', 'FAR', 'specific_humidity_gsp',
                            'flight_phase', 'trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection'
                        ]
                        dataframes.append(df[selected_columns].copy())

                else:
                    print(f"{trajectory}: Length mismatch across flights! Unique lengths: {unique_lengths}")
                    print("Skipping this dataset to maintain consistency.")

                print("\n")

# CONCATENATE ALL FLIGHTS INTO ONE DATAFRAME
final_df = pd.concat(dataframes, ignore_index=True)
final_df['fuel_flow_py'] = 2 * final_df['fuel_flow_per_engine']
final_df.drop(columns=['fuel_flow_per_engine'], inplace=True)

print(f"Collected {len(final_df)} rows from {len(dataframes)} flight data files.")

# Define threshold for fuel flow outliers
outlier_threshold = 2.25

# Identify rows where fuel flow is an outlier
outlier_mask = final_df['fuel_flow'] >= outlier_threshold
outlier_count = outlier_mask.sum()

# If there are outliers, apply interpolation
if outlier_count > 0:
    print(f"Detected {outlier_count} outliers in 'fuel_flow' (>= {outlier_threshold} kg/s). Applying interpolation...")

    # Replace outlier values with NaN
    columns_to_interpolate = [
        'fuel_flow', 'ei_nox', 'nvpm_ei_n', 'thrust_setting_meem',
        'TT3', 'PT3', 'FAR', 'specific_humidity_gsp'
    ]
    final_df.loc[outlier_mask, columns_to_interpolate] = np.nan

    # Apply linear interpolation along the index
    final_df[columns_to_interpolate] = final_df[columns_to_interpolate].interpolate(method='linear')

    # Print results
    print(f"Successfully interpolated missing values for {outlier_count} rows.")
else:
    print("No extreme outliers detected. No interpolation needed.")




# # Map engine names for display in the plot
# engine_display_names = {
#     'GTF1990': 'CFM1990/2000',  # Combined label
#     'GTF2000': 'CFM1990/2000',  # Combined label
#     'GTF': 'GTF',
#     'GTF2035': 'GTF2035',
#     'GTF2035_wi': 'GTF2035WI'
# }
#
# # Use Matplotlib colors for consistency
# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# engine_groups = {
#     'GTF1990': {'marker': '^', 'color': 'tab:blue'},
#     'GTF2000': {'marker': '^', 'color': 'tab:blue'},
#     'GTF': {'marker': 'o', 'color': 'tab:green'},
#     'GTF2035': {'marker': 's', 'color': 'tab:red'},
#     'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}  # Next matplotlib color
# }
#
# # X-axis variables and their units for labeling (not titles)
# x_vars = {
#     'TT3': ('TT3', 'TT3 [K]'),
#     'PT3': ('PT3', 'PT3 [bar]'),
#     'FAR': ('FAR', 'FAR [-]'),
#     'specific_humidity_gsp': ('Specific Humidity', 'Specific Humidity [kg/kg]')
# }
#
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# axs = axs.flatten()
#
# legend_handles = {}
# seen_cfm = False
#
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
#     ax.set_ylabel('EI NOx (g/kg fuel)')
#     ax.set_title(f'EI NOx vs {title_label}')
#
# # Place legend **in the top-left corner** of the **first plot (TT3 vs EI NOx)**
# axs[0].legend(handles=legend_handles.values(), loc='upper left', title="Engine")
#
# plt.tight_layout()
# plt.savefig('results_report/emissions/nox_emissions_tt3_pt3_far_scatter.png', format='png')
# # #plt.show()
#
#
# # Engine display names and colors
# engine_display_names = {
#     'GTF1990': 'CFM1990',
#     'GTF2000': 'CFM2000',
#     'GTF': 'GTF',
#     'GTF2035': 'GTF2035',
#     'GTF2035_wi': 'GTF2035WI'
# }
#
# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# engine_groups = {
#     'GTF1990': {'marker': '^', 'color': 'tab:blue'},
#     'GTF2000': {'marker': '^', 'color': 'tab:orange'},
#     'GTF': {'marker': 'o', 'color': 'tab:green'},
#     'GTF2035': {'marker': 's', 'color': 'tab:red'},
#     'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}
# }
#
# x_vars = {
#     'TT3': ('TT3', 'TT3 [K]'),
#     'PT3': ('PT3', 'PT3 [bar]'),
#     'FAR': ('FAR', 'FAR [-]'),
#     'thrust_setting_meem': ('Thrust Setting', 'Thrust Setting [-]')
# }
#
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# axs = axs.flatten()
#
# legend_handles = {}
#
# for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
#     ax = axs[i]
#
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine)]
#
#         # Exclude SAF != 0
#         if engine in ['GTF2035', 'GTF2035_wi']:
#             subset = subset[subset['saf_level'] == 0]
#
#         if not subset.empty:
#             ax.scatter(subset[x_var], subset['nvpm_ei_n'],
#                        label=engine_display_names[engine], marker=style['marker'],
#                        color=style['color'], alpha=0.3, s=10)
#
#             if engine_display_names[engine] not in legend_handles:
#                 legend_handles[engine_display_names[engine]] = mlines.Line2D(
#                     [], [], color=style['color'], marker=style['marker'], linestyle='None',
#                     markersize=8, label=engine_display_names[engine]
#                 )
#
#     ax.set_xlabel(x_label)
#     ax.set_ylabel('EI nvPM Number (#/kg)')
#     ax.set_title(f'EI nvPM Number vs {title_label}')
#     ax.set_yscale('log')
#
# plt.tight_layout()
# axs[1].legend(handles=legend_handles.values(), loc='lower right', title="Engine")
# plt.subplots_adjust(hspace=0.3, wspace=0.25)
# plt.savefig('results_report/emissions/nvpm_emissions_no_saf_scatter.png', format='png')
# # #plt.show()
#
# # #plt.show()
#
# # Consistent engine display names and colors as in previous plots
# engine_display_names = ['CFM1990', 'CFM2000', 'GTF', 'GTF2035', 'GTF2035WI']
# engine_colors = {
#     'CFM1990': 'tab:blue',
#     'CFM2000': 'tab:orange',
#     'GTF': 'tab:green',
#     'GTF2035': 'tab:red',
#     'GTF2035WI': 'purple'  # Matplotlib default color cycle index 4 is purple
# }
#
# # ICAO data
# thrust_setting_icao = [0.07, 0.3, 0.85, 1]
#
# icao_data = {
#     'GTF': [5.78e15, 3.85e14, 1.60e15, 1.45e15],
#     'GTF2035': [5.78e15, 3.85e14, 1.60e15, 1.45e15],
#     'GTF2035WI': [5.78e15, 3.85e14, 1.60e15, 1.45e15],
#     'CFM1990': [4.43e15, 9.03e15, 2.53e15, 1.62e15],
#     'CFM2000': [7.98e14, 4.85e14, 1.39e15, 1.02e15],
# }
#
# # Plot
# plt.figure(figsize=(8, 6))
# for engine in engine_display_names:
#     plt.plot(thrust_setting_icao, icao_data[engine], marker='o', label=engine, color=engine_colors[engine])
#
# plt.xlabel('Thrust Setting [-]')
# plt.ylabel('EI nvPM Number (#/kg)')
# plt.title('ICAO EI nvPM Number vs Thrust Setting (SLS)')
# plt.yscale('log')
# plt.legend(title='Engine')
# plt.grid(True, which='both', linestyle='--', linewidth=0.7)
# plt.savefig('results_report/emissions/nvpm_emissions_icao_lto.png', format='png')
# # #plt.show()
#
#
#
# # Use Matplotlib default color cycle for consistency
# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# engine_display_names = {
#     'GTF2035_0': 'GTF2035 SAF0',
#     'GTF2035_20': 'GTF2035 SAF20',
#     'GTF2035_100': 'GTF2035 SAF100',
#     'GTF2035_wi_0': 'GTF2035WI SAF0',
#     'GTF2035_wi_20': 'GTF2035WI SAF20',
#     'GTF2035_wi_100': 'GTF2035WI SAF100'
# }
#
# # Colors based on Plot 1 (use same color for GTF2035WI across SAF levels)
# engine_groups = {
#     'GTF2035_0': {'marker': 's', 'color': 'tab:red'},
#     'GTF2035_20': {'marker': 's', 'color': 'tab:pink'},
#     'GTF2035_100': {'marker': 's', 'color': 'tab:grey'},
#     'GTF2035_wi_0': {'marker': 'D', 'color': default_colors[4]},  # Same as in Plot 1
#     'GTF2035_wi_20': {'marker': 'D', 'color': 'tab:olive'},
#     'GTF2035_wi_100': {'marker': 'D', 'color': 'tab:cyan'}
# }
#
# x_vars = {
#     'TT3': ('TT3', 'TT3 [K]'),
#     'PT3': ('PT3', 'PT3 [bar]'),
#     'FAR': ('FAR', 'FAR [-]'),
#     'thrust_setting_meem': ('Thrust Setting', 'Thrust Setting [-]')
# }
#
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# axs = axs.flatten()
#
# legend_handles = {}
#
# for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
#     ax = axs[i]
#
#     for engine, style in engine_groups.items():
#         base_engine = 'GTF2035' if 'wi' not in engine else 'GTF2035_wi'
#         saf_level = int(engine.split('_')[-1])
#
#         subset = final_df[(final_df['engine'] == base_engine) & (final_df['saf_level'] == saf_level)]
#
#         if not subset.empty:
#             ax.scatter(subset[x_var], subset['nvpm_ei_n'],
#                        label=engine_display_names[engine], marker=style['marker'],
#                        color=style['color'], alpha=0.3, s=10)
#
#             if engine_display_names[engine] not in legend_handles:
#                 legend_handles[engine_display_names[engine]] = mlines.Line2D(
#                     [], [], color=style['color'], marker=style['marker'], linestyle='None',
#                     markersize=8, label=engine_display_names[engine]
#                 )
#
#     ax.set_xlabel(x_label)
#     ax.set_ylabel('EI nvPM Number (#/kg)')
#     ax.set_title(f'EI nvPM Number vs {title_label}')
#     ax.set_yscale('log')
#
# plt.tight_layout()
# axs[1].legend(handles=legend_handles.values(), loc='lower right', title="Engine")
# plt.subplots_adjust(hspace=0.3, wspace=0.25)
# plt.savefig('results_report/emissions/nvpm_emissions_saf_scatter.png', format='png')
# # #plt.show()
#
#
# # Engine display names and colors
# engine_display_names = {
#     'GTF1990': 'CFM1990',
#     'GTF2000': 'CFM2000',
#     'GTF': 'GTF',
#     'GTF2035': 'GTF2035',
#     'GTF2035_wi': 'GTF2035WI'
# }
#
# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# engine_groups = {
#     'GTF1990': {'marker': '^', 'color': 'tab:blue'},
#     'GTF2000': {'marker': '^', 'color': 'tab:orange'},
#     'GTF': {'marker': 'o', 'color': 'tab:green'},
#     'GTF2035': {'marker': 's', 'color': 'tab:red'},
#     'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}
# }
#
# x_vars = {
#     'TT3': ('TT3', 'TT3 [K]'),
#     'PT3': ('PT3', 'PT3 [bar]'),
#     'thrust_setting_meem': ('Thrust Setting', 'Thrust Setting [-]'),
#     'FAR': ('FAR', 'FAR [-]')
# }
#
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# axs = axs.flatten()
#
# legend_handles = {}
#
# for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
#     ax = axs[i]
#
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine)]
#
#         # Exclude SAF != 0
#         if engine in ['GTF2035', 'GTF2035_wi']:
#             subset = subset[subset['saf_level'] == 0]
#
#         if not subset.empty:
#             ax.scatter(subset[x_var], subset['fuel_flow'],
#                        label=engine_display_names[engine], marker=style['marker'],
#                        color=style['color'], alpha=0.3, s=10)
#
#             if engine_display_names[engine] not in legend_handles:
#                 legend_handles[engine_display_names[engine]] = mlines.Line2D(
#                     [], [], color=style['color'], marker=style['marker'], linestyle='None',
#                     markersize=8, label=engine_display_names[engine]
#                 )
#
#     ax.set_xlabel(x_label)
#     ax.set_ylabel('Fuel Flow [kg/s]')
#     ax.set_title(f'Fuel Flow vs {title_label}')
#
# plt.tight_layout()
# axs[1].legend(handles=legend_handles.values(), loc='lower right', title="Engine")
# plt.subplots_adjust(hspace=0.3, wspace=0.25)
# # plt.savefig('results_report/emissions/fuel_flow_no_saf_scatter.png', format='png')
# # #plt.show()
#
#
#
# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#
# engine_display_names = {
#     'GTF2035_0': 'GTF2035 SAF0',
#     'GTF2035_20': 'GTF2035 SAF20',
#     'GTF2035_100': 'GTF2035 SAF100',
#     'GTF2035_wi_0': 'GTF2035WI SAF0',
#     'GTF2035_wi_20': 'GTF2035WI SAF20',
#     'GTF2035_wi_100': 'GTF2035WI SAF100'
# }
#
# engine_groups = {
#     'GTF2035_0': {'marker': 's', 'color': 'tab:red'},
#     'GTF2035_20': {'marker': 's', 'color': 'tab:pink'},
#     'GTF2035_100': {'marker': 's', 'color': 'tab:grey'},
#     'GTF2035_wi_0': {'marker': 'D', 'color': default_colors[4]},
#     'GTF2035_wi_20': {'marker': 'D', 'color': 'tab:olive'},
#     'GTF2035_wi_100': {'marker': 'D', 'color': 'tab:cyan'}
# }
#
# x_vars = {
#     'TT3': ('TT3', 'TT3 [K]'),
#     'PT3': ('PT3', 'PT3 [bar]'),
#     'thrust_setting_meem': ('Thrust Setting', 'Thrust Setting [-]'),
#     'FAR': ('FAR', 'FAR [-]')
# }
#
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# axs = axs.flatten()
#
# legend_handles = {}
#
# for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
#     ax = axs[i]
#
#     for engine, style in engine_groups.items():
#         base_engine = 'GTF2035' if 'wi' not in engine else 'GTF2035_wi'
#         saf_level = int(engine.split('_')[-1])
#
#         subset = final_df[(final_df['engine'] == base_engine) & (final_df['saf_level'] == saf_level)]
#
#         if not subset.empty:
#             ax.scatter(subset[x_var], subset['fuel_flow'],
#                        label=engine_display_names[engine], marker=style['marker'],
#                        color=style['color'], alpha=0.3, s=10)
#
#             if engine_display_names[engine] not in legend_handles:
#                 legend_handles[engine_display_names[engine]] = mlines.Line2D(
#                     [], [], color=style['color'], marker=style['marker'], linestyle='None',
#                     markersize=8, label=engine_display_names[engine]
#                 )
#
#     ax.set_xlabel(x_label)
#     ax.set_ylabel('Fuel Flow [kg/s]')
#     ax.set_title(f'Fuel Flow vs {title_label}')
#
# plt.tight_layout()
# axs[1].legend(handles=legend_handles.values(), loc='lower right', title="Engine")
# plt.subplots_adjust(hspace=0.3, wspace=0.25)
# # plt.savefig('results_report/emissions/fuel_flow_saf_scatter.png', format='png')
# # #plt.show()


"""SCATTER PLOT GSP AND PYCONTRAILS"""
# # Filter dataset to include only 'GTF' engine model and specific flight phases
gtf_cruise_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'cruise')].copy()
gtf_climb_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'climb')].copy()
gtf_descent_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'descent')].copy()

# Calculate the ratio of fuel flow between PyContrails and GSP
gtf_cruise_df['fuel_flow_ratio'] = gtf_cruise_df['fuel_flow_py'] / gtf_cruise_df['fuel_flow']
gtf_climb_df['fuel_flow_ratio'] = gtf_climb_df['fuel_flow_py'] / gtf_climb_df['fuel_flow']
gtf_descent_df['fuel_flow_ratio'] = gtf_descent_df['fuel_flow_py'] / gtf_descent_df['fuel_flow']

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Define plot style
color = 'tab:green'
marker = 'o'

# Scatter plot for cruise
axs[0].scatter(gtf_cruise_df['fuel_flow'], gtf_cruise_df['fuel_flow_ratio'],
               label='Cruise', marker=marker, color=color, alpha=0.3, s=10)
axs[0].axhline(y=1, color='r', linestyle='--')
axs[0].set_xlabel("Fuel Flow from GSP (kg/s)")
axs[0].set_ylabel("Fuel Flow Ratio (PyContrails / GSP)")
axs[0].set_title("Cruise")

# Scatter plot for climb
axs[1].scatter(gtf_climb_df['fuel_flow'], gtf_climb_df['fuel_flow_ratio'],
               label='Climb', marker=marker, color=color, alpha=0.3, s=10)
axs[1].axhline(y=1, color='r', linestyle='--')
axs[1].set_xlabel("Fuel Flow from GSP (kg/s)")
axs[1].set_title("Climb")

# Scatter plot for descent
axs[2].scatter(gtf_descent_df['fuel_flow'], gtf_descent_df['fuel_flow_ratio'],
               label='Descent', marker=marker, color=color, alpha=0.3, s=10)
axs[2].axhline(y=1, color='r', linestyle='--')
axs[2].set_xlabel("Fuel Flow from GSP (kg/s)")
axs[2].set_title("Descent")

# Formatting
for ax in axs:
    ax.grid(True)

# Legend (shared)
legend_handles = [
    mlines.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=8, label='GTF'),
    mlines.Line2D([], [], color='r', linestyle='--', label='Equal Fuel Flow')
]

axs[0].legend(handles=legend_handles, loc='upper left', title="Legend")

# Adjust layout
plt.tight_layout()
plt.savefig(f'results_report/fuel_correction/gtf_pycontrails_gsp_correlation.png', format='png')
# #plt.show()


"""CURVE FIT CRUISE"""


# # Define the piecewise linear function with an optimized breakpoint
def piecewise_linear(x, x_break, m1, b1, m2):
    """
    Piecewise linear function with optimized breakpoint.

    Parameters:
    x       - Input fuel flow data (GSP fuel flow)
    x_break - The breakpoint where the slope changes
    m1      - Slope of the first segment (before breakpoint)
    b1      - Intercept of the first segment
    m2      - Slope of the second segment (after breakpoint)

    Returns:
    y       - Output fuel flow (PyContrails fuel flow)
    """
    x = np.asarray(x)  # Ensure x is a NumPy array
    return np.where(x < x_break, m1 * x + b1, m2 * (x - x_break) + (m1 * x_break + b1))


# Extract data from cruise phase
x_data = gtf_cruise_df['fuel_flow'].values  # Fuel Flow from GSP
y_data = gtf_cruise_df['fuel_flow_py'].values  # Fuel Flow from PyContrails

# Provide an initial guess for the breakpoint and slopes
x_break_initial = np.median(x_data)  # Start with the median as a rough estimate
m1_initial = 1.0  # Initial guess for first slope
b1_initial = 0.0  # Initial guess for intercept
m2_initial = 1.0  # Initial guess for second slope

initial_guess = [x_break_initial, m1_initial, b1_initial, m2_initial]

# Fit the piecewise linear model
params, _ = curve_fit(piecewise_linear, x_data, y_data, p0=initial_guess, maxfev=10000)

# Extract optimized parameters
x_break_opt, m1_opt, b1_opt, m2_opt = params

# Generate the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = piecewise_linear(x_fit, *params)

# Apply the correction to GSP fuel flow
gtf_cruise_df['fuel_flow_corrected'] = piecewise_linear(gtf_cruise_df['fuel_flow'], *params)

# Plot the original and corrected data
plt.figure(figsize=(10, 7))

# Scatter plot of original data
plt.scatter(x_data, y_data, color='green', alpha=0.3, label="Original Data")

# Plot fitted piecewise linear curve
plt.plot(x_fit, y_fit, color='blue', linewidth=2, label="Fitted Piecewise Linear Curve")

# Scatter plot of corrected data
plt.scatter(gtf_cruise_df['fuel_flow'], gtf_cruise_df['fuel_flow_corrected'], color='red', alpha=0.3,
            label="Corrected Data")

# Vertical line indicating the optimized breakpoint
plt.axvline(x_break_opt, color='black', linestyle='--', label=f"Optimized Breakpoint: {x_break_opt:.3f}")

# Labels and title
plt.xlabel("Fuel Flow from GSP (kg/s)")
plt.ylabel("Fuel Flow (kg/s)")
plt.title("Fuel Flow Correction - Optimized Piecewise Linear Fit")
plt.legend()
plt.grid(True)
plt.savefig(f'results_report/fuel_correction/gtf_pycontrails_gsp_correlation_cruise.png', format='png')
# Show the plot
# #plt.show()

# Print optimized parameters
print(f"Optimized Breakpoint: {x_break_opt:.3f}")
print(f"Slope before breakpoint: {m1_opt:.3f}")
print(f"Intercept before breakpoint: {b1_opt:.3f}")
print(f"Slope after breakpoint: {m2_opt:.3f}")
"""CURVE FIT CLIMB"""


# Define a polynomial function (second-degree)
def poly_fit(x, a, b, c):
    return a * x**2 + b * x + c

# Extract data from climb phase
x_data = gtf_climb_df['fuel_flow'].values  # Fuel Flow from GSP
y_data = gtf_climb_df['fuel_flow_py'].values  # Fuel Flow from PyContrails

# Fit a second-degree polynomial model
params, _ = curve_fit(poly_fit, x_data, y_data)

# Extract optimized parameters
a_opt, b_opt, c_opt = params

# Generate the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = poly_fit(x_fit, *params)

# ✅ Apply the correction to GSP fuel flow (Renamed to fuel_flow_corrected)
gtf_climb_df['fuel_flow_corrected'] = poly_fit(gtf_climb_df['fuel_flow'], *params)

# 🔹 Plot the original and corrected data
plt.figure(figsize=(10, 7))

# Scatter plot of original data
plt.scatter(x_data, y_data, color='green', alpha=0.3, label="Original Data")

# Plot fitted polynomial curve
plt.plot(x_fit, y_fit, color='blue', linewidth=2, label="Fitted Polynomial Curve")

# Scatter plot of corrected data
plt.scatter(gtf_climb_df['fuel_flow'], gtf_climb_df['fuel_flow_corrected'], color='red', alpha=0.3,
            label="Corrected Data")

# Labels and title
plt.xlabel("Fuel Flow from GSP (kg/s)")
plt.ylabel("Fuel Flow from PyContrails (kg/s)")
plt.title("Fuel Flow Correction - Polynomial Fit for Climb Phase")
plt.legend()
plt.grid(True)
plt.savefig(f'results_report/fuel_correction/gtf_pycontrails_gsp_correlation_climb.png', format='png')
# Show the plot
# #plt.show()

# ✅ Print optimized parameters
print(f"Optimized Polynomial Coefficients: a={a_opt:.5f}, b={b_opt:.5f}, c={c_opt:.5f}")

"""SCATTER PLOT GSP CORRECTED PYCONTRAILS"""

gtf_cruise_df['fuel_flow_ratio_corrected'] = gtf_cruise_df['fuel_flow_py'] / gtf_cruise_df['fuel_flow_corrected']
gtf_climb_df['fuel_flow_ratio_corrected'] = gtf_climb_df['fuel_flow_py'] / gtf_climb_df['fuel_flow_corrected']
# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Define plot style
color = 'tab:green'
marker = 'o'

# Scatter plot for cruise
axs[0].scatter(gtf_cruise_df['fuel_flow_corrected'], gtf_cruise_df['fuel_flow_ratio_corrected'],
               label='Cruise', marker=marker, color=color, alpha=0.3, s=10)
axs[0].axhline(y=1, color='r', linestyle='--')
axs[0].set_xlabel("Fuel Flow from GSP (kg/s)")
axs[0].set_ylabel("Fuel Flow Ratio (PyContrails / GSP)")
axs[0].set_title("Cruise")

# Scatter plot for climb
axs[1].scatter(gtf_climb_df['fuel_flow_corrected'], gtf_climb_df['fuel_flow_ratio_corrected'],
               label='Climb', marker=marker, color=color, alpha=0.3, s=10)
axs[1].axhline(y=1, color='r', linestyle='--')
axs[1].set_xlabel("Fuel Flow from GSP (kg/s)")
axs[1].set_title("Climb")

# Scatter plot for descent
axs[2].scatter(gtf_descent_df['fuel_flow'], gtf_descent_df['fuel_flow_ratio'],
               label='Descent', marker=marker, color=color, alpha=0.3, s=10)
axs[2].axhline(y=1, color='r', linestyle='--')
axs[2].set_xlabel("Fuel Flow from GSP (kg/s)")
axs[2].set_title("Descent")

# Formatting
for ax in axs:
    ax.grid(True)

# Legend (shared)
legend_handles = [
    mlines.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=8, label='GTF'),
    mlines.Line2D([], [], color='r', linestyle='--', label='Equal Fuel Flow')
]

axs[0].legend(handles=legend_handles, loc='upper left', title="Legend")

# Adjust layout
plt.tight_layout()
plt.savefig(f'results_report/fuel_correction/gtf_pycontrails_gsp_correlation_corrected.png', format='png')
# #plt.show()

"""CORRECT THE GTF FUEL FLOW"""

# ================================
# DEFINE CORRECTION FUNCTIONS
# ================================

# Define the piecewise linear function (Used for Cruise Phase)
def piecewise_linear(x, x_break, m1, b1, m2):
    x = np.asarray(x)  # Ensure x is a NumPy array
    return np.where(x < x_break, m1 * x + b1, m2 * (x - x_break) + (m1 * x_break + b1))

# Define a polynomial function (Used for Climb Phase)
def poly_fit(x, a, b, c):
    return a * x**2 + b * x + c

# ================================
# FIT GTF CORRECTIONS (BASELINE)
# ================================

# Initialize `fuel_flow_corrected` in final_df
final_df['fuel_flow_corrected'] = 0.0
# final_df['correction_factor'] = 0.0

# Extract GTF cruise and climb data
gtf_cruise_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'cruise')].copy()
gtf_climb_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'climb')].copy()

# Fit Cruise Correction (Piecewise Linear)
x_data_cruise = gtf_cruise_df['fuel_flow'].values
y_data_cruise = gtf_cruise_df['fuel_flow_py'].values
cruise_params, _ = curve_fit(piecewise_linear, x_data_cruise, y_data_cruise, p0=[np.median(x_data_cruise), 1.0, 0.0, 1.0], maxfev=10000)

# Fit Climb Correction (Polynomial)
x_data_climb = gtf_climb_df['fuel_flow'].values
y_data_climb = gtf_climb_df['fuel_flow_py'].values
climb_params, _ = curve_fit(poly_fit, x_data_climb, y_data_climb)

# Print Optimized Parameters for Verification
print("\n=== Optimized Cruise Fit Parameters ===")
print(f"Break Point (x_break): {cruise_params[0]:.5f}")
print(f"Slope before breakpoint (m1): {cruise_params[1]:.5f}")
print(f"Intercept before breakpoint (b1): {cruise_params[2]:.5f}")
print(f"Slope after breakpoint (m2): {cruise_params[3]:.5f}")

print("\n=== Optimized Climb Fit Parameters ===")
print(f"a (Quadratic term): {climb_params[0]:.5f}")
print(f"b (Linear term): {climb_params[1]:.5f}")
print(f"c (Intercept): {climb_params[2]:.5f}")

# Step 1: Initialize the `fuel_flow_corrected` column with zeros
final_df['fuel_flow_corrected'] = 0.0  # Ensure no pre-existing values interfere

# Step 2: Apply Cruise Correction (Piecewise Linear)
mask_cruise = (final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'cruise')
final_df.loc[mask_cruise, 'fuel_flow_corrected'] = piecewise_linear(
    final_df.loc[mask_cruise, 'fuel_flow'], *cruise_params
)

# Step 3: Apply Climb Correction (Polynomial)
mask_climb = (final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'climb')
final_df.loc[mask_climb, 'fuel_flow_corrected'] = poly_fit(
    final_df.loc[mask_climb, 'fuel_flow'], *climb_params
)

# Step 4: Copy Descent Fuel Flow Directly
mask_descent = (final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'descent')
final_df.loc[mask_descent, 'fuel_flow_corrected'] = final_df.loc[mask_descent, 'fuel_flow']

# Step 5: Ensure no GTF flights have zero fuel_flow_corrected values
gtf_zero_fuel_flow = final_df[(final_df['engine'] == 'GTF') & (final_df['fuel_flow_corrected'] == 0)]
if not gtf_zero_fuel_flow.empty:
    print("\n⚠️ WARNING: Some GTF flights still have zero values after correction!")
    print(gtf_zero_fuel_flow[['trajectory', 'season', 'diurnal', 'flight_phase', 'fuel_flow_corrected']])

# Step 6: Ensure all other engine configs remain zero
other_engines = final_df[final_df['engine'] != 'GTF']
if (other_engines['fuel_flow_corrected'] != 0).any():
    print("\n❌ ERROR: Some non-GTF engines have nonzero corrected fuel flow!")
else:
    print("\n✅ All non-GTF engine configurations correctly have zero fuel_flow_corrected.")

# Step 7: Print summary statistics for validation
print("\nFuel Flow Correction Summary:")
print(final_df.groupby(['engine', 'flight_phase'])[['fuel_flow', 'fuel_flow_corrected']].mean())

"""CHECK BY SCATTERING AGAIN AFTER CHANGING FINAL_DF"""
# # Filter final_df for GTF engine only
# gtf_final_df = final_df[final_df['engine'] == 'GTF']
# gtf_final_df['fuel_flow_ratio_corrected'] = gtf_final_df['fuel_flow_py'] /  gtf_final_df['fuel_flow_corrected']
# # Separate flight phases
# gtf_cruise_df = gtf_final_df[gtf_final_df['flight_phase'] == 'cruise']
# gtf_climb_df = gtf_final_df[gtf_final_df['flight_phase'] == 'climb']
# gtf_descent_df = gtf_final_df[gtf_final_df['flight_phase'] == 'descent']
#
# # Create figure and subplots
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# # Define plot style
# color = 'tab:blue'
# marker = 'o'
#
# # Scatter plot for cruise
# axs[0].scatter(gtf_cruise_df['fuel_flow_corrected'], gtf_cruise_df['fuel_flow_ratio_corrected'],
#                label='Cruise', marker=marker, color=color, alpha=0.3, s=10)
# axs[0].axhline(y=1, color='r', linestyle='--')
# axs[0].set_xlabel("Corrected Fuel Flow (kg/s)")
# axs[0].set_ylabel("Fuel Flow PyContrails (kg/s)")
# axs[0].set_title("Cruise")
#
# # Scatter plot for climb
# axs[1].scatter(gtf_climb_df['fuel_flow_corrected'], gtf_climb_df['fuel_flow_ratio_corrected'],
#                label='Climb', marker=marker, color=color, alpha=0.3, s=10)
# axs[1].axhline(y=1, color='r', linestyle='--')
# axs[1].set_xlabel("Corrected Fuel Flow (kg/s)")
# axs[1].set_title("Climb")
#
# # Scatter plot for descent
# axs[2].scatter(gtf_descent_df['fuel_flow_corrected'], gtf_descent_df['fuel_flow_ratio_corrected'],
#                label='Descent', marker=marker, color=color, alpha=0.3, s=10)
# axs[2].axhline(y=1, color='r', linestyle='--')
# axs[2].set_xlabel("Corrected Fuel Flow (kg/s)")
# axs[2].set_title("Descent")
#
# # Formatting
# for ax in axs:
#     ax.grid(True)
#
# # Legend (shared)
# legend_handles = [
#     mlines.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=8, label='GTF'),
#     mlines.Line2D([], [], color='r', linestyle='--', label='Equal Fuel Flow')
# ]
#
# axs[0].legend(handles=legend_handles, loc='upper left', title="Legend")
#
# # Adjust layout
# plt.tight_layout()
# #plt.show()

"""""PLOT SOME FLIGHT PATHS TO LOOK AT FUEL FLOW CORRECTION"""
#plot some flights
# Get unique combinations of trajectory, season, and diurnal for GTF
unique_missions = final_df[final_df['engine'] == 'GTF'][['trajectory', 'season', 'diurnal']] #.drop_duplicates()

# Select the first 3 unique flights (ensures no duplicate routes are selected)
selected_missions = unique_missions.sample(n=5,random_state=42)

# Loop through each selected mission and create a separate figure
for _, row in selected_missions.iterrows():
    mission = row['trajectory']
    season = row['season']
    diurnal = row['diurnal']

    # Filter for this exact flight (ensuring only one flight per plot)
    mission_df = final_df[
        (final_df['engine'] == 'GTF') &
        (final_df['trajectory'] == mission) &
        (final_df['season'] == season) &
        (final_df['diurnal'] == diurnal)
    ].copy()

    mission_df = mission_df.sort_values(by='index')  # Ensure proper time sequence

    # Create a new figure for each mission
    plt.figure(figsize=(10, 6))

    # Plot PyContrails fuel flow
    plt.plot(mission_df['index'], mission_df['fuel_flow_py'], label='PyContrails', color='tab:blue', linestyle='-', marker='o', markersize=2.5)

    # Plot GSP fuel flow
    plt.plot(mission_df['index'], mission_df['fuel_flow'], label='GSP', color='tab:orange', linestyle='-', marker='o', markersize=2.5)

    # Plot Corrected GSP fuel flow
    plt.plot(mission_df['index'], mission_df['fuel_flow_corrected'], label='GSP Corrected', color='tab:green', linestyle='-', marker='o', markersize=2.5)

    # Set the title with trajectory, season, and diurnal
    plt.title(f"Fuel Flow Comparison - {mission} | Season: {season} | {diurnal.capitalize()}")

    # Formatting
    plt.xlabel("Time in minutes")  # Keep index but rename the label
    plt.ylabel("Fuel Flow (kg/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results_report/fuel_correction/gtf_corrected_mission_{mission}_{season}_{diurnal}.png', format='png')
    # Show the plot for each mission separately
    #plt.show()



"""FUTURE AND SAF FUEL FLOW CORRECTIONS"""


# Ensure we have corrected fuel flow before plotting
if 'fuel_flow_corrected' not in final_df.columns:
    raise ValueError("fuel_flow_corrected column is missing! Ensure corrections are applied before plotting.")

# Toggle to include or exclude the descent phase
include_descent = True  # Set to True to include descent phase in plots

# Filter dataset to include only relevant engine models and SAF levels
gtf_df = final_df[final_df['engine'] == 'GTF'].copy()

gtf2035_saf_dfs = {saf: final_df[(final_df['engine'] == 'GTF2035') & (final_df['saf_level'] == saf)].copy() for saf in
                   [0, 20, 100]}
gtf2035_wi_saf_dfs = {saf: final_df[(final_df['engine'] == 'GTF2035_wi') & (final_df['saf_level'] == saf)].copy() for
                      saf in [0, 20, 100]}

# Flight phases (descent optional)
flight_phases = ['cruise', 'climb']
if include_descent:
    flight_phases.append('descent')

# Phase-based dictionaries
gtf_phase_dfs = {phase: gtf_df[gtf_df['flight_phase'] == phase] for phase in flight_phases}
gtf2035_phase_dfs = {
    saf: {phase: gtf2035_saf_dfs[saf][gtf2035_saf_dfs[saf]['flight_phase'] == phase] for phase in flight_phases} for saf
    in [0, 20, 100]}
gtf2035_wi_phase_dfs = {
    saf: {phase: gtf2035_wi_saf_dfs[saf][gtf2035_wi_saf_dfs[saf]['flight_phase'] == phase] for phase in flight_phases}
    for saf in [0, 20, 100]}

# Define colors and markers per engine
engine_groups = {
    'GTF2035_0': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_20': {'marker': 's', 'color': 'tab:pink'},
    'GTF2035_100': {'marker': 's', 'color': 'tab:grey'},
    'GTF2035_wi_0': {'marker': 'D', 'color': 'tab:blue'},
    'GTF2035_wi_20': {'marker': 'D', 'color': 'tab:olive'},
    'GTF2035_wi_100': {'marker': 'D', 'color': 'tab:cyan'}
}


# Define linear function for fitting
def linear_fit(x, a, b):
    return a * x + b


# Phases to plot separately
phases = ['cruise', 'climb']

for phase in phases:
    # 📌 GTF vs GTF2035 Plot
    plt.figure(figsize=(10, 6))

    for saf in [0, 20, 100]:
        print(f"\n🔍 Processing {phase.capitalize()} Phase | SAF: {saf} (GTF2035)")

        # Merge datasets on the common columns
        merged_df = gtf_phase_dfs[phase].merge(
            gtf2035_phase_dfs[saf][phase],
            on=['trajectory', 'season', 'diurnal', 'flight_phase', 'index'],
            suffixes=('_gtf', f'_gtf2035_{saf}')
        )

        # Extract fuel flow values
        x = merged_df['fuel_flow_gtf'].values  # GTF fuel flow
        y = merged_df[f'fuel_flow_gtf2035_{saf}'].values  # GTF2035 fuel flow

        # Scatter plot
        plt.scatter(x, y, label=f'GTF2035 SAF {saf}', alpha=0.3, s=10)

        # Fit a linear model
        params, _ = curve_fit(linear_fit, x, y)
        a, b = params  # Extract slope and intercept

        # Plot fitted line
        x_fit = np.linspace(min(x), max(x), 100)
        plt.plot(x_fit, linear_fit(x_fit, a, b), linestyle='--',
                 label=f'Fit SAF {saf} (y = {a:.4f}x + {b:.4f})')

    # Add y = x reference line
    plt.plot(x_fit, x_fit, linestyle='-', color='black', alpha=0.7, label='y = x (Reference)')

    # Formatting
    plt.xlabel("GTF Fuel Flow (kg/s)")
    plt.ylabel("GTF2035 Fuel Flow (kg/s)")
    plt.title(f"Correlation Between GTF and GTF2035 ({phase.capitalize()} - Before Correction)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results_report/fuel_correction/gtf_vs_gtf2035_{phase}_correlation.png', format='png')

    # 📌 GTF vs GTF2035WI Plot
    plt.figure(figsize=(10, 6))

    for saf in [0, 20, 100]:
        print(f"\n🔍 Processing {phase.capitalize()} Phase | SAF: {saf} (GTF2035WI)")

        # Merge datasets on the common columns
        merged_df = gtf_phase_dfs[phase].merge(
            gtf2035_wi_phase_dfs[saf][phase],
            on=['trajectory', 'season', 'diurnal', 'flight_phase', 'index'],
            suffixes=('_gtf', f'_gtf2035wi_{saf}')
        )

        # Extract fuel flow values
        x = merged_df['fuel_flow_gtf'].values  # GTF fuel flow
        y = merged_df[f'fuel_flow_gtf2035wi_{saf}'].values  # GTF2035WI fuel flow

        # Scatter plot
        plt.scatter(x, y, label=f'GTF2035WI SAF {saf}', alpha=0.3, s=10)

        # Fit a linear model
        params, _ = curve_fit(linear_fit, x, y)
        a, b = params  # Extract slope and intercept

        # Plot fitted line
        x_fit = np.linspace(min(x), max(x), 100)
        plt.plot(x_fit, linear_fit(x_fit, a, b), linestyle='--',
                 label=f'Fit SAF {saf} (y = {a:.4f}x + {b:.4f})')

    # Add y = x reference line
    plt.plot(x_fit, x_fit, linestyle='-', color='black', alpha=0.7, label='y = x (Reference)')

    # Formatting
    plt.xlabel("GTF Fuel Flow (kg/s)")
    plt.ylabel("GTF2035WI Fuel Flow (kg/s)")
    plt.title(f"Correlation Between GTF and GTF2035WI ({phase.capitalize()} - Before Correction)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results_report/fuel_correction/gtf_vs_gtf2035wi_{phase}_correlation.png', format='png')

# Dictionary to store fit parameters for SAF levels (cruise & climb separately)
gtf2035_fit_params = {'cruise': {}, 'climb': {}}
gtf2035wi_fit_params = {'cruise': {}, 'climb': {}}

# Compute fit parameters separately for cruise & climb
for saf in [0, 20, 100]:
    for phase in ['cruise', 'climb']:  # Only fit for cruise & climb
        print(f"\n🔍 Checking Column Names for GTF2035 SAF {saf} | Phase: {phase}")

        # Merge full dataset for GTF and GTF2035 (for the specific phase)
        merged_df = gtf_phase_dfs[phase].merge(
            gtf2035_phase_dfs[saf][phase].copy(),  # Explicit copy to avoid view warning
            on=['trajectory', 'season', 'diurnal', 'flight_phase', 'index'],
            suffixes=('_gtf', f'_gtf2035_{saf}')
        )

        # print("🔹 Columns in merged_df:")
        # print(merged_df.columns)

        # Fit linear model for GTF2035
        params, _ = curve_fit(linear_fit, merged_df['fuel_flow_gtf'], merged_df[f'fuel_flow_gtf2035_{saf}'])
        gtf2035_fit_params[phase][saf] = params  # Store (a, b) for later use
        print(f"✅ GTF2035 SAF {saf} | Phase: {phase} → Fit: y = {params[0]:.4f}x + {params[1]:.4f}")

        # Merge full dataset for GTF and GTF2035WI (for the specific phase)
        merged_df_wi = gtf_phase_dfs[phase].merge(
            gtf2035_wi_phase_dfs[saf][phase].copy(),  # Explicit copy to avoid view warning
            on=['trajectory', 'season', 'diurnal', 'flight_phase', 'index'],
            suffixes=('_gtf', f'_gtf2035wi_{saf}')
        )

        # Fit linear model for GTF2035WI
        params_wi, _ = curve_fit(linear_fit, merged_df_wi['fuel_flow_gtf'], merged_df_wi[f'fuel_flow_gtf2035wi_{saf}'])
        gtf2035wi_fit_params[phase][saf] = params_wi  # Store (a, b) for later use
        print(f"✅ GTF2035WI SAF {saf} | Phase: {phase} → Fit: y = {params_wi[0]:.4f}x + {params_wi[1]:.4f}")



# Apply the fit separately for cruise & climb
for saf in [0, 20, 100]:
    for phase in ['cruise', 'climb']:
        a_gtf2035, b_gtf2035 = gtf2035_fit_params[phase][saf]
        a_gtf2035wi, b_gtf2035wi = gtf2035wi_fit_params[phase][saf]

        print(f"\n🔍 Applying correction for GTF2035 SAF {saf} | Phase: {phase}")

        # ✅ Align GTF and GTF2035 using merge on trajectory, season, diurnal, and index
        merged_df = gtf2035_phase_dfs[saf][phase].merge(
            gtf_phase_dfs[phase][['trajectory', 'season', 'diurnal', 'index', 'fuel_flow_corrected']],
            on=['trajectory', 'season', 'diurnal', 'index'],  # Match rows properly
            suffixes=('', '_gtf')
        )

        # ✅ Apply correction
        merged_df['fuel_flow_corrected'] = a_gtf2035 * merged_df['fuel_flow_corrected_gtf'] + b_gtf2035

        # ✅ Update the original dataframe with corrected values
        gtf2035_phase_dfs[saf][phase] = merged_df

        print(f"✅ Correction applied successfully for GTF2035 SAF {saf} - {phase}")

        # ✅ Repeat for GTF2035WI
        merged_df_wi = gtf2035_wi_phase_dfs[saf][phase].merge(
            gtf_phase_dfs[phase][['trajectory', 'season', 'diurnal', 'index', 'fuel_flow_corrected']],
            on=['trajectory', 'season', 'diurnal', 'index'],
            suffixes=('', '_gtf')
        )

        merged_df_wi['fuel_flow_corrected'] = a_gtf2035wi * merged_df_wi['fuel_flow_corrected_gtf'] + b_gtf2035wi

        gtf2035_wi_phase_dfs[saf][phase] = merged_df_wi

        print(f"✅ Correction applied successfully for GTF2035WI SAF {saf} - {phase}")

# Copy descent fuel flow directly only if `include_descent` is True
if include_descent:
    for saf in [0, 20, 100]:
        if 'descent' in flight_phases:  # Ensure descent exists
            gtf2035_phase_dfs[saf]['descent'].loc[:, 'fuel_flow_corrected'] = gtf2035_phase_dfs[saf]['descent']['fuel_flow']
            gtf2035_wi_phase_dfs[saf]['descent'].loc[:, 'fuel_flow_corrected'] = gtf2035_wi_phase_dfs[saf]['descent']['fuel_flow']
    print("✅ Descent fuel flow copied directly.")

print("✅ GTF2035 and GTF2035WI fuel_flow_corrected values updated correctly!")


# Function to generate plots
def plot_fuel_flow_comparison(corrected=False):
    suffix = '_corrected' if corrected else ''
    title_suffix = "Corrected" if corrected else "Original"
    xlabel = "Corrected Fuel Flow from GTF (kg/s)" if corrected else "Fuel Flow from GTF (kg/s)"

    num_phases = len(flight_phases)
    fig, axs = plt.subplots(2, num_phases, figsize=(6 * num_phases, 10), sharey=True)

    if num_phases == 1:
        axs = axs.reshape(2, 1)  # Ensure proper indexing when only one phase is present

    merge_cols = ['trajectory', 'season', 'diurnal', 'flight_phase', 'index']

    legend_handles = []  # Store legend items

    for i, phase in enumerate(flight_phases):
        ax = axs[0, i]

        for saf in [0, 20, 100]:
            merged_df = gtf_phase_dfs[phase].merge(
                gtf2035_phase_dfs[saf][phase], on=merge_cols, suffixes=('_gtf', f'_gtf2035_{saf}')
            )

            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()

            # Check for duplicate column names
            duplicate_cols = merged_df.columns[merged_df.columns.duplicated()].tolist()
            if duplicate_cols:
                print(f"🚨 Duplicate column names detected: {duplicate_cols}")

            # Check for missing values
            nan_counts = merged_df.isna().sum()
            print("🚨 NaN count per column after merge:")
            print(nan_counts[nan_counts > 0])

            # Dynamically choose the correct column names
            fuel_flow_column_gtf = 'fuel_flow_corrected_gtf' if corrected else 'fuel_flow_gtf'
            fuel_flow_column_gtf2035 = f'fuel_flow_corrected_gtf2035_{saf}' if corrected else f'fuel_flow_gtf2035_{saf}'

            # Check if the expected columns exist
            if fuel_flow_column_gtf not in merged_df.columns or fuel_flow_column_gtf2035 not in merged_df.columns:
                print(f"🚨 ERROR: Missing expected columns in merged_df for {phase} | SAF {saf}!")
                continue  # Skip plotting if columns are missing

            # Compute fuel flow ratio
            merged_df[f'fuel_flow_ratio{suffix}'] = merged_df[fuel_flow_column_gtf2035] / merged_df[fuel_flow_column_gtf]

            # Check for NaNs in the computed ratio
            if merged_df[f'fuel_flow_ratio{suffix}'].isna().any():
                print(f"⚠️ WARNING: NaN values detected in fuel flow ratio for {phase} | SAF {saf}!")

            # Scatter plot
            ax.scatter(merged_df[fuel_flow_column_gtf], merged_df[f'fuel_flow_ratio{suffix}'],
                       label=f'GTF2035 SAF {saf} / GTF', marker=engine_groups[f'GTF2035_{saf}']['marker'],
                       color=engine_groups[f'GTF2035_{saf}']['color'], alpha=0.3, s=10)

            if i == 0:  # Add legend items only once
                legend_handles.append(mlines.Line2D([], [], color=engine_groups[f'GTF2035_{saf}']['color'],
                                                    marker=engine_groups[f'GTF2035_{saf}']['marker'], linestyle='None',
                                                    markersize=8, label=f'GTF2035 SAF {saf}'))


        ax.axhline(y=1, color='r', linestyle='--', label="Equal Fuel Flow")
        ax.set_title(f"GTF2035 vs GTF - {phase.capitalize()} - {title_suffix}")
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(f"Fuel Flow Ratio (GTF2035 / GTF) - {title_suffix}")

    axs[0, 0].legend(handles=legend_handles, loc='upper left', title="GTF2035 Legend")

    legend_handles = []  # Reset legend items for GTF2035WI

    for i, phase in enumerate(flight_phases):
        ax = axs[1, i]

        for saf in [0, 20, 100]:
            merged_df = gtf_phase_dfs[phase].merge(
                gtf2035_wi_phase_dfs[saf][phase], on=merge_cols, suffixes=('_gtf', f'_gtf2035wi_{saf}')
            )

            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()

            # Check for duplicate column names
            duplicate_cols = merged_df.columns[merged_df.columns.duplicated()].tolist()
            if duplicate_cols:
                print(f"🚨 Duplicate column names detected: {duplicate_cols}")

            nan_counts = merged_df.isna().sum()
            print("🚨 NaN count per column after merge:")
            print(nan_counts[nan_counts > 0])

            fuel_flow_column_gtf2035wi = f'fuel_flow_corrected_gtf2035wi_{saf}' if corrected else f'fuel_flow_gtf2035wi_{saf}'

            if fuel_flow_column_gtf not in merged_df.columns or fuel_flow_column_gtf2035wi not in merged_df.columns:
                print(f"🚨 ERROR: Missing expected columns in merged_df for {phase} | SAF {saf}!")
                continue

            merged_df[f'fuel_flow_ratio{suffix}'] = merged_df[fuel_flow_column_gtf2035wi] / merged_df[fuel_flow_column_gtf]

            if merged_df[f'fuel_flow_ratio{suffix}'].isna().any():
                print(f"⚠️ WARNING: NaN values detected in fuel flow ratio for {phase} | SAF {saf}!")

            ax.scatter(merged_df[fuel_flow_column_gtf], merged_df[f'fuel_flow_ratio{suffix}'],
                       label=f'GTF2035WI SAF {saf} / GTF', marker=engine_groups[f'GTF2035_wi_{saf}']['marker'],
                       color=engine_groups[f'GTF2035_wi_{saf}']['color'], alpha=0.3, s=10)

            if i == 0:  # Add legend items only once
                legend_handles.append(mlines.Line2D([], [], color=engine_groups[f'GTF2035_wi_{saf}']['color'],
                                                    marker=engine_groups[f'GTF2035_wi_{saf}']['marker'],
                                                    linestyle='None',
                                                    markersize=8, label=f'GTF2035WI SAF {saf}'))

        ax.axhline(y=1, color='r', linestyle='--', label="Equal Fuel Flow")
        ax.set_xlabel(xlabel)
        ax.set_title(f"GTF2035WI vs GTF - {phase.capitalize()} - {title_suffix}")



    axs[1, 0].legend(handles=legend_handles, loc='upper left', title="GTF2035WI Legend")

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.savefig(f'results_report/fuel_correction/gtf_vs_gtf2035_correlation_corrected_{corrected}.png', format='png')


    # #plt.show()


# Plot Uncorrected Data
plot_fuel_flow_comparison(corrected=False)

# Plot Corrected Data
plot_fuel_flow_comparison(corrected=True)
#plt.show()


"""APPLY FUEL FLOW CORRECTIONS TO FUTURE ENGINES IN final_df"""

# Ensure `fuel_flow_corrected` exists for GTF before applying to future engines
if 'fuel_flow_corrected' not in final_df.columns:
    raise ValueError("🚨 fuel_flow_corrected column is missing! Ensure corrections are applied for GTF first.")

# Initialize `fuel_flow_corrected` for future engines (set to 0 initially)
final_df.loc[final_df['engine'] != 'GTF', 'fuel_flow_corrected'] = 0.0

# ✅ Extract **only SAF 0's** GTF corrected fuel flow
gtf_fuel_flow_corrected = final_df[
    (final_df['engine'] == 'GTF') &
    (final_df['saf_level'] == 0)  # Always use SAF 0 correction
][['trajectory', 'season', 'diurnal', 'index', 'fuel_flow_corrected']]

# ✅ Merge this into `final_df` to align rows correctly
final_df = final_df.merge(gtf_fuel_flow_corrected, on=['trajectory', 'season', 'diurnal', 'index'], suffixes=('', '_gtf'))

# Apply corrections for each SAF level and phase
for saf in [0, 20, 100]:  # SAF 20 & 100 now correctly use SAF 0 GTF correction
    for phase in ['cruise', 'climb']:
        # Get linear fit parameters for GTF2035 and GTF2035WI
        a_gtf2035, b_gtf2035 = gtf2035_fit_params[phase][saf]
        a_gtf2035wi, b_gtf2035wi = gtf2035wi_fit_params[phase][saf]

        print(f"\n🔍 Applying correction for GTF2035 & GTF2035WI | SAF {saf} | Phase: {phase}")

        # Apply correction for GTF2035
        mask_gtf2035 = (final_df['engine'] == 'GTF2035') & (final_df['saf_level'] == saf) & (final_df['flight_phase'] == phase)
        final_df.loc[mask_gtf2035, 'fuel_flow_corrected'] = (
            a_gtf2035 * final_df.loc[mask_gtf2035, 'fuel_flow_corrected_gtf'] + b_gtf2035
        )

        # ✅ Validation Check for GTF2035
        nonzero_gtf2035 = final_df.loc[mask_gtf2035, 'fuel_flow_corrected'].gt(0).sum()
        nan_gtf2035 = final_df.loc[mask_gtf2035, 'fuel_flow_corrected'].isna().sum()
        print(f"✅ GTF2035 SAF {saf} | {phase} updated in final_df | Nonzero: {nonzero_gtf2035} | NaNs: {nan_gtf2035}")

        # Apply correction for GTF2035WI
        mask_gtf2035wi = (final_df['engine'] == 'GTF2035_wi') & (final_df['saf_level'] == saf) & (final_df['flight_phase'] == phase)
        final_df.loc[mask_gtf2035wi, 'fuel_flow_corrected'] = (
            a_gtf2035wi * final_df.loc[mask_gtf2035wi, 'fuel_flow_corrected_gtf'] + b_gtf2035wi
        )

        # ✅ Validation Check for GTF2035WI
        nonzero_gtf2035wi = final_df.loc[mask_gtf2035wi, 'fuel_flow_corrected'].gt(0).sum()
        nan_gtf2035wi = final_df.loc[mask_gtf2035wi, 'fuel_flow_corrected'].isna().sum()
        print(f"✅ GTF2035WI SAF {saf} | {phase} updated in final_df | Nonzero: {nonzero_gtf2035wi} | NaNs: {nan_gtf2035wi}")

# ✅ Drop temporary GTF corrected column to clean up
final_df.drop(columns=['fuel_flow_corrected_gtf'], inplace=True)

# Copy descent fuel flow directly (if descent phase is included)
if 'descent' in flight_phases:
    for saf in [0, 20, 100]:
        mask_gtf2035_descent = (final_df['engine'] == 'GTF2035') & (final_df['saf_level'] == saf) & (final_df['flight_phase'] == 'descent')
        mask_gtf2035wi_descent = (final_df['engine'] == 'GTF2035_wi') & (final_df['saf_level'] == saf) & (final_df['flight_phase'] == 'descent')

        final_df.loc[mask_gtf2035_descent, 'fuel_flow_corrected'] = final_df.loc[mask_gtf2035_descent, 'fuel_flow']
        final_df.loc[mask_gtf2035wi_descent, 'fuel_flow_corrected'] = final_df.loc[mask_gtf2035wi_descent, 'fuel_flow']

        # ✅ Validation Check for Descent Phase
        nonzero_gtf2035_descent = final_df.loc[mask_gtf2035_descent, 'fuel_flow_corrected'].gt(0).sum()
        nonzero_gtf2035wi_descent = final_df.loc[mask_gtf2035wi_descent, 'fuel_flow_corrected'].gt(0).sum()

        print(f"✅ Descent fuel flow copied for GTF2035 SAF {saf} | Nonzero: {nonzero_gtf2035_descent}")
        print(f"✅ Descent fuel flow copied for GTF2035WI SAF {saf} | Nonzero: {nonzero_gtf2035wi_descent}")

print("\n🚀 Final fuel_flow_corrected values updated in final_df for all future engines!")



"""LETS SEE IF IT WORKED"""

# Function to generate plots from final_df
def plot_fuel_flow_comparison_from_final_df(corrected=False):
    suffix = '_corrected' if corrected else ''
    title_suffix = "Corrected" if corrected else "Original"
    xlabel = "Corrected Fuel Flow from GTF (kg/s)" if corrected else "Fuel Flow from GTF (kg/s)"

    num_phases = len(flight_phases)
    fig, axs = plt.subplots(2, num_phases, figsize=(6 * num_phases, 10), sharey=True)

    if num_phases == 1:
        axs = axs.reshape(2, 1)  # Ensure proper indexing when only one phase is present

    merge_cols = ['trajectory', 'season', 'diurnal', 'flight_phase', 'index']

    legend_handles = []  # Store legend items

    for i, phase in enumerate(flight_phases):
        ax = axs[0, i]

        for saf in [0, 20, 100]:
            print(f"\n🔍 Merging GTF with GTF2035 | Phase: {phase} | SAF: {saf}")

            # Filter GTF and GTF2035 from final_df
            gtf_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == phase)]
            gtf2035_df = final_df[(final_df['engine'] == 'GTF2035') & (final_df['saf_level'] == saf) & (final_df['flight_phase'] == phase)]

            # Merge on common columns
            merged_df = gtf_df.merge(gtf2035_df, on=merge_cols, suffixes=('_gtf', f'_gtf2035_{saf}'))

            # Print columns for debugging
            print("📌 Columns in merged_df:", list(merged_df.columns))

            # Remove duplicate columns
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()

            # Dynamically determine fuel flow columns after merging
            fuel_flow_column_gtf = 'fuel_flow_corrected_gtf' if corrected else 'fuel_flow_gtf'
            fuel_flow_column_gtf2035 = f'fuel_flow_corrected_gtf2035_{saf}' if corrected else f'fuel_flow_gtf2035_{saf}'

            # Check if expected columns exist
            if fuel_flow_column_gtf not in merged_df.columns or fuel_flow_column_gtf2035 not in merged_df.columns:
                print(f"🚨 ERROR: Missing expected columns in merged_df for {phase} | SAF {saf}!")
                continue  # Skip plotting if columns are missing

            # Compute fuel flow ratio
            merged_df[f'fuel_flow_ratio{suffix}'] = merged_df[fuel_flow_column_gtf2035] / merged_df[fuel_flow_column_gtf]

            # Scatter plot
            ax.scatter(merged_df[fuel_flow_column_gtf], merged_df[f'fuel_flow_ratio{suffix}'],
                       label=f'GTF2035 SAF {saf} / GTF', marker=engine_groups[f'GTF2035_{saf}']['marker'],
                       color=engine_groups[f'GTF2035_{saf}']['color'], alpha=0.3, s=10)

            if i == 0:  # Add legend items only once
                legend_handles.append(mlines.Line2D([], [], color=engine_groups[f'GTF2035_{saf}']['color'],
                                                    marker=engine_groups[f'GTF2035_{saf}']['marker'], linestyle='None',
                                                    markersize=8, label=f'GTF2035 SAF {saf}'))

        ax.axhline(y=1, color='r', linestyle='--', label="Equal Fuel Flow")
        ax.set_title(f"GTF2035 vs GTF - {phase.capitalize()} - {title_suffix}")
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(f"Fuel Flow Ratio (GTF2035 / GTF) - {title_suffix}")

    axs[0, 0].legend(handles=legend_handles, loc='upper left', title="GTF2035 Legend")

    legend_handles = []  # Reset legend items for GTF2035WI

    for i, phase in enumerate(flight_phases):
        ax = axs[1, i]

        for saf in [0, 20, 100]:
            print(f"\n🔍 Merging GTF with GTF2035WI | Phase: {phase} | SAF: {saf}")

            # Filter GTF and GTF2035WI from final_df
            gtf_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == phase)]
            gtf2035wi_df = final_df[(final_df['engine'] == 'GTF2035_wi') & (final_df['saf_level'] == saf) & (final_df['flight_phase'] == phase)]

            # Merge on common columns
            merged_df = gtf_df.merge(gtf2035wi_df, on=merge_cols, suffixes=('_gtf', f'_gtf2035wi_{saf}'))

            # Print columns for debugging
            print("📌 Columns in merged_df:", list(merged_df.columns))

            # Remove duplicate columns
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()

            # Dynamically determine fuel flow columns after merging
            fuel_flow_column_gtf2035wi = f'fuel_flow_corrected_gtf2035wi_{saf}' if corrected else f'fuel_flow_gtf2035wi_{saf}'

            if fuel_flow_column_gtf not in merged_df.columns or fuel_flow_column_gtf2035wi not in merged_df.columns:
                print(f"🚨 ERROR: Missing expected columns in merged_df for {phase} | SAF {saf}!")
                continue

            merged_df[f'fuel_flow_ratio{suffix}'] = merged_df[fuel_flow_column_gtf2035wi] / merged_df[fuel_flow_column_gtf]

            ax.scatter(merged_df[fuel_flow_column_gtf], merged_df[f'fuel_flow_ratio{suffix}'],
                       label=f'GTF2035WI SAF {saf} / GTF', marker=engine_groups[f'GTF2035_wi_{saf}']['marker'],
                       color=engine_groups[f'GTF2035_wi_{saf}']['color'], alpha=0.3, s=10)

            if i == 0:  # Add legend items only once
                legend_handles.append(mlines.Line2D([], [], color=engine_groups[f'GTF2035_wi_{saf}']['color'],
                                                    marker=engine_groups[f'GTF2035_wi_{saf}']['marker'],
                                                    linestyle='None',
                                                    markersize=8, label=f'GTF2035WI SAF {saf}'))

        ax.axhline(y=1, color='r', linestyle='--', label="Equal Fuel Flow")
        ax.set_xlabel(xlabel)
        ax.set_title(f"GTF2035WI vs GTF - {phase.capitalize()} - {title_suffix}")

    axs[1, 0].legend(handles=legend_handles, loc='upper left', title="GTF2035WI Legend")

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()


# Plot Uncorrected Data from final_df
plot_fuel_flow_comparison_from_final_df(corrected=False)

# Plot Corrected Data from final_df
plot_fuel_flow_comparison_from_final_df(corrected=True)

"""PLOT SOME FLIGHT PATHS TO LOOK AT FUEL FLOW CORRECTION"""
# Get unique combinations of trajectory, season, and diurnal for GTF
unique_missions = final_df[final_df['engine'] == 'GTF'][['trajectory', 'season', 'diurnal']]

# Select the first 5 unique flights (ensures no duplicate routes are selected)
selected_missions = unique_missions.sample(n=5, random_state=42)

# Loop through each selected mission and create a separate figure
for _, row in selected_missions.iterrows():
    mission = row['trajectory']
    season = row['season']
    diurnal = row['diurnal']

    # Filter for this exact flight (ensuring only one flight per plot)
    gtf_mission_df = final_df[
        (final_df['engine'] == 'GTF') &
        (final_df['trajectory'] == mission) &
        (final_df['season'] == season) &
        (final_df['diurnal'] == diurnal)
    ].copy()

    gtf2035_mission_df = final_df[
        (final_df['engine'] == 'GTF2035') &
        (final_df['saf_level'] == 0) &  # Only SAF=0
        (final_df['trajectory'] == mission) &
        (final_df['season'] == season) &
        (final_df['diurnal'] == diurnal)
    ].copy()

    gtf2035wi_mission_df = final_df[
        (final_df['engine'] == 'GTF2035_wi') &
        (final_df['saf_level'] == 0) &  # Only SAF=0
        (final_df['trajectory'] == mission) &
        (final_df['season'] == season) &
        (final_df['diurnal'] == diurnal)
    ].copy()

    # Sort values to ensure proper time sequence
    gtf_mission_df = gtf_mission_df.sort_values(by='index')
    gtf2035_mission_df = gtf2035_mission_df.sort_values(by='index')
    gtf2035wi_mission_df = gtf2035wi_mission_df.sort_values(by='index')

    # Create a new figure for each mission
    plt.figure(figsize=(10, 6))

    # # Plot PyContrails fuel flow (GTF)
    # plt.plot(gtf_mission_df['index'], gtf_mission_df['fuel_flow_py'], label='GTF PyContrails', color='tab:blue', linestyle='-', marker='o', markersize=2.5)

    # # Plot GSP fuel flow (GTF)
    # plt.plot(gtf_mission_df['index'], gtf_mission_df['fuel_flow'], label='GTF GSP', color='tab:orange', linestyle='-', marker='o', markersize=2.5)

    # Plot Corrected GSP fuel flow (GTF)
    plt.plot(gtf_mission_df['index'], gtf_mission_df['fuel_flow_corrected'], label='GTF GSP Corrected', color='tab:green', linestyle='-', marker='o', markersize=2.5)

    # Plot GTF2035 (SAF=0) Corrected fuel flow
    plt.plot(gtf2035_mission_df['index'], gtf2035_mission_df['fuel_flow_corrected'], label='GTF2035 (SAF=0) Corrected', color='tab:red', linestyle='-', marker='o', markersize=2.5)

    # Plot GTF2035WI (SAF=0) Corrected fuel flow
    plt.plot(gtf2035wi_mission_df['index'], gtf2035wi_mission_df['fuel_flow_corrected'], label='GTF2035WI (SAF=0) Corrected', color='tab:purple', linestyle='-', marker='o', markersize=2.5)

    # Set the title with trajectory, season, and diurnal
    plt.title(f"Fuel Flow Comparison - {mission} | Season: {season} | {diurnal.capitalize()}")

    # Formatting
    plt.xlabel("Time in minutes")  # Keep index but rename the label
    plt.ylabel("Fuel Flow (kg/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results_report/fuel_correction/gtf_future_{mission}_{season}_{diurnal}.png', format='png')

    # Create a new figure for each mission
    plt.figure(figsize=(10, 6))

    # Plot Corrected GSP fuel flow (GTF)
    plt.plot(gtf_mission_df['index'], gtf_mission_df['fuel_flow'], label='GTF GSP Uncorrected',
             color='tab:green', linestyle='-', marker='o', markersize=2.5)

    # Plot GTF2035 (SAF=0) Corrected fuel flow
    plt.plot(gtf2035_mission_df['index'], gtf2035_mission_df['fuel_flow'], label='GTF2035 (SAF=0) Uncorrected',
             color='tab:red', linestyle='-', marker='o', markersize=2.5)

    # Plot GTF2035WI (SAF=0) Corrected fuel flow
    plt.plot(gtf2035wi_mission_df['index'], gtf2035wi_mission_df['fuel_flow'],
             label='GTF2035WI (SAF=0) Uncorrected', color='tab:purple', linestyle='-', marker='o', markersize=2.5)

    # Set the title with trajectory, season, and diurnal
    plt.title(f"Fuel Flow Comparison - {mission} | Season: {season} | {diurnal.capitalize()} Uncorrected")

    # Formatting
    plt.xlabel("Time in minutes")  # Keep index but rename the label
    plt.ylabel("Fuel Flow (kg/s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results_report/fuel_correction/gtf_future_{mission}_{season}_{diurnal}_uncorrected.png', format='png')

    # Show the plot
    # plt.show()
#plt.show()
plt.show()

# Define the filename
csv_filename = "results_df_corrected_fuel_emissions_v2.csv"

# Save final_df as CSV
final_df.to_csv(csv_filename, index=False)

print(f"✅ `final_df` successfully saved as: {csv_filename}")