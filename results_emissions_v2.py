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

base_path = 'main_results_figures/results'

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
                            'index', 'fuel_flow', 'fuel_flow_per_engine', 'ei_nox', 'nvpm_ei_n',
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
# # plt.show()
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
# # plt.show()
#
# # plt.show()
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
# # plt.show()
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
# # plt.show()
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
# # plt.show()
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
# # plt.show()



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
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

# Show the plot
# plt.show()

# Print optimized parameters
print(f"Optimized Breakpoint: {x_break_opt:.3f}")
print(f"Slope before breakpoint: {m1_opt:.3f}")
print(f"Intercept before breakpoint: {b1_opt:.3f}")
print(f"Slope after breakpoint: {m2_opt:.3f}")
"""climb fit!!"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

# Show the plot
# plt.show()

# ✅ Print optimized parameters
print(f"Optimized Polynomial Coefficients: a={a_opt:.5f}, b={b_opt:.5f}, c={c_opt:.5f}")

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
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.lines as mlines

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

# ================================
# APPLY CORRECTIONS TO ALL ENGINES
# ================================

for engine_model in ['GTF', 'GTF2035', 'GTF2035_wi']:
    engine_df = final_df[final_df['engine'] == engine_model].copy()

    # Apply cruise correction
    cruise_mask = engine_df['flight_phase'] == 'cruise'
    engine_df.loc[cruise_mask, 'fuel_flow_corrected'] = piecewise_linear(
        engine_df.loc[cruise_mask, 'fuel_flow'], *cruise_params
    )

    # Apply climb correction
    climb_mask = engine_df['flight_phase'] == 'climb'
    engine_df.loc[climb_mask, 'fuel_flow_corrected'] = poly_fit(
        engine_df.loc[climb_mask, 'fuel_flow'], *climb_params
    )

    # Merge back into final DataFrame
    final_df.loc[engine_df.index, 'fuel_flow_corrected'] = engine_df['fuel_flow_corrected']

print("\n✅ Fuel flow corrections applied to GTF, GTF2035, and GTF2035WI.")

# ================================
# PLOT ORIGINAL & CORRECTED RATIOS
# ================================

# Filter corrected datasets
gtf_cruise_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'cruise')].copy()
gtf_climb_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'climb')].copy()
gtf_descent_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'descent')].copy()

# Compute Original Ratios
gtf_cruise_df['fuel_flow_ratio'] = gtf_cruise_df['fuel_flow_py'] / gtf_cruise_df['fuel_flow']
gtf_climb_df['fuel_flow_ratio'] = gtf_climb_df['fuel_flow_py'] / gtf_climb_df['fuel_flow']
gtf_descent_df['fuel_flow_ratio'] = gtf_descent_df['fuel_flow_py'] / gtf_descent_df['fuel_flow']

# Compute Corrected Ratios
gtf_cruise_df['fuel_flow_ratio_corrected'] = gtf_cruise_df['fuel_flow_py'] / gtf_cruise_df['fuel_flow_corrected']
gtf_climb_df['fuel_flow_ratio_corrected'] = gtf_climb_df['fuel_flow_py'] / gtf_climb_df['fuel_flow_corrected']
gtf_descent_df['fuel_flow_ratio_corrected'] = gtf_descent_df['fuel_flow_py'] / gtf_descent_df['fuel_flow_corrected']

# Create subplots (Before & After Correction)
fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharey=True)

# Scatter plot for cruise (Original)
axs[0, 0].scatter(gtf_cruise_df['fuel_flow'], gtf_cruise_df['fuel_flow_ratio'], color='tab:green', alpha=0.3, s=10)
axs[0, 0].axhline(y=1, color='r', linestyle='--')
axs[0, 0].set_ylabel("Fuel Flow Ratio (PyContrails / GSP)")
axs[0, 0].set_xlabel("Fuel Flow from GSP (kg/s)", labelpad=10)
axs[0, 0].set_title("Cruise - Original", fontsize=14)

# Scatter plot for climb (Original)
axs[0, 1].scatter(gtf_climb_df['fuel_flow'], gtf_climb_df['fuel_flow_ratio'], color='tab:green', alpha=0.3, s=10)
axs[0, 1].axhline(y=1, color='r', linestyle='--')
axs[0, 1].set_xlabel("Fuel Flow from GSP (kg/s)", labelpad=10)
axs[0, 1].set_title("Climb - Original", fontsize=14)

# Scatter plot for descent (Original)
axs[0, 2].scatter(gtf_descent_df['fuel_flow'], gtf_descent_df['fuel_flow_ratio'], color='tab:green', alpha=0.3, s=10)
axs[0, 2].axhline(y=1, color='r', linestyle='--')
axs[0, 2].set_xlabel("Fuel Flow from GSP (kg/s)", labelpad=10)
axs[0, 2].set_title("Descent - Original", fontsize=14)

# Scatter plot for cruise (Corrected)
axs[1, 0].scatter(gtf_cruise_df['fuel_flow_corrected'], gtf_cruise_df['fuel_flow_ratio_corrected'], color='tab:blue', alpha=0.3, s=10)
axs[1, 0].axhline(y=1, color='r', linestyle='--')
axs[1, 0].set_xlabel("Corrected Fuel Flow from GSP (kg/s)", labelpad=10)
axs[1, 0].set_ylabel("Fuel Flow Ratio (PyContrails / Corrected GSP)")
axs[1, 0].set_title("Cruise - Corrected", fontsize=14)

# Scatter plot for climb (Corrected)
axs[1, 1].scatter(gtf_climb_df['fuel_flow_corrected'], gtf_climb_df['fuel_flow_ratio_corrected'], color='tab:blue', alpha=0.3, s=10)
axs[1, 1].axhline(y=1, color='r', linestyle='--')
axs[1, 1].set_xlabel("Corrected Fuel Flow from GSP (kg/s)", labelpad=10)
axs[1, 1].set_title("Climb - Corrected", fontsize=14)

# Scatter plot for descent (Corrected)
axs[1, 2].scatter(gtf_descent_df['fuel_flow'], gtf_descent_df['fuel_flow_ratio'], color='tab:blue', alpha=0.3, s=10)
axs[1, 2].axhline(y=1, color='r', linestyle='--')
axs[1, 2].set_xlabel("Fuel Flow from GSP (kg/s)", labelpad=10)
axs[1, 2].set_title("Descent", fontsize=14)

# Formatting
for ax in axs.flatten():
    ax.grid(True)

# Adjust layout to prevent overlap
plt.subplots_adjust(hspace=0.5, top=0.90, bottom=0.10)  # Increase spacing

# Set a global title
plt.suptitle("Fuel Flow Ratio Before and After Correction", fontsize=16, y=1.00)

# plt.tight_layout()
# plt.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# # ================================
# # DEFINE CORRECTION FUNCTIONS
# # ================================
#
# # Initialize fuel_flow_corrected with the original fuel_flow values
# final_df['fuel_flow_corrected'] = 0.0
#
# # Piecewise Linear Function (Used for Cruise Phase)
# def piecewise_linear(x, x_break, m1, b1, m2):
#     """
#     Piecewise linear function with an optimized breakpoint.
#
#     Parameters:
#     x       - Fuel Flow from GSP
#     x_break - Optimized breakpoint where slope changes
#     m1      - Slope of first segment (before breakpoint)
#     b1      - Intercept of first segment
#     m2      - Slope of second segment (after breakpoint)
#
#     Returns:
#     y       - Corrected Fuel Flow from PyContrails
#     """
#     x = np.asarray(x)  # Ensure x is a NumPy array
#     return np.where(x < x_break, m1 * x + b1, m2 * (x - x_break) + (m1 * x_break + b1))
#
# # Polynomial Function (Used for Climb Phase)
# def poly_fit(x, a, b, c):
#     """
#     Second-degree polynomial function for curve fitting.
#
#     Parameters:
#     x - Fuel Flow from GSP
#     a, b, c - Optimized polynomial coefficients
#
#     Returns:
#     y - Corrected Fuel Flow from PyContrails
#     """
#     return a * x**2 + b * x + c
#
# # ================================
# # APPLY CORRECTION TO GTF (BASELINE CURVE FITTING)
# # ================================
#
# def apply_fuel_flow_correction(df, phase, correction_function, initial_guess):
#     """
#     Applies fuel flow correction using the specified function.
#
#     Parameters:
#     df: DataFrame - The input DataFrame containing 'fuel_flow' and 'fuel_flow_py'
#     phase: str - Flight phase ('cruise' or 'climb')
#     correction_function: function - The function to apply (piecewise_linear or poly_fit)
#     initial_guess: list - Initial guess for curve fitting parameters
#
#     Returns:
#     df: DataFrame - Updated DataFrame with 'fuel_flow_corrected'
#     params: list - Optimized parameters for correction function
#     """
#     # Extract only the relevant phase data
#     phase_df = df[df['flight_phase'] == phase].copy()
#
#     # Check if phase data exists
#     if phase_df.empty:
#         print(f"No data for phase: {phase}")
#         return df, None  # Return original if no data exists
#
#     # Extract X (GSP Fuel Flow) and Y (PyContrails Fuel Flow)
#     x_data = phase_df['fuel_flow'].values
#     y_data = phase_df['fuel_flow_py'].values
#
#     # Fit the selected correction model
#     params, _ = curve_fit(correction_function, x_data, y_data, p0=initial_guess, maxfev=10000)
#
#     # # Ensure `fuel_flow_corrected` exists in final_df before updating
#     # if 'fuel_flow_corrected' not in final_df.columns:
#     #     final_df.loc[:, 'fuel_flow_corrected'] = final_df['fuel_flow']  # Ensure column assignment
#
#     # Apply correction to fuel flow and store results
#     phase_df['fuel_flow_corrected'] = correction_function(phase_df['fuel_flow'], *params)
#
#     # Merge corrected values back into main DataFrame
#     df.update(phase_df[['fuel_flow_corrected']])
#
#     return df, params  # Return both the updated DataFrame and the fitted parameters
#
# # ================================
# # 1. APPLY FITTING TO GTF (BASELINE)
# # ================================
#
# # Initial guesses for fitting
# initial_guess_cruise = [np.median(final_df['fuel_flow']), 1.0, 0.0, 1.0]  # Cruise (Piecewise Linear)
# initial_guess_climb = [1.0, 1.0, 0.0]  # Climb (Polynomial)
#
# # Apply cruise correction to GTF and store cruise fit parameters
# final_df, cruise_params = apply_fuel_flow_correction(final_df[final_df['engine'] == 'GTF'], 'cruise', piecewise_linear, initial_guess_cruise)
#
# # Apply climb correction to GTF and store climb fit parameters
# final_df, climb_params = apply_fuel_flow_correction(final_df[final_df['engine'] == 'GTF'], 'climb', poly_fit, initial_guess_climb)
#
# # ================================
# # 2. APPLY GTF FITS TO GTF2035 & GTF2035WI (NO REFITTING)
# # ================================
#
#
# # # Apply corrections for GTF2035 and GTF2035WI using precomputed GTF parameters
# # for engine_model in ['GTF2035', 'GTF2035_wi']:
# #     engine_df = final_df[final_df['engine'] == engine_model].copy()
# #
# #     # Apply cruise correction where flight_phase == 'cruise'
# #     cruise_mask = engine_df['flight_phase'] == 'cruise'
# #     engine_df.loc[cruise_mask, 'fuel_flow_corrected'] = piecewise_linear(
# #         engine_df.loc[cruise_mask, 'fuel_flow'], *cruise_params
# #     )
# #
# #     # Apply climb correction where flight_phase == 'climb'
# #     climb_mask = engine_df['flight_phase'] == 'climb'
# #     engine_df.loc[climb_mask, 'fuel_flow_corrected'] = poly_fit(
# #         engine_df.loc[climb_mask, 'fuel_flow'], *climb_params
# #     )
# #
# #     # Merge back into final DataFrame
# #     final_df.update(engine_df[['fuel_flow_corrected']])
# #
# # # Print confirmation
# # print("Fuel flow corrections applied to GTF, GTF2035, and GTF2035WI using GTF-derived curve fits.")
# # # print("Columns in final_df:", final_df.columns)
# # # print("Columns in gtf_cruise_df:", gtf_cruise_df.columns)
#
# # ================================
# # PLOT ORIGINAL FUEL FLOW RATIO
# # ================================
#
# # Filter dataset to include only 'GTF' engine model and specific flight phases
# gtf_cruise_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'cruise')].copy()
# gtf_climb_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'climb')].copy()
# gtf_descent_df = final_df[(final_df['engine'] == 'GTF') & (final_df['flight_phase'] == 'descent')].copy()
#
# # Calculate the ratio of fuel flow between PyContrails and GSP
# gtf_cruise_df['fuel_flow_ratio'] = gtf_cruise_df['fuel_flow_py'] / gtf_cruise_df['fuel_flow']
# gtf_climb_df['fuel_flow_ratio'] = gtf_climb_df['fuel_flow_py'] / gtf_climb_df['fuel_flow']
# gtf_descent_df['fuel_flow_ratio'] = gtf_descent_df['fuel_flow_py'] / gtf_descent_df['fuel_flow']
#
# # Create subplots
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# # Define plot style
# color = 'tab:green'
# marker = 'o'
#
# # Scatter plot for cruise
# axs[0].scatter(gtf_cruise_df['fuel_flow'], gtf_cruise_df['fuel_flow_ratio'],
#                label='Cruise', marker=marker, color=color, alpha=0.3, s=10)
# axs[0].axhline(y=1, color='r', linestyle='--')
# axs[0].set_xlabel("Fuel Flow from GSP (kg/s)")
# axs[0].set_ylabel("Fuel Flow Ratio (PyContrails / GSP)")
# axs[0].set_title("Cruise")
#
# # Scatter plot for climb
# axs[1].scatter(gtf_climb_df['fuel_flow'], gtf_climb_df['fuel_flow_ratio'],
#                label='Climb', marker=marker, color=color, alpha=0.3, s=10)
# axs[1].axhline(y=1, color='r', linestyle='--')
# axs[1].set_xlabel("Fuel Flow from GSP (kg/s)")
# axs[1].set_title("Climb")
#
# # Scatter plot for descent
# axs[2].scatter(gtf_descent_df['fuel_flow'], gtf_descent_df['fuel_flow_ratio'],
#                label='Descent', marker=marker, color=color, alpha=0.3, s=10)
# axs[2].axhline(y=1, color='r', linestyle='--')
# axs[2].set_xlabel("Fuel Flow from GSP (kg/s)")
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
# plt.show()
#
# # ================================
# # PLOT CORRECTED FUEL FLOW RATIO
# # ================================
#
# # Calculate the ratio using the corrected fuel flow
# gtf_cruise_df['fuel_flow_ratio_corrected'] = gtf_cruise_df['fuel_flow_py'] / gtf_cruise_df['fuel_flow_corrected']
# gtf_climb_df['fuel_flow_ratio_corrected'] = gtf_climb_df['fuel_flow_py'] / gtf_climb_df['fuel_flow_corrected']
# gtf_descent_df['fuel_flow_ratio_corrected'] = gtf_descent_df['fuel_flow_py'] / gtf_descent_df['fuel_flow_corrected']
#
# # Create new figure for corrected data
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# # Scatter plot for cruise
# axs[0].scatter(gtf_cruise_df['fuel_flow_corrected'], gtf_cruise_df['fuel_flow_ratio_corrected'],
#                label='Cruise (Corrected)', marker=marker, color='tab:blue', alpha=0.3, s=10)
# axs[0].axhline(y=1, color='r', linestyle='--')
# axs[0].set_xlabel("Corrected Fuel Flow from GSP (kg/s)")
# axs[0].set_ylabel("Fuel Flow Ratio (PyContrails / Corrected GSP)")
# axs[0].set_title("Cruise - Corrected")
#
# # Scatter plot for climb
# axs[1].scatter(gtf_climb_df['fuel_flow_corrected'], gtf_climb_df['fuel_flow_ratio_corrected'],
#                label='Climb (Corrected)', marker=marker, color='tab:blue', alpha=0.3, s=10)
# axs[1].axhline(y=1, color='r', linestyle='--')
# axs[1].set_xlabel("Corrected Fuel Flow from GSP (kg/s)")
# axs[1].set_title("Climb - Corrected")
#
# # Scatter plot for descent
# axs[2].scatter(gtf_descent_df['fuel_flow_corrected'], gtf_descent_df['fuel_flow_ratio_corrected'],
#                label='Descent (Corrected)', marker=marker, color='tab:blue', alpha=0.3, s=10)
# axs[2].axhline(y=1, color='r', linestyle='--')
# axs[2].set_xlabel("Corrected Fuel Flow from GSP (kg/s)")
# axs[2].set_title("Descent - Corrected")
#
# plt.tight_layout()
# plt.show()

#
#
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
#
# # Filter dataset to include only relevant engine models and SAF levels
# gtf_df = final_df[final_df['engine'] == 'GTF'].copy()
#
# # GTF2035 with SAF variations
# gtf2035_saf_dfs = {
#     saf: final_df[(final_df['engine'] == 'GTF2035') & (final_df['saf_level'] == saf)].copy()
#     for saf in [0, 20, 100]
# }
#
# # GTF2035WI with SAF variations
# gtf2035_wi_saf_dfs = {
#     saf: final_df[(final_df['engine'] == 'GTF2035_wi') & (final_df['saf_level'] == saf)].copy()
#     for saf in [0, 20, 100]
# }
#
# # Flight phases (keeping 'descent' without renaming)
# flight_phases = ['cruise', 'climb', 'descent']
#
# # Split data into phases
# gtf_phase_dfs = {phase: gtf_df[gtf_df['flight_phase'] == phase] for phase in flight_phases}
# gtf2035_phase_dfs = {
#     saf: {phase: gtf2035_saf_dfs[saf][gtf2035_saf_dfs[saf]['flight_phase'] == phase] for phase in flight_phases} for saf
#     in [0, 20, 100]}
# gtf2035_wi_phase_dfs = {
#     saf: {phase: gtf2035_wi_saf_dfs[saf][gtf2035_wi_saf_dfs[saf]['flight_phase'] == phase] for phase in flight_phases}
#     for saf in [0, 20, 100]}
#
# # Define colors and markers per engine
# engine_groups = {
#     'GTF2035_0': {'marker': 's', 'color': 'tab:red'},
#     'GTF2035_20': {'marker': 's', 'color': 'tab:pink'},
#     'GTF2035_100': {'marker': 's', 'color': 'tab:grey'},
#     'GTF2035_wi_0': {'marker': 'D', 'color': 'tab:blue'},
#     'GTF2035_wi_20': {'marker': 'D', 'color': 'tab:olive'},
#     'GTF2035_wi_100': {'marker': 'D', 'color': 'tab:cyan'}
# }
#
# # Create a single figure with two rows (one for GTF2035 vs GTF, one for GTF2035WI vs GTF)
# fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
#
# # Merge columns including index to ensure correct waypoint comparison
# merge_cols = ['trajectory', 'season', 'diurnal', 'flight_phase', 'index']
#
# # Plot GTF2035 vs GTF (Fuel Flow Ratio) with GTF Fuel Flow on x-axis
# for i, phase in enumerate(flight_phases):
#     ax = axs[0, i]
#
#     for saf in [0, 20, 100]:
#         # Merge datasets ensuring waypoints align (index-based merging)
#         merged_df = gtf_phase_dfs[phase].merge(gtf2035_phase_dfs[saf][phase], on=merge_cols,
#                                                suffixes=('_gtf', f'_gtf2035_{saf}'))
#
#         # Compute fuel flow ratio
#         merged_df['fuel_flow_ratio'] = merged_df[f'fuel_flow_gtf2035_{saf}'] / merged_df['fuel_flow_gtf']
#
#         # Scatter plot with GTF fuel flow on x-axis
#         ax.scatter(merged_df['fuel_flow_gtf'], merged_df['fuel_flow_ratio'],
#                    label=f'GTF2035 SAF {saf} / GTF', marker=engine_groups[f'GTF2035_{saf}']['marker'],
#                    color=engine_groups[f'GTF2035_{saf}']['color'], alpha=0.3, s=10)
#
#     ax.axhline(y=1, color='r', linestyle='--', label="Equal Fuel Flow")
#     ax.set_title(f"GTF2035 vs GTF - {phase.capitalize()}")
#     if i == 0:
#         ax.set_ylabel("Fuel Flow Ratio (GTF2035 / GTF)")
#     else:
#         ax.set_ylabel("")
#
# # Remove X-axis labels from the top row
# for ax in axs[0]:
#     ax.set_xlabel("")
#
# axs[0, 0].legend(handles=[
#     mlines.Line2D([], [], color=engine_groups['GTF2035_0']['color'], marker=engine_groups['GTF2035_0']['marker'],
#                   linestyle='None', markersize=8, label="GTF2035 SAF 0"),
#     mlines.Line2D([], [], color=engine_groups['GTF2035_20']['color'], marker=engine_groups['GTF2035_20']['marker'],
#                   linestyle='None', markersize=8, label="GTF2035 SAF 20"),
#     mlines.Line2D([], [], color=engine_groups['GTF2035_100']['color'], marker=engine_groups['GTF2035_100']['marker'],
#                   linestyle='None', markersize=8, label="GTF2035 SAF 100"),
#     mlines.Line2D([], [], color='r', linestyle='--', label="Equal Fuel Flow")
# ], loc='upper left', title="Legend")
#
# # Plot GTF2035WI vs GTF (Fuel Flow Ratio) with GTF Fuel Flow on x-axis
# for i, phase in enumerate(flight_phases):
#     ax = axs[1, i]
#
#     for saf in [0, 20, 100]:
#         # Merge datasets ensuring waypoints align (index-based merging)
#         merged_df = gtf_phase_dfs[phase].merge(gtf2035_wi_phase_dfs[saf][phase], on=merge_cols,
#                                                suffixes=('_gtf', f'_gtf2035wi_{saf}'))
#
#         # Compute fuel flow ratio
#         merged_df['fuel_flow_ratio'] = merged_df[f'fuel_flow_gtf2035wi_{saf}'] / merged_df['fuel_flow_gtf']
#
#         # Scatter plot with GTF fuel flow on x-axis
#         ax.scatter(merged_df['fuel_flow_gtf'], merged_df['fuel_flow_ratio'],
#                    label=f'GTF2035WI SAF {saf} / GTF', marker=engine_groups[f'GTF2035_wi_{saf}']['marker'],
#                    color=engine_groups[f'GTF2035_wi_{saf}']['color'], alpha=0.3, s=10)
#
#     ax.axhline(y=1, color='r', linestyle='--', label="Equal Fuel Flow")
#     ax.set_xlabel("Fuel Flow from GTF (kg/s)")
#     ax.set_title(f"GTF2035WI vs GTF - {phase.capitalize()}")
#     if i == 0:
#         ax.set_ylabel("Fuel Flow Ratio (GTF2035WI / GTF)")
#     else:
#         ax.set_ylabel("")
#
# axs[1, 0].legend(handles=[
#     mlines.Line2D([], [], color=engine_groups['GTF2035_wi_0']['color'], marker=engine_groups['GTF2035_wi_0']['marker'],
#                   linestyle='None', markersize=8, label="GTF2035WI SAF 0"),
#     mlines.Line2D([], [], color=engine_groups['GTF2035_wi_20']['color'],
#                   marker=engine_groups['GTF2035_wi_20']['marker'], linestyle='None', markersize=8,
#                   label="GTF2035WI SAF 20"),
#     mlines.Line2D([], [], color=engine_groups['GTF2035_wi_100']['color'],
#                   marker=engine_groups['GTF2035_wi_100']['marker'], linestyle='None', markersize=8,
#                   label="GTF2035WI SAF 100"),
#     mlines.Line2D([], [], color='r', linestyle='--', label="Equal Fuel Flow")
# ], loc='upper left', title="Legend")
#
# # Adjust layout and increase vertical spacing
# plt.subplots_adjust(hspace=0.4)
# plt.tight_layout()
# plt.show()
#
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
#
# # Filter dataset to include only relevant engine models and SAF levels
# gtf_df = final_df[final_df['engine'] == 'GTF'].copy()
#
# # GTF2035 with SAF variations
# gtf2035_saf_dfs = {
#     saf: final_df[(final_df['engine'] == 'GTF2035') & (final_df['saf_level'] == saf)].copy()
#     for saf in [0, 20, 100]
# }
#
# # GTF2035WI with SAF variations
# gtf2035_wi_saf_dfs = {
#     saf: final_df[(final_df['engine'] == 'GTF2035_wi') & (final_df['saf_level'] == saf)].copy()
#     for saf in [0, 20, 100]
# }
#
# # Flight phases (keeping 'descent' without renaming)
# flight_phases = ['cruise', 'climb', 'descent']
#
# # Split data into phases
# gtf_phase_dfs = {phase: gtf_df[gtf_df['flight_phase'] == phase] for phase in flight_phases}
# gtf2035_phase_dfs = {
#     saf: {phase: gtf2035_saf_dfs[saf][gtf2035_saf_dfs[saf]['flight_phase'] == phase] for phase in flight_phases}
#     for saf in [0, 20, 100]
# }
# gtf2035_wi_phase_dfs = {
#     saf: {phase: gtf2035_wi_saf_dfs[saf][gtf2035_wi_saf_dfs[saf]['flight_phase'] == phase] for phase in flight_phases}
#     for saf in [0, 20, 100]
# }
#
# # Define colors and markers per engine
# engine_groups = {
#     'GTF2035_0': {'marker': 's', 'color': 'tab:red'},
#     'GTF2035_20': {'marker': 's', 'color': 'tab:pink'},
#     'GTF2035_100': {'marker': 's', 'color': 'tab:grey'},
#     'GTF2035_wi_0': {'marker': 'D', 'color': 'tab:blue'},
#     'GTF2035_wi_20': {'marker': 'D', 'color': 'tab:olive'},
#     'GTF2035_wi_100': {'marker': 'D', 'color': 'tab:cyan'}
# }
#
# # Merge columns including index to ensure correct waypoint comparison
# merge_cols = ['trajectory', 'season', 'diurnal', 'flight_phase', 'index']
#
# # Function to plot before and after correction
# def plot_fuel_flow_correction(corrected=False):
#     fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
#
#     ratio_column = 'fuel_flow_ratio_corrected' if corrected else 'fuel_flow_ratio'
#     x_axis_label = "Fuel Flow from GTF (kg/s) Corrected" if corrected else "Fuel Flow from GTF (kg/s)"
#
#     # Plot GTF2035 vs GTF
#     for i, phase in enumerate(flight_phases):
#         ax = axs[0, i]
#
#         for saf in [0, 20, 100]:
#             merged_df = gtf_phase_dfs[phase].merge(gtf2035_phase_dfs[saf][phase], on=merge_cols,
#                                                    suffixes=('_gtf', f'_gtf2035_{saf}'))
#
#             # Compute fuel flow ratio before or after correction
#             merged_df[ratio_column] = merged_df[f'fuel_flow_gtf2035_{saf}'] / merged_df[
#                 'fuel_flow_corrected_gtf' if corrected else 'fuel_flow_gtf']
#
#             ax.scatter(merged_df['fuel_flow_gtf'], merged_df[ratio_column],
#                        label=f'GTF2035 SAF {saf} / GTF',
#                        marker=engine_groups[f'GTF2035_{saf}']['marker'],
#                        color=engine_groups[f'GTF2035_{saf}']['color'], alpha=0.3, s=10)
#
#         ax.axhline(y=1, color='r', linestyle='--', label="Equal Fuel Flow")
#         ax.set_title(f"GTF2035 vs GTF - {phase.capitalize()}")
#         if i == 0:
#             ax.set_ylabel(f"Fuel Flow Ratio (GTF2035 / GTF) {'Corrected' if corrected else ''}")
#         else:
#             ax.set_ylabel("")
#
#     for ax in axs[0]:
#         ax.set_xlabel("")
#
#     axs[0, 0].legend(handles=[
#         mlines.Line2D([], [], color=engine_groups['GTF2035_0']['color'], marker=engine_groups['GTF2035_0']['marker'],
#                       linestyle='None', markersize=8, label="GTF2035 SAF 0"),
#         mlines.Line2D([], [], color=engine_groups['GTF2035_20']['color'], marker=engine_groups['GTF2035_20']['marker'],
#                       linestyle='None', markersize=8, label="GTF2035 SAF 20"),
#         mlines.Line2D([], [], color=engine_groups['GTF2035_100']['color'],
#                       marker=engine_groups['GTF2035_100']['marker'], linestyle='None', markersize=8,
#                       label="GTF2035 SAF 100"),
#         mlines.Line2D([], [], color='r', linestyle='--', label="Equal Fuel Flow")
#     ], loc='upper left', title="Legend")
#
#     # Plot GTF2035WI vs GTF
#     for i, phase in enumerate(flight_phases):
#         ax = axs[1, i]
#
#         for saf in [0, 20, 100]:
#             merged_df = gtf_phase_dfs[phase].merge(gtf2035_wi_phase_dfs[saf][phase], on=merge_cols,
#                                                    suffixes=('_gtf', f'_gtf2035wi_{saf}'))
#
#             # Compute fuel flow ratio before or after correction
#             merged_df[ratio_column] = merged_df[f'fuel_flow_gtf2035wi_{saf}'] / merged_df[
#                 'fuel_flow_corrected_gtf' if corrected else 'fuel_flow_gtf']
#
#             ax.scatter(merged_df['fuel_flow_gtf'], merged_df[ratio_column],
#                        label=f'GTF2035WI SAF {saf} / GTF',
#                        marker=engine_groups[f'GTF2035_wi_{saf}']['marker'],
#                        color=engine_groups[f'GTF2035_wi_{saf}']['color'], alpha=0.3, s=10)
#
#         ax.axhline(y=1, color='r', linestyle='--', label="Equal Fuel Flow")
#         ax.set_xlabel(x_axis_label)
#         ax.set_title(f"GTF2035WI vs GTF - {phase.capitalize()}")
#         if i == 0:
#             ax.set_ylabel(f"Fuel Flow Ratio (GTF2035WI / GTF) {'Corrected' if corrected else ''}")
#         else:
#             ax.set_ylabel("")
#
#     axs[1, 0].legend(handles=[
#         mlines.Line2D([], [], color=engine_groups['GTF2035_wi_0']['color'],
#                       marker=engine_groups['GTF2035_wi_0']['marker'], linestyle='None', markersize=8,
#                       label="GTF2035WI SAF 0"),
#         mlines.Line2D([], [], color=engine_groups['GTF2035_wi_20']['color'],
#                       marker=engine_groups['GTF2035_wi_20']['marker'], linestyle='None', markersize=8,
#                       label="GTF2035WI SAF 20"),
#         mlines.Line2D([], [], color=engine_groups['GTF2035_wi_100']['color'],
#                       marker=engine_groups['GTF2035_wi_100']['marker'], linestyle='None', markersize=8,
#                       label="GTF2035WI SAF 100"),
#         mlines.Line2D([], [], color='r', linestyle='--', label="Equal Fuel Flow")
#     ], loc='upper left', title="Legend")
#
#     plt.subplots_adjust(hspace=0.4)
#     plt.tight_layout()
#     plt.show()
#
# # Generate plots
# print("Plotting BEFORE correction...")
# plot_fuel_flow_correction(corrected=False)
#
# print("Plotting AFTER correction...")
# plot_fuel_flow_correction(corrected=True)
#
#
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Ensure we have corrected fuel flow before plotting
if 'fuel_flow_corrected' not in final_df.columns:
    raise ValueError("fuel_flow_corrected column is missing! Ensure corrections are applied before plotting.")

# Filter dataset to include only relevant engine models and SAF levels
gtf_df = final_df[final_df['engine'] == 'GTF'].copy()

gtf2035_saf_dfs = {saf: final_df[(final_df['engine'] == 'GTF2035') & (final_df['saf_level'] == saf)].copy() for saf in [0, 20, 100]}
gtf2035_wi_saf_dfs = {saf: final_df[(final_df['engine'] == 'GTF2035_wi') & (final_df['saf_level'] == saf)].copy() for saf in [0, 20, 100]}

# Flight phases (keeping 'descent' without renaming)
flight_phases = ['cruise', 'climb', 'descent']

# Phase-based dictionaries
gtf_phase_dfs = {phase: gtf_df[gtf_df['flight_phase'] == phase] for phase in flight_phases}
gtf2035_phase_dfs = {saf: {phase: gtf2035_saf_dfs[saf][gtf2035_saf_dfs[saf]['flight_phase'] == phase] for phase in flight_phases} for saf in [0, 20, 100]}
gtf2035_wi_phase_dfs = {saf: {phase: gtf2035_wi_saf_dfs[saf][gtf2035_wi_saf_dfs[saf]['flight_phase'] == phase] for phase in flight_phases} for saf in [0, 20, 100]}

# Define colors and markers per engine
engine_groups = {
    'GTF2035_0': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_20': {'marker': 's', 'color': 'tab:pink'},
    'GTF2035_100': {'marker': 's', 'color': 'tab:grey'},
    'GTF2035_wi_0': {'marker': 'D', 'color': 'tab:blue'},
    'GTF2035_wi_20': {'marker': 'D', 'color': 'tab:olive'},
    'GTF2035_wi_100': {'marker': 'D', 'color': 'tab:cyan'}
}


# Function to generate plots
def plot_fuel_flow_comparison(corrected=False):
    suffix = '_corrected' if corrected else ''
    title_suffix = "Corrected" if corrected else "Original"
    xlabel = "Corrected Fuel Flow from GTF (kg/s)" if corrected else "Fuel Flow from GTF (kg/s)"

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    merge_cols = ['trajectory', 'season', 'diurnal', 'flight_phase', 'index']

    legend_handles = []  # Store legend items

    for i, phase in enumerate(flight_phases):
        ax = axs[0, i]
        for saf in [0, 20, 100]:
            merged_df = gtf_phase_dfs[phase].merge(gtf2035_phase_dfs[saf][phase], on=merge_cols, suffixes=('_gtf', f'_gtf2035_{saf}'))

            # Dynamically choose the correct column
            fuel_flow_column_gtf = 'fuel_flow_corrected_gtf' if corrected else 'fuel_flow_gtf'
            fuel_flow_column_gtf2035 = f'fuel_flow_corrected_gtf2035_{saf}' if corrected else f'fuel_flow_gtf2035_{saf}'

            merged_df[f'fuel_flow_ratio{suffix}'] = merged_df[fuel_flow_column_gtf2035] / merged_df[fuel_flow_column_gtf]

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
            merged_df = gtf_phase_dfs[phase].merge(gtf2035_wi_phase_dfs[saf][phase], on=merge_cols, suffixes=('_gtf', f'_gtf2035wi_{saf}'))

            fuel_flow_column_gtf2035wi = f'fuel_flow_corrected_gtf2035wi_{saf}' if corrected else f'fuel_flow_gtf2035wi_{saf}'

            merged_df[f'fuel_flow_ratio{suffix}'] = merged_df[fuel_flow_column_gtf2035wi] / merged_df[fuel_flow_column_gtf]

            ax.scatter(merged_df[fuel_flow_column_gtf], merged_df[f'fuel_flow_ratio{suffix}'],
                       label=f'GTF2035WI SAF {saf} / GTF', marker=engine_groups[f'GTF2035_wi_{saf}']['marker'],
                       color=engine_groups[f'GTF2035_wi_{saf}']['color'], alpha=0.3, s=10)

        ax.axhline(y=1, color='r', linestyle='--', label="Equal Fuel Flow")
        ax.set_xlabel(xlabel)
        ax.set_title(f"GTF2035WI vs GTF - {phase.capitalize()} - {title_suffix}")

    axs[1, 0].legend(handles=legend_handles, loc='upper left', title="GTF2035WI Legend")

    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.show()


# Plot Uncorrected Data
plot_fuel_flow_comparison(corrected=False)

# Plot Corrected Data
plot_fuel_flow_comparison(corrected=True)


