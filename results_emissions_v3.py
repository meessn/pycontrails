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
                            'index', 'time', 'fuel_flow', 'ei_nox', 'nvpm_ei_n',
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
dt = 60
final_df['nox'] = final_df['ei_nox']*final_df['fuel_flow']*dt #unit is kg (kg/kg fuel * kg fuel/s * s )
final_df['nvpm_n'] = final_df['nvpm_ei_n']*final_df['fuel_flow']*dt #unit is # (#/kg fuel * kg fuel/s * s )
print(f"Collected {len(final_df)} rows from {len(dataframes)} flight data files.")
# print(final_df['fuel_flow'])


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
# #plt.savefig('results_report/emissions/nox_emissions_tt3_pt3_far_scatter.png', format='png')
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
# # plt.savefig('results_report/emissions/nvpm_emissions_no_saf_scatter.png', format='png')
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
# plt.grid(True, which='both', linestyle='--', linewidth=1.0)
# # plt.savefig('results_report/emissions/nvpm_emissions_icao_lto.png', format='png')
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
# # plt.savefig('results_report/emissions/nvpm_emissions_saf_scatter.png', format='png')
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
# plt.savefig('results_report/emissions/fuel_flow_no_saf_scatter.png', format='png')
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
# plt.savefig('results_report/emissions/fuel_flow_saf_scatter.png', format='png')
# plt.show()

# Define flight phases

"""ei nox vs waypoint """
flight_phases = ['climb', 'cruise', 'descent']
phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}

# Default engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF1990': {'marker': '^', 'color': 'tab:blue'},
    'GTF2000': {'marker': '^', 'color': 'tab:orange'},
    'GTF': {'marker': 'o', 'color': 'tab:green'},
    'GTF2035': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}
}

saf_colors = {
    ('GTF2035', 0): 'tab:red',
    ('GTF2035', 20): 'tab:pink',
    ('GTF2035', 100): 'tab:grey',
    ('GTF2035_wi', 0): default_colors[4],
    ('GTF2035_wi', 20): 'tab:olive',
    ('GTF2035_wi', 100): 'tab:cyan'
}

# Scatter Plots - Without SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
        subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
        if not subset.empty:
            ax.scatter(subset['index'], subset['ei_nox'] * 1000, label=engine_display_names[engine], marker=style['marker'],
                       color=style['color'], alpha=0.3, s=10)
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    if phase == 'climb':
        ax.set_xlim(0, subset['index'].max() + 5)
    else:
        ax.set_xlim(subset['index'].min() - 5, subset['index'].max() + 5)

axs[0].set_ylabel('EI NOx (g/kg fuel)')
fig.suptitle("EI NOx vs Time in Minutes (Without SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[2].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    # Extract color from scatter plot (PathCollection)
    rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
    solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
    marker = handle.get_paths()[0]  # Keep marker style consistent

    legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
                                        linestyle='None', markersize=8, label=label))

# Apply the updated handles to the legend
axs[2].legend(handles=legend_handles, loc='upper left', title='Engine', frameon=True, markerscale=1.0)
plt.savefig('results_report/emissions/ei_nox_no_saf_scatter_waypoints.png', format='png')
# plt.show()

# Scatter Plots - With SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine in ['GTF2035', 'GTF2035_wi']:
        for saf in [0, 20, 100]:
            subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
            subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
            if not subset.empty:
                label = f"{engine_display_names[engine]} SAF{saf}"
                ax.scatter(subset['index'], subset['ei_nox'] * 1000, label=label, marker=engine_groups[engine]['marker'],
                           color=saf_colors[(engine, saf)], alpha=0.3, s=10)
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    if phase == 'climb':
        ax.set_xlim(0, subset['index'].max() + 5)
    else:
        ax.set_xlim(subset['index'].min() - 5, subset['index'].max() + 5)

axs[0].set_ylabel('EI NOx (g/kg fuel)')
fig.suptitle("EI NOx vs Time in Minutes (With SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[2].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    # Extract color from scatter plot (PathCollection)
    rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
    solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
    marker = handle.get_paths()[0]  # Keep marker style consistent

    legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
                                        linestyle='None', markersize=8, label=label))

# Apply the updated handles to the legend
axs[2].legend(handles=legend_handles, loc='upper left', title='Engine & SAF Level', frameon=True, markerscale=1.0)
plt.savefig('results_report/emissions/ei_nox_saf_scatter_waypoints.png', format='png')
# plt.show()

# Function to introduce NaN gaps in the dataset
def introduce_gaps(df):
    df = df.sort_values(by='index')
    index_diff = df['index'].diff()
    gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
    df.loc[index_diff > gap_threshold, 'ei_nox'] = None  # Assign NaN where gaps exist
    return df

# Function to ensure gaps persist in average computation
def compute_avg_with_gaps(df):
    df = introduce_gaps(df)
    avg_subset = df.groupby('index')['ei_nox'].mean() * 1000
    return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))

# Line Plots - Without SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
        subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]  # Apply climb phase filter
        if not subset.empty:
            avg_subset = compute_avg_with_gaps(subset)
            ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    # ax.legend(title="Engine")

axs[0].set_ylabel('EI NOx (g/kg fuel)')
fig.suptitle("EI NOx Average vs Time in Minutes (Without SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[2].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))

# Apply the updated handles to the legend
axs[2].legend(handles=legend_handles, loc='upper left', title='Engine', frameon=True)
plt.savefig('results_report/emissions/ei_nox_no_saf_average_waypoints.png', format='png')
# plt.show()

# Line Plots - With SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine in ['GTF2035', 'GTF2035_wi']:
        for saf in [0, 20, 100]:
            subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
            subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]  # Apply climb phase filter
            if not subset.empty:
                avg_subset = compute_avg_with_gaps(subset)
                ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=saf_colors[(engine, saf)], alpha=1.0,
                        label=f"{engine_display_names[engine]} SAF{saf}")
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    # ax.legend(title="Engine & SAF Level")

axs[0].set_ylabel('EI NOx (g/kg fuel)')
fig.suptitle("EI NOx Average vs Time in Minutes (With SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[2].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))

# Apply the updated handles to the legend
axs[2].legend(handles=legend_handles, loc='upper left', title='Engine & SAF Level', frameon=True)
plt.savefig('results_report/emissions/ei_nox_saf_average_waypoints.png', format='png')
# plt.show()

"""ei nvPM"""

# Define flight phases
flight_phases = ['climb', 'cruise', 'descent']
phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}

# Default engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF1990': {'marker': '^', 'color': 'tab:blue'},
    'GTF2000': {'marker': '^', 'color': 'tab:orange'},
    'GTF': {'marker': 'o', 'color': 'tab:green'},
    'GTF2035': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}
}

saf_colors = {
    ('GTF2035', 0): 'tab:red',
    ('GTF2035', 20): 'tab:pink',
    ('GTF2035', 100): 'tab:grey',
    ('GTF2035_wi', 0): default_colors[4],
    ('GTF2035_wi', 20): 'tab:olive',
    ('GTF2035_wi', 100): 'tab:cyan'
}

# Function to introduce NaN gaps in the dataset
def introduce_gaps(df):
    df = df.sort_values(by='index')
    index_diff = df['index'].diff()
    gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
    df.loc[index_diff > gap_threshold, 'nvpm_ei_n'] = None  # Assign NaN where gaps exist
    return df

# Function to ensure gaps persist in average computation
def compute_avg_with_gaps(df):
    df = introduce_gaps(df)
    avg_subset = df.groupby('index')['nvpm_ei_n'].mean()
    return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))

# Scatter Plots - Without SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
        subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
        if not subset.empty:
            ax.scatter(subset['index'], subset['nvpm_ei_n'], label=engine_display_names[engine], marker=style['marker'],
                       color=style['color'], alpha=0.3, s=10)
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    ax.set_yscale('log')
    # ax.legend(title="Engine")

axs[0].set_ylabel('EI nvPM Number (#/kg fuel)')
fig.suptitle("EI nvPM Number vs Time in Minutes (Without SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    # Extract color from scatter plot (PathCollection)
    rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
    solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
    marker = handle.get_paths()[0]  # Keep marker style consistent

    legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
                                        linestyle='None', markersize=8, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True, markerscale=1.0)
plt.savefig('results_report/emissions/ei_nvpm_no_saf_scatter_waypoints.png', format='png')
# plt.show()

# Scatter Plots - With SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine in ['GTF2035', 'GTF2035_wi']:
        for saf in [0, 20, 100]:
            subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
            subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
            if not subset.empty:
                label = f"{engine_display_names[engine]} SAF{saf}"
                ax.scatter(subset['index'], subset['nvpm_ei_n'], label=label, marker=engine_groups[engine]['marker'],
                           color=saf_colors[(engine, saf)], alpha=0.3, s=10)
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    ax.set_yscale('log')
    # ax.legend(title="Engine & SAF Level")

axs[0].set_ylabel('EI nvPM Number (#/kg fuel)')
fig.suptitle("EI nvPM Number vs Time in Minutes (With SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    # Extract color from scatter plot (PathCollection)
    rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
    solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
    marker = handle.get_paths()[0]  # Keep marker style consistent

    legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
                                        linestyle='None', markersize=8, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True, markerscale=1.0)
plt.savefig('results_report/emissions/ei_nvpm_saf_scatter_waypoints.png', format='png')
# plt.show()

# Line Plots - Without SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
        subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
        if not subset.empty:
            avg_subset = compute_avg_with_gaps(subset)
            ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    ax.set_yscale('log')
    # ax.legend(title="Engine")

axs[0].set_ylabel('EI nvPM Number (#/kg fuel)')
fig.suptitle("EI nvPM Number Average vs Time in Minutes (Without SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True)
plt.savefig('results_report/emissions/ei_nvpm_no_saf_average_waypoints.png', format='png')
# plt.show()

# Line Plots - With SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine in ['GTF2035', 'GTF2035_wi']:
        for saf in [0, 20, 100]:
            subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
            subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
            if not subset.empty:
                avg_subset = compute_avg_with_gaps(subset)
                ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=saf_colors[(engine, saf)], alpha=1.0,
                        label=f"{engine_display_names[engine]} SAF{saf}")
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    ax.set_yscale('log')
    # ax.legend(title="Engine & SAF Level")

axs[0].set_ylabel('EI nvPM Number (#/kg fuel)')
fig.suptitle("EI nvPM Number Average vs Time in Minutes (With SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True)
plt.savefig('results_report/emissions/ei_nvpm_saf_average_waypoints.png', format='png')
# plt.show()

"""FUEL FLOW"""

# Define flight phases
flight_phases = ['climb', 'cruise', 'descent']
phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}

# Default engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF1990': {'marker': '^', 'color': 'tab:blue'},
    'GTF2000': {'marker': '^', 'color': 'tab:orange'},
    'GTF': {'marker': 'o', 'color': 'tab:green'},
    'GTF2035': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}
}

saf_colors = {
    ('GTF2035', 0): 'tab:red',
    ('GTF2035', 20): 'tab:pink',
    ('GTF2035', 100): 'tab:grey',
    ('GTF2035_wi', 0): default_colors[4],
    ('GTF2035_wi', 20): 'tab:olive',
    ('GTF2035_wi', 100): 'tab:cyan'
}

# Function to introduce NaN gaps in the dataset
def introduce_gaps(df):
    df = df.sort_values(by='index')
    index_diff = df['index'].diff()
    gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
    df.loc[index_diff > gap_threshold, 'fuel_flow'] = None  # Assign NaN where gaps exist
    return df

# Function to ensure gaps persist in average computation
def compute_avg_with_gaps(df):
    df = introduce_gaps(df)
    avg_subset = df.groupby('index')['fuel_flow'].mean()
    return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))

# Scatter Plots - Without SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
        subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
        if not subset.empty:
            ax.scatter(subset['index'], subset['fuel_flow'], label=engine_display_names[engine], marker=style['marker'],
                       color=style['color'], alpha=0.3, s=10)
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    #ax.set_yscale('log')
    # ax.legend(title="Engine")

axs[0].set_ylabel('Fuel Flow (kg/s)')
fig.suptitle("Fuel Flow vs Time in Minutes (Without SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    # Extract color from scatter plot (PathCollection)
    rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
    solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
    marker = handle.get_paths()[0]  # Keep marker style consistent

    legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
                                        linestyle='None', markersize=8, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True, markerscale=1.0)
plt.savefig('results_report/emissions/fuel_no_saf_scatter_waypoints.png', format='png')
# plt.show()

# Scatter Plots - With SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine in ['GTF2035', 'GTF2035_wi']:
        for saf in [0, 20, 100]:
            subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
            subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
            if not subset.empty:
                label = f"{engine_display_names[engine]} SAF{saf}"
                ax.scatter(subset['index'], subset['fuel_flow'], label=label, marker=engine_groups[engine]['marker'],
                           color=saf_colors[(engine, saf)], alpha=0.3, s=10)
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    #ax.set_yscale('log')
    # ax.legend(title="Engine & SAF Level")

axs[0].set_ylabel('Fuel Flow (kg/s)')
fig.suptitle("Fuel Flow vs Time in Minutes (With SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    # Extract color from scatter plot (PathCollection)
    rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
    solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
    marker = handle.get_paths()[0]  # Keep marker style consistent

    legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
                                        linestyle='None', markersize=8, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True, markerscale=1.0)
plt.savefig('results_report/emissions/fuel_saf_scatter_waypoints.png', format='png')
# plt.show()

# Line Plots - Without SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
        subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
        if not subset.empty:
            avg_subset = compute_avg_with_gaps(subset)
            ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    #ax.set_yscale('log')
    # ax.legend(title="Engine")

axs[0].set_ylabel('Fuel Flow (kg/s)')
fig.suptitle("Fuel Flow Average vs Time in Minutes (Without SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True)
plt.savefig('results_report/emissions/fuel_no_saf_average_waypoints.png', format='png')
# plt.show()

# Line Plots - With SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine in ['GTF2035', 'GTF2035_wi']:
        for saf in [0, 20, 100]:
            subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
            subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
            if not subset.empty:
                avg_subset = compute_avg_with_gaps(subset)
                ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=saf_colors[(engine, saf)], alpha=1.0,
                        label=f"{engine_display_names[engine]} SAF{saf}")
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    #ax.set_yscale('log')
    # ax.legend(title="Engine & SAF Level")

axs[0].set_ylabel('Fuel Flow (kg/s)')
fig.suptitle("Fuel Flow Average vs Time in Minutes (With SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True)
plt.savefig('results_report/emissions/fuel_saf_average_waypoints.png', format='png')
# plt.show()

"""NOx instead of EI_NOx"""

# Define flight phases
flight_phases = ['climb', 'cruise', 'descent']
phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}

# Default engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF1990': {'marker': '^', 'color': 'tab:blue'},
    'GTF2000': {'marker': '^', 'color': 'tab:orange'},
    'GTF': {'marker': 'o', 'color': 'tab:green'},
    'GTF2035': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}
}

saf_colors = {
    ('GTF2035', 0): 'tab:red',
    ('GTF2035', 20): 'tab:pink',
    ('GTF2035', 100): 'tab:grey',
    ('GTF2035_wi', 0): default_colors[4],
    ('GTF2035_wi', 20): 'tab:olive',
    ('GTF2035_wi', 100): 'tab:cyan'
}

# Function to introduce NaN gaps in the dataset
def introduce_gaps(df):
    df = df.sort_values(by='index')
    index_diff = df['index'].diff()
    gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
    df.loc[index_diff > gap_threshold, 'nox'] = None  # Assign NaN where gaps exist
    return df

# Function to ensure gaps persist in average computation
def compute_avg_with_gaps(df):
    df = introduce_gaps(df)
    avg_subset = df.groupby('index')['nox'].mean()
    return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))

# Line Plots - Without SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
        subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]  # Apply climb phase filter
        if not subset.empty:
            avg_subset = compute_avg_with_gaps(subset)
            ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    # ax.legend(title="Engine")

axs[0].set_ylabel('NOx (kg)')
fig.suptitle("NOx Average vs Time in Minutes (Without SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[2].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))

# Apply the updated handles to the legend
axs[2].legend(handles=legend_handles, loc='upper left', title='Engine', frameon=True)
plt.savefig('results_report/emissions/nox_no_saf_average_waypoints.png', format='png')

"""nvPM instead of EI_nvPM"""

# Define flight phases
flight_phases = ['climb', 'cruise', 'descent']
phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}

# Default engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF1990': {'marker': '^', 'color': 'tab:blue'},
    'GTF2000': {'marker': '^', 'color': 'tab:orange'},
    'GTF': {'marker': 'o', 'color': 'tab:green'},
    'GTF2035': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}
}

saf_colors = {
    ('GTF2035', 0): 'tab:red',
    ('GTF2035', 20): 'tab:pink',
    ('GTF2035', 100): 'tab:grey',
    ('GTF2035_wi', 0): default_colors[4],
    ('GTF2035_wi', 20): 'tab:olive',
    ('GTF2035_wi', 100): 'tab:cyan'
}

# Function to introduce NaN gaps in the dataset
def introduce_gaps(df):
    df = df.sort_values(by='index')
    index_diff = df['index'].diff()
    gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
    df.loc[index_diff > gap_threshold, 'nvpm_n'] = None  # Assign NaN where gaps exist
    return df

# Function to ensure gaps persist in average computation
def compute_avg_with_gaps(df):
    df = introduce_gaps(df)
    avg_subset = df.groupby('index')['nvpm_n'].mean()
    return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))

# Line Plots - Without SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
        subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
        if not subset.empty:
            avg_subset = compute_avg_with_gaps(subset)
            ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    ax.set_yscale('log')
    # ax.legend(title="Engine")

axs[0].set_ylabel('nvPM Number (#)')
fig.suptitle("nvPM Number Average vs Time in Minutes (Without SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
# Create new handles with solid lines instead of scatter markers
legend_handles = []
for handle, label in zip(handles, labels):
    legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True)
plt.savefig('results_report/emissions/nvpm_no_saf_average_waypoints.png', format='png')
# plt.show()

# Line Plots - With SAF
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine in ['GTF2035', 'GTF2035_wi']:
        for saf in [0, 20, 100]:
            subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
            subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
            if not subset.empty:
                avg_subset = compute_avg_with_gaps(subset)
                ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=saf_colors[(engine, saf)], alpha=1.0,
                        label=f"{engine_display_names[engine]} SAF{saf}")
    ax.set_xlabel('Time in minutes')
    ax.set_title(phase_titles[phase])
    ax.set_yscale('log')
    # ax.legend(title="Engine & SAF Level")

axs[0].set_ylabel('nvPM Number (#)')
fig.suptitle("nvPM Number Average vs Time in Minutes (With SAF)", fontsize=14)
plt.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))

# Apply the updated handles to the legend
axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True)
plt.savefig('results_report/emissions/nvpm_saf_average_waypoints.png', format='png')
plt.show()