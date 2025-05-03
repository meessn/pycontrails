import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.ticker import FuncFormatter
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

                            if saf == 0:
                                df['ei_co2_conservative'] = 3.825
                                df['ei_co2_optimistic'] = 3.825
                                df['ei_h2o'] = 1.237
                            elif saf == 20:
                                df['ei_co2_conservative'] = 3.75
                                df['ei_co2_optimistic'] = 3.1059
                                df['ei_h2o'] = 1.264
                            elif saf == 100:
                                df['ei_co2_conservative'] = 3.4425
                                df['ei_co2_optimistic'] = 0.2295
                                df['ei_h2o'] = 1.370

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

                        #Ensure 'cocip_atr20' exists in df, fillwith 0 if missing
                        if 'cocip_atr20' not in df.columns:
                            df['cocip_atr20'] = 0

                        selected_columns = [
                            'index', 'time','altitude', 'fuel_flow', 'ei_nox', 'nvpm_ei_n',
                            'thrust_setting_meem', 'TT3', 'PT3', 'FAR', 'specific_humidity_gsp',
                            'flight_phase', 'trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection',
                            'accf_sac_aCCF_O3', 'accf_sac_aCCF_CH4', 'accf_sac_aCCF_CO2', 'ei_co2_conservative',
                            'ei_co2_optimistic', 'ei_h2o', 'cocip_atr20', 'accf_sac_aCCF_Cont', 'accf_sac_aCCF_H2O', 'accf_sac_segment_length_km',
                            'accf_sac_accf_contrail_cocip', 'accf_sac_pcfa', 'accf_sac_issr', 'accf_sac_sac', 'engine_efficiency', 'LHV', 'air_pressure', 'air_temperature', 'accf_sac_geopotential'
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

# Add calculations per waypoint to final_df
final_df['nox_impact'] = final_df['fuel_flow'] * dt * (final_df['accf_sac_aCCF_O3'] + final_df['accf_sac_aCCF_CH4'] * 1.29) * final_df['ei_nox']


# # Filter rows where accf_sac_pcfa < 0.9 AND cocip_atr20 != 0
# condition = (final_df['accf_sac_pcfa'] < 0.9) & final_df['cocip_atr20'].notna() & (final_df['cocip_atr20'] != 0)
#
#
# # Count the number of such rows
# count = condition.sum()
#
# print(f"Number of times where accf_sac_pcfa < 0.9 and cocip_atr20 != 0: {count}")

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
# # Ensure 'cocip_atr20' exists before using it
# if 'cocip_atr20' in final_df.columns:
#     final_df['contrail_atr20_cocip'] = final_df['cocip_atr20'].fillna(0) * 0.42
# else:
#     final_df['contrail_atr20_cocip'] = 0
#
# # ACCF contrail impact (aCCF weighted by segment length)
# final_df['contrail_atr20_accf'] = final_df['accf_sac_aCCF_Cont'] * final_df['accf_sac_segment_length_km']
#
# # ACCF-CoCiP-PCFA contrail impact
# final_df['contrail_atr20_accf_cocip_pcfa'] = final_df['accf_sac_aCCF_Cont'] * final_df['accf_sac_segment_length_km']
#
# # Climate impact variants
# final_df['climate_non_co2_cocip'] = final_df['nox_impact'] + final_df['h2o_impact'] + final_df['contrail_atr20_cocip']
# final_df['climate_total_cons_cocip'] = final_df['climate_non_co2_cocip'] + final_df['co2_impact_cons']
# final_df['climate_total_opti_cocip'] = final_df['climate_non_co2_cocip'] + final_df['co2_impact_opti']
#
# final_df['climate_non_co2_accf'] = final_df['nox_impact'] + final_df['h2o_impact'] + final_df['contrail_atr20_accf']
# final_df['climate_total_cons_accf'] = final_df['climate_non_co2_accf'] + final_df['co2_impact_cons']
# final_df['climate_total_opti_accf'] = final_df['climate_non_co2_accf'] + final_df['co2_impact_opti']
#
# final_df['climate_non_co2_accf_cocip_pcfa'] = final_df['nox_impact'] + final_df['h2o_impact'] + final_df['contrail_atr20_accf_cocip_pcfa']
# final_df['climate_total_cons_accf_cocip_pcfa'] = final_df['climate_non_co2_accf_cocip_pcfa'] + final_df['co2_impact_cons']
# final_df['climate_total_opti_accf_cocip_pcfa'] = final_df['climate_non_co2_accf_cocip_pcfa'] + final_df['co2_impact_opti']
# print(final_df['fuel_flow'])


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

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

legend_handles = {}
seen_cfm = False

for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
    ax = axs[i]

    for engine, style in engine_groups.items():
        subset = final_df[final_df['engine'] == engine]
        if not subset.empty:
            ax.scatter(subset[x_var], subset['ei_nox'] * 1000,
                       label=engine_display_names[engine], marker=style['marker'],
                       color=style['color'], alpha=0.3, s=10)

            if engine in ['GTF1990', 'GTF2000']:
                if not seen_cfm:
                    legend_handles['CFM1990/2000'] = mlines.Line2D(
                        [], [], color=style['color'], marker=style['marker'], linestyle='None', markersize=8, label='CFM1990/2000'
                    )
                    seen_cfm = True
            else:
                if engine_display_names[engine] not in legend_handles:
                    legend_handles[engine_display_names[engine]] = mlines.Line2D(
                        [], [], color=style['color'], marker=style['marker'], linestyle='None', markersize=8, label=engine_display_names[engine]
                    )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(r'$EI_{\mathrm{NOx}}$ (g/ kg Fuel)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    if i % 2 == 1:  # Right-hand column (remove y label & ticks)
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

# Place legend in top-left of first subplot
axs[0].legend(handles=legend_handles.values(), loc='upper left', title="Engine", title_fontsize=13, fontsize=12)

plt.tight_layout()
plt.savefig('results_report/emissions/nox_emissions_tt3_pt3_far_scatter.png', format='png')

# #plt.show()


# Engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2008',
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

x_vars = {
    'TT3': ('TT3', 'TT3 (K)'),
    'PT3': ('PT3', 'PT3 (bar)'),
    'FAR': ('FAR', 'FAR (-)'),
    'thrust_setting_meem': ('Thrust Setting', 'Thrust Setting (-)')
}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

legend_handles = {}

for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
    ax = axs[i]

    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine)]

        # Exclude SAF != 0
        if engine in ['GTF2035', 'GTF2035_wi']:
            subset = subset[subset['saf_level'] == 0]

        if not subset.empty:
            ax.scatter(subset[x_var], subset['nvpm_ei_n'],
                       label=engine_display_names[engine], marker=style['marker'],
                       color=style['color'], alpha=0.3, s=10)

            if engine_display_names[engine] not in legend_handles:
                legend_handles[engine_display_names[engine]] = mlines.Line2D(
                    [], [], color=style['color'], marker=style['marker'], linestyle='None',
                    markersize=8, label=engine_display_names[engine]
                )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(r'$EI_{\mathrm{nvPM,number}}$ (# / kg Fuel)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_yscale('log')

    if i % 2 == 1:  # Right-hand column
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

axs[1].legend(handles=legend_handles.values(), loc='lower right', title="Engine", title_fontsize=13, fontsize=12)

plt.tight_layout()
# plt.subplots_adjust(hspace=0.3, wspace=0.25)
plt.savefig('results_report/emissions/nvpm_emissions_no_saf_scatter.png', format='png')

# #plt.show()

# #plt.show()

# Consistent engine display names and colors as in previous plots
engine_display_names = ['CFM1990', 'CFM2008', 'GTF', 'GTF2035', 'GTF2035WI']
engine_colors = {
    'CFM1990': 'tab:blue',
    'CFM2008': 'tab:orange',
    'GTF': 'tab:green',
    'GTF2035': 'tab:red',
    'GTF2035WI': 'purple'  # Matplotlib default color cycle index 4 is purple
}

# ICAO data
thrust_setting_icao = [0.07, 0.3, 0.85, 1]

icao_data = {
    'GTF': [5.78e15, 3.85e14, 1.60e15, 1.45e15],
    'GTF2035': [5.78e15, 3.85e14, 1.60e15, 1.45e15],
    'GTF2035WI': [5.78e15, 3.85e14, 1.60e15, 1.45e15],
    'CFM1990': [4.43e15, 9.03e15, 2.53e15, 1.62e15],
    'CFM2008': [7.98e14, 4.85e14, 1.39e15, 1.02e15],
}

# Plot
plt.figure(figsize=(8, 6))
for engine in engine_display_names:
    plt.plot(thrust_setting_icao, icao_data[engine], marker='o', label=engine, color=engine_colors[engine])

plt.xlabel('Thrust Setting (-)')
plt.ylabel(f'$EI_{{\\mathrm{{nvPM,number}}}}$ (# / kg Fuel)')
plt.title('ICAO $EI_{{\\mathrm{{nvPM,number}}}}$ vs Thrust Setting (SLS)')
plt.yscale('log')
plt.legend(title='Engine')
plt.grid(True, which='both', linestyle='--', linewidth=1.0)
plt.savefig('results_report/emissions/nvpm_emissions_icao_lto.png', format='png')
# #plt.show()



# Use Matplotlib default color cycle for consistency
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_display_names = {
    'GTF2035_0': 'GTF2035 SAF0',
    'GTF2035_20': 'GTF2035 SAF20',
    'GTF2035_100': 'GTF2035 SAF100',
    'GTF2035_wi_0': 'GTF2035WI SAF0',
    'GTF2035_wi_20': 'GTF2035WI SAF20',
    'GTF2035_wi_100': 'GTF2035WI SAF100'
}

# Colors based on Plot 1 (use same color for GTF2035WI across SAF levels)
engine_groups = {
    'GTF2035_0': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_20': {'marker': 's', 'color': 'tab:pink'},
    'GTF2035_100': {'marker': 's', 'color': 'tab:grey'},
    'GTF2035_wi_0': {'marker': 'D', 'color': default_colors[4]},  # Same as in Plot 1
    'GTF2035_wi_20': {'marker': 'D', 'color': 'tab:olive'},
    'GTF2035_wi_100': {'marker': 'D', 'color': 'tab:cyan'}
}

x_vars = {
    'TT3': ('TT3', 'TT3 (K)'),
    'PT3': ('PT3', 'PT3 (bar)'),
    'FAR': ('FAR', 'FAR (-)'),
    'thrust_setting_meem': ('Thrust Setting', 'Thrust Setting (-)')
}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

legend_handles = {}

for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
    ax = axs[i]

    for engine, style in engine_groups.items():
        base_engine = 'GTF2035' if 'wi' not in engine else 'GTF2035_wi'
        saf_level = int(engine.split('_')[-1])

        subset = final_df[(final_df['engine'] == base_engine) & (final_df['saf_level'] == saf_level)]

        if not subset.empty:
            ax.scatter(subset[x_var], subset['nvpm_ei_n'],
                       label=engine_display_names[engine], marker=style['marker'],
                       color=style['color'], alpha=0.3, s=10)

            if engine_display_names[engine] not in legend_handles:
                legend_handles[engine_display_names[engine]] = mlines.Line2D(
                    [], [], color=style['color'], marker=style['marker'], linestyle='None',
                    markersize=8, label=engine_display_names[engine]
                )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(r'$EI_{\mathrm{nvPM,number}}$ (# / kg Fuel)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_yscale('log')

    if i % 2 == 1:  # Right column
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

axs[1].legend(handles=legend_handles.values(), loc='lower right', title="Engine", title_fontsize=13, fontsize=12)

plt.tight_layout()
# plt.subplots_adjust(hspace=0.3, wspace=0.25)
plt.savefig('results_report/emissions/nvpm_emissions_saf_scatter.png', format='png')

# #plt.show()


# Engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2008',
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

x_vars = {
    'TT3': ('TT3', 'TT3 (K)'),
    'PT3': ('PT3', 'PT3 (bar)'),
    'FAR': ('FAR', 'FAR (-)'),
    'thrust_setting_meem': ('Thrust Setting', 'Thrust Setting (-)')
}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

legend_handles = {}

for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
    ax = axs[i]

    for engine, style in engine_groups.items():
        subset = final_df[(final_df['engine'] == engine)]

        # Exclude SAF != 0
        if engine in ['GTF2035', 'GTF2035_wi']:
            subset = subset[subset['saf_level'] == 0]

        if not subset.empty:
            ax.scatter(subset[x_var], subset['fuel_flow'],
                       label=engine_display_names[engine], marker=style['marker'],
                       color=style['color'], alpha=0.3, s=10)

            if engine_display_names[engine] not in legend_handles:
                legend_handles[engine_display_names[engine]] = mlines.Line2D(
                    [], [], color=style['color'], marker=style['marker'], linestyle='None',
                    markersize=8, label=engine_display_names[engine]
                )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Fuel Flow (kg/s)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    if i % 2 == 1:  # Right column
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

axs[1].legend(handles=legend_handles.values(), loc='lower right', title="Engine", title_fontsize=13, fontsize=12)

plt.tight_layout()
# plt.subplots_adjust(hspace=0.3, wspace=0.25)
plt.savefig('results_report/emissions/fuel_flow_no_saf_scatter.png', format='png')

# #plt.show()



default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_display_names = {
    'GTF2035_0': 'GTF2035 SAF0',
    'GTF2035_20': 'GTF2035 SAF20',
    'GTF2035_100': 'GTF2035 SAF100',
    'GTF2035_wi_0': 'GTF2035WI SAF0',
    'GTF2035_wi_20': 'GTF2035WI SAF20',
    'GTF2035_wi_100': 'GTF2035WI SAF100'
}

engine_groups = {
    'GTF2035_0': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_20': {'marker': 's', 'color': 'tab:pink'},
    'GTF2035_100': {'marker': 's', 'color': 'tab:grey'},
    'GTF2035_wi_0': {'marker': 'D', 'color': default_colors[4]},
    'GTF2035_wi_20': {'marker': 'D', 'color': 'tab:olive'},
    'GTF2035_wi_100': {'marker': 'D', 'color': 'tab:cyan'}
}

x_vars = {
    'TT3': ('TT3', 'TT3 (K)'),
    'PT3': ('PT3', 'PT3 (bar)'),
    'FAR': ('FAR', 'FAR (-)'),
    'thrust_setting_meem': ('Thrust Setting', 'Thrust Setting (-)')
}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

legend_handles = {}

for i, (x_var, (title_label, x_label)) in enumerate(x_vars.items()):
    ax = axs[i]

    for engine, style in engine_groups.items():
        base_engine = 'GTF2035' if 'wi' not in engine else 'GTF2035_wi'
        saf_level = int(engine.split('_')[-1])

        subset = final_df[(final_df['engine'] == base_engine) & (final_df['saf_level'] == saf_level)]

        if not subset.empty:
            ax.scatter(subset[x_var], subset['fuel_flow'],
                       label=engine_display_names[engine], marker=style['marker'],
                       color=style['color'], alpha=0.3, s=10)

            if engine_display_names[engine] not in legend_handles:
                legend_handles[engine_display_names[engine]] = mlines.Line2D(
                    [], [], color=style['color'], marker=style['marker'], linestyle='None',
                    markersize=8, label=engine_display_names[engine]
                )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Fuel Flow (kg/s)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    if i % 2 == 1:  # Right column
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelleft=False)

axs[1].legend(handles=legend_handles.values(), loc='lower right', title="Engine", title_fontsize=13, fontsize=12)

plt.tight_layout()
# plt.subplots_adjust(hspace=0.3, wspace=0.25)
plt.savefig('results_report/emissions/fuel_flow_saf_scatter.png', format='png')

# plt.show()
#
# Define flight phases

"""ei nox vs waypoint """
flight_phases = ['climb', 'cruise', 'descent']
phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}

# Default engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2008',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF1990': {'marker': '^', 'color': 'tab:orange'},
    'GTF2000': {'marker': '^', 'color': 'tab:blue'},
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

# # Scatter Plots - Without SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
#         subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#         if not subset.empty:
#             ax.scatter(subset['index'], subset['ei_nox'] * 1000, label=engine_display_names[engine], marker=style['marker'],
#                        color=style['color'], alpha=0.3, s=10)
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     if phase == 'climb':
#         ax.set_xlim(0, subset['index'].max() + 5)
#     else:
#         ax.set_xlim(subset['index'].min() - 5, subset['index'].max() + 5)
#
# axs[0].set_ylabel(f'$EI_{{\\mathrm{{NOx}}}}$')
# fig.suptitle(f"$EI_{{\\mathrm{{NOx}}}}$ vs Time in Minutes (Without SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[2].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     # Extract color from scatter plot (PathCollection)
#     rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
#     solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
#     marker = handle.get_paths()[0]  # Keep marker style consistent
#
#     legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
#                                         linestyle='None', markersize=8, label=label))
#
# # Apply the updated handles to the legend
# axs[2].legend(handles=legend_handles, loc='upper left', title='Engine', frameon=True, markerscale=1.0)
# plt.savefig('results_report/emissions/ei_nox_no_saf_scatter_waypoints.png', format='png')
# # plt.show()

# Create 3 subplots, one for each phase
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        subset = final_df[
            (final_df['engine'] == engine) &
            (final_df['flight_phase'] == phase) &
            (final_df['saf_level'] == 0)
        ]
        if not subset.empty:
            ax.scatter(subset['altitude'], subset['ei_nox'] * 1000,
                       label=engine_display_names[engine],
                       marker=style['marker'], color=style['color'],
                       alpha=0.3, s=10)

    ax.set_xlabel('Altitude (m)')
    ax.set_title(phase_titles[phase])
    ax.grid(True)

axs[0].set_ylabel(r'$EI_{\mathrm{NOx}}$ (g / kg Fuel)')
fig.suptitle(r"$EI_{\mathrm{NOx}}$ vs Altitude (Without SAF)", fontsize=14)

# Create legend from the last plot's data
handles, labels = axs[2].get_legend_handles_labels()
legend_handles = []
for handle, label in zip(handles, labels):
    rgba_color = handle.get_facecolor()[0]
    solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)
    marker = handle.get_paths()[0]
    legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
                                        linestyle='None', markersize=8, label=label))

axs[2].legend(handles=legend_handles, loc='upper left', title='Engine', frameon=True, markerscale=1.0)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
# plt.savefig('results_report/emissions/ei_nox_vs_altitude_phases.png', format='png')
# plt.show()
#
# # Scatter Plots - With SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine in ['GTF2035', 'GTF2035_wi']:
#         for saf in [0, 20, 100]:
#             subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
#             subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#             if not subset.empty:
#                 label = f"{engine_display_names[engine]} SAF{saf}"
#                 ax.scatter(subset['index'], subset['ei_nox'] * 1000, label=label, marker=engine_groups[engine]['marker'],
#                            color=saf_colors[(engine, saf)], alpha=0.3, s=10)
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     if phase == 'climb':
#         ax.set_xlim(0, subset['index'].max() + 5)
#     else:
#         ax.set_xlim(subset['index'].min() - 5, subset['index'].max() + 5)
#
# axs[0].set_ylabel(f'$EI_{{\\mathrm{{NOx}}}}$ (g / kg Fuel)')
# fig.suptitle(f"$EI_{{\\mathrm{{NOx}}}}$ vs Time in Minutes (With SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[2].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     # Extract color from scatter plot (PathCollection)
#     rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
#     solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
#     marker = handle.get_paths()[0]  # Keep marker style consistent
#
#     legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
#                                         linestyle='None', markersize=8, label=label))
#
# # Apply the updated handles to the legend
# axs[2].legend(handles=legend_handles, loc='upper left', title='Engine & SAF Level', frameon=True, markerscale=1.0)
# plt.savefig('results_report/emissions/ei_nox_saf_scatter_waypoints.png', format='png')
# # plt.show()
#
# # Function to introduce NaN gaps in the dataset
# def introduce_gaps(df):
#     df = df.sort_values(by='index')
#     index_diff = df['index'].diff()
#     gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
#     df.loc[index_diff > gap_threshold, 'ei_nox'] = None  # Assign NaN where gaps exist
#     return df
#
# # Function to ensure gaps persist in average computation
# def compute_avg_with_gaps(df):
#     df = introduce_gaps(df)
#     avg_subset = df.groupby('index')['ei_nox'].mean() * 1000
#     return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))
# #
# # Line Plots - Without SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
#         subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]  # Apply climb phase filter
#         if not subset.empty:
#             avg_subset = compute_avg_with_gaps(subset)
#             ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     # ax.legend(title="Engine")
#
# axs[0].set_ylabel(f'$EI_{{\\mathrm{{NOx}}}}$ (g/kg Fuel)')
# fig.suptitle(f"$EI_{{\\mathrm{{NOx}}}}$ Average vs Time in Minutes (Without SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[2].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))
#
# # Apply the updated handles to the legend
# axs[2].legend(handles=legend_handles, loc='upper left', title='Engine', frameon=True)
# plt.savefig('results_report/emissions/ei_nox_no_saf_average_waypoints.png', format='png')
# # plt.show()
def introduce_altitude_gaps(df, altitude_column='altitude', value_column='ei_nox', gap_threshold=100):
    df = df.sort_values(by=altitude_column)
    alt_diff = df[altitude_column].diff()
    df.loc[alt_diff > gap_threshold, value_column] = None
    return df

def compute_binned_avg(df, bin_size=50):
    # Create altitude bins (e.g. every 500m)
    df = df.copy()
    df['alt_bin'] = (df['altitude'] // bin_size) * bin_size
    grouped = df.groupby('alt_bin')['ei_nox'].mean() * 1000  # Convert to g/kg
    return grouped.sort_index()

def compute_binned_avg_with_gaps(df, bin_size=50, min_count=5):
    df = df.copy()
    df['alt_bin'] = (df['altitude'] // bin_size) * bin_size
    grouped = df.groupby('alt_bin')['ei_nox'].agg(['mean', 'count'])

    # Filter based on count
    grouped = grouped[grouped['count'] >= min_count]

    avg = grouped['mean'] * 1000  # Convert to g/kg
    # Reindex to full range with NaNs where no data
    full_index = np.arange(df['alt_bin'].min(), df['alt_bin'].max() + bin_size, bin_size)
    return avg.reindex(full_index)

def compute_binned_stats_with_gaps(df, bin_size=250, min_count=5):
    df = df.copy()
    df['alt_bin'] = (df['altitude'] // bin_size) * bin_size
    grouped = df.groupby('alt_bin')['ei_nox'].agg(['mean', 'std', 'count'])
    grouped = grouped[grouped['count'] >= min_count]
    grouped['mean'] *= 1000  # g/kg
    grouped['std'] *= 1000
    full_index = np.arange(df['alt_bin'].min(), df['alt_bin'].max() + bin_size, bin_size)
    return grouped.reindex(full_index)

# Plot setup
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(flight_phases):
    ax = axs[i]
    for engine, style in engine_groups.items():
        if engine == 'GTF1990':
            continue  # Skip plotting CFM1990

        subset = final_df[
            (final_df['engine'] == engine) &
            (final_df['flight_phase'] == phase) &
            (final_df['saf_level'] == 0)
        ]
        if not subset.empty:
            stats = compute_binned_stats_with_gaps(subset)

            # Shaded area
            ax.fill_between(stats.index / 1000,
                            stats['mean'] - stats['std'],
                            stats['mean'] + stats['std'],
                            color=style['color'], alpha=0.2)
            # Line plot
            ax.plot(stats.index / 1000, stats['mean'], color=style['color'], linewidth=2.0,
                    label=engine_display_names[engine])

    ax.set_xlabel('Altitude (km)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.set_title(phase_titles[phase], fontsize=16)
    ax.grid(True)

axs[0].set_ylabel(r'$EI_{\mathrm{NOx}}$ (Mean $\pm$ Std) (g / kg Fuel)', fontsize=16)
axs[0].tick_params(axis='y', labelsize=12)
# fig.suptitle(r"$EI_{\mathrm{NOx}}$ vs Altitude", fontsize=14)

# Legend with CFM1990 added manually first
legend_handles = [
    mlines.Line2D([], [], color=engine_groups['GTF1990']['color'], linestyle='-', linewidth=2, label='CFM1990')
]

seen_labels = {'CFM1990'}
handles, labels = axs[2].get_legend_handles_labels()
for handle, label in zip(handles, labels):
    if label not in seen_labels:
        legend_handles.append(
            mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label)
        )
        seen_labels.add(label)

axs[2].legend(handles=legend_handles, loc='upper left', title='Engine', frameon=True)

plt.tight_layout()
# plt.subplots_adjust(top=0.88)
# plt.savefig('results_report/emissions/ei_nox_mean_std_altitude_phases.png', format='png')
# plt.show()


def compute_binned_stats(df, value_column, bin_column='altitude', bin_size=250, min_count=5, scale=1.0):
    df = df.copy()
    df['bin'] = (df[bin_column] // bin_size) * bin_size
    grouped = df.groupby('bin')[value_column].agg(['mean', 'std', 'count'])
    grouped = grouped[grouped['count'] >= min_count]
    grouped['mean'] *= scale
    grouped['std'] *= scale
    full_index = np.arange(df['bin'].min(), df['bin'].max() + bin_size, bin_size)
    return grouped.reindex(full_index)

def plot_metric_vs_altitude(final_df, value_column, y_label, filename, scale=1.0, bin_size=250, log_y=False, skip_cfm1990=False,
    legend_location='upper left', legend_axis=2, y_limits=None):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, phase in enumerate(flight_phases):
        ax = axs[i]
        for engine, style in engine_groups.items():
            if skip_cfm1990 and engine == 'GTF1990':
                continue  # Optionally skip plotting CFM1990

            subset = final_df[
                (final_df['engine'] == engine) &
                (final_df['flight_phase'] == phase) &
                (final_df['saf_level'] == 0)
            ]
            if not subset.empty:
                stats = compute_binned_stats(subset, value_column=value_column, bin_size=bin_size, scale=scale)
                print(stats['count'].min(), stats['count'].max())
                ax.fill_between(stats.index / 1000,
                                stats['mean'] - stats['std'],
                                stats['mean'] + stats['std'],
                                color=style['color'], alpha=0.2)
                ax.plot(stats.index / 1000, stats['mean'], color=style['color'], linewidth=2.0,
                        label=engine_display_names[engine])

        ax.set_xlabel('Altitude (km)', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.set_title(phase_titles[phase], fontsize=16)
        ax.grid(True)
        if log_y:
            ax.set_yscale('log')
        if y_limits is not None:
            ax.set_ylim(y_limits)

    axs[0].set_ylabel(y_label, fontsize=16)
    axs[0].tick_params(axis='y', labelsize=12)

    # Legend
    legend_handles = [
        mlines.Line2D([], [], color=engine_groups['GTF1990']['color'], linestyle='-', linewidth=2, label='CFM1990')
    ]
    seen_labels = {'CFM1990'}
    handles, labels = axs[2].get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            legend_handles.append(
                mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label)
            )
            seen_labels.add(label)

    axs[legend_axis].legend(handles=legend_handles, loc=legend_location, title='Engine', frameon=True)
    plt.tight_layout()
    plt.savefig(filename, format='png')
    # plt.show()

plot_metric_vs_altitude(
    final_df, value_column='ei_nox',
    y_label='$EI_{\mathrm{NOx}}$ (Mean $\pm$ Std) (g / kg Fuel)',
    scale=1000,
    bin_size=250,
    filename='results_report/emissions/ei_nox_mean_std_altitude.png',
    log_y=False,
    skip_cfm1990=True,
    legend_location='upper left',
    legend_axis=2
)

plot_metric_vs_altitude(
    final_df, value_column='nvpm_ei_n',
    y_label='$EI_{{\\mathrm{{nvPM,number}}}}$ (Mean $\pm$ Std) (# / kg Fuel)',
    scale=1,
    bin_size=250,
    filename='results_report/emissions/nvpm_ei_n_mean_std_altitude.png',
    log_y=True,
    skip_cfm1990=False,
    legend_location='lower left',
    legend_axis=0,
    y_limits=(8e13, 1.5e16)
)

plot_metric_vs_altitude(
    final_df, value_column='nox',
    y_label='NOx (Mean $\pm$ Std) (kg)',
    scale=1,
    bin_size=250,
    filename='results_report/emissions/nox_mean_std_altitude.png',
    log_y=False,
    skip_cfm1990=True,
    legend_location='upper left',
    legend_axis=2
)

plot_metric_vs_altitude(
    final_df, value_column='nvpm_n',
    y_label='nvPM Number (Mean $\pm$ Std) (#)',
    scale=1,
    bin_size=250,
    filename='results_report/emissions/nvpm_n_mean_std_altitude.png',
    log_y=True,
    skip_cfm1990=False,
    legend_location='lower left',
    legend_axis=0,
    y_limits=(1e15, 8e17)
)

plot_metric_vs_altitude(
    final_df, value_column='fuel_flow',
    y_label='Fuel Flow (Mean $\pm$ Std) (kg/s)',
    scale=1,
    bin_size=250,
    filename='results_report/emissions/fuel_flow_mean_std_altitude.png',
    log_y=False,
    skip_cfm1990=True,
    legend_location='lower left',
    legend_axis=0
)

def plot_metric_vs_altitude_saf(
    final_df,
    value_column,
    y_label,
    filename,
    scale=1.0,
    bin_size=250,
    log_y=False,
    legend_location='upper left',
    legend_axis=2,
    y_limits=None
):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, phase in enumerate(flight_phases):
        ax = axs[i]

        for engine in ['GTF2035', 'GTF2035_wi']:
            for saf in [0, 20, 100]:
                subset = final_df[
                    (final_df['engine'] == engine) &
                    (final_df['flight_phase'] == phase) &
                    (final_df['saf_level'] == saf)
                ]

                if not subset.empty:
                    stats = compute_binned_stats(
                        subset,
                        value_column=value_column,
                        bin_size=bin_size,
                        scale=scale
                    )
                    ax.fill_between(stats.index / 1000,
                                    stats['mean'] - stats['std'],
                                    stats['mean'] + stats['std'],
                                    color=saf_colors[(engine, saf)], alpha=0.2)
                    ax.plot(stats.index / 1000, stats['mean'], color=saf_colors[(engine, saf)], linewidth=2.0,
                            label=f"{engine_display_names[engine]} SAF{saf}")


        ax.set_xlabel('Altitude (km)', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.set_title(phase_titles[phase], fontsize=16)
        ax.grid(True)

        if log_y:
            ax.set_yscale('log')
        if y_limits is not None:
            ax.set_ylim(y_limits)

    axs[0].set_ylabel(y_label, fontsize=16)
    axs[0].tick_params(axis='y', labelsize=12)

    # Legend creation
    handles, labels = axs[legend_axis].get_legend_handles_labels()
    legend_handles = []
    seen = set()
    for handle, label in zip(handles, labels):
        if label not in seen:
            legend_handles.append(
                mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label)
            )
            seen.add(label)

    axs[legend_axis].legend(handles=legend_handles, loc=legend_location, title='Engine & SAF Level', frameon=True)

    plt.tight_layout()
    plt.savefig(filename, format='png')
    # plt.show()

plot_metric_vs_altitude_saf(
    final_df,
    value_column='nvpm_ei_n',
    y_label=r'$EI_{\mathrm{nvPM}}$ (Mean $\pm$ Std) (# / kg Fuel)',
    filename='results_report/emissions/nvpm_ei_saf_mean_std_altitude.png',
    scale=1,
    bin_size=250,
    log_y=True,
    y_limits=(5e13, 5e15),
    legend_location='lower left',
    legend_axis=0
)

plot_metric_vs_altitude_saf(
    final_df,
    value_column='nvpm_n',
    y_label='nvPM Number (Mean $\pm$ Std) (#)',
    filename='results_report/emissions/nvpm_n_saf_mean_std_altitude.png',
    scale=1,
    bin_size=250,
    log_y=True,
    y_limits=(2e14, 2e17),
    legend_location='lower left',
    legend_axis=0
)
plt.show()



#
# # Line Plots - With SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine in ['GTF2035', 'GTF2035_wi']:
#         for saf in [0, 20, 100]:
#             subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
#             subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]  # Apply climb phase filter
#             if not subset.empty:
#                 avg_subset = compute_avg_with_gaps(subset)
#                 ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=saf_colors[(engine, saf)], alpha=1.0,
#                         label=f"{engine_display_names[engine]} SAF{saf}")
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     # ax.legend(title="Engine & SAF Level")
#
# axs[0].set_ylabel(f'$EI_{{\\mathrm{{NOx}}}}$ (g/kg Fuel)')
# fig.suptitle(f"$EI_{{\\mathrm{{NOx}}}}$ Average vs Time in Minutes (With SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[2].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))
#
# # Apply the updated handles to the legend
# axs[2].legend(handles=legend_handles, loc='upper left', title='Engine & SAF Level', frameon=True)
# plt.savefig('results_report/emissions/ei_nox_saf_average_waypoints.png', format='png')
# # plt.show()
#
# """ei nvPM"""
#
# # Define flight phases
# flight_phases = ['climb', 'cruise', 'descent']
# phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}
#
# # Default engine display names and colors
# engine_display_names = {
#     'GTF1990': 'CFM1990',
#     'GTF2000': 'CFM2008',
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
# saf_colors = {
#     ('GTF2035', 0): 'tab:red',
#     ('GTF2035', 20): 'tab:pink',
#     ('GTF2035', 100): 'tab:grey',
#     ('GTF2035_wi', 0): default_colors[4],
#     ('GTF2035_wi', 20): 'tab:olive',
#     ('GTF2035_wi', 100): 'tab:cyan'
# }
#
# # Function to introduce NaN gaps in the dataset
# def introduce_gaps(df):
#     df = df.sort_values(by='index')
#     index_diff = df['index'].diff()
#     gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
#     df.loc[index_diff > gap_threshold, 'nvpm_ei_n'] = None  # Assign NaN where gaps exist
#     return df
#
# # Function to ensure gaps persist in average computation
# def compute_avg_with_gaps(df):
#     df = introduce_gaps(df)
#     avg_subset = df.groupby('index')['nvpm_ei_n'].mean()
#     return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))
#
# # Scatter Plots - Without SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
#         subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#         if not subset.empty:
#             ax.scatter(subset['index'], subset['nvpm_ei_n'], label=engine_display_names[engine], marker=style['marker'],
#                        color=style['color'], alpha=0.3, s=10)
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     ax.set_yscale('log')
#     # ax.legend(title="Engine")
#
# axs[0].set_ylabel('$EI_{{\\mathrm{{nvPM,number}}}}$ (#/kg fuel)')
# fig.suptitle("$EI_{{\\mathrm{{nvPM,number}}}}$ vs Time in Minutes (Without SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     # Extract color from scatter plot (PathCollection)
#     rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
#     solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
#     marker = handle.get_paths()[0]  # Keep marker style consistent
#
#     legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
#                                         linestyle='None', markersize=8, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True, markerscale=1.0)
# plt.savefig('results_report/emissions/ei_nvpm_no_saf_scatter_waypoints.png', format='png')
# # plt.show()
#
# # Scatter Plots - With SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine in ['GTF2035', 'GTF2035_wi']:
#         for saf in [0, 20, 100]:
#             subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
#             subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#             if not subset.empty:
#                 label = f"{engine_display_names[engine]} SAF{saf}"
#                 ax.scatter(subset['index'], subset['nvpm_ei_n'], label=label, marker=engine_groups[engine]['marker'],
#                            color=saf_colors[(engine, saf)], alpha=0.3, s=10)
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     ax.set_yscale('log')
#     # ax.legend(title="Engine & SAF Level")
#
# axs[0].set_ylabel('$EI_{{\\mathrm{{nvPM,number}}}}$ (#/kg fuel)')
# fig.suptitle("$EI_{{\\mathrm{{nvPM,number}}}}$ vs Time in Minutes (With SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     # Extract color from scatter plot (PathCollection)
#     rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
#     solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
#     marker = handle.get_paths()[0]  # Keep marker style consistent
#
#     legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
#                                         linestyle='None', markersize=8, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True, markerscale=1.0)
# plt.savefig('results_report/emissions/ei_nvpm_saf_scatter_waypoints.png', format='png')
# # plt.show()
#
# # Line Plots - Without SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
#         subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#         if not subset.empty:
#             avg_subset = compute_avg_with_gaps(subset)
#             ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     ax.set_yscale('log')
#     # ax.legend(title="Engine")
#
# axs[0].set_ylabel(f'$EI_{{\\mathrm{{nvPM,number}}}}$ (#/kg fuel)')
# fig.suptitle(f"$EI_{{\\mathrm{{nvPM,number}}}}$ Average vs Time in Minutes (Without SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True)
# plt.savefig('results_report/emissions/ei_nvpm_no_saf_average_waypoints.png', format='png')
# # plt.show()
#
# # Line Plots - With SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine in ['GTF2035', 'GTF2035_wi']:
#         for saf in [0, 20, 100]:
#             subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
#             subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#             if not subset.empty:
#                 avg_subset = compute_avg_with_gaps(subset)
#                 ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=saf_colors[(engine, saf)], alpha=1.0,
#                         label=f"{engine_display_names[engine]} SAF{saf}")
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     ax.set_yscale('log')
#     # ax.legend(title="Engine & SAF Level")
#
# axs[0].set_ylabel(f'$EI_{{\\mathrm{{nvPM,number}}}}$ (#/kg fuel)')
# fig.suptitle(f"$EI_{{\\mathrm{{nvPM,number}}}}$ Average vs Time in Minutes (With SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True)
# plt.savefig('results_report/emissions/ei_nvpm_saf_average_waypoints.png', format='png')
# # plt.show()
#
# """FUEL FLOW"""
#
# # Define flight phases
# flight_phases = ['climb', 'cruise', 'descent']
# phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}
#
# # Default engine display names and colors
# engine_display_names = {
#     'GTF1990': 'CFM1990',
#     'GTF2000': 'CFM2008',
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
# saf_colors = {
#     ('GTF2035', 0): 'tab:red',
#     ('GTF2035', 20): 'tab:pink',
#     ('GTF2035', 100): 'tab:grey',
#     ('GTF2035_wi', 0): default_colors[4],
#     ('GTF2035_wi', 20): 'tab:olive',
#     ('GTF2035_wi', 100): 'tab:cyan'
# }
#
# # Function to introduce NaN gaps in the dataset
# def introduce_gaps(df):
#     df = df.sort_values(by='index')
#     index_diff = df['index'].diff()
#     gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
#     df.loc[index_diff > gap_threshold, 'fuel_flow'] = None  # Assign NaN where gaps exist
#     return df
#
# # Function to ensure gaps persist in average computation
# def compute_avg_with_gaps(df):
#     df = introduce_gaps(df)
#     avg_subset = df.groupby('index')['fuel_flow'].mean()
#     return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))
#
# # Scatter Plots - Without SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
#         subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#         if not subset.empty:
#             ax.scatter(subset['index'], subset['fuel_flow'], label=engine_display_names[engine], marker=style['marker'],
#                        color=style['color'], alpha=0.3, s=10)
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     #ax.set_yscale('log')
#     # ax.legend(title="Engine")
#
# axs[0].set_ylabel('Fuel Flow (kg/s)')
# fig.suptitle("Fuel Flow vs Time in Minutes (Without SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     # Extract color from scatter plot (PathCollection)
#     rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
#     solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
#     marker = handle.get_paths()[0]  # Keep marker style consistent
#
#     legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
#                                         linestyle='None', markersize=8, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True, markerscale=1.0)
# plt.savefig('results_report/emissions/fuel_no_saf_scatter_waypoints.png', format='png')
# # plt.show()
#
# # Scatter Plots - With SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine in ['GTF2035', 'GTF2035_wi']:
#         for saf in [0, 20, 100]:
#             subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
#             subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#             if not subset.empty:
#                 label = f"{engine_display_names[engine]} SAF{saf}"
#                 ax.scatter(subset['index'], subset['fuel_flow'], label=label, marker=engine_groups[engine]['marker'],
#                            color=saf_colors[(engine, saf)], alpha=0.3, s=10)
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     #ax.set_yscale('log')
#     # ax.legend(title="Engine & SAF Level")
#
# axs[0].set_ylabel('Fuel Flow (kg/s)')
# fig.suptitle("Fuel Flow vs Time in Minutes (With SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     # Extract color from scatter plot (PathCollection)
#     rgba_color = handle.get_facecolor()[0]  # Extract first color in RGBA format
#     solid_color = (rgba_color[0], rgba_color[1], rgba_color[2], 1.0)  # Force alpha to 1.0
#     marker = handle.get_paths()[0]  # Keep marker style consistent
#
#     legend_handles.append(mlines.Line2D([], [], color=solid_color, marker=marker,
#                                         linestyle='None', markersize=8, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True, markerscale=1.0)
# plt.savefig('results_report/emissions/fuel_saf_scatter_waypoints.png', format='png')
# # plt.show()
#
# # Line Plots - Without SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
#         subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#         if not subset.empty:
#             avg_subset = compute_avg_with_gaps(subset)
#             ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     #ax.set_yscale('log')
#     # ax.legend(title="Engine")
#
# axs[0].set_ylabel('Fuel Flow (kg/s)')
# fig.suptitle("Fuel Flow Average vs Time in Minutes (Without SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True)
# plt.savefig('results_report/emissions/fuel_no_saf_average_waypoints.png', format='png')
# # plt.show()
#
# # Line Plots - With SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine in ['GTF2035', 'GTF2035_wi']:
#         for saf in [0, 20, 100]:
#             subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
#             subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#             if not subset.empty:
#                 avg_subset = compute_avg_with_gaps(subset)
#                 ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=saf_colors[(engine, saf)], alpha=1.0,
#                         label=f"{engine_display_names[engine]} SAF{saf}")
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     #ax.set_yscale('log')
#     # ax.legend(title="Engine & SAF Level")
#
# axs[0].set_ylabel('Fuel Flow (kg/s)')
# fig.suptitle("Fuel Flow Average vs Time in Minutes (With SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True)
# plt.savefig('results_report/emissions/fuel_saf_average_waypoints.png', format='png')
# # plt.show()
#
# """NOx instead of EI_NOx"""
#
# # Define flight phases
# flight_phases = ['climb', 'cruise', 'descent']
# phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}
#
# # Default engine display names and colors
# engine_display_names = {
#     'GTF1990': 'CFM1990',
#     'GTF2000': 'CFM2008',
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
# saf_colors = {
#     ('GTF2035', 0): 'tab:red',
#     ('GTF2035', 20): 'tab:pink',
#     ('GTF2035', 100): 'tab:grey',
#     ('GTF2035_wi', 0): default_colors[4],
#     ('GTF2035_wi', 20): 'tab:olive',
#     ('GTF2035_wi', 100): 'tab:cyan'
# }
#
# # Function to introduce NaN gaps in the dataset
# def introduce_gaps(df):
#     df = df.sort_values(by='index')
#     index_diff = df['index'].diff()
#     gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
#     df.loc[index_diff > gap_threshold, 'nox'] = None  # Assign NaN where gaps exist
#     return df
#
# # Function to ensure gaps persist in average computation
# def compute_avg_with_gaps(df):
#     df = introduce_gaps(df)
#     avg_subset = df.groupby('index')['nox'].mean()
#     return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))
#
# # Line Plots - Without SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
#         subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]  # Apply climb phase filter
#         if not subset.empty:
#             avg_subset = compute_avg_with_gaps(subset)
#             ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     # ax.legend(title="Engine")
#
# axs[0].set_ylabel('NOx (kg)')
# fig.suptitle("NOx Average vs Time in Minutes (Without SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[2].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))
#
# # Apply the updated handles to the legend
# axs[2].legend(handles=legend_handles, loc='upper left', title='Engine', frameon=True)
# plt.savefig('results_report/emissions/nox_no_saf_average_waypoints.png', format='png')
#
# """nvPM instead of EI_nvPM"""
#
# # Define flight phases
# flight_phases = ['climb', 'cruise', 'descent']
# phase_titles = {'climb': 'Climb Phase', 'cruise': 'Cruise Phase', 'descent': 'Descent Phase'}
#
# # Default engine display names and colors
# engine_display_names = {
#     'GTF1990': 'CFM1990',
#     'GTF2000': 'CFM2008',
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
# saf_colors = {
#     ('GTF2035', 0): 'tab:red',
#     ('GTF2035', 20): 'tab:pink',
#     ('GTF2035', 100): 'tab:grey',
#     ('GTF2035_wi', 0): default_colors[4],
#     ('GTF2035_wi', 20): 'tab:olive',
#     ('GTF2035_wi', 100): 'tab:cyan'
# }
#
# # Function to introduce NaN gaps in the dataset
# def introduce_gaps(df):
#     df = df.sort_values(by='index')
#     index_diff = df['index'].diff()
#     gap_threshold = 5  # Define a threshold to introduce NaNs when gaps appear
#     df.loc[index_diff > gap_threshold, 'nvpm_n'] = None  # Assign NaN where gaps exist
#     return df
#
# # Function to ensure gaps persist in average computation
# def compute_avg_with_gaps(df):
#     df = introduce_gaps(df)
#     avg_subset = df.groupby('index')['nvpm_n'].mean()
#     return avg_subset.reindex(range(df['index'].min(), df['index'].max() + 1))
#
# # Line Plots - Without SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine, style in engine_groups.items():
#         subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == 0)]
#         subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#         if not subset.empty:
#             avg_subset = compute_avg_with_gaps(subset)
#             ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=style['color'], alpha=1.0, label=engine_display_names[engine])
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     ax.set_yscale('log')
#     # ax.legend(title="Engine")
#
# axs[0].set_ylabel('nvPM Number (#)')
# fig.suptitle("nvPM Number Average vs Time in Minutes (Without SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# # Create new handles with solid lines instead of scatter markers
# legend_handles = []
# for handle, label in zip(handles, labels):
#     legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine', frameon=True)
# plt.savefig('results_report/emissions/nvpm_no_saf_average_waypoints.png', format='png')
# # plt.show()
#
# # Line Plots - With SAF
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#
# for i, phase in enumerate(flight_phases):
#     ax = axs[i]
#     for engine in ['GTF2035', 'GTF2035_wi']:
#         for saf in [0, 20, 100]:
#             subset = final_df[(final_df['engine'] == engine) & (final_df['flight_phase'] == phase) & (final_df['saf_level'] == saf)]
#             subset = subset[(subset['flight_phase'] != 'climb') | (subset['index'] <= 39)]
#             if not subset.empty:
#                 avg_subset = compute_avg_with_gaps(subset)
#                 ax.plot(avg_subset.index, avg_subset.values, linestyle='-', color=saf_colors[(engine, saf)], alpha=1.0,
#                         label=f"{engine_display_names[engine]} SAF{saf}")
#     ax.set_xlabel('Time in minutes')
#     ax.set_title(phase_titles[phase])
#     ax.set_yscale('log')
#     # ax.legend(title="Engine & SAF Level")
#
# axs[0].set_ylabel('nvPM Number (#)')
# fig.suptitle("nvPM Number Average vs Time in Minutes (With SAF)", fontsize=14)
# plt.tight_layout()
# handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = []
# for handle, label in zip(handles, labels):
#     legend_handles.append(mlines.Line2D([], [], color=handle.get_color(), linestyle='-', linewidth=2, label=label))
#
# # Apply the updated handles to the legend
# axs[0].legend(handles=legend_handles, loc='lower left', title='Engine & SAF Level', frameon=True)
# plt.savefig('results_report/emissions/nvpm_saf_average_waypoints.png', format='png')
# # plt.show()

final_df['waypoint_key'] = (
    final_df['trajectory'] + '_' +
    final_df['season'] + '_' +
    final_df['diurnal'] + '_' +
    final_df['index'].astype(str)
)
# final_df = final_df[
#     (final_df['accf_sac_pcfa'].fillna(0) != 0) &
#     (final_df['accf_sac_issr'].fillna(0) != 0)
# ]
# Add a new column to represent the full engine config
final_df['engine_config'] = final_df['engine'] + '_SAF' + final_df['saf_level'].astype(str)



# # Select both values to pivot
# df_pcfa_issr = final_df[['waypoint_key', 'engine_config', 'accf_sac_pcfa', 'accf_sac_issr']].copy()
#
# # Pivot both metrics separately
# pcfa_wide = df_pcfa_issr.pivot(index='waypoint_key', columns='engine_config', values='accf_sac_pcfa')
# issr_wide = df_pcfa_issr.pivot(index='waypoint_key', columns='engine_config', values='accf_sac_issr')
#
# all_pcfa_zero = (pcfa_wide.fillna(0) == 0).all(axis=1)
# all_issr_zero = (issr_wide.fillna(0) == 0).all(axis=1)
#
# both_zero_waypoints = all_pcfa_zero & all_issr_zero
# print(f"Number of waypoints where pcfa and issr are both 0 for all engines: {both_zero_waypoints.sum()}")
# # 1. Get the waypoint_keys to drop
# waypoints_to_drop = both_zero_waypoints[both_zero_waypoints].index.tolist()
#
# # 2. Filter final_df to keep only non-zero-relevant rows
# final_df = final_df[~final_df['waypoint_key'].isin(waypoints_to_drop)]
#
#
#
# pcfa_df = final_df[['waypoint_key', 'engine_config', 'accf_sac_pcfa']].copy()
#
# pcfa_wide = pcfa_df.pivot(index='waypoint_key', columns='engine_config', values='accf_sac_pcfa')
#
# baseline = pcfa_wide['GTF1990_SAF0']
#
# relative_diff = (
#     pcfa_wide.subtract(baseline, axis=0)
#     .divide(baseline.replace(0, np.nan), axis=0)
#     * 100
# ).fillna(0)
#
# relative_diff = relative_diff.reset_index()
#
engine_order = [
    'GTF1990_SAF0',
    'GTF2000_SAF0',
    'GTF_SAF0',
    'GTF2035_SAF0',
    'GTF2035_SAF20',
    'GTF2035_SAF100',
    'GTF2035_wi_SAF0',
    'GTF2035_wi_SAF20',
    'GTF2035_wi_SAF100'
]
#
# mean_diff = relative_diff.drop(columns='waypoint_key').mean().reindex(engine_order)
# print(mean_diff)
# # Merge diurnal info back into relative_diff using waypoint_key
# diurnal_map = final_df.drop_duplicates(subset=['waypoint_key'])[['waypoint_key', 'diurnal']]
# relative_diff = relative_diff.merge(diurnal_map, on='waypoint_key', how='left')
#
# # Helper to get stats
# def compute_stats(df, label):
#     subset = df[engine_order]
#     return pd.DataFrame({
#         f'{label}_mean': subset.mean(),
#         f'{label}_min': subset.min(),
#         f'{label}_max': subset.max()
#     })
#
# # Full set
# all_stats = compute_stats(relative_diff, 'all')
#
# # Daytime only
# daytime_stats = compute_stats(relative_diff[relative_diff['diurnal'] == 'daytime'], 'day')
#
# # Nighttime only
# nighttime_stats = compute_stats(relative_diff[relative_diff['diurnal'] == 'nighttime'], 'night')
#
# # Combine them into one table
# summary_stats = pd.concat([all_stats, daytime_stats, nighttime_stats], axis=1)
# pd.set_option('display.float_format', lambda x: f'{x:6.2f}')
# pd.set_option('display.max_columns', None)  # Show all columns
# print(summary_stats)
# # Create a boolean DataFrame: True where engine_config PCFA > baseline (GTF1990_SAF0)
# pcfa_higher_than_baseline = pcfa_wide.gt(pcfa_wide['GTF1990_SAF0'], axis=0)
#
# # Drop the baseline column itself (it's always False or NaN)
# pcfa_higher_than_baseline = pcfa_higher_than_baseline.drop(columns='GTF1990_SAF0')
#
# # Count how many times each engine config is higher than GTF1990
# higher_count = pcfa_higher_than_baseline.sum().reindex(engine_order[1:])  # skip baseline in result
#
# # Total number of valid comparisons (exclude NaNs)
# valid_comparisons = pcfa_wide.notna().sum().reindex(engine_order[1:])  # again skip baseline
#
# # Calculate percentage frequency of being higher
# higher_freq_pct = (higher_count / valid_comparisons * 100).round(2)
#
# # Combine into a summary table
# freq_summary = pd.DataFrame({
#     'times_higher_than_GTF1990': higher_count,
#     'total_valid': valid_comparisons,
#     'frequency_pct': higher_freq_pct
# })
#
# print(freq_summary)
#
# # Step 1: Get wide format again
# pcfa_df = final_df[['waypoint_key', 'engine_config', 'accf_sac_pcfa', 'diurnal']].copy()
# pcfa_wide = pcfa_df.pivot(index='waypoint_key', columns='engine_config', values='accf_sac_pcfa')
#
# # Step 2: Map diurnal info
# diurnal_map = final_df.drop_duplicates(subset=['waypoint_key'])[['waypoint_key', 'diurnal']]
# pcfa_wide = pcfa_wide.merge(diurnal_map, on='waypoint_key', how='left')
#
# # Step 3: Full, day, and night subsets
# day_df = pcfa_wide[pcfa_wide['diurnal'] == 'daytime']
# night_df = pcfa_wide[pcfa_wide['diurnal'] == 'nighttime']
#
# # Drop 'diurnal' for analysis
# def compute_frequency(df):
#     df = df.drop(columns='diurnal')
#
#     # Make sure everything is numeric
#     df = df.apply(pd.to_numeric, errors='coerce')
#
#     baseline = df['GTF1990_SAF0']
#     comparison = df.gt(baseline, axis=0).drop(columns='GTF1990_SAF0')
#
#     total = df.notna().drop(columns='GTF1990_SAF0').sum()
#     higher = comparison.sum()
#     freq = (higher / total * 100).round(2)
#
#     return pd.DataFrame({
#         'count_higher': higher,
#         'total_valid': total,
#         'frequency_pct': freq
#     })
#
#
# # Step 4: Compute for each time group
# freq_all = compute_frequency(pcfa_wide).rename(columns=lambda c: f"all_{c}")
# freq_day = compute_frequency(day_df).rename(columns=lambda c: f"day_{c}")
# freq_night = compute_frequency(night_df).rename(columns=lambda c: f"night_{c}")
#
# # Step 5: Combine
# freq_summary = pd.concat([freq_all, freq_day, freq_night], axis=1).reindex(engine_order[1:])  # exclude baseline
#
# # Display cleanly
# import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.float_format', lambda x: f'{x:.2f}')
#
# print(freq_summary)
#
# # Compare with GTF1990 baseline
# baseline = pcfa_wide['GTF1990_SAF0']
#
# # Prepare result storage
# deviation_counts = []
#
# # Loop through engines from index 1 onward
# for i in range(1, len(engine_order)):
#     current_engine = engine_order[i]
#     prior_engines = engine_order[1:i]  # exclude baseline itself
#
#     # Check where all prior engines equal the baseline
#     prior_equal = (pcfa_wide[prior_engines].values == baseline.to_numpy()[:, np.newaxis]).all(axis=1)
#
#
#     # Check where current engine is NOT equal to baseline
#     current_differs = pcfa_wide[current_engine] != baseline
#
#     # Combine conditions
#     condition = prior_equal & current_differs
#
#     # Count
#     deviation_counts.append({
#         'engine_config': current_engine,
#         'count_prior_engines_equal_baseline': condition.sum()
#     })
#
# # Convert to DataFrame
# deviation_summary = pd.DataFrame(deviation_counts).set_index('engine_config')
#
# # Show it
# print("\n🔍 Cases where prior engines matched baseline, but current engine deviated:")
# print(deviation_summary)

# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
#
# # Customize this list with missions you want to inspect
# selected_missions = [
#     # {"trajectory": "bos_fll", "diurnal": "nighttime", "season": "2023-08-06"},
#     {"trajectory": "cts_tpe", "diurnal": "nighttime", "season": "2023-11-06"},
#     # {"trajectory": "dus_tos", "diurnal": "nighttime", "season": "2023-08-06"},
#     # {"trajectory": "gru_lim", "diurnal": "nighttime", "season": "2023-08-06"},
#     # {"trajectory": "hel_kef", "diurnal": "nighttime", "season": "2023-08-06"},
#     # {"trajectory": "lhr_ist", "diurnal": "nighttime", "season": "2023-08-06"},
#     # {"trajectory": "sfo_dfw", "diurnal": "nighttime", "season": "2023-08-06"},
#     {"trajectory": "sin_maa", "diurnal": "nighttime", "season": "2023-11-06"}
# ]
#
# # Color + marker assignment per engine config
# engine_configs = final_df['engine_config'].unique()
# default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# color_map = {cfg: default_colors[i % len(default_colors)] for i, cfg in enumerate(engine_configs)}
#
# for mission in selected_missions:
#     traj = mission['trajectory']
#     season = mission['season']
#     diurnal = mission['diurnal']
#
#     # Filter the dataframe
#     mission_df = final_df[
#         (final_df['trajectory'] == traj) &
#         (final_df['season'] == season) &
#         (final_df['diurnal'] == diurnal)
#     ]
#
#     if mission_df.empty:
#         print(f"⚠️ No data found for {traj} - {season} - {diurnal}")
#         continue
#
#     # Prepare the base plot
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     # Plot the common accf_sac_issr (shared across engines)
#     issr_series = (
#         mission_df.groupby('index')['accf_sac_issr']
#         .first()
#         .sort_index()
#     )
#     ax.plot(
#         issr_series.index,
#         issr_series.values,
#         color='black',
#         linewidth=2,
#         label='aCCF ISSR'
#     )
#
#     # Plot accf_sac_sac for each engine config in defined order
#     for engine_config in engine_order:
#         subset = mission_df[mission_df['engine_config'] == engine_config].sort_values(by='index')
#         if not subset.empty:
#             clean_label = engine_config.replace("_", " ").replace("wi", "WI")
#             ax.plot(
#                 subset['index'],
#                 subset['accf_sac_sac'],
#                 label=clean_label,
#                 color=color_map[engine_config],
#                 linewidth=1.5,
#                 alpha=0.9
#             )
#
#     clean_traj = traj.replace("_", "-").upper()
#     ax.set_title(f"aCCF SAC Comparison - {clean_traj}, {season}, {diurnal.capitalize()}")
#     ax.set_xlabel("Time in Minutes")
#     ax.set_ylabel("aCCF Contrail Formation Parameters (SAC and ISSR)")
#     # Custom legend position for specific mission
#     if traj == "sin_maa" and diurnal == "nighttime":
#         ax.legend(title="Engine Config", loc='upper right', fontsize=9)
#     else:
#         ax.legend(title="Engine Config", loc='best', fontsize=9)
#     ax.grid(True)
#
#     plt.tight_layout()
#     # plt.savefig(f"results_report/accf_sac_plots/accf_comparison_{traj}_{season}_{diurnal}.png", dpi=300)
#     # plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Constants
# cp = 1004
# eps = 0.6222
#
# # Filter for the mission of interest
# mission_df = final_df[
#     (final_df['trajectory'] == 'sin_maa') &
#     (final_df['season'] == '2023-08-06') &
#     (final_df['diurnal'] == 'nighttime') &
#     (final_df['altitude'] > 8000) &
#     (final_df['flight_phase'] == 'cruise')
# ]
#
# engine_configs = mission_df['engine_config'].unique()
# g_points = []
# tcrit_points = []
# index_labels = []
#
# # Define label offsets (custom x, y offsets to avoid clutter)
# xy_offsets = {
#     'GTF_SAF0': (-10, 8),
#     'CFM2008_SAF0': (10, -10),
#     'GTF2035_SAF0': (-10, 10),
#     'GTF2035_SAF20': (10, 12),
#     'GTF2035_SAF100': (-12, -12),
#     'GTF2035WI_SAF0': (-20, 8),
#     'GTF2035WI_SAF20': (-10, -16),
#     'GTF2035WI_SAF100': (8, 6)
# }
#
# for engine in engine_configs:
#     if "GTF1990" in engine:
#         continue  # Skip GTF1990
#
#     subset = mission_df[mission_df['engine_config'] == engine]
#     if subset.empty:
#         continue
#
#     eta = subset['engine_efficiency'].mean()
#
#     # Reference row
#     ref = subset.iloc[80]
#     c0 = -2.64e-11
#     c1 = 2.46e-16
#     a = 1.17e-13
#     b = -1.04e-18
#     numerator = ref['accf_sac_aCCF_O3'] - (c0 + c1 * ref['accf_sac_geopotential'])
#     denominator = a + b * ref['accf_sac_geopotential']
#     Q = ref['LHV'] * 1000
#     EI_H2O = ref['ei_h2o']
#     P = ref['air_pressure']
#     T_amb = numerator / denominator
#
#     # Compute G
#     G = (EI_H2O * cp * P) / (eps * Q * (1 - eta))
#     if G > 0.053:
#         log_term = np.log(G - 0.053)
#         T_crit = -46.46 + 9.43 * log_term + 0.720 * log_term**2 + 273.15
#
#         g_points.append(G)
#         tcrit_points.append(T_crit)
#
#         # Clean label
#         label_clean = engine.replace("GTF2000", "CFM2008")
#         label_clean = label_clean.replace("_", " ").replace("wi", "WI")
#         index_labels.append(label_clean)
#
# # Plot theoretical curve
# G_vals = np.linspace(1.20, 1.8, 100)
# valid = G_vals > 0.053
# T_crit_curve = np.full_like(G_vals, np.nan)
# T_crit_curve[valid] = -46.46 + 9.43 * np.log(G_vals[valid] - 0.053) + 0.720 * (np.log(G_vals[valid] - 0.053))**2 + 273.15
#
# # Plot
# plt.figure(figsize=(10, 6))
# plt.plot(G_vals, T_crit_curve, label='$T_{crit}$ (K) vs G - Theory', color='blue')
# plt.scatter(g_points, tcrit_points, color='red', label='Engine')
#
# # Add annotations with adjusted offsets
# for x, y, label in zip(g_points, tcrit_points, index_labels):
#     y_offset = 12 if label == "GTF2035 SAF100" else 6  # Slightly more vertical offset for just this label
#     plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, y_offset), ha='right', fontsize=8)
#
# plt.xlabel('G')
# plt.ylabel('Critical Temperature $T_{crit}$ (K)')
# plt.title('$T_{crit}$ vs G with Engine Data\nSIN-MAA - 2023-08-06 - Nighttime')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# # plt.savefig("results_report/physics/T_crit_vs_G_cleaned_labels_noclip.png", dpi=300)
# plt.show()
#
#
# # Map trajectory names to nicer display labels (edit as needed)
# trajectory_labels = {
#     "bos_fll": "BOS → FLL",
#     "cts_tpe": "CTS → TPE",
#     "dus_tos": "DUS → TOS",
#     "gru_lim": "GRU → LIM",
#     "hel_kef": "HEL → KEF",
#     "lhr_ist": "LHR → IST",
#     "sfo_dfw": "SFO → DFW",
#     "sin_maa": "SIN → MAA"
# }
#
# # Filter by common season + diurnal
# season = "2023-08-06"
# diurnal = "nighttime"
#
# # Create the figure
# plt.figure(figsize=(12, 6))
#
# for mission in selected_missions:
#     traj = mission['trajectory']
#     this_season = mission['season']
#     this_diurnal = mission['diurnal']
#
#     if this_season != season or this_diurnal != diurnal:
#         continue
#
#     # Filter mission data
#     mission_df = final_df[
#         (final_df['trajectory'] == traj) &
#         (final_df['season'] == this_season) &
#         (final_df['diurnal'] == this_diurnal)
#     ]
#
#     if mission_df.empty:
#         print(f"⚠️ No data for {traj}")
#         continue
#
#     # Get ISSR series (only one needed per mission)
#     issr_series = (
#         mission_df.groupby('index')['accf_sac_issr']
#         .first()
#         .sort_index()
#     )
#
#     label = trajectory_labels.get(traj, traj.upper())
#     plt.plot(issr_series.index, issr_series.values, label=label, linewidth=2)
#
# # Final plot styling
# plt.title(f"ISSR aCCF Comparison across Missions\nSeason: {season} | Diurnal: {diurnal.capitalize()}", fontsize=14)
# plt.xlabel("Time in Minutes")
# plt.ylabel("aCCF ISSR Value")
# plt.grid(True)
# plt.legend(title="Mission Route", fontsize=9)
# plt.tight_layout()
# # plt.savefig(f"results_report/accf_sac_plots/issr_comparison_{season}_{diurnal}.png", dpi=300)
# plt.show()
#


