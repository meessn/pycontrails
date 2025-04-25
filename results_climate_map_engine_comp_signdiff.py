import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import math
from pycontrails.physics.thermo import e_sat_liquid, e_sat_ice
import numpy as np

def generate_engine_display(df):
    """
    Generates the engine_display column based on existing engine names and SAF levels.

    Parameters:
        df (DataFrame): The input DataFrame containing engine configurations.

    Returns:
        DataFrame: Updated DataFrame with an 'engine_display' column.
    """

    def format_engine_name(row):
        """
        Formats engine name by including SAF and water injection levels.
        """
        engine = row['engine']  # Assuming engine names are in a column called 'engine'

        # Add SAF level if present
        # Add water injection if present
        if engine == "GTF1990":
            engine = "CFM1990"
        if engine == "GTF2000":
            engine = "CFM2000"
        if 'water_injection' in row and row['water_injection'] > 0:
            print('yes')
            engine = engine.replace("_wi", "")
            engine += "WI"
        if 'saf_level' in row and row['saf_level'] != 0:
            engine += f" - {int(row['saf_level'])}"

        print(engine)

        return engine

    # Apply function to each row
    df['engine_display'] = df.apply(format_engine_name, axis=1)

    return df

def plot_trajectory_subfigures(trajectory, flight_date, time_of_day, save_fig=False):
    """
    Plots multiple subfigures showing different engine models for a selected trajectory, highlighting EF values and the trajectory path.
    """
    base_results_dir = 'main_results_figures/results'
    base_figures_dir = 'main_results_figures/figures'

    if trajectory == 'hel_kef' and flight_date == '2023-11-06' and time_of_day == 'daytime':
        engine_models = {
            "GTF": True,
            "GTF2035": True,
            "GTF2035_wi": True
        }
    else:
        engine_models = {
            "GTF1990": True,
            "GTF2000": True,
            "GTF": True,
            "GTF2035": True,
            "GTF2035_wi": True
        }
    print(engine_models)
    # SAF values based on engine model
    saf_dict = {
        "0": True,
        "20": True,
        "100": True
    }

    # Water injection values
    water_injection_values = {
        "GTF2035_wi": "15",
        "default": "0"
    }

    # Filter selected engine models
    selected_engines = [engine for engine, enabled in engine_models.items() if enabled]

    df_dict = {}  # Initialize before using it
    ef_min_global, ef_max_global = float('inf'), float('-inf')

    for engine in selected_engines:
        saf_levels = saf_dict.keys() if engine in ["GTF2035", "GTF2035_wi"] else ["0"]
        water_injection = water_injection_values.get(engine, "0")

        for saf in saf_levels:
            parquet_path = os.path.join(base_results_dir, trajectory,
                                        f"{trajectory}_{flight_date}_{time_of_day}/climate/mees/era5model/co_cont_{engine}_{saf}_{water_injection}.parquet")
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                df['ef'] *= 0.42
                df_dict[(engine, saf)] = df
                ef_min_global = min(ef_min_global, df['ef'].min())
                ef_max_global = max(ef_max_global, df['ef'].max())
            else:
                df_dict[(engine, saf)] = None

    # Now, determine number of subplots based on the actual number of datasets

    valid_datasets = []

    for engine in selected_engines:
        saf_levels = saf_dict.keys() if engine in ["GTF2035", "GTF2035_wi"] else ["0"]
        for saf in saf_levels:
            valid_datasets.append((engine, saf))
    print(valid_datasets)
    num_subplots = len(valid_datasets)
    ncols = 3
    nrows = math.ceil(num_subplots / ncols) if num_subplots > 0 else 1

    # Create the figure with the correct number of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows),
                             subplot_kw={'projection': ccrs.PlateCarree()})

    # Ensure axes is always a list
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    subplot_idx = 0

    max_abs_ef = max(abs(ef_min_global), abs(ef_max_global))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs_ef, vcenter=0, vmax=max_abs_ef)

    # Initialize global bounds
    global_lon_min, global_lon_max = float('inf'), float('-inf')
    global_lat_min, global_lat_max = float('inf'), float('-inf')

    # Compute bounds across all datasets
    for (engine, saf), df in df_dict.items():
        if df is not None:
            global_lon_min = min(global_lon_min, df['longitude'].min())
            global_lon_max = max(global_lon_max, df['longitude'].max())
            global_lat_min = min(global_lat_min, df['latitude'].min())
            global_lat_max = max(global_lat_max, df['latitude'].max())
        else:
            continue



    for (engine, saf), df in df_dict.items():
        war_value = "15" if engine == "GTF2035_wi" else "0"
        flight_path_csv = os.path.join(base_results_dir, trajectory,
                                       f"{trajectory}_{flight_date}_{time_of_day}/climate/mees/era5model/{engine}_SAF_{saf}_A20N_full_WAR_{war_value}_climate.csv")

        if os.path.exists(flight_path_csv):
            flight_df = pd.read_csv(flight_path_csv)

            # Update global bounds with trajectory path
            global_lon_min = min(global_lon_min, flight_df['longitude'].min())
            global_lon_max = max(global_lon_max, flight_df['longitude'].max())
            global_lat_min = min(global_lat_min, flight_df['latitude'].min())
            global_lat_max = max(global_lat_max, flight_df['latitude'].max())

    # Expand bounds by ±2 degrees for better visibility
    if trajectory == 'hel_kef':
        buffer_lon = 5
        buffer_lat = 20  # Increase latitude buffer
    else:
        buffer_lon = buffer_lat = 1

    global_lon_min -= buffer_lon
    global_lon_max += buffer_lon
    global_lat_min -= buffer_lat
    global_lat_max += buffer_lat

    for (engine, saf) in valid_datasets:
        df = df_dict[(engine, saf)]
        ax = axes[subplot_idx]
        ax.add_feature(cfeature.COASTLINE, edgecolor='#D0D0D0')
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#D0D0D0')
        ax.add_feature(cfeature.LAND, edgecolor='#D0D0D0', facecolor='#EAEAEA')

        # Scatter plot (EF values)
        if df is not None:
            sc = ax.scatter(df['longitude'], df['latitude'], c=df['ef'], cmap='coolwarm', norm=norm, alpha=0.8,
                        transform=ccrs.PlateCarree())

        war_value = "15" if engine == "GTF2035_wi" else "0"
        flight_path_csv = os.path.join(base_results_dir, trajectory,
                                       f"{trajectory}_{flight_date}_{time_of_day}/climate/mees/era5model/{engine}_SAF_{saf}_A20N_full_WAR_{war_value}_climate.csv")
        # Plot trajectory if available
        if os.path.exists(flight_path_csv):
            flight_df = pd.read_csv(flight_path_csv)
            if 'cocip_atr20' not in flight_df.columns:
                flight_df['cocip_atr20'] = 0  # Create column and fill with zeros
            else:
                flight_df['cocip_atr20'] = flight_df['cocip_atr20'].fillna(0)  # Fill existing NaNs with 0
                flight_df['cocip_atr20'] *= 0.42

            for i in range(len(flight_df) - 1):
                lon_segment = [flight_df['longitude'].iloc[i], flight_df['longitude'].iloc[i + 1]]
                lat_segment = [flight_df['latitude'].iloc[i], flight_df['latitude'].iloc[i + 1]]

                # Choose color: Green if ATR20 is nonzero, Black otherwise
                color = 'green' if flight_df['cocip_atr20'].iloc[i] != 0 else 'black'

                ax.plot(lon_segment, lat_segment, color=color, linewidth=1,
                        transform=ccrs.PlateCarree())

        # Apply global bounds
        ax.set_extent([global_lon_min, global_lon_max, global_lat_min, global_lat_max], crs=ccrs.PlateCarree())

        # Rename specific engines
        title_engine = engine.replace("GTF1990", "CFM1990").replace("GTF2000", "CFM2000").replace("GTF2035_wi",
                                                                                                  "GTF2035WI")

        # Construct title without "- 0" when SAF is 0
        title = f"{title_engine}" if saf == "0" else f"{title_engine} - {saf}"
        ax.set_title(title)
        subplot_idx += 1

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # Colorbar axis
    # fig.colorbar(sc, cax=cbar_ax, label="Contrail EF (J)")
    for ax in axes[subplot_idx:]:
        ax.set_visible(False)
    # Adjust spacing between subplots to fix layout issues
    plt.subplots_adjust(left=0.05, right=0.80, top=0.92, bottom=0.05, wspace=0.1, hspace=0.1)
    plt.suptitle(f"CoCiP Contrail EF Evolution", fontsize=16,
                 fontweight='bold')
    # Move the colorbar slightly left so it's not too squeezed
    cbar_ax = fig.add_axes([0.82, 0.15, 0.03, 0.7])  # (left, bottom, width, height)
    fig.colorbar(sc, cax=cbar_ax, label="Contrail EF (J), including efficacy")

    if save_fig:
        save_path = f'results_report/specialcases/ef_sign_change_compared_to_1990_{trajectory}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_path}")

    # plt.show()
    print(f"Subfigure plot generated successfully for {trajectory} on {flight_date} during {time_of_day}.")


def plot_cocip_atr20_evolution(trajectory, flight_date, time_of_day, save_fig=False):
    """
    Plots CoCiP ATR20 evolution along the trajectory for different engine models and SAF levels.
    """
    base_results_dir = 'main_results_figures/results'
    base_figures_dir = 'main_results_figures/figures'

    if trajectory == 'hel_kef' and flight_date == '2023-11-06' and time_of_day == 'daytime':
        engine_models = {
            "GTF": True,
            "GTF2035": True,
            "GTF2035_wi": True
        }
    else:
        engine_models = {
            "GTF1990": True,
            "GTF2000": True,
            "GTF": True,
            "GTF2035": True,
            "GTF2035_wi": True
        }

    # SAF values
    saf_dict = {
        "0": True,
        "20": True,
        "100": True
    }

    # Water injection values
    water_injection_values = {
        "GTF2035_wi": "15",
        "default": "0"
    }

    # Filter selected engine models
    selected_engines = [engine for engine, enabled in engine_models.items() if enabled]

    # Initialize global bounds
    global_lon_min, global_lon_max = float('inf'), float('-inf')
    global_lat_min, global_lat_max = float('inf'), float('-inf')
    atr_min_global, atr_max_global = float('inf'), float('-inf')

    df_dict = {}

    # Load data and compute bounds
    for engine in selected_engines:
        saf_levels = saf_dict.keys() if engine in ["GTF2035", "GTF2035_wi"] else ["0"]
        water_injection = water_injection_values.get(engine, "0")

        for saf in saf_levels:
            flight_path_csv = os.path.join(base_results_dir, trajectory,
                                           f"{trajectory}_{flight_date}_{time_of_day}/climate/mees/era5model/{engine}_SAF_{saf}_A20N_full_WAR_{water_injection}_climate.csv")

            if os.path.exists(flight_path_csv):
                flight_df = pd.read_csv(flight_path_csv)

                if 'cocip_atr20' not in flight_df.columns:
                    flight_df['cocip_atr20'] = 0  # Create column and fill with zeros
                else:
                    flight_df['cocip_atr20'] = flight_df['cocip_atr20'].fillna(0)  # Fill existing NaNs with 0
                    flight_df['cocip_atr20'] *= 0.42

                # Store dataframe for later plotting
                df_dict[(engine, saf)] = flight_df

                # Update global bounds
                global_lon_min = min(global_lon_min, flight_df['longitude'].min())
                global_lon_max = max(global_lon_max, flight_df['longitude'].max())
                global_lat_min = min(global_lat_min, flight_df['latitude'].min())
                global_lat_max = max(global_lat_max, flight_df['latitude'].max())

                # Update global ATR20 min/max
                atr_min_global = min(atr_min_global, flight_df['cocip_atr20'].min())
                atr_max_global = max(atr_max_global, flight_df['cocip_atr20'].max())

    valid_datasets = [(engine, saf) for (engine, saf), df in df_dict.items() if df is not None]
    # Expand bounds by ±2 degrees
    if trajectory == 'hel_kef':
        buffer_lon = 5
        buffer_lat = 20  # Increase latitude buffer
    else:
        buffer_lon = buffer_lat = 1

    global_lon_min -= buffer_lon
    global_lon_max += buffer_lon
    global_lat_min -= buffer_lat
    global_lat_max += buffer_lat

    # Set up subplots
    num_subplots = len(df_dict)
    ncols = 3
    nrows = -(-num_subplots // ncols)  # Equivalent to math.ceil(num_subplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    # Normalize ATR20 values globally
    max_abs_atr20 = max(abs(atr_min_global), abs(atr_max_global))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs_atr20, vcenter=0.0, vmax=max_abs_atr20)

    # Plot each engine model & SAF level
    subplot_idx = 0
    for (engine, saf) in valid_datasets:
        flight_df = df_dict[(engine, saf)]
        ax = axes[subplot_idx]
        ax.add_feature(cfeature.COASTLINE, edgecolor='#D0D0D0')
        ax.add_feature(cfeature.BORDERS, linestyle=':',edgecolor='#D0D0D0')
        ax.add_feature(cfeature.LAND, edgecolor='#D0D0D0', facecolor='#EAEAEA')

        # Rename engines in titles
        title_engine = engine.replace("GTF1990", "CFM1990").replace("GTF2000", "CFM2000").replace("GTF2035_wi", "GTF2035WI")
        title = f"{title_engine}" if saf == "0" else f"{title_engine} - {saf}"
        ax.set_title(title)

        for i in range(len(flight_df) - 1):
            lon_segment = [flight_df['longitude'].iloc[i], flight_df['longitude'].iloc[i + 1]]
            lat_segment = [flight_df['latitude'].iloc[i], flight_df['latitude'].iloc[i + 1]]

            # Choose color: Green if ATR20 is nonzero, Black otherwise
            color = 'green' if flight_df['cocip_atr20'].iloc[i] != 0 else 'black'

            ax.plot(lon_segment, lat_segment, color=color, linewidth=1)

        # Scatter ATR20 values
        sc = ax.scatter(flight_df['longitude'], flight_df['latitude'], c=flight_df['cocip_atr20'], cmap='coolwarm', norm=norm,
                        alpha=0.8, label="CoCiP P-ATR20, including efficacy")

        # Apply global bounds
        ax.set_extent([global_lon_min, global_lon_max, global_lat_min, global_lat_max], crs=ccrs.PlateCarree())

        subplot_idx += 1

    for ax in axes[subplot_idx:]:
        ax.set_visible(False)
    # Adjust layout to minimize whitespace
    plt.subplots_adjust(left=0.05, right=0.75, top=0.92, bottom=0.05, wspace=0.05, hspace=0.1)
    plt.suptitle(f"CoCiP Contrail P-ATR20", fontsize=16,
                 fontweight='bold')

    # Add a single colorbar for all plots
    if num_subplots < 9:
        cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    else:
        cbar_ax = fig.add_axes([0.77, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="CoCiP P-ATR20, including efficacy")

    if save_fig:
        save_path = f'results_report/specialcases/patr20_sign_change_compared_to_1990_{trajectory}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_path}")

    # plt.show()
    print(f"CoCiP ATR20 Evolution plot generated successfully for {trajectory} on {flight_date} during {time_of_day}.")

# Example usage

import numpy as np
import matplotlib.pyplot as plt

def plot_engine_barplot(df, df_name):
    """
    Plots a grouped bar chart showing Contrail, NOx, and Total Climate Impact (Conservative & Optimistic) for each engine.

    Parameters:
        df (DataFrame): The input dataframe containing values.
        df_name (str): Name of the dataframe (for saving the plot).
    """
    # Define engines to plot
    # engines_to_plot = ['CFM1990', 'CFM2000', 'GTF', 'GTF2035', 'GTF2035 - 20', 'GTF2035 - 100', 'GTF2035WI', 'GTF2035WI - 20', 'GTF2035WI - 100']
    # saf_levels = [20, 100]  # SAF levels for GTF2035 variants

    # Define legend titles
    legend_titles = {
        'contrail_atr20_cocip_sum': 'Contrail (CoCiP)',
        'contrail_atr20_accf_cocip_pcfa_sum': 'Contrail (aCCF)',
        'nox_impact_sum': 'NOx',
        'h2o_impact_sum': 'H₂O',
        'co2_impact_cons_sum': 'CO₂ (Conservative)',
        'co2_impact_opti_sum': 'CO₂ (Optimistic)'
    }

    metric_color_map = {
        "nox_impact_sum": "tab:blue",
        "h2o_impact_sum": "tab:grey",
        "co2_impact_cons_sum": "tab:orange",
        "contrail_atr20_accf_cocip_pcfa_sum": "tab:red",
        "contrail_atr20_cocip_sum": "tab:green",
    }

    # Select relevant columns
    # metrics = ['contrail_atr20_cocip_sum', 'nox_impact_sum', 'climate_total_cons_sum', 'climate_total_opti_sum']
    metrics = ['contrail_atr20_cocip_sum', 'contrail_atr20_accf_cocip_pcfa_sum', 'nox_impact_sum', 'h2o_impact_sum', 'co2_impact_cons_sum']
    # df_filtered = df[df['engine_display'].isin(engines_to_plot)]
    # df_filtered = df
    # Compute mean values per engine type


    # Define full list of engines and labels for default case
    x_order_full = [
        "CFM1990", "CFM2000", "GTF",
        "GTF2035", "GTF2035 - 20", "GTF2035 - 100",
        "GTF2035WI", "GTF2035WI - 20", "GTF2035WI - 100"
    ]

    x_labels_full = [
        "CFM1990", "CFM2000", "GTF",
        "GTF2035", "GTF2035\n-20", "GTF2035\n-100",
        "GTF2035WI", "GTF2035WI\n-20", "GTF2035WI\n-100"
    ]

    # Conditional filter for specific case
    if df_name == "df_hel_kef_sign_gtf":
        engines_to_plot = [
            "GTF",
            "GTF2035", "GTF2035 - 20", "GTF2035 - 100",
            "GTF2035WI", "GTF2035WI - 20", "GTF2035WI - 100"
        ]
        df_filtered = df[df['engine_display'].isin(engines_to_plot)]
        x_order = engines_to_plot
        x_labels = [label.replace(" - ", "\n-") for label in engines_to_plot]
    else:
        df_filtered = df
        x_order = x_order_full
        x_labels = x_labels_full

    # Ensure ordering
    grouped = df_filtered.groupby("engine_display")[metrics].mean().reset_index()
    grouped = grouped.set_index("engine_display").reindex(x_order).reset_index()

    # Plot settings
    width = 0.13  # Width of each bar
    x = np.arange(len(x_order))  # X positions for the bars

    plt.figure(figsize=(12, 6))

    # Plot each metric as a separate bar group
    for i, metric in enumerate(metrics):
        color = metric_color_map.get(metric, None)  # Use color map; fallback to default if not found
        plt.bar(x + i * width, grouped[metric], alpha=0.7, label=legend_titles[metric], width=width, color=color)

    # Labeling and formatting
    plt.ylabel("P-ATR20 (K)")
    plt.title("Climate Impact Contributions per Engine")
    center_offset = (len(metrics) - 1) / 2 * width  # Now 4 metrics → offset = 0.3 if width=0.2
    plt.xticks(x + center_offset, x_labels, rotation=0, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save plot
    filename = f"results_report/specialcases/engine_barplot_{df_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")


"""sign flip compared to 1990"""
trajectory = 'dus_tos'
flight_date = '2023-02-06'
time_of_day = 'daytime'

plot_trajectory_subfigures(trajectory, flight_date, time_of_day, save_fig=True)
#
plot_cocip_atr20_evolution(trajectory, flight_date, time_of_day, save_fig=True)

df = pd.read_csv('results_main_simulations.csv')
df_dus_tos = df[(df['trajectory'] == trajectory) &
                     (df['season'] == flight_date) &
                     (df['diurnal'] == time_of_day)]

df_dus_tos = generate_engine_display(df_dus_tos)
plot_engine_barplot(df_dus_tos, 'df_dus_tos_sign_1990')

"""sign flip compared to gtf"""
trajectory = 'hel_kef'
flight_date = '2023-11-06'
time_of_day = 'daytime'

plot_trajectory_subfigures(trajectory, flight_date, time_of_day, save_fig=True)

plot_cocip_atr20_evolution(trajectory, flight_date, time_of_day, save_fig=True)

df = pd.read_csv('results_main_simulations.csv')
df_hel_kef = df[(df['trajectory'] == trajectory) &
                     (df['season'] == flight_date) &
                     (df['diurnal'] == time_of_day)]

df_hel_kef = generate_engine_display(df_hel_kef)
plot_engine_barplot(df_hel_kef, 'df_hel_kef_sign_gtf')

"""no contrail for 1990 2000"""
trajectory = 'bos_fll'
flight_date = '2023-08-06'
time_of_day = 'nighttime'

plot_trajectory_subfigures(trajectory, flight_date, time_of_day, save_fig=True)
#
plot_cocip_atr20_evolution(trajectory, flight_date, time_of_day, save_fig=True)

df = pd.read_csv('results_main_simulations.csv')
df_bos_fll = df[(df['trajectory'] == trajectory) &
                     (df['season'] == flight_date) &
                     (df['diurnal'] == time_of_day)]

df_bos_fll = generate_engine_display(df_bos_fll)
plot_engine_barplot(df_bos_fll, 'df_bos_fll_no_cfm')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycontrails.physics.thermo import e_sat_liquid, e_sat_ice
import os

# CONFIG
trajectory = "bos_fll"
flight_date = "bos_fll_2023-08-06_nighttime"
time_of_day = "nighttime"
weather_model = "era5model"
prediction = "mees"
aircraft = "A20N_full"
target_index = 128

engine_configs = [
    ("GTF1990", 0, 0),
    ("GTF2000", 0, 0),
    ("GTF", 0, 0),
    ("GTF2035", 0, 0),
    ("GTF2035", 20, 0),
    ("GTF2035", 100, 0),
    ("GTF2035_wi", 0, 15),
    ("GTF2035_wi", 20, 15),
    ("GTF2035_wi", 100, 15),
]

def get_engine_display_name(engine, saf, war):
    if engine == "GTF1990":
        return "CFM1990"
    elif engine == "GTF2000":
        return "CFM2008"
    elif engine == "GTF":
        return "GTF"
    elif engine == "GTF2035":
        return f"GTF2035" if saf == 0 else f"GTF2035 - {saf}"
    elif engine == "GTF2035_wi":
        return f"GTF2035WI" if saf == 0 else f"GTF2035WI - {saf}"
    else:
        return engine  # fallback

plt.figure(figsize=(10, 7))

# Saturation curves
T_K = np.linspace(222, 240, 200)
e_liquid = e_sat_liquid(T_K)
e_ice = e_sat_ice(T_K)

plt.plot(T_K, e_liquid, label='Saturation over Liquid (Pa)', color='blue', linestyle='--')
plt.plot(T_K, e_ice, label='Saturation over Ice (Pa)', color='purple', linestyle='--')

for engine, saf, war in engine_configs:
    saf_str = str(saf)
    war_str = str(war)

    filename = f"{engine}_SAF_{saf_str}_{aircraft}_WAR_{war_str}_climate.csv"
    path = f"main_results_figures/results/{trajectory}/{flight_date}/climate/{prediction}/{weather_model}/{filename}"

    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        continue

    df = pd.read_csv(path)

    row = df[df['index'] == target_index]
    if row.empty:
        print(f"⚠️ No data at index {target_index} in {filename}")
        continue

    row = row.iloc[0]
    persistent = row.get('cocip_persistent_1', 0)
    atr20 = row.get('cocip_atr20', 0)
    print(f"{engine} | SAF {saf}% | WAR {war} → persistent: {persistent}, ATR20: {atr20:.2e}")
    T_k_point = row['cocip_T_sat_liquid']
    G = row['cocip_G']

    e_point = e_sat_liquid(T_k_point)
    # if 'cocip_air_temperature_1' in row and row['cocip_air_temperature_1'] != 0 and pd.notna(
    #         row['cocip_air_temperature_1']):
    #     T_amb = row['cocip_air_temperature_1']
    #     print(T_amb)
    #     rhi_amb = row['cocip_rhi_1']
    #
    # else:
    #     T_amb = row['air_temperature']
    #     print(T_amb)
    #     rhi_amb = row['cocip_rhi']
    T_amb = row['air_temperature']
    print(T_amb)
    rhi_amb = row['cocip_rhi']
    e_amb = rhi_amb * e_sat_ice(T_amb)
        # Mixing line
    T_mix = np.linspace(222, 240, 100)
    e_mix = e_point + G * (T_mix - T_k_point)



    # Plot
    # plt.scatter([T_k_point], [e_point], s=60, label=f'{label} pt')
    # plt.plot(T_mix, e_mix, linestyle='-', label=f'{label} line')

    # Define temperature range around T_amb
    T_range = np.linspace(T_amb, 240, 200)  # or choose a narrower range if needed

    # Compute corresponding vapor pressures along the mixing line
    e_mix = e_amb + G * (T_range - T_amb)

    # Plot the mixing line
    label = get_engine_display_name(engine, saf, war)
    plt.plot(T_range, e_mix, linestyle='-', label=label)


    # if 'cocip_T_critical_sac' in row and pd.notna(row['cocip_T_critical_sac']):
    #     T_crit_sac = row['cocip_T_critical_sac']
    #     plt.axvline(
    #         x=T_crit_sac,
    #         color='gray',
    #         linestyle=(0, (2, 4)),  # dashed pattern
    #         linewidth=1.5,
    #         label=f'{label} | T_crit_sac ({T_crit_sac:.2f} K)'
    #     )

    # Plot vertical striped/dashed line at that temperature
plt.scatter([T_amb], [e_amb], s=40, marker='x', color='black', label='Ambient Conditions')
# plt.axvline(
#     x=T_amb,
#     color='black',
#     linestyle=(0, (5, 10)),  # custom dashed pattern
#     linewidth=2,
#     label=f'Ambient Temp at index {target_index} ({T_amb:.2f} K)'
# )
plt.xlabel('Temperature (K)')
plt.xlim(222, 240)
plt.ylabel('Saturation Vapor Pressure (Pa)')
plt.title(f'Schmidt-Appleman Diagram for Engines during Cruise')
plt.legend(fontsize='small', loc='upper left', ncol=2)
plt.grid(True)
plt.tight_layout()
filename = 'results_report/specialcases/sac_diagram_bos_fll.png'
plt.savefig(filename, dpi=300, bbox_inches="tight")
plt.show()

# Constants
cp = 1004
eps = 0.6222
g_points = []
tcrit_points = []
index_labels = []
universal_air_temp = None

for engine, saf, war in engine_configs:
    if engine == "GTF1990":
        continue

    saf_str = str(saf)
    war_str = str(war)

    filename = f"{engine}_SAF_{saf_str}_{aircraft}_WAR_{war_str}_climate.csv"
    path = f"main_results_figures/results/{trajectory}/{flight_date}/climate/{prediction}/{weather_model}/{filename}"

    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        continue

    df = pd.read_csv(path)

    if len(df) <= target_index:
        print(f"⚠️ Index {target_index} not found in {filename}")
        continue

    row = df.iloc[target_index]

    eta = row.get('engine_efficiency')
    if pd.isna(eta) or eta == 0:
        print(f"⚠️ Invalid engine efficiency for {filename}")
        continue

    # Store air_temperature just once
    if universal_air_temp is None:
        if 'air_temperature' in row and pd.notna(row['air_temperature']):
            universal_air_temp = row['air_temperature']

    # SAC T_crit
    c0, c1 = -2.64e-11, 2.46e-16
    a, b = 1.17e-13, -1.04e-18
    geop = row['accf_sac_geopotential']
    accf_o3 = row['accf_sac_aCCF_O3']

    numerator = accf_o3 - (c0 + c1 * geop)
    denominator = a + b * geop
    T_amb = numerator / denominator

    Q = row['LHV'] * 1000
    EI_H2O = row['ei_h2o']
    P = row['air_pressure']
    G = row['cocip_G']

    if G > 0.053:
        log_term = np.log(G - 0.053)
        # T_crit = -46.46 + 9.43 * log_term + 0.720 * log_term**2 + 273.15
        T_crit = row['cocip_T_critical_sac']
        g_points.append(G)
        tcrit_points.append(T_crit)

        label = get_engine_display_name(engine, saf, war)
        index_labels.append(label)

# Theoretical curve
G_vals = np.linspace(1.20, 1.8, 100)
valid = G_vals > 0.053
T_crit_curve = np.full_like(G_vals, np.nan)
T_crit_curve[valid] = -46.46 + 9.43 * np.log(G_vals[valid] - 0.053) + 0.720 * (np.log(G_vals[valid] - 0.053))**2 + 273.15

# Plot
plt.figure(figsize=(10, 6))
# plt.plot(G_vals, T_crit_curve, label='$T_{crit}$ Theory', color='blue')
plt.scatter(g_points, tcrit_points, color='red', label='Critical Temperature Threshold')

# Annotate
for x, y, label in zip(g_points, tcrit_points, index_labels):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 0), ha='left', fontsize=8)

# # Horizontal line for air temperature
# if universal_air_temp is not None:
#     plt.axhline(universal_air_temp, color='green', linestyle='--', linewidth=1.5,
#                 label=f'Ambient Air Temperature')

plt.xlabel('G')
plt.xlim(1.35, 1.83)
plt.ylabel('Temperature (K)')
plt.title('$T_{crit}$ vs G \nBOS-FLL - 2023-08-06 - Nighttime')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results_report/specialcases/T_crit_vs_G_with_air_temp_line.png", dpi=300)
plt.show()




# """total climate impact negative"""
# trajectory = 'sin_maa'
# flight_date = '2023-05-05'
# time_of_day = 'daytime'
#
# # plot_trajectory_subfigures(trajectory, flight_date, time_of_day, save_fig=True)
#
# # plot_cocip_atr20_evolution(trajectory, flight_date, time_of_day, save_fig=True)
#
#
#
# df = pd.read_csv('results_main_simulations.csv')
# df_sin_maa = df[(df['trajectory'] == 'sin_maa') &
#                      (df['season'] == '2023-05-05') &
#                      (df['diurnal'] == 'daytime')]
#
# df_sin_maa = generate_engine_display(df_sin_maa)
# # plot_engine_barplot(df_sin_maa, 'df_sin_maa_neg_tot_clim')

plt.show()