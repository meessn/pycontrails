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
import numpy as np

def plot_trajectory_subfigures(trajectory, flight_date, time_of_day, save_fig=False):
    """
    Plots multiple subfigures showing different engine models for a selected trajectory, highlighting EF values and the trajectory path.
    """
    base_results_dir = 'main_results_figures/results'
    base_figures_dir = 'main_results_figures/figures'

    # Engine models to run
    engine_models = {
        "GTF1990": True,
        "GTF2000": True,
        "GTF": True,
        "GTF2035": True,
        "GTF2035_wi": True
    }

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
                df_dict[(engine, saf)] = df
                ef_min_global = min(ef_min_global, df['ef'].min())
                ef_max_global = max(ef_max_global, df['ef'].max())

    # Now, determine number of subplots based on the actual number of datasets
    num_subplots = len(df_dict)
    ncols = 3
    nrows = math.ceil(num_subplots / ncols) if num_subplots > 0 else 1  # Ensure at least 1 row

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
        global_lon_min = min(global_lon_min, df['longitude'].min())
        global_lon_max = max(global_lon_max, df['longitude'].max())
        global_lat_min = min(global_lat_min, df['latitude'].min())
        global_lat_max = max(global_lat_max, df['latitude'].max())



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
    buffer = 1
    global_lon_min -= buffer
    global_lon_max += buffer
    global_lat_min -= buffer
    global_lat_max += buffer

    for (engine, saf), df in df_dict.items():
        ax = axes[subplot_idx]
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')

        # Scatter plot (EF values)
        sc = ax.scatter(df['longitude'], df['latitude'], c=df['ef'], cmap='coolwarm', norm=norm, alpha=0.8,
                        transform=ccrs.PlateCarree())

        # Plot trajectory if available
        if os.path.exists(flight_path_csv):
            flight_df = pd.read_csv(flight_path_csv)
            ax.plot(flight_df['longitude'], flight_df['latitude'], color='k', linewidth=0.8,
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

    # Adjust spacing between subplots to fix layout issues
    plt.subplots_adjust(left=0.05, right=0.80, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)

    # Move the colorbar slightly left so it's not too squeezed
    cbar_ax = fig.add_axes([0.82, 0.15, 0.03, 0.7])  # (left, bottom, width, height)
    fig.colorbar(sc, cax=cbar_ax, label="Contrail EF (J)")

    if save_fig:
        save_path = 'results_report/specialcases/sign_change_compared_to_1990_dus_tos.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_path}")

    plt.show()
    print(f"Subfigure plot generated successfully for {trajectory} on {flight_date} during {time_of_day}.")


trajectory = 'dus_tos'
flight_date = '2023-02-06'
time_of_day = 'daytime'

plot_trajectory_subfigures(trajectory, flight_date, time_of_day, save_fig=True)