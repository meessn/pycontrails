import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob


def plot_flight_data(flight_dirs):
    # Storage for min/max values across flights
    lat_min, lat_max = float('inf'), float('-inf')
    lon_min, lon_max = float('inf'), float('-inf')
    rf_lw_min, rf_lw_max = float('inf'), float('-inf')
    rf_sw_min, rf_sw_max = float('inf'), float('-inf')
    ef_min, ef_max = float('inf'), float('-inf')

    # Storage for data to plot later
    flight_data = []

    # First pass: Load data and determine global min/max values
    for flight_dir in flight_dirs:
        parquet_path = glob.glob(os.path.join(flight_dir, 'co_cont_GTF_0_0.parquet'))[0]
        csv_path = glob.glob(os.path.join(flight_dir, 'GTF_SAF_0_A20N_full_WAR_0_climate.csv'))[0]

        cocip_df = pd.read_parquet(parquet_path)
        fcocip_df = pd.read_csv(csv_path)

        lat_min = min(lat_min, cocip_df['latitude'].min())
        lat_max = max(lat_max, cocip_df['latitude'].max())
        lon_min = min(lon_min, cocip_df['longitude'].min())
        lon_max = max(lon_max, cocip_df['longitude'].max())

        rf_lw_min = min(rf_lw_min, cocip_df['rf_lw'].min())
        rf_lw_max = max(rf_lw_max, cocip_df['rf_lw'].max())

        rf_sw_min = min(rf_sw_min, cocip_df['rf_sw'].min())
        rf_sw_max = max(rf_sw_max, cocip_df['rf_sw'].max())

        ef_min = min(ef_min, cocip_df['ef'].min())
        ef_max = max(ef_max, cocip_df['ef'].max())

        flight_data.append((fcocip_df, cocip_df, os.path.basename(flight_dir)))

    print(f"Global Latitude range: {lat_min} to {lat_max}")
    print(f"Global Longitude range: {lon_min} to {lon_max}")
    print(f"Global RF_LW range: {rf_lw_min} to {rf_lw_max}")
    print(f"Global RF_SW range: {rf_sw_min} to {rf_sw_max}")
    print(f"Global EF range: {ef_min} to {ef_max}")

    # Second pass: Plotting with consistent limits
    for i, (fcocip_df, cocip_df, flight_name) in enumerate(flight_data):
        ## Long Wave RF Plot
        plt.figure()
        ax1 = plt.axes()

        ax1.plot(fcocip_df['longitude'], fcocip_df['latitude'], color='k', label='Flight path')

        sc1 = ax1.scatter(
            cocip_df['longitude'],
            cocip_df['latitude'],
            c=cocip_df['rf_lw'],
            cmap='Reds',
            vmin=rf_lw_min,
            vmax=rf_lw_max,
            label='Contrail LW RF (W/m2)'
        )

        ax1.set_xlim(lon_min, lon_max)
        ax1.set_ylim(lat_min, lat_max)

        ax1.legend()
        plt.title(f"{flight_name} - Long Wave Radiative Forcing of Contrail")
        plt.colorbar(sc1, ax=ax1, label='rf_lw')
        plt.savefig(f'LW_RF_{flight_name}.png')
        plt.close()

        ## Short Wave RF Plot
        plt.figure()
        ax2 = plt.axes()

        ax2.plot(fcocip_df['longitude'], fcocip_df['latitude'], color='k', label='Flight path')

        sc2 = ax2.scatter(
            cocip_df['longitude'],
            cocip_df['latitude'],
            c=cocip_df['rf_sw'],
            cmap='Blues_r',
            vmin=rf_sw_min,
            vmax=rf_sw_max,
            label='Contrail SW RF (W/m2)'
        )

        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)

        ax2.legend()
        plt.title(f"{flight_name} - Short Wave Radiative Forcing of Contrail")
        plt.colorbar(sc2, ax=ax2, label='rf_sw')
        plt.savefig(f'SW_RF_{flight_name}.png')
        plt.close()

        ## Energy Forcing Evolution Plot
        plt.figure()
        ax3 = plt.axes()

        ax3.plot(fcocip_df['longitude'], fcocip_df['latitude'], color='k', label='Flight path')

        max_abs = max(abs(ef_min), abs(ef_max))
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

        sc3 = ax3.scatter(
            cocip_df['longitude'],
            cocip_df['latitude'],
            c=cocip_df['ef'],
            cmap='coolwarm',
            norm=norm,
            alpha=0.8,
            label="Contrail EF (J)"
        )

        ax3.set_xlim(lon_min, lon_max)
        ax3.set_ylim(lat_min, lat_max)

        ax3.legend()
        plt.title(f"{flight_name} - Contrail Energy Forcing Evolution")
        cbar = plt.colorbar(sc3, ax=ax3, label='ef')
        cbar.formatter.set_powerlimits((0, 0))
        plt.savefig(f'EF_Evolution_{flight_name}.png')
        plt.close()

    print("Plots saved as LW_RF_<flight_name>.png, SW_RF_<flight_name>.png, and EF_Evolution_<flight_name>.png")


# Specify the directories containing the parquet and CSV files for the two flights you want to compare
prediction = 'mees'
weather_model = 'era5model'
prediction_2 = 'mees'
weather_model_2 = 'era5'
flight1_dir = f"main_results_figures/results/malaga/malaga/climate/{prediction}/{weather_model}"
flight2_dir = f"main_results_figures/results/malaga/malaga/climate/{prediction_2}/{weather_model_2}"

plot_flight_data([flight1_dir, flight2_dir])
