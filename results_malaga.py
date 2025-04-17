import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob


def plot_flight_data(flight_dirs, output_dirs, engine_models):
    # Storage for min/max values across flights
    lat_min, lat_max = float('inf'), float('-inf')
    lon_min, lon_max = float('inf'), float('-inf')
    rf_lw_min, rf_lw_max = float('inf'), float('-inf')
    rf_sw_min, rf_sw_max = float('inf'), float('-inf')
    ef_min, ef_max = float('inf'), float('-inf')

    # Storage for data to plot later
    flight_data = []

    # First pass: Load data and determine global min/max values
    for flight_dir, engine_model in zip(flight_dirs, engine_models):
        parquet_path = glob.glob(os.path.join(flight_dir, f'co_cont_{engine_model}_0_0.parquet'))[0]
        csv_path = glob.glob(os.path.join(flight_dir, f'{engine_model}_SAF_0_A20N_full_WAR_0_climate.csv'))[0]

        cocip_df = pd.read_parquet(parquet_path)
        cocip_df['ef'] = cocip_df['ef']*0.42
        fcocip_df = pd.read_csv(csv_path)

        # Latitude and longitude from both fcocip_df and cocip_df
        lat_min = min(lat_min, cocip_df['latitude'].min(), fcocip_df['latitude'].min())
        lat_max = max(lat_max, cocip_df['latitude'].max(), fcocip_df['latitude'].max())
        lon_min = min(lon_min, cocip_df['longitude'].min(), fcocip_df['longitude'].min())
        lon_max = max(lon_max, cocip_df['longitude'].max(), fcocip_df['longitude'].max())

        rf_lw_min = min(rf_lw_min, cocip_df['rf_lw'].min())
        rf_lw_max = max(rf_lw_max, cocip_df['rf_lw'].max())

        rf_sw_min = min(rf_sw_min, cocip_df['rf_sw'].min())
        rf_sw_max = max(rf_sw_max, cocip_df['rf_sw'].max())

        ef_min = min(ef_min, cocip_df['ef'].min())
        ef_max = max(ef_max, cocip_df['ef'].max())

        flight_data.append((fcocip_df, cocip_df))

    # Add buffer of 0.5 degrees to lat/lon limits
    lat_min -= 0.5
    lat_max += 0.5
    lon_min -= 0.5
    lon_max += 0.5

    print(f"Buffered Latitude range: {lat_min} to {lat_max}")
    print(f"Buffered Longitude range: {lon_min} to {lon_max}")
    print(f"Global RF_LW range: {rf_lw_min} to {rf_lw_max}")
    print(f"Global RF_SW range: {rf_sw_min} to {rf_sw_max}")
    print(f"Global EF range: {ef_min} to {ef_max}")

    # Second pass: Plotting with consistent limits
    for (fcocip_df, cocip_df), output_dir, engine_model in zip(flight_data, output_dirs, engine_models):
        os.makedirs(output_dir, exist_ok=True)

        ## Long Wave RF Plot
        plt.figure()
        ax1 = plt.axes()

        ax1.plot(fcocip_df['longitude'], fcocip_df['latitude'], color='k', label='Flight Path')

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
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.legend()
        plt.title("Long Wave Radiative Forcing of Contrail")
        plt.colorbar(sc1, ax=ax1, label='Long Wave Radiative Forcing (W/m2)')
        plt.savefig(os.path.join(output_dir, f'{engine_model}_SAF_0_cocip_lw_rf.png'))
        plt.close()

        ## Short Wave RF Plot
        plt.figure()
        ax2 = plt.axes()

        ax2.plot(fcocip_df['longitude'], fcocip_df['latitude'], color='k', label='Flight Path')

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
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.legend()
        plt.title("Short Wave Radiative Forcing of Contrail")
        plt.colorbar(sc2, ax=ax2, label='Short Wave Radiative Forcing (W/m2)')
        plt.savefig(os.path.join(output_dir, f'{engine_model}_SAF_0_cocip_sw_rf.png'))
        plt.close()

        ## Energy Forcing Evolution Plot
        plt.figure()
        ax3 = plt.axes()

        ax3.plot(fcocip_df['longitude'], fcocip_df['latitude'], color='k', label='Flight Path')

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
        ax3.set_xlabel("Longitude")
        ax3.set_ylabel("Latitude")
        ax3.legend()
        plt.title("Contrail Energy Forcing Evolution")
        cbar = plt.colorbar(sc3, ax=ax3)
        cbar.set_label('Energy Forcing (J), including efficacy', fontsize=10)
        cbar.formatter.set_powerlimits((0, 0))
        plt.savefig(os.path.join(output_dir, f'{engine_model}_SAF_0_cocip_ef_evolution.png'))
        plt.close()

    print("Plots saved in the corresponding directories.")


# Specify the directories containing the parquet and CSV files for the two flights you want to compare
prediction = 'mees'
weather_model = 'era5model'
engine_model_1 = 'GTF'
prediction_2 = 'pycontrails'
weather_model_2 = 'era5model'
engine_model_2 = 'GTF'

flight1_dir = f"main_results_figures/results/malaga/malaga/climate/{prediction}/{weather_model}"
flight2_dir = f"main_results_figures/results/malaga/malaga/climate/{prediction_2}/{weather_model_2}"

output1_dir = f"main_results_figures/figures/malaga/malaga/climate/{prediction}/{weather_model}/cocip/{engine_model_1}/pycontrails"
output2_dir = f"main_results_figures/figures/malaga/malaga/climate/{prediction_2}/{weather_model_2}/cocip/{engine_model_2}/pycontrails"

plot_flight_data([flight1_dir, flight2_dir], [output1_dir, output2_dir], [engine_model_1, engine_model_2])

csv_path_1 = glob.glob(os.path.join(flight1_dir, f'{engine_model_1}_SAF_0_A20N_full_WAR_0_climate.csv'))[0]
df_1 = pd.read_csv(csv_path_1)

csv_path_2 = glob.glob(os.path.join(flight2_dir, f'{engine_model_2}_SAF_0_A20N_full_WAR_0_climate.csv'))[0]
df_2 = pd.read_csv(csv_path_2)

# Compute total sums for each variable in both datasets
total_fuel_flow_1 = df_1['fuel_flow'].sum() # Per engine
total_fuel_flow_gsp_2 = df_2['fuel_flow'].sum()

total_ei_nox_1 = df_1['ei_nox'].sum()
total_ei_nox_gsp_2 = df_2['ei_nox'].sum()

total_ei_nvpm_1 = df_1['nvpm_ei_n'].sum()
total_ei_nvpm_gsp_2 = df_2['nvpm_ei_n'].sum()

total_nox_1 = (df_1['ei_nox'] * df_1['fuel_flow'] ).sum()
total_nox_gsp_2 = (df_2['ei_nox'] * df_2['fuel_flow']).sum()

total_nvpm_1 = (df_1['nvpm_ei_n'] * (df_1['fuel_flow'] )).sum()
total_nvpm_gsp_2 = (df_2['nvpm_ei_n'] * df_2['fuel_flow']).sum()

# Compute percentage differences
percentage_differences = {
    'Fuel Flow': ((total_fuel_flow_gsp_2 - total_fuel_flow_1) / total_fuel_flow_1) * 100,
    'EI NOx': ((total_ei_nox_gsp_2 - total_ei_nox_1) / total_ei_nox_1) * 100,
    'NOx': ((total_nox_gsp_2 - total_nox_1) / total_nox_1) * 100,
    'EI nvPM': ((total_ei_nvpm_gsp_2 - total_ei_nvpm_1) / total_ei_nvpm_1) * 100,
    'nvPM': ((total_nvpm_gsp_2 - total_nvpm_1) / total_nvpm_1) * 100
}

# Create a bar plot
plt.figure(figsize=(8, 6))
x_emissions = np.arange(len(percentage_differences))
plt.bar(x_emissions, list(percentage_differences.values()), width=0.5)
plt.xticks(x_emissions, list(percentage_differences.keys()))
plt.xlabel("Metric")
plt.ylabel("Percentage Difference (%)")
plt.title("Emissions - Relative Difference Compared To Baseline")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f'results_report/climate_sensitivity_chapter/{prediction}_{prediction_2}_{weather_model}_{weather_model_2}_emissions.png', format='png')



dt = 60

total_co2_impact_1 = (df_1['fuel_flow']*dt*df_1['accf_sac_aCCF_CO2']).sum()
total_co2_impact_gsp_2 = (df_2['fuel_flow']*dt*df_2['accf_sac_aCCF_CO2']).sum()
print(total_co2_impact_1)
print(total_co2_impact_gsp_2)
total_nox_impact_1 = (df_1['fuel_flow']*dt*(df_1['accf_sac_aCCF_O3']+df_1['accf_sac_aCCF_CH4']*1.29)*df_1['ei_nox']).sum()
total_nox_impact_gsp_2 = (df_2['fuel_flow']*dt*(df_2['accf_sac_aCCF_O3']+df_2['accf_sac_aCCF_CH4']*1.29)*df_2['ei_nox']).sum()
print(total_nox_impact_1)
print(total_nox_impact_gsp_2)
total_cocip_atr20_impact_1 = np.absolute(df_1['cocip_atr20'].sum()*0.42)
total_cocip_atr20_impact_gsp_2 = np.absolute(df_2['cocip_atr20'].sum()*0.42)
# total_cocip_atr20_impact_1 = df_1['cocip_atr20'].sum()*0.42
# total_cocip_atr20_impact_gsp_2 = df_2['cocip_atr20'].sum()*0.42
print(total_cocip_atr20_impact_1)
print(total_cocip_atr20_impact_gsp_2)
# total_non_co2_impact_1 = df_1['accf_sac_nox_impact'].sum()+df_1['cocip_atr20'].sum()
# total_non_co2_impact_gsp_2 = df_2['accf_sac_nox_impact'].sum()+df_2['cocip_atr20'].sum()
#
# total_impact_1 =df_1['accf_sac_nox_impact'].sum()+df_1['cocip_atr20'].sum()+df_1['accf_sac_co2_impact'].sum()
# total_impact_gsp_2 =df_2['accf_sac_nox_impact'].sum()+df_2['cocip_atr20'].sum()+df_2['accf_sac_co2_impact'].sum()

impact_labels = ['CO2', 'NOx', 'Contrails']
percentage_climate_differences = [
    ((total_co2_impact_gsp_2 - total_co2_impact_1) / total_co2_impact_1) * 100,
    ((total_nox_impact_gsp_2 - total_nox_impact_1) / total_nox_impact_1) * 100,
    ((total_cocip_atr20_impact_gsp_2 - total_cocip_atr20_impact_1) / total_cocip_atr20_impact_1) * 100
    # ((total_non_co2_impact_gsp_2 - total_non_co2_impact_1) / total_non_co2_impact_1) * 100,
    # ((total_impact_gsp_2 - total_impact_1) / total_impact_1) * 100
]

plt.figure(figsize=(8, 6))
x_climate = np.arange(len(impact_labels))
plt.bar(x_climate, percentage_climate_differences, width=0.25)
plt.xticks(x_climate, impact_labels)
plt.xlabel("Metric")
plt.ylabel("Percentage Difference (%)")
plt.title("Climate Impact (P-ATR20) - Relative Difference Compared To Baseline")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f'results_report/climate_sensitivity_chapter/{prediction}_{prediction_2}_{weather_model}_{weather_model_2}_climate.png', format='png')
plt.show()

# plt.figure(figsize=(10, 6))
# # plt.plot(df_gsp.index, df_gsp['fuel_flow_per_engine'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
# plt.plot(df_1.index, df_1['fuel_flow_gsp'], label='GSP 60s Interval', linestyle='-', marker='o', markersize=2.5)
# plt.plot(df_2.index, df_2['fuel_flow_gsp'], label='GSP Simplified Cruise', linestyle='-', marker='o', markersize=2.5)
# # plt.plot(df_piano.index, df_piano['fuel_flow_piano'], label='PianoX', linestyle='-', marker='o', markersize=2.5)
# plt.title('Fuel Flow')
# plt.xlabel('Time in minutes')
# plt.ylabel('Fuel Flow (kg/s)')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'results_report/climate_sensitivity_chapter/fuel_flow_cr_appr.png', format='png')
# # plt.close()
# plt.show()