import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

flight_trajectories_to_simulate = [
    "bos_fll", "cts_tpe", "dus_tos", "gru_lim", "hel_kef", "lhr_ist", "sfo_dfw", "sin_maa"
]

base_results_dir = 'main_results_figures/results'
base_figures_dir = 'main_results_figures/figures'

# User-defined variables
flight_date = '2023-02-06'
time_of_day = 'daytime'

ef_min_global = float('inf')
ef_max_global = float('-inf')
dataframes = []

# First pass to get global min and max for color scale
for flight in flight_trajectories_to_simulate:
    parquet_path = os.path.join(base_results_dir, flight, f"{flight}_{flight_date}_{time_of_day}/climate/mees/era5model/co_cont_GTF_0_0.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        df['flight'] = flight
        dataframes.append(df)
        ef_min_global = min(ef_min_global, df['ef'].min())
        ef_max_global = max(ef_max_global, df['ef'].max())

# Combine data
combined_df = pd.concat(dataframes, ignore_index=True)

# Plot combined map
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')

# Plot flight paths
for flight in flight_trajectories_to_simulate:
    flight_path_csv = os.path.join(base_results_dir, flight, f"{flight}_{flight_date}_{time_of_day}/climate/mees/era5model/GTF_SAF_0_A20N_full_WAR_0_climate.csv")
    if os.path.exists(flight_path_csv):
        flight_path_df = pd.read_csv(flight_path_csv)
        ax.plot(
            flight_path_df['longitude'],
            flight_path_df['latitude'],
            color='k', linewidth=0.8, transform=ccrs.PlateCarree(), label=flight
        )

# Scatter plot with global colormap
max_abs_ef = max(abs(ef_min_global), abs(ef_max_global))
norm = mcolors.TwoSlopeNorm(vmin=-max_abs_ef, vcenter=0, vmax=max_abs_ef)
sc = ax.scatter(
    combined_df['longitude'],
    combined_df['latitude'],
    c=combined_df['ef'],
    cmap='coolwarm',
    norm=norm,
    alpha=0.8,
    transform=ccrs.PlateCarree()
)

plt.colorbar(sc, ax=ax, label="Contrail EF (J)")
ax.set_title(f"Contrail EF Evolution - {flight_date}_{time_of_day} GTF")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Save plot
save_path = os.path.join(base_figures_dir, f"combined_{flight_date}_{time_of_day}_climate_mees_era5model_cocip_GTF_ef_evolution.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
# plt.savefig(save_path, dpi=300)
# plt.close()
plt.show()
print(f"Combined plot generated successfully for {flight_date} {time_of_day}.")
