# import pandas as pd
#
# # Load the CSV file (update 'your_file.csv' with the actual filename)
# prediction = "mees"
# file_path = f"main_results_figures/results/malaga/malaga/climate/{prediction}/era5model/GTF_SAF_0_A20N_full_WAR_0_cli_cont.csv"
# df = pd.read_csv(file_path)
#
# # Ensure the file contains data
# if df.empty:
#     print("The CSV file is empty.")
# else:
#     # Extract column names and values from the first row
#     for column in df.columns:
#         value = df[column].iloc[0]  # Get the first value for each column
#         print(f"{column}: {value}")

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.colors as mcolors

# matplotlib.use('Agg')  # Prevents GUI windows
import copy
from pycontrails.core.met import MetDataset, MetVariable, MetDataArray
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5, ERA5ModelLevel
from pycontrails.datalib.ecmwf.variables import PotentialVorticity
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import ExponentialBoostHumidityScaling
from pycontrails.models.issr import ISSR
from pycontrails.physics import units
from pycontrails.models.accf import ACCF
from pycontrails.datalib import ecmwf
from pycontrails.core.fuel import JetA, SAF20, SAF100
from pycontrails.models.cocip.output_formats import flight_waypoint_summary_statistics, \
    contrail_flight_summary_statistics
from pycontrails.physics.thermo import rh
from pycontrails.core.met_var import RelativeHumidity
from pycontrails.core.cache import DiskCacheStore
from pathlib import Path
import os
import pickle

# # Path to the Parquet file
# parquet_path = 'main_results_figures/results/malaga/malaga/climate/mees/era5model/cocip_contrail.parquet'
#
# # Read the Parquet file into a DataFrame
# cocip = pd.read_parquet(parquet_path)
#
# # Display the DataFrame (or you can work with it as needed)
# print(cocip.head())
# plt.figure()
# ax3 = plt.axes()
# # Extract exact colors from the 'coolwarm' colormap
# coolwarm = plt.get_cmap("coolwarm")
#
# blue_rgb = coolwarm(0.0)  # Exact blue from coolwarm
# gray_rgb = coolwarm(0.5)  # Exact gray from coolwarm (center)
# red_rgb = coolwarm(1.0)  # Exact red from coolwarm
#
# def create_blue_gray_colormap():
#     """ Custom colormap from exact coolwarm blue to coolwarm gray (for negative values). """
#     colors = [blue_rgb, gray_rgb]  # Coolwarm blue → Coolwarm gray
#     return mcolors.LinearSegmentedColormap.from_list("BlueGray", colors)
#
# def create_gray_red_colormap():
#     """ Custom colormap from exact coolwarm gray to coolwarm red (for positive values). """
#     colors = [gray_rgb, red_rgb]  # Coolwarm gray → Coolwarm red
#     return mcolors.LinearSegmentedColormap.from_list("GrayRed", colors)
#
# rf_lw_min = cocip["rf_lw"].min()
# rf_lw_max = cocip["rf_lw"].max()
#
# if rf_lw_min == rf_lw_max:  # All values are identical
#     if rf_lw_min == 0:  # Special case: all zero
#         vmin, vmax = -1, 1  # Avoid collapsed colormap
#     else:
#         vmin, vmax = rf_lw_min - 0.1 * abs(rf_lw_min), rf_lw_max + 0.1 * abs(rf_lw_max)  # Small buffer
#     norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#     cmap = "coolwarm"
#
# elif rf_lw_max <= 0:  # All values are negative → Gray at the top (vmax=0)
#     vcenter = (rf_lw_min + 0) / 2  # Middle value, prevents vcenter == vmax error
#     norm = mcolors.TwoSlopeNorm(vmin=rf_lw_min, vcenter=vcenter, vmax=0)
#     cmap = create_blue_gray_colormap()  # Uses exact coolwarm colors
#
# elif rf_lw_min >= 0:  # All values are positive → Gray at the bottom (vmin=0)
#     vcenter = (0 + rf_lw_max) / 2  # Middle value, prevents vcenter == vmin error
#     norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=vcenter, vmax=rf_lw_max)
#     cmap = create_gray_red_colormap()  # Uses exact coolwarm colors
#
# else:  # Mixed positive and negative values → Use standard coolwarm
#     max_abs = max(abs(rf_lw_min), abs(rf_lw_max))
#     norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
#     cmap = "coolwarm"  # No modification needed
#
# # Scatter plot with the selected colormap
# sc = ax3.scatter(
#     cocip["longitude"],
#     cocip["latitude"],
#     c=cocip["rf_lw"],
#     cmap=cmap,
#     norm=norm,
#     alpha=0.8,
#     label="Contrail EF (J)",
# )
# plt.show()
#
# # Add colorbar and format it
# cbar = plt.colorbar(sc, ax=ax3, label="rf_lw")
# cbar.formatter.set_powerlimits((0, 0))
# sc.set_clim(norm.vmin, norm.vmax)
#
# ax3.legend()
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Contrail Energy Forcing Evolution")

climate_csv = 'main_results_figures/results/malaga/malaga/climate/mees/era5model/GTF_SAF_0_A20N_full_WAR_0_climate.csv'
df = pd.read_csv(climate_csv)
df['cocip_atr20'] = df['cocip_atr20'].fillna(0)
plt.figure(figsize=(10, 6))
plt.plot(df['index'], df['accf_sac_contrails_atr20'], label='aCCF')
plt.plot(df['index'], df['cocip_atr20'], label='CoCiP')
plt.title('Contrail warming impact (P-ATR20)')
plt.xlabel('Time in minutes')
plt.ylabel('Degrees K ')
plt.legend()
plt.grid(True)
plt.savefig('main_results_figures/figures/malaga/malaga/climate/mees/era5model/accf_vs_cocip.png', format='png')