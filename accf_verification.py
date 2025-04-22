import pandas as pd
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.colors as mcolors
# matplotlib.use('Agg')  # Prevents GUI windows
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
from pycontrails.physics.units import m_to_pl, pl_to_m
from pycontrails.models.cocip.output_formats import flight_waypoint_summary_statistics, contrail_flight_summary_statistics
from pycontrails.physics.thermo import rh
from pycontrails.core.met_var import RelativeHumidity
from pycontrails.core.cache import DiskCacheStore
from pathlib import Path
import os
import pickle


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.accf import ACCF
from pycontrails.physics import units

time_bounds = ("2015-12-18 00:00", "2015-12-18 01:00")
local_cache_dir_era5p = Path("F:/era5pressure/accf_verification")
local_cachestore_era5p = DiskCacheStore(cache_dir=local_cache_dir_era5p)
pressure_levels = [250]
era5pl = ERA5(
            time=time_bounds,
            variables=Cocip.met_variables + Cocip.optional_met_variables + (ecmwf.PotentialVorticity,) + (ecmwf.RelativeHumidity,),
            pressure_levels=pressure_levels,
            cachestore=local_cachestore_era5p
        )
era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,), cachestore=local_cachestore_era5p)

# Download data from ERA5 (or open from cache)
met = era5pl.open_metdataset()  # meteorology
rad = era5sl.open_metdataset()

ac = ACCF(
        met=met,
        surface=rad,
        params={
            "emission_scenario": "pulse",
            "accf_v": "V1.0",  "issr_rhi_threshold": 0.9, "efficacy": False, "PMO": True,
            "horizontal_resolution": 0.25,
            "forecast_step": 0,
            "pfca": "PCFA-ISSR",
            "unit_K_per_kg_fuel": False
            # "sac_eta": fl.dataframe['engine_efficiency']
            # "pfca": "PCFA-SAC"
        },
        verify_met=False
    )
ds = ac.eval()

# Custom colormap: white -> yellow -> orange -> red
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
value_points = [5e-13, 1.1e-12, 1.55e-12]  # white, yellow, orange, red
color_points = ["white", "orange", "red"]

# Normalize the values to 0–1 range for colormap creation
normed = [(v - min(value_points)) / (max(value_points) - min(value_points)) for v in value_points]
custom_cmap = LinearSegmentedColormap.from_list("custom", list(zip(normed, color_points)))

lon_vals = ds["longitude"].data  # or .values
lat_vals = ds["latitude"].data

lon_mask = (lon_vals >= -20) & (lon_vals <= 40)
lat_mask = (lat_vals >= 30) & (lat_vals <= 70)

lon_subset = ds["longitude"].data[lon_mask]
lat_subset = ds["latitude"].data[lat_mask]
da = ds["aCCF_O3"].data  # This gives you the underlying xarray DataArray
print(da.dims)
level_val = da.coords["level"].values[0]
time_val = da.coords["time"].values[0]

print(f"Level index 0 corresponds to level = {level_val}")
print(f"Time index 0 corresponds to time = {time_val}")
# Select a slice at level=0, time=0
data_slice = da.isel(level=0, time=0)
# Get index positions
lon_idx = np.where(lon_mask)[0]
lat_idx = np.where(lat_mask)[0]

# Subset by index
data_subset = data_slice.isel(
    longitude=lon_idx,
    latitude=lat_idx
)

data_subset = data_subset.T
# Plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as ticker

# Make sure plot_data is lat x lon
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.15)

# Main plot
ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

# Plotting
p = ax.pcolor(
    lon_subset,
    lat_subset,
    data_subset,
    cmap=custom_cmap,
    vmin=5e-13,
    vmax=1.55e-12,
    transform=ccrs.PlateCarree()
)

# Add map features
ax.coastlines(resolution='50m', linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none')

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Extract geopotential field (same level and time)
geo = ds["geopotential"].data.isel(level=0, time=0)  # or "z" or "phi" depending on your dataset

# Subset using same index masks as before
geo_subset = geo.isel(longitude=lon_idx, latitude=lat_idx)

# Transpose if needed
geo_plot = geo_subset.T
print(np.nanmin(geo_plot), np.nanmax(geo_plot))
# Contour levels
contour_levels = np.arange(94500, 106500, 1500)

# Draw contour lines
contours = ax.contour(
    lon_subset,
    lat_subset,
    geo_plot,
    levels=[100500],
    colors='black',
    linewidths=1,
    transform=ccrs.PlateCarree()
)

# Label all contours with exponential notation
ax.clabel(
    contours,
    fmt={100500: r"$1.005 \times 10^5$"},     # Labels like 1e+05, 9e+04, etc.
    inline=True,
    fontsize=10
)

# Labels and title
ax.set_title("O3 aCCF (K/kg(NO2)) at 00:00 UTC 250 hPa", fontsize=12)

# Colorbar
# Define colorbar tick range from 5e-13 to 1.5e-12 in steps of 0.5e-13
ticks = np.arange(5e-13, 1.45e-12, 0.5e-13)

# Set only a few labels (e.g., every 2nd or 3rd tick)
label_ticks = ticks[::3]  # show every 2nd tick label

cax = fig.add_subplot(gs[1])
cbar = plt.colorbar(
    p,
    cax=cax,
    orientation='horizontal',
    ticks=ticks,     # your existing ticks
    extend='both'
)

# Now override tick labels: only show selected, others blank
cbar.ax.set_xticklabels([f"{t:.2e}" if t in label_ticks else "" for t in ticks])

# Set extent to match the domain: 20W to 40E, 30N to 70N
ax.set_extent([-20, 40, 30, 70], crs=ccrs.PlateCarree())
plt.savefig('results_report/accf_verification/ozone_climaccf.png', dpi=300, bbox_inches='tight')
# plt.tight_layout()


# Get the underlying xarray DataArray
ch4_da = ds["aCCF_CH4"].data
nox_da = ds["aCCF_NOx"].data
h2o_da = ds["aCCF_H2O"].data

# Select level 0 and time 0
ch4_slice = ch4_da.isel(level=0, time=0)
nox_slice = nox_da.isel(level=0, time=0)
h2o_slice = h2o_da.isel(level=0, time=0)
# Use same longitude/latitude index masks
ch4_plot_data = ch4_slice.isel(
    longitude=lon_idx,
    latitude=lat_idx
).T  # transpose to (lat, lon)

nox_plot_data = nox_slice.isel(
    longitude=lon_idx,
    latitude=lat_idx
).T  # transpose to (lat, lon)

h2o_plot_data = h2o_slice.isel(
    longitude=lon_idx,
    latitude=lat_idx
).T  # transpose to (lat, lon)

# Custom blue-to-white colormap
blue_shades = LinearSegmentedColormap.from_list(
    "blue_custom",
    ["#081d58", "#2171b5", "#6baed6", "#bdd7e7", "#f7fbff"]
)
# Tick settings
vmin = -5.2e-13
vmax = -4.3e-13
step = 0.3e-14  # step = 0.03e-13
ticks_ch4 = np.arange(vmin, vmax+ 1e-15, step)
print(ticks_ch4)
label_ticks_ch4 = ticks_ch4[::5]
print(label_ticks_ch4)
# Create the plot
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.15)

# Main plot
ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

# Plot the data
p = ax.pcolor(
    lon_subset,
    lat_subset,
    ch4_plot_data,
    cmap=blue_shades,
    vmin=vmin,
    vmax=vmax,
    transform=ccrs.PlateCarree()
)

# Map features
ax.coastlines(resolution='50m', linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none')

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Set extent
ax.set_extent([-20, 40, 30, 70], crs=ccrs.PlateCarree())

# Title
ax.set_title("CH4 aCCF + PMO aCCF (K/kg(NOx)) at 00:00 UTC 250 hPa", fontsize=12)

cax = fig.add_subplot(gs[1])
cbar = plt.colorbar(
    p,
    cax=cax,
    orientation='horizontal',
    ticks=ticks_ch4,     # your existing ticks
    extend='both'
)

# Set sparse tick labels
cbar.ax.set_xticklabels([f"{t:.2e}" if t in label_ticks_ch4 else "" for t in ticks_ch4])
cbar.set_label("K/kg(NOx)")
plt.savefig('results_report/accf_verification/methane_climaccf.png', dpi=300, bbox_inches='tight')


value_points = [2e-13, 6e-13, 1e-12]  # white, yellow, orange, red
color_points = ["white", "orange", "red"]

# Normalize the values to 0–1 range for colormap creation
normed = [(v - min(value_points)) / (max(value_points) - min(value_points)) for v in value_points]
custom_cmap = LinearSegmentedColormap.from_list("custom", list(zip(normed, color_points)))
# Tick settings
vmin = 2e-13
vmax = 1e-12
step = 0.2e-13  # tick spacing
ticks_nox = np.arange(vmin, vmax + 1e-15, step)
label_ticks_nox = ticks_nox[::5]  # label every 5th tick

# Create the plot
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.15)

# Main plot
ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

# Plot the data
p = ax.pcolor(
    lon_subset,
    lat_subset,
    nox_plot_data,
    cmap=custom_cmap,
    vmin=vmin,
    vmax=vmax,
    transform=ccrs.PlateCarree()
)

# Map features
ax.coastlines(resolution='50m', linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none')

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Set extent
ax.set_extent([-20, 40, 30, 70], crs=ccrs.PlateCarree())

# Title
ax.set_title("NOx aCCF (K/kg(NOx)) at 00:00 UTC 250 hPa", fontsize=12)

cax = fig.add_subplot(gs[1])
cbar = plt.colorbar(
    p,
    cax=cax,
    orientation='horizontal',
    ticks=ticks_nox,     # your existing ticks
    extend='both'
)

# Set sparse tick labels
cbar.ax.set_xticklabels([f"{t:.2e}" if i % 5 == 0 else "" for i, t in enumerate(ticks_nox)])
cbar.set_label("K/kg(NOx)")
plt.savefig('results_report/accf_verification/nox_climaccf.png', dpi=300, bbox_inches='tight')



value_points = [2e-16, 6e-16, 1e-15]  # white, yellow, orange, red
color_points = ["#FFF5B1", "orange", "red"]

# Normalize the values to 0–1 range for colormap creation
normed = [(v - min(value_points)) / (max(value_points) - min(value_points)) for v in value_points]
custom_cmap = LinearSegmentedColormap.from_list("custom", list(zip(normed, color_points)))
# Tick settings
vmin = 2e-16
vmax = 1e-15
step = 0.5e-16  # tick spacing
ticks_h2o = np.arange(vmin, vmax + 1e-16, step)
label_ticks_h2o = ticks_h2o[::2]  # label every 2nd tick
print(ticks_h2o)
print(label_ticks_h2o)
# Create the plot
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1], hspace=0.15)

# Main plot
ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

# Plot the data
p = ax.pcolor(
    lon_subset,
    lat_subset,
    h2o_plot_data,
    cmap=custom_cmap,
    vmin=vmin,
    vmax=vmax,
    transform=ccrs.PlateCarree()
)

# Map features
ax.coastlines(resolution='50m', linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none')

# Gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Set extent
ax.set_extent([-20, 40, 30, 70], crs=ccrs.PlateCarree())

# Title
ax.set_title("H2O aCCF (K/kg(h2o)) at 00:00 UTC 250 hPa", fontsize=12)

cax = fig.add_subplot(gs[1])
cbar = plt.colorbar(
    p,
    cax=cax,
    orientation='horizontal',
    ticks=ticks_h2o,     # your existing ticks
    extend='both'
)

# Set sparse tick labels
cbar.ax.set_xticklabels([f"{t:.2e}" if i % 5 == 0 else "" for i, t in enumerate(ticks_h2o)])
cbar.set_label("K/kg(fuel)")
plt.savefig('results_report/accf_verification/water_vapour_climaccf.png', dpi=300, bbox_inches='tight')

plt.show()
