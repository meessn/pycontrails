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
from pycontrails.models.cocip.output_formats import flight_waypoint_summary_statistics, contrail_flight_summary_statistics
from pycontrails.physics.thermo import rh
from pycontrails.core.met_var import RelativeHumidity
from pycontrails.core.cache import DiskCacheStore
from pathlib import Path
import os
import pickle

save_path_contrail = f'results/malaga/climate/mees/era5model/cocip_contrail.parquet'

contrail = pd.read_parquet(save_path_contrail)
plt.figure()
ax1 = plt.axes()



# Plot contrail LW RF
contrail.plot.scatter(
    "longitude",
    "latitude",
    c="rf_lw",
    cmap="Reds",
    ax=ax1,
    label="Contrail LW RF",
)

ax1.legend()




# Create a new figure for the second plot
plt.figure()
ax2 = plt.axes()



# Plot contrail SW RF
contrail.plot.scatter(
    "longitude",
    "latitude",
    c="rf_sw",
    cmap="Blues_r",
    ax=ax2,
    label="Contrail SW RF",
)

ax2.legend()



plt.figure()
ax3 = plt.axes()




# Get absolute max value for symmetric colormap
ef_min = contrail["ef"].min()
ef_max = contrail["ef"].max()
max_abs = max(abs(ef_min), abs(ef_max))  # Ensure symmetry

# Normalize colormap around 0, using symmetric min/max values
norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

# Use Matplotlib's scatter instead of Pandas
sc = ax3.scatter(
    contrail["longitude"],
    contrail["latitude"],
    c=contrail["ef"],
    cmap="coolwarm",
    norm=norm,  # Symmetric colormap
    alpha=0.7,  # Make points slightly transparent for better visibility
    label="Contrail EF",
)

# Add colorbar
cbar = plt.colorbar(sc, ax=ax3, label="Energy Forcing (EF)")
cbar.formatter.set_powerlimits((0, 0))  # Ensure scientific notation format

ax3.legend()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Contrail Energy Forcing Evolution")

plt.show()