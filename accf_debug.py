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

# save_path_contrail = f'results/malaga/climate/mees/era5model/cocip_contrail.parquet'
#
# contrail = pd.read_parquet(save_path_contrail)
# plt.figure()
# ax1 = plt.axes()
#
#
#
# # Plot contrail LW RF
# contrail.plot.scatter(
#     "longitude",
#     "latitude",
#     c="rf_lw",
#     cmap="Reds",
#     ax=ax1,
#     label="Contrail LW RF",
# )
#
# ax1.legend()
#
#
#
#
# # Create a new figure for the second plot
# plt.figure()
# ax2 = plt.axes()
#
#
#
# # Plot contrail SW RF
# contrail.plot.scatter(
#     "longitude",
#     "latitude",
#     c="rf_sw",
#     cmap="Blues_r",
#     ax=ax2,
#     label="Contrail SW RF",
# )
#
# ax2.legend()
#
#
#
# plt.figure()
# ax3 = plt.axes()
#
#
#
#
# # Get absolute max value for symmetric colormap
# ef_min = contrail["ef"].min()
# ef_max = contrail["ef"].max()
# max_abs = max(abs(ef_min), abs(ef_max))  # Ensure symmetry
#
# # Normalize colormap around 0, using symmetric min/max values
# norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
#
# # Use Matplotlib's scatter instead of Pandas
# sc = ax3.scatter(
#     contrail["longitude"],
#     contrail["latitude"],
#     c=contrail["ef"],
#     cmap="coolwarm",
#     norm=norm,  # Symmetric colormap
#     alpha=0.7,  # Make points slightly transparent for better visibility
#     label="Contrail EF",
# )
#
# # Add colorbar
# cbar = plt.colorbar(sc, ax=ax3, label="Energy Forcing (EF)")
# cbar.formatter.set_powerlimits((0, 0))  # Ensure scientific notation format
#
# ax3.legend()
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Contrail Energy Forcing Evolution")
#
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.accf import ACCF
from pycontrails.physics import units
trajectory = 'malaga'
flight = 'malaga'
engine_model = 'GTF'
SAF = 0
formatted_values = [0,0,0]
aircraft = 'A20N_full'
file_path = f'main_results_figures/results/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv'
df = pd.read_csv(file_path)

pressure_levels = (1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175)
if flight == 'malaga':
    time_bounds = ("2024-06-07 09:00", "2024-06-08 02:00")
    local_cache_dir_era5m = Path("F:/era5model/malaga")
    variables_model = ("t", "q", "u", "v", "w", "ciwc", "vo", "clwc")

local_cache_dir_era5p = Path("F:/era5pressure/Cache")
local_cachestore_era5p = DiskCacheStore(cache_dir=local_cache_dir_era5p)

era5pl = ERA5(
            time=time_bounds,
            variables=Cocip.met_variables + Cocip.optional_met_variables + (ecmwf.PotentialVorticity,) + (ecmwf.RelativeHumidity,),
            pressure_levels=pressure_levels,
            cachestore=local_cachestore_era5p
        )
era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,), cachestore=local_cachestore_era5p)

met = era5pl.open_metdataset()
rad = era5sl.open_metdataset()

accf_issr = ACCF(
    met=met,
    surface=rad,
    params={
        "emission_scenario": "pulse",
        "accf_v": "V1.0",  "issr_rhi_threshold": 0.9, "efficacy": True, "PMO": False,
        "horizontal_resolution": 0.25,
        "forecast_step": None,
        "pfca": "PCFA-ISSR",
        "unit_K_per_kg_fuel": True
        # "sac_eta": fl.dataframe['engine_efficiency']
        # "pfca": "PCFA-SAC"
    },
    verify_met=False
    )

df_climate = pd.read_csv('main_results_figures/results/malaga/malaga/climate/mees/era5model/GTF_SAF_0_A20N_full_WAR_0_climate.csv')

fl = Flight(data=df)
fa_issr = accf_issr.eval(fl)
df_accf_issr = fa_issr.dataframe.copy()
plt.figure(figsize=(10, 6))
# plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_CH4'], label="aCCF CH4")
# plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_O3'], label="aCCF O3")
plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_NOx'], label="aCCF NOx climaccf")
plt.plot(df_climate['index'], df_climate['ei_nox']*df_climate['accf_issr_aCCF_NOx'], label='aCCF NOx EI This Work')
plt.plot(df_climate['index'], df_climate['accf_issr_aCCF_CO2'], label='aCCF CO2')
plt.title(f'NOx - aCCF along {flight} Flight')
plt.xlabel('Time in minutes')
plt.ylabel('Degrees K / kg fuel')
plt.legend()
plt.grid(True)
# plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/climate/{prediction}/{weather_model}/accf_issr/{engine_model}_SAF_{SAF}_nox_accf.png', format='png')
# plt.close()
plt.figure(figsize=(10, 6))
# plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_CH4'], label="aCCF CH4")
# plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_O3'], label="aCCF O3")
plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_NOx']*df_accf_issr['fuel_flow_gsp']*2, label="aCCF NOx climaccf")
plt.plot(df_climate['index'], df_climate['ei_nox']*df_climate['accf_issr_aCCF_NOx']*df_climate['fuel_flow'], label='aCCF NOx EI This Work')
plt.plot(df_climate['index'], df_climate['accf_issr_aCCF_CO2']*df_climate['fuel_flow'], label='aCCF CO2')
plt.title(f'NOx - aCCF along {flight} Flight')
plt.xlabel('Time in minutes')
plt.ylabel('Degrees K')
plt.legend()
plt.grid(True)
plt.show()