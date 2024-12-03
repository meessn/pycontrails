import numpy as np
import os
import xarray as xr
import pandas as pd
import subprocess
import constants
import json
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors

import sys
import pickle
from pycontrails import Flight, MetDataset
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import HistogramMatching
from pycontrails.models.ps_model import PSFlight

# from ps_model.ps_model import PSFlight
# import ps_model.ps_grid
from pycontrails.models.issr import ISSR
from pycontrails.physics import units
from pycontrails.models.emissions import Emissions
from pycontrails.models.accf import ACCF
from pycontrails.datalib import ecmwf

"""FLIGHT PARAMETERS"""
engine_model = 'GTF'        # GTF , GTF2035
water_injection = [0, 0, 0]     # WAR climb cruise approach/descent
SAF = 0                         # 0, 20, 100 unit = %
flight = 'malaga'
aircraft = 'A20N_full'        # A20N ps model, A20N_wf is change in Thrust and t/o and idle fuel flows
                            # A20N_wf_opr is with changed nominal opr and bpr
                            # A20N_full has also the eta 1 and 2 and psi_0

# Convert the water_injection values to strings, replacing '.' with '_'
formatted_values = [str(value).replace('.', '_') for value in water_injection]
file_path = f'results/{flight}/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv'

df_read = pd.read_csv(file_path)

columns_required = ['index', 'longitude', 'latitude', 'altitude', 'groundspeed', 'time', 'flight_id', 'air_temperature'
                            , 'eastward_wind', 'northward_wind', 'true_airspeed', 'aircraft_mass', 'specific_humidity',
                    'air_pressure', 'rhi', 'flight_phase', 'WAR', 'engine_model', 'SAF', 'mach', 'PT3', 'TT3', 'TT4',
                    'FAR', 'fuel_flow_gsp', 'thrust_gsp', 'EI_nvpm_number_p3t3_meem']

df = df_read[columns_required]

df = df.rename(columns={
    'EI_nvpm_number_p3t3_meem': 'nvpm_ei_n'
})

df['fuel_flow'] = 2*df['fuel_flow_gsp']
df['thrust'] = 2*df['thrust_gsp']
df['air_pressure'] = df['air_pressure']*10**5
q_fuel = 43.13e6
df['engine_efficiency'] = (df['thrust_gsp']*1000*df['true_airspeed']) / (df['fuel_flow_gsp']*q_fuel)
df['wingspan'] = 35.8

# df = df[df['altitude'] > 9500]

fl = Flight(data=df)
"""------RETRIEVE METEOROLOGIC DATA----------------------------------------------"""

time_bounds = ("2024-06-07 9:00", "2024-06-08 02:00")
pressure_levels = (1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175) #hpa
pressure_levels = (500, 450, 400, 350, 300, 250, 225, 200, 175) #hpa

era5pl = ERA5(
    time=time_bounds,
    variables=Cocip.met_variables + Cocip.optional_met_variables + (ecmwf.PotentialVorticity,) + (ecmwf.RelativeHumidity,),
    pressure_levels=pressure_levels,
)
era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,))

# download data from ERA5 (or open from cache)
met = era5pl.open_metdataset() # meteorology
rad = era5sl.open_metdataset() # radiation


"""ISSRs"""
issr = ISSR(met=met, humidity_scaling=HistogramMatching())
source = MetDataset(
    xr.Dataset(
        coords={
            "time": ["2024-06-07T10:30"],
            "longitude": np.arange(-4, 3, 0.25),
            "latitude": np.arange(35, 55, 0.25),
            "level": units.ft_to_pl(np.arange(27000, 40000, 1000)),
        }
    )
)
da = issr.eval(source).data["issr"]

# # Use altitude_ft as the vertical coordinate for plotting
# da["altitude_m"] = units.pl_to_m(da["level"]).round().astype(int)
# da = da.swap_dims(level="altitude_m")
# da = da.sel(altitude_m=da["altitude_m"].values[::-1])
# da = da.squeeze()
#
# da.plot(x="longitude", y="latitude", col="altitude_m", col_wrap=4, add_colorbar=True);
# # plt.savefig('figures/issr_regions.png')
# plt.savefig(f'figures/{flight}/climate/issr_regions.png', format='png')
"""-------------------------------------------------------------------------------------"""
"""---------------------CoCiP-----------------------------------------------------------"""
# def _eval_aircraft_performance(
#     aircraft_performance: AircraftPerformance | None, flight: Flight
# ) -> Flight:
#     """Evaluate the :class:`AircraftPerformance` model.
#
#     Parameters
#     ----------
#     aircraft_performance : AircraftPerformance | None
#         Input aircraft performance model
#     flight : Flight
#         Flight to evaluate
#
#     Returns
#     -------
#     Flight
#         Output from aircraft performance model
#
#     Raises
#     ------
#     ValueError
#         If ``aircraft_performance`` is None
#     """
#
#     ap_vars = {"wingspan", "engine_efficiency", "fuel_flow", "aircraft_mass"}
#     missing = ap_vars.difference(flight).difference(flight.attrs)
#     if not missing:
#         return flight
#
#     if aircraft_performance is None:
#         msg = (
#             f"An AircraftPerformance model parameter is required if the flight does not contain "
#             f"the following variables: {ap_vars}. This flight is missing: {missing}. "
#             "Instantiate the Cocip model with an AircraftPerformance model. "
#             "For example, 'Cocip(..., aircraft_performance=PSFlight(...))'."
#         )
#         raise ValueError(msg)
#
#     return aircraft_performance.eval(source=flight, copy_source=False)
#
#
# def _eval_emissions(emissions: Emissions, flight: Flight) -> Flight:
#     """Evaluate the :class:`Emissions` model.
#
#     Parameters
#     ----------
#     emissions : Emissions
#         Emissions model
#     flight : Flight
#         Flight to evaluate
#
#     Returns
#     -------
#     Flight
#         Output from emissions model
#     """
#
#     emissions_outputs = "nvpm_ei_n"
#     if flight.ensure_vars(emissions_outputs, False):
#         return flight
#     return emissions.eval(source=flight, copy_source=False)

# Required for cocip:
#   nvpm_ei_n
#   engine_efficiency
#   ei_h2o is default for fuel: ei_h2o: float = 1.23

cocip = Cocip(
    met=met, rad=rad, humidity_scaling=HistogramMatching()
)
fcocip = cocip.eval(fl)
# fl
#
# plt.figure()
# fcocip.dataframe.plot.scatter(
#     x="longitude",
#     y="latitude",
#     c="ef",
#     cmap="coolwarm",
#     vmin=-1e13,
#     vmax=1e13,
#     title="EF generated by flight waypoint",
# );
# plt.savefig(f'figures/{flight}/climate/cocip_ef_flight_path.png', format='png')
# # plt.savefig('figures/cocip_ef_flight_path')
# #
# #
# # Plot the first figure and save it
# plt.figure()
# ax1 = plt.axes()
#
# # Plot flight path
# cocip.source.dataframe.plot(
#     "longitude",
#     "latitude",
#     color="k",
#     ax=ax1,
#     label="Flight path",
# )
#
# # Plot contrail LW RF
# cocip.contrail.plot.scatter(
#     "longitude",
#     "latitude",
#     c="rf_lw",
#     cmap="Reds",
#     ax=ax1,
#     label="Contrail LW RF",
# )
#
# ax1.legend()
# # plt.savefig('figures/cocip_lw_rf')
# plt.savefig(f'figures/{flight}/climate/cocip_lw_rf.png', format='png')
#
# # Create a new figure for the second plot
# plt.figure()
# ax2 = plt.axes()
#
# # Plot flight path (assuming you want to plot it again)
# cocip.source.dataframe.plot(
#     "longitude",
#     "latitude",
#     color="k",
#     ax=ax2,
#     label="Flight path",
# )
#
# # Plot contrail SW RF
# cocip.contrail.plot.scatter(
#     "longitude",
#     "latitude",
#     c="rf_sw",
#     cmap="Blues_r",
#     ax=ax2,
#     label="Contrail SW RF",
# )
#
# ax2.legend()
# # plt.savefig('figures/cocip_sw_rf')
# plt.savefig(f'figures/{flight}/climate/cocip_sw_rf.png', format='png')
# plt.show()

# """ACCF"""
#
ac = ACCF(met=met, surface=rad)
fa = ac.eval(fl)
#
# # Waypoint duration in seconds
# dt_sec = fa.segment_duration()
#
# # kg fuel per contrail
# fuel_burn = fa["fuel_flow"] * dt_sec
#
# # Get impacts in degrees K per waypoint
# warming_contrails = fuel_burn * fa["aCCF_Cont"]
# warming_merged = fuel_burn * fa["aCCF_merged"]
#
# f, (ax5, ax6) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
# ax5.plot(fa["time"], warming_contrails, label="Contrails")
# ax5.plot(fa["time"], warming_merged, label="Combined ACCFs")
# ax5.set_ylabel("Degrees K")
# ax5.set_title("Warming impact by waypoint")
# ax5.legend()
#
# ax6.plot(fa["time"], np.cumsum(warming_contrails), label="Contrails")
# ax6.plot(fa["time"], np.cumsum(warming_merged), label="Combined ACCFs")
# ax6.legend()
# ax6.set_xlabel("Waypoint Time")
# ax6.set_ylabel("Degrees K")
# ax6.set_title("Cumulative warming impact");
# # plt.savefig("figures/accf")
#
# # Optionally, show the plots if needed
# plt.show()

