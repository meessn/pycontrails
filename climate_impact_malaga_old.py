import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
import scipy

from pycontrails import Flight, MetDataset
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import HistogramMatching
from pycontrails.models.ps_model import PSFlight
from pycontrails.models.issr import ISSR
from pycontrails.physics import units
from pycontrails.models.emissions import Emissions
from pycontrails.models.accf import ACCF
from pycontrails.datalib import ecmwf



"""------READ FLIGHT CSV AND PREPARE FORMAT---------------------------------------"""
df = pd.read_csv("malaga_flight.csv")
df = df.rename(columns={'geoaltitude': 'altitude', 'groundspeed': 'true_airspeed', 'timestamp':'time'})
df = df.drop(['Unnamed: 0', 'icao24', 'callsign'], axis=1)
df = df[df['altitude'] > 30000]
column_order = ['longitude', 'latitude', 'altitude', 'true_airspeed', 'time']
df = df[column_order]
df['altitude'] = df['altitude']*0.3048 #foot to meters
attrs= {
    "flight_id" : "malaga",
    "aircraft_type": "A320"
}
fl = Flight(df, attrs=attrs)

"""PLOT ORIGINAL DATA FLIGHT"""
fig2, ax2 = plt.subplots()
ax2 = fl.dataframe.plot.scatter(x="longitude", y="latitude", figsize=(12, 8))
ax2.set_title('Flight Path Original')
# plt.savefig('figures/flight_data.png')

"""SAMPLE AND FILL DATA"""
fl = fl.resample_and_fill(freq="60s", drop=False) # recommended for CoCiP
fl.dataframe.loc[23, 'true_airspeed'] = (fl.dataframe.loc[22, 'true_airspeed'] + fl.dataframe.loc[24, 'true_airspeed']) / 2
fl.dataframe.loc[83, 'true_airspeed'] = (fl.dataframe.loc[82, 'true_airspeed'] + fl.dataframe.loc[84, 'true_airspeed']) / 2
# fl.dataframe.loc[100, 'true_airspeed'] = (fl.dataframe.loc[99, 'true_airspeed'] + fl.dataframe.loc[101, 'true_airspeed']) / 2

"""PLOT SAMPLED DATA FLIGHT"""
fig3, ax3 = plt.subplots()
ax3 = fl.dataframe.plot.scatter(x="longitude", y="latitude", figsize=(12, 8))
ax3.set_title('Flight Path SAMPLED')
# plt.savefig('figures/flight_data_sampled.png')

# fl = fl.dataframe.interpolate()
"""------------------------------------------------------------------------------"""
"""------RETRIEVE METEOROLOGIC DATA----------------------------------------------"""

time_bounds = ("2024-06-07 9:00", "2024-06-08 02:00")
pressure_levels = (350, 300, 250, 225, 200, 175)

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

# Use altitude_ft as the vertical coordinate for plotting
da["altitude_ft"] = units.pl_to_ft(da["level"]).round().astype(int)
da = da.swap_dims(level="altitude_ft")
da = da.sel(altitude_ft=da["altitude_ft"].values[::-1])
da = da.squeeze()

da.plot(x="longitude", y="latitude", col="altitude_ft", col_wrap=4, add_colorbar=False);
# plt.savefig('figures/issr_regions.png')



"""-------------------------------------------------------------------------------"""


"""-----RUN AIRCRAFT PERFORMANCE MODEL--------------------------------------------"""
perf = PSFlight(met=met)
fp = perf.eval(fl)


"""PLOT EFFICIENCY AND FUEL MASS FLOW"""
fig, ax = plt.subplots()
# fig.set_size_inches(9,5)
axf = ax.twinx()
ax.set_title('Aircraft Performance')
axf.set_ylabel('Fuel Flow [kg/s]')
ax.set_ylabel('Engine Efficiency')
# axf.set_yticks([])
fp.dataframe.plot(ax=ax, x="time", y="engine_efficiency", style="r", legend=False)
fp.dataframe.plot(ax=axf, x="time", y="fuel_flow", style="k", legend=False)
_ = axf.legend(
    [ax.get_lines()[0], axf.get_lines()[0]],
    ["Engine Efficiency", "Fuel Flow"],
    bbox_to_anchor=(0.5, 0.25),
)
fig.tight_layout()
# plt.savefig('figures/ps_model_eff_fuel')

"""PLOT THRUST"""
fig1, ax1 = plt.subplots()
ax1 = fp.dataframe.plot(x="time", y="thrust", style="b", legend=False)
def kN_formatter(x, pos):
    return f'{x * 1e-3:.1f}'
ax1.yaxis.set_major_formatter(FuncFormatter(kN_formatter))
ax1.set_title('Aircraft Thrust vs Time')
ax1.set_ylabel('Thrust [kN]')
# plt.savefig('figures/ps_model_thrust')


"""--------------------------------------------------------------------------------"""
"""---------EMISSIONS MODEL -------------------------------------------------------"""
emissions = Emissions(met=met, humidity_scaling=HistogramMatching())
f_test = fp.dataframe.drop('thrust', axis=1)
f_test = Flight(f_test)
fe = emissions.eval(f_test)
fe

# plt.figure()
fig4, ax4 = plt.subplots()
axfu = ax4.twinx()

fe.dataframe.plot(ax=ax4, x="time", y="nvpm_mass", style="r", legend=False)
fe.dataframe.plot(ax=axfu, x="time", y="fuel_flow", style="b", legend=False)
ax4.set_ylabel('nvPM Mass [kg]')
axfu.set_ylabel('Fuel Flow [kg / s]')
ax4.set_title('nvPM Mass and Fuel Flow')
axfu.legend(
    [
        ax4.get_lines()[0],
        axfu.get_lines()[0],
    ],
    ["nvPM Mass", "Fuel Flow"],
    bbox_to_anchor=(0.5, 0.25),
)
# plt.savefig('figures/emi_nvpm_fuel')

fig5, ax5 = plt.subplots()
axc = ax5.twinx()
def c_formatter(x, pos):
    return f'{x*10**3 :.1f}'
axc.yaxis.set_major_formatter(FuncFormatter(c_formatter))
fe.dataframe.plot(ax=ax5, x="time", y="nox", style="r", legend=False)
fe.dataframe.plot(ax=axc, x="time", y="co", style="b", legend=False)
ax5.set_ylabel('NOx [kg]')
axc.set_ylabel('CO [g]')
ax5.set_title('NOx and CO emissions')
axc.legend(
    [
        ax5.get_lines()[0],
        axc.get_lines()[0],
    ],
    ["NOx", "CO"],
    bbox_to_anchor=(0.5, 0.25),
)
# plt.savefig('figures/emi_nox_co')


"""-------------------------------------------------------------------------------------"""
"""---------------------CoCiP-----------------------------------------------------------"""
cocip = Cocip(
    met=met, rad=rad, aircraft_performance=PSFlight(), humidity_scaling=HistogramMatching()
)
fcocip = cocip.eval(fl)
# fl
#
plt.figure()
fcocip.dataframe.plot.scatter(
    x="longitude",
    y="latitude",
    c="ef",
    cmap="coolwarm",
    vmin=-1e13,
    vmax=1e13,
    title="EF generated by flight waypoint",
);
# plt.savefig('figures/cocip_ef_flight_path')
#
#
# Plot the first figure and save it
plt.figure()
ax1 = plt.axes()

# Plot flight path
cocip.source.dataframe.plot(
    "longitude",
    "latitude",
    color="k",
    ax=ax1,
    label="Flight path",
)

# Plot contrail LW RF
cocip.contrail.plot.scatter(
    "longitude",
    "latitude",
    c="rf_lw",
    cmap="Reds",
    ax=ax1,
    label="Contrail LW RF",
)

ax1.legend()
# plt.savefig('figures/cocip_lw_rf')

# Create a new figure for the second plot
plt.figure()
ax2 = plt.axes()

# Plot flight path (assuming you want to plot it again)
cocip.source.dataframe.plot(
    "longitude",
    "latitude",
    color="k",
    ax=ax2,
    label="Flight path",
)

# Plot contrail SW RF
cocip.contrail.plot.scatter(
    "longitude",
    "latitude",
    c="rf_sw",
    cmap="Blues_r",
    ax=ax2,
    label="Contrail SW RF",
)

ax2.legend()
# plt.savefig('figures/cocip_sw_rf')

"""MET DATA FOR ACCF"""
time_bounds = ("2024-06-07 9:00", "2024-06-08 02:00")
pressure_levels = (350, 300, 250, 225, 200, 175)


"""ACCF"""

ac = ACCF(met=met, surface=rad)
fa = ac.eval(fp)

# Waypoint duration in seconds
dt_sec = fa.segment_duration()

# kg fuel per contrail
fuel_burn = fa["fuel_flow"] * dt_sec

# Get impacts in degrees K per waypoint
fa['warming_contrails'] = fuel_burn * fa["aCCF_Cont"]
fa['warming_merged'] = fuel_burn * fa["aCCF_merged"]

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
# plt.savefig("figures/accf")

fa['index'] = range(len(fa))


plt.figure(figsize=(10, 6))
plt.plot(fa['index'], fa['warming_contrails'])
plt.title('Contrail warming impact')
plt.xlabel('Time in minutes')
plt.ylabel('Degrees K ')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/malaga/climate/contrail_accf_impact_pycontrails_emissions.png', format='png')

# Optionally, show the plots if needed
plt.show()

