import pandas as pd
import numpy as np
import xarray as xr
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

def add_relative_humidity_to_metdataset(met: MetDataset) -> MetDataset:
    """Compute and add relative humidity to the MetDataset."""
    # Ensure required variables exist
    required_vars = ["specific_humidity", "air_temperature", "air_pressure"]
    met.ensure_vars(required_vars)

    # Extract necessary variables
    q = met["specific_humidity"].data  # Specific humidity
    T = met["air_temperature"].data  # Temperature in Kelvin
    p = met["air_pressure"].data  # Pressure in Pa

    # Compute relative humidity
    rh_values = rh(q, T, p)

    # Add relative humidity to the dataset
    met[RelativeHumidity.standard_name] = xr.DataArray(
        rh_values,
        coords=met.data.coords,
        dims=met.data.dims,
        attrs={
            "units": RelativeHumidity.units,
            "long_name": RelativeHumidity.long_name,
            "description": RelativeHumidity.description,
            "grib1_id": RelativeHumidity.grib1_id,
            "ecmwf_id": RelativeHumidity.ecmwf_id,
            "grib2_id": RelativeHumidity.grib2_id,
            "amip": RelativeHumidity.amip,
            "level_type": RelativeHumidity.level_type,
        },
    )

    return met




"""FLIGHT PARAMETERS"""
engine_model = 'GTF'        # GTF , GTF2035
water_injection = [0, 0, 0]     # WAR climb cruise approach/descent
SAF = 0    # 0, 20, 100 unit = %
#VERGEET NIET SAF LHV EN H2O en CO2  MEE TE GEVEN AAN PYCONTRAILS EN ACCF!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
flight = 'malaga'
aircraft = 'A20N_full'        # A20N ps model, A20N_wf is change in Thrust and t/o and idle fuel flows
prediction = 'mees'            #mees or pycontrails
                            # A20N_wf_opr is with changed nominal opr and bpr
                            # A20N_full has also the eta 1 and 2 and psi_0
diurnal = 'day'             # day / night
weather_model = 'era5'      # era5 / era5model


# Convert the water_injection values to strings, replacing '.' with '_'
formatted_values = [str(value).replace('.', '_') for value in water_injection]
file_path = f'results/{flight}/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv'

df = pd.read_csv(file_path)
if prediction == 'pycontrails':
    columns_to_drop = [
        'air_pressure'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df['ei_nox'] = df['nox_ei']
    df['ei_co2'] = df['ei_co2_conservative']


if prediction != 'pycontrails':
    columns_to_drop = [
        'nox_ei', 'co_ei', 'hc_ei', 'nvpm_ei_m', 'nvpm_ei_n', 'co2', 'h2o',
        'so2', 'sulphates', 'oc', 'nox', 'co', 'hc', 'nvpm_mass', 'nvpm_number'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    df = df.rename(columns={
        'EI_nvpm_number_p3t3_meem': 'nvpm_ei_n',
    })

    df['ei_nox'] = df['EI_nox_p3t3'] / 1000
    df['nvpm_ei_m'] = df['EI_nvpm_mass_p3t3_meem'] / 10**6

    df = df.drop(columns=['EI_nox_p3t3', 'EI_nvpm_mass_p3t3_meem'], errors='ignore')

    """Correct inputs for pycontrails climate impact methods -> compute everything for two engines"""
    df['fuel_flow'] = 2*df['fuel_flow_gsp']
    df['thrust'] = 2*df['thrust_gsp']
    df['air_pressure'] = df['air_pressure']*10**5
    df['ei_co2'] = df['ei_co2_conservative']
    q_fuel = df['LHV'].iloc[0]*1000
    df['engine_efficiency'] = (df['thrust_gsp']*1000*df['true_airspeed']) / (df['fuel_flow_gsp']*q_fuel)

SAF = df['SAF'].iloc[0]
if SAF == 0:
    fuel = JetA()
elif SAF == 20:
    fuel = SAF20()
elif SAF == 100:
    fuel = SAF100()
else:
    raise ValueError(f"Unsupported SAF value: {SAF}")

#wingspan needed as aircraft / engine are not defined (extra safety measure that data does not get overwritten)
df['wingspan'] = 35.8



fl = Flight(data=df, fuel=fuel)
fl_issr = Flight(data=df.copy(), fuel=fuel)
fl_cocip = Flight(data=df.copy(), fuel=fuel)
fl_accf = Flight(data=df.copy(), fuel=fuel)
# print("fl" , fl.dataframe['rhi'].sum)
"""------RETRIEVE METEOROLOGIC DATA----------------------------------------------"""

# time_bounds = ("2018-06-15 12:00:00", "2018-06-15 13:00:00")pressure_levels = [250]
time_bounds = ("2024-06-07 09:00", "2024-06-08 02:00")
pressure_levels = (1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175) #hpa
# pressure_levels = (350, 300, 250, 225, 200, 175)



if weather_model == 'era5':
    era5pl = ERA5(
        time=time_bounds,
        variables=Cocip.met_variables + Cocip.optional_met_variables + (ecmwf.PotentialVorticity,) + (ecmwf.RelativeHumidity,),
        pressure_levels=pressure_levels,
    )
    era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,))

    # download data from ERA5 (or open from cache)
    met = era5pl.open_metdataset().copy() # meteorology
    met_issr = era5pl.open_metdataset().copy()
    met_cocip = era5pl.open_metdataset().copy()
    met_accf = era5pl.open_metdataset().copy()
    rad = era5sl.open_metdataset().copy() # radiation
    rad_issr = era5sl.open_metdataset().copy()
    rad_cocip = era5sl.open_metdataset().copy()
    rad_accf = era5sl.open_metdataset().copy()
elif weather_model == 'era5model':
    # url = "https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions"
    # df_weather = pd.read_html(url, na_values="-", index_col="n")[0].rename_axis("model_level")
    # df_weather.loc[67:137]  # model levels 70 - 90 agree with our ERA5ModelLevel object below
    # Define pressure levels
    pressure_levels_10 = np.arange(150, 400, 10)  # 150 to 400 with steps of 10
    pressure_levels_50 = np.arange(400, 1001, 50)  # 400 to 1000 with steps of 50

    # Combine the two arrays
    pressure_levels_model = np.concatenate((pressure_levels_10, pressure_levels_50))
    # paths = ["C:/era5model/malaga/7bb44ca286a873689d7b8884bcd7d548.nc", "C:/era5model/malaga/67e727ad0e2ad65747f2db9add2d5ad1.nc"]
    local_cache_dir = Path("C:/era5model/malaga")
    local_cachestore = DiskCacheStore(cache_dir=local_cache_dir)

    era5ml = ERA5ModelLevel(
        time=time_bounds,
        variables=("t", "q", "u", "v", "w", "ciwc", "vo", "clwc"),
        # paths=paths,
        # grid=1,  # horizontal resolution, 0.25 by default
        model_levels=range(67, 133),
        pressure_levels=pressure_levels_model,
        cachestore=local_cachestore
    )
    met = era5ml.open_metdataset()
    met = add_relative_humidity_to_metdataset(met)


    era5sl = ERA5(
        time=time_bounds,
        variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,)
        # grid=1,
        # pressure_levels=pressure_levels,
    )
    rad = era5sl.open_metdataset()





"""use ssdr to check day / night"""
# Step 1: Perform the intersection
intersected_values = fl.intersect_met(rad['surface_solar_downward_radiation'])
fl["surface_solar_downward_radiation"] = intersected_values
total_solar_radiation = intersected_values.sum()
# diurnal = fl.get("diurnal", None)  # Ensure 'diurnal' is set in the Flight object
if diurnal == "night" and total_solar_radiation != 0:
    raise Warning("Error: Solar radiation is non-zero for a nighttime flight.")
elif diurnal == "day" and total_solar_radiation == 0:
    raise Warning("Warning: Solar radiation is zero for a daytime flight. Check the data.")

    # Additional Check: Report any zero values for daytime flights
    zero_indices = [i for i, val in enumerate(intersected_values) if val == 0]
    if zero_indices:
        print(f"Warning: Solar radiation is zero at the following waypoints (indices): {zero_indices}")

# Optional: Print results for debugging
print(f"Total Solar Radiation: {total_solar_radiation}")
print(f"Diurnal: {diurnal}")

"""ISSRs"""
# issr_mds = ISSR(met=met, humidity_scaling=ExponentialBoostHumidityScaling(rhi_adj=0.9779, rhi_boost_exponent=1.635,
#                                                                         clip_upper=1.65)).eval()
#
# issr = issr_mds["issr"]
# da = issr.data.isel(time=0)
# target_levels = [200, 300, 400, 600, 800, 1000]
# da = da.sel(level=target_levels, method="nearest")
# da["altitude_m"] = units.pl_to_m(da["level"]).round().astype(int)
# da = da.swap_dims(level="altitude_m")
# da = da.sel(altitude_m=da["altitude_m"].values[::-1])
# da = da.squeeze()
# da.plot(x="longitude", y="latitude",  col="altitude_m", col_wrap=3, cmap="Reds", figsize=(12, 12));
# plt.savefig(f'figures/{flight}/climate/issr_regions.png', format='png')

issr_flight = ISSR(met=met_issr, humidity_scaling=ExponentialBoostHumidityScaling(rhi_adj=0.9779, rhi_boost_exponent=1.635,
                                                                        clip_upper=1.65)).eval(source=fl_issr)

df_climate_results = fl_issr.dataframe.copy() #issr_flight.dataframe.copy()
df_issr_flight = issr_flight.dataframe.copy()
new_columns_issr_flight = df_issr_flight.drop(columns=df_climate_results.columns, errors='ignore')
new_columns_issr_flight.columns = ['issr_' + col for col in new_columns_issr_flight.columns]
# print("issr" , issr_flight.dataframe['rhi'].sum)
fig, ax = plt.subplots(figsize=(10, 6))

# Create colormap with red for ISSR and blue for non-ISSR
cmap = ListedColormap(["b", "r"])

ax.scatter(issr_flight["longitude"], issr_flight["latitude"], c=issr_flight["issr"], cmap=cmap)


# Create legend
legend_elements = [
    plt.Line2D([0], [0], marker="o", color="w", label="ISSR", markerfacecolor="r", markersize=10),
    plt.Line2D(
        [0], [0], marker="o", color="w", label="non-ISSR", markerfacecolor="b", markersize=10
    ),
]
ax.legend(handles=legend_elements, loc="upper left")

ax.set(xlabel="longitude", ylabel="latitude");
if prediction == 'pycontrails':
    plt.savefig(f'figures/{flight}/climate/pycontrails/issr_regions_along_flight.png', format='png')
elif weather_model == 'era5model':
    plt.savefig(f'figures/{flight}/climate/era5model/issr_regions_along_flight.png', format='png')
else:
    plt.savefig(f'figures/{flight}/climate/issr_regions_along_flight.png', format='png')


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
# humidity scaling to combat era5 ice-supersaturation under-representation
cocip = Cocip(
    met=met_cocip, rad=rad_cocip, humidity_scaling=ExponentialBoostHumidityScaling(rhi_adj=0.9779, rhi_boost_exponent=1.635,
                                                                        clip_upper=1.65), verbose_outputs=True,
    compute_atr20=True, process_emissions=False
)
fcocip = cocip.eval(fl_cocip)
# df_fcocip = fcocip.dataframe.copy()
# print("cocip" , fcocip.dataframe['rhi'].sum)


fcocip_eval_flight = flight_waypoint_summary_statistics(fcocip, cocip.contrail)
fcocip_eval_contrail = contrail_flight_summary_statistics(fcocip_eval_flight)
df_climate_contrail_results = fcocip_eval_contrail.copy()
if prediction == 'pycontrails':
    df_climate_contrail_results.to_csv(
        f'results/{flight}/climate/pycontrails/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}_climate_contrails.csv')
elif weather_model == 'era5model':
    df_climate_contrail_results.to_csv(
        f'results/{flight}/climate/era5model/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}_climate_contrails.csv')
else:
    df_climate_contrail_results.to_csv(
        f'results/{flight}/climate/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}_climate_contrails.csv')

df_fcocip = fcocip_eval_flight.dataframe.copy()
new_columns_fcocip = df_fcocip.drop(columns=df_climate_results.columns, errors='ignore')
new_columns_fcocip.columns = ['cocip_' + col for col in new_columns_fcocip.columns]
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
if prediction == 'pycontrails':
    plt.savefig(f'figures/{flight}/climate/pycontrails/cocip_ef_flight_path.png', format='png')
elif weather_model == 'era5model':
    plt.savefig(f'figures/{flight}/climate/era5model/cocip_ef_flight_path.png', format='png')
else:
    plt.savefig(f'figures/{flight}/climate/cocip_ef_flight_path.png', format='png')


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
if prediction == 'pycontrails':
    plt.savefig(f'figures/{flight}/climate/pycontrails/cocip_lw_rf.png', format='png')
elif weather_model == 'era5model':
    plt.savefig(f'figures/{flight}/climate/era5model/cocip_lw_rf.png', format='png')
else:
    plt.savefig(f'figures/{flight}/climate/cocip_lw_rf.png', format='png')


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
if prediction == 'pycontrails':
    plt.savefig(f'figures/{flight}/climate/pycontrails/cocip_sw_rf.png', format='png')
elif weather_model == 'era5model':
    plt.savefig(f'figures/{flight}/climate/era5model/cocip_sw_rf.png', format='png')
else:
    plt.savefig(f'figures/{flight}/climate/cocip_sw_rf.png', format='png')



# """ACCF"""
#
if weather_model == 'era5' or weather_model == 'era5model':
    accf = ACCF(
        met=met_accf,
        surface=rad_accf,
        params={
            "emission_scenario": "pulse",
            "accf_v": "V1.0",  "issr_rhi_threshold": 0.9, "efficacy": True, "PMO": False,
            "horizontal_resolution": 0.25,
            "forecast_step": 1.0
            # "pfca": "PCFA-SAC",
            # "sac_eta": fl.dataframe['engine_efficiency']
            # "pfca": "PCFA-SAC"
        }
        # verify_met=True
    )
    fa = accf.eval(fl_accf)

    # Waypoint duration in seconds
    # dt_sec = fa.segment_duration()
    df_accf = fa.dataframe.copy()
    # kg fuel per contrail
    df_accf['fuel_burn'] = df_accf["fuel_flow"] * 60

    # Get impacts in degrees K per waypoint
    df_accf['nox_impact'] = df_accf['fuel_burn'] * df_accf["aCCF_NOx"] * df_accf['ei_nox']
    if df_accf['SAF'].iloc[0] != 0:
        df_accf['co2_impact_conservative'] = df_accf['fuel_burn'] * df_accf["aCCF_CO2"] * df_accf['ei_co2_conservative']
        df_accf['co2_impact_optimistic'] = df_accf['fuel_burn'] * df_accf["aCCF_CO2"] * df_accf['ei_co2_optimistic']
    else:
        df_accf['co2_impact'] = df_accf['fuel_burn'] * df_accf["aCCF_CO2"] * df_accf['ei_co2']
    df_accf['warming_contrails'] = df_accf['fuel_burn'] * df_accf["aCCF_Cont"]






    plt.figure(figsize=(10, 6))
    plt.plot(df_accf['index'], df_accf['aCCF_CH4'], label="aCCF CH4")
    plt.plot(df_accf['index'], df_accf['aCCF_O3'], label="aCCF O3")
    plt.plot(df_accf['index'], df_accf['aCCF_NOx'], label="aCCF NOx")
    plt.title('aCCF K / kg species')
    plt.xlabel('Time in minutes')
    plt.ylabel('Degrees K / kg species')
    plt.legend()
    plt.grid(True)
    if prediction == 'pycontrails':
        plt.savefig(f'figures/{flight}/climate/pycontrails/nox_accf.png', format='png')
    elif weather_model == 'era5model':
        plt.savefig(f'figures/{flight}/climate/era5model/nox_accf.png', format='png')
    else:
        plt.savefig(f'figures/{flight}/climate/nox_accf.png', format='png')


    plt.figure(figsize=(10, 6))
    plt.plot(df_accf['index'], df_accf['aCCF_Cont'])
    plt.title('Contrail warming impact aCCF K / kg fuel')
    plt.xlabel('Time in minutes')
    plt.ylabel('Degrees K / kg fuel ')
    # plt.legend()
    plt.grid(True)
    if prediction == 'pycontrails':
        plt.savefig(f'figures/{flight}/climate/pycontrails/contrail_accf.png', format='png')
    elif weather_model == 'era5model':
        plt.savefig(f'figures/{flight}/climate/era5model/contrail_accf.png', format='png')
    else:
        plt.savefig(f'figures/{flight}/climate/contrail_accf.png', format='png')


    plt.figure(figsize=(10, 6))
    plt.plot(df_accf['index'], df_accf['warming_contrails'])
    plt.title('Contrail warming impact')
    plt.xlabel('Time in minutes')
    plt.ylabel('Degrees K ')
    # plt.legend()
    plt.grid(True)
    if prediction == 'pycontrails':
        plt.savefig(f'figures/{flight}/climate/pycontrails/contrail_accf_impact.png', format='png')
    elif weather_model == 'era5model':
        plt.savefig(f'figures/{flight}/climate/era5model/contrail_accf_impact.png', format='png')
    else:
        plt.savefig(f'figures/{flight}/climate/contrail_accf_impact.png', format='png')


    plt.figure(figsize=(10, 6))
    plt.plot(df_accf['index'], df_accf['nox_impact'], label="NOx")
    if df_accf['SAF'].iloc[0] != 0:
        plt.plot(df_accf['index'], df_accf['co2_impact_conservative'], label="CO2 Conservative")
        plt.plot(df_accf['index'], df_accf['co2_impact_optimistic'], label="CO2 Optimistic")
    else:
        plt.plot(df_accf['index'], df_accf['co2_impact'], label="CO2")
    plt.title('Warming impact by waypoint')
    plt.xlabel('Time in minutes')
    plt.ylabel('Degrees K')
    plt.legend()
    plt.grid(True)
    if prediction == 'pycontrails':
        plt.savefig(f'figures/{flight}/climate/pycontrails/nox_co2_impact.png', format='png')
    elif weather_model == 'era5model':
        plt.savefig(f'figures/{flight}/climate/era5model/nox_co2_impact.png', format='png')
    else:
        plt.savefig(f'figures/{flight}/climate/nox_co2_impact.png', format='png')


# elif weather_model == 'era5model':
    # df_accf = fl.dataframe.copy()

new_columns_df_accf = df_accf.drop(columns=df_climate_results.columns, errors='ignore')
# new_columns_df_accf = new_columns_df_accf.drop(['sac'], axis=1)
new_columns_df_accf.columns = ['accf_' + col for col in new_columns_df_accf.columns]
# Define the shared columns to check
shared_columns = ['longitude', 'latitude', 'altitude']  # Columns to compare

# Function to check shared columns for mismatches across three dataframes
def check_shared_columns(df1, df2, df3, shared_columns):
    for col in shared_columns:
        if not (df1[col].equals(df2[col]) and df2[col].equals(df3[col])):
            raise ValueError(f"Mismatched values in column: {col}")

# Check if shared columns match between df_fcocip, df_accf, and df_third
check_shared_columns(df_fcocip, df_accf, df_issr_flight, shared_columns)

# Concatenate new columns to the base DataFrame
df_climate_results = pd.concat([df_climate_results,  new_columns_issr_flight, new_columns_fcocip, new_columns_df_accf], axis=1)

if prediction == 'pycontrails':
    df_climate_results.to_csv(
        f'results/{flight}/climate/pycontrails/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}_climate.csv')
elif weather_model == 'era5model':
    df_climate_results.to_csv(
        f'results/{flight}/climate/era5model/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}_climate.csv')
else:
    df_climate_results.to_csv(
        f'results/{flight}/climate/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}_climate.csv')