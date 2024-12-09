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
import scipy
from emission_index import p3t3_nox, p3t3_nvpm, p3t3_nvpm_mass, meem_nvpm
from emission_index import NOx_correlation_de_boer, NOx_correlation_kypriandis_optimized_tf, NOx_correlation_kaiser_optimized_tf
from emission_index import p3t3_nvpm_meem, p3t3_nvpm_meem_mass, p3t3_nox_xue
# from piano import altitude_ft_sla
import sys
import pickle
from pycontrails.models.ps_model.ps_model import fuel_mass_flow_rate
from pycontrails import Flight, MetDataset
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import HistogramMatching
from pycontrails.models.ps_model import PSFlight
from openap import FuelFlow
# from ps_model.ps_model import PSFlight
# import ps_model.ps_grid
from pycontrails.models.issr import ISSR
from pycontrails.physics import units
from pycontrails.models.emissions import Emissions
from pycontrails.models.accf import ACCF
from pycontrails.datalib import ecmwf

with open('p3t3_graphs_sls.pkl', 'rb') as f:
    loaded_functions = pickle.load(f)

interp_func_far = loaded_functions['interp_func_far']
interp_func_pt3 = loaded_functions['interp_func_pt3']


"""FLIGHT PARAMETERS"""
engine_model = 'GTF'        # GTF , GTF2035
water_injection = [0, 0, 0]     # WAR climb cruise approach/descent
SAF = 0                     # 0, 20, 100 unit = %
flight = 'malaga'
aircraft = 'A20N_full'        # A20N ps model, A20N_wf is change in Thrust and t/o and idle fuel flows
                            # A20N_wf_opr is with changed nominal opr and bpr
                            # A20N_full has also the eta 1 and 2 and psi_0

# Gasturb reference for GTF war 0 0 0 saf 0 malaga A20N_full
if engine_model == 'GTF' and water_injection[0] == 0 and water_injection[1] == 0 and water_injection[2] == 0 and SAF ==0 and flight == 'malaga' and aircraft == 'A20N_full':
    data_gasturb = {
        "index": [2, 10, 20, 87, 110, 127, 137],
        "fuel_flow_gasturb": [0.6142, 0.4753, 0.2247, 0.2174, 0.1428, 0.1126, 0.1016]
    }

    # Create a DataFrame from the provided data
    df_gasturb = pd.DataFrame(data_gasturb)
    df_gasturb.set_index('index', inplace=True)

    df_piano = pd.read_csv(f"pianoX_malaga.csv", delimiter=';', decimal=',', index_col='index')






"""------READ FLIGHT CSV AND PREPARE FORMAT---------------------------------------"""
df = pd.read_csv(f"{flight}_flight.csv")
df = df.rename(columns={'geoaltitude': 'altitude', 'groundspeed': 'groundspeed', 'timestamp':'time'})
df = df.dropna(subset=['callsign'])
df = df.dropna(subset=['altitude'])
df = df.drop(['Unnamed: 0', 'icao24', 'callsign'], axis=1)

# df = df[df['altitude'] > 1900]
column_order = ['longitude', 'latitude', 'altitude', 'groundspeed', 'time']
df = df[column_order]
df['altitude'] = df['altitude']*0.3048 #foot to meters
df['groundspeed'] = df['groundspeed']*0.514444444
attrs = {
    "flight_id" : "34610D",
    "aircraft_type": f"{aircraft}",
    "engine_uid": "01P22PW163"
}
fl = Flight(df, attrs=attrs)
print('flight length', fl.length)


"""SAMPLE AND FILL DATA"""
fl = fl.resample_and_fill(freq="60s", drop=False) # recommended for CoCiP
fl.dataframe['groundspeed'] = fl.dataframe['groundspeed'].interpolate(method='linear', inplace=True)


"""------RETRIEVE METEOROLOGIC DATA----------------------------------------------"""

time_bounds = ("2024-06-07 9:00", "2024-06-08 02:00")
pressure_levels = (1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175) #hpa

era5pl = ERA5(
    time=time_bounds,
    variables=Cocip.met_variables + Cocip.optional_met_variables + (ecmwf.PotentialVorticity,) + (ecmwf.RelativeHumidity,),
    pressure_levels=pressure_levels,
)
era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,))

# download data from ERA5 (or open from cache)
met = era5pl.open_metdataset() # meteorology
rad = era5sl.open_metdataset() # radiation


"""-----RUN AIRCRAFT PERFORMANCE MODEL--------------------------------------------"""

perf = PSFlight(
    met=met,
    fill_low_altitude_with_isa_temperature=True,  # Estimate temperature using ISA
    fill_low_altitude_with_zero_wind=True
)
fp = perf.eval(fl)
file_name_1 = f'results/{flight}/{flight}_model_{engine_model}_SAF_0_aircraft_A20N_full_WAR_0_0_0.csv'
df_gsp = pd.read_csv(file_name_1)

file_name_2 = f'pollman_properties.csv'
df_poll = pd.read_csv(file_name_2)

merged_df = pd.merge(df_gsp, df_poll, on="air_temperature", how="inner")
q_fuel = 43.13e6
merged_df['eta_gsp'] = (merged_df['thrust_gsp']*1000*merged_df['true_airspeed']) / (merged_df['fuel_flow_gsp']*q_fuel)


merged_df['fuel_flow_poll_efficiencies_gsp'] = merged_df.apply(
    lambda row: fuel_mass_flow_rate(
        row['air_pressure_y'],
        row['air_temperature'],
        row['mach_y'],
        row['c_t'],
        row['eta_gsp'],
        row['wing_surface'],
        row['q_fuel']
    ),
    axis=1
)

plt.figure(figsize=(10, 6))
# plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_py'], label='Pycontrails', linestyle='-', marker='o')
plt.plot(df_gsp.index, df_gsp['fuel_flow_per_engine'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['fuel_flow_gsp'], label='GSP', linestyle='-', marker='o', markersize=2.5)
try:
    if data_gasturb:
        plt.scatter(df_gasturb.index, df_gasturb['fuel_flow_gasturb'], label='GasTurb', marker='o', s=25, color='red')
        plt.plot(df_piano.index, df_piano['fuel_flow_piano'], label='PianoX', linestyle='-', marker='o', markersize=2.5)
        # plt.plot(df_gsp.index, df_gsp['fuel_flow_openap']/2, label='OpenAP', linestyle='-', marker='o', markersize=2.5)
except NameError:
    print("Variable does not exist, skipping.")
plt.plot(df_gsp.index, merged_df['fuel_flow_poll_efficiencies_gsp']/2, label='Pycontrails - GSP eta', linestyle='-', marker='o', markersize=2.5, color='purple')
plt.title('Fuel Flow')
plt.xlabel('Time in minutes')
plt.ylabel('Fuel Flow (kg/s)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/fuel_flow_poll_gsp_efficiencies.png', format='png')
# plt.show()