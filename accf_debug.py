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
# trajectory = 'malaga'
# flight = 'malaga'
# engine_model = 'GTF'
# SAF = 0
# formatted_values = [0,0,0]
# aircraft = 'A20N_full'
# file_path = f'main_results_figures/results/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv'
# df = pd.read_csv(file_path)
#
# pressure_levels = (1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175)
# if flight == 'malaga':
#     time_bounds = ("2024-06-07 09:00", "2024-06-08 02:00")
#     local_cache_dir_era5m = Path("F:/era5model/malaga")
#     variables_model = ("t", "q", "u", "v", "w", "ciwc", "vo", "clwc")
#
# local_cache_dir_era5p = Path("F:/era5pressure/Cache")
# local_cachestore_era5p = DiskCacheStore(cache_dir=local_cache_dir_era5p)
#
# era5pl = ERA5(
#             time=time_bounds,
#             variables=Cocip.met_variables + Cocip.optional_met_variables + (ecmwf.PotentialVorticity,) + (ecmwf.RelativeHumidity,),
#             pressure_levels=pressure_levels,
#             cachestore=local_cachestore_era5p
#         )
# era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,), cachestore=local_cachestore_era5p)
#
# met = era5pl.open_metdataset()
# rad = era5sl.open_metdataset()
#
# accf_issr = ACCF(
#     met=met,
#     surface=rad,
#     params={
#         "emission_scenario": "pulse",
#         "accf_v": "V1.0",  "issr_rhi_threshold": 0.9, "efficacy": True, "PMO": False,
#         "horizontal_resolution": 0.25,
#         "forecast_step": None,
#         "pfca": "PCFA-ISSR",
#         "unit_K_per_kg_fuel": True
#         # "sac_eta": fl.dataframe['engine_efficiency']
#         # "pfca": "PCFA-SAC"
#     },
#     verify_met=False
#     )

df_climate = pd.read_csv('main_results_figures/results/malaga/malaga/climate/mees/era5model/GTF_SAF_0_A20N_full_WAR_0_climate.csv')
S = 1360
fl = Flight(data=df_climate)
segment_lengths = fl.segment_length()
df_climate['segment_length_m'] = segment_lengths
df_climate['segment_length_km'] = df_climate['segment_length_m'] / 1000
# Extract row 78 (note: index 77 because Python is 0-based)
row = df_climate.iloc[77]

# Extract values
t = row['air_temperature']
gp = row['accf_sac_geopotential']
Fin = row['accf_sac_Fin']
phi_deg = row['latitude']     # assuming the column is named 'latitude'
N = 159
print(t)
print(gp)
print(Fin)
# Compute aCCFs
accf_o3 = -5.20e-11 + 2.30e-13 * t + 4.85e-16 * gp - 2.04e-18 * t * gp
accf_ch4 = -9.83e-13 + 1.99e-18 * gp - 6.32e-16 * Fin + 6.12e-21 * gp * Fin

# efficacy and scaling factor
accf_o3 = accf_o3*1.37/1.97
accf_ch4 = accf_ch4*1.18/2.03
# PMO factor
pmo = 0.29*accf_ch4

# Total NOx aCCF
accf_nox = accf_o3 + accf_ch4 + pmo

# Print results
print(f"aCCF_O3: {accf_o3:.4e} K/kg(NO2)")
print(f"aCCF_CH4: {accf_ch4:.4e} K/kg(NO2)")
print(f"Total NOx aCCF: {accf_nox:.4e} K/kg(NO2)")

accf_o3_yin = -2.64e-11 + 1.17e-13 * t + 2.46e-16 * gp - 1.04e-18 * t * gp
accf_ch4_yin = -4.84e-13 + 9.79e-19 * gp - 3.11e-16 * Fin + 3.01e-21 * gp * Fin
# efficacy and scaling factor
accf_o3_yin = accf_o3_yin *1.37
accf_ch4_yin = accf_ch4_yin *1.18
pmo_yin = 0.29*accf_ch4_yin
accf_nox_yin = accf_o3_yin + accf_ch4_yin + pmo_yin
print(f"aCCF_O3_yin: {accf_o3_yin:.4e} K/kg(NO2)")
print(f"aCCF_CH4_yin: {accf_ch4_yin:.4e} K/kg(NO2)")
print(f"Total NOx aCCF yin: {accf_nox_yin:.4e} K/kg(NO2)")
phi = np.deg2rad(phi_deg)

# Solar declination (δ), also in radians
delta = np.deg2rad(-23.44 * np.cos(np.deg2rad(360 / 365 * (N + 10))))

# cos(theta)
cos_theta = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta)

# Final calculation of Fin
Fin = S * cos_theta
print(Fin)

m = 11400
pl = m_to_pl(11400)
print(pl)

pl_accf_max = 300
pl_accf_min = 150
m_accf_lowest = pl_to_m(300)
m_accf_highest = pl_to_m(150)
print('accf lowest altitude (m)', m_accf_lowest)
print('accf highest altitude (m)', m_accf_highest)

dt = 60  # seconds per row
df_climate['accf_h2o'] = df_climate['accf_sac_aCCF_H2O'] * df_climate['fuel_flow'] * dt
df_climate['accf_co2'] = df_climate['accf_sac_aCCF_CO2'] * df_climate['fuel_flow'] * dt

# Filter rows above 9163.8m
df_high = df_climate[df_climate['altitude'] > 9160].copy()

# Extract variables
t = df_high['air_temperature']
gp = df_high['accf_sac_geopotential']
Fin = df_high['accf_sac_Fin']
fuel_flow = df_high['fuel_flow']
ei_nox = df_high['ei_nox']

# --- Common contributions ---
# df_high['accf_h2o'] = df_high['accf_sac_aCCF_H2O'] * fuel_flow * 60
# df_high['accf_co2'] = df_high['accf_sac_aCCF_CO2'] * fuel_flow * 60

# --- Method 1: NOx default ---
accf_o3 = -5.20e-11 + 2.30e-13 * t + 4.85e-16 * gp - 2.04e-18 * t * gp
accf_ch4 = -9.83e-13 + 1.99e-18 * gp - 6.32e-16 * Fin + 6.12e-21 * gp * Fin
accf_o3 *= 1.37 / 1.97
accf_ch4 *= 1.18 / 2.03
df_high['accf_nox_default'] = (accf_o3 + accf_ch4 + 0.29 * accf_ch4) * ei_nox * fuel_flow * 60

# --- Method 2: NOx yin (with factors) ---
accf_o3_yin = -2.64e-11 + 1.17e-13 * t + 2.46e-16 * gp - 1.04e-18 * t * gp
accf_ch4_yin = -4.84e-13 + 9.79e-19 * gp - 3.11e-16 * Fin + 3.01e-21 * gp * Fin
accf_o3_yin_scaled = accf_o3_yin * 1.37
accf_ch4_yin_scaled = accf_ch4_yin * 1.18
df_high['accf_nox_yin'] = (accf_o3_yin_scaled + accf_ch4_yin_scaled + 0.29 * accf_ch4_yin_scaled) * ei_nox * fuel_flow * 60

# --- Method 3: NOx yin (no scaling) ---
df_high['accf_nox_yin_nofactor'] = (accf_o3_yin + accf_ch4_yin + 0.29 * accf_ch4_yin) * ei_nox * fuel_flow * 60
# --- Contrail Climate Impact (P-ATR20) ---
contrail_impact_1 = 4.06e-11    # pressure level era5
contrail_impact_2 = 8.10e-12    # model level era5

df_high['contrail_impact_3'] = df_high['accf_sac_aCCF_Cont'] * df_climate['segment_length_km']
contrail_impact_3 = df_high['contrail_impact_3'].sum() / 0.42
print(f"Contrail Climate Impact 3: {contrail_impact_3:.4e}")

# malaga is a daytime contrail.
def dCont_accf(olr):
    return 1e-10 * (-1.7 -0.0088*olr) * 0.0151

# Apply dCont_accf conditionally
df_high['contrail_impact_4_accf'] = df_high.apply(
    lambda row: dCont_accf(row['accf_sac_olr']) if pd.notna(row['cocip_atr20']) and row['cocip_atr20'] != 0 else 0,
    axis=1
)
# # Apply dCont_accf conditionally
# df_high['contrail_impact_4_accf'] = df_high.apply(
#     lambda row: dCont_accf(row['accf_sac_olr'], row['accf_sac_pcfa']),
#     axis=1
# )

df_high['contrail_impact_4'] = df_high['contrail_impact_4_accf'] * df_climate['segment_length_km']
contrail_impact_4 = df_high['contrail_impact_4'].sum()
print(f"Contrail Climate Impact 4: {contrail_impact_4:.4e}")




def total_impact(df, nox_col, contrail=None, contrail_factor=1.0):
    total_nox = df[nox_col].sum()
    total_h2o = df['accf_h2o'].sum()
    total_co2 = df['accf_co2'].sum()
    contrail_val = contrail * contrail_factor if contrail else 0.0
    total = total_nox + total_h2o + total_co2 + contrail_val
    return {
        'NOx': total_nox,
        'H2O': total_h2o,
        'CO2': total_co2,
        'Contrail': contrail_val,
        'Total': total
    }

# --- Scenarios ---
impact_sets = {
    'No Contrail Impact': {
        'Default NOx': total_impact(df_high, 'accf_nox_default'),
        'Yin NOx': total_impact(df_high, 'accf_nox_yin'),
        'Yin NOx (No Factor)': total_impact(df_high, 'accf_nox_yin_nofactor')
    },
    'With Contrail (4.06e-11)': {
        'Default NOx': total_impact(df_high, 'accf_nox_default', contrail_impact_1, contrail_factor=0.42),
        'Yin NOx': total_impact(df_high, 'accf_nox_yin', contrail_impact_1, contrail_factor=0.42),
        'Yin NOx (No Factor)': total_impact(df_high, 'accf_nox_yin_nofactor', contrail_impact_1, contrail_factor=1.0)
    },
    'With Contrail (8.10e-12)': {
        'Default NOx': total_impact(df_high, 'accf_nox_default', contrail_impact_2, contrail_factor=0.42),
        'Yin NOx': total_impact(df_high, 'accf_nox_yin', contrail_impact_2, contrail_factor=0.42),
        'Yin NOx (No Factor)': total_impact(df_high, 'accf_nox_yin_nofactor', contrail_impact_2, contrail_factor=1.0)
    },
    'With Contrail (Method 3 accf)': {
        'Default NOx': total_impact(df_high, 'accf_nox_default', contrail_impact_3, contrail_factor=0.42),
        'Yin NOx': total_impact(df_high, 'accf_nox_yin', contrail_impact_3, contrail_factor=0.42),
        'Yin NOx (No Factor)': total_impact(df_high, 'accf_nox_yin_nofactor', contrail_impact_3, contrail_factor=1.0)
    },
    'With Contrail (Method 4 accf with cocip issr/sac)': {
        'Default NOx': total_impact(df_high, 'accf_nox_default', contrail_impact_4, contrail_factor=0.42),
        'Yin NOx': total_impact(df_high, 'accf_nox_yin', contrail_impact_4, contrail_factor=0.42),
        'Yin NOx (No Factor)': total_impact(df_high, 'accf_nox_yin_nofactor', contrail_impact_4, contrail_factor=1.0)
    }
}

# --- Print breakdown ---
for title, impacts in impact_sets.items():
    print(f"\n--- {title} ---")
    for method, data in impacts.items():
        print(f"\n{method}:")
        for key in ['NOx', 'H2O', 'CO2', 'Contrail']:
            print(f"  {key}: {100 * data[key] / data['Total']:.2f}%")
        print(f"  TOTAL Climate Impact: {data['Total']:.4e}")

file_path = f'main_results_figures/results/bos_fll/bos_fll_2023-02-06_nighttime/climate/mees/era5model/GTF_SAF_0_A20N_full_WAR_0_climate.csv'
df_night = pd.read_csv(file_path)
print(df_night['cocip_atr20'].sum())
# file_path = f'main_results_figures/results/malaga/malaga/emissions/GTF_SAF_0_A20N_full_WAR_0_0_0.csv'
# df = pd.read_csv(file_path)
# columns_to_drop = [
#             'nox_ei', 'co_ei', 'hc_ei', 'nvpm_ei_m', 'nvpm_ei_n', 'co2', 'h2o',
#             'so2', 'sulphates', 'oc', 'nox', 'co', 'hc', 'nvpm_mass', 'nvpm_number'
#         ]
# df = df.drop(columns=columns_to_drop, errors='ignore')
#
# df = df.rename(columns={
#     'ei_nvpm_number_p3t3_meem': 'nvpm_ei_n',
#     'rhi': 'rhi_emissions',
#     'specific_humidity': 'specific_humidity_emissions'
# })
#
# df['ei_nox'] = df['ei_nox_p3t3'] / 1000
# df['nvpm_ei_m'] = df['ei_nvpm_mass_p3t3_meem'] / 10**6
#
# df = df.drop(columns=['ei_nox_p3t3', 'ei_nvpm_mass_p3t3_meem'], errors='ignore')
#
# """Correct inputs for pycontrails climate impact methods -> compute everything for two engines"""
#
# df['fuel_flow'] = 2*df['fuel_flow_gsp']
#
# df['thrust'] = 2*df['thrust_gsp']
# df['air_pressure'] = df['air_pressure']*10**5
# df['ei_co2'] = df['ei_co2_conservative']
# q_fuel = df['LHV'].iloc[0]*1000
# df['engine_efficiency'] = (df['thrust_gsp']*1000*df['true_airspeed']) / ((df['fuel_flow']/2)*q_fuel)
# fl = Flight(data=df)
# time_bounds = ("2024-06-07 09:00", "2024-06-08 02:00")
# # time_bounds = ("2024-06-07 09:00", "2024-06-07 09:10")
# local_cache_dir_era5m = Path("F:/era5model/malaga")
# variables_model = ("t", "q", "u", "v", "w", "ciwc", "vo", "clwc")
# local_cachestore_era5m = DiskCacheStore(cache_dir=local_cache_dir_era5m)
# pressure_levels_10 = np.arange(150, 400, 10)  # 150 to 400 with steps of 10
# pressure_levels_50 = np.arange(400, 1001, 50)  # 400 to 1000 with steps of 50
#
# # Combine the two arrays
# pressure_levels_model = np.concatenate((pressure_levels_10, pressure_levels_50))
# # Extract min/max longitude and latitude from the dataframe
# west = fl.dataframe["longitude"].min() - 50  # Subtract 1 degree for west buffer
# east = fl.dataframe["longitude"].max() + 50  # Add 1 degree for east buffer
# south = fl.dataframe["latitude"].min() - 50  # Subtract 1 degree for south buffer
# north = fl.dataframe["latitude"].max() + 50  # Add 1 degree for north buffer
#
# # Define the bounding box with altitude range
# bbox = (west, south, 150, east, north, 1000)  # (west, south, min-level, east, north, max-level)
#
# era5ml = ERA5ModelLevel(
#             time=time_bounds,
#             variables=variables_model,
#             # paths=paths,
#             # grid=1,  # horizontal resolution, 0.25 by default
#             model_levels=range(67, 133),
#             pressure_levels=pressure_levels_model,
#             cachestore=local_cachestore_era5m
#         )
# met = era5ml.open_metdataset()
# met = met.downselect(bbox=bbox)
#
# # intersect geopotential (model level)
# # fl.intersect_met(met["geopotential"])
# # print(fl.dataframe["geopotential"])
# df_model = fl.dataframe.copy()
# # Day of year (constant for all rows here)
# N = 159
# S = 1360  # Solar constant in W/m^2
#
# # Convert latitude to radians
# phi = np.deg2rad(df_model['latitude'])
#
# # Solar declination in radians (same for all rows on day N)
# delta = np.deg2rad(-23.44 * np.cos(np.deg2rad(360 / 365 * (N + 10))))
#
# # Compute cos(theta) for all rows
# cos_theta = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta)
#
# # Calculate Fin (top-of-atmosphere solar flux)
# df_model['Fin'] = S * cos_theta
#
# # --- Common contributions ---
# df_model['accf_h2o'] = df_climate['accf_h2o']
# df_model['accf_co2'] = df_climate['accf_co2']
#
# df_model = df_model[df_model['altitude'] > 9160].copy()
# # Extract variables
# t = df_model['air_temperature']
# gp = df_model['geopotential']
# Fin = df_model['Fin']
# fuel_flow = df_model['fuel_flow']
# ei_nox = df_model['ei_nox']
#
#
#
# # --- Method 1: NOx default ---
# accf_o3 = -5.20e-11 + 2.30e-13 * t + 4.85e-16 * gp - 2.04e-18 * t * gp
# accf_ch4 = -9.83e-13 + 1.99e-18 * gp - 6.32e-16 * Fin + 6.12e-21 * gp * Fin
# accf_o3 *= 1.37 / 1.97
# accf_ch4 *= 1.18 / 2.03
# df_model['accf_nox_default'] = (accf_o3 + accf_ch4 + 0.29 * accf_ch4) * ei_nox * fuel_flow * 60
#
# # --- Method 2: NOx yin (with factors) ---
# accf_o3_yin = -2.64e-11 + 1.17e-13 * t + 2.46e-16 * gp - 1.04e-18 * t * gp
# accf_ch4_yin = -4.84e-13 + 9.79e-19 * gp - 3.11e-16 * Fin + 3.01e-21 * gp * Fin
# accf_o3_yin_scaled = accf_o3_yin * 1.37 / 1.97
# accf_ch4_yin_scaled = accf_ch4_yin * 1.18 / 2.03
# df_model['accf_nox_yin'] = (accf_o3_yin_scaled + accf_ch4_yin_scaled + 0.29 * accf_ch4_yin_scaled) * ei_nox * fuel_flow * 60
#
# # --- Method 3: NOx yin (no scaling) ---
# df_model['accf_nox_yin_nofactor'] = (accf_o3_yin + accf_ch4_yin + 0.29 * accf_ch4_yin) * ei_nox * fuel_flow * 60
# # --- Contrail Climate Impact (P-ATR20) ---
# contrail_impact_1 = 4.06e-11
# contrail_impact_2 = 8.10e-12
#
# def total_impact(df, nox_col, contrail=None, contrail_factor=1.0):
#     total_nox = df[nox_col].sum()
#     total_h2o = df['accf_h2o'].sum()
#     total_co2 = df['accf_co2'].sum()
#     contrail_val = contrail * contrail_factor if contrail else 0.0
#     total = total_nox + total_h2o + total_co2 + contrail_val
#     return {
#         'NOx': total_nox,
#         'H2O': total_h2o,
#         'CO2': total_co2,
#         'Contrail': contrail_val,
#         'Total': total
#     }
#
# # --- Scenarios ---
# impact_sets = {
#     'No Contrail Impact (nox model)': {
#         'Default NOx': total_impact(df_model, 'accf_nox_default'),
#         'Yin NOx': total_impact(df_model, 'accf_nox_yin'),
#         'Yin NOx (No Factor)': total_impact(df_model, 'accf_nox_yin_nofactor')
#     },
#     'With Contrail (4.06e-11) (nox model)': {
#         'Default NOx': total_impact(df_model, 'accf_nox_default', contrail_impact_1, contrail_factor=0.42),
#         'Yin NOx': total_impact(df_model, 'accf_nox_yin', contrail_impact_1, contrail_factor=0.42),
#         'Yin NOx (No Factor)': total_impact(df_model, 'accf_nox_yin_nofactor', contrail_impact_1, contrail_factor=1.0)
#     },
#     'With Contrail (8.10e-12) (nox model)': {
#         'Default NOx': total_impact(df_model, 'accf_nox_default', contrail_impact_2, contrail_factor=0.42),
#         'Yin NOx': total_impact(df_model, 'accf_nox_yin', contrail_impact_2, contrail_factor=0.42),
#         'Yin NOx (No Factor)': total_impact(df_model, 'accf_nox_yin_nofactor', contrail_impact_2, contrail_factor=1.0)
#     }
# }
#
# # --- Print breakdown ---
# for title, impacts in impact_sets.items():
#     print(f"\n--- {title} ---")
#     for method, data in impacts.items():
#         print(f"\n{method}:")
#         for key in ['NOx', 'H2O', 'CO2', 'Contrail']:
#             print(f"  {key}: {100 * data[key] / data['Total']:.2f}%")
#         print(f"  TOTAL Climate Impact (nox model): {data['Total']:.4e}")
# fl = Flight(data=df)
# fa_issr = accf_issr.eval(fl)
# df_accf_issr = fa_issr.dataframe.copy()
# plt.figure(figsize=(10, 6))
# # plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_CH4'], label="aCCF CH4")
# # plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_O3'], label="aCCF O3")
# plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_NOx'], label="aCCF NOx climaccf")
# plt.plot(df_climate['index'], df_climate['ei_nox']*df_climate['accf_issr_aCCF_NOx'], label='aCCF NOx EI This Work')
# plt.plot(df_climate['index'], df_climate['accf_issr_aCCF_CO2'], label='aCCF CO2')
# plt.title(f'aCCF along {flight} Flight')
# plt.xlabel('Time in minutes')
# plt.ylabel('Degrees K / kg fuel')
# plt.legend()
# plt.grid(True)
# plt.savefig('results_report/portions/proof/accf_along_flight.png', format='png')
# # plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/climate/{prediction}/{weather_model}/accf_issr/{engine_model}_SAF_{SAF}_nox_accf.png', format='png')
# # plt.close()
# plt.figure(figsize=(10, 6))
# # plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_CH4'], label="aCCF CH4")
# # plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_O3'], label="aCCF O3")
# plt.plot(df_accf_issr['index'], df_accf_issr['aCCF_NOx']*df_accf_issr['fuel_flow_gsp']*2, label="aCCF NOx climaccf")
# plt.plot(df_climate['index'], df_climate['ei_nox']*df_climate['accf_issr_aCCF_NOx']*df_climate['fuel_flow'], label='aCCF NOx EI This Work')
# plt.plot(df_climate['index'], df_climate['accf_issr_aCCF_CO2']*df_climate['fuel_flow'], label='aCCF CO2')
# plt.title(f'Climate Impact along {flight} Flight')
# plt.xlabel('Time in minutes')
# plt.ylabel('Degrees K')
# plt.legend()
# plt.grid(True)
# plt.savefig('results_report/portions/proof/climate_impact_along_flight.png', format='png')
# plt.show()