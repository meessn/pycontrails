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
accf_o3_yin = accf_o3_yin*1.37/1.97
accf_ch4_yin = accf_ch4_yin*1.18/2.03
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

# Filter rows above 9163.8m
df_high = df_climate[df_climate['altitude'] > 9160].copy()

# Extract variables
t = df_high['air_temperature']
gp = df_high['accf_sac_geopotential']
Fin = df_high['accf_sac_Fin']
fuel_flow = df_high['fuel_flow']
ei_nox = df_high['ei_nox']

# --- Common contributions ---
df_high['accf_h2o'] = df_high['accf_sac_aCCF_H2O'] * fuel_flow * 60
df_high['accf_co2'] = df_high['accf_sac_aCCF_CO2'] * fuel_flow * 60

# --- Method 1: NOx default ---
accf_o3 = -5.20e-11 + 2.30e-13 * t + 4.85e-16 * gp - 2.04e-18 * t * gp
accf_ch4 = -9.83e-13 + 1.99e-18 * gp - 6.32e-16 * Fin + 6.12e-21 * gp * Fin
accf_o3 *= 1.37 / 1.97
accf_ch4 *= 1.18 / 2.03
df_high['accf_nox_default'] = (accf_o3 + accf_ch4 + 0.29 * accf_ch4) * ei_nox * fuel_flow * 60

# --- Method 2: NOx yin (with factors) ---
accf_o3_yin = -2.64e-11 + 1.17e-13 * t + 2.46e-16 * gp - 1.04e-18 * t * gp
accf_ch4_yin = -4.84e-13 + 9.79e-19 * gp - 3.11e-16 * Fin + 3.01e-21 * gp * Fin
accf_o3_yin_scaled = accf_o3_yin * 1.37 / 1.97
accf_ch4_yin_scaled = accf_ch4_yin * 1.18 / 2.03
df_high['accf_nox_yin'] = (accf_o3_yin_scaled + accf_ch4_yin_scaled + 0.29 * accf_ch4_yin_scaled) * ei_nox * fuel_flow * 60

# --- Method 3: NOx yin (no scaling) ---
df_high['accf_nox_yin_nofactor'] = (accf_o3_yin + accf_ch4_yin + 0.29 * accf_ch4_yin) * ei_nox * fuel_flow * 60
# --- Contrail Climate Impact (P-ATR20) ---
contrail_impact_1 = 4.06e-11
contrail_impact_2 = 8.10e-12

# --- Compute totals ---
def total_impact(df, nox_col, include_contrail=False, contrail_val=0.0):
    total_nox = df[nox_col].sum()
    total_h2o = df['accf_h2o'].sum()
    total_co2 = df['accf_co2'].sum()
    total = total_nox + total_h2o + total_co2 + (contrail_val if include_contrail else 0.0)
    return {
        'NOx': total_nox,
        'H2O': total_h2o,
        'CO2': total_co2,
        'Contrail': contrail_val if include_contrail else 0.0,
        'Total': total
    }

# --- Calculate all scenarios ---
impact_sets = {
    'No Contrail Impact': {
        'Default NOx': total_impact(df_high, 'accf_nox_default'),
        'Yin NOx': total_impact(df_high, 'accf_nox_yin'),
        'Yin NOx (No Factor)': total_impact(df_high, 'accf_nox_yin_nofactor')
    },
    'With Contrail Impact (4.06e-11)': {
        'Default NOx': total_impact(df_high, 'accf_nox_default', True, contrail_impact_1),
        'Yin NOx': total_impact(df_high, 'accf_nox_yin', True, contrail_impact_1),
        'Yin NOx (No Factor)': total_impact(df_high, 'accf_nox_yin_nofactor', True, contrail_impact_1)
    },
    'With Contrail Impact (8.10e-12)': {
        'Default NOx': total_impact(df_high, 'accf_nox_default', True, contrail_impact_2),
        'Yin NOx': total_impact(df_high, 'accf_nox_yin', True, contrail_impact_2),
        'Yin NOx (No Factor)': total_impact(df_high, 'accf_nox_yin_nofactor', True, contrail_impact_2)
    }
}

# --- Print nicely ---
for title, impacts in impact_sets.items():
    print(f"\n--- {title} ---")
    for method, data in impacts.items():
        print(f"\n{method}:")
        for key in ['NOx', 'H2O', 'CO2', 'Contrail']:
            print(f"  {key}: {100 * data[key] / data['Total']:.2f}%")
        print(f"  TOTAL Impact: {data['Total']:.4e}")



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