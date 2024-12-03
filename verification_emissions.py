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
from emission_index import p3t3_nvpm_meem, p3t3_nvpm_meem_mass
# from piano import altitude_ft_sla
import sys
import pickle
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
engine_model = 'GTF2035'        # GTF , GTF2035
water_injection = [10, 5, 0]     # WAR climb cruise approach/descent
SAF = 0                         # 0, 20, 100 unit = %
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
# fl2 = fl
# for key in met:
# fl['air_temperature'] = fl.intersect_met(met['air_temperature'])
# fl.dataframe['air_temperature'] = fl.dataframe['air_temperature'].interpolate(method='linear', inplace=True)
# if pd.isna(fl.dataframe['air_temperature'].iloc[0]):
#         fl.dataframe['air_temperature'].iloc[0] = fl.dataframe['air_temperature'].iloc[1]
#
# if pd.isna(fl.dataframe['air_temperature'].iloc[-1]):
#         fl.dataframe['air_temperature'].iloc[-1] = fl.dataframe['air_temperature'].iloc[-2]

# fl['specific_humidity'] = fl.intersect_met(met['specific_humidity'])
# fl.dataframe['specific_humidity'] = fl.dataframe['specific_humidity'].interpolate(method='linear', inplace=True)
# if pd.isna(fl.dataframe['specific_humidity'].iloc[0]):
#         fl.dataframe['specific_humidity'].iloc[0] = fl.dataframe['specific_humidity'].iloc[1]
#
# if pd.isna(fl.dataframe['specific_humidity'].iloc[-1]):
#         fl.dataframe['specific_humidity'].iloc[-1] = fl.dataframe['specific_humidity'].iloc[-2]

# perf = PSFlight(met=met)
perf = PSFlight(
    met=met,
    fill_low_altitude_with_isa_temperature=True,  # Estimate temperature using ISA
    fill_low_altitude_with_zero_wind=True
)
fp = perf.eval(fl)
# if pd.isna(fp.dataframe['air_temperature'].iloc[0]):
#     fp.dataframe['air_temperature'].iloc[0] = fp.dataframe['air_temperature'].iloc[1]
# if pd.isna(fp.dataframe['air_temperature'].iloc[-1]):
#     fp.dataframe['air_temperature'].iloc[-1] = fp.dataframe['air_temperature'].iloc[-2]
# fp.dataframe['air_temperature'] = fp.dataframe['air_temperature'].interpolate(method='linear', inplace=True)
# fp.dataframe['specific_humidity'] = fp.dataframe['specific_humidity'].interpolate(method='linear', inplace=True)
# fp2 = perf.eval(fp)
# """----prepare file for pianox-----------------"""
# piano = fp.dataframe.copy()
# piano['mach'] = piano['true_airspeed'] / np.sqrt(constants.kappa*constants.R_d* piano['air_temperature'])
# piano['altitude_ft'] = piano['altitude']*constants.m_to_ft
#
# piano['altitude_ft_sla'] = piano.apply(
#     lambda row: altitude_ft_sla(
#         row['altitude_ft']
#     ),
#     axis=1
# )

"""---------EMISSIONS MODEL FFM2 + ICAO-------------------------------------------------------"""
emissions = Emissions(met=met, humidity_scaling=HistogramMatching())
fe = emissions.eval(fp)

# print("Thrust at index 30:", fp.dataframe['thrust'].iloc[30])
# print("Fuel Flow at index 30:", fe.dataframe['fuel_flow'].iloc[30])
# print("NOx at index 30:", fe.dataframe['nox'].iloc[30])
# print("Specific Humidity at index 30:", fe.dataframe['specific_humidity'].iloc[30])
# print("Altitude at index 30:", fe.dataframe['altitude'].iloc[30])
# print("Air Temperature at index 30:", fe.dataframe['air_temperature'].iloc[30])
# print("NVPM Mass at index 30:", fe.dataframe['nvpm_mass'].iloc[30])
# print("NVPM Number at index 30:", fe.dataframe['nvpm_number'].iloc[30])


# Extract the DataFrame from the Flight object
df = fe.dataframe



"""CREATE FLIGHT PHASE COLUMN"""
# Add a column for altitude change
# Add a column for altitude change
df['altitude_change'] = df['altitude'].diff()

# Define thresholds
climb_threshold = 50       # Minimum altitude change per step for climb
descent_threshold = -50    # Maximum altitude change per step for descent
cruise_min_altitude = 0.95 * df['altitude'].max()  # Minimum altitude for cruise
# Initialize the flight phase column
df['flight_phase'] = 'cruise'

# Classify climb and descent phases based on altitude change and altitude threshold
df.loc[(df['altitude_change'] > climb_threshold), 'flight_phase'] = 'climb'
df.loc[(df['altitude_change'] < descent_threshold), 'flight_phase'] = 'descent'

# Ensure cruise is set correctly for regions above the altitude threshold
df.loc[(df['altitude'] > cruise_min_altitude) &
       (df['flight_phase'] == 'cruise'), 'flight_phase'] = 'cruise'

# Everything else is not cruise: Assign "climb" or "descent" based on neighboring values
for i in range(1, len(df) - 1):  # Avoid the first and last rows
    if df.loc[i, 'altitude'] <= cruise_min_altitude and df.loc[i, 'flight_phase'] == 'cruise':
        # Check neighbors
        if df.loc[i - 1, 'flight_phase'] == 'climb' or df.loc[i + 1, 'flight_phase'] == 'climb':
            df.loc[i, 'flight_phase'] = 'climb'
        elif df.loc[i - 1, 'flight_phase'] == 'descent' or df.loc[i + 1, 'flight_phase'] == 'descent':
            df.loc[i, 'flight_phase'] = 'descent'

# Smooth transitions: Ensure consecutive points with the same slope share the same phase
for i in range(1, len(df)):
    if df.loc[i, 'flight_phase'] != df.loc[i - 1, 'flight_phase']:
        # Ensure previous point aligns with the phase of the interval
        df.loc[i - 1, 'flight_phase'] = df.loc[i, 'flight_phase']
"""plot altitude"""
phase_colors = {
    'climb': 'blue',
    'cruise': 'green',
    'descent': 'red'
}

# Create the plot
plt.figure(figsize=(12, 6))

# Loop through each row and plot segments based on flight phase
for i in range(len(df) - 1):
    # Get the current and next rows
    x_values = [i, i + 1]
    y_values = [df['altitude'].iloc[i], df['altitude'].iloc[i + 1]]

    # Determine the phase (and corresponding color)
    phase = df['flight_phase'].iloc[i]
    color = phase_colors.get(phase, 'black')  # Default to black if phase is missing

    # Plot a line segment for this portion
    plt.plot(x_values, y_values, color=color)

# Add labels, title, and grid
plt.xlabel('Index')
plt.ylabel('Altitude')
plt.title('Altitude vs Index with Flight Phases (Single Line, Colored Sections)')
plt.grid(True)

# Create a legend for the phases
for phase, color in phase_colors.items():
    plt.plot([], [], color=color, label=phase)  # Dummy plot for the legend

plt.legend(title="Flight Phase")
plt.savefig(f'figures/{flight}/flight_phases.png', format='png')
# plt.show()

"""Add config columns"""
# Define a function to map flight phases to WAR values
def assign_war(phase):
    if phase == 'climb':
        return water_injection[0]
    elif phase == 'cruise':
        return water_injection[1]
    elif phase == 'descent':
        return water_injection[2]
    else:
        return None  # Optional: Handle unexpected flight phases

# Apply the function to create the WAR column
df['WAR'] = df['flight_phase'].apply(assign_war)
df['engine_model'] = engine_model
df['SAF'] = SAF

if water_injection[0] != 0 or water_injection[1] != 0 or water_injection[2] != 0:
    df_water = pd.read_csv(f'results/{flight}/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_0_0_0.csv')
    df_water['W3_no_water_injection'] = df_water['W3']
    df['W3_no_water_injection'] = df_water['W3_no_water_injection']
    df['water_injection_kg_s'] = df['W3_no_water_injection'] * df['WAR']/100
else:
    df['water_injection_kg_s'] = 0

# """take selection of points for verification"""
#
# # Filter for the climb phase (positive altitude change)
# climb_point = df[df['altitude_change'] > 0].iloc[0]  # First point during climb
#
# # Filter for cruise phase (near-zero altitude change)
# cruise_points = df[(df['altitude_change'].abs() < 1)].sample(3, random_state=42)  # Select 3 random points during cruise
#
# # Filter for descent phase (negative altitude change)
# descent_point = df[df['altitude_change'] < 0].iloc[-3]  # Third to last point during descent
#
# # Drop auxiliary column
df = df.drop(columns=['altitude_change'])
#
# # Display the updated DataFrame
# print(df)
""" END """
""""AVERAGE CRUISE HEIGHT"""
average_cruise_altitude = df[df['flight_phase'] == 'cruise']['altitude'].mean()
# """"""
#
# # Combine the selected points into a new DataFrame using pd.concat()
# selected_points = pd.concat([pd.DataFrame([climb_point]), cruise_points, pd.DataFrame([descent_point])])
# # selected_points = pd.DataFrame([climb_point])
#
# # Drop the auxiliary column
# selected_points = selected_points.drop(columns=['altitude_change'])
#
# # Display the new DataFrame with the selected 5 points
# print("Selected flight points:")
# print(selected_points)

# columns_to_keep = ['altitude', 'air_temperature', 'air_pressure', 'specific_humidity',  'true_airspeed', 'thrust', 'fuel_flow', 'nox',
                  #  'nvpm_mass', 'nvpm_number', 'flight_phase']

# verify_df = selected_points[columns_to_keep]
# verify_df = df[columns_to_keep].copy()
verify_df = df.copy()
print("New DataFrame with selected columns:")
print(verify_df.head())  # Show the first few rows as an example

kappa = constants.kappa
R_d = constants.R_d
verify_csv_df = verify_df.copy()
verify_csv_df['mach'] = verify_csv_df['true_airspeed'] / np.sqrt(constants.kappa*constants.R_d* verify_csv_df['air_temperature'])
verify_csv_df['air_pressure'] = verify_csv_df['air_pressure'] / 10**5
verify_csv_df['thrust_per_engine'] = verify_csv_df['thrust'] / 2000
verify_csv_df['fuel_flow_per_engine'] = verify_csv_df['fuel_flow'] / 2

verify_csv_df['EI_nox_py'] = verify_csv_df['nox']*1000 / (60*verify_csv_df['fuel_flow'])
verify_csv_df['EI_nvpm_mass_py'] = verify_csv_df['nvpm_mass']*1e6 / (60*verify_csv_df['fuel_flow'])
verify_csv_df['EI_nvpm_number_py'] = verify_csv_df['nvpm_number'] / (60*verify_csv_df['fuel_flow'])

"""DELETE NAN ROWS!!!!!!!!!!!!!!!!!!!!!!!!!!!1"""
print('deleted rows:', verify_csv_df[verify_csv_df.isna().any(axis=1)].shape[0])
verify_csv_df = verify_csv_df.dropna()

verify_csv_df.to_csv('verify_df.csv', sep=';', decimal=',', index=False)
verify_csv_df.to_csv('input.csv', index=True, index_label='index')
python32_path = r"C:\Users\Mees Snoek\AppData\Local\Programs\Python\Python39-32\python.exe"
# Paths for input and output CSV files
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
input_csv_path = os.path.join(current_directory, "input.csv")
output_csv_path = os.path.join(current_directory, "output.csv")

try:
    # Run the subprocess
    subprocess.run(
        [python32_path, 'gsp_api.py', input_csv_path, output_csv_path],
        check=True  # Raises an error if the subprocess fails
    )
except subprocess.CalledProcessError as e:
    print(f"Subprocess failed with error: {e}")
    print(f"Subprocess output: {e.output if hasattr(e, 'output') else 'No output available'}")
    print(f"Subprocess stderr: {e.stderr if hasattr(e, 'stderr') else 'No stderr available'}")


# Read the results back into the main DataFrame
results_df = pd.read_csv(output_csv_path)

# Merge the results back into the original DataFrame
df_gsp = pd.read_csv(input_csv_path)  # Load the original DataFrame
df_gsp = df_gsp.merge(results_df, on='index', how='left')

print(df_gsp)
df_gsp['WAR_gsp'] = (df_gsp['water_injection_kg_s'] / df_gsp['W3'])*100

"""NOx p3t3"""
df_gsp['EI_nox_p3t3'] = df_gsp.apply(
    lambda row: p3t3_nox(
        row['PT3'],
        row['TT3'],
        interp_func_far,
        interp_func_pt3,
        row['specific_humidity'],
        row['WAR_gsp']
    ),
    axis=1
)

"""NOx De Boer"""
df_gsp['EI_nox_boer'] = df_gsp.apply(
    lambda row: NOx_correlation_de_boer(
        row['PT3'],
        row['TT3'],
        row['TT4'],
        row['WAR_gsp']
    ),
    axis=1
)

"""NOx Kaiser"""
df_gsp['EI_nox_kaiser'] = df_gsp.apply(
    lambda row: NOx_correlation_kaiser_optimized_tf(
        row['PT3'],
        row['TT3'],
        row['WAR_gsp']
    ),
    axis=1
)

"""NOx kypriandis optimized"""
df_gsp['EI_nox_kypriandis'] = df_gsp.apply(
    lambda row: NOx_correlation_kypriandis_optimized_tf(
        row['PT3'],
        row['TT3'],
        row['TT4'],
        row['WAR_gsp']
    ),
    axis=1
)
#
df_gsp['EI_nvpm_number_p3t3'] = df_gsp.apply(
    lambda row: p3t3_nvpm(
        row['PT3'],
        row['TT3'],
        row['FAR'],
        interp_func_far,
        interp_func_pt3,
        row['SAF']
    ),
    axis=1
)
#
df_gsp['EI_nvpm_mass_p3t3'] = df_gsp.apply(
    lambda row: p3t3_nvpm_mass(
        row['PT3'],
        row['TT3'],
        row['FAR'],
        interp_func_far,
        interp_func_pt3,
        row['SAF']
    ),
    axis=1
)
"""P3T3 _MEEM"""
df_gsp['EI_nvpm_number_p3t3_meem'] = df_gsp.apply(
    lambda row: p3t3_nvpm_meem(
        row['PT3'],
        row['TT3'],
        row['FAR'],
        interp_func_far,
        interp_func_pt3,
        row['SAF']
    ),
    axis=1
)

df_gsp['EI_nvpm_mass_p3t3_meem'] = df_gsp.apply(
    lambda row: p3t3_nvpm_meem_mass(
        row['PT3'],
        row['TT3'],
        row['FAR'],
        interp_func_far,
        interp_func_pt3,
        row['SAF']
    ),
    axis=1
)


"""MEEM"""
print("average cruise altitude", average_cruise_altitude)
df_gsp[['EI_mass_meem', 'EI_number_meem']] = df_gsp.apply(
    lambda row: pd.Series(meem_nvpm(
        row['altitude'],
        row['mach'],
        average_cruise_altitude,
        row['flight_phase'],
        row['SAF']
    )),
    axis=1
)
print(df_gsp[['EI_nvpm_mass_p3t3', 'EI_nvpm_number_p3t3', 'EI_mass_meem', 'EI_number_meem']])
#
# df_gsp.to_csv('verify_df_p3t3_api.csv', sep=';', decimal=',', index=False)

# Plot A: EI_NOx
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['EI_nox_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nox_p3t3'], label='P3T3', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nox_boer'], label='Boer', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nox_kaiser'], label='Kaiser', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nox_kypriandis'], label='Kypriandis', linestyle='-', marker='o', markersize=2.5)
plt.title('EI_NOx')
plt.xlabel('Time in minutes')
plt.ylabel('EI_NOx (g/ kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/ei_nox.png', format='png')

# Plot A: EI_NOx
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['EI_nox_py'], label='Pycontrails', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_nox_p3t3'], label='P3T3', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_nox_boer'], label='Boer', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_nox_kaiser'], label='Kaiser', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_nox_kypriandis'], label='Kypriandis', linestyle='-')
plt.title('EI_NOx')
plt.xlabel('Time in minutes')
plt.ylabel('EI_NOx (g/ kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/ei_nox_no_markers.png', format='png')


# Plot B: EI_nvpm_mass
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['EI_nvpm_mass_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nvpm_mass_p3t3'], label='P3T3', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_mass_meem'], label='MEEM', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nvpm_mass_p3t3_meem'], label='P3T3 - MEEM', linestyle='-', marker='o', markersize=2.5)
plt.title('EI_nvpm_mass')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvpm_mass (mg / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/ei_nvpm_mass.png', format='png')

# Plot B: EI_nvpm_mass
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['EI_nvpm_mass_py'], label='Pycontrails', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_nvpm_mass_p3t3'], label='P3T3', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_mass_meem'], label='MEEM', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_nvpm_mass_p3t3_meem'], label='P3T3 - MEEM', linestyle='-')
plt.title('EI_nvpm_mass')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvpm_mass (mg / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/ei_nvpm_mass_no_markers.png', format='png')

# Plot C: EI_nvpm_number
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_p3t3'], label='P3T3', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_number_meem'], label='MEEM', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_p3t3_meem'], label='P3T3 - MEEM', linestyle='-', marker='o', markersize=2.5)
plt.title('EI_nvpm_number')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvpm_number (# / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/ei_nvpm_number.png', format='png')

plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_py'], label='Pycontrails', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_p3t3'], label='P3T3', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_number_meem'], label='MEEM', linestyle='-')
plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_p3t3_meem'], label='P3T3 - MEEM', linestyle='-')
plt.title('EI_nvpm_number')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvpm_number (# / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/ei_nvpm_number_no_markers.png', format='png')

# # Plot D: EI_nvpm_mass
# plt.figure(figsize=(10, 6))
# # plt.plot(df_gsp.index, df_gsp['EI_nvpm_mass_py'], label='Pycontrails', linestyle='-', marker='o')
# plt.plot(df_gsp.index, df_gsp['EI_nvpm_mass_p3t3'], label='P3T3', linestyle='-', marker='x')
# plt.plot(df_gsp.index, df_gsp['EI_mass_meem'], label='MEEM', linestyle='-', marker='s')
# plt.title('Plot D: EI_nvpm_mass')
# plt.xlabel('Index')
# plt.ylabel('EI_nvpm_mass')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'figures/{flight}/ei_nvpm_mass_p3t3_meem.png', format='png')
#
# # Plot E: EI_nvpm_number
# plt.figure(figsize=(10, 6))
# # plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_py'], label='Pycontrails', linestyle='-', marker='o')
# plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_p3t3'], label='P3T3', linestyle='-', marker='x')
# plt.plot(df_gsp.index, df_gsp['EI_number_meem'], label='MEEM', linestyle='-', marker='s')
# plt.title('Plot E: EI_nvpm_number')
# plt.xlabel('Index')
# plt.ylabel('EI_nvpm_number')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'figures/{flight}/ei_nvpm_number_p3t3_meem.png', format='png')

# Plot E: fuel flow
#openap

# if engine_model == 'GTF' and water_injection[0] == 0 and water_injection[1] == 0 and water_injection[2] == 0 and SAF ==0 and flight == 'malaga' and aircraft == 'A20N_full':
#     fuelflow = FuelFlow(ac='a320')
#
#     df_gsp['fuel_flow_openap'] = df_gsp.apply(
#         lambda row: fuelflow.at_thrust(
#             row['thrust'],
#             row['altitude']*constants.m_to_ft
#         ),
#         axis=1
#     )


plt.figure(figsize=(10, 6))
# plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_py'], label='Pycontrails', linestyle='-', marker='o')
plt.plot(df_gsp.index, df_gsp['fuel_flow_per_engine'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['fuel_flow_gsp'], label='GSP', linestyle='-', marker='o', markersize=2.5)
try:
    if data_gasturb:
        plt.scatter(df_gasturb.index, df_gasturb['fuel_flow_gasturb'], label='GasTurb', marker='o', s=25, color='green')
        # plt.plot(df_gsp.index, df_gsp['fuel_flow_openap']/2, label='OpenAP', linestyle='-', marker='o', markersize=2.5)
except NameError:
    print("Variable does not exist, skipping.")
plt.title('Fuel Flow')
plt.xlabel('Time in minutes')
plt.ylabel('Fuel Flow (kg/s)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/fuel_flow.png', format='png')


plt.figure(figsize=(10, 6))
# plt.plot(df_gsp.index, df_gsp['EI_nvpm_number_py'], label='Pycontrails', linestyle='-', marker='o')
plt.plot(df_gsp.index, df_gsp['thrust_per_engine'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['thrust_gsp'], label='GSP', linestyle='-', marker='o', markersize=2.5)
plt.title('Thrust')
plt.xlabel('Time in minutes')
plt.ylabel('Thrust (kN)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/thrust.png', format='png')

# Convert the water_injection values to strings, replacing '.' with '_'
formatted_values = [str(value).replace('.', '_') for value in water_injection]

df_gsp.to_csv(f'results/{flight}/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv')