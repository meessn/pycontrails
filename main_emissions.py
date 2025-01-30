import numpy as np
import os
import pandas as pd
import subprocess
import constants
from matplotlib import pyplot as plt
from emission_index import p3t3_nox
from emission_index import p3t3_nvpm_meem, p3t3_nvpm_meem_mass, thrust_setting
import pickle
from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import HistogramMatching, ExponentialBoostHumidityScaling
from pycontrails.models.ps_model import PSFlight
from pycontrails.models.emissions import Emissions
from pycontrails.datalib import ecmwf



"""FLIGHT PARAMETERS"""
engine_model = 'GTF'        # GTF , GTF2035, GTF1990, GTF2000
water_injection = [0, 0, 0]     # WAR climb cruise approach/descent
SAF = 0                   # 0, 20, 100 unit = %
flight = 'malaga'
aircraft = 'A20N_full'        # A20N ps model, A20N_wf is change in Thrust and t/o and idle fuel flows
                            # A20N_wf_opr is with changed nominal opr and bpr
                            # A20N_full has also the eta 1 and 2 and psi_0

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

if engine_model == 'GTF' or engine_model == 'GTF2035':
    engine_uid = '01P22PW163'
elif engine_model == 'GTF1990':
    engine_uid = '1CM009'
elif engine_model == 'GTF2000':
    engine_uid = '3CM026'
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}. ")

attrs = {
    "flight_id" : "34610D",
    "aircraft_type": f"{aircraft}",
    "engine_uid": f"{engine_uid}"
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

"""---------EMISSIONS MODEL FFM2 + ICAO-------------------------------------------------------"""
emissions = Emissions(met=met, humidity_scaling=ExponentialBoostHumidityScaling(rhi_adj=0.9779, rhi_boost_exponent=1.635,
                                                                        clip_upper=1.65))
fe = emissions.eval(fp)

# Extract the DataFrame from the Flight object
df = fe.dataframe

"""CREATE FLIGHT PHASE COLUMN"""

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
plt.xlabel('Flight Time in Minutes')
plt.ylabel('Altitude')
plt.title(f'Altitude Profile of {flight} to Amsterdam Flight')
plt.grid(True)

# Create a legend for the phases
for phase, color in phase_colors.items():
    plt.plot([], [], color=color, label=phase)  # Dummy plot for the legend

plt.legend(title="Flight Phase")
plt.savefig(f'main_results_figures/figures/{flight}/emissions/flight_phases.png', format='png')
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

if SAF == 0:
    LHV = 43031 #kJ/kg
    ei_h2o = 1.237
    ei_co2_conservative = 3.16
    ei_co2_optimistic = 3.16
elif SAF == 20:
    LHV = ((43031*1000) + 10700*SAF)/1000
    ei_h2o = 1.237*(14.1/13.8)
    ei_co2_conservative = 3.16*0.9*0.2 + 0.8*3.16
    ei_co2_optimistic = 3.16*0.06*0.2 + 0.8*3.16
elif SAF == 100:
    LHV = ((43031*1000) + 10700*SAF)/1000
    ei_h2o = 1.237 * (15.3/13.8)
    ei_co2_conservative = 3.16*0.9
    ei_co2_optimistic = 3.16*0.06
else:
    print('error: not a correct saf value')

df['LHV'] = LHV
df['ei_h2o'] = ei_h2o
df['ei_co2_conservative'] = ei_co2_conservative
df['ei_co2_optimistic'] = ei_co2_optimistic

if water_injection[0] != 0 or water_injection[1] != 0 or water_injection[2] != 0:
    df_water = pd.read_csv(f'main_results_figures/results/{flight}/emissions/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_0_0_0.csv')
    df_water['W3_no_water_injection'] = df_water['W3_no_specific_humid']
    df['W3_no_water_injection'] = df_water['W3_no_water_injection']
    df['water_injection_kg_s'] = df['W3_no_water_injection'] * (df['WAR']/100 - df['specific_humidity'])
    df['water_injection_kg_s'] = df['water_injection_kg_s'].clip(lower=0) #no negative water injection if 0 WAR is present
else:
    df['water_injection_kg_s'] = 0

# # Drop auxiliary column
df = df.drop(columns=['altitude_change'])
#
""" END """
""""AVERAGE CRUISE HEIGHT"""
average_cruise_altitude = df[df['flight_phase'] == 'cruise']['altitude'].mean()


kappa = constants.kappa
R_d = constants.R_d

df['mach'] = df['true_airspeed'] / np.sqrt(constants.kappa*constants.R_d* df['air_temperature'])
df['air_pressure'] = df['air_pressure'] / 10**5
df['thrust_per_engine'] = df['thrust'] / 2000
df['fuel_flow_per_engine'] = df['fuel_flow'] / 2

df['ei_nox_py'] = df['nox']*1000 / (60*df['fuel_flow'])
df['ei_nvpm_mass_py'] = df['nvpm_mass']*1e6 / (60*df['fuel_flow'])
df['ei_nvpm_number_py'] = df['nvpm_number'] / (60*df['fuel_flow'])

"""DELETE NAN ROWS!!!!!!!!!!!!!!!!!!!!!!!!!!!1"""
try:
    # Identify rows with NaN values
    nan_rows = df[df.isna().any(axis=1)].index
    deleted_rows_count = len(nan_rows)

    # Check if any NaN row is not the first or last row
    if any((row_index > 0) & (row_index < len(df) - 1) for row_index in nan_rows):
        raise ValueError("NaN detected in a non-edge row. Proceeding with deletion, but this may affect results.")

    # Print the number of rows being deleted
    print("Deleted rows:", deleted_rows_count)

    # Drop NaN rows
    df = df.dropna()

except ValueError as e:
    print(f"Warning: {e}")

df.to_csv('input.csv', index=True, index_label='index')
python32_path = r"C:\Users\Mees Snoek\AppData\Local\Programs\Python\Python39-32\python.exe"
# Paths for input and output CSV files
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
input_csv_path = os.path.join(current_directory, "input.csv")
output_csv_path = os.path.join(current_directory, "output.csv")

if engine_model != 'GTF2000': #avoid computing GTF2000 again, as GSP model is the same as GTF1990
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
elif engine_model == 'GTF2000':
    # GTF2000 always run after 1990, so output.csv can be used. However do check if the column values correspond and error if not
    formatted_values = [str(value).replace('.', '_') for value in water_injection]
    gtf1990_file_path = (
        f"main_results_figures/results/{flight}/emissions/"
        f"{flight}_model_GTF1990_SAF_{SAF}_aircraft_{aircraft}_WAR_"
        f"{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv"
    )

    output_df = pd.read_csv(output_csv_path)
    gtf1990_df = pd.read_csv(gtf1990_file_path)

    # Check if all columns in output.csv exist in GTF1990 file
    missing_columns = [col for col in output_df.columns if col not in gtf1990_df.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing in the GTF1990 file: {missing_columns}")

    # Subset GTF1990 to only the columns in output.csv
    gtf1990_df_subset = gtf1990_df[output_df.columns]

    # Compare dataframes row-wise
    mismatched_rows = (output_df != gtf1990_df_subset).any(axis=1)

    if mismatched_rows.any():
        # Log mismatches for debugging
        mismatches = output_df[mismatched_rows].compare(gtf1990_df_subset[mismatched_rows])
        print("Found mismatched data:")
        print(mismatches)
        raise ValueError("Mismatch detected between output.csv and GTF1990 file.")
    else:
        print("Validation successful: All columns and row values match.")
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}. ")

# Read the results back into the main DataFrame
results_df = pd.read_csv(output_csv_path)

# Merge the results back into the original DataFrame
df_gsp = pd.read_csv(input_csv_path)  # Load the original DataFrame
df_gsp = df_gsp.merge(results_df, on='index', how='left')

df_gsp['W3_no_specific_humid'] = df_gsp['W3'] / (1+df_gsp['specific_humidity']) #pure air, without water from ambience

df_gsp['WAR_gsp'] = ((df_gsp['water_injection_kg_s'] + df_gsp['specific_humidity']*df_gsp['W3_no_specific_humid']) / df_gsp['W3_no_specific_humid'])*100 #%

if engine_model == 'GTF' or engine_model == 'GTF2035':
    with open('p3t3_graphs_sls.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)

    interp_func_far = loaded_functions['interp_func_far']
    interp_func_pt3 = loaded_functions['interp_func_pt3']
elif engine_model == 'GTF1990' or engine_model == 'GTF2000':
    with open('p3t3_graphs_sls_1990_2000.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)

    interp_func_far = loaded_functions['interp_func_far']
    interp_func_pt3 = loaded_functions['interp_func_pt3']
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}. ")

df_gsp['thrust_setting_meem'] = df_gsp.apply(
    lambda row: thrust_setting(
        engine_model,
        row['TT3'],
        interp_func_pt3
    ),
    axis=1
)

"""NOx p3t3"""
df_gsp['ei_nox_p3t3'] = df_gsp.apply(
    lambda row: p3t3_nox(
        row['PT3'],
        row['TT3'],
        interp_func_far,
        interp_func_pt3,
        row['specific_humidity'],
        row['WAR_gsp'],
        engine_model
    ),
    axis=1
)
#

"""P3T3 _MEEM"""
df_gsp['ei_nvpm_number_p3t3_meem'] = df_gsp.apply(
    lambda row: p3t3_nvpm_meem(
        row['PT3'],
        row['TT3'],
        row['FAR'],
        interp_func_far,
        interp_func_pt3,
        row['SAF'],
        row['thrust_setting_meem'],
        engine_model
    ),
    axis=1
)

df_gsp['ei_nvpm_mass_p3t3_meem'] = df_gsp.apply(
    lambda row: p3t3_nvpm_meem_mass(
        row['PT3'],
        row['TT3'],
        row['FAR'],
        interp_func_far,
        interp_func_pt3,
        row['SAF'],
        row['thrust_setting_meem'],
        engine_model
    ),
    axis=1
)

# Plot A: EI_NOx
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nox_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
if engine_model != 'GTF1990':
    plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3'], label='P3T3', linestyle='-', marker='o', markersize=2.5)
plt.title('EI_NOx')
plt.xlabel('Time in minutes')
plt.ylabel('EI_NOx (g/ kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'main_results_figures/figures/{flight}/emissions/ei_nox.png', format='png')

# Plot A: EI_NOx
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nox_py'], label='Pycontrails', linestyle='-')
plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3'], label='P3T3', linestyle='-')
plt.title('EI_NOx')
plt.xlabel('Time in minutes')
plt.ylabel('EI_NOx (g/ kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'main_results_figures/figures/{flight}/emissions/ei_nox_no_markers.png', format='png')

# Plot A: EI_NOx
plt.figure(figsize=(10, 6))
plt.plot(df_gsp['TT3'], df_gsp['ei_nox_py'], label='Pycontrails', linestyle='-')
plt.plot(df_gsp['TT3'], df_gsp['ei_nox_p3t3'], label='P3T3', linestyle='-')
plt.title('EI_NOx')
plt.xlabel('TT3')
plt.ylabel('EI_NOx (g/ kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'main_results_figures/figures/{flight}/emissions/ei_nox_tt3.png', format='png')
# plt.show()

# Plot B: EI_nvpm_mass
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_p3t3_meem'], label='P3T3 - MEEM', linestyle='-', marker='o', markersize=2.5)
plt.title('EI_nvPM_mass')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvPM_mass (mg / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'main_results_figures/figures/{flight}/emissions/ei_nvpm_mass.png', format='png')

# Plot B: EI_nvpm_mass
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_py'], label='Pycontrails', linestyle='-')
plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_p3t3_meem'], label='P3T3 - MEEM', linestyle='-')
plt.title('EI_nvPM_mass')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvPM_mass (mg / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'main_results_figures/figures/{flight}/emissions/ei_nvpm_mass_no_markers.png', format='png')

# Plot C: EI_nvpm_number
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3_meem'], label='P3T3 - MEEM', linestyle='-', marker='o', markersize=2.5)
plt.title('EI_nvPM_number')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvPM_number (# / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'main_results_figures/figures/{flight}/emissions/ei_nvpm_number.png', format='png')

plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_py'], label='Pycontrails', linestyle='-')
plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3_meem'], label='P3T3 - MEEM', linestyle='-')
plt.title('EI_nvPM_number')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvPM_number (# / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'main_results_figures/figures/{flight}/emissions/ei_nvpm_number_no_markers.png', format='png')


plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['fuel_flow_per_engine'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['fuel_flow_gsp'], label='GSP', linestyle='-', marker='o', markersize=2.5)
plt.title('Fuel Flow')
plt.xlabel('Time in minutes')
plt.ylabel('Fuel Flow (kg/s)')
plt.legend()
plt.grid(True)
plt.savefig(f'main_results_figures/figures/{flight}/emissions/fuel_flow.png', format='png')


plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['thrust_per_engine'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['thrust_gsp'], label='GSP', linestyle='-', marker='o', markersize=2.5)
plt.title('Thrust')
plt.xlabel('Time in minutes')
plt.ylabel('Thrust (kN)')
plt.legend()
plt.grid(True)
plt.savefig(f'main_results_figures/figures/{flight}/emissions/thrust.png', format='png')

# Convert the water_injection values to strings, replacing '.' with '_'
formatted_values = [str(value).replace('.', '_') for value in water_injection]

df_gsp.to_csv(f'main_results_figures/results/{flight}/emissions/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv')