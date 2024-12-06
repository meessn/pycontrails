"""
take flight

-pick 3 climb points

-pick 3 cruise points

-pick 3 descent points

0 - 20% WAR per 0.5% steps

per point: run GSP model (2035)

Retrieve fuel flow and other parameters

calculate P3T3 nox

make a plot for each point with title {stage_of_flight} point {number} EI_nox y axis and fuel flow x axis -> WAR colour

"""

import numpy as np
import os
import pandas as pd
import subprocess
import constants
from matplotlib import pyplot as plt
import pickle
from pycontrails import Flight, MetDataset
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import HistogramMatching
from pycontrails.models.ps_model import PSFlight
from pycontrails.models.emissions import Emissions
from pycontrails.datalib import ecmwf

from emission_index import p3t3_nox_wi, p3t3_nvpm_meem
with open('p3t3_graphs_sls.pkl', 'rb') as f:
    loaded_functions = pickle.load(f)

interp_func_far = loaded_functions['interp_func_far']
interp_func_pt3 = loaded_functions['interp_func_pt3']


"""FLIGHT PARAMETERS"""
engine_model = 'GTF2035_wi_gass'        # GTF , GTF2035
water_injection = [0, 0, 0]     # WAR climb cruise approach/descent
SAF = 0                         # 0, 20, 100 unit = %
flight = 'malaga'
aircraft = 'A20N_full'        # ps model, A20N_wf is change in Thrust and t/o and idle fuel flows
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

perf = PSFlight(
    met=met,
    fill_low_altitude_with_isa_temperature=True,  # Estimate temperature using ISA
    fill_low_altitude_with_zero_wind=True
)
fp = perf.eval(fl)


"""---------EMISSIONS MODEL FFM2 + ICAO-------------------------------------------------------"""
emissions = Emissions(met=met, humidity_scaling=HistogramMatching())
fe = emissions.eval(fp)



# Extract the DataFrame from the Flight object
df = fe.dataframe

# Identify climb, cruise, and descent phases based on altitude changes
df['altitude_change'] = df['altitude'].diff()



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

df_water = pd.read_csv(f'results/{flight}/{flight}_model_GTF2035_SAF_{SAF}_aircraft_{aircraft}_WAR_0_0_0.csv')
df_water['W3_no_water_injection'] = df_water['W3_no_specific_humid']
df['W3_no_water_injection'] = df_water['W3_no_water_injection']

df_tsfc_2020 = pd.read_csv(f'results/{flight}/{flight}_model_GTF_SAF_{SAF}_aircraft_{aircraft}_WAR_0_0_0.csv')
df_tsfc_2020['tsfc_2020'] = (df_tsfc_2020['fuel_flow_gsp']*1000) / df_tsfc_2020['thrust_gsp']
df_tsfc_2020['tsfc_2035_max'] = df_tsfc_2020['tsfc_2020'] * 0.87
df['tsfc_2020'] = df_tsfc_2020['tsfc_2020']
df['tsfc_2035_max'] = df_tsfc_2020['tsfc_2035_max']

df_tsfc_2020['fuel_2020'] = df_tsfc_2020['fuel_flow_gsp']*2
df_tsfc_2020['fuel_2035_max'] = df_tsfc_2020['fuel_2020'] * 0.87
df['fuel_2020'] = df_tsfc_2020['fuel_2020']
df['fuel_2035_max'] = df_tsfc_2020['fuel_2035_max']
# # Drop auxiliary column
df = df.drop(columns=['altitude_change'])

""" END """
""""AVERAGE CRUISE HEIGHT"""
average_cruise_altitude = df[df['flight_phase'] == 'cruise']['altitude'].mean()

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


"""here we pick the points (9 points in total), 
create new df for each point with same values in each row, but different WAR%"""
# Constants




WAR_VALUES = np.arange(0, 25.5, 0.5)  # WAR values from 0.00 to 0.20
NUM_POINTS_PER_PHASE = 3  # Number of points to select per flight phase

verify_csv_df['original_index'] = verify_csv_df.index

points = []
# for phase in ['climb', 'cruise', 'descent']:
#     phase_points = verify_csv_df[verify_csv_df['flight_phase'] == phase].sample(NUM_POINTS_PER_PHASE, random_state=40)
#     points.append(phase_points)

# Manually specify indices for each phase
manual_indices = {
    'climb': [5, 12, 24],  # Example indices for climb phase
    'cruise': [33, 65, 100],  # Example indices for cruise phase
    'descent': [112, 116, 131]  # Example indices for descent phase
}

# Loop through the phases and select points based on manual indices
for phase, indices in manual_indices.items():
    phase_points = verify_csv_df.loc[indices]
    points.append(phase_points)

# Combine all selected points into a single DataFrame
print('points', points)
selected_points = pd.concat(points)


python32_path = r"C:\Users\Mees Snoek\AppData\Local\Programs\Python\Python39-32\python.exe"
# Paths for input and output CSV files
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)


# Directory for saving outputs
output_dir = os.path.join(current_directory, "water_injection_optimized")
os.makedirs(output_dir, exist_ok=True)

manual_points_indices = selected_points['original_index'].values
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

# Highlight the manual points
for idx in manual_points_indices:
    plt.scatter(idx, df['altitude'].iloc[idx], color='orange', zorder=5)

# Add labels, title, and grid
plt.xlabel('Index')
plt.ylabel('Altitude')
plt.title('Altitude vs Index with Flight Phases (Single Line, Colored Sections)')
plt.grid(True)

# Create a legend for the phases
for phase, color in phase_colors.items():
    plt.plot([], [], color=color, label=phase)  # Dummy plot for the legend

plt.legend(title="Flight Phase")
plot_path = os.path.join(output_dir, f"flight_phases_chosen_points.png")
plt.savefig(plot_path, format='png')


# df['water_injection_kg_s'] = df['W3_no_water_injection'] * df['WAR']/100

print(verify_csv_df.columns)
# Loop over each selected point
for i, (_, point_row) in enumerate(selected_points.iterrows()):
    # Create a new DataFrame for this point with varying WAR values
    point_df = pd.DataFrame([point_row.to_dict()] * len(WAR_VALUES))
    point_df['WAR'] = WAR_VALUES  # Add the WAR column
    point_df['water_injection_kg_s'] = point_df['W3_no_water_injection'] * (point_df['WAR'] / 100 - point_df['specific_humidity'])
    point_df['water_injection_kg_s'] = point_df['water_injection_kg_s'].clip(lower=0)
    # Reset the index and ensure 'index' column exists
    point_df.reset_index(drop=False, inplace=True)  # Add a unique 'index' column
    # # Ensure the original index from df is preserved
    # point_df['original_index'] = point_row['original_index']  # Add original index from df

    # Save the point DataFrame to a temporary input file
    point_input_path = os.path.join(output_dir, f"point_{i}_input.csv")
    point_output_path = os.path.join(output_dir, f"point_{i}_output.csv")
    # point_df.reset_index(drop=True).to_csv(point_input_path, index=False)
    # Reset the index and save the DataFrame with the index labeled as 'index'
    point_df.to_csv(point_input_path, index=False)
    # Execute the subprocess for this point
    try:
        subprocess.run(
            [python32_path, 'gsp_api.py', point_input_path, point_output_path],
            check=True  # Raises an error if the subprocess fails
        )
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with error: {e}")
        print(f"Subprocess output: {e.output if hasattr(e, 'output') else 'No output available'}")
        print(f"Subprocess stderr: {e.stderr if hasattr(e, 'stderr') else 'No stderr available'}")

    # Read the results back and merge into the main DataFrame
    # print('test')

    results_df = pd.read_csv(point_output_path)
    point_results_df = pd.read_csv(point_input_path)

    point_results_df = point_results_df.merge(results_df, on='index', how='left')
    point_results_df['W3_no_specific_humid'] = point_results_df['W3'] / (1 + point_results_df['specific_humidity'])

    point_results_df['WAR_gsp'] = ((point_results_df['water_injection_kg_s'] + point_results_df['specific_humidity'] * point_results_df['W3_no_specific_humid']) / point_results_df[
        'W3_no_specific_humid']) * 100

    point_results_df['EI_nox_p3t3_wi'] = point_results_df.apply(
        lambda row: p3t3_nox_wi(
            row['PT3'],
            row['TT3'],
            interp_func_far,
            interp_func_pt3,
            row['WAR_gsp']
        ),
        axis=1
    )

    point_results_df['EI_nvpm_number_p3t3_meem'] = point_results_df.apply(
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

    # Save the combined results for this point
    combined_output_path = os.path.join(output_dir, f"{engine_model}_point_{i}_combined.csv")
    point_results_df.to_csv(combined_output_path, index=False)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot EI_nox_p3t3_wi and TSFC on the primary y-axis
    line1, = ax1.plot(
        point_results_df['WAR'],
        point_results_df['EI_nox_p3t3_wi'],
        label='EI_NOx',
        linestyle='-',
        marker='o'
    )
    line2, = ax1.plot(
        point_results_df['WAR'],
        (point_results_df['fuel_flow_gsp'] * 1000) / point_results_df['thrust_gsp'],
        label='TSFC (g/kNs)',
        linestyle='--',
        marker='o'
    )

    # Add horizontal lines for TSFC state-of-the-art and 2035 maximum
    line3 = ax1.axhline(
        y=point_results_df['tsfc_2020'].iloc[0],
        color='blue',
        linestyle='--',
        linewidth=2,
        label='TSFC 2020 (State-of-the-Art)'
    )
    line4 = ax1.axhline(
        y=point_results_df['tsfc_2035_max'].iloc[0],
        color='red',
        linestyle=':',
        linewidth=2,
        label='TSFC 2035 Max'
    )

    # Customize primary y-axis
    ax1.set_xlabel("WAR in combustor [%]")
    ax1.set_ylabel("EI_nox (g/kg fuel) & TSFC (g/kNs)")
    ax1.grid(True)

    # Create secondary y-axis
    ax2 = ax1.twinx()

    # Plot EI_nvpm_number_p3t3_meem on the secondary y-axis
    line5, = ax2.plot(
        point_results_df['WAR'],
        point_results_df['EI_nvpm_number_p3t3_meem'],
        label='EI_nvPM_number',
        color='purple',
        linestyle='-.',
        marker='o'
    )

    # Customize secondary y-axis
    ax2.set_ylabel("EI_nvPM_number (#)")

    # Combine legends from both axes
    lines = [line1, line2, line3, line4, line5]
    labels = [l.get_label() for l in lines]
    fig.legend(
        lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=True
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    # Title and layout adjustments
    plt.title(f"WAR vs EI_nox_p3t3_wi, TSFC, and EI_nvpm_number - {point_row['flight_phase']}")

    plot_path = os.path.join(output_dir, f"{engine_model}_point_{i}_plot_nvpm_nox_tsfc_war.png")
    plt.savefig(plot_path, format='png')
    plt.close()

    # plot with TSFC, NOx and nvpm_number vs WAR
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot EI_nox_p3t3_wi and fuel flow on the primary y-axis
    ax1.plot(
        point_results_df['WAR'],
        point_results_df['EI_nox_p3t3_wi'],
        label='EI_NOx',
        linestyle='-',
        marker='o'
    )
    ax1.plot(
        point_results_df['WAR'],
        point_results_df['fuel_flow_gsp']*2*10,
        label='Total Aircraft Fuel Flow (kg/s)',
        linestyle='--',
        marker='o'
    )

    # Add horizontal lines for TSFC state-of-the-art and 2035 maximum
    ax1.axhline(
        y=point_results_df['fuel_2020'].iloc[0]*10,
        color='blue',
        linestyle='--',
        linewidth=2,
        label='Total Fuel Flow 2020 (State-of-the-Art) (scaled)'
    )
    ax1.axhline(
        y=point_results_df['fuel_2035_max'].iloc[0]*10,
        color='red',
        linestyle=':',
        linewidth=2,
        label='Total Fuel Flow 2035 Max (scaled)'
    )

    # Customize primary y-axis
    ax1.set_xlabel("WAR in combustor [%]")
    ax1.set_ylabel("EI_nox (g/kg fuel) & Total Aircaft Fuel Flow (scaled by 10) (kg/s)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Plot EI_nvpm_number_p3t3_meem on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(
        point_results_df['WAR'],
        point_results_df['EI_nvpm_number_p3t3_meem'],
        label='EI_nvPM_number',
        color='purple',
        linestyle='-.',
        marker='o'
    )

    # Customize secondary y-axis
    ax2.set_ylabel("EI_nvPM_number (#)")
    ax2.legend(loc="upper right")

    # Title and layout adjustments
    plt.title(f"WAR vs EI_nox_p3t3_wi, Aircraft Fuel Flow (scaled by 10) - {point_row['flight_phase']}, and EI_nvpm_number")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{engine_model}_point_{i}_plot_nvpm_nox_fuel_flow_war.png")
    plt.savefig(plot_path, format='png')
    plt.close()




    # Plot EI_nox_p3t3_wi vs fuel_flow_gsp with color-coded WAR values
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        point_results_df['fuel_flow_gsp'],
        point_results_df['EI_nox_p3t3_wi'],
        c=point_results_df['WAR'],
        cmap='viridis',  # Use a color map to represent WAR values
        edgecolor='k'
    )
    plt.colorbar(scatter, label="WAR Value")
    plt.xlabel("Fuel Flow (gsp)")
    plt.ylabel("EI_nox_p3t3_wi")
    plt.title(f"Point {i} - Original Index {point_row['original_index']} - {point_row['flight_phase']}")
    plot_path = os.path.join(output_dir, f"{engine_model}_point_{i}_plot_war_wf_nox.png")
    plt.savefig(plot_path, format='png')
    plt.close()

    # Plot EI_nox_p3t3_wi vs TSFC with color-coded WAR values
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        (point_results_df['fuel_flow_gsp']*1000)/point_results_df['thrust_gsp'],
        point_results_df['EI_nox_p3t3_wi'],
        c=point_results_df['WAR'],
        cmap='viridis',  # Use a color map to represent WAR values
        edgecolor='k'
    )
    plt.colorbar(scatter, label="WAR Value")
    plt.xlabel("TSFC (g/kNs)")
    plt.ylabel("EI_nox_p3t3_wi")
    plt.title(f"Point {i} - Original Index {point_row['original_index']} - {point_row['flight_phase']}")
    plot_path = os.path.join(output_dir, f"{engine_model}_point_{i}_plot_war_tsfc_nox.png")
    plt.savefig(plot_path, format='png')
    plt.close()

    # plt.figure(figsize=(10, 6))
    # plt.plot(point_results_df['WAR_gsp'], point_results_df['TT3'], linestyle='-', marker='o')
    # plt.title('WI effect on TT3')
    # plt.xlabel('WAR GSP [%]')
    # plt.ylabel('TT3 [K]')
    # # plt.legend()
    # plt.grid(True)
    # plot_path = os.path.join(output_dir, f"{engine_model}_point_{i}_plot_tt3.png")
    # plt.savefig(plot_path, format='png')
    # plt.close()
    # Assuming WAR = 0 is the first row in the DataFrame
    # Calculate the percentage change for TT3
    # Assuming WAR = 0 is the first row in the DataFrame
    variables = [
        {"name": "TT3", "title": "WI Effect on TT3", "unit": "K"},
        {"name": "TT4", "title": "WI Effect on TT4", "unit": "K"},
        {"name": "PT3", "title": "WI Effect on PT3", "unit": "bar"},
        {"name": "FAR", "title": "WI Effect on FAR", "unit": "-"}
    ]

    for var in variables:
        var_name = var["name"]
        var_title = var["title"]
        var_unit = var["unit"]
        try:
            # Calculate percentage change relative to WAR = 0
            war_0_value = point_results_df.loc[point_results_df['WAR'] == 0, var_name].iloc[0]
            point_results_df[f"{var_name}_pct_change"] = (point_results_df[var_name] - war_0_value) / war_0_value * 100

            # Create the plot
            plt.figure(figsize=(10, 6))
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot absolute values
            ax1.plot(point_results_df['WAR'], point_results_df[var_name], linestyle='-', marker='o', color='b')
            ax1.set_xlabel('WAR [%]')
            ax1.set_ylabel(f"{var_name} [{var_unit}]")
            ax1.tick_params(axis='y')
            ax1.grid(True)

            # Create second y-axis for percentage change
            ax2 = ax1.twinx()
            ax2.plot(point_results_df['WAR'], point_results_df[f"{var_name}_pct_change"], linestyle='--', color='b')
            ax2.set_ylabel(f"{var_name} Change [%]")
            ax2.tick_params(axis='y')

            # Set title and save the plot
            plt.title(var_title)
            plot_path = os.path.join(output_dir, f"{engine_model}_point_{i}_plot_{var_name.lower()}_with_pct_change.png")
            plt.savefig(plot_path, format='png')
            plt.close()
        except IndexError:
            print("No plot available")

    # plt.figure(figsize=(10, 6))
    # plt.plot(point_results_df['WAR_gsp'], point_results_df['TT4'], linestyle='-', marker='o')
    # plt.title('WI effect on TT4')
    # plt.xlabel('WAR GSP [%]')
    # plt.ylabel('TT4 [K]')
    # # plt.legend()
    # plt.grid(True)
    # plot_path = os.path.join(output_dir, f"{engine_model}_point_{i}_plot_tt4.png")
    # plt.savefig(plot_path, format='png')
    # plt.close()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(point_results_df['WAR_gsp'], point_results_df['PT3'], linestyle='-', marker='o')
    # plt.title('WI effect on PT3')
    # plt.xlabel('WAR GSP [%]')
    # plt.ylabel('PT3 [bar]')
    # # plt.legend()
    # plt.grid(True)
    # plot_path = os.path.join(output_dir, f"{engine_model}_point_{i}_plot_pt3.png")
    # plt.savefig(plot_path, format='png')
    # plt.close()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(point_results_df['WAR_gsp'], point_results_df['FAR'], linestyle='-', marker='o')
    # plt.title('WI effect on FAR')
    # plt.xlabel('WAR GSP [%]')
    # plt.ylabel('FAR [-]')
    # # plt.legend()
    # plt.grid(True)
    # plot_path = os.path.join(output_dir, f"{engine_model}_point_{i}_plot_pt3.png")
    # plt.savefig(plot_path, format='png')
    # plt.close()

