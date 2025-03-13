import numpy as np
import os
import pandas as pd
import subprocess
import constants
from matplotlib import pyplot as plt
from pycontrails.datalib.ecmwf import ERA5, ERA5ModelLevel
from emission_index import p3t3_nox
from emission_index import p3t3_nvpm_meem, p3t3_nvpm_meem_mass, thrust_setting,meem_nvpm
import pickle
from pycontrails import Flight
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import HistogramMatching, ExponentialBoostHumidityScaling
from pycontrails.models.ps_model import PSFlight
from pycontrails.models.emissions import Emissions
from pycontrails.datalib import ecmwf
from pycontrails.core.cache import DiskCacheStore
from pathlib import Path
import copy

import warnings
# root_dir = "flight_trajectories/processed_flights"
#
# flight_trajectories_to_simulate = {
#     "bos_fll": True,
#     "cts_tpe": False,
#     "dus_tos": False,
#     "gru_lim": False,
#     "hel_kef": False,
#     "lhr_ist": False,
#     "sfo_dfw": False,
#     "sin_maa": False
# }
#
# time_bounds_dict = {
#     "2023-02-06": ("2023-02-05 14:00", "2023-02-07 11:00"),
#     "2023-05-05": ("2023-05-04 14:00", "2023-05-06 11:00"),
#     "2023-08-06": ("2023-08-05 14:00", "2023-08-07 11:00"),
#     "2023-11-06": ("2023-11-05 14:00", "2023-11-07 11:00")
# }
#
# engine_models = {
#     "GTF1990": True,
#     "GTF2000": True,
#     "GTF": True,
#     "GTF2035": True,
#     "GTF2035_wi_gass_on_design": True
# }
#
# saf_dict = {
#     "SAF20": True,
#     "SAF100": True
# }
#
#
# """FLIGHT PARAMETERS"""
# engine_model = 'GTF'        # GTF , GTF2035, GTF1990, GTF2000
# water_injection = [0, 0, 0]     # WAR climb cruise approach/descent
# SAF = 0                   # 0, 20, 100 unit = %
# flight = 'malaga'
# aircraft = 'A20N_full'        # A20N ps model, A20N_wf is change in Thrust and t/o and idle fuel flows
#                             # A20N_wf_opr is with changed nominal opr and bpr
#                             # A20N_full has also the eta 1 and 2 and psi_0


def run_emissions(trajectory, flight_path, engine_model, water_injection, SAF, aircraft, time_bounds):
    """Runs emissions calculations for a specific flight configuration."""

    flight = os.path.basename(flight_path).replace('.csv', '')
    print(f"\nRunning emissions for {flight} | Engine: {engine_model} | SAF: {SAF} | Water Injection: {water_injection}")

    # Define the output directory
    output_csv_dir = f"main_results_figures/results/{trajectory}/{flight}/emissions/"

    # Ensure the directory exists
    os.makedirs(output_csv_dir, exist_ok=True)

    # Define the output directory
    output_dir = f"main_results_figures/figures/{trajectory}/{flight}/emissions/"

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load flight data
    df = pd.read_csv(flight_path)
    if flight == "malaga":
        df = df.rename(columns={'geoaltitude': 'altitude', 'groundspeed': 'groundspeed', 'timestamp':'time'})
        df = df.dropna(subset=['callsign'])
        df = df.dropna(subset=['altitude'])
        df = df.drop(['Unnamed: 0', 'icao24', 'callsign'], axis=1)

        # df = df[df['altitude'] > 1900]
        column_order = ['longitude', 'latitude', 'altitude', 'groundspeed', 'time']
        df = df[column_order]
        df['altitude'] = df['altitude']*0.3048 #foot to meters
        df['groundspeed'] = df['groundspeed']*0.514444444

    if engine_model == 'GTF' or engine_model == 'GTF2035' or engine_model == 'GTF2035_wi':
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



    if flight == "malaga":
        """SAMPLE AND FILL DATA"""
        fl = fl.resample_and_fill(freq="60s", drop=False) # recommended for CoCiP
        fl.dataframe['groundspeed'] = fl.dataframe['groundspeed'].interpolate(method='linear', inplace=True)

    """------RETRIEVE METEOROLOGIC DATA----------------------------------------------"""

    # time_bounds = ("2024-06-07 9:00", "2024-06-08 02:00")

    pressure_levels_10 = np.arange(150, 400, 10)  # 150 to 400 with steps of 10
    pressure_levels_50 = np.arange(400, 1001, 50)  # 400 to 1000 with steps of 50
    pressure_levels_model = np.concatenate((pressure_levels_10, pressure_levels_50))

    if flight == 'malaga':
        local_cache_dir = Path("F:/era5model/malaga")
        variables_model = ("t", "q", "u", "v", "w", "ciwc", "vo", "clwc")
    else:
        local_cache_dir = Path("F:/era5model/flights")
        variables_model = ("t", "q", "u", "v", "w", "ciwc")

    local_cachestore = DiskCacheStore(cache_dir=local_cache_dir)

    era5ml = ERA5ModelLevel(
                    time=time_bounds,
                    variables=variables_model,
                    model_levels=range(67, 133),
                    pressure_levels=pressure_levels_model,
                    cachestore=local_cachestore
                )
    # era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables + (ecmwf.SurfaceSolarDownwardRadiation,))

    # download data from ERA5 (or open from cache)
    met = era5ml.open_metdataset()
    # Extract min/max longitude and latitude from the dataframe
    west = fl.dataframe["longitude"].min() - 50  # Subtract 1 degree for west buffer
    east = fl.dataframe["longitude"].max() + 50 # Add 1 degree for east buffer
    south = fl.dataframe["latitude"].min() - 50  # Subtract 1 degree for south buffer
    north = fl.dataframe["latitude"].max() + 50  # Add 1 degree for north buffer

    # Define the bounding box with altitude range
    bbox = (west, south, 150, east, north, 1000)  # (west, south, min-level, east, north, max-level)
    met = met.downselect(bbox=bbox)
    met_ps = copy.deepcopy(met)#era5ml.open_metdataset() # meteorology
    met_emi = copy.deepcopy(met)
    # rad = era5sl.open_metdataset() # radiation


    # fl_test = copy.deepcopy(fl)
    # print(fl_test.intersect_met(met_ps['specific_humidity']))
    """-----RUN AIRCRAFT PERFORMANCE MODEL--------------------------------------------"""

    perf = PSFlight(
        met=met_ps,
        fill_low_altitude_with_isa_temperature=True,  # Estimate temperature using ISA
        fill_low_altitude_with_zero_wind=True
    )
    fp = perf.eval(fl)
    df_p = fp.dataframe
    df_p.update(df_p.select_dtypes(include=[np.number]).interpolate(method='linear', limit_area='inside'))
    fp = Flight(df_p, attrs=attrs)

    """---------EMISSIONS MODEL FFM2 + ICAO-------------------------------------------------------"""
    emissions = Emissions(met=met_emi, humidity_scaling=ExponentialBoostHumidityScaling(rhi_adj=0.9779, rhi_boost_exponent=1.635,
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
    plt.title(f'Altitude Profile of {flight} Flight')
    plt.grid(True)

    # Create a legend for the phases
    for phase, color in phase_colors.items():
        plt.plot([], [], color=color, label=phase)  # Dummy plot for the legend

    plt.legend(title="Flight Phase")
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_flight_phases.png', format='png')
    plt.close()# plt.show()

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
        df_water = pd.read_csv(f'main_results_figures/results/{trajectory}/{flight}/emissions/GTF2035_SAF_{SAF}_{aircraft}_WAR_0_0_0.csv')
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

    """DELETE NAN ROWS (EXCEPT FOR 'callsign' OR 'icao24')"""

    try:
        # Exclude 'callsign' and 'icao24' when checking for NaN values
        relevant_cols = df.drop(columns=['callsign', 'icao24'], errors='ignore')
        numeric_cols = relevant_cols.select_dtypes(include='number').columns

        # Identify all rows containing NaN in relevant columns
        nan_rows = df[relevant_cols.isna().any(axis=1)].index

        if nan_rows.empty:
            print("No NaN values found. Skipping deletion.")
        else:
            rows_to_delete = []  # Track rows to delete
            interpolate_needed = False

            for row_index in nan_rows:
                # Check if 'ei_nox_py' exists in the DataFrame
                if 'ei_nox_py' in df.columns:
                    all_previous_ei_nox_py_nan = df.iloc[:row_index]['ei_nox_py'].isna().all()
                    all_remaining_ei_nox_py_nan = df.iloc[row_index + 1:]['ei_nox_py'].isna().all()
                else:
                    all_previous_ei_nox_py_nan = all_remaining_ei_nox_py_nan = True

                # If all previous or all remaining rows have only NaN in `ei_nox_py`, delete as an edge row
                if all_previous_ei_nox_py_nan or all_remaining_ei_nox_py_nan:
                    rows_to_delete.append(row_index)
                else:
                    interpolate_needed = True

                    # Otherwise, it's a middle NaN → Check the closest valid rows before and after
                    prev_valid = df.iloc[:row_index].dropna(subset=['ei_nox_py']).index[-1] if not df.iloc[
                                                                                               :row_index].dropna(
                        subset=['ei_nox_py']).empty else None
                    next_valid = df.iloc[row_index + 1:].dropna(subset=['ei_nox_py']).index[0] if not df.iloc[
                                                                                                  row_index + 1:].dropna(
                        subset=['ei_nox_py']).empty else None

                    print(f"NaN detected in a non-edge row at index {row_index}. ")
                    print(f"First valid row before: {prev_valid}, first valid row after: {next_valid}.")
                    # raise ValueError(
                    #     f"NaN detected in a non-edge row at index {row_index}. "
                    #     f"First valid row before: {prev_valid}, first valid row after: {next_valid}."
                    # )

            if interpolate_needed:
                # Interpolate only numeric columns
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
                print("Interpolation performed for non-edge NaNs.")


            if rows_to_delete:
                df.drop(rows_to_delete, inplace=True)
                print(f"Total rows deleted: {len(rows_to_delete)}")

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

        # Read the results back into the main DataFrame
        results_df = pd.read_csv(output_csv_path)

    elif engine_model == 'GTF2000':  # Special handling for GTF2000 → Copy from GTF1990
        formatted_values = [str(value).replace('.', '_') for value in water_injection]
        gtf1990_file_path = (
            f"main_results_figures/results/{trajectory}/{flight}/emissions/"
            f"GTF1990_SAF_{SAF}_{aircraft}_WAR_"
            f"{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv"
        )

        gtf1990_df = pd.read_csv(gtf1990_file_path)

        # Make a copy of `df` for GTF2000 (to keep naming clearer)
        df_2000 = pd.read_csv(input_csv_path)

        # Columns to copy from GTF1990
        columns_to_copy = ['index', 'PT3', 'TT3', 'TT4', 'specific_humidity_gsp', 'FAR', 'fuel_flow_gsp', 'thrust_gsp', 'W3']

        if not df_2000['time'].equals(gtf1990_df['time']):
            print("GTF2000 'time' head():")
            print(df_2000['time'].head(10))

            print("\nGTF1990 'time' head():")
            print(gtf1990_df['time'].head(10))

            print("\nGTF2000 'time' tail():")
            print(df_2000['time'].tail(10))

            print("\nGTF1990 'time' tail():")
            print(gtf1990_df['time'].tail(10))

            # Find mismatches
            for i, (t1, t2) in enumerate(zip(df_2000['time'], gtf1990_df['time'])):
                if t1 != t2:
                    print(f"Mismatch at index {i}: GTF2000 time = {t1}, GTF1990 time = {t2}")
                    break

            raise ValueError("Mismatch detected: GTF2000 and GTF1990 'time' columns do not match.")

        # Copy over the required columns into `df_2000`
        for col in columns_to_copy:
            df_2000[col] = gtf1990_df[col]

        # Use df_2000 as `results_df` to keep downstream logic uniform
        results_df = df_2000[columns_to_copy].copy()

    else:
        raise ValueError(f"Unsupported engine_model: {engine_model}")



    # Merge the results back into the original DataFrame
    df_gsp = pd.read_csv(input_csv_path)  # Load the original DataFrame
    df_gsp = df_gsp.merge(results_df, on='index', how='left')

    df_gsp['W3_no_specific_humid'] = df_gsp['W3'] / (1+df_gsp['specific_humidity']) #pure air, without water from ambience

    df_gsp['WAR_gsp'] = ((df_gsp['water_injection_kg_s'] + df_gsp['specific_humidity']*df_gsp['W3_no_specific_humid']) / df_gsp['W3_no_specific_humid'])*100 #%

    # df_gsp = df_gsp.interpolate(method='linear', limit_area='inside')
    df_gsp.update(df_gsp.select_dtypes(include=[np.number]).interpolate(method='linear', limit_area='inside'))
    # Load interpolation functions based on engine model
    if engine_model in ('GTF', 'GTF2035', 'GTF2035_wi'):
        with open('p3t3_graphs_sls_gtf_corr.pkl', 'rb') as f:
            loaded_functions = pickle.load(f)
    elif engine_model in ('GTF1990', 'GTF2000'):
        with open('p3t3_graphs_sls_1990_2000.pkl', 'rb') as f:
            loaded_functions = pickle.load(f)
    else:
        raise ValueError(f"Unsupported engine_model: {engine_model}.")

    interp_func_far = loaded_functions['interp_func_far']
    interp_func_pt3 = loaded_functions['interp_func_pt3']

    # Get interpolation function bounds
    x_min, x_max = interp_func_far.x[0], interp_func_far.x[-1]

    # Get min and max TT3 from df_gsp
    tt3_min = df_gsp['TT3'].min()
    tt3_max = df_gsp['TT3'].max()

    # Identify out-of-bounds values
    out_of_bounds_mask = (df_gsp['TT3'] < x_min) | (df_gsp['TT3'] > x_max)
    out_of_bounds_values = df_gsp.loc[out_of_bounds_mask, 'TT3']

    if not out_of_bounds_values.empty:
        warnings.warn(f"TT3 values in df_gsp are outside the interpolation range ({x_min}, {x_max}). "
                      f"Min TT3: {tt3_min}, Max TT3: {tt3_max}. Extreme values are clipped.")

        print(f"Number of TT3 values out of bounds: {out_of_bounds_values.shape[0]}")
        print("Out-of-bounds TT3 values:", out_of_bounds_values.tolist())

        # Clamp values to stay within bounds
        df_gsp['TT3'] = df_gsp['TT3'].clip(lower=x_min, upper=x_max)




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

    df_gsp[['ei_mass_meem', 'ei_number_meem']] = df_gsp.apply(
        lambda row: pd.Series(meem_nvpm(
            row['altitude'],
            row['mach'],
            average_cruise_altitude,
            row['flight_phase'],
            row['SAF']
        )),
        axis=1
    )

    # Plot A: EI_NOx
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nox_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3'], label='P3T3', linestyle='-', marker='o', markersize=2.5)
    plt.title('EI_NOx')
    plt.xlabel('Time in minutes')
    plt.ylabel('EI_NOx (g/ kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nox.png', format='png')
    plt.close()

    # Plot A: EI_NOx
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nox_py'], label='Pycontrails', linestyle='-')
    plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3'], label='P3T3', linestyle='-')
    plt.title('EI_NOx')
    plt.xlabel('Time in minutes')
    plt.ylabel('EI_NOx (g/ kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nox_no_markers.png', format='png')
    plt.close()

    # Plot A: EI_NOx
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp['TT3'], df_gsp['ei_nox_py'], label='Pycontrails', linestyle='-')
    plt.plot(df_gsp['TT3'], df_gsp['ei_nox_p3t3'], label='P3T3', linestyle='-')
    plt.title('EI_NOx')
    plt.xlabel('TT3')
    plt.ylabel('EI_NOx (g/ kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nox_tt3.png', format='png')
    plt.close()# plt.show()

    # Plot B: EI_nvpm_mass
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_p3t3_meem'], label='P3T3 - MEEM', linestyle='-', marker='o', markersize=2.5)
    plt.title('EI_nvPM_mass')
    plt.xlabel('Time in minutes')
    plt.ylabel('EI_nvPM_mass (mg / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nvpm_mass.png', format='png')
    plt.close()

    # Plot B: EI_nvpm_mass
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_py'], label='Pycontrails', linestyle='-')
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_p3t3_meem'], label='P3T3 - MEEM', linestyle='-')
    plt.title('EI_nvPM_mass')
    plt.xlabel('Time in minutes')
    plt.ylabel('EI_nvPM_mass (mg / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nvpm_mass_no_markers.png', format='png')
    plt.close()

    # Plot C: EI_nvpm_number
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3_meem'], label='P3T3 - MEEM', linestyle='-', marker='o', markersize=2.5)
    plt.title('EI_nvPM_number')
    plt.xlabel('Time in minutes')
    plt.ylabel('EI_nvPM_number (# / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nvpm_number.png', format='png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_py'], label='Pycontrails', linestyle='-')
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3_meem'], label='P3T3 - MEEM', linestyle='-')
    plt.title('EI_nvPM_number')
    plt.xlabel('Time in minutes')
    plt.ylabel('EI_nvPM_number (# / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nvpm_number_no_markers.png', format='png')
    plt.close()

    df_piano = pd.read_csv(f"pianoX_malaga.csv", delimiter=';', decimal=',', index_col='index')

    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['fuel_flow_per_engine'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_gsp.index, df_gsp['fuel_flow_gsp'], label='GSP', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_piano.index, df_piano['fuel_flow_piano'], label='PianoX', linestyle='-', marker='o', markersize=2.5)
    plt.title('Fuel Flow')
    plt.xlabel('Time in minutes')
    plt.ylabel('Fuel Flow (kg/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_fuel_flow.png', format='png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['thrust_per_engine'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_gsp.index, df_gsp['thrust_gsp'], label='GSP', linestyle='-', marker='o', markersize=2.5)
    plt.title('Thrust')
    plt.xlabel('Time in minutes')
    plt.ylabel('Thrust (kN)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_thrust.png', format='png')
    plt.close()
    # Convert the water_injection values to strings, replacing '.' with '_'
    formatted_values = [str(value).replace('.', '_') for value in water_injection]

    df_gsp.to_csv(f'main_results_figures/results/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv')

    return True