import numpy as np
import os
import pandas as pd
import subprocess
import constants
from matplotlib import pyplot as plt
from pycontrails.datalib.ecmwf import ERA5, ERA5ModelLevel
from emission_index import p3t3_nox
from emission_index import p3t3_nvpm_meem, p3t3_nvpm_meem_mass, thrust_setting,meem_nvpm, p3t3_nvpm_piecewise
from emission_index import NOx_correlation_de_boer, NOx_correlation_kypriandis_optimized_tf, NOx_correlation_kyprianidis
from emission_index import NOx_correlation_kaiser_optimized_tf, NOx_correlation_kaiser, p3t3_nox_xue
from emission_index import p3t3_nvpm, p3t3_nvpm_mass
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



def run_emissions_verification(trajectory, flight_path, engine_model, water_injection, SAF, aircraft, time_bounds):
    """Runs emissions calculations for a specific flight configuration."""
    if trajectory != 'malaga':
        raise ValueError("run_emissions_verification should only be called for 'malaga'.")

    if engine_model == 'GTF' and water_injection[0] == 0 and water_injection[1] == 0 and water_injection[2] == 0 and SAF == 0  and aircraft == 'A20N_full':
        df_piano = pd.read_csv(f"pianoX_malaga.csv", delimiter=';', decimal=',', index_col='index')



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
    if engine_model in ('GTF'):
        with open('p3t3_graphs_sls_gtf_final.pkl', 'rb') as f:
            loaded_functions = pickle.load(f)
    elif engine_model in ('GTF2035', 'GTF2035_wi'):
        with open('p3t3_graphs_sls_gtf2035_final.pkl', 'rb') as f:
            loaded_functions = pickle.load(f)
    elif engine_model in ('GTF1990', 'GTF2000'):
        with open('p3t3_graphs_sls_1990_2000_final.pkl', 'rb') as f:
            loaded_functions = pickle.load(f)
    else:
        raise ValueError(f"Unsupported engine_model: {engine_model}.")

    interp_func_far = loaded_functions['interp_func_far']
    interp_func_pt3 = loaded_functions['interp_func_pt3']
    interp_func_fgr = loaded_functions['interp_func_fgr']
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
            interp_func_pt3,
            interp_func_fgr
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

    # if gtf and no water injection
    if engine_model == 'GTF' and water_injection[0] == 0 and water_injection[1] == 0 and water_injection[
        2] == 0 and SAF == 0 and aircraft == 'A20N_full':
        """NOx Kaiser"""
        df_gsp['ei_nox_kaiser_opt'] = df_gsp.apply(
            lambda row: NOx_correlation_kaiser_optimized_tf(
                row['PT3'],
                row['TT3'],
                row['specific_humidity'],
                row['WAR_gsp']
            ),
            axis=1
        )

        """NOx kypriandis optimized"""
        df_gsp['ei_nox_kypriandis_opt'] = df_gsp.apply(
            lambda row: NOx_correlation_kypriandis_optimized_tf(
                row['PT3'],
                row['TT3'],
                row['TT4'],
                row['specific_humidity'],
                row['WAR_gsp'],
            ),
            axis=1
        )

        df_gsp['ei_nox_boer'] = df_gsp.apply(
            lambda row: NOx_correlation_de_boer(
                row['PT3'],
                row['TT3'],
                row['TT4'],
                row['specific_humidity'],
                row['WAR_gsp']
            ),
            axis=1
        )

        """NOx Kaiser"""
        df_gsp['ei_nox_kaiser'] = df_gsp.apply(
            lambda row: NOx_correlation_kaiser(
                row['PT3'],
                row['TT3'],
                row['specific_humidity'],
                row['WAR_gsp']
            ),
            axis=1
        )

        """NOx kypriandis optimized"""
        df_gsp['ei_nox_kypriandis'] = df_gsp.apply(
            lambda row: NOx_correlation_kyprianidis(
                row['PT3'],
                row['TT3'],
                row['TT4'],
                row['specific_humidity'],
                row['WAR_gsp'],
            ),
            axis=1
        )

        df_gsp['ei_nvpm_number_p3t3'] = df_gsp.apply(
            lambda row: p3t3_nvpm(
                row['PT3'],
                row['TT3'],
                row['FAR'],
                interp_func_far,
                interp_func_pt3,
                row['SAF'],
                row['thrust_setting_meem']
            ),
            axis=1
        )

        df_gsp['ei_nvpm_number_p3t3_piecewise'] = df_gsp.apply(
            lambda row: p3t3_nvpm_piecewise(
                row['PT3'],
                row['TT3'],
                row['FAR'],
                interp_func_far,
                interp_func_pt3,
                row['SAF'],
                row['thrust_setting_meem']
            ),
            axis=1
        )
        #
        df_gsp['ei_nvpm_mass_p3t3'] = df_gsp.apply(
            lambda row: p3t3_nvpm_mass(
                row['PT3'],
                row['TT3'],
                row['FAR'],
                interp_func_far,
                interp_func_pt3,
                row['SAF'],
                row['thrust_setting_meem']
            ),
            axis=1
        )





    # if water injection != 0:
    if  water_injection[0] != 0 or water_injection[1] != 0 or water_injection[
        2] != 0:

        df_gsp['ei_nox_p3t3_xue'] = df_gsp.apply(
            lambda row: p3t3_nox_xue(
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


        """NOx Kaiser"""
        df_gsp['ei_nox_kaiser'] = df_gsp.apply(
            lambda row: NOx_correlation_kaiser(
                row['PT3'],
                row['TT3'],
                row['specific_humidity'],
                row['WAR_gsp']
            ),
            axis=1
        )

        """NOx kypriandis optimized"""
        df_gsp['ei_nox_kypriandis'] = df_gsp.apply(
            lambda row: NOx_correlation_kyprianidis(
                row['PT3'],
                row['TT3'],
                row['TT4'],
                row['specific_humidity'],
                row['WAR_gsp'],
            ),
            axis=1
        )
    #p3t3 kaiser
    #p3t3 xue
    # kaiser
    #kypriandis

    # Plot A: $EI_{{\\mathrm{{NOx}}}}$
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nox_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3'], label='P3T3', linestyle='-', marker='o', markersize=2.5)
    plt.title(f'$EI_{{\\mathrm{{NOx}}}}$')
    plt.xlabel('Time (Minutes)')
    plt.ylabel(f'$EI_{{\\mathrm{{NOx}}}}$ (g / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nox.png', format='png')
    plt.close()

    # Plot A: $EI_{{\\mathrm{{NOx}}}}$
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nox_py'], label='FFM2 Dubois (2006) (Pycontrails)', linestyle='-')
    plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3'], label='P3T3 (This Work)', linestyle='-')
    if engine_model == 'GTF' and water_injection[0] == 0 and water_injection[1] == 0 and water_injection[
        2] == 0 and SAF == 0 and aircraft == 'A20N_full':
        # plt.plot(df_gsp.index, df_gsp['ei_nox_boer'], label='De Boer', linestyle='-')
        plt.plot(df_gsp.index, df_gsp['ei_nox_kaiser'], label='Kaiser (2022)', linestyle='-')
        # plt.plot(df_gsp.index, df_gsp['ei_nox_kaiser_opt'], label='Kaiser Opt.', linestyle='-')
        plt.plot(df_gsp.index, df_gsp['ei_nox_kypriandis'], label='Kyprianidis (2015)', linestyle='-')
        # plt.plot(df_gsp.index, df_gsp['ei_nox_kypriandis_opt'], label='Kyprianidis Opt.', linestyle='-')
        plt.plot(df_piano.index, df_piano['ei_nox_piano'], label='PianoX (2008)', linestyle='-')
    plt.title(f'$EI_{{\\mathrm{{NOx}}}}$')
    plt.xlabel('Time (Minutes)')
    plt.ylabel(f'$EI_{{\\mathrm{{NOx}}}}$ (g / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nox_no_markers.png', format='png')
    plt.close()

    if (water_injection[0] != 0 or water_injection[1] != 0 or water_injection[
        2] != 0) and (engine_model == 'GTF2035' or engine_model == 'GTF2035_wi'):
        plt.figure(figsize=(10, 6))
        plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3'], label='P3T3 Kaiser (This Work)', linestyle='-')
        plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3_xue'], label='P3T3 Xue (2016)', linestyle='-')
        plt.plot(df_gsp.index, df_gsp['ei_nox_kaiser'], label='Kaiser (2022)', linestyle='-')
        plt.plot(df_gsp.index, df_gsp['ei_nox_kypriandis'], label='Kyprianidis (2015)', linestyle='-')
        plt.title(f'$EI_{{\\mathrm{{NOx}}}}$ Prediction for Steam Injection {water_injection[0]}% ')
        plt.xlabel('Time (Minutes)')
        plt.ylabel(f'$EI_{{\\mathrm{{NOx}}}}$ (g / kg Fuel)')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nox_WAR_correlations.png',
            format='png')
        plt.close()

    # Plot A: $EI_{{\\mathrm{{NOx}}}}$
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp['TT3'], df_gsp['ei_nox_py'], label='Pycontrails', linestyle='-')
    plt.plot(df_gsp['TT3'], df_gsp['ei_nox_p3t3'], label='P3T3', linestyle='-')
    plt.title(f'$EI_{{\\mathrm{{NOx}}}}$')
    plt.xlabel('TT3')
    plt.ylabel(f'$EI_{{\\mathrm{{NOx}}}}$ (g / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nox_tt3.png', format='png')
    plt.close()# plt.show()

    # Plot B: EI_nvpm_mass
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_p3t3_meem'], label='Adjusted MEEM (This Work)', linestyle='-', marker='o', markersize=2.5)
    plt.title(f'$EI_{{\\mathrm{{nvPM,mass}}}}$')
    plt.xlabel('Time (Minutes)')
    plt.ylabel(f'$EI_{{\\mathrm{{nvPM,mass}}}}$ (mg / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nvpm_mass.png', format='png')
    plt.close()

    # Plot B: EI_nvpm_mass
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_py'], label='T4/T2 Teoh (2020) (Pycontrails)', linestyle='-')
    if engine_model == 'GTF' and water_injection[0] == 0 and water_injection[1] == 0 and water_injection[
        2] == 0 and SAF == 0 and aircraft == 'A20N_full':
        plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_p3t3'], label='P3T3 Saluja (2023)', linestyle='-')
        plt.plot(df_gsp.index, df_gsp['ei_mass_meem'], label='MEEM Ahrens (2022)', linestyle='-')
        plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_p3t3_meem'], label='Adjusted MEEM (This Work)', linestyle='-')
    else:
        plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_p3t3_meem'], label='Adjusted MEEM (This Work)', linestyle='-')
    plt.title(f'$EI_{{\\mathrm{{nvPM,mass}}}}$')
    plt.xlabel('Time (Minutes)')
    plt.ylabel(f'$EI_{{\\mathrm{{nvPM,mass}}}}$ (mg / kg Fuel)')
    plt.ylim(0,45)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nvpm_mass_no_markers.png', format='png')
    plt.close()

    # Plot C: EI_nvpm_number
    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3_meem'], label='Adjusted MEEM (This Work)', linestyle='-', marker='o', markersize=2.5)
    plt.title(f'$EI_{{\\mathrm{{nvPM,number}}}}$')
    plt.xlabel('Time (Minutes)')
    plt.ylabel(f'$EI_{{\\mathrm{{nvPM,number}}}}$ (# / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nvpm_number.png', format='png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_py'], label='T4/T2 Teoh (2020) (Pycontrails)', linestyle='-')
    if engine_model == 'GTF' and water_injection[0] == 0 and water_injection[1] == 0 and water_injection[
        2] == 0 and SAF == 0 and aircraft == 'A20N_full':
        plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3'], label='P3T3 Saluja (2023)', linestyle='-')
        plt.plot(df_gsp.index, df_gsp['ei_number_meem'], label='MEEM Ahrens (2022)', linestyle='-')
        plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3_meem'], label='Adjusted MEEM (This Work)', linestyle='-')
        # plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3_piecewise'], label='P3T3 Piecewise Corr', linestyle='-')
    else:
        plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3_meem'], label='Adjusted MEEM (This Work)', linestyle='-')
    plt.title(f'$EI_{{\\mathrm{{nvPM,number}}}}$')
    plt.xlabel('Time (Minutes)')
    plt.ylabel(f'$EI_{{\\mathrm{{nvPM,number}}}}$ (# / kg Fuel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_ei_nvpm_number_no_markers.png', format='png')
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['fuel_flow_per_engine'], label='Poll-Schumann (2021) (Pycontrails)', linestyle='-')
    plt.plot(df_gsp.index, df_gsp['fuel_flow_gsp'], label='GSP (This Work)', linestyle='-')
    if engine_model == 'GTF' and water_injection[0] == 0 and water_injection[1] == 0 and water_injection[2] == 0 and SAF == 0  and aircraft == 'A20N_full':
        plt.plot(df_piano.index, df_piano['fuel_flow_piano'], label='PianoX (2008)', linestyle='-')
    plt.title('Fuel Flow')
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Fuel Flow (kg/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_fuel_flow.png', format='png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_gsp.index, df_gsp['thrust_per_engine'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
    plt.plot(df_gsp.index, df_gsp['thrust_gsp'], label='GSP', linestyle='-', marker='o', markersize=2.5)
    plt.title('Thrust')
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Thrust (kN)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_thrust.png', format='png')
    plt.close()
    # Convert the water_injection values to strings, replacing '.' with '_'
    formatted_values = [str(value).replace('.', '_') for value in water_injection]

    df_gsp.to_csv(f'main_results_figures/results/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv')


    if engine_model == 'GTF' and water_injection[0] == 0 and water_injection[1] == 0 and water_injection[2] == 0 and SAF == 0  and aircraft == 'A20N_full':
        df_luft = pd.read_csv("20250219_Selected_Reference_Aircraft_Missions_luftbauhaus.csv", delimiter=";",
                              decimal=',')  # Adjust delimiter if needed

        # Convert duration to cumulative time in minutes
        df_luft["Cumulative Time (min)"] = df_luft["Duration [s]"].cumsum() / 60

        # Generate full-minute timestamps
        time_min = np.arange(0, int(np.ceil(df_luft["Cumulative Time (min)"].max())) + 1, 1)

        # Interpolate values for the new DataFrame
        df_luft_interp = pd.DataFrame({"Cumulative Time (min)": time_min})
        df_luft_interp["Fuel Flow per Engine [kg/s]"] = np.interp(time_min, df_luft["Cumulative Time (min)"],
                                                                  df_luft["Fuel Flow per Engine [kg/s]"])
        df_luft_interp["Total Aircraft Thrust [N]"] = np.interp(time_min, df_luft["Cumulative Time (min)"],
                                                                df_luft["Total Aircraft Thrust [N]"])
        df_luft_interp["Altitude [m]"] = np.interp(time_min, df_luft["Cumulative Time (min)"], df_luft["Altitude [m]"])
        df_luft_interp["Mach [-]"] = np.interp(time_min, df_luft["Cumulative Time (min)"], df_luft["Mach [-]"])

        # Compute Engine Thrust (kN)
        df_luft_interp["Engine Thrust [kN]"] = (0.5 * df_luft_interp["Total Aircraft Thrust [N]"]) / 1000

        df_malaga_luft = df_gsp.copy()

        # Find the starting altitude in df2
        start_altitude = df_malaga_luft.iloc[0]["altitude"]

        # Find the first index in df1 where altitude matches the start altitude of df2
        start_index = (df_luft_interp["Altitude [m]"] - start_altitude).abs().idxmin()

        # Get the corresponding cumulative time in df1
        start_time = df_luft_interp.loc[start_index, "Cumulative Time (min)"]

        # Assign cumulative time in df2 starting from start_time
        df_malaga_luft["Cumulative Time (min)"] = np.arange(start_time, start_time + len(df_malaga_luft))

        df_piano_luft = pd.read_csv(f"pianoX_malaga.csv", delimiter=';', decimal=',', index_col='index')

        # Find the first index in df_malaga_luft
        start_index_malaga = df_malaga_luft.index.min()  # Smallest index in df_malaga_luft
        start_index_piano = df_piano_luft.index.min()  # Smallest index in df_piano_luft (36 in this case)

        # Find the corresponding cumulative time for df_piano_luft's first index in df_malaga_luft
        start_time_piano = df_malaga_luft.loc[start_index_piano, "Cumulative Time (min)"]

        # Assign cumulative time to df_piano_luft based on df_malaga_luft's timeline
        df_piano_luft["Cumulative Time (min)"] = np.arange(start_time_piano, start_time_piano + len(df_piano_luft))

        # Plot Fuel Flow per Engine vs. Time
        plt.figure(figsize=(10, 6))
        plt.plot(df_luft_interp["Cumulative Time (min)"], df_luft_interp["Fuel Flow per Engine [kg/s]"],
                 label="Verification Flight Luftfahrt Bauhaus", linestyle="-", marker="o", markersize=2.5, color='tab:purple')
        plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["fuel_flow_per_engine"], label="pycontrails (AGP-AMS)",
                 linestyle="-", marker="o", markersize=2.5, color='tab:blue')
        plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["fuel_flow_gsp"], label="GSP (AGP-AMS)", linestyle="-",
                 marker="o", markersize=2.5, color='tab:orange')
        plt.plot(df_piano_luft["Cumulative Time (min)"], df_piano_luft['fuel_flow_piano'], label='PianoX (AGP-AMS)',
                 linestyle='-', marker='o', markersize=2.5, color='tab:green')
        plt.title("Fuel Flow Over Time")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Fuel Flow (kg/s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_luftbauhaus_fuel.png',
            format='png')
        # plt.show()

        # Plot Thrust per Engine vs. Time
        plt.figure(figsize=(10, 6))
        plt.plot(df_luft_interp["Cumulative Time (min)"], df_luft_interp["Engine Thrust [kN]"],
                 label="Verification Flight Luftfahrt Bauhaus", linestyle="-", marker="o", markersize=2.5, color='tab:purple')
        plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["thrust_per_engine"], label="pycontrails (AGP-AMS)",
                 linestyle="-", marker="o", markersize=2.5, color='tab:blue')
        # plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["thrust_gsp"], label="GSP", linestyle="-", marker="o", markersize=2.5, color='tab:orange')
        plt.title("Thrust per Engine Over Time")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Thrust (kN)")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_luftbauhaus_thrust.png',
            format='png')
        # plt.show()

        # Plot Altitude vs. Time
        plt.figure(figsize=(10, 6))
        plt.plot(df_luft_interp["Cumulative Time (min)"], df_luft_interp["Altitude [m]"], label="Verification Flight Luftfahrt Bauhaus",
                 linestyle="-", marker="o", markersize=2.5, color='tab:purple')
        plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["altitude"], label="pycontrails (AGP-AMS)",
                 linestyle="-", marker="o", markersize=2.5, color='tab:blue')
        plt.title("Altitude Over Time")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Altitude (m)")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_luftbauhaus_altitude.png',
            format='png')
        # plt.show()

        # Plot Mach vs. Time
        plt.figure(figsize=(10, 6))
        plt.plot(df_luft_interp["Cumulative Time (min)"], df_luft_interp["Mach [-]"], label="Verification Flight Luftfahrt Bauhaus",
                 linestyle="-", marker="o", markersize=2.5, color='tab:purple')
        plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["mach"], label="pycontrails (AGP-AMS)", linestyle="-",
                 marker="o", markersize=2.5, color='tab:blue')
        plt.title("Mach Over Time")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Mach")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(
            f'main_results_figures/figures/{trajectory}/{flight}/emissions/{engine_model}_SAF_{SAF}_luftbauhaus_mach.png',
            format='png')

        # === CONFIG ===
        output_dir = f"main_results_figures/results/{trajectory}/{flight}/emissions/"
        # os.makedirs(output_dir, exist_ok=True)

        # === Fuel Flow Comparison ===
        fuel_flow_methods = {
            "Poll-Schumann (2021) (Pycontrails)": df_gsp['fuel_flow_per_engine'],
            "GSP (This Work)": df_gsp['fuel_flow_gsp']
        }

        fuel_flow_summary = pd.DataFrame()
        for phase in ['climb', 'cruise', 'descent']:
            ref_sum = df_gsp[df_gsp['flight_phase'] == phase]['fuel_flow_gsp'].sum()
            for method, series in fuel_flow_methods.items():
                values = df_gsp
                phase_sum = values[values['flight_phase'] == phase][series.name].sum()
                fuel_flow_summary.loc[method, phase] = 100 * (phase_sum - ref_sum) / ref_sum

        ref_total = df_gsp['fuel_flow_gsp'].sum()
        for method, series in fuel_flow_methods.items():
            total_sum = series.sum()
            fuel_flow_summary.loc[method, 'Total'] = 100 * (total_sum - ref_total) / ref_total

        fuel_flow_summary.to_csv(os.path.join(output_dir, "fuel_flow_comparison_phases.csv"))

        # === NOx Comparison ===
        nox_methods = {
            "FFM2 Dubois (2006) (Pycontrails)": df_gsp['ei_nox_py'],
            "P3T3 (This Work)": df_gsp['ei_nox_p3t3'],
            "Kaiser (2022)": df_gsp['ei_nox_kaiser'],
            "Kyprianidis (2015)": df_gsp['ei_nox_kypriandis']
        }

        nox_summary = pd.DataFrame()
        for phase in ['climb', 'cruise', 'descent']:
            ref_sum = df_gsp[df_gsp['flight_phase'] == phase]['ei_nox_p3t3'].sum()
            for method, series in nox_methods.items():
                phase_sum = df_gsp[df_gsp['flight_phase'] == phase][series.name].sum()
                nox_summary.loc[method, phase] = 100 * (phase_sum - ref_sum) / ref_sum

        ref_total = df_gsp['ei_nox_p3t3'].sum()
        for method, series in nox_methods.items():
            total_sum = series.sum()
            nox_summary.loc[method, 'Total'] = 100 * (total_sum - ref_total) / ref_total

        nox_summary.to_csv(os.path.join(output_dir, "nox_comparison_phases.csv"))

        # === nvPM Comparison ===
        nvpm_methods = {
            "T4/T2 Teoh (2020) (Pycontrails)": df_gsp['ei_nvpm_number_py'],
            "P3T3 Saluja (2023)": df_gsp['ei_nvpm_number_p3t3'],
            "MEEM Ahrens (2022)": df_gsp['ei_number_meem'],
            "Adjusted MEEM (This Work)": df_gsp['ei_nvpm_number_p3t3_meem']
        }

        nvpm_summary = pd.DataFrame()
        for phase in ['climb', 'cruise', 'descent']:
            ref_sum = df_gsp[df_gsp['flight_phase'] == phase]['ei_nvpm_number_p3t3_meem'].sum()
            for method, series in nvpm_methods.items():
                phase_sum = df_gsp[df_gsp['flight_phase'] == phase][series.name].sum()
                nvpm_summary.loc[method, phase] = 100 * (phase_sum - ref_sum) / ref_sum

        ref_total = df_gsp['ei_nvpm_number_p3t3_meem'].sum()
        for method, series in nvpm_methods.items():
            total_sum = series.sum()
            nvpm_summary.loc[method, 'Total'] = 100 * (total_sum - ref_total) / ref_total

        nvpm_summary.to_csv(os.path.join(output_dir, "nvpm_comparison_phases.csv"))

    return True