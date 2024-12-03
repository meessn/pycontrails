import pandas as pd
import matplotlib.pyplot as plt

"""FLIGHT PARAMETERS"""
engine_model = 'GTF'        # GTF , GTF2035
water_injection = [0, 0, 0]     # WAR climb cruise approach/descent
SAF = 0                         # 0, 20, 100 unit = %
flight = 'malaga'
aircraft = 'A20N_full'        # A20N ps model, A20N_wf is change in Thrust and t/o and idle fuel flows
                            # A20N_wf_opr is with changed nominal opr and bpr
                            # A20N_full has also the eta 1 and 2 and psi_0

# Convert the water_injection values to strings, replacing '.' with '_'
formatted_values = [str(value).replace('.', '_') for value in water_injection]

water_injections = [0, 0, 0] [17, 14, 16]
fuel_flow_gsp_data = {}
nox_p3t3_data = {}

# Loop through SAF values and read corresponding files
for water_injection in water_injections:
    file_name = f'results/{flight}/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_values[0]}_{formatted_values[1]}_{formatted_values[2]}.csv'
    try:
        # Read the CSV file
        df = pd.read_csv(file_name)

        # Extract the nvPM_p3t3_meem column and store it
        fuel_flow_gsp_data[engine] = df['fuel_flow_gsp']
        nox_p3t3_data[engine] = df['EI_nox_p3t3']
        nvPM_number_data[engine] = df['EI_nvpm_number_p3t3_meem']
        nvPM_mass_data[engine] = df['EI_nvpm_mass_p3t3_meem']

    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except KeyError:
        print(f"'nvPM_p3t3_meem' column not found in file: {file_name}")

# Plot the data
plt.figure(figsize=(10, 6))
for engine, fuel in fuel_flow_gsp_data.items():
    plt.plot(fuel, label=f"Engine {engine}")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("Fuel flow [kg/s]")
plt.title("Fuel flow for different engines")
plt.legend(title="Engine")
plt.grid()
plt.savefig(f'figures/{flight}/EI_fuel_flow_engines.png', format='png')

plt.figure(figsize=(10, 6))
for engine, nox in nox_p3t3_data.items():
    plt.plot(nox, label=f"Engine {engine}")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_NOx")
plt.title("P3T3 EI_NOx for different engines")
plt.legend(title="Engine")
plt.grid()
plt.savefig(f'figures/{flight}/EI_nox_water_injection.png', format='png')

