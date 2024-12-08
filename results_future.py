import pandas as pd
import matplotlib.pyplot as plt



# Parameters
flight = 'malaga' # Replace with your flight identifier
engine_model = 'GTF', 'GTF2035'  # Replace with your engine model
SAF = 0  # List of SAF configurations
fuel_flow_gsp_data = {}
nox_p3t3_data = {}
nvPM_number_data = {}  # To store data for plotting
nvPM_mass_data = {}
# Loop through SAF values and read corresponding files
for engine in engine_model:
    file_name = f'results/{flight}/{flight}_model_{engine}_SAF_0_aircraft_A20N_full_WAR_0_0_0.csv'
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

plt.xlabel("Time (m)")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("Fuel flow [kg/s]")
plt.title("Fuel flow for different engines")
plt.legend(title="Engine")
plt.grid()
plt.savefig(f'figures/{flight}/future_engines/EI_fuel_flow_engines.png', format='png')

plt.figure(figsize=(10, 6))
for engine, nox in nox_p3t3_data.items():
    plt.plot(nox, label=f"Engine {engine}")

plt.xlabel("Time (m)")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_NOx")
plt.title("P3T3 EI_NOx for different engines")
plt.legend(title="Engine")
plt.grid()
plt.savefig(f'figures/{flight}/future_engines/EI_nox_engines.png', format='png')

plt.figure(figsize=(10, 6))
for engine, nvPM_number in nvPM_number_data.items():
    plt.plot(nvPM_number, label=f"Engine {engine}")

plt.xlabel("Time (m)")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_nvpm_number")
plt.title("P3T3-MEEM EI_nvPM_number for different engines")
plt.legend(title="Engine")
plt.grid()
plt.savefig(f'figures/{flight}/future_engines/EI_nvpm_number_engines.png', format='png')

plt.figure(figsize=(10, 6))
for engine, nvPM_mass in nvPM_mass_data.items():
    plt.plot(nvPM_mass, label=f"Engine {engine}")

plt.xlabel("Time (m)")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_nvpm_mass")
plt.title("P3T3-MEEM EI_nvPM_mass for different engines")
plt.legend(title="Engine")
plt.grid()
plt.savefig(f'figures/{flight}/future_engines/EI_nvpm_mass_engines.png', format='png')