import pandas as pd
import matplotlib.pyplot as plt



# Parameters
flight = 'malaga' # Replace with your flight identifier
engine_model = 'GTF'  # Replace with your engine model
SAF_values = [0, 20, 100]  # List of SAF configurations
nvPM_number_data = {}  # To store data for plotting
nvPM_mass_data = {}
fuel_flow_gsp_data = {}
nox_p3t3_data = {}
# Loop through SAF values and read corresponding files
for SAF in SAF_values:
    file_name = f'results/{flight}/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_A20N_full_WAR_0_0_0.csv'
    try:
        # Read the CSV file
        df = pd.read_csv(file_name)

        # Extract the nvPM_p3t3_meem column and store it
        fuel_flow_gsp_data[SAF] = df['fuel_flow_gsp']
        nox_p3t3_data[SAF] = df['EI_nox_p3t3']
        nvPM_number_data[SAF] = df['EI_nvpm_number_p3t3_meem']
        nvPM_mass_data[SAF] = df['EI_nvpm_mass_p3t3_meem']

    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except KeyError:
        print(f"'nvPM_p3t3_meem' column not found in file: {file_name}")

# Plot the data
plt.figure(figsize=(10, 6))
for SAF, nvPM_number in nvPM_number_data.items():
    plt.plot(nvPM_number, label=f"SAF {SAF}%")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_nvpm_number")
plt.title("P3T3-MEEM nvPM number for different SAF blends")
plt.legend(title="SAF blends")
plt.grid()
plt.savefig(f'figures/{flight}/saf_effect/EI_nvpm_number_SAF.png', format='png')

plt.figure(figsize=(10, 6))
for SAF, nvPM_mass in nvPM_mass_data.items():
    plt.plot(nvPM_mass, label=f"SAF {SAF}%")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_nvpm_mass")
plt.title("P3T3-MEEM nvPM mass for different SAF blends")
plt.legend(title="SAF blends")
plt.grid()
plt.savefig(f'figures/{flight}/saf_effect/EI_nvpm_mass_SAF.png', format='png')

plt.figure(figsize=(10, 6))
for SAF, fuel_flow in fuel_flow_gsp_data.items():
    plt.plot(fuel_flow, label=f"SAF {SAF}%")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("Fuel flow [kg/s]")
plt.title("Fuel flow for different SAF blends")
plt.legend(title="SAF")
plt.grid()
plt.savefig(f'figures/{flight}/saf_effect/fuel_flow_saf.png', format='png')

plt.figure(figsize=(10, 6))
for SAF, nox in nox_p3t3_data.items():
    plt.plot(nox, label=f"SAF {SAF}%")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_NOx")
plt.title("P3T3 EI_NOx for different SAF blends")
plt.legend(title="SAF")
plt.grid()
plt.savefig(f'figures/{flight}/saf_effect/EI_nox_saf.png', format='png')