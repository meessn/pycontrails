import pandas as pd
import matplotlib.pyplot as plt



# Parameters
flight = 'malaga' # Replace with your flight identifier
engine_model = 'GTF2035'  # Replace with your engine model
SAF_values = [0, 20, 100]  # List of SAF configurations
nvPM_number_data = {}  # To store data for plotting
nvPM_mass_data = {}
fuel_flow_gsp_data = {}
nox_p3t3_data = {}
# Loop through SAF values and read corresponding files
for SAF in SAF_values:
    file_name = f'../main_results_figures/results/malaga/malaga/emissions/{engine_model}_SAF_{SAF}_A20N_full_WAR_0_0_0.csv'
    try:
        # Read the CSV file
        df = pd.read_csv(file_name)

        # Extract the nvPM_p3t3_meem column and store it
        fuel_flow_gsp_data[SAF] = df['fuel_flow_gsp']
        nox_p3t3_data[SAF] = df['ei_nox_p3t3']
        nvPM_number_data[SAF] = df['ei_nvpm_number_p3t3_meem']
        nvPM_mass_data[SAF] = df['ei_nvpm_mass_p3t3_meem']

    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except KeyError:
        print(f"'nvPM_p3t3_meem' column not found in file: {file_name}")

# Plot the data
plt.figure(figsize=(10, 6))
for SAF, nvPM_number in nvPM_number_data.items():
    plt.plot(nvPM_number, label=f"SAF {SAF}%")

plt.xlabel("Time (minutes)")  # Or replace "Time (minutes)" with a relevant x-axis label if needed
plt.ylabel(f'$EI_{{\\mathrm{{nvPM,number}}}}$ (# / kg Fuel)')
plt.title(f"$EI_{{\\mathrm{{nvPM,number}}}}$ for different SAF blends")
plt.legend(title="SAF blends")
plt.grid()
plt.savefig(f'../results_report/performance_emissions_chapter/saf_effect/EI_nvpm_number_SAF_{engine_model}.png', format='png')

plt.figure(figsize=(10, 6))
for SAF, nvPM_mass in nvPM_mass_data.items():
    plt.plot(nvPM_mass, label=f"SAF {SAF}%")

plt.xlabel("Time (minutes)")  # Or replace "Time (minutes)" with a relevant x-axis label if needed
plt.ylabel(f'$EI_{{\\mathrm{{nvPM,mass}}}}$ (mg / kg Fuel)')
plt.title(f"$EI_{{\\mathrm{{nvPM,mass}}}}$ for different SAF blends")
plt.legend(title="SAF blends")
plt.grid()
plt.savefig(f'../results_report/performance_emissions_chapter/saf_effect/EI_nvpm_mass_SAF_{engine_model}.png', format='png')

plt.figure(figsize=(10, 6))
for SAF, fuel_flow in fuel_flow_gsp_data.items():
    plt.plot(fuel_flow, label=f"SAF {SAF}%")

plt.xlabel("Time (minutes)")  # Or replace "Time (minutes)" with a relevant x-axis label if needed
plt.ylabel("Fuel flow (kg/s)")
plt.title("Fuel flow for different SAF blends")
plt.legend(title="SAF")
plt.grid()
plt.savefig(f'../results_report/performance_emissions_chapter/saf_effect/fuel_flow_saf_{engine_model}.png', format='png')

plt.figure(figsize=(10, 6))
for SAF, nox in nox_p3t3_data.items():
    plt.plot(nox, label=f"SAF {SAF}%")

plt.xlabel("Time (minutes)")  # Or replace "Time (minutes)" with a relevant x-axis label if needed
plt.ylabel(f"$EI_{{\\mathrm{{NOx}}}}$ (g / kg fuel)")
plt.title(f"$EI_{{\\mathrm{{NOx}}}}$ for different SAF blends")
plt.legend(title="SAF")
plt.grid()
plt.savefig(f'../results_report/performance_emissions_chapter/saf_effect/EI_nox_saf_{engine_model}.png', format='png')