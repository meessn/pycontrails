import pandas as pd
import matplotlib.pyplot as plt

"""FLIGHT PARAMETERS"""
flight = 'malaga'

# File paths and their corresponding legends
file_info = [
    {
        "file": "../main_results_figures/results/malaga/malaga/emissions/GTF2035_SAF_0_A20N_full_WAR_0_0_0.csv",
        "legend": "No Water Injection"
    },
    {
        "file": "../main_results_figures/results/malaga/malaga/emissions/GTF2035_SAF_0_A20N_full_WAR_2_5_2_5_2_5.csv",
        "legend": "Water Injection Steam 2.5% 2.5% 2.5%"
    },
    {
        "file": "../main_results_figures/results/malaga/malaga/emissions/GTF2035_SAF_0_A20N_full_WAR_5_5_5.csv",
        "legend": "Water Injection Steam 5% 5% 5%"
    },
    {
        "file": "../main_results_figures/results/malaga/malaga/emissions/GTF2035_SAF_0_A20N_full_WAR_10_10_10.csv",
        "legend": "Water Injection Steam 10% 10% 10%"
    }

]

# Data storage
fuel_flow_gsp_data = {}
nox_p3t3_data = {}
nvpm_number_data = {}

# Loop through file information to extract data
for info in file_info:
    try:
        # Read the CSV file
        df = pd.read_csv(info["file"])
        legend = info["legend"]

        # Extract the fuel flow, EI_NOx, and EI_nvPM_number data
        fuel_flow_gsp_data[legend] = df['fuel_flow_gsp']
        nox_p3t3_data[legend] = df['ei_nox_p3t3']
        nvpm_number_data[legend] = df['ei_nvpm_number_p3t3_meem']
    except FileNotFoundError:
        print(f"File not found: {info['file']}")
    except KeyError as e:
        print(f"Column {e} not found in file: {info['file']}")

# Plot fuel flow
plt.figure(figsize=(10, 6))
for legend, fuel in fuel_flow_gsp_data.items():
    plt.plot(fuel, label=legend)

plt.xlabel("Time (minutes)")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("Fuel flow (kg/s)")
plt.title("Fuel Flow")
plt.legend(title="Configuration")
plt.grid()
plt.savefig(f'../results_report/performance_emissions_chapter/water_injection/amounts/fuel_flow_water_injection.png', format='png')
plt.close()

# Plot EI_NOx
plt.figure(figsize=(10, 6))
for legend, nox in nox_p3t3_data.items():
    plt.plot(nox, label=legend)

plt.xlabel("Time (minutes)")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel(f"$EI_{{\\mathrm{{NOx}}}}$ (g / kg fuel)")
plt.title(f"$EI_{{\\mathrm{{NOx}}}}$")
plt.legend(title="Configuration")
plt.grid()
plt.savefig(f'../results_report/performance_emissions_chapter/water_injection/amounts/EI_nox_water_injection.png', format='png')
plt.close()

# Plot EI_nvpm_number_p3t3_meem
plt.figure(figsize=(10, 6))
for legend, nvpm in nvpm_number_data.items():
    plt.plot(nvpm, label=legend)

plt.xlabel("Time (minutes)")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel(f'$EI_{{\\mathrm{{nvPM,number}}}}$ (# / kg Fuel)')
plt.title(f"$EI_{{\\mathrm{{nvPM,number}}}}$")
plt.legend(title="Configuration")
plt.grid()
plt.savefig(f'../results_report/performance_emissions_chapter/water_injection/amounts/EI_nvpm_number_water_injection.png', format='png')
plt.close()

