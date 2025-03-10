import pandas as pd
import matplotlib.pyplot as plt

# Parameters
flight = 'malaga'  # Replace with your flight identifier
engine_model = ['GTF', 'GTF_corr'] #['GTF1990', 'GTF2000', 'GTF', 'GTF2035']  # Replace with your engine model
SAF = 0  # SAF configuration
fuel_flow_gsp_data = {}
fuel_flow_pycontrails_data = {}
nox_p3t3_data = {}
nvPM_number_data = {}
nvPM_mass_data = {}
nox_pycontrails = {}
nvPM_number_py = {}
nvPM_mass_py = {}
nvPM_number_meem = {}
nvPM_mass_meem = {}
# Loop through engine models and read corresponding files
for engine in engine_model:
    file_name = f'../main_results_figures/results/{flight}/{flight}/emissions/{engine}_SAF_0_A20N_full_WAR_0_0_0.csv'
    try:
        df = pd.read_csv(file_name)

        fuel_flow_gsp_data[engine] = df['fuel_flow_gsp']
        fuel_flow_pycontrails_data[engine] = df['fuel_flow_per_engine']
        nox_p3t3_data[engine] = df['ei_nox_p3t3']
        nox_pycontrails[engine] = df['ei_nox_py']
        nvPM_number_data[engine] = df['ei_nvpm_number_p3t3_meem']
        nvPM_mass_data[engine] = df['ei_nvpm_mass_p3t3_meem']
        nvPM_number_py[engine] = df['ei_nvpm_number_py']
        nvPM_mass_py[engine] = df['ei_nvpm_mass_py']
        nvPM_number_meem[engine] = df['ei_number_meem']
        nvPM_mass_meem[engine] = df['ei_mass_meem']

    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except KeyError as e:
        print(f"Column {e} not found in file: {file_name}")

# Engine name mapping for legend
engine_legend_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF_corr': 'GTF Improved GSP',
    'GTF2035': 'GTF2035'
}

# Plot Fuel Flow
plt.figure(figsize=(10, 6))
# Plot PyContrails fuel flow only once
if 'GTF' in fuel_flow_pycontrails_data:
    plt.plot(fuel_flow_pycontrails_data['GTF'], label="pycontrails")
    print(fuel_flow_pycontrails_data['GTF'])
for engine, fuel in fuel_flow_gsp_data.items():
    plt.plot(fuel, label=f"{engine_legend_names.get(engine, engine)}")

plt.xlabel("Time (m)")
plt.ylabel("Fuel flow [kg/s]")
plt.title("Fuel flow for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig(f'../main_results_figures/figures/{flight}/{flight}/emissions/EI_fuel_flow_engines_1990_2000.png', format='png')

# Plot EI NOx
plt.figure(figsize=(10, 6))
if 'GTF' in nox_pycontrails:
    plt.plot(nox_pycontrails['GTF'], label="pycontrails")
for engine, nox in nox_p3t3_data.items():
    plt.plot(nox, label=f"{engine_legend_names.get(engine, engine)}")

plt.xlabel("Time (m)")
plt.ylabel("EI_NOx")
plt.title("P3T3 EI_NOx for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig(f'../main_results_figures/figures/{flight}/{flight}/emissions/EI_nox_engines_1990_2000.png', format='png')

# Plot EI nvPM Number
plt.figure(figsize=(10, 6))
if 'GTF' in nvPM_number_py:
    plt.plot(nvPM_number_py['GTF'], label="pycontrails")
    plt.plot(nvPM_number_meem['GTF'], label="MEEM")
for engine, nvPM_number in nvPM_number_data.items():
    plt.plot(nvPM_number, label=f"{engine_legend_names.get(engine, engine)}")

plt.xlabel("Time (m)")
plt.ylabel("EI_nvpm_number")
plt.title("P3T3-MEEM EI_nvPM_number for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig(f'../main_results_figures/figures/{flight}/{flight}/emissions/EI_nvpm_number_engines_1990_2000.png', format='png')

# Plot EI nvPM Mass
plt.figure(figsize=(10, 6))
if 'GTF' in nvPM_mass_py:
    plt.plot(nvPM_mass_py['GTF'], label="pycontrails")
    plt.plot(nvPM_mass_meem['GTF'], label="MEEM")
for engine, nvPM_mass in nvPM_mass_data.items():
    plt.plot(nvPM_mass, label=f"{engine_legend_names.get(engine, engine)}")

plt.xlabel("Time (m)")
plt.ylabel("EI_nvpm_mass")
plt.title("P3T3-MEEM EI_nvPM_mass for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig(f'../main_results_figures/figures/{flight}/{flight}/emissions/EI_nvpm_mass_engines_1990_2000.png', format='png')

plt.show()
