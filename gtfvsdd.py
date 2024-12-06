import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_nvpm = 'gtfvsdd/edb-emissions-databank_v30__web_ python_analysis_nvpm.csv'
file_gaseous = 'gtfvsdd/edb-emissions-databank_v30__web_ python_analysis_gaseous.csv'

nvpm_data = pd.read_csv(file_nvpm, delimiter=';', decimal=',')
gaseous_data = pd.read_csv(file_gaseous, delimiter=';', decimal=',')

# Function to classify engine types
def classify_engine(engine_id):
    if "CFM56" in engine_id:
        return "CFM56"
    elif "PW" in engine_id or "PW1100G" in engine_id:
        return "PW GTF"
    elif "LEAP" in engine_id:
        return "LEAP"
    else:
        return "Other"

# Add engine type column
nvpm_data['EngineType'] = nvpm_data['Engine Identification'].apply(classify_engine)
gaseous_data['EngineType'] = gaseous_data['Engine Identification'].apply(classify_engine)

# Define stages and colors/markers
stages = ['T/O', 'C/O', 'App', 'Idle']
colors = {'CFM56': 'red', 'PW GTF': 'green', 'LEAP': 'orange'}
markers = {'CFM56': 'o', 'PW GTF': 's', 'LEAP': '^'}

# Function to calculate actual NOx emissions (g/s) in gaseous_data
def calculate_nox_actual_emissions(data, stage):
    fuel_flow_col = f'Fuel Flow {stage} (kg/sec)'
    nox_ei_col = f'NOx EI {stage} (g/kg)'
    if fuel_flow_col in data.columns and nox_ei_col in data.columns:
        data[f'NOx_Actual_{stage} (g/s)'] = data[fuel_flow_col] * data[nox_ei_col]

# Function to calculate actual nvPM emissions (#/s) in nvpm_data
def calculate_nvpm_actual_emissions(data, stage):
    fuel_flow_col = f'Fuel Flow {stage} (kg/sec)'
    nvpm_einum_col = f'nvPM EInum_SL {stage} (#/kg)'
    if fuel_flow_col in data.columns and nvpm_einum_col in data.columns:
        data[f'nvPM_Actual_{stage} (#/s)'] = data[fuel_flow_col] * data[nvpm_einum_col]

# Calculate emissions for all stages
for stage in stages:
    calculate_nox_actual_emissions(gaseous_data, stage)
    calculate_nvpm_actual_emissions(nvpm_data, stage)

# Function to create subplots for all stages
def create_subplot_figure(plot_type, y_data_template, ylabel, title_template, data, filename):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, stage in enumerate(stages):
        ax = axs[i]
        for engine_type in ['CFM56', 'PW GTF', 'LEAP']:
            filtered_data = data[data['EngineType'] == engine_type]
            y_data = y_data_template.format(stage=stage)
            ax.scatter(filtered_data['Rated Thrust (kN)'], filtered_data[y_data],
                       color=colors[engine_type], marker=markers[engine_type],
                       label=engine_type, alpha=0.7)
        ax.set_title(f'{stage}')
        ax.set_xlabel('Rated Thrust (kN)')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if i == 0:  # Only include legend in the first subplot
            ax.legend()

    fig.suptitle(title_template, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close()

# Function to create subplots with a logarithmic y-axis for all stages
def create_subplot_figure_log(plot_type, y_data_template, ylabel, title_template, data, filename):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, stage in enumerate(stages):
        ax = axs[i]
        for engine_type in ['CFM56', 'PW GTF', 'LEAP']:
            filtered_data = data[data['EngineType'] == engine_type]
            y_data = y_data_template.format(stage=stage)
            ax.scatter(filtered_data['Rated Thrust (kN)'], filtered_data[y_data],
                       color=colors[engine_type], marker=markers[engine_type],
                       label=engine_type, alpha=0.7)
        ax.set_yscale('log')  # Set logarithmic scale for y-axis
        ax.set_title(f'{stage}')
        ax.set_xlabel('Rated Thrust (kN)')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if i == 0:  # Only include legend in the first subplot
            ax.legend()

    fig.suptitle(title_template, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close()

# Create combined plots for each plot type
create_subplot_figure(
    'Fuel Flow vs Rated Thrust',
    'Fuel Flow {stage} (kg/sec)',
    'Fuel Flow (kg/s)',
    'Fuel Flow vs Rated Thrust',
    gaseous_data,
    'gtfvsdd/fuel_flow_vs_thrust_combined.png'
)

create_subplot_figure(
    'NOx EI vs Rated Thrust',
    'NOx EI {stage} (g/kg)',
    'NOx EI (g/kg)',
    'NOx EI vs Rated Thrust',
    gaseous_data,
    'gtfvsdd/nox_ei_vs_thrust_combined.png'
)

create_subplot_figure(
    'nvPM EI vs Rated Thrust',
    'nvPM EInum_SL {stage} (#/kg)',
    'nvPM EI (#/kg)',
    'nvPM EI vs Rated Thrust',
    nvpm_data,
    'gtfvsdd/nvpm_ei_vs_thrust_combined.png'
)

create_subplot_figure(
    'NOx Actual Emissions vs Rated Thrust',
    'NOx_Actual_{stage} (g/s)',
    'NOx Emissions (g/s)',
    'NOx Actual Emissions vs Rated Thrust',
    gaseous_data,
    'gtfvsdd/nox_actual_vs_thrust_combined.png'
)

create_subplot_figure(
    'nvPM Actual Emissions vs Rated Thrust',
    'nvPM_Actual_{stage} (#/s)',
    'nvPM Emissions (#/s)',
    'nvPM Actual Emissions vs Rated Thrust',
    nvpm_data,
    'gtfvsdd/nvpm_actual_vs_thrust_combined.png'
)

create_subplot_figure_log(
    'nvPM EI vs Rated Thrust (Log)',
    'nvPM EInum_SL {stage} (#/kg)',
    'nvPM EI (#/kg)',
    'nvPM EI vs Rated Thrust (Logarithmic)',
    nvpm_data,
    'gtfvsdd/nvpm_ei_vs_thrust_combined_log.png'
)

create_subplot_figure_log(
    'nvPM Actual Emissions vs Rated Thrust (Log)',
    'nvPM_Actual_{stage} (#/s)',
    'nvPM Emissions (#/s)',
    'nvPM Actual Emissions vs Rated Thrust (Logarithmic)',
    nvpm_data,
    'gtfvsdd/nvpm_actual_vs_thrust_combined_log.png'
)

print("Combined scatter plot figures have been generated and saved.")