import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the emissions results CSV
results_df = pd.read_csv('results_main_simulations.csv')
print(results_df.columns.tolist())
# ---- FIRST: GTF1990 as baseline ---- #
baseline_1990_df = results_df[results_df['engine'] == 'GTF1990']
merged_1990_df = results_df.merge(baseline_1990_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

metrics_to_compare = [
    'fuel_kg_sum', 'ei_co2_conservative_sum', 'ei_co2_optimistic_sum',
    'ei_nox_sum',   'ei_nvpm_num_sum', 'co2_conservative_sum', 'co2_optimistic_sum', 'nox_sum', 'nvpm_num_sum'
]

for metric in metrics_to_compare:
    merged_1990_df[f'{metric}_change'] = 100 * (merged_1990_df[metric] - merged_1990_df[f'{metric}_baseline']) / merged_1990_df[f'{metric}_baseline']

columns_to_drop = [col for col in merged_1990_df.columns if '_baseline' in col]
merged_1990_df = merged_1990_df.drop(columns=columns_to_drop)

average_1990_df = merged_1990_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
average_1990_df = average_1990_df.round(1)

# ---- SECOND: GTF as baseline ---- #
baseline_gtf_df = results_df[(results_df['engine'] == 'GTF') & (results_df['saf_level'] == 0)]
merged_gtf_df = results_df.merge(baseline_gtf_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

for metric in metrics_to_compare:
    merged_gtf_df[f'{metric}_change'] = 100 * (merged_gtf_df[metric] - merged_gtf_df[f'{metric}_baseline']) / merged_gtf_df[f'{metric}_baseline']

columns_to_drop = [col for col in merged_gtf_df.columns if '_baseline' in col]
merged_gtf_df = merged_gtf_df.drop(columns=columns_to_drop)

average_gtf_df = merged_gtf_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
average_gtf_df = average_gtf_df.round(1)

# Remove GTF1990 and GTF2000 from GTF comparison results
average_gtf_df = average_gtf_df[~average_gtf_df['engine'].isin(['GTF1990', 'GTF2000'])]

# ---- SAVE ---- #
average_1990_df.to_csv('results_report/emissions/all_emissions_changes_vs_GTF1990.csv', index=False)
average_gtf_df.to_csv('results_report/emissions/all_emissions_changes_vs_GTF.csv', index=False)

print("Saved both CSVs successfully!")
# Load CSV

"""GTF1990"""
df_1990 = pd.read_csv("results_report/emissions/all_emissions_changes_vs_GTF1990.csv")  # Adjust path if needed
# Engine names & colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF1990': {'color': 'tab:blue'},
    'GTF2000': {'color': 'tab:orange'},
    'GTF': {'color': 'tab:green'},
    'GTF2035': {'color': 'tab:red'},
    'GTF2035_wi': {'color': default_colors[4]}
}

saf_colors = {
    ('GTF2035', 0): 'tab:red',
    ('GTF2035', 20): 'tab:pink',
    ('GTF2035', 100): 'tab:grey',
    ('GTF2035_wi', 0): default_colors[4],
    ('GTF2035_wi', 20): 'tab:olive',
    ('GTF2035_wi', 100): 'tab:cyan'
}

# Function to format engine names correctly
def format_engine_name(engine, saf_level, water_injection):
    formatted_name = engine_display_names.get(engine, engine)
    if saf_level > 0:
        formatted_name += f"\n-{saf_level}"
    # if water_injection > 0:
    #     formatted_name += "WI"
    return formatted_name

# Get engine colors
def get_color(engine, saf_level):
    return saf_colors.get((engine, saf_level), engine_groups.get(engine, {}).get('color', 'black'))

# Process data
df_1990["Formatted Engine"] = df_1990.apply(lambda row: format_engine_name(row["engine"], row["saf_level"], row["water_injection"]), axis=1)
df_1990["Color"] = df_1990.apply(lambda row: get_color(row["engine"], row["saf_level"]), axis=1)

# Set baseline (CFM1990) to 100
df_1990["fuel_kg_sum_change"] += 100
df_1990["ei_nox_sum_change"] += 100
df_1990["nox_sum_change"] += 100
df_1990["ei_nvpm_num_sum_change"] += 100
df_1990["nvpm_num_sum_change"] += 100
df_1990["co2_conservative_sum_change"] += 100
df_1990["co2_optimistic_sum_change"] += 100
# Enforce correct sorting order
engine_order = [
    "CFM1990", "CFM2000", "GTF",
    "GTF2035", "GTF2035\n-20", "GTF2035\n-100",
    "GTF2035WI", "GTF2035WI\n-20", "GTF2035WI\n-100"
]
df_1990["SortOrder"] = df_1990["Formatted Engine"].apply(lambda x: engine_order.index(x) if x in engine_order else len(engine_order))
df_1990 = df_1990.sort_values(by="SortOrder").drop(columns=["SortOrder"])
x_labels = df_1990["Formatted Engine"]

### ---- PLOT 1: FUEL BURN ---- ###
plt.figure(figsize=(10, 6))
plt.bar(x_labels, df_1990["fuel_kg_sum_change"], color=df_1990["Color"], edgecolor="black")
plt.axhline(100, color="black", linestyle="--")
plt.ylabel("Relative Fuel Burn (%)")
plt.title("Fuel Burn Relative to CFM1990")
plt.xticks(rotation=0, ha="center")  # Ensuring horizontal labels
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('results_report/emissions/fuel_flow_comp_all_flights_gtf1990.png', format='png')

### ---- PLOT 2: NOX EMISSIONS ---- ###
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_1990))
width = 0.4

bars1 = ax.bar(x - width/2, df_1990["ei_nox_sum_change"], width=width, label="EI NOx", color=df_1990["Color"], edgecolor="black", hatch="//")
bars2 = ax.bar(x + width/2, df_1990["nox_sum_change"], width=width, label="NOx", color=df_1990["Color"], edgecolor="black")

ax.axhline(100, color="black", linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=0, ha="center")  # Ensuring horizontal labels
ax.set_ylabel("Relative NOx Emissions (%)")
ax.set_title("NOx Emissions Relative to CFM1990")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
# plt.show()
plt.savefig('results_report/emissions/nox_comp_all_flights_gtf1990.png', format='png')

### ---- PLOT 3: nvPM EMISSIONS ---- ###
fig, ax = plt.subplots(figsize=(10, 6))

bars3 = ax.bar(x - width/2, df_1990["ei_nvpm_num_sum_change"], width=width, label="EI nvPM Number", color=df_1990["Color"], edgecolor="black", hatch="..")
bars4 = ax.bar(x + width/2, df_1990["nvpm_num_sum_change"], width=width, label="nvPM Number", color=df_1990["Color"], edgecolor="black")

ax.axhline(100, color="black", linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=0, ha="center")  # Ensuring horizontal labels
ax.set_ylabel("Relative nvPM Emissions (%)")
ax.set_title("nvPM Emissions Relative to CFM1990")
from matplotlib.patches import Patch

# Custom legend patches
legend_patches = [
    Patch(facecolor="tab:blue", edgecolor="black", hatch="..", label="EI nvPM Number"),
    Patch(facecolor="tab:blue", edgecolor="black", label="nvPM Number"),
]

ax.legend(handles=legend_patches, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
# plt.show()
plt.savefig('results_report/emissions/nvpm_comp_all_flights_gtf1990.png', format='png')

"""PLOT CO2 emissions"""

fig, ax = plt.subplots(figsize=(10, 6))

bars5 = ax.bar(x - width/2, df_1990["co2_conservative_sum_change"], width=width, label="CO2 Conservative", color=df_1990["Color"], edgecolor="black", hatch="..")
bars6 = ax.bar(x + width/2, df_1990["co2_optimistic_sum_change"], width=width, label="CO2 Optimistic", color=df_1990["Color"], edgecolor="black")

ax.axhline(100, color="black", linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=0, ha="center")  # Ensuring horizontal labels
ax.set_ylabel("Relative CO2 Emissions (%)")
ax.set_title("CO2 Emissions Relative to CFM1990")

# Custom legend patches with smaller hatch pattern
legend_patches = [
    Patch(facecolor="tab:green", edgecolor="black", hatch="..", label="CO2 Conservative"),
    Patch(facecolor="tab:green", edgecolor="black", label="CO2 Optimistic"),
]

ax.legend(handles=legend_patches, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('results_report/emissions/co2_comp_all_flights_gtf1990.png', format='png')
"""GTF"""


# Load new CSV file with GTF as baseline
df_gtf = pd.read_csv("results_report/emissions/all_emissions_changes_vs_GTF.csv")

# Engine names & colors
engine_display_names = {
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF': {'color': 'tab:green'},
    'GTF2035': {'color': 'tab:red'},
    'GTF2035_wi': {'color': default_colors[4]}
}

saf_colors = {
    ('GTF2035', 0): 'tab:red',
    ('GTF2035', 20): 'tab:pink',
    ('GTF2035', 100): 'tab:grey',
    ('GTF2035_wi', 0): default_colors[4],
    ('GTF2035_wi', 20): 'tab:olive',
    ('GTF2035_wi', 100): 'tab:cyan'
}

# Function to format engine names correctly
def format_engine_name(engine, saf_level, water_injection):
    formatted_name = engine_display_names.get(engine, engine)
    if saf_level > 0:
        formatted_name += f"\n-{saf_level}"  # Add newline before SAF level
    return formatted_name

# Get engine colors
def get_color(engine, saf_level):
    return saf_colors.get((engine, saf_level), engine_groups.get(engine, {}).get('color', 'black'))

# Process data
df_gtf["Formatted Engine"] = df_gtf.apply(lambda row: format_engine_name(row["engine"], row["saf_level"], row["water_injection"]), axis=1)
df_gtf["Color"] = df_gtf.apply(lambda row: get_color(row["engine"], row["saf_level"]), axis=1)

# Set baseline (GTF) to 100
df_gtf["fuel_kg_sum_change"] += 100
df_gtf["ei_nox_sum_change"] += 100
df_gtf["nox_sum_change"] += 100
df_gtf["ei_nvpm_num_sum_change"] += 100
df_gtf["nvpm_num_sum_change"] += 100
df_gtf["co2_conservative_sum_change"] += 100
df_gtf["co2_optimistic_sum_change"] += 100

# Enforce correct sorting order (excluding CFM1990 and CFM2000)
engine_order = [
    "GTF",
    "GTF2035", "GTF2035\n-20", "GTF2035\n-100",
    "GTF2035WI", "GTF2035WI\n-20", "GTF2035WI\n-100"
]
df_gtf["SortOrder"] = df_gtf["Formatted Engine"].apply(lambda x: engine_order.index(x) if x in engine_order else len(engine_order))
df_gtf = df_gtf.sort_values(by="SortOrder").drop(columns=["SortOrder"])
x_labels = df_gtf["Formatted Engine"]

### ---- PLOT 1: FUEL BURN ---- ###
plt.figure(figsize=(10, 6))
plt.bar(x_labels, df_gtf["fuel_kg_sum_change"], color=df_gtf["Color"], edgecolor="black")
plt.axhline(100, color="black", linestyle="--")
plt.ylabel("Relative Fuel Burn (%)")
plt.title("Fuel Burn Relative to GTF")
plt.xticks(rotation=0, ha="center")  # Ensuring horizontal labels
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
# plt.show()
plt.savefig('results_report/emissions/fuel_flow_comp_all_flights_gtf.png', format='png')
### ---- PLOT 2: NOX EMISSIONS ---- ###
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_gtf))
width = 0.4

bars1 = ax.bar(x - width/2, df_gtf["ei_nox_sum_change"], width=width, label="EI NOx", color=df_gtf["Color"], edgecolor="black", hatch="//")
bars2 = ax.bar(x + width/2, df_gtf["nox_sum_change"], width=width, label="NOx", color=df_gtf["Color"], edgecolor="black")

ax.axhline(100, color="black", linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=0, ha="center")  # Ensuring horizontal labels
ax.set_ylabel("Relative NOx Emissions (%)")
ax.set_title("NOx Emissions Relative to GTF")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
# plt.show()
plt.savefig('results_report/emissions/nox_comp_all_flights_gtf.png', format='png')
### ---- PLOT 3: nvPM EMISSIONS ---- ###
fig, ax = plt.subplots(figsize=(10, 6))

bars3 = ax.bar(x - width/2, df_gtf["ei_nvpm_num_sum_change"], width=width, label="EI nvPM Number", color=df_gtf["Color"], edgecolor="black", hatch="..")
bars4 = ax.bar(x + width/2, df_gtf["nvpm_num_sum_change"], width=width, label="nvPM Number", color=df_gtf["Color"], edgecolor="black")

ax.axhline(100, color="black", linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=0, ha="center")  # Ensuring horizontal labels
ax.set_ylabel("Relative nvPM Emissions (%)")
ax.set_title("nvPM Emissions Relative to GTF")

# Custom legend patches with smaller hatch pattern
legend_patches = [
    Patch(facecolor="tab:green", edgecolor="black", hatch="..", label="EI nvPM Number"),
    Patch(facecolor="tab:green", edgecolor="black", label="nvPM Number"),
]

ax.legend(handles=legend_patches, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('results_report/emissions/nvpm_comp_all_flights_gtf.png', format='png')
# plt.show()

"""PLOT CO2 emissions"""

fig, ax = plt.subplots(figsize=(10, 6))

bars5 = ax.bar(x - width/2, df_gtf["co2_conservative_sum_change"], width=width, label="CO2 Conservative", color=df_gtf["Color"], edgecolor="black", hatch="..")
bars6 = ax.bar(x + width/2, df_gtf["co2_optimistic_sum_change"], width=width, label="CO2 Optimistic", color=df_gtf["Color"], edgecolor="black")

ax.axhline(100, color="black", linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=0, ha="center")  # Ensuring horizontal labels
ax.set_ylabel("Relative CO2 Emissions (%)")
ax.set_title("CO2 Emissions Relative to GTF")

# Custom legend patches with smaller hatch pattern
legend_patches = [
    Patch(facecolor="tab:green", edgecolor="black", hatch="..", label="CO2 Conservative"),
    Patch(facecolor="tab:green", edgecolor="black", label="CO2 Optimistic"),
]

ax.legend(handles=legend_patches, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('results_report/emissions/co2_comp_all_flights_gtf.png', format='png')
plt.show()