import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
results_df = pd.read_csv('results_main_simulations.csv')

# Default engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

engine_groups = {
    'GTF1990': {'marker': '^', 'color': 'tab:blue'},
    'GTF2000': {'marker': '^', 'color': 'tab:orange'},
    'GTF': {'marker': 'o', 'color': 'tab:green'},
    'GTF2035': {'marker': 's', 'color': 'tab:red'},
    'GTF2035_wi': {'marker': 'D', 'color': default_colors[4]}
}

saf_colors = {
    ('GTF2035', 0): 'tab:red',
    ('GTF2035', 20): 'tab:pink',
    ('GTF2035', 100): 'tab:grey',
    ('GTF2035_wi', 0): default_colors[4],
    ('GTF2035_wi', 20): 'tab:olive',
    ('GTF2035_wi', 100): 'tab:cyan'
}

# Scatter plot without SAF consideration
plt.figure(figsize=(10, 6))
for engine, group in engine_groups.items():
    if engine in ['GTF2035', 'GTF2035_wi']:
        subset = results_df[(results_df['engine'] == engine) & (results_df['saf_level'] == 0)]
    else:
        subset = results_df[results_df['engine'] == engine]
    plt.scatter(subset['climate_non_co2'], subset['co2_impact_cons_sum'],
                label=engine_display_names[engine], marker=group['marker'], color=group['color'])

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Climate Non-CO2 Impact")
plt.ylabel("Climate CO2 Impact (Conservative)")
plt.title("Climate Impact Scatter Plot (Without SAF Consideration)")
plt.legend()
plt.show()

# Scatter plot with SAF (Conservative)
plt.figure(figsize=(10, 6))
for (engine, saf_level), color in saf_colors.items():
    subset = results_df[(results_df['engine'] == engine) & (results_df['saf_level'] == saf_level)]
    plt.scatter(subset['climate_non_co2'], subset['co2_impact_cons_sum'],
                label=f"{engine_display_names[engine]} SAF {saf_level}%", marker=engine_groups[engine]['marker'], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Climate Non-CO2 Impact")
plt.ylabel("Climate CO2 Impact (Conservative)")
plt.title("Climate Impact Scatter Plot (With SAF Consideration - Conservative)")
plt.legend()
plt.show()

# Scatter plot with SAF (Optimistic)
plt.figure(figsize=(10, 6))
for (engine, saf_level), color in saf_colors.items():
    subset = results_df[(results_df['engine'] == engine) & (results_df['saf_level'] == saf_level)]
    plt.scatter(subset['climate_non_co2'], subset['co2_impact_opti_sum'],
                label=f"{engine_display_names[engine]} SAF {saf_level}%", marker=engine_groups[engine]['marker'], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Climate Non-CO2 Impact")
plt.ylabel("Climate CO2 Impact (Optimistic)")
plt.title("Climate Impact Scatter Plot (With SAF Consideration - Optimistic)")
plt.legend()
plt.show()
