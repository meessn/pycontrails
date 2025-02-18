import pandas as pd

# Load the emissions results CSV
results_df = pd.read_csv('results_main_simulations.csv')
print(results_df.columns.tolist())
# ---- FIRST: GTF1990 as baseline ---- #
baseline_1990_df = results_df[results_df['engine'] == 'GTF1990']
merged_1990_df = results_df.merge(baseline_1990_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

metrics_to_compare = [
    'fuel_kg_sum', 'energy_sum', 'ei_co2_conservative_sum', 'ei_co2_optimistic_sum',
    'ei_nox_sum',  'ei_nvpm_mass_sum',  'ei_nvpm_num_sum'
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

