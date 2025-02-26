import pandas as pd

# Load the emissions results CSV
results_df = pd.read_csv('results_main_simulations.csv')

# Map trajectories to regions
region_mapping = {
    'hel_kef': 'upper_temperate',
    'dus_tos': 'upper_temperate',
    'lhr_ist': 'temperate',
    'cts_tpe': 'temperate',
    'bos_fll': 'temperate',
    'sfo_dfw': 'temperate',
    'sin_maa': 'tropical',
    'gru_lim': 'tropical'
}

# Add region column
results_df['region'] = results_df['trajectory'].map(region_mapping)

metrics_to_compare = [
    'fuel_kg_sum', 'ei_co2_conservative_sum', 'ei_co2_optimistic_sum',
    'ei_nox_sum',   'ei_nvpm_num_sum', 'co2_conservative_sum', 'co2_optimistic_sum', 'nox_sum', 'nvpm_num_sum'
]

# Helper function to calculate percentage changes and average by region
def calculate_regional_changes(baseline_engine, baseline_saf=0):
    baseline_df = results_df[(results_df['engine'] == baseline_engine) & (results_df['saf_level'] == baseline_saf)]
    merged_df = results_df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

    for metric in metrics_to_compare:
        merged_df[f'{metric}_change'] = 100 * (merged_df[metric] - merged_df[f'{metric}_baseline']) / merged_df[f'{metric}_baseline']

    columns_to_drop = [col for col in merged_df.columns if '_baseline' in col]
    merged_df = merged_df.drop(columns=columns_to_drop)

    # Aggregate by region
    for region in ['upper_temperate', 'temperate', 'tropical']:
        region_df = merged_df[merged_df['region'] == region]
        average_df = region_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
        average_df = average_df.round(1)
        baseline_label = baseline_engine.lower()
        average_df.to_csv(f'results_report/emissions/regions/region_{region}_vs_{baseline_label}.csv', index=False)

# Calculate for both GTF1990 and GTF baselines
calculate_regional_changes('GTF1990')
calculate_regional_changes('GTF')

print("Saved all regional CSVs successfully!")
