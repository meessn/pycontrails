import pandas as pd

# Load the emissions results CSV
results_df = pd.read_csv('results_main_simulations.csv')

# Map Land/Ocean categories from the provided table (updated bos_fll to ocean)
land_ocean_mapping = {
    'hel_kef': 'ocean',
    'dus_tos': 'land',
    'lhr_ist': 'land',
    'cts_tpe': 'ocean',
    'bos_fll': 'ocean',  # UPDATED TO OCEAN
    'sfo_dfw': 'land',
    'sin_maa': 'ocean',
    'gru_lim': 'land'
}

# Assign Land/Ocean/Mix category
results_df['land_ocean'] = results_df['trajectory'].map(land_ocean_mapping)

# Define metrics to compare
metrics_to_compare = [
    'fuel_kg_sum', 'ei_co2_conservative_sum', 'ei_co2_optimistic_sum',
    'ei_nox_sum',   'ei_nvpm_num_sum', 'co2_conservative_sum', 'co2_optimistic_sum', 'nox_sum', 'nvpm_num_sum'
]

# Helper function to calculate percentage changes and average by Land/Ocean
def calculate_land_ocean_changes(baseline_engine, baseline_saf=0):
    baseline_df = results_df[(results_df['engine'] == baseline_engine) & (results_df['saf_level'] == baseline_saf)]
    merged_df = results_df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

    for metric in metrics_to_compare:
        merged_df[f'{metric}_change'] = 100 * (merged_df[metric] - merged_df[f'{metric}_baseline']) / merged_df[f'{metric}_baseline']

    columns_to_drop = [col for col in merged_df.columns if '_baseline' in col]
    merged_df = merged_df.drop(columns=columns_to_drop)

    # Aggregate by Land/Ocean
    for category in ['land', 'ocean']:
        category_df = merged_df[merged_df['land_ocean'] == category]
        average_df = category_df.groupby(['engine', 'saf_level', 'water_injection'])[
            [f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
        average_df = average_df.round(1)
        baseline_label = baseline_engine.lower()
        average_df.to_csv(f'results_report/emissions/land_ocean/land_ocean_{category}_vs_{baseline_label}.csv', index=False)

# Calculate for both GTF1990 and GTF baselines
calculate_land_ocean_changes('GTF1990')
calculate_land_ocean_changes('GTF')

print("Saved all Land/Ocean CSVs successfully!")
