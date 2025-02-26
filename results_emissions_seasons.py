import pandas as pd

# Load the emissions results CSV
results_df = pd.read_csv('results_main_simulations.csv')

# Map seasons for GRU -> LIM and SIN -> MAA
season_mapping = {
    'gru_lim': {'2023-02-06': 'summer', '2023-05-05': 'autumn', '2023-08-06': 'winter', '2023-11-06': 'spring'},
    'sin_maa': {'2023-02-06': 'winter', '2023-05-05': 'summer', '2023-08-06': 'monsoon', '2023-11-06': 'postmonsoon'}
}

# Create a new column 'season_adjusted'
def map_season(row):
    if row['trajectory'] in season_mapping:
        return season_mapping[row['trajectory']].get(row['season'], row['season'])
    # Map default seasons for other flights
    season_default = {
        '2023-02-06': 'winter',
        '2023-05-05': 'spring',
        '2023-08-06': 'summer',
        '2023-11-06': 'autumn'
    }
    return season_default.get(row['season'], row['season'])


results_df['season_adjusted'] = results_df.apply(map_season, axis=1)

# Define metrics to compare
metrics_to_compare = [
    'fuel_kg_sum', 'ei_co2_conservative_sum', 'ei_co2_optimistic_sum',
    'ei_nox_sum',   'ei_nvpm_num_sum', 'co2_conservative_sum', 'co2_optimistic_sum', 'nox_sum', 'nvpm_num_sum'
]

# Helper function to calculate percentage changes and average by season
def calculate_seasonal_changes(baseline_engine, baseline_saf=0):
    baseline_df = results_df[(results_df['engine'] == baseline_engine) & (results_df['saf_level'] == baseline_saf)]
    merged_df = results_df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

    for metric in metrics_to_compare:
        merged_df[f'{metric}_change'] = 100 * (merged_df[metric] - merged_df[f'{metric}_baseline']) / merged_df[f'{metric}_baseline']

    columns_to_drop = [col for col in merged_df.columns if '_baseline' in col]
    merged_df = merged_df.drop(columns=columns_to_drop)

    # General seasons
    general_seasons_df = merged_df[merged_df['trajectory'] != 'sin_maa']
    for season in ['winter', 'spring', 'summer', 'autumn']:
        season_df = general_seasons_df[general_seasons_df['season_adjusted'] == season]
        average_df = season_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
        average_df = average_df.round(1)
        baseline_label = baseline_engine.lower()
        average_df.to_csv(f'results_report/emissions/seasonal/season_{season}_vs_{baseline_label}.csv', index=False)

    # SIN -> MAA specific seasons
    sinmaa_df = merged_df[merged_df['trajectory'] == 'sin_maa']
    for season in ['winter', 'summer', 'monsoon', 'postmonsoon']:
        season_df = sinmaa_df[sinmaa_df['season_adjusted'] == season]
        average_df = season_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
        average_df = average_df.round(1)
        baseline_label = baseline_engine.lower()
        average_df.to_csv(f'results_report/emissions/seasonal/season_sinmaa_{season}_vs_{baseline_label}.csv', index=False)

# Calculate for both GTF1990 and GTF baselines
calculate_seasonal_changes('GTF1990')
calculate_seasonal_changes('GTF')
print(results_df[['trajectory', 'season', 'season_adjusted']].drop_duplicates())

print("Saved all seasonal CSVs successfully!")


