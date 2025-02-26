import pandas as pd

# Load the emissions results CSV
results_df = pd.read_csv('results_main_simulations.csv')

# ---- COMMON CONFIG ---- #
metrics_to_compare = [
    'fuel_kg_sum', 'ei_co2_conservative_sum', 'ei_co2_optimistic_sum',
    'ei_nox_sum',   'ei_nvpm_num_sum', 'co2_conservative_sum', 'co2_optimistic_sum', 'nox_sum', 'nvpm_num_sum'
]

def calculate_average_changes(results_df, baseline_engine, baseline_saf=0, diurnal_filter=None):
    # Apply diurnal filtering if needed
    if diurnal_filter:
        results_df = results_df[results_df['diurnal'] == diurnal_filter]

    if baseline_engine == 'GTF1990':
        baseline_df = results_df[results_df['engine'] == 'GTF1990']
    elif baseline_engine == 'GTF':
        baseline_df = results_df[(results_df['engine'] == 'GTF') & (results_df['saf_level'] == baseline_saf)]
    else:
        raise ValueError("Unsupported baseline engine!")

    merged_df = results_df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

    for metric in metrics_to_compare:
        merged_df[f'{metric}_change'] = 100 * (merged_df[metric] - merged_df[f'{metric}_baseline']) / merged_df[f'{metric}_baseline']

    columns_to_drop = [col for col in merged_df.columns if '_baseline' in col]
    merged_df = merged_df.drop(columns=columns_to_drop)

    average_df = merged_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
    average_df = average_df.round(1)

    # Remove GTF1990 and GTF2000 when using GTF as baseline
    if baseline_engine == 'GTF':
        average_df = average_df[~average_df['engine'].isin(['GTF1990', 'GTF2000'])]

    return average_df


# ---- CALCULATE AND SAVE ---- #

# 1. GTF1990 as baseline
average_1990_day_df = calculate_average_changes(results_df, 'GTF1990', diurnal_filter='daytime')
average_1990_night_df = calculate_average_changes(results_df, 'GTF1990', diurnal_filter='nighttime')

average_1990_day_df.to_csv('results_report/emissions/all_emissions_changes_vs_GTF1990_daytime.csv', index=False)
average_1990_night_df.to_csv('results_report/emissions/all_emissions_changes_vs_GTF1990_nighttime.csv', index=False)

# 2. GTF as baseline
average_gtf_day_df = calculate_average_changes(results_df, 'GTF', diurnal_filter='daytime')
average_gtf_night_df = calculate_average_changes(results_df, 'GTF', diurnal_filter='nighttime')

average_gtf_day_df.to_csv('results_report/emissions/all_emissions_changes_vs_GTF_daytime.csv', index=False)
average_gtf_night_df.to_csv('results_report/emissions/all_emissions_changes_vs_GTF_nighttime.csv', index=False)

print("Saved daytime and nighttime CSVs successfully!")
