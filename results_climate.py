import pandas as pd
import numpy as np

# Load the results CSV
results_df = pd.read_csv('results_main_simulations.csv')

# Filter out special cases
excluded_categories = ['special cases']
filtered_df = results_df[~results_df['contrail_category'].isin(excluded_categories)]

# Split into no contrail, warming, cooling
no_contrail_df = filtered_df[filtered_df['contrail_category'] == 'no contrail']
warming_df = filtered_df[filtered_df['contrail_category'] == 'warming']
cooling_df = filtered_df[filtered_df['contrail_category'] == 'cooling']

# Baseline: GTF1990, saf_level = 0
baseline_df = filtered_df[(filtered_df['engine'] == 'GTF1990') & (filtered_df['saf_level'] == 0)]

# Define metrics for each category
common_metrics = [
    'nox_impact_sum', 'co2_impact_cons_sum', 'co2_impact_opti_sum', 'h2o_impact_sum',
    'climate_non_co2', 'climate_total_cons_sum', 'climate_total_opti_sum'
]
contrail_metrics = ['contrail_atr20_cocip_sum', 'contrail_atr20_accf_sum'] + common_metrics


# Helper function to calculate percentage changes
def calculate_percentage_changes(df, metrics):
    merged_df = df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))
    problematic_rows = []

    for metric in metrics:
        baseline_metric = merged_df[f'{metric}_baseline']

        # Avoid division by zero by marking problematic rows
        zero_baseline_mask = baseline_metric == 0

        if zero_baseline_mask.any():
            problematic_entries = merged_df.loc[zero_baseline_mask, ['trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection']]
            problematic_rows.extend(problematic_entries.to_dict('records'))

        # Perform the percentage change calculation
        merged_df[f'{metric}_change'] = np.where(
            zero_baseline_mask,
            np.inf,  # or np.nan if you prefer
            100 * (merged_df[metric] - baseline_metric) / baseline_metric
        )

    # Collect entries with any 'inf' values in the change columns
    inf_rows = merged_df.loc[merged_df[[f'{metric}_change' for metric in metrics]].isin([np.inf, -np.inf]).any(axis=1),
                             ['trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection']]

    if not inf_rows.empty:
        print("Rows resulting in 'inf' due to division by zero in baseline:")
        print(inf_rows)

    # Drop baseline columns and compute averages
    columns_to_drop = [col for col in merged_df.columns if '_baseline' in col]
    merged_df = merged_df.drop(columns=columns_to_drop)
    average_df = merged_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics]].mean().reset_index()

    return average_df.round(1), problematic_rows


# Calculate changes
no_contrail_changes, _ = calculate_percentage_changes(no_contrail_df, common_metrics)
warming_changes, problematic_warming_rows = calculate_percentage_changes(warming_df, contrail_metrics)
cooling_changes, problematic_cooling_rows = calculate_percentage_changes(cooling_df, contrail_metrics)

# Rearrange columns for warming and cooling so contrail change columns come BEFORE non-CO2 and total climate
def reorder_columns(df):
    core_cols = ['engine', 'saf_level', 'water_injection']
    nox_co2_h2o_cols = [
        'nox_impact_sum_change',
        'co2_impact_cons_sum_change',
        'co2_impact_opti_sum_change',
        'h2o_impact_sum_change',
    ]
    contrail_cols = [
        'contrail_atr20_cocip_sum_change',
        'contrail_atr20_accf_sum_change',
    ]
    climate_total_cols = [
        'climate_non_co2_change',
        'climate_total_cons_sum_change',
        'climate_total_opti_sum_change',
    ]

    reordered_cols = core_cols + nox_co2_h2o_cols + contrail_cols + climate_total_cols
    return df[reordered_cols]

warming_changes = reorder_columns(warming_changes)
cooling_changes = reorder_columns(cooling_changes)

# Save outputs
no_contrail_changes.to_csv('results_report/climate/no_contrail_changes_vs_GTF1990.csv', index=False)
warming_changes.to_csv('results_report/climate/warming_changes_vs_GTF1990.csv', index=False)
cooling_changes.to_csv('results_report/climate/cooling_changes_vs_GTF1990.csv', index=False)

print("Saved climate impact percentage change CSVs for no contrail, warming, and cooling cases successfully!")

# Optional: Report problematic rows
if problematic_warming_rows:
    print(f"Problematic rows in warming (division by zero in baseline): {problematic_warming_rows}")
if problematic_cooling_rows:
    print(f"Problematic rows in cooling (division by zero in baseline): {problematic_cooling_rows}")
