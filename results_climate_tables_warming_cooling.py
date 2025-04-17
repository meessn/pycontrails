import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

results_df = pd.read_csv('results_main_simulations.csv')

# Group by trajectory, season, diurnal
grouped = results_df.groupby(['trajectory', 'season', 'diurnal'])

# Create empty lists to collect the records for each group
cooling_records = []
warming_records = []

for name, group in grouped:
    values = group['contrail_atr20_cocip_sum']

    if (values < 0).all():
        group_info = group.copy()
        group_info['trajectory'] = name[0]
        group_info['season'] = name[1]
        group_info['diurnal'] = name[2]
        cooling_records.append(group.copy())
    elif (values > 0).all():
        group_info = group.copy()
        group_info['trajectory'] = name[0]
        group_info['season'] = name[1]
        group_info['diurnal'] = name[2]
        warming_records.append(group.copy())

# Convert to DataFrames
cooling_df = pd.concat(cooling_records, ignore_index=True)
warming_df = pd.concat(warming_records, ignore_index=True)

saf_levels = [20, 100]

# Engine display names
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2008',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

def add_engine_display(df):
    return df.assign(
        engine_display=df.apply(
            lambda row: f"{engine_display_names[row['engine']]}" + (
                f"\n-{row['saf_level']}" if row['engine'] in ['GTF2035', 'GTF2035_wi'] and row['saf_level'] in saf_levels else ""
            ), axis=1
        )
    )

# Apply to both
cooling_df = add_engine_display(cooling_df)
warming_df = add_engine_display(warming_df)

# Preview
print("Cooling combinations:")
print(cooling_df)

print("\nWarming combinations:")
print(warming_df)

# Baseline: GTF1990, saf_level = 0
baseline_df = results_df[(results_df['engine'] == 'GTF1990') & (results_df['saf_level'] == 0)]

# Define metrics for comparison
common_metrics = [
    'nox_impact_sum', 'co2_impact_cons_sum', 'co2_impact_opti_sum', 'h2o_impact_sum',
    'climate_non_co2_cocip', 'climate_non_co2_accf', 'climate_non_co2_accf_cocip_pcfa',
    'climate_total_cons_cocip', 'climate_total_opti_cocip', 'climate_total_cons_accf', 'climate_total_opti_accf',
    'climate_total_cons_accf_cocip_pcfa', 'climate_total_opti_accf_cocip_pcfa'
]
contrail_metrics = ['contrail_atr20_cocip_sum', 'contrail_atr20_accf_sum', 'contrail_atr20_accf_cocip_pcfa_sum'] + common_metrics

missing = set(cooling_df[['trajectory', 'season', 'diurnal']].apply(tuple, axis=1)) - \
          set(baseline_df[['trajectory', 'season', 'diurnal']].apply(tuple, axis=1))
print("Missing baseline combos:", missing)
print(cooling_df.columns)
def calculate_relative_changes(df, metrics):
    merged_df = df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))


    for metric in metrics:


        baseline_metric = np.abs(merged_df[f'{metric}_baseline'])
        new_metric = np.abs(merged_df[metric])

        # Apply Normalized Relative Difference Formula
        merged_df[f'{metric}_relative_change'] = np.where(
            (baseline_metric + new_metric) == 0,
            np.nan,  # Assign NaN if both values are zero (no valid comparison)
            (new_metric - baseline_metric) / (new_metric + baseline_metric)
        )


    # Drop baseline columns
    columns_to_drop = [col for col in merged_df.columns if '_baseline' in col]
    merged_df = merged_df.drop(columns=columns_to_drop)

    # Keep only relevant columns
    flight_level_df = merged_df

    return flight_level_df


cooling_df_changes = calculate_relative_changes(cooling_df, contrail_metrics)
warming_df_changes = calculate_relative_changes(warming_df, contrail_metrics)

# Define engine order
engine_order = [
    "CFM1990", "CFM2008", "GTF",
    "GTF2035", "GTF2035\n-20", "GTF2035\n-100",
    "GTF2035WI", "GTF2035WI\n-20", "GTF2035WI\n-100"
]

# Format x-axis labels for SAF levels
engine_labels = {
    "CFM1990": "CFM1990",
    "CFM2008": "CFM2008",
    "GTF": "GTF",
    "GTF2035": "GTF2035",
    "GTF2035 - 20": "GTF2035\n-20",
    "GTF2035 - 100": "GTF2035\n-100",
    "GTF2035WI": "GTF2035WI",
    "GTF2035WI - 20": "GTF2035WI\n-20",
    "GTF2035WI - 100": "GTF2035WI\n-100"
}

def plot_cooling_warming_barplot(cooling_df, warming_df, df_name, metric='climate_total_cons_cocip_relative_change'):
    """
    Creates a grouped bar plot comparing cooling vs. warming climate impact groups for different engine configurations.
    """

    # Use the custom engine_display field (already added earlier)
    cooling_df = cooling_df[cooling_df['engine_display'].isin(engine_order)]
    warming_df = warming_df[warming_df['engine_display'].isin(engine_order)]

    # Average metric per engine_display
    cooling_avg = cooling_df.groupby("engine_display")[metric].mean().reset_index()
    warming_avg = warming_df.groupby("engine_display")[metric].mean().reset_index()

    # Merge by engine_display
    merged_df = pd.merge(cooling_avg, warming_avg, on='engine_display', suffixes=('_cooling', '_warming'))

    # Convert to Relative Climate Impact % (RASD formula)
    for col in [f"{metric}_cooling", f"{metric}_warming"]:
        merged_df[col] = (2 * merged_df[col]) / (1 - merged_df[col]) * 100
        merged_df[col] += 100

    # Sort based on engine_display order
    merged_df = merged_df.set_index("engine_display").reindex(engine_order).reset_index()

    # Plotting
    width = 0.35
    x = np.arange(len(merged_df))

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, merged_df[f"{metric}_cooling"], width=width, label="Cooling Contrail Scenarios", alpha=0.7)
    plt.bar(x + width/2, merged_df[f"{metric}_warming"], width=width, label="Warming Contrail Scenarios", alpha=0.7)

    plt.ylabel("Relative Climate Impact (%)")
    plt.title("Climate Impact Relative to GTF1990 — Cooling vs. Warming Contrails")
    plt.xticks(x, [engine_labels.get(eng, eng) for eng in merged_df['engine_display']], rotation=0, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save
    filename = f"results_report/barplot/contrail_type/contrail_type_barplot_{metric}_{df_name}.png".replace(" ", "_")
    # plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")


plot_cooling_warming_barplot(
    cooling_df_changes,
    warming_df_changes,
    df_name='df_cooling_warming',
    metric='contrail_atr20_cocip_sum_relative_change'
)
plt.show()