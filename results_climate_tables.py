import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import os
import numpy as np
import math
# Load the dataset
results_df = pd.read_csv('results_main_simulations.csv')

# Default engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2008',
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


# Default engine display names and colors
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2008',
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
    ('GTF1990', 0): 'tab:blue',
    ('GTF2000', 0): 'tab:orange',
    ('GTF', 0): 'tab:green',
    ('GTF2035', 0): 'tab:red',
    ('GTF2035', 20): 'tab:pink',
    ('GTF2035', 100): 'tab:grey',
    ('GTF2035_wi', 0): default_colors[4],
    ('GTF2035_wi', 20): 'tab:olive',
    ('GTF2035_wi', 100): 'tab:cyan'
}

season_colors = {'2023-02-06': 'tab:blue', '2023-05-05': 'tab:green', '2023-08-06': 'tab:red',
                 '2023-11-06': 'tab:brown'}
diurnal_colors = {'daytime': 'tab:blue', 'nighttime': 'tab:red'}
contrail_colors = {'formed': 'tab:green', 'not_formed': 'tab:red'}

# Load the results CSV
results_df = pd.read_csv('results_main_simulations.csv')
# ----- COCIP Method -----
# Identify flights where any engine produces a contrail (COCIP)
contrail_status_cocip = results_df.groupby(['trajectory', 'season', 'diurnal'])['contrail_atr20_cocip_sum'].sum().reset_index()
contrail_status_cocip['contrail_formed'] = contrail_status_cocip['contrail_atr20_cocip_sum'] != 0

# Merge this back
results_df_cocip = results_df.merge(
    contrail_status_cocip[['trajectory', 'season', 'diurnal', 'contrail_formed']],
    on=['trajectory', 'season', 'diurnal'],
    how='left'
)

# Classify flights
contrail_no_df_cocip = results_df_cocip[results_df_cocip['contrail_formed'] == False]
contrail_yes_df_cocip = results_df_cocip[results_df_cocip['contrail_formed'] == True]
print('cocip no df', len(contrail_no_df_cocip))
print('cocip yes df', len(contrail_yes_df_cocip))
# Strict (engine-level) classification
contrail_strict_cocip = results_df_cocip.copy()
contrail_strict_cocip['contrail_binary'] = contrail_strict_cocip['contrail_atr20_cocip_sum'] != 0

contrail_counts_cocip = contrail_strict_cocip.groupby(['trajectory', 'season', 'diurnal'])['contrail_binary'].sum().reset_index()
contrail_counts_cocip.rename(columns={'contrail_binary': 'num_engines_with_contrail'}, inplace=True)

full_contrail_flights_cocip = contrail_counts_cocip[contrail_counts_cocip['num_engines_with_contrail'] == 9]
contrail_yes_all_df_cocip = results_df.merge(
    full_contrail_flights_cocip[['trajectory', 'season', 'diurnal']],
    on=['trajectory', 'season', 'diurnal'],
    how='inner'
)
print('cocip all df', len(contrail_yes_all_df_cocip))

# ----- ACCF Method -----
# Identify flights where any engine produces a contrail (ACCF)
contrail_status_accf = results_df.groupby(['trajectory', 'season', 'diurnal'])['contrail_atr20_accf_cocip_pcfa_sum'].sum().reset_index()
contrail_status_accf['contrail_formed'] = contrail_status_accf['contrail_atr20_accf_cocip_pcfa_sum'] != 0

# Merge this back
results_df_accf = results_df.merge(
    contrail_status_accf[['trajectory', 'season', 'diurnal', 'contrail_formed']],
    on=['trajectory', 'season', 'diurnal'],
    how='left'
)

# Classify flights
contrail_no_df_accf = results_df_accf[results_df_accf['contrail_formed'] == False]
contrail_yes_df_accf = results_df_accf[results_df_accf['contrail_formed'] == True]
print('accf no df', len(contrail_no_df_accf))
print('accf yes df', len(contrail_yes_df_accf))
# Strict (engine-level) classification
contrail_strict_accf = results_df_accf.copy()
contrail_strict_accf['contrail_binary'] = contrail_strict_accf['contrail_atr20_accf_cocip_pcfa_sum'] != 0

contrail_counts_accf = contrail_strict_accf.groupby(['trajectory', 'season', 'diurnal'])['contrail_binary'].sum().reset_index()
contrail_counts_accf.rename(columns={'contrail_binary': 'num_engines_with_contrail'}, inplace=True)

full_contrail_flights_accf = contrail_counts_accf[contrail_counts_accf['num_engines_with_contrail'] == 9]
contrail_yes_all_df_accf = results_df.merge(
    full_contrail_flights_accf[['trajectory', 'season', 'diurnal']],
    on=['trajectory', 'season', 'diurnal'],
    how='inner'
)
print('accf all df', len(contrail_yes_all_df_accf))


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

def summarize_contrail_metric(df, group_cols, contrail_cols):
    grouped = df[group_cols + contrail_cols].copy()
    summaries = grouped.groupby(group_cols).agg({
        col: ['sum', lambda x: x[x > 0].sum(), lambda x: x[x < 0].sum()] for col in contrail_cols
    })

    # Fix column naming
    new_cols = []
    for col in contrail_cols:
        new_cols.extend([
            f"{col}_net",
            f"{col}_warming",
            f"{col}_cooling"
        ])

    summaries.columns = new_cols
    summaries = summaries.reset_index()
    return summaries

def scale_dataframe(df, scale=1e10):
    df_scaled = df.copy()
    # Select only numeric columns (excluding group columns like 'engine', 'saf_level')
    group_cols = ['engine', 'saf_level']
    numeric_cols = df_scaled.columns.difference(group_cols, sort=False)
    numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(df_scaled[col])]
    df_scaled[numeric_cols] = df_scaled[numeric_cols] * scale
    return df_scaled

contrail_only_metrics = [m for m in contrail_metrics if m.startswith('contrail_atr20')]

# 1. All flights
fleet_summary_all = results_df.groupby(['engine', 'saf_level'])[contrail_metrics].sum().reset_index()
fleet_summary_all_contrail_parts = summarize_contrail_metric(results_df, ['engine', 'saf_level'], contrail_only_metrics)
fleet_summary_all_full = pd.merge(fleet_summary_all, fleet_summary_all_contrail_parts, on=['engine', 'saf_level'], how='left')

# 2. COCIP
metrics_cocip = [m for m in contrail_metrics if m not in ['contrail_atr20_accf_sum', 'contrail_atr20_accf_cocip_pcfa_sum']]
fleet_summary_cocip = contrail_yes_df_cocip.groupby(['engine', 'saf_level'])[metrics_cocip].sum().reset_index()
fleet_summary_cocip_contrail_parts = summarize_contrail_metric(
    contrail_yes_df_cocip, ['engine', 'saf_level'], [m for m in contrail_only_metrics if 'accf' not in m]
)
fleet_summary_cocip_full = pd.merge(fleet_summary_cocip, fleet_summary_cocip_contrail_parts, on=['engine', 'saf_level'], how='left')

# 3. ACCF
metrics_accf = [m for m in contrail_metrics if m not in ['contrail_atr20_cocip_sum', 'contrail_atr20_accf_sum']]
fleet_summary_accf = contrail_yes_df_accf.groupby(['engine', 'saf_level'])[metrics_accf].sum().reset_index()
fleet_summary_accf_contrail_parts = summarize_contrail_metric(
    contrail_yes_df_accf, ['engine', 'saf_level'], [m for m in contrail_only_metrics if 'cocip_sum' not in m and 'accf_sum' not in m]
)
fleet_summary_accf_full = pd.merge(fleet_summary_accf, fleet_summary_accf_contrail_parts, on=['engine', 'saf_level'], how='left')

# ----------------- FILTER + SCALE -----------------

# Always include these
base_emissions = ['nox_impact_sum', 'co2_impact_cons_sum', 'co2_impact_opti_sum', 'h2o_impact_sum']

# Filter COCIP: 'cocip' in name but not 'accf_cocip_pcfa', + emissions
cocip_cols = ['engine', 'saf_level'] + base_emissions + [
    col for col in fleet_summary_cocip_full.columns if 'cocip' in col and 'accf_cocip_pcfa' not in col and 'accf' not in col
]
fleet_summary_cocip_filtered = fleet_summary_cocip_full[list(dict.fromkeys(cocip_cols))]  # remove duplicates
fleet_summary_cocip_scaled = scale_dataframe(fleet_summary_cocip_filtered)

# Filter ACCF: must contain 'accf_cocip_pcfa', + emissions
accf_cols = ['engine', 'saf_level'] + base_emissions + [
    col for col in fleet_summary_accf_full.columns if 'accf_cocip_pcfa' in col
]
fleet_summary_accf_filtered = fleet_summary_accf_full[list(dict.fromkeys(accf_cols))]
fleet_summary_accf_scaled = scale_dataframe(fleet_summary_accf_filtered)

# Scale all-flights full summary too
print(fleet_summary_all_full.dtypes)
all_cols = ['engine', 'saf_level'] + base_emissions + [
    col for col in fleet_summary_all_full.columns if 'accf_cocip_pcfa' in col or 'cocip' in col
]

fleet_summary_all_filtered = fleet_summary_all_full[list(dict.fromkeys(all_cols))]
fleet_summary_all_scaled = scale_dataframe(fleet_summary_all_filtered)


engine_order = ['GTF1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi']

def reorder_engines(df):
    df = df.copy()
    df['engine'] = pd.Categorical(df['engine'], categories=engine_order, ordered=True)
    return df.sort_values(['engine', 'saf_level']).reset_index(drop=True)

fleet_summary_all_scaled = reorder_engines(fleet_summary_all_scaled)
fleet_summary_cocip_scaled = reorder_engines(fleet_summary_cocip_scaled)
fleet_summary_accf_scaled = reorder_engines(fleet_summary_accf_scaled)

# ----------------- SAVE -----------------
output_dir = 'results_report/fleet'
os.makedirs(output_dir, exist_ok=True)

fleet_summary_all_scaled.to_csv(f'{output_dir}/fleet_summary_all_scaled.csv', index=False)
fleet_summary_cocip_scaled.to_csv(f'{output_dir}/fleet_summary_cocip_filtered_scaled.csv', index=False)
fleet_summary_accf_scaled.to_csv(f'{output_dir}/fleet_summary_accf_filtered_scaled.csv', index=False)


def compute_relative_diff(df):
    df = df.copy()

    # Reference: GTF1990 at SAF level 0
    baseline_row = df[(df['engine'] == 'GTF1990') & (df['saf_level'] == 0)]
    if baseline_row.empty:
        raise ValueError("Baseline row 'GTF1990' with SAF level 0 not found.")

    baseline = baseline_row.iloc[0]

    diffs = []
    for _, row in df.iterrows():
        if row['engine'] == 'GTF1990' and row['saf_level'] == 0:
            continue  # skip baseline

        row_diff = {
            'engine': row['engine'],
            'saf_level': row['saf_level']
        }

        for col in df.columns:
            if col in ['engine', 'saf_level']:
                continue
            base_val = baseline[col]
            print(base_val)
            if base_val != 0:
                row_diff[col] = ((abs(row[col]) - abs(base_val))/ abs(base_val)) * 100
            else:
                row_diff[col] = None  # avoid division by zero

        diffs.append(row_diff)

    return pd.DataFrame(diffs)


fleet_rel_all = compute_relative_diff(fleet_summary_all_scaled)
fleet_rel_cocip = compute_relative_diff(fleet_summary_cocip_scaled)
fleet_rel_accf = compute_relative_diff(fleet_summary_accf_scaled)

fleet_rel_all.to_csv(f'{output_dir}/fleet_summary_all_rel_diff.csv', index=False)
fleet_rel_cocip.to_csv(f'{output_dir}/fleet_summary_cocip_rel_diff.csv', index=False)
fleet_rel_accf.to_csv(f'{output_dir}/fleet_summary_accf_rel_diff.csv', index=False)

def calculate_relative_changes(df, metrics):
    merged_df = df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

    for metric in metrics:
        baseline_metric = np.abs(merged_df[f'{metric}_baseline'])
        new_metric = np.abs(merged_df[metric])

        merged_df[f'{metric}_relative_change'] = np.where(
            baseline_metric == 0,
            np.nan, #nan to avoid dilution of results, if both zero or baseline only
            (new_metric - baseline_metric) / baseline_metric
        )

    # Drop baseline columns
    columns_to_drop = [col for col in merged_df.columns if '_baseline' in col]
    merged_df = merged_df.drop(columns=columns_to_drop)

    # Keep only relevant columns
    flight_level_df = merged_df

    return flight_level_df

results_df_changes = calculate_relative_changes(results_df, contrail_metrics)
contrail_no_cocip_changes= calculate_relative_changes(contrail_no_df_cocip, common_metrics)
contrail_yes_cocip_changes = calculate_relative_changes(contrail_yes_df_cocip, contrail_metrics)
contrail_yes_all_cocip_changes = calculate_relative_changes(contrail_yes_all_df_cocip, contrail_metrics)
contrail_no_accf_changes= calculate_relative_changes(contrail_no_df_accf, common_metrics)
contrail_yes_accf_changes = calculate_relative_changes(contrail_yes_df_accf, contrail_metrics)
contrail_yes_all_accf_changes = calculate_relative_changes(contrail_yes_all_df_accf, contrail_metrics)

# Engine display names
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2008',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}


# Function to generate bar chart with error bars



# Engine display names
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2008',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

metric_titles = {
        'nox_impact_sum_relative_change': 'NOx',
        'contrail_atr20_cocip_sum_relative_change': 'Contrail (CoCiP)',
        'contrail_atr20_accf_cocip_pcfa_sum_relative_change': 'Contrail (aCCF)',
        'climate_non_co2_cocip_relative_change': 'Non-CO₂ (Contrail CoCiP)',
        'climate_non_co2_accf_cocip_pcfa_relative_change': 'Non-CO₂ (Contrail aCCF)',
        'climate_total_cons_cocip_relative_change': 'Total (Contrail CoCiP)',
        'climate_total_cons_accf_cocip_pcfa_relative_change': 'Total (Contrail aCCF)',
        'climate_total_opti_cocip_relative_change': 'Total (Contrail CoCiP)',
        'climate_total_opti_accf_cocip_pcfa_relative_change': 'Total (Contrail aCCF)',
        'co2_impact_cons_sum_relative_change': 'CO₂',
        'co2_impact_opti_sum_relative_change': 'CO₂',
        'h2o_impact_sum_relative_change': 'H₂O'
    }

legend_titles = {
        'contrail_atr20_cocip_sum_relative_change': 'Contrail (CoCiP)',
        'contrail_atr20_accf_cocip_pcfa_sum_relative_change': 'Contrail (aCCF)',
        'contrail_atr20_accf_sum_relative_change': 'Contrail (aCCF without nvPM Correction)',
        'nox_impact_sum_relative_change': 'NOx',
        'co2_impact_cons_sum_relative_change': 'CO₂',
        'co2_impact_opti_sum_relative_change': 'CO₂',
        'co2_impact_midpoint_sum_relative_change': 'CO₂',
        'climate_total_midpoint_cocip_relative_change': 'Total (Contrail CoCiP)',
        'climate_total_midpoint_accf_cocip_pcfa_relative_change': 'Total (Contrail aCCF)',
        'climate_non_co2_cocip_relative_change': 'Non-CO₂ (Contrail CoCiP)',
        'climate_non_co2_accf_cocip_pcfa_relative_change': 'Non-CO₂ (Contrail aCCF)',
        'climate_total_cons_cocip_relative_change': 'Total (Contrail CoCiP) Conservative',
        'climate_total_cons_accf_cocip_pcfa_relative_change': 'Total (Contrail aCCF) Conservative',
        'climate_total_opti_cocip_relative_change': 'Total (Contrail CoCiP) Optimistic',
        'climate_total_opti_accf_cocip_pcfa_relative_change': 'Total (Contrail aCCF) Optimistic',
        'h2o_impact_sum_relative_change': 'H₂O'
    }

metric_abbr_map = {
    'nox_impact_sum_relative_change': 'NOx',
    'contrail_atr20_cocip_sum_relative_change': 'CtrC',
    'contrail_atr20_accf_cocip_pcfa_sum_relative_change': 'CtrA',
    'climate_non_co2_cocip_relative_change': 'NonCO2C',
    'climate_non_co2_accf_cocip_pcfa_relative_change': 'NonCO2A',
    'climate_total_cons_cocip_relative_change': 'TotC_cons',
    'climate_total_cons_accf_cocip_pcfa_relative_change': 'TotA_cons',
    'climate_total_opti_cocip_relative_change': 'TotC_opti',
    'climate_total_opti_accf_cocip_pcfa_relative_change': 'TotA_opti',
    'co2_impact_cons_sum_relative_change': 'CO2_cons',
    'co2_impact_opti_sum_relative_change': 'CO2_opti',
    'h2o_impact_sum_relative_change': 'H2O'
}




from matplotlib.colors import to_rgb
def darken_color(color, amount=0.6):
    """
    Darkens a given matplotlib color.
    `amount` < 1 makes the color darker.
    """
    r, g, b = to_rgb(color)
    return (r * amount, g * amount, b * amount)



def plot_rad_barplot(df, df_name, metrics=['climate_total_cons_sum_relative_change']):
    """
    Plots a grouped bar chart of relative difference (%) after applying the transformation formula.

    Parameters:
        df (DataFrame): The input dataframe containing RASD values.
        df_name (str): Name of the dataframe (for saving the plot).
        metrics (list): A list of column names representing RASD values to be plotted.
    """

    # Engines to plot, including baseline CFM1990
    engines_to_plot = ['GTF1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi']
    saf_levels = [20, 100]  # Only GTF2035 variants get SAF levels

    # Generate display names including SAF levels
    df['engine_display'] = df.apply(
        lambda row: f"{engine_display_names[row['engine']]}" + (
            f"\n{row['saf_level']}" if row['engine'] in ['GTF2035', 'GTF2035_wi'] and row['saf_level'] in saf_levels else ""
        ), axis=1
    )

    # Filter relevant engines (including CFM1990)
    df_filtered = df[df['engine'].isin(engines_to_plot)]

    # Compute median  per engine type for each metric
    grouped = df_filtered.groupby("engine_display")[metrics].median()

    # # Apply transformation: (2 * rasd) / (1 - rasd)
    # for metric in metrics:
    #     grouped[metric] = (2 * grouped[metric]) / (1 - grouped[metric])

    for metric in metrics:
        grouped[metric] = grouped[metric] * 100 + 100

    for metric in metrics:
        grouped.loc["CFM1990", metric] = 100

    grouped = grouped.reset_index()

    # Print value counts for verification
    print("Value counts per engine:")
    print(df_filtered['engine_display'].value_counts())

    # Define x-axis order, including baseline CFM1990
    x_order = [
        "CFM1990", "CFM2008", "GTF",
        "GTF2035", "GTF2035\n20", "GTF2035\n100",
        "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
    ]

    # Ensure ordering
    grouped = grouped.set_index("engine_display").reindex(x_order).reset_index()

    # Adjust bar width based on the number of metrics
    width = 0.6 if len(metrics) == 1 else 0.15
    x = np.arange(len(x_order))

    plt.figure(figsize=(12, 6))

    use_contrail_model = "no" not in df_name.lower()

    for i, metric in enumerate(metrics):
        raw_label = legend_titles.get(metric, metric.replace("_relative_change", "").replace("_", " "))

        if not use_contrail_model and raw_label.startswith("Total (Contrail"):
            # Strip the "(Contrail ...)" part but keep "Total" and the suffix (e.g., Conservative/Optimistic)
            if ")" in raw_label:
                suffix = raw_label.split(")", 1)[1].strip()
                legend_label = f"Total {suffix}" if suffix else "Total"
            else:
                legend_label = "Total"
        elif not use_contrail_model and raw_label.startswith("Non-CO₂ (Contrail"):
            # Strip the "(Contrail ...)" part entirely
            legend_label = "Non-CO₂"
        else:
            legend_label = raw_label
        plt.bar(x + i * (width if len(metrics) > 1 else 0), grouped[metric],
                alpha=0.7, label=legend_label, width=width)

    # Generate title parts, ensuring "Total" and "CO₂" appear only once
    title_parts = []
    seen_total = seen_co2 = False

    for metric in metrics:
        if metric in ['climate_total_cons_cocip_relative_change',
                      'climate_total_opti_cocip_relative_change']:
            if not seen_total:
                title_parts.append("Total (Contrail CoCiP)" if use_contrail_model else "Total")
                seen_total = True
        elif metric in ['climate_total_cons_accf_cocip_pcfa_relative_change',
                        'climate_total_opti_accf_cocip_pcfa_relative_change']:
            if not seen_total:
                title_parts.append("Total (Contrail aCCF)" if use_contrail_model else "Total")
                seen_total = True
        elif metric in ['co2_impact_cons_sum_relative_change', 'co2_impact_opti_sum_relative_change']:
            if not seen_co2:
                title_parts.append("CO₂")
                seen_co2 = True
        elif metric in metric_titles:
            label = metric_titles[metric]
            if not use_contrail_model and label.startswith("Non-CO₂ (Contrail"):
                label = "Non-CO₂"
            title_parts.append(label)

    # Create the title
    plot_title = " & ".join(title_parts) + " Climate Impact Relative to CFM1990"

    # plt.xlabel("Engine Type")
    plt.ylabel("Relative Climate Impact (%)")
    plt.title(plot_title)
    plt.xticks(x + (width * (len(metrics) - 1) / 2 if len(metrics) > 1 else 0), x_order, rotation=0, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Generate abbreviated filename
    metric_abbreviations = [legend_titles.get(m, m.replace("_relative_change", "").replace("_", "")) for m in metrics]
    metric_str = "_".join(metric_abbreviations)
    filename = f"results_report/barplot/rad_barplot_{df_name}_{metric_str}.png".replace(" ", "_")

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")

# plot_rad_barplot(results_df_changes, "results_df", metrics=['co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change', 'nox_impact_sum_relative_change'])
#
# plot_rad_barplot(contrail_no_accf_changes, "contrail_no_accf", metrics=['nox_impact_sum_relative_change'])
# plot_rad_barplot(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])
#
# plot_rad_barplot(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['nox_impact_sum_relative_change'])
# plot_rad_barplot(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])
#
#
# plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
# plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change','climate_non_co2_cocip_relative_change'])
# plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['contrail_atr20_cocip_sum_relative_change'])
#
#
# plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])
#
# plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
# plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change','climate_non_co2_accf_cocip_pcfa_relative_change'])
# plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])
# plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['contrail_atr20_accf_cocip_pcfa_sum_relative_change'])


from matplotlib.colors import to_rgba

def faded_edge(color, alpha=0.7):
    return to_rgba(color, alpha)





def plot_rad_barplot_v3(df, df_name, metrics=['climate_total_cons_sum_relative_change']):
    metric_color_map = {
        "nox_impact": "tab:blue",
        "co2_impact": "tab:orange",
        "climate_non_co2": "tab:purple",
        "climate_total": "tab:cyan",
        "contrail_atr20_accf": "tab:red",
        "contrail_atr20_cocip": "tab:green",
        "h2o_impact": "tab:grey"
    }

    def get_metric_color(metric_name):
        for key in metric_color_map:
            if metric_name.startswith(key):
                return metric_color_map[key]
        return None
    # Engines to plot, including baseline CFM1990
    engines_to_plot = ['GTF1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi']
    saf_levels = [20, 100]  # Only GTF2035 variants get SAF levels

    # Generate display names including SAF levels
    df['engine_display'] = df.apply(
        lambda row: f"{engine_display_names[row['engine']]}" + (
            f"\n{row['saf_level']}" if row['engine'] in ['GTF2035', 'GTF2035_wi'] and row[
                'saf_level'] in saf_levels else ""
        ), axis=1
    )

    # Filter relevant engines (including CFM1990)
    df_filtered = df[df['engine'].isin(engines_to_plot)]

    # Compute mean and standard deviation per engine type for each metric
    grouped = df_filtered.groupby("engine_display")[metrics].median()

    # # Apply transformation: (2 * rasd) / (1 - rasd)
    # for metric in metrics:
    #     grouped[metric] = (2 * grouped[metric]) / (1 - grouped[metric])

    for metric in metrics:
        grouped[metric] = grouped[metric] * 100 + 100

    for metric in metrics:
        grouped.loc["CFM1990", metric] = 100

    grouped = grouped.reset_index()

    print("Value counts per engine:")
    print(df_filtered['engine_display'].value_counts())

    x_order = [
        "CFM1990", "CFM2008", "GTF",
        "GTF2035", "GTF2035\n20", "GTF2035\n100",
        "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
    ]

    grouped = grouped.set_index("engine_display").reindex(x_order).reset_index()

    width = 0.35
    x = np.arange(len(x_order))

    plt.figure(figsize=(12, 6))
    use_contrail_model = "no" not in df_name.lower()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_co2 = colors[0]
    color_nox = colors[1]


    # Track legends added per color for SAF bars
    legend_elements = []
    legend_labels_used = set()
    saf_legend_by_color = set()

    # Determine how many actual bar groups we'll draw (either solo or cons-opti pair)
    plotted_metrics = []
    skip_next = False
    for idx, metric in enumerate(metrics):
        if skip_next:
            skip_next = False
            continue
        if '_cons_' in metric and metric.replace('_cons_', '_opti_') in metrics:
            plotted_metrics.append((metric, metric.replace('_cons_', '_opti_')))
            skip_next = True
        elif '_opti_' in metric and metric.replace('_opti_', '_cons_') in metrics:
            continue
        else:
            plotted_metrics.append(metric)

    n_bar_groups = len(plotted_metrics)

    # Width logic
    base_width = 0.5
    raw_width = base_width / max(n_bar_groups, 1)
    width = min(max(raw_width, 0.15), 0.35)

    # Plot
    for bar_index, entry in enumerate(plotted_metrics):
        if isinstance(entry, tuple):
            cons_metric, opti_metric = entry
            raw_color = get_metric_color(cons_metric) or colors[bar_index % len(colors)]
            edge = faded_edge(raw_color, alpha=0.7)
            dark_color = darken_color(raw_color, amount=0.7)
            bar_color = faded_edge(raw_color, alpha=0.7)

            legend_label = legend_titles.get(opti_metric, opti_metric.replace("_relative_change", "").replace("_", " "))
            if legend_label not in legend_labels_used:
                legend_elements.append(Patch(facecolor=bar_color, edgecolor=edge, label=legend_label))
                legend_labels_used.add(legend_label)

            cons_values = grouped[cons_metric]
            opti_values = grouped[opti_metric]

            for j, label in enumerate(x_order):
                if pd.isna(cons_values[j]) or pd.isna(opti_values[j]):
                    continue

                x_offset = x[j] + (bar_index - n_bar_groups / 2 + 0.5) * width
                bottom_val = min(cons_values[j], opti_values[j])
                top_val = max(cons_values[j], opti_values[j])
                delta = top_val - bottom_val
                mid_val = (cons_values[j] + opti_values[j]) / 2

                plt.bar(x_offset, mid_val, width=width, color=bar_color, zorder=2)
                if "\n20" in label or "\n100" in label:
                    plt.bar(x_offset, delta, bottom=bottom_val, width=width * 0.05,
                            color='white', edgecolor=dark_color, linewidth=1.5, zorder=3)

                    cap_width = width * 0.5
                    plt.hlines([bottom_val, top_val],
                               x_offset - cap_width / 2, x_offset + cap_width / 2,
                               color=dark_color, linewidth=1.5, zorder=4)

            if raw_color not in saf_legend_by_color:
                saf_legend = Line2D([0], [0], color=dark_color, linewidth=2.5,
                                    label='SAF Production Pathway Range')
                legend_elements.append(saf_legend)
                saf_legend_by_color.add(raw_color)

        else:
            metric = entry
            raw_color = get_metric_color(metric) or colors[bar_index % len(colors)]
            edge = faded_edge(raw_color, alpha=0.7)
            bar_color = raw_color

            alpha = 0.7
            legend_label = legend_titles.get(metric, metric.replace("_relative_change", "").replace("_", " "))
            if legend_label not in legend_labels_used:
                legend_elements.append(Patch(facecolor=bar_color, alpha=alpha, label=legend_label))
                legend_labels_used.add(legend_label)

            x_offset = x + (bar_index - n_bar_groups / 2 + 0.5) * width
            plt.bar(x_offset, grouped[metric], alpha=alpha,
                    width=width, color=bar_color, linewidth=1.0)

    # # Add SAF legend at the end if used
    # if saf_pathway_used:
    #     legend_elements.append(Patch(facecolor='white', edgecolor=color_co2, hatch='//',
    #                                  label='SAF Production Pathway Dependency'))

    # Finalize legend
    if legend_elements:
        plt.legend(handles=legend_elements)
    title_parts = []
    seen_total = seen_co2 = False

    for metric in metrics:
        if metric in ['climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change']:
            if not seen_total:
                title_parts.append("Total (Contrail CoCiP)" if use_contrail_model else "Total")
                seen_total = True
        elif metric in ['climate_total_cons_accf_cocip_pcfa_relative_change',
                        'climate_total_opti_accf_cocip_pcfa_relative_change']:
            if not seen_total:
                title_parts.append("Total (Contrail aCCF)" if use_contrail_model else "Total")
                seen_total = True
        elif metric in ['co2_impact_cons_sum_relative_change', 'co2_impact_opti_sum_relative_change']:
            if not seen_co2:
                title_parts.append("CO₂")
                seen_co2 = True
        elif metric in metric_titles:
            label = metric_titles[metric]
            if not use_contrail_model and label.startswith("Non-CO₂ (Contrail"):
                label = "Non-CO₂"
            title_parts.append(label)

    plot_title = " & ".join(title_parts) + " Climate Impact Relative to CFM1990"

    plt.ylabel("Relative Climate Impact (%)")
    plt.title(plot_title)
    plt.xticks(x, x_order, rotation=0, ha="center")




    if legend_elements:
        plt.legend(handles=legend_elements)
    else:
        plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)

    metric_abbreviations = [legend_titles.get(m, m.replace("_relative_change", "").replace("_", "")) for m in metrics]
    metric_str = "_".join([metric_abbr_map.get(m, m[:8]) for m in metrics])
    filename = f"results_report/barplot/rad_{df_name}_{metric_str}.png".replace(" ", "_")
    # Add headroom to y-axis
    y_max = 1.05 * grouped[metrics].max().max()
    plt.ylim(0, y_max)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")

plot_rad_barplot_v3(results_df_changes, "results_df", metrics=['co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change', 'nox_impact_sum_relative_change', 'h2o_impact_sum_relative_change'])
plot_rad_barplot_v3(results_df_changes, "results_df", metrics=['co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change', 'nox_impact_sum_relative_change'])
# plt.show()
plot_rad_barplot_v3(contrail_no_accf_changes, "contrail_no_accf", metrics=['nox_impact_sum_relative_change'])
plot_rad_barplot_v3(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot_v3(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])

plot_rad_barplot_v3(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['h2o_impact_sum_relative_change','nox_impact_sum_relative_change','climate_non_co2_cocip_relative_change'])
plot_rad_barplot_v3(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot_v3(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])
plot_rad_barplot_v3(contrail_no_cocip_changes, "no_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change', 'climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])

plot_rad_barplot_v3(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
plot_rad_barplot_v3(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['h2o_impact_sum_relative_change','nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change','climate_non_co2_cocip_relative_change'])
plot_rad_barplot_v3(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['contrail_atr20_cocip_sum_relative_change'])


plot_rad_barplot_v3(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot_v3(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])
plot_rad_barplot_v3(contrail_yes_cocip_changes, "yes_cocip", metrics=['climate_non_co2_cocip_relative_change', 'climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])
plot_rad_barplot_v3(contrail_yes_cocip_changes, "yes_cocip", metrics=['climate_non_co2_cocip_relative_change','co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change', 'climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])
plot_rad_barplot_v3(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
plot_rad_barplot_v3(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['h2o_impact_sum_relative_change' ,'nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change', 'climate_non_co2_accf_cocip_pcfa_relative_change'])
plot_rad_barplot_v3(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot_v3(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])
plot_rad_barplot_v3(contrail_yes_accf_changes, "yes_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])
plot_rad_barplot_v3(contrail_yes_accf_changes, "yes_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change', 'climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])
plot_rad_barplot_v3(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['contrail_atr20_accf_cocip_pcfa_sum_relative_change'])







def plot_grouped_boxplot_v6(df, df_name, metrics, ylim=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Affine2D
    import matplotlib.colors as mcolors
    plt.clf()
    plt.close('all')
    df = df.copy()
    is_contrail_off = 'no' in df_name.lower()
    # Display names
    engine_display_names = {
        'GTF1990': 'CFM1990',
        'GTF2000': 'CFM2008',
        'GTF': 'GTF',
        'GTF2035': 'GTF2035',
        'GTF2035_wi': 'GTF2035WI',
        'CFM_joined': 'CFM1990/CFM2008'
    }

    # Color mapping
    metric_color_map = {
        "nox_impact": "tab:blue",
        "co2_impact": "tab:orange",
        "climate_non_co2": "tab:purple",
        "climate_total": "tab:cyan",
        "contrail_atr20_accf": "tab:red",
        "contrail_atr20_cocip": "tab:green",
        "h2o_impact": "tab:grey"
    }

    def get_metric_color(metric_name):
        for key in metric_color_map:
            if metric_name.startswith(key):
                return metric_color_map[key]
        return "tab:gray"

    # Detect midpoint metrics
    midpoint_metrics = {}
    new_metrics = []
    for metric in metrics:
        if '_cons_' in metric:
            opti_metric = metric.replace('_cons_', '_opti_')
            if opti_metric in df.columns:
                midpoint_name = metric.replace('_cons_', '_midpoint_')
                df[midpoint_name] = df[metric]  # copy data
                midpoint_metrics[midpoint_name] = {
                    'cons': metric,
                    'opti': opti_metric
                }
                new_metrics.append(midpoint_name)
            else:
                new_metrics.append(metric)
        else:
            new_metrics.append(metric)

    metrics = new_metrics

    # Metrics eligible for CFM joining
    joinable_metrics = {
        'nox_impact_sum_relative_change',
        'h2o_impact_sum_relative_change',
        'co2_impact_cons_sum_relative_change',
        'co2_impact_opti_sum_relative_change',
        'co2_impact_midpoint_sum_relative_change'
    }
    should_join_cfm = all(m in joinable_metrics for m in metrics)

    if should_join_cfm:
        df['engine'] = df['engine'].replace({'GTF1990': 'CFM_joined', 'GTF2000': 'CFM_joined'})
        engines_to_plot = ['CFM_joined', 'GTF', 'GTF2035', 'GTF2035_wi']
        x_order = [
            "CFM1990/CFM2008", "GTF",
            "GTF2035", "GTF2035\n20", "GTF2035\n100",
            "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
        ]
    else:
        engines_to_plot = ['GTF1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi']
        x_order = [
            "CFM1990", "CFM2008", "GTF",
            "GTF2035", "GTF2035\n20", "GTF2035\n100",
            "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
        ]

    saf_levels = [20, 100]
    df_filtered = df[df['engine'].isin(engines_to_plot)].copy()

    df_filtered['engine_display'] = df_filtered.apply(
        lambda row: f"{engine_display_names[row['engine']]}" +
                    (f"\n{row['saf_level']}" if row['engine'] in ['GTF2035', 'GTF2035_wi']
                                                and row['saf_level'] in saf_levels else ""),
        axis=1
    )

    for metric in metrics:
        df_filtered[metric] = df_filtered[metric] * 100

    df_melted = df_filtered.melt(
        id_vars=['engine_display'],
        value_vars=metrics,
        var_name='metric',
        value_name='value'
    )

    current_color_map = {metric: get_metric_color(metric) for metric in metrics}
    legend_labels = {}
    for metric in metrics:
        label = legend_titles.get(metric, metric.replace("_", " "))
        if is_contrail_off:
            label = label.replace(" (Contrail CoCiP)", "").replace(" (Contrail aCCF)", "")
            label = label.replace(" (CoCiP)", "").replace(" (aCCF)", "")
        legend_labels[metric] = label

    if len(metrics) == 1:
        box_width = 0.4  # Narrower for single metric
    else:
        box_width = 0.7  # Default width for multiple metrics
    plt.figure(figsize=(12, 6))
    metric_count = len(metrics)
    ax = plt.gca()  # Get axis early so we can use it for drawing bars



    ax = sns.boxplot(
        data=df_melted,
        x='engine_display',
        y='value',
        hue='metric',
        order=x_order,
        palette=current_color_map,
        showfliers=False,
        width=box_width
    )

    for patch in ax.patches:
        facecolor = patch.get_facecolor()
        patch.set_facecolor((*facecolor[:3], 0.7))

    # plt.title("Relative Climate Impact Comparison")
    plt.ylabel("Relative Climate Impact (%)")
    plt.xlabel("")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=0, ha="center")


    for i, engine_display in enumerate(x_order):
        base_x = i
        for j, metric in enumerate(metrics):
            offset = (-0.5 + (j + 0.5) / metric_count) * box_width
            x = base_x + offset

            engine_mask = df_filtered['engine_display'] == engine_display

            if metric in midpoint_metrics:
                cons = midpoint_metrics[metric]['cons']
                opti = midpoint_metrics[metric]['opti']
                cons_vals = df_filtered.loc[engine_mask, cons] * 100
                opti_vals = df_filtered.loc[engine_mask, opti] * 100
                all_vals = pd.concat([cons_vals, opti_vals])
            else:
                all_vals = df_filtered.loc[engine_mask, metric]

            if all_vals.empty:
                continue

            if metric in midpoint_metrics:
                q3_cons = cons_vals.quantile(0.75)
                q3_opti = opti_vals.quantile(0.75)
                bar_height = (q3_cons + q3_opti) / 2
            else:
                bar_height = all_vals.quantile(0.75)

            q1 = all_vals.quantile(0.25)
            q3 = all_vals.quantile(0.75)

            # Skip drawing the bar only if the IQR crosses zero
            if q1 < 0 < q3 or q3 < 0 < q1:
                continue

            base_rgb = mcolors.to_rgba(get_metric_color(metric))
            bar_color = (*base_rgb[:3], 0.3)
            bar_width = box_width / metric_count

            ax.add_patch(Rectangle(
                (x - bar_width / 2, 0),
                bar_width,
                bar_height,
                facecolor=bar_color,
                edgecolor=None,
                linewidth=0,
                zorder=1.5
            ))

    mean_marker_size = 30 if metric_count == 1 else 20 if metric_count == 2 else 10

    xticks = ax.get_xticks()
    box_positions = {}
    for i, engine_display in enumerate(x_order):
        base_x = xticks[i]
        for j, metric in enumerate(metrics):
            offset = (-0.5 + (j + 0.5) / metric_count) * box_width
            box_positions[(engine_display, metric)] = base_x + offset

    suppress_means = (
            'climate_non_co2_cocip_relative_change' in metrics or
            'climate_non_co2_accf_cocip_pcfa_relative_change' in metrics or
            'climate_total_midpoint_accf_cocip_pcfa_relative_change' in metrics or
            'climate_total_midpoint_cocip_relative_change' in metrics)
    suppress_means = True
    print(metrics)
    if not suppress_means:
        for (engine, metric), group in df_melted.groupby(['engine_display', 'metric']):
            if (engine, metric) not in box_positions:
                continue
            if metric == 'climate_non_co2_cocip_relative_change':
                continue  # Skip mean for this metric
            x = box_positions[(engine, metric)]
            """OPTI AND CONS SAF approach have same spread of data -> translate box and other components to show the mean of opti cons (so not cons or opt but mean scenario )"""
            if metric in midpoint_metrics:
                cons = midpoint_metrics[metric]['cons']
                opti = midpoint_metrics[metric]['opti']
                engine_mask = df_filtered['engine_display'] == engine
                mean_cons = df_filtered.loc[engine_mask, cons] * 100
                mean_opti = df_filtered.loc[engine_mask, opti] * 100
                mean_val = (mean_cons.mean() + mean_opti.mean()) / 2
            else:
                mean_val = group['value'].mean()

            plt.scatter(x, mean_val, color='black', marker='D',
                        s=mean_marker_size, zorder=10)

    # Shift entire boxplot vertically for midpoint metrics
    for metric in midpoint_metrics:
        cons = midpoint_metrics[metric]['cons']
        opti = midpoint_metrics[metric]['opti']

        for engine in df_filtered['engine_display'].unique():
            x = box_positions.get((engine, metric))
            engine_mask = df_filtered['engine_display'] == engine
            if not engine_mask.any():
                continue

            cons_median = df_filtered.loc[engine_mask, cons].median() * 100
            opti_median = df_filtered.loc[engine_mask, opti].median() * 100
            target_median = (cons_median + opti_median) / 2
            dy = target_median - cons_median

            # Precompute RGBA for current metric
            target_color = mcolors.to_rgba(current_color_map[metric], alpha=0.7)
            target_x = round(box_positions.get((engine, metric), -999), 2)

            for patch in reversed(ax.patches):  # reverse to avoid index shifting
                path = patch.get_path()
                verts = path.vertices
                transformed = patch.get_patch_transform().transform(verts)
                box_x_center = round(np.mean(transformed[:, 0]), 3)
                facecolor = patch.get_facecolor()

                # Debug print to understand the match failure
                # print(f"Checking patch: center={box_x_center}, target_x={target_x}, "
                #       f"color={facecolor}, target_color={target_color}")

                # Color difference
                color_diff = np.abs(np.array(facecolor) - np.array(target_color))
                # print(f"Color diff: {color_diff}")

                if np.isclose(box_x_center, target_x, atol=0.05) and \
                        np.allclose(facecolor, target_color, atol=0.15):
                    # print('test')
                    # Remove the original box
                    # print("Original box width:", max(transformed[:, 0]) - min(transformed[:, 0]))
                    transform = Affine2D().translate(0, dy) + ax.transData
                    patch.set_transform(transform)
                    break

            # Shift Line2D elements: median, whiskers, caps
            for line in ax.lines:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) != 2 or len(ydata) != 2:
                    continue

                x_mid = sum(xdata) / 2
                if abs(x_mid - box_positions.get((engine, metric), -999)) > 0.01:
                    continue

                # median
                if ydata[0] == ydata[1] and abs(ydata[0] - cons_median) < 0.01:
                    transform = Affine2D().translate(0, dy) + ax.transData
                    line.set_transform(transform)
                # whiskers
                elif xdata[0] == xdata[1]:
                    transform = Affine2D().translate(0, dy) + ax.transData
                    line.set_transform(transform)
                # caps
                elif abs(ydata[0] - ydata[1]) < 1e-3:
                    transform = Affine2D().translate(0, dy) + ax.transData
                    line.set_transform(transform)



    # xticks = ax.get_xticks()
    # ax.set_xlim(xticks[0] - 0.05, xticks[-1] + 0.05)
    # Legend
    # --- Draw SAF Production Pathway Range Lines (from cons median to opti median) ---
    saf_line_drawn_for = set()
    # SAF line z-order logic
    saf_line_on_top = any(m in metrics for m in [
        'climate_total_midpoint_accf_cocip_pcfa_relative_change',
        'climate_total_midpoint_cocip_relative_change'
    ])
    zorder_saf = 20 if saf_line_on_top else 1.4

    final_handles = []
    saf_legend_added_for = set()

    for metric in metrics:
        color = current_color_map[metric]
        label = legend_titles.get(metric, metric.replace("_", " "))
        if 'no' in df_name.lower():
            label = label.split(' (')[0]  # Removes any parenthetical content
        patch = Patch(facecolor=color, alpha=0.7, label=label)
        final_handles.append(patch)

        if metric in midpoint_metrics and metric not in saf_legend_added_for:
            cons_col = midpoint_metrics[metric]['cons']
            opti_col = midpoint_metrics[metric]['opti']
            base_color = get_metric_color(metric)

            # SAF line color logic (apply special color only to total climate)
            if metric in ['climate_total_midpoint_cocip_relative_change',
                          'climate_total_midpoint_accf_cocip_pcfa_relative_change']:
                line_color = 'tab:blue'
                this_zorder = 20
            else:
                line_color = mcolors.to_rgba(base_color, alpha=1.0)
                this_zorder = 1.4

            for engine in df_filtered['engine_display'].unique():
                engine_mask = df_filtered['engine_display'] == engine
                if not engine_mask.any():
                    continue

                x = box_positions.get((engine, metric))
                if x is None:
                    continue

                cons_median = df_filtered.loc[engine_mask, cons_col].median() * 100
                opti_median = df_filtered.loc[engine_mask, opti_col].median() * 100

                if np.isclose(cons_median, opti_median, atol=0.01):
                    continue

                cap_width = box_width / metric_count * 1.0

                ax.plot([x, x], [cons_median, opti_median],
                        color=line_color, linewidth=1.5, zorder=this_zorder)

                ax.hlines([cons_median, opti_median],
                          x - cap_width / 2, x + cap_width / 2,
                          color=line_color, linewidth=1.5, zorder=this_zorder)

            # Only add SAF legend once per metric (and with correct color)
            final_handles.append(Line2D([0], [0],
                                        color=line_color,
                                        linewidth=1.5,
                                        label="SAF Production Pathway Range"))
            saf_legend_added_for.add(metric)

    # Mean marker at the end
    if not suppress_means:
        final_handles.append(Line2D([0], [0], marker='D', color='black',
                                    label='Mean', markersize=6, linestyle='None'))

    plt.legend(handles=final_handles, title=None)

    cocip_metrics = {
        'climate_total_cons_cocip_relative_change',
        'climate_total_opti_cocip_relative_change',
        'climate_total_midpoint_cocip_relative_change',
        'climate_non_co2_cocip_relative_change',
        'contrail_atr20_cocip_sum_relative_change'
    }

    accf_metrics = {
        'climate_total_cons_accf_cocip_pcfa_relative_change',
        'climate_total_opti_accf_cocip_pcfa_relative_change',
        'climate_total_midpoint_accf_cocip_pcfa_relative_change',
        'climate_non_co2_accf_cocip_pcfa_relative_change',
        'contrail_atr20_accf_cocip_pcfa_sum_relative_change'
    }

    used_cocip = [m for m in metrics if m in cocip_metrics]
    used_accf = [m for m in metrics if m in accf_metrics]

    title_parts = []
    seen_total = seen_co2 = seen_nonco2 = seen_contrail = False

    for metric in metrics:
        raw_label = metric_titles.get(metric, metric.replace("_relative_change", "").replace("_", " "))
        suffix = ""

        if not is_contrail_off:
            if metric in cocip_metrics and len(used_cocip) == 1:
                suffix = " (Contrail CoCiP)"
            elif metric in accf_metrics and len(used_accf) == 1:
                suffix = " (Contrail aCCF)"

        if metric.startswith("co2_impact") and not seen_co2:
            label = "CO₂" + suffix
            seen_co2 = True
        elif metric.startswith("h2o_impact") and "H₂O" not in title_parts:
            label = "H₂O" + suffix
        elif metric.startswith("nox_impact") and "NOx" not in title_parts:
            label = "NOx" + suffix
        elif "contrail" in metric and not seen_contrail:
            label = "Contrail" + suffix
            seen_contrail = True
        elif "non_co2" in metric and not seen_nonco2:
            label = "Non-CO₂" + suffix
            seen_nonco2 = True
        elif metric.startswith("climate_total") and not seen_total:
            label = "Total" + suffix
            seen_total = True
        else:
            label = raw_label + suffix


        if is_contrail_off:
            label = label.replace(" (Contrail CoCiP)", "").replace(" (Contrail aCCF)", "")
            label = label.replace(" (CoCiP)", "").replace(" (aCCF)", "")

        title_parts.append(label)

    # Add model label at the end ONLY if >1 metrics from a group
    if not is_contrail_off:
        if len(used_cocip) > 1:
            title_parts.append("(Contrail CoCiP)")
        elif len(used_accf) > 1:
            title_parts.append("(Contrail aCCF)")

    plot_title = ", ".join(title_parts) + " Climate Impact Relative to CFM1990"
    plt.title(plot_title)
    # Font sizes
    plt.xticks(fontsize=12)  # x-axis tick labels
    plt.yticks(fontsize=12)  # y-axis tick labels
    plt.ylabel(ax.get_ylabel(), fontsize=13)  # y-axis label
    plt.title(plot_title, fontsize=14)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    metric_abbreviations = [legend_titles.get(m, m.replace("_relative_change", "").replace("_", "")) for m in metrics]
    metric_str = "_".join(metric_abbreviations)  # Combine them with underscores

    filename = f"results_report/boxplot/boxplot_grouped_{df_name}_{metric_str}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300)
    print(f"Saved grouped plot: {filename}")


# plot_grouped_boxplot_v6(results_df_changes, "results_df", metrics=['nox_impact_sum_relative_change'])
#
# plot_grouped_boxplot_v6(results_df_changes, "results_df", metrics=['nox_impact_sum_relative_change', 'h2o_impact_sum_relative_change'])
#
# plot_grouped_boxplot_v6(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['contrail_atr20_cocip_sum_relative_change'])
# plot_grouped_boxplot_v6(results_df_changes, "results_df", metrics=['nox_impact_sum_relative_change', 'h2o_impact_sum_relative_change',
#                           'co2_impact_cons_sum_relative_change'])

plot_grouped_boxplot_v6(results_df_changes, "results_df", metrics=['co2_impact_cons_sum_relative_change', 'nox_impact_sum_relative_change', 'h2o_impact_sum_relative_change'])
plot_grouped_boxplot_v6(results_df_changes, "results_df", metrics=['nox_impact_sum_relative_change', 'h2o_impact_sum_relative_change'])
# plot_grouped_boxplot_v6(results_df_changes, "results_df_nm", metrics=['co2_impact_cons_sum_relative_change', 'nox_impact_sum_relative_change', 'h2o_impact_sum_relative_change'])
plot_grouped_boxplot_v6(results_df_changes, "results_df", metrics=['co2_impact_cons_sum_relative_change', 'nox_impact_sum_relative_change'])
# plt.show()
plot_grouped_boxplot_v6(contrail_no_accf_changes, "contrail_no_accf", metrics=['nox_impact_sum_relative_change'])
plot_grouped_boxplot_v6(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change'])
plot_grouped_boxplot_v6(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change'])

plot_grouped_boxplot_v6(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['h2o_impact_sum_relative_change','nox_impact_sum_relative_change','climate_non_co2_cocip_relative_change'], ylim=[-105,5])
plot_grouped_boxplot_v6(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change'])
plot_grouped_boxplot_v6(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_total_cons_cocip_relative_change'])
plot_grouped_boxplot_v6(contrail_no_cocip_changes, "no_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change', 'climate_total_cons_cocip_relative_change'], ylim=[-105,5])

plot_grouped_boxplot_v6(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
plot_grouped_boxplot_v6(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['h2o_impact_sum_relative_change','nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change','climate_non_co2_cocip_relative_change'], ylim=[-105,50])
plot_grouped_boxplot_v6(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['contrail_atr20_cocip_sum_relative_change'], ylim=[-105, 40])
# plot_grouped_boxplot_v6(contrail_yes_cocip_changes, "contrail_yes_cocip_nm", metrics=['contrail_atr20_cocip_sum_relative_change'], ylim=[-105, 40])
plt.show()

plot_grouped_boxplot_v6(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change'])
plot_grouped_boxplot_v6(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_total_cons_cocip_relative_change'])
plot_grouped_boxplot_v6(contrail_yes_cocip_changes, "yes_cocip", metrics=['climate_non_co2_cocip_relative_change', 'climate_total_cons_cocip_relative_change'])
plot_grouped_boxplot_v6(contrail_yes_cocip_changes, "yes_cocip", metrics=['climate_non_co2_cocip_relative_change','co2_impact_cons_sum_relative_change', 'climate_total_cons_cocip_relative_change'], ylim=[-105,50])
plot_grouped_boxplot_v6(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
plot_grouped_boxplot_v6(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['h2o_impact_sum_relative_change' ,'nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change', 'climate_non_co2_accf_cocip_pcfa_relative_change'], ylim=[-105,50])
plot_grouped_boxplot_v6(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change'])
plot_grouped_boxplot_v6(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change'])
plot_grouped_boxplot_v6(contrail_yes_accf_changes, "yes_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'climate_total_cons_accf_cocip_pcfa_relative_change'])
plot_grouped_boxplot_v6(contrail_yes_accf_changes, "yes_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change', 'climate_total_cons_accf_cocip_pcfa_relative_change'], ylim=[-105,50])
plot_grouped_boxplot_v6(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
plt.show()
contrail_night_cocip_changes = contrail_yes_cocip_changes[contrail_yes_cocip_changes['diurnal'] == 'nighttime'].copy()
contrail_day_cocip_changes = contrail_yes_cocip_changes[contrail_yes_cocip_changes['diurnal'] == 'daytime'].copy()
plot_grouped_boxplot_v6(contrail_night_cocip_changes, "contrail_night_cocip", metrics=['h2o_impact_sum_relative_change','nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change','climate_non_co2_cocip_relative_change'], ylim=[-105,50])
plot_grouped_boxplot_v6(contrail_day_cocip_changes, "contrail_day_cocip", metrics=['h2o_impact_sum_relative_change','nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change','climate_non_co2_cocip_relative_change'], ylim=[-105,50])



def plot_emissions_boxplot(df, df_name, metrics=['nox_impact_sum', 'contrail_atr20_cocip_sum', 'contrail_atr20_accf_cocip_pcfa_sum']):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    supported_metrics = {
        'nox_impact_sum',
        'contrail_atr20_cocip_sum',
        'contrail_atr20_accf_cocip_pcfa_sum'
    }
    assert set(metrics).issubset(supported_metrics), f"Only supports these metrics: {supported_metrics}"

    df = df.copy()

    # Engine display names
    engine_display_names = {
        'GTF1990': 'CFM1990',
        'GTF2000': 'CFM2008',
        'GTF': 'GTF',
        'GTF2035': 'GTF2035',
        'GTF2035_wi': 'GTF2035WI',
    }

    saf_levels = [20, 100]
    engines = ['GTF1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi']
    df = df[df['engine'].isin(engines)].copy()

    # Remove zero values for all metrics
    for metric in metrics:
        if metric in df.columns:
            df = df[df[metric] != 0]

    # Engine label with SAF
    df['engine_display'] = df.apply(
        lambda row: f"{engine_display_names[row['engine']]}" +
                    (f"\n{row['saf_level']}" if row['engine'] in ['GTF2035', 'GTF2035_wi']
                                              and row['saf_level'] in saf_levels else ""),
        axis=1
    )

    # Melt data
    df_melted = df.melt(
        id_vars=['engine_display'],
        value_vars=metrics,
        var_name='metric',
        value_name='value'
    )

    df_melted['abs_value'] = df_melted['value'].abs()

    # Color map
    metric_color_map = {
        'nox_impact_sum': 'tab:blue',
        'contrail_atr20_cocip_sum': 'tab:green',
        'contrail_atr20_accf_cocip_pcfa_sum': 'tab:red'
    }

    legend_titles = {
        'nox_impact_sum': 'NOx Impact',
        'contrail_atr20_cocip_sum': 'Contrail Impact (CoCiP)',
        'contrail_atr20_accf_cocip_pcfa_sum': 'Contrail Impact (aCCF)'
    }

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=df_melted,
        x='engine_display',
        y='abs_value',
        hue='metric',
        palette=metric_color_map,
        showfliers=True,
        width=0.6
    )

    ax.set_yscale('log')
    ax.set_ylabel("Absolute Climate Impact P-ATR20 (K)")
    ax.set_xlabel("")
    # ax.set_title("NOx and Contrail Climate Impact")
    plot_title = "NOx and Contrail Climate Impact"  # or set dynamically if needed
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel(ax.get_ylabel(), fontsize=13)
    plt.title(plot_title, fontsize=14)

    ax.grid(True, linestyle='--', alpha=0.5)

    # Mean markers
    xticks = ax.get_xticks()
    metric_count = len(metrics)
    box_positions = {}
    for i, engine_display in enumerate(sorted(df['engine_display'].unique())):
        base_x = xticks[i]
        for j, metric in enumerate(metrics):
            offset = (-0.5 + (j + 0.5) / metric_count) * 0.6
            box_positions[(engine_display, metric)] = base_x + offset

    for (engine, metric), group in df_melted.groupby(['engine_display', 'metric']):
        x = box_positions.get((engine, metric))
        if x is not None:
            plt.scatter(x, group['abs_value'].mean(), color='black', marker='D', s=30, zorder=10)

    # Legend
    legend_elements = [
        Patch(facecolor=metric_color_map[m], alpha=0.7, label=legend_titles.get(m, m.replace('_', ' ').capitalize()))
        for m in metrics
    ]
    legend_elements.append(Line2D([0], [0], marker='D', color='black', label='Mean', markersize=6, linestyle='None'))
    plt.legend(handles=legend_elements, title=None)

    plt.tight_layout()
    filename = f"results_report/boxplot/emissions_boxplot_{df_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300)
    print(f"Saved emissions plot: {filename}")


plot_emissions_boxplot(contrail_yes_cocip_changes, "contrail_yes_cocip")
plot_emissions_boxplot(results_df_changes, "contrail_results_df")
plt.show()
def export_relative_difference_csv(df, df_name, metrics=['climate_total_cons_sum_relative_change']):
    """
    Computes the relative difference (2 * rasd) / (1 - rasd) * 100 for specified metrics
    and saves it as a CSV file, correctly handling SAF levels.

    Parameters:
        df (DataFrame): The input dataframe containing RASD values.
        df_name (str): The name of the CSV file to save.
        metrics (list): A list of column names to be included in the CSV.
    """

    # Engines to include in the CSV
    engines_to_include = ['GTF1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi']
    saf_levels = [20, 100]  # SAF levels to be considered separately

    # Create the engine display column with SAF level distinctions
    if df.empty:
        print(f"⚠️ WARNING: DataFrame {df_name} is empty. Skipping export.")
        return
    df['engine_display'] = df.apply(
        lambda row: f"{engine_display_names[row['engine']]}" + (
            f" - {row['saf_level']}" if row['engine'] in ['GTF2035', 'GTF2035_wi'] and row['saf_level'] in saf_levels else ""
        ), axis=1
    )
    # print(df)
    # Filter relevant engines
    df_filtered = df[df['engine'].isin(engines_to_include)]
    # print(df_filtered)
    # Compute mean per engine type for each metric
    grouped = df_filtered.groupby("engine_display")[metrics].median()


    for metric in metrics:
        grouped[metric] *= 100

    # Rename columns using axis_titles dictionary (now correctly legend_titles in your code)
    grouped = grouped.rename(columns=legend_titles)

    # Reset index for saving to CSV
    grouped = grouped.reset_index()

    # Define the correct order for engine_display (ensuring SAF 20 comes before SAF 100)
    ordered_engines = [
        "CFM1990", "CFM2008", "GTF",
        "GTF2035", "GTF2035 - 20", "GTF2035 - 100",
        "GTF2035WI", "GTF2035WI - 20", "GTF2035WI - 100"
    ]

    # Ensure correct ordering
    grouped = grouped.set_index("engine_display").reindex(ordered_engines).reset_index()

    # Define the filename format
    filename = f'results_report/climate/{df_name}_rad_vs_gtf1990.csv'

    # Save as CSV
    grouped.to_csv(filename, index=False)
    print(f"CSV saved: {filename}")


metrics_csv_contrail_yes_cocip = ['h2o_impact_sum_relative_change','nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change',
                'climate_non_co2_cocip_relative_change',
               'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change',
                'climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'
               ]

metrics_csv_contrail_no_cocip = ['h2o_impact_sum_relative_change','nox_impact_sum_relative_change',
                'climate_non_co2_cocip_relative_change',
               'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change',
                'climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'
               ]

metrics_csv_contrail_yes_accf = ['h2o_impact_sum_relative_change','nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change',
                'climate_non_co2_accf_cocip_pcfa_relative_change',
               'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change',
                'climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'
               ]

metrics_csv_contrail_no_accf = ['h2o_impact_sum_relative_change','nox_impact_sum_relative_change',
                'climate_non_co2_accf_cocip_pcfa_relative_change',
               'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change',
                'climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'
               ]



export_relative_difference_csv(contrail_yes_accf_changes, "contrail_yes_accf", metrics_csv_contrail_yes_accf)
#
export_relative_difference_csv(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics_csv_contrail_yes_cocip)
#
export_relative_difference_csv(contrail_no_accf_changes, "contrail_no_accf", metrics_csv_contrail_no_accf)
export_relative_difference_csv(contrail_no_cocip_changes, "contrail_no_cocip", metrics_csv_contrail_no_cocip)

# """DIURNAL"""
contrail_yes_accf_day = contrail_yes_accf_changes[contrail_yes_accf_changes['diurnal'] == 'daytime'].copy()
contrail_yes_accf_night = contrail_yes_accf_changes[contrail_yes_accf_changes['diurnal'] == 'nighttime'].copy()

contrail_no_accf_day = contrail_no_accf_changes[contrail_no_accf_changes['diurnal'] == 'daytime'].copy()
contrail_no_accf_night = contrail_no_accf_changes[contrail_no_accf_changes['diurnal'] == 'nighttime'].copy()

contrail_no_cocip_day = contrail_no_cocip_changes[contrail_no_cocip_changes['diurnal'] == 'daytime'].copy()
contrail_no_cocip_night = contrail_no_cocip_changes[contrail_no_cocip_changes['diurnal'] == 'nighttime'].copy()

contrail_yes_cocip_day = contrail_yes_cocip_changes[contrail_yes_cocip_changes['diurnal'] == 'daytime'].copy()
contrail_yes_cocip_night = contrail_yes_cocip_changes[contrail_yes_cocip_changes['diurnal'] == 'nighttime'].copy()

print(contrail_yes_accf_changes.columns)
#
# #rasd
# plot_rasd_barplot(contrail_yes_day, "contrail_yes_day", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rasd_barplot(contrail_yes_night, "contrail_yes_night", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
#
# plot_rasd_barplot(contrail_yes_all_day, "contrail_yes_all_day", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rasd_barplot(contrail_yes_all_night, "contrail_yes_all_night", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
#
#
# plot_rasd_barplot(contrail_no_day, "contrail_no_day", metrics=['nox_impact_sum_relative_change','co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rasd_barplot(contrail_no_night, "contrail_no_night", metrics=['nox_impact_sum_relative_change','co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# #rad
# plot_rad_barplot(contrail_yes_accf_day, "contrail_yes_accf_day", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
# plot_rad_barplot(contrail_yes_accf_night, "contrail_yes_accf_night", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
#
# plot_rad_barplot(contrail_yes_all_day, "contrail_yes_all_day", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_yes_all_night, "contrail_yes_all_night", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
#
# plot_rad_barplot(contrail_no_day, "contrail_no_day", metrics=['nox_impact_sum_relative_change','co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_no_night, "contrail_no_night", metrics=['nox_impact_sum_relative_change','co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
#
# #tables
export_relative_difference_csv(contrail_yes_accf_day, "contrail_yes_accf_day", metrics_csv_contrail_yes_accf)
export_relative_difference_csv(contrail_yes_accf_night, "contrail_yes_accf_night", metrics_csv_contrail_yes_accf)

export_relative_difference_csv(contrail_no_accf_day, "contrail_no_accf_day", metrics_csv_contrail_no_accf)
export_relative_difference_csv(contrail_no_accf_night, "contrail_no_accf_night", metrics_csv_contrail_no_accf)

export_relative_difference_csv(contrail_no_cocip_day, "contrail_no_cocip_day", metrics_csv_contrail_no_cocip)
export_relative_difference_csv(contrail_no_cocip_night, "contrail_no_cocip_night", metrics_csv_contrail_no_cocip)

export_relative_difference_csv(contrail_yes_cocip_day, "contrail_yes_cocip_day", metrics_csv_contrail_yes_cocip)
export_relative_difference_csv(contrail_yes_cocip_night, "contrail_yes_cocip_night", metrics_csv_contrail_yes_cocip)

"""SEASON"""
def assign_season_astro_simple(df):
    """
    Assigns a 'season_astro' column based on the four fixed season dates.
    Handles special cases for 'GRU → LIM' (Southern Hemisphere) and 'SIN → MAA'.
    """

    # Convert date column to datetime if not already
    df['season'] = pd.to_datetime(df['season'])

    # Define the season mapping for normal flights (Northern Hemisphere)
    season_mapping = {
        '2023-02-06': 'winter',
        '2023-05-05': 'spring',
        '2023-08-06': 'summer',
        '2023-11-06': 'autumn'
    }

    # Define the season mapping for GRU → LIM (Southern Hemisphere: Swap Summer & Winter, Spring & Autumn)
    season_mapping_gru_lim = {
        '2023-02-06': 'summer',
        '2023-05-05': 'autumn',
        '2023-08-06': 'winter',
        '2023-11-06': 'spring'
    }



    # Function to apply correct season mapping
    def get_season(row):
        date_str = row['season'].strftime('%Y-%m-%d')  # Convert to string for mapping
        trajectory = row['trajectory']

        if trajectory == 'gru_lim':
            return season_mapping_gru_lim.get(date_str, 'Unknown')
        else:
            return season_mapping.get(date_str, 'Unknown')

    # Apply function to DataFrame
    df['season_astro'] = df.apply(get_season, axis=1)

    return df
#
contrail_yes_accf_changes = assign_season_astro_simple(contrail_yes_accf_changes)
contrail_yes_cocip_changes = assign_season_astro_simple(contrail_yes_cocip_changes)
contrail_no_accf_changes = assign_season_astro_simple(contrail_no_accf_changes)
contrail_no_cocip_changes = assign_season_astro_simple(contrail_no_cocip_changes)
#
contrail_yes_accf_winter = contrail_yes_accf_changes[contrail_yes_accf_changes['season_astro'] == 'winter'].copy()
contrail_yes_accf_spring = contrail_yes_accf_changes[contrail_yes_accf_changes['season_astro'] == 'spring'].copy()
contrail_yes_accf_summer = contrail_yes_accf_changes[contrail_yes_accf_changes['season_astro'] == 'summer'].copy()
contrail_yes_accf_autumn = contrail_yes_accf_changes[contrail_yes_accf_changes['season_astro'] == 'autumn'].copy()

contrail_yes_cocip_winter = contrail_yes_cocip_changes[contrail_yes_cocip_changes['season_astro'] == 'winter'].copy()
contrail_yes_cocip_spring = contrail_yes_cocip_changes[contrail_yes_cocip_changes['season_astro'] == 'spring'].copy()
contrail_yes_cocip_summer = contrail_yes_cocip_changes[contrail_yes_cocip_changes['season_astro'] == 'summer'].copy()
contrail_yes_cocip_autumn = contrail_yes_cocip_changes[contrail_yes_cocip_changes['season_astro'] == 'autumn'].copy()

contrail_no_accf_winter = contrail_no_accf_changes[contrail_no_accf_changes['season_astro'] == 'winter'].copy()
contrail_no_accf_spring = contrail_no_accf_changes[contrail_no_accf_changes['season_astro'] == 'spring'].copy()
contrail_no_accf_summer = contrail_no_accf_changes[contrail_no_accf_changes['season_astro'] == 'summer'].copy()
contrail_no_accf_autumn = contrail_no_accf_changes[contrail_no_accf_changes['season_astro'] == 'autumn'].copy()

contrail_no_cocip_winter = contrail_no_cocip_changes[contrail_no_cocip_changes['season_astro'] == 'winter'].copy()
contrail_no_cocip_spring = contrail_no_cocip_changes[contrail_no_cocip_changes['season_astro'] == 'spring'].copy()
contrail_no_cocip_summer = contrail_no_cocip_changes[contrail_no_cocip_changes['season_astro'] == 'summer'].copy()
contrail_no_cocip_autumn = contrail_no_cocip_changes[contrail_no_cocip_changes['season_astro'] == 'autumn'].copy()
#
# plot_rad_barplot(contrail_yes_winter, "contrail_yes_winter", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_yes_spring, "contrail_yes_spring", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_yes_summer, "contrail_yes_summer", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_yes_autumn, "contrail_yes_autumn", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
#
export_relative_difference_csv(contrail_yes_accf_winter, "contrail_yes_accf_winter", metrics_csv_contrail_yes_accf)
export_relative_difference_csv(contrail_yes_accf_spring, "contrail_yes_accf_spring", metrics_csv_contrail_yes_accf)
export_relative_difference_csv(contrail_yes_accf_summer, "contrail_yes_accf_summer", metrics_csv_contrail_yes_accf)
export_relative_difference_csv(contrail_yes_accf_autumn, "contrail_yes_accf_autumn", metrics_csv_contrail_yes_accf)
#
export_relative_difference_csv(contrail_yes_cocip_winter, "contrail_yes_cocip_winter", metrics_csv_contrail_yes_cocip)
export_relative_difference_csv(contrail_yes_cocip_spring, "contrail_yes_cocip_spring", metrics_csv_contrail_yes_cocip)
export_relative_difference_csv(contrail_yes_cocip_summer, "contrail_yes_cocip_summer", metrics_csv_contrail_yes_cocip)
export_relative_difference_csv(contrail_yes_cocip_autumn, "contrail_yes_cocip_autumn", metrics_csv_contrail_yes_cocip)
# plot_rad_barplot(contrail_no_winter, "contrail_no_winter", metrics=['nox_impact_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_no_spring, "contrail_no_spring", metrics=['nox_impact_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_no_summer, "contrail_no_summer", metrics=['nox_impact_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_no_autumn, "contrail_no_autumn", metrics=['nox_impact_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
#
export_relative_difference_csv(contrail_no_accf_winter, "contrail_no_accf_winter", metrics_csv_contrail_no_accf)
export_relative_difference_csv(contrail_no_accf_spring, "contrail_no_accf_spring", metrics_csv_contrail_no_accf)
export_relative_difference_csv(contrail_no_accf_summer, "contrail_no_accf_summer", metrics_csv_contrail_no_accf)
export_relative_difference_csv(contrail_no_accf_autumn, "contrail_no_accf_autumn", metrics_csv_contrail_no_accf)

export_relative_difference_csv(contrail_no_cocip_winter, "contrail_no_cocip_winter", metrics_csv_contrail_no_cocip)
export_relative_difference_csv(contrail_no_cocip_spring, "contrail_no_cocip_spring", metrics_csv_contrail_no_cocip)
export_relative_difference_csv(contrail_no_cocip_summer, "contrail_no_cocip_summer", metrics_csv_contrail_no_cocip)
export_relative_difference_csv(contrail_no_cocip_autumn, "contrail_no_cocip_autumn", metrics_csv_contrail_no_cocip)


contrail_yes_cocip_day.to_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_day.csv", index=False)
contrail_yes_cocip_night.to_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_night.csv", index=False)

contrail_yes_cocip_winter.to_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_winter.csv", index=False)
contrail_yes_cocip_spring.to_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_spring.csv", index=False)
contrail_yes_cocip_summer.to_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_summer.csv", index=False)
contrail_yes_cocip_autumn.to_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_autumn.csv", index=False)



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
def plot_nvpm_correction_barplot(df):
    metric_accf = "contrail_atr20_accf_sum_relative_change"
    metric_pcfa = "contrail_atr20_accf_cocip_pcfa_sum_relative_change"

    # Set colors
    base_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]  # tab:red
    faded_color = mcolors.to_rgba(base_color, alpha=0.7)
    hatch_style = "//"

    # Define display names
    engine_display_names = {
        'CFM1990': 'CFM1990',
        'GTF2000': 'CFM2008',
        'GTF': 'GTF',
        'GTF2035': 'GTF2035',
        'GTF2035_wi': 'GTF2035WI'
    }

    saf_levels = [20, 100]
    engines_to_plot = [
        "CFM1990", "CFM2008", "GTF",
        "GTF2035", "GTF2035\n20", "GTF2035\n100",
        "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
    ]

    # Fix engine_display names with SAF levels
    df['engine_display'] = df.apply(
        lambda row: (
            f"{engine_display_names.get(row['engine'], row['engine'])}" +
            (f"\n{row['saf_level']}" if row['engine'] in ['GTF2035', 'GTF2035_wi']
             and row['saf_level'] in saf_levels else "")
        ), axis=1
    )
    print(df['engine_display'])

    # Filter and group
    df_filtered = df[df['engine_display'].isin(engines_to_plot)]
    grouped = df_filtered.groupby("engine_display")[[metric_accf, metric_pcfa]].median()
    for name, group in df_filtered.groupby("engine_display"):
        print(f"\n{name}")
        print(group[[metric_accf, metric_pcfa]])

    for metric in [metric_accf, metric_pcfa]:
        grouped[metric] = grouped[metric] * 100 + 100

    # Ensure baseline
    grouped.loc["CFM1990", [metric_accf, metric_pcfa]] = 100

    grouped = grouped.reset_index()
    grouped = grouped.set_index("engine_display").reindex(engines_to_plot).reset_index()

    # Plotting
    x = np.arange(len(engines_to_plot))
    width = 0.35

    accf_vals = grouped[metric_accf]
    pcfa_vals = grouped[metric_pcfa]

    plt.figure(figsize=(12, 6))
    plt.bar(x, pcfa_vals, width=width, color=faded_color, label="Contrail (aCCF)", edgecolor=faded_color)

    for i in range(len(x)):
        val_accf = accf_vals[i]
        val_pcfa = pcfa_vals[i]

        if not np.isnan(val_accf) and not np.isnan(val_pcfa) and val_accf > val_pcfa:
            delta = val_accf - val_pcfa
            plt.bar(x[i], delta, bottom=val_pcfa, width=width,
                    color="white", edgecolor=faded_color, hatch=hatch_style,
                    label="nvPM reduction" if i == 0 else "", zorder=3)

    plt.xticks(x, engines_to_plot)
    plt.ylabel("Relative Climate Impact (%)")
    plt.title("Contrail (aCCF) Climate Impact Relative to CFM1990")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Legend
    handles = [
        Patch(facecolor=faded_color, label="Contrail (aCCF)"),
        Patch(facecolor="white", edgecolor=faded_color, hatch=hatch_style, label="nvPM Reduction Effect")
    ]
    plt.legend(handles=handles)

    plt.tight_layout()
    plt.savefig("results_report/barplot/contrail_accf_nvpm_corrected.png", dpi=300)
    # plt.show()




plot_nvpm_correction_barplot(contrail_yes_accf_changes)

def plot_grouped_boxplot_nvpm(df, df_name, metrics):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Affine2D
    import matplotlib.colors as mcolors
    plt.clf()
    plt.close('all')
    df = df.copy()
    is_contrail_off = 'no' in df_name.lower()
    # Display names
    engine_display_names = {
        'GTF1990': 'CFM1990',
        'GTF2000': 'CFM2008',
        'GTF': 'GTF',
        'GTF2035': 'GTF2035',
        'GTF2035_wi': 'GTF2035WI',
        'CFM_joined': 'CFM1990/CFM2008'
    }

    # Color mapping
    metric_color_map = {
        "nox_impact": "tab:blue",
        "co2_impact": "tab:orange",
        "climate_non_co2": "tab:purple",
        "climate_total": "tab:cyan",
        "contrail_atr20_accf_cocip_pcfa_sum_relative_change": "tab:red",
        "contrail_atr20_accf_sum_relative_change": "#756bb1",
        "contrail_atr20_cocip": "tab:green",
        "h2o_impact": "tab:grey"
    }

    def get_metric_color(metric_name):
        # Exact match first
        if metric_name in metric_color_map:
            return metric_color_map[metric_name]
        # Fallback to prefix matching if needed
        for key in metric_color_map:
            if metric_name.startswith(key):
                return metric_color_map[key]
        return "tab:gray"

    # Detect midpoint metrics
    midpoint_metrics = {}
    new_metrics = []
    for metric in metrics:
        if '_cons_' in metric:
            opti_metric = metric.replace('_cons_', '_opti_')
            if opti_metric in df.columns:
                midpoint_name = metric.replace('_cons_', '_midpoint_')
                df[midpoint_name] = df[metric]  # copy data
                midpoint_metrics[midpoint_name] = {
                    'cons': metric,
                    'opti': opti_metric
                }
                new_metrics.append(midpoint_name)
            else:
                new_metrics.append(metric)
        else:
            new_metrics.append(metric)

    metrics = new_metrics

    # Metrics eligible for CFM joining
    joinable_metrics = {
        'nox_impact_sum_relative_change',
        'h2o_impact_sum_relative_change',
        'co2_impact_cons_sum_relative_change',
        'co2_impact_opti_sum_relative_change',
        'co2_impact_midpoint_sum_relative_change'
    }
    should_join_cfm = all(m in joinable_metrics for m in metrics)

    if should_join_cfm:
        df['engine'] = df['engine'].replace({'GTF1990': 'CFM_joined', 'GTF2000': 'CFM_joined'})
        engines_to_plot = ['CFM_joined', 'GTF', 'GTF2035', 'GTF2035_wi']
        x_order = [
            "CFM1990/CFM2008", "GTF",
            "GTF2035", "GTF2035\n20", "GTF2035\n100",
            "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
        ]
    else:
        engines_to_plot = ['GTF1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi']
        x_order = [
            "CFM1990", "CFM2008", "GTF",
            "GTF2035", "GTF2035\n20", "GTF2035\n100",
            "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
        ]

    saf_levels = [20, 100]
    df_filtered = df[df['engine'].isin(engines_to_plot)].copy()

    df_filtered['engine_display'] = df_filtered.apply(
        lambda row: f"{engine_display_names[row['engine']]}" +
                    (f"\n{row['saf_level']}" if row['engine'] in ['GTF2035', 'GTF2035_wi']
                                                and row['saf_level'] in saf_levels else ""),
        axis=1
    )

    for metric in metrics:
        df_filtered[metric] = df_filtered[metric] * 100

    df_melted = df_filtered.melt(
        id_vars=['engine_display'],
        value_vars=metrics,
        var_name='metric',
        value_name='value'
    )

    current_color_map = {metric: get_metric_color(metric) for metric in metrics}
    legend_labels = {}
    for metric in metrics:
        label = legend_titles.get(metric, metric.replace("_", " "))
        if is_contrail_off:
            label = label.replace(" (Contrail CoCiP)", "").replace(" (Contrail aCCF)", "")
            label = label.replace(" (CoCiP)", "").replace(" (aCCF)", "")
        legend_labels[metric] = label

    if len(metrics) == 1:
        box_width = 0.4  # Narrower for single metric
    else:
        box_width = 0.7  # Default width for multiple metrics
    plt.figure(figsize=(12, 6))
    metric_count = len(metrics)
    ax = plt.gca()  # Get axis early so we can use it for drawing bars



    ax = sns.boxplot(
        data=df_melted,
        x='engine_display',
        y='value',
        hue='metric',
        order=x_order,
        palette=current_color_map,
        showfliers=False,
        width=box_width
    )

    for patch in ax.patches:
        facecolor = patch.get_facecolor()
        patch.set_facecolor((*facecolor[:3], 0.7))


    # plt.title("Relative Climate Impact Comparison")
    plt.ylabel("Relative Climate Impact (%)")
    plt.xlabel("")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=0, ha="center")


    for i, engine_display in enumerate(x_order):
        base_x = i
        for j, metric in enumerate(metrics):
            offset = (-0.5 + (j + 0.5) / metric_count) * box_width
            x = base_x + offset

            engine_mask = df_filtered['engine_display'] == engine_display

            if metric in midpoint_metrics:
                cons = midpoint_metrics[metric]['cons']
                opti = midpoint_metrics[metric]['opti']
                cons_vals = df_filtered.loc[engine_mask, cons] * 100
                opti_vals = df_filtered.loc[engine_mask, opti] * 100
                all_vals = pd.concat([cons_vals, opti_vals])
            else:
                all_vals = df_filtered.loc[engine_mask, metric]

            if all_vals.empty:
                continue

            if metric in midpoint_metrics:
                q3_cons = cons_vals.quantile(0.75)
                q3_opti = opti_vals.quantile(0.75)
                bar_height = (q3_cons + q3_opti) / 2
            else:
                bar_height = all_vals.quantile(0.75)

            q1 = all_vals.quantile(0.25)
            q3 = all_vals.quantile(0.75)

            # Skip drawing the bar only if the IQR crosses zero
            if q1 < 0 < q3 or q3 < 0 < q1:
                continue

            base_rgb = mcolors.to_rgba(get_metric_color(metric))
            bar_color = (*base_rgb[:3], 0.3)
            bar_width = box_width / metric_count

            ax.add_patch(Rectangle(
                (x - bar_width / 2, 0),
                bar_width,
                bar_height,
                facecolor=bar_color,
                edgecolor=None,
                linewidth=0,
                zorder=1.5
            ))

    mean_marker_size = 30 if metric_count == 1 else 20 if metric_count == 2 else 10

    xticks = ax.get_xticks()
    box_positions = {}
    for i, engine_display in enumerate(x_order):
        base_x = xticks[i]
        for j, metric in enumerate(metrics):
            offset = (-0.5 + (j + 0.5) / metric_count) * box_width
            box_positions[(engine_display, metric)] = base_x + offset

    suppress_means = (
            'climate_non_co2_cocip_relative_change' in metrics or
            'climate_non_co2_accf_cocip_pcfa_relative_change' in metrics or
            'climate_total_midpoint_accf_cocip_pcfa_relative_change' in metrics or
            'climate_total_midpoint_cocip_relative_change' in metrics)
    print(metrics)
    if not suppress_means:
        for (engine, metric), group in df_melted.groupby(['engine_display', 'metric']):
            if (engine, metric) not in box_positions:
                continue
            if metric == 'climate_non_co2_cocip_relative_change':
                continue  # Skip mean for this metric
            x = box_positions[(engine, metric)]
            """OPTI AND CONS SAF approach have same spread of data -> translate box and other components to show the mean of opti cons (so not cons or opt but mean scenario )"""
            if metric in midpoint_metrics:
                cons = midpoint_metrics[metric]['cons']
                opti = midpoint_metrics[metric]['opti']
                engine_mask = df_filtered['engine_display'] == engine
                mean_cons = df_filtered.loc[engine_mask, cons] * 100
                mean_opti = df_filtered.loc[engine_mask, opti] * 100
                mean_val = (mean_cons.mean() + mean_opti.mean()) / 2
            else:
                mean_val = group['value'].mean()

            plt.scatter(x, mean_val, color='black', marker='D',
                        s=mean_marker_size, zorder=10)

    # Shift entire boxplot vertically for midpoint metrics
    for metric in midpoint_metrics:
        cons = midpoint_metrics[metric]['cons']
        opti = midpoint_metrics[metric]['opti']

        for engine in df_filtered['engine_display'].unique():
            x = box_positions.get((engine, metric))
            engine_mask = df_filtered['engine_display'] == engine
            if not engine_mask.any():
                continue

            cons_median = df_filtered.loc[engine_mask, cons].median() * 100
            opti_median = df_filtered.loc[engine_mask, opti].median() * 100
            target_median = (cons_median + opti_median) / 2
            dy = target_median - cons_median

            # Precompute RGBA for current metric
            target_color = mcolors.to_rgba(current_color_map[metric], alpha=0.7)
            target_x = round(box_positions.get((engine, metric), -999), 2)

            for patch in reversed(ax.patches):  # reverse to avoid index shifting
                path = patch.get_path()
                verts = path.vertices
                transformed = patch.get_patch_transform().transform(verts)
                box_x_center = round(np.mean(transformed[:, 0]), 3)
                facecolor = patch.get_facecolor()

                # Debug print to understand the match failure
                # print(f"Checking patch: center={box_x_center}, target_x={target_x}, "
                #       f"color={facecolor}, target_color={target_color}")

                # Color difference
                color_diff = np.abs(np.array(facecolor) - np.array(target_color))
                # print(f"Color diff: {color_diff}")

                if np.isclose(box_x_center, target_x, atol=0.05) and \
                        np.allclose(facecolor, target_color, atol=0.15):
                    # print('test')
                    # Remove the original box
                    # print("Original box width:", max(transformed[:, 0]) - min(transformed[:, 0]))
                    transform = Affine2D().translate(0, dy) + ax.transData
                    patch.set_transform(transform)
                    break

            # Shift Line2D elements: median, whiskers, caps
            for line in ax.lines:
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) != 2 or len(ydata) != 2:
                    continue

                x_mid = sum(xdata) / 2
                if abs(x_mid - box_positions.get((engine, metric), -999)) > 0.01:
                    continue

                # median
                if ydata[0] == ydata[1] and abs(ydata[0] - cons_median) < 0.01:
                    transform = Affine2D().translate(0, dy) + ax.transData
                    line.set_transform(transform)
                # whiskers
                elif xdata[0] == xdata[1]:
                    transform = Affine2D().translate(0, dy) + ax.transData
                    line.set_transform(transform)
                # caps
                elif abs(ydata[0] - ydata[1]) < 1e-3:
                    transform = Affine2D().translate(0, dy) + ax.transData
                    line.set_transform(transform)



    # xticks = ax.get_xticks()
    # ax.set_xlim(xticks[0] - 0.05, xticks[-1] + 0.05)
    # Legend
    # --- Draw SAF Production Pathway Range Lines (from cons median to opti median) ---
    saf_line_drawn_for = set()
    # SAF line z-order logic
    saf_line_on_top = any(m in metrics for m in [
        'climate_total_midpoint_accf_cocip_pcfa_relative_change',
        'climate_total_midpoint_cocip_relative_change'
    ])
    zorder_saf = 20 if saf_line_on_top else 1.4

    final_handles = []
    saf_legend_added_for = set()

    for metric in metrics:
        color = current_color_map[metric]
        label = legend_titles.get(metric, metric.replace("_", " "))
        if 'no' in df_name.lower():
            label = label.split(' (')[0]  # Removes any parenthetical content
        patch = Patch(facecolor=color, alpha=0.7, label=label)
        final_handles.append(patch)

        if metric in midpoint_metrics and metric not in saf_legend_added_for:
            cons_col = midpoint_metrics[metric]['cons']
            opti_col = midpoint_metrics[metric]['opti']
            base_color = get_metric_color(metric)

            # SAF line color logic (apply special color only to total climate)
            if metric in ['climate_total_midpoint_cocip_relative_change',
                          'climate_total_midpoint_accf_cocip_pcfa_relative_change']:
                line_color = 'tab:blue'
                this_zorder = 20
            else:
                line_color = mcolors.to_rgba(base_color, alpha=1.0)
                this_zorder = 1.4

            for engine in df_filtered['engine_display'].unique():
                engine_mask = df_filtered['engine_display'] == engine
                if not engine_mask.any():
                    continue

                x = box_positions.get((engine, metric))
                if x is None:
                    continue

                cons_median = df_filtered.loc[engine_mask, cons_col].median() * 100
                opti_median = df_filtered.loc[engine_mask, opti_col].median() * 100

                if np.isclose(cons_median, opti_median, atol=0.01):
                    continue

                cap_width = box_width / metric_count * 1.0

                ax.plot([x, x], [cons_median, opti_median],
                        color=line_color, linewidth=1.5, zorder=this_zorder)

                ax.hlines([cons_median, opti_median],
                          x - cap_width / 2, x + cap_width / 2,
                          color=line_color, linewidth=1.5, zorder=this_zorder)

            # Only add SAF legend once per metric (and with correct color)
            final_handles.append(Line2D([0], [0],
                                        color=line_color,
                                        linewidth=1.5,
                                        label="SAF Production Pathway Range"))
            saf_legend_added_for.add(metric)

    # Mean marker at the end
    if not suppress_means:
        final_handles.append(Line2D([0], [0], marker='D', color='black',
                                    label='Mean', markersize=6, linestyle='None'))

    plt.legend(handles=final_handles, title=None)

    cocip_metrics = {
        'climate_total_cons_cocip_relative_change',
        'climate_total_opti_cocip_relative_change',
        'climate_total_midpoint_cocip_relative_change',
        'climate_non_co2_cocip_relative_change',
        'contrail_atr20_cocip_sum_relative_change'
    }

    accf_metrics = {
        'climate_total_cons_accf_cocip_pcfa_relative_change',
        'climate_total_opti_accf_cocip_pcfa_relative_change',
        'climate_total_midpoint_accf_cocip_pcfa_relative_change',
        'climate_non_co2_accf_cocip_pcfa_relative_change',
        'contrail_atr20_accf_cocip_pcfa_sum_relative_change'
    }

    used_cocip = [m for m in metrics if m in cocip_metrics]
    used_accf = [m for m in metrics if m in accf_metrics]

    title_parts = []
    seen_total = seen_co2 = seen_nonco2 = seen_contrail = False

    for metric in metrics:
        raw_label = metric_titles.get(metric, metric.replace("_relative_change", "").replace("_", " "))
        suffix = ""

        if not is_contrail_off:
            if metric in cocip_metrics and len(used_cocip) == 1:
                suffix = " (Contrail CoCiP)"
            elif metric in accf_metrics and len(used_accf) == 1:
                suffix = " (Contrail aCCF)"

        if metric.startswith("co2_impact") and not seen_co2:
            label = "CO₂" + suffix
            seen_co2 = True
        elif metric.startswith("h2o_impact") and "H₂O" not in title_parts:
            label = "H₂O" + suffix
        elif metric.startswith("nox_impact") and "NOx" not in title_parts:
            label = "NOx" + suffix
        elif "contrail" in metric and not seen_contrail:
            label = "Contrail" + suffix
            seen_contrail = True
        elif "non_co2" in metric and not seen_nonco2:
            label = "Non-CO₂" + suffix
            seen_nonco2 = True
        elif metric.startswith("climate_total") and not seen_total:
            label = "Total" + suffix
            seen_total = True
        else:
            label = raw_label + suffix


        if is_contrail_off:
            label = label.replace(" (Contrail CoCiP)", "").replace(" (Contrail aCCF)", "")
            label = label.replace(" (CoCiP)", "").replace(" (aCCF)", "")

        title_parts.append(label)

    # Add model label at the end ONLY if >1 metrics from a group
    if not is_contrail_off:
        if len(used_cocip) > 1:
            title_parts.append("(Contrail CoCiP)")
        elif len(used_accf) > 1:
            title_parts.append("(Contrail aCCF)")

    plot_title = "Contrail aCCF nvPM Correction Effect"
    plt.title(plot_title)
    # Font sizes
    plt.xticks(fontsize=12)  # x-axis tick labels
    plt.yticks(fontsize=12)  # y-axis tick labels
    plt.ylabel(ax.get_ylabel(), fontsize=13)  # y-axis label
    plt.title(plot_title, fontsize=14)
    plt.ylim(-105,85)
    yticks = np.arange(-100, 81, 20)  # Covers a little beyond your ylim for clean ticks
    plt.yticks(yticks)
    plt.tight_layout()
    metric_abbreviations = [legend_titles.get(m, m.replace("_relative_change", "").replace("_", "")) for m in metrics]
    metric_str = "_".join(metric_abbreviations)  # Combine them with underscores

    filename = f"results_report/boxplot/boxplot_grouped_{df_name}_{metric_str}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300)
    print(f"Saved grouped plot: {filename}")


plot_grouped_boxplot_nvpm(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['contrail_atr20_accf_sum_relative_change','contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
plt.show()