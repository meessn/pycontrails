import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
        'co2_impact_opti_sum_relative_change': 'CO₂'
    }

legend_titles = {
        'contrail_atr20_cocip_sum_relative_change': 'Contrail (CoCiP)',
        'contrail_atr20_accf_cocip_pcfa_sum_relative_change': 'Contrail (aCCF)',
        'nox_impact_sum_relative_change': 'NOx',
        'co2_impact_cons_sum_relative_change': 'CO2 Conservative',
        'co2_impact_opti_sum_relative_change': 'CO2 Optimistic',
        'climate_non_co2_cocip_relative_change': 'Non-CO₂ (Contrail CoCiP)',
        'climate_non_co2_accf_cocip_pcfa_relative_change': 'Non-CO₂ (Contrail aCCF)',
        'climate_total_cons_cocip_relative_change': 'Total (Contrail CoCiP) Conservative',
        'climate_total_cons_accf_cocip_pcfa_relative_change': 'Total (Contrail aCCF) Conservative',
        'climate_total_opti_cocip_relative_change': 'Total (Contrail CoCiP) Optimistic',
        'climate_total_opti_accf_cocip_pcfa_relative_change': 'Total (Contrail aCCF) Optimistic'
    }
# Function to generate bar chart with multiple metrics and error bars
# Function to generate bar chart with multiple metrics and error bars
def plot_rasd_barplot(df, df_name, metrics=['climate_total_cons_sum_relative_change']):
    """
    Plots a grouped bar chart of mean RASD values with error bars for multiple metrics in a given dataframe.

    Parameters:
        df (DataFrame): The input dataframe containing RASD values.
        df_name (str): Name of the dataframe (for saving the plot).
        metrics (list): A list of column names representing RASD values to be plotted.
    """
    engines_to_plot = ['GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi']
    saf_levels = [20, 100]  # Only GTF2035 variants get SAF levels

    # Generate a new column for display names including SAF levels
    df['engine_display'] = df.apply(
        lambda row: f"{engine_display_names[row['engine']]}" + (
            f"\n-{row['saf_level']}" if row['engine'] in ['GTF2035', 'GTF2035_wi'] and row[
                'saf_level'] in saf_levels else ""
        ), axis=1
    )

    # Filter relevant engines
    df_filtered = df[df['engine'].isin(engines_to_plot)]

    # Compute mean and standard deviation per engine type for each metric
    grouped = df_filtered.groupby("engine_display")[metrics].agg(['mean', 'std'])

    # Flatten MultiIndex columns
    grouped.columns = [f"{metric}_{agg}" for metric, agg in grouped.columns]
    grouped = grouped.reset_index()

    # Print value counts for verification
    print("Value counts per engine:")
    print(df_filtered['engine_display'].value_counts())

    # Define x-axis order
    x_order = [
        "CFM2008", "GTF",
        "GTF2035", "GTF2035\n-20", "GTF2035\n-100",
        "GTF2035WI", "GTF2035WI\n-20", "GTF2035WI\n-100"
    ]

    # Ensure ordering
    grouped = grouped.set_index("engine_display").reindex(x_order).reset_index()

    # Adjust bar width based on the number of metrics
    width = 0.6 if len(metrics) == 1 else 0.15  # Wider bars if only one metric
    x = np.arange(len(x_order))  # X locations for groups

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
        plt.bar(x + i * (width if len(metrics) > 1 else 0), grouped[f"{metric}_mean"], yerr=grouped[f"{metric}_std"],
                capsize=5,
                alpha=0.7, label=legend_label, width=width)

    # Mapping of metric names to title components


    # Generate title parts
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
    plot_title = " & ".join(title_parts) + " Climate Impact compared to CFM1990 (RASD)"

    # plt.xlabel("Engine Type")
    plt.ylabel("Mean RASD (Climate Impact compared to CFM1990) (Error: STD)")
    plt.title(plot_title)
    plt.xticks(x + (width * (len(metrics) - 1) / 2 if len(metrics) > 1 else 0), x_order, rotation=0, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save figure
    # Generate abbreviated filename
    metric_abbreviations = [legend_titles.get(m, m.replace("_relative_change", "").replace("_", "")) for m in metrics]
    metric_str = "_".join(metric_abbreviations)  # Combine them with underscores
    filename = f"results_report/barplot_error/rasd_barplot_{df_name}_{metric_str}.png".replace(" ", "_")  # Ensure no spaces

    # Save figure with the new filename
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")


plot_rasd_barplot(results_df_changes, "results_df", metrics=['co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change', 'nox_impact_sum_relative_change'])

plot_rasd_barplot(contrail_no_accf_changes, "contrail_no_accf", metrics=['nox_impact_sum_relative_change'])
plot_rasd_barplot(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])

plot_rasd_barplot(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['nox_impact_sum_relative_change'])
plot_rasd_barplot(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])


plot_rasd_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
plot_rasd_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change','climate_non_co2_cocip_relative_change'])
plot_rasd_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['contrail_atr20_cocip_sum_relative_change'])


plot_rasd_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])

plot_rasd_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
plot_rasd_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change','climate_non_co2_accf_cocip_pcfa_relative_change'])
plot_rasd_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])
plot_rasd_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['contrail_atr20_accf_cocip_pcfa_sum_relative_change'])


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

    # Compute mean and standard deviation per engine type for each metric
    grouped = df_filtered.groupby("engine_display")[metrics].mean()

    # Apply transformation: (2 * rasd) / (1 - rasd)
    for metric in metrics:
        grouped[metric] = (2 * grouped[metric]) / (1 - grouped[metric])

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

plot_rad_barplot(results_df_changes, "results_df", metrics=['co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change', 'nox_impact_sum_relative_change'])

plot_rad_barplot(contrail_no_accf_changes, "contrail_no_accf", metrics=['nox_impact_sum_relative_change'])
plot_rad_barplot(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_no_accf_changes, "contrail_no_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])

plot_rad_barplot(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['nox_impact_sum_relative_change'])
plot_rad_barplot(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_no_cocip_changes, "contrail_no_cocip", metrics=['climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])


plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change','climate_non_co2_cocip_relative_change'])
plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['contrail_atr20_cocip_sum_relative_change'])


plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_non_co2_cocip_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_yes_cocip_changes, "contrail_yes_cocip", metrics=['climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'])

plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change','climate_non_co2_accf_cocip_pcfa_relative_change'])
plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_non_co2_accf_cocip_pcfa_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'])
plot_rad_barplot(contrail_yes_accf_changes, "contrail_yes_accf", metrics=['contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
# plot_rad_barplot(
#     contrail_yes_changes[
#         (contrail_yes_changes['season'] == '2023-05-05') &
#         (contrail_yes_changes['diurnal'] == 'daytime')
#     ],
#     "worst",
#     metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change']
# )
# print('amount of spring daytime', contrail_yes_changes[
#         (contrail_yes_changes['season'] == '2023-05-05') &
#         (contrail_yes_changes['diurnal'] == 'daytime')])
#
# plot_rad_barplot(
#     contrail_yes_changes[
#         (contrail_yes_changes['season'] == '2023-05-05') &
#         (contrail_yes_changes['diurnal'] == 'daytime')
#     ],
#     "worst",
#     metrics=['climate_non_co2_relative_change' ,'climate_total_cons_sum_relative_change']
# )


# plot_rad_barplot(contrail_yes_all_changes, "contrail_yes_all", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
# plot_rad_barplot(contrail_yes_all_changes, "contrail_yes_all", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
# plot_rad_barplot(contrail_yes_all_changes, "contrail_yes_all", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])

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
    grouped = df_filtered.groupby("engine_display")[metrics].mean()

    # Apply transformation: (2 * rasd) / (1 - rasd) * 100
    for metric in metrics:
        grouped[metric] = (2 * grouped[metric]) / (1 - grouped[metric]) * 100

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


metrics_csv_contrail_yes_cocip = ['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change',
                'climate_non_co2_cocip_relative_change',
               'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change',
                'climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'
               ]

metrics_csv_contrail_no_cocip = ['nox_impact_sum_relative_change',
                'climate_non_co2_cocip_relative_change',
               'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change',
                'climate_total_cons_cocip_relative_change', 'climate_total_opti_cocip_relative_change'
               ]

metrics_csv_contrail_yes_accf = ['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change',
                'climate_non_co2_accf_cocip_pcfa_relative_change',
               'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change',
                'climate_total_cons_accf_cocip_pcfa_relative_change', 'climate_total_opti_accf_cocip_pcfa_relative_change'
               ]

metrics_csv_contrail_no_accf = ['nox_impact_sum_relative_change',
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
plot_rad_barplot(contrail_yes_accf_day, "contrail_yes_accf_day", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
plot_rad_barplot(contrail_yes_accf_night, "contrail_yes_accf_night", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_accf_cocip_pcfa_sum_relative_change'])
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
