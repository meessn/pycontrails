import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
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

# Identify flights where any engine produces a contrail
contrail_status = results_df.groupby(['trajectory', 'season', 'diurnal'])['contrail_atr20_cocip_sum'].sum().reset_index()

# Mark flights where at least one engine generates a contrail
contrail_status['contrail_formed'] = contrail_status['contrail_atr20_cocip_sum'] != 0

# Merge this back with the original dataset
results_df = results_df.merge(contrail_status[['trajectory', 'season', 'diurnal', 'contrail_formed']],
                              on=['trajectory', 'season', 'diurnal'], how='left')

# Classify flights correctly
contrail_no_df = results_df[results_df['contrail_formed'] == False]  # If no engines created a contrail
contrail_yes_df = results_df[results_df['contrail_formed'] == True]  # If at least one engine created a contrail

print(contrail_no_df)
print(contrail_yes_df)


# Baseline: GTF1990, saf_level = 0
baseline_df = results_df[(results_df['engine'] == 'GTF1990') & (results_df['saf_level'] == 0)]

# Define metrics for comparison
common_metrics = [
    'nox_impact_sum', 'co2_impact_cons_sum', 'co2_impact_opti_sum', 'h2o_impact_sum',
    'climate_non_co2', 'climate_total_cons_sum', 'climate_total_opti_sum'
]
contrail_metrics = ['contrail_atr20_cocip_sum', 'contrail_atr20_accf_sum'] + common_metrics

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


contrail_no_changes= calculate_relative_changes(contrail_no_df, common_metrics)
contrail_yes_changes = calculate_relative_changes(contrail_yes_df, contrail_metrics)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Engine display names
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}


# Function to generate bar chart with error bars

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Engine display names
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2000',
    'GTF': 'GTF',
    'GTF2035': 'GTF2035',
    'GTF2035_wi': 'GTF2035WI'
}

metric_titles = {
        'nox_impact_sum_relative_change': 'NOx',
        'contrail_atr20_cocip_sum_relative_change': 'Contrail',
        'climate_non_co2_relative_change': 'Non-CO₂',
        'climate_total_cons_sum_relative_change': 'Total',
        'climate_total_opti_sum_relative_change': 'Total',
        'co2_impact_cons_sum_relative_change': 'CO₂',
        'co2_impact_opti_sum_relative_change': 'CO₂'
    }

legend_titles = {
        'contrail_atr20_cocip_sum_relative_change': 'Contrail',
        'nox_impact_sum_relative_change': 'NOx',
        'co2_impact_cons_sum_relative_change': 'CO2 Conservative',
        'co2_impact_opti_sum_relative_change': 'CO2 Optimistic',
        'climate_non_co2_relative_change': 'Non-CO2',
        'climate_total_cons_sum_relative_change': 'Total Climate Impact Conservative',
        'climate_total_opti_sum_relative_change': 'Total Climate Impact Optimistic'
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
        "CFM2000", "GTF",
        "GTF2035", "GTF2035\n-20", "GTF2035\n-100",
        "GTF2035WI", "GTF2035WI\n-20", "GTF2035WI\n-100"
    ]

    # Ensure ordering
    grouped = grouped.set_index("engine_display").reindex(x_order).reset_index()

    # Adjust bar width based on the number of metrics
    width = 0.6 if len(metrics) == 1 else 0.15  # Wider bars if only one metric
    x = np.arange(len(x_order))  # X locations for groups

    plt.figure(figsize=(12, 6))



    for i, metric in enumerate(metrics):
        legend_label = legend_titles.get(metric, metric.replace("_relative_change", "").replace("_", " "))

        plt.bar(x + i * (width if len(metrics) > 1 else 0), grouped[f"{metric}_mean"], yerr=grouped[f"{metric}_std"],
                capsize=5,
                alpha=0.7, label=legend_label, width=width)

    # Mapping of metric names to title components


    # Generate title parts, ensuring "Total" and "CO₂" appear only once
    title_parts = []
    seen_total = seen_co2 = False

    for metric in metrics:
        if metric in ['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change']:
            if not seen_total:
                title_parts.append("Total")
                seen_total = True
        elif metric in ['co2_impact_cons_sum_relative_change', 'co2_impact_opti_sum_relative_change']:
            if not seen_co2:
                title_parts.append("CO₂")
                seen_co2 = True
        elif metric in metric_titles:
            title_parts.append(metric_titles[metric])

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



plot_rasd_barplot(contrail_yes_changes, "contrail_yes", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
plot_rasd_barplot(contrail_yes_changes, "contrail_yes", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_yes_changes, "contrail_yes", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])

plot_rasd_barplot(contrail_no_changes, "contrail_no", metrics=['nox_impact_sum_relative_change'])
plot_rasd_barplot(contrail_no_changes, "contrail_no", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_no_changes, "contrail_no", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])



def plot_rad_barplot(df, df_name, metrics=['climate_total_cons_sum_relative_change']):
    """
    Plots a grouped bar chart of relative difference (%) after applying the transformation formula.

    Parameters:
        df (DataFrame): The input dataframe containing RASD values.
        df_name (str): Name of the dataframe (for saving the plot).
        metrics (list): A list of column names representing RASD values to be plotted.
    """

    # Engines to plot, including baseline CFM1990
    engines_to_plot = ['CFM1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi']
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
        "CFM1990", "CFM2000", "GTF",
        "GTF2035", "GTF2035\n20", "GTF2035\n100",
        "GTF2035WI", "GTF2035WI\n20", "GTF2035WI\n100"
    ]

    # Ensure ordering
    grouped = grouped.set_index("engine_display").reindex(x_order).reset_index()

    # Adjust bar width based on the number of metrics
    width = 0.6 if len(metrics) == 1 else 0.15
    x = np.arange(len(x_order))

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        # Use axis_titles mapping for the legend label, default to cleaned-up column name if missing
        legend_label = legend_titles.get(metric, metric.replace("_relative_change", "").replace("_", " "))

        plt.bar(x + i * (width if len(metrics) > 1 else 0), grouped[metric],
                alpha=0.7, label=legend_label, width=width)

    # Generate title parts, ensuring "Total" and "CO₂" appear only once
    title_parts = []
    seen_total = seen_co2 = False

    for metric in metrics:
        if metric in ['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change']:
            if not seen_total:
                title_parts.append("Total")
                seen_total = True
        elif metric in ['co2_impact_cons_sum_relative_change', 'co2_impact_opti_sum_relative_change']:
            if not seen_co2:
                title_parts.append("CO₂")
                seen_co2 = True
        elif metric in metric_titles:
            title_parts.append(metric_titles[metric])

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


plot_rad_barplot(contrail_yes_changes, "contrail_yes", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
plot_rad_barplot(contrail_yes_changes, "contrail_yes", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_yes_changes, "contrail_yes", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])

plot_rad_barplot(contrail_no_changes, "contrail_no", metrics=['nox_impact_sum_relative_change'])
plot_rad_barplot(contrail_no_changes, "contrail_no", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_no_changes, "contrail_no", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])


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
        "CFM1990", "CFM2000", "GTF",
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


metrics_csv_contrail_yes = ['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change',
                'climate_non_co2_relative_change',
               'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change',
                'climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'
               ]

metrics_csv_contrail_no = ['nox_impact_sum_relative_change',
                'climate_non_co2_relative_change',
               'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change',
                'climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'
               ]

export_relative_difference_csv(contrail_yes_changes, "contrail_yes", metrics_csv_contrail_yes)


export_relative_difference_csv(contrail_no_changes, "contrail_no", metrics_csv_contrail_no)


"""DIURNAL"""


"""SEASON"""



plt.show()