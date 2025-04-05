import pandas as pd
import matplotlib.pyplot as plt
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
contrail_strict = results_df.copy()
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


contrail_strict['contrail_binary'] = contrail_strict['contrail_atr20_cocip_sum'] != 0

# Step 2: Group by flight, sum how many engines had non-zero contrail
contrail_counts = contrail_strict.groupby(['trajectory', 'season', 'diurnal'])['contrail_binary'].sum().reset_index()
contrail_counts.rename(columns={'contrail_binary': 'num_engines_with_contrail'}, inplace=True)

# Step 3: Keep only flights where ALL 9 engines generated contrails
full_contrail_flights = contrail_counts[contrail_counts['num_engines_with_contrail'] == 9]

# Step 4: Merge to get full rows (for all engines) only for those flights
contrail_yes_all_df = results_df.merge(
    full_contrail_flights[['trajectory', 'season', 'diurnal']],
    on=['trajectory', 'season', 'diurnal'],
    how='inner'
)

print(contrail_yes_all_df)


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
contrail_yes_all_changes = calculate_relative_changes(contrail_yes_all_df, contrail_metrics)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Engine display names
engine_display_names = {
    'GTF1990': 'CFM1990',
    'GTF2000': 'CFM2008',
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
    'GTF2000': 'CFM2008',
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
plot_rasd_barplot(contrail_yes_changes, "contrail_yes", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change','climate_non_co2_relative_change'])
plot_rasd_barplot(contrail_yes_changes, "contrail_yes", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_yes_changes, "contrail_yes", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])

plot_rasd_barplot(contrail_no_changes, "contrail_no", metrics=['nox_impact_sum_relative_change'])
plot_rasd_barplot(contrail_no_changes, "contrail_no", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_no_changes, "contrail_no", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])

plot_rasd_barplot(contrail_yes_all_changes, "contrail_yes_all", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
plot_rasd_barplot(contrail_yes_all_changes, "contrail_yes_all", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_yes_all_changes, "contrail_yes_all", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])

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

plot_rad_barplot(
    contrail_yes_changes[
        (contrail_yes_changes['season'] == '2023-05-05') &
        (contrail_yes_changes['diurnal'] == 'daytime')
    ],
    "worst",
    metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change']
)
print('amount of spring daytime', contrail_yes_changes[
        (contrail_yes_changes['season'] == '2023-05-05') &
        (contrail_yes_changes['diurnal'] == 'daytime')])

plot_rad_barplot(
    contrail_yes_changes[
        (contrail_yes_changes['season'] == '2023-05-05') &
        (contrail_yes_changes['diurnal'] == 'daytime')
    ],
    "worst",
    metrics=['climate_non_co2_relative_change' ,'climate_total_cons_sum_relative_change']
)

plot_rad_barplot(contrail_no_changes, "contrail_no", metrics=['nox_impact_sum_relative_change'])
plot_rad_barplot(contrail_no_changes, "contrail_no", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_no_changes, "contrail_no", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])

plot_rad_barplot(contrail_yes_all_changes, "contrail_yes_all", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change'])
plot_rad_barplot(contrail_yes_all_changes, "contrail_yes_all", metrics=['climate_non_co2_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_yes_all_changes, "contrail_yes_all", metrics=['climate_total_cons_sum_relative_change', 'climate_total_opti_sum_relative_change'])

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
#
#
export_relative_difference_csv(contrail_no_changes, "contrail_no", metrics_csv_contrail_no)


"""DIURNAL"""
contrail_yes_day = contrail_yes_changes[contrail_yes_changes['diurnal'] == 'daytime'].copy()
contrail_yes_night = contrail_yes_changes[contrail_yes_changes['diurnal'] == 'nighttime'].copy()

contrail_no_day = contrail_no_changes[contrail_no_changes['diurnal'] == 'daytime'].copy()
contrail_no_night = contrail_no_changes[contrail_no_changes['diurnal'] == 'nighttime'].copy()

contrail_yes_all_day = contrail_yes_all_changes[contrail_yes_all_changes['diurnal'] == 'daytime'].copy()
contrail_yes_all_night = contrail_yes_all_changes[contrail_yes_all_changes['diurnal'] == 'nighttime'].copy()

#rasd
plot_rasd_barplot(contrail_yes_day, "contrail_yes_day", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_yes_night, "contrail_yes_night", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])

plot_rasd_barplot(contrail_yes_all_day, "contrail_yes_all_day", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_yes_all_night, "contrail_yes_all_night", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])


plot_rasd_barplot(contrail_no_day, "contrail_no_day", metrics=['nox_impact_sum_relative_change','co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rasd_barplot(contrail_no_night, "contrail_no_night", metrics=['nox_impact_sum_relative_change','co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
#rad
plot_rad_barplot(contrail_yes_day, "contrail_yes_day", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_yes_night, "contrail_yes_night", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])

plot_rad_barplot(contrail_yes_all_day, "contrail_yes_all_day", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_yes_all_night, "contrail_yes_all_night", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])

plot_rad_barplot(contrail_no_day, "contrail_no_day", metrics=['nox_impact_sum_relative_change','co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_no_night, "contrail_no_night", metrics=['nox_impact_sum_relative_change','co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])

#tables
export_relative_difference_csv(contrail_yes_day, "contrail_yes_day", metrics_csv_contrail_yes)
export_relative_difference_csv(contrail_yes_night, "contrail_yes_night", metrics_csv_contrail_yes)

export_relative_difference_csv(contrail_no_day, "contrail_no_day", metrics_csv_contrail_no)
export_relative_difference_csv(contrail_no_night, "contrail_no_night", metrics_csv_contrail_no)

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

contrail_yes_changes = assign_season_astro_simple(contrail_yes_changes)
contrail_no_changes = assign_season_astro_simple(contrail_no_changes)

contrail_yes_winter = contrail_yes_changes[contrail_yes_changes['season_astro'] == 'winter'].copy()
contrail_yes_spring = contrail_yes_changes[contrail_yes_changes['season_astro'] == 'spring'].copy()
contrail_yes_summer = contrail_yes_changes[contrail_yes_changes['season_astro'] == 'summer'].copy()
contrail_yes_autumn = contrail_yes_changes[contrail_yes_changes['season_astro'] == 'autumn'].copy()

contrail_no_winter = contrail_no_changes[contrail_no_changes['season_astro'] == 'winter'].copy()
contrail_no_spring = contrail_no_changes[contrail_no_changes['season_astro'] == 'spring'].copy()
contrail_no_summer = contrail_no_changes[contrail_no_changes['season_astro'] == 'summer'].copy()
contrail_no_autumn = contrail_no_changes[contrail_no_changes['season_astro'] == 'autumn'].copy()

plot_rad_barplot(contrail_yes_winter, "contrail_yes_winter", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_yes_spring, "contrail_yes_spring", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_yes_summer, "contrail_yes_summer", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_yes_autumn, "contrail_yes_autumn", metrics=['nox_impact_sum_relative_change', 'contrail_atr20_cocip_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])

export_relative_difference_csv(contrail_yes_winter, "contrail_yes_winter", metrics_csv_contrail_yes)
export_relative_difference_csv(contrail_yes_spring, "contrail_yes_spring", metrics_csv_contrail_yes)
export_relative_difference_csv(contrail_yes_summer, "contrail_yes_summer", metrics_csv_contrail_yes)
export_relative_difference_csv(contrail_yes_autumn, "contrail_yes_autumn", metrics_csv_contrail_yes)

plot_rad_barplot(contrail_no_winter, "contrail_no_winter", metrics=['nox_impact_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_no_spring, "contrail_no_spring", metrics=['nox_impact_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_no_summer, "contrail_no_summer", metrics=['nox_impact_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])
plot_rad_barplot(contrail_no_autumn, "contrail_no_autumn", metrics=['nox_impact_sum_relative_change', 'co2_impact_cons_sum_relative_change','co2_impact_opti_sum_relative_change'])

export_relative_difference_csv(contrail_no_winter, "contrail_no_winter", metrics_csv_contrail_no)
export_relative_difference_csv(contrail_no_spring, "contrail_no_spring", metrics_csv_contrail_no)
export_relative_difference_csv(contrail_no_summer, "contrail_no_summer", metrics_csv_contrail_no)
export_relative_difference_csv(contrail_no_autumn, "contrail_no_autumn", metrics_csv_contrail_no)


season_dfs = {
    "contrail_yes_changes": contrail_yes_changes,
    "contrail_no_changes": contrail_no_changes,
    "contrail_yes_day": contrail_yes_day,
    "contrail_yes_night": contrail_yes_night,
    "contrail_no_day": contrail_no_day,
    "contrail_no_night": contrail_no_night,
    "contrail_yes_winter": contrail_yes_winter,
    "contrail_yes_spring": contrail_yes_spring,
    "contrail_yes_summer": contrail_yes_summer,
    "contrail_yes_autumn": contrail_yes_autumn,
    "contrail_no_winter": contrail_no_winter,
    "contrail_no_spring": contrail_no_spring,
    "contrail_no_summer": contrail_no_summer,
    "contrail_no_autumn": contrail_no_autumn
}

# Loop through each DataFrame and print row count
for name, df in season_dfs.items():
    print(f"{name}: {df.shape[0]} rows")





# Define engine order
engine_order = [
    "CFM1990", "CFM2008", "GTF",
    "GTF2035", "GTF2035 - 20", "GTF2035 - 100",
    "GTF2035WI", "GTF2035WI - 20", "GTF2035WI - 100"
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

def plot_day_night_barplot(day_df, night_df, df_name, metric='climate_total_cons_sum_relative_change'):
    """
    Creates a grouped bar plot comparing daytime vs. nighttime climate impact for different engines.

    Parameters:
        day_df (DataFrame): Data for daytime missions.
        night_df (DataFrame): Data for nighttime missions.
        metric (str): The column name representing the metric to be plotted.
    """

    # Filter only engines that are in our predefined order
    day_df = day_df[day_df['engine_display'].isin(engine_order)]
    night_df = night_df[night_df['engine_display'].isin(engine_order)]

    # Merge dataframes based on engine_display
    merged_df = pd.merge(day_df[['engine_display', metric]], night_df[['engine_display', metric]],
                         on='engine_display', suffixes=('_day', '_night'))

    # **Add 100 to metric values**
    for col in [f"{metric}_day", f"{metric}_night"]:
        merged_df[col] = merged_df[col] + 100

    # Sort DataFrame based on predefined engine order
    merged_df = merged_df.set_index("engine_display").reindex(engine_order).reset_index()

    # Bar plot setup
    width = 0.35
    x = np.arange(len(merged_df))

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, merged_df[f"{metric}_day"], width=width, label="Daytime", alpha=0.7)
    plt.bar(x + width/2, merged_df[f"{metric}_night"], width=width, label="Nighttime", alpha=0.7)

    plt.ylabel("Relative Climate Impact (%)")
    plt.title(f"{metric}: Day vs Night")
    plt.xticks(x, [engine_labels[eng] for eng in merged_df['engine_display']], rotation=0, ha="center")  # No rotation
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save figure
    filename = f"results_report/barplot/day_night_barplot_{metric}_{df_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")


def plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, df_name, metric='climate_total_cons_sum_relative_change'):
    """
    Creates a grouped bar plot comparing seasonal climate impact for different engines.

    Parameters:
        winter_df (DataFrame): Data for winter missions.
        spring_df (DataFrame): Data for spring missions.
        summer_df (DataFrame): Data for summer missions.
        autumn_df (DataFrame): Data for autumn missions.
        metric (str): The column name representing the metric to be plotted.
    """

    # Filter only engines that are in our predefined order
    winter_df = winter_df[winter_df['engine_display'].isin(engine_order)]
    spring_df = spring_df[spring_df['engine_display'].isin(engine_order)]
    summer_df = summer_df[summer_df['engine_display'].isin(engine_order)]
    autumn_df = autumn_df[autumn_df['engine_display'].isin(engine_order)]

    # Merge seasonal data based on engine_display
    merged_df = pd.merge(winter_df[['engine_display', metric]], spring_df[['engine_display', metric]],
                         on='engine_display', suffixes=('_winter', '_spring'))
    merged_df = pd.merge(merged_df, summer_df[['engine_display', metric]], on='engine_display')
    merged_df = pd.merge(merged_df, autumn_df[['engine_display', metric]], on='engine_display',
                         suffixes=('_summer', '_autumn'))

    # **Add 100 to metric values**
    for col in [f"{metric}_winter", f"{metric}_spring", f"{metric}_summer", f"{metric}_autumn"]:
        merged_df[col] = merged_df[col] + 100

    # Sort DataFrame based on predefined engine order
    merged_df = merged_df.set_index("engine_display").reindex(engine_order).reset_index()

    # Bar plot setup
    width = 0.2
    x = np.arange(len(merged_df))

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, merged_df[f"{metric}_winter"], width=width, label="Winter", alpha=0.7)
    plt.bar(x - 0.5 * width, merged_df[f"{metric}_spring"], width=width, label="Spring", alpha=0.7)
    plt.bar(x + 0.5 * width, merged_df[f"{metric}_summer"], width=width, label="Summer", alpha=0.7)
    plt.bar(x + 1.5 * width, merged_df[f"{metric}_autumn"], width=width, label="Autumn", alpha=0.7)

    plt.ylabel("Relative Climate Impact (%)")
    plt.title(f"{metric} Climate Impact Seasonal Effect")
    plt.xticks(x, [engine_labels[eng] for eng in merged_df['engine_display']], rotation=0, ha="center")  # No rotation
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save figure
    filename = f"results_report/barplot/seasonal_barplot_{metric}_{df_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")



# Load CSV files
day_df = pd.read_csv("results_report/climate/contrail_yes_day_rad_vs_gtf1990.csv")
night_df = pd.read_csv("results_report/climate/contrail_yes_night_rad_vs_gtf1990.csv")

# Call day/night barplot function
plot_day_night_barplot(day_df, night_df, 'contrails_yes', metric="Contrail")
plot_day_night_barplot(day_df, night_df, 'contrails_yes', metric="NOx")
plot_day_night_barplot(day_df, night_df, 'contrails_yes', metric="Non-CO2")
plot_day_night_barplot(day_df, night_df, 'contrails_yes', metric="Total Climate Impact Conservative")
plot_day_night_barplot(day_df, night_df, 'contrails_yes', metric="Total Climate Impact Optimistic")

# Load seasonal CSV files
winter_df = pd.read_csv("results_report/climate/contrail_yes_winter_rad_vs_gtf1990.csv")
autumn_df = pd.read_csv("results_report/climate/contrail_yes_autumn_rad_vs_gtf1990.csv")
spring_df = pd.read_csv("results_report/climate/contrail_yes_spring_rad_vs_gtf1990.csv")
summer_df = pd.read_csv("results_report/climate/contrail_yes_summer_rad_vs_gtf1990.csv")

# Call seasonal barplot function
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'contrails_yes',metric="Total Climate Impact Conservative")
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'contrails_yes',metric="Total Climate Impact Optimistic")
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df,'contrails_yes', metric="NOx")
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df,'contrails_yes', metric="Non-CO2")
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'contrails_yes',metric="Contrail")

# Load CSV files
day_df = pd.read_csv("results_report/climate/contrail_no_day_rad_vs_gtf1990.csv")
night_df = pd.read_csv("results_report/climate/contrail_no_night_rad_vs_gtf1990.csv")

# Call day/night barplot function
# plot_day_night_barplot(day_df, night_df, 'contrails_no', metric="Contrail")
plot_day_night_barplot(day_df, night_df, 'contrails_no', metric="NOx")
plot_day_night_barplot(day_df, night_df, 'contrails_no', metric="Non-CO2")
plot_day_night_barplot(day_df, night_df, 'contrails_no', metric="Total Climate Impact Conservative")

# Load seasonal CSV files
winter_df = pd.read_csv("results_report/climate/contrail_no_winter_rad_vs_gtf1990.csv")
autumn_df = pd.read_csv("results_report/climate/contrail_no_autumn_rad_vs_gtf1990.csv")
spring_df = pd.read_csv("results_report/climate/contrail_no_spring_rad_vs_gtf1990.csv")
summer_df = pd.read_csv("results_report/climate/contrail_no_summer_rad_vs_gtf1990.csv")

# Call seasonal barplot function
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'contrails_no',metric="Total Climate Impact Conservative")
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df,'contrails_no', metric="NOx")
# plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'contrails_no',metric="Contrail")

plt.show()