import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the emissions results CSV
results_df = pd.read_csv('results_main_simulations.csv')

# ---- COMMON CONFIG ---- #
metrics_to_compare = [
    'fuel_kg_sum', 'ei_co2_conservative_sum', 'ei_co2_optimistic_sum',
    'ei_nox_sum',   'ei_nvpm_num_sum', 'co2_conservative_sum', 'co2_optimistic_sum', 'nox_sum', 'nvpm_num_sum', 'nox_impact_sum'
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

def compute_day_night_ratios(results_df, metric):
    """
    Compute the average day/total and night/total ratios per engine config.
    """
    # Group identifiers
    group_cols = ['engine', 'saf_level', 'water_injection', 'trajectory', 'season']

    # Split day and night
    day_df = results_df[results_df['diurnal'] == 'daytime']
    night_df = results_df[results_df['diurnal'] == 'nighttime']

    # Merge on trajectory + season
    merged = pd.merge(
        day_df[group_cols + [metric]],
        night_df[group_cols + [metric]],
        on=group_cols,
        suffixes=('_day', '_night')
    )

    # Compute fractions
    merged['total'] = merged[f'{metric}_day'] + merged[f'{metric}_night']
    merged['day_frac'] = merged[f'{metric}_day'] / merged['total']
    merged['night_frac'] = merged[f'{metric}_night'] / merged['total']

    # Average across all trajectories/seasons
    ratio_df = merged.groupby(['engine', 'saf_level', 'water_injection'])[['day_frac', 'night_frac']].mean().reset_index()
    return ratio_df

def prepare_stacked_bar_data(day_df, night_df, results_df, metric, metric_display_name):
    """
    Prepares data for the stacked bar showing total emissions split into day/night.
    """
    # Compute the average day/night ratio for each engine
    ratio_df = compute_day_night_ratios(results_df, metric)

    # Merge with day/night bar values
    merged = pd.merge(
        day_df[['engine', 'saf_level', 'water_injection', metric_display_name]],
        night_df[['engine', 'saf_level', 'water_injection', metric_display_name]],
        on=['engine', 'saf_level', 'water_injection'],
        suffixes=('_day', '_night')
    )

    # Add +100 to convert from relative to actual bar heights
    merged['day_val'] = merged[f'{metric_display_name}_day'] + 100
    merged['night_val'] = merged[f'{metric_display_name}_night'] + 100

    # Compute total (for stacked bar height)
    merged['total_val'] = (merged['day_val'] + merged['night_val']) / 2

    # Merge with day/night fractions
    merged = pd.merge(merged, ratio_df, on=['engine', 'saf_level', 'water_injection'])

    # Compute stacked parts
    merged['stacked_day'] = merged['total_val'] * merged['day_frac']
    merged['stacked_night'] = merged['total_val'] * merged['night_frac']

    # Format engine names for plotting
    merged = generate_engine_display(merged)
    return merged

def generate_engine_display(df):
    """
    Generates the engine_display column based on existing engine names and SAF levels.

    Parameters:
        df (DataFrame): The input DataFrame containing engine configurations.

    Returns:
        DataFrame: Updated DataFrame with an 'engine_display' column.
    """

    def format_engine_name(row):
        """
        Formats engine name by including SAF and water injection levels.
        """
        engine = row['engine']  # Assuming engine names are in a column called 'engine'

        # Add SAF level if present
        # Add water injection if present
        if engine == "GTF1990":
            engine = "CFM1990"
        if engine == "GTF2000":
            engine = "CFM2008"
        if 'water_injection' in row and row['water_injection'] > 0:
            engine = engine.replace("_wi", "")
            engine += "WI"
        if 'saf_level' in row and row['saf_level'] != 0:
            engine += f" - {int(row['saf_level'])}"



        return engine

    # Apply function to each row
    df['engine_display'] = df.apply(format_engine_name, axis=1)

    return df


def plot_stacked_day_night_bars(stacked_df, day_df, night_df, metric_display_name, df_name):
    """
    Plots 3 bars per engine:
    - Left: Stacked bar (Total emissions = Day + Night), with diagonal hatching
    - Middle: Day only (solid)
    - Right: Night only (solid)
    """
    # Order bars by engine
    stacked_df = stacked_df[stacked_df['engine_display'].isin(engine_order)].set_index('engine_display').reindex(engine_order).reset_index()
    day_df = day_df.set_index('engine_display').reindex(engine_order).reset_index()
    night_df = night_df.set_index('engine_display').reindex(engine_order).reset_index()

    width = 0.25
    x = np.arange(len(stacked_df))

    # Colors
    day_color = "#1f77b4"   # blue
    night_color = "#ff7f0e" # orange

    plt.figure(figsize=(14, 6))

    # ---- LEFT BAR: TOTAL (stacked with hatching) ----
    bar_day_total = plt.bar(
        x - width, stacked_df['stacked_day'], width=width,
        label="Day Contribution To Total", color=day_color, hatch='\\',
        edgecolor="white", alpha=0.7
    )
    bar_night_total = plt.bar(
        x - width, stacked_df['stacked_night'], bottom=stacked_df['stacked_day'], width=width,
        label="Night Contribution To Total", color=night_color, hatch='\\',
        edgecolor="white", alpha=0.7
    )

    # ---- MIDDLE BAR: DAY ONLY (solid) ----
    bar_day = plt.bar(
        x, day_df[metric_display_name] + 100, width=width,
        label="Day", color=day_color, alpha=0.7
    )

    # ---- RIGHT BAR: NIGHT ONLY (solid) ----
    bar_night = plt.bar(
        x + width, night_df[metric_display_name] + 100, width=width,
        label="Night", color=night_color, alpha=0.7
    )

    # Axis labels and ticks
    plt.ylabel("Relative Emissions (%)")
    plt.title(f"{metric_display_name}: Total vs Day/Night Breakdown")
    plt.xticks(x, [engine_labels[eng] for eng in stacked_df['engine_display']], rotation=0)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Custom Legend
    handles = [bar_day_total[0], bar_night_total[0], bar_day[0], bar_night[0]]
    labels = ["Day Total", "Night Total", "Day", "Night"]
    plt.legend(handles, labels)

    # Save to file
    filename = f"results_report/emissions/striped_day_night_barplot_{metric_display_name}_{df_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved striped stacked bar plot: {filename}")


average_1990_day_df = generate_engine_display(average_1990_day_df)
average_1990_night_df = generate_engine_display(average_1990_night_df)
average_1990_day_df = average_1990_day_df.rename(columns={
    "fuel_kg_sum_change": "Fuel Flow",
    "nox_sum_change": "NOx",
    "nvpm_num_sum_change": "nvPM Number",
    "nox_impact_sum_change": "NOx Climate Impact"
})
average_1990_night_df = average_1990_night_df.rename(columns={
    "fuel_kg_sum_change": "Fuel Flow",
    "nox_sum_change": "NOx",
    "nvpm_num_sum_change": "nvPM Number",
    "nox_impact_sum_change": "NOx Climate Impact"
})

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

    plt.ylabel("Relative Emissions (%)")
    plt.title(f"{metric}: Day vs Night")
    plt.xticks(x, [engine_labels[eng] for eng in merged_df['engine_display']], rotation=0, ha="center")  # No rotation
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save figure
    filename = f"results_report/emissions/day_night_barplot_{metric}_{df_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")


plot_day_night_barplot(average_1990_day_df, average_1990_night_df, 'df_emissions_1990', metric="NOx")
plot_day_night_barplot(average_1990_day_df, average_1990_night_df, 'df_emissions_1990', metric="nvPM Number")
plot_day_night_barplot(average_1990_day_df, average_1990_night_df, 'df_emissions_1990', metric="Fuel Flow")


stacked_nox_df = prepare_stacked_bar_data(
    average_1990_day_df, average_1990_night_df, results_df,
    metric='nox_sum', metric_display_name='NOx'
)
plot_stacked_day_night_bars(stacked_nox_df, average_1990_day_df, average_1990_night_df, 'NOx', '1990_emissions')

# For nvPM
stacked_nvpm_df = prepare_stacked_bar_data(
    average_1990_day_df, average_1990_night_df, results_df,
    metric='nvpm_num_sum', metric_display_name='nvPM Number'
)
plot_stacked_day_night_bars(stacked_nvpm_df, average_1990_day_df, average_1990_night_df, 'nvPM Number', '1990_emissions')

# For Fuel
stacked_fuel_df = prepare_stacked_bar_data(
    average_1990_day_df, average_1990_night_df, results_df,
    metric='fuel_kg_sum', metric_display_name='Fuel Flow'
)
plot_stacked_day_night_bars(stacked_fuel_df, average_1990_day_df, average_1990_night_df, 'Fuel Flow', '1990_emissions')

# For NOx Climate Impact
stacked_nox_clim_df = prepare_stacked_bar_data(
    average_1990_day_df, average_1990_night_df, results_df,
    metric='nox_impact_sum', metric_display_name='NOx Climate Impact'
)
plot_stacked_day_night_bars(
    stacked_nox_clim_df, average_1990_day_df, average_1990_night_df,
    'NOx Climate Impact', '1990_emissions'
)

plt.show()