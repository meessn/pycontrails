import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

engine_order = [
    "CFM1990", "CFM2000", "GTF",
    "GTF2035", "GTF2035 - 20", "GTF2035 - 100",
    "GTF2035WI", "GTF2035WI - 20", "GTF2035WI - 100"
]

# Format x-axis labels for SAF levels
engine_labels = {
    "CFM1990": "CFM1990",
    "CFM2000": "CFM2000",
    "GTF": "GTF",
    "GTF2035": "GTF2035",
    "GTF2035 - 20": "GTF2035\n-20",
    "GTF2035 - 100": "GTF2035\n-100",
    "GTF2035WI": "GTF2035WI",
    "GTF2035WI - 20": "GTF2035WI\n-20",
    "GTF2035WI - 100": "GTF2035WI\n-100"
}


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
            engine = "CFM2000"
        if 'water_injection' in row and row['water_injection'] > 0:
            engine = engine.replace("_wi", "")
            engine += "WI"
        if 'saf_level' in row and row['saf_level'] != 0:
            engine += f" - {int(row['saf_level'])}"



        return engine

    # Apply function to each row
    df['engine_display'] = df.apply(format_engine_name, axis=1)

    return df

average_1990_day_df = generate_engine_display(average_1990_day_df)
average_1990_night_df = generate_engine_display(average_1990_night_df)
average_1990_day_df = average_1990_day_df.rename(columns={"fuel_kg_sum_change": "Fuel Flow",
                        "nox_sum_change": "NOx",
                        "nvpm_num_sum_change": "nvPM Number"})
average_1990_night_df = average_1990_night_df.rename(columns={"fuel_kg_sum_change": "Fuel Flow",
                        "nox_sum_change": "NOx",
                        "nvpm_num_sum_change": "nvPM Number"})

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

plt.show()