import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the emissions results CSV
results_df = pd.read_csv('results_main_simulations.csv')

# Map seasons for GRU -> LIM and SIN -> MAA
season_mapping = {
    'gru_lim': {'2023-02-06': 'summer', '2023-05-05': 'autumn', '2023-08-06': 'winter', '2023-11-06': 'spring'}
    # 'sin_maa': {'2023-02-06': 'winter', '2023-05-05': 'summer', '2023-08-06': 'monsoon', '2023-11-06': 'postmonsoon'}
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
    general_seasons_df = merged_df
    for season in ['winter', 'spring', 'summer', 'autumn']:
        season_df = general_seasons_df[general_seasons_df['season_adjusted'] == season]
        average_df = season_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
        average_df = average_df.round(1)
        baseline_label = baseline_engine.lower()
        average_df.to_csv(f'results_report/emissions/seasonal/season_{season}_vs_{baseline_label}.csv', index=False)

    # # SIN -> MAA specific seasons
    # sinmaa_df = merged_df[merged_df['trajectory'] == 'sin_maa']
    # for season in ['winter', 'summer', 'monsoon', 'postmonsoon']:
    #     season_df = sinmaa_df[sinmaa_df['season_adjusted'] == season]
    #     average_df = season_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
    #     average_df = average_df.round(1)
    #     baseline_label = baseline_engine.lower()
    #     average_df.to_csv(f'results_report/emissions/seasonal/season_sinmaa_{season}_vs_{baseline_label}.csv', index=False)

# Calculate for both GTF1990 and GTF baselines
calculate_seasonal_changes('GTF1990')
calculate_seasonal_changes('GTF')
print(results_df[['trajectory', 'season', 'season_adjusted']].drop_duplicates())

print("Saved all seasonal CSVs successfully!")

### open csv's

winter_df = pd.read_csv("results_report/emissions/seasonal/season_winter_vs_gtf1990.csv")
autumn_df = pd.read_csv("results_report/emissions/seasonal/season_autumn_vs_gtf1990.csv")
spring_df = pd.read_csv("results_report/emissions/seasonal/season_spring_vs_gtf1990.csv")
summer_df = pd.read_csv("results_report/emissions/seasonal/season_summer_vs_gtf1990.csv")

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

winter_df = generate_engine_display(winter_df)
autumn_df = generate_engine_display(autumn_df)
spring_df = generate_engine_display(spring_df)
summer_df = generate_engine_display(summer_df)

winter_df = winter_df.rename(columns={"fuel_kg_sum_change": "Fuel Flow",
                        "nox_sum_change": "NOx",
                        "nvpm_num_sum_change": "nvPM Number"})
autumn_df = autumn_df.rename(columns={"fuel_kg_sum_change": "Fuel Flow",
                        "nox_sum_change": "NOx",
                        "nvpm_num_sum_change": "nvPM Number"})
spring_df = spring_df.rename(columns={"fuel_kg_sum_change": "Fuel Flow",
                        "nox_sum_change": "NOx",
                        "nvpm_num_sum_change": "nvPM Number"})
summer_df = summer_df.rename(columns={"fuel_kg_sum_change": "Fuel Flow",
                        "nox_sum_change": "NOx",
                        "nvpm_num_sum_change": "nvPM Number"})


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

    plt.ylabel("Relative Emissions (%)")
    plt.title(f"{metric} Emissions: Seasonal Effect")
    plt.xticks(x, [engine_labels[eng] for eng in merged_df['engine_display']], rotation=0, ha="center")  # No rotation
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save figure
    filename = f"results_report/emissions/seasonal/seasonal_barplot_{metric}_{df_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")

plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'df_emissions_1990',metric="Fuel Flow")
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df,'df_emissions_1990', metric="NOx")
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'df_emissions_1990',metric="nvPM Number")

plt.show()