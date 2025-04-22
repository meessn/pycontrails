import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

day_df = pd.read_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_day.csv")
night_df = pd.read_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_night.csv")

winter_df = pd.read_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_winter.csv")
spring_df = pd.read_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_spring.csv")
summer_df = pd.read_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_summer.csv")
autumn_df = pd.read_csv("results_report/climate/diurnal_seasonal_entire_csv/contrail_yes_cocip_autumn.csv")

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

# def plot_day_night_barplot(day_df, night_df, df_name, metric='climate_total_cons_sum_relative_change'):
#     """
#     Creates a grouped bar plot comparing daytime vs. nighttime climate impact for different engines.
#
#     Parameters:
#         day_df (DataFrame): Data for daytime missions.
#         night_df (DataFrame): Data for nighttime missions.
#         metric (str): The column name representing the metric to be plotted.
#     """
#
#     # Filter only engines that are in our predefined order
#     day_df = day_df[day_df['engine_display'].isin(engine_order)]
#     night_df = night_df[night_df['engine_display'].isin(engine_order)]
#
#     # Take mean RASD values per engine first
#     day_df = day_df.groupby("engine_display")[metric].mean().reset_index()
#     night_df = night_df.groupby("engine_display")[metric].mean().reset_index()
#
#     # Merge dataframes based on engine_display
#     merged_df = pd.merge(day_df[['engine_display', metric]], night_df[['engine_display', metric]],
#                          on='engine_display', suffixes=('_day', '_night'))
#
#     # Apply RASD to Relative Climate Impact % conversion
#     for col in [f"{metric}_day", f"{metric}_night"]:
#         merged_df[col] = (2 * merged_df[col]) / (1 - merged_df[col]) * 100
#
#     # Now add 100 to all values
#     for col in [f"{metric}_day", f"{metric}_night"]:
#         merged_df[col] = merged_df[col] + 100
#
#     # Sort DataFrame based on predefined engine order
#     merged_df = merged_df.set_index("engine_display").reindex(engine_order).reset_index()
#
#     # Bar plot setup
#     width = 0.35
#     x = np.arange(len(merged_df))
#
#     plt.figure(figsize=(12, 6))
#     plt.bar(x - width/2, merged_df[f"{metric}_day"], width=width, label="Daytime", alpha=0.7)
#     plt.bar(x + width/2, merged_df[f"{metric}_night"], width=width, label="Nighttime", alpha=0.7)
#
#     plt.ylabel("Relative Climate Impact (%)")
#     plt.title(f"Contrail Climate Impact: Diurnal Effect Relative to CFM1990")
#     plt.xticks(x, [engine_labels[eng] for eng in merged_df['engine_display']], rotation=0, ha="center")  # No rotation
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.5)
#
#     # Save figure
#     filename = f"results_report/barplot/diurnal/day_night_barplot_{metric}_{df_name}.png".replace(" ", "_")
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     print(f"Saved plot as: {filename}")


# def plot_day_night_barplot_stacked(day_df, night_df, df_name, metric='climate_total_cons_sum_relative_change'):
#     """
#     Plots a stacked bar chart for day and night contributions of climate impact,
#     using relative changes applied to the baseline (CFM1990) split.
#
#     Parameters:
#         day_df (DataFrame): Data for daytime missions.
#         night_df (DataFrame): Data for nighttime missions.
#         df_name (str): Label for saving plots or debugging.
#         metric (str): The column name representing the metric to be analyzed.
#     """
#
#     # ------------------- Step 1: Compute CFM1990 day/night split ------------------- #
#     cfm_df = pd.concat([day_df, night_df])
#     cfm_df = cfm_df[cfm_df['engine'] == 'GTF1990']
#
#     pivoted = cfm_df.pivot_table(
#         index=['trajectory', 'season'],
#         columns='diurnal',
#         values='contrail_atr20_cocip_sum'
#     )
#
#     total_pairs = len(pivoted)
#     pivoted = pivoted.dropna(subset=['daytime', 'nighttime'])
#     valid_pairs = len(pivoted)
#
#     print(f"[{df_name}] Total possible trajectory-season pairs: {total_pairs}")
#     print(f"[{df_name}] Valid day-night pairs (used): {valid_pairs}")
#     print(f"[{df_name}] Discarded due to missing day or night: {total_pairs - valid_pairs}")
#
#     pivoted['daytime_abs'] = np.abs(pivoted['daytime'])
#     pivoted['nighttime_abs'] = np.abs(pivoted['nighttime'])
#
#     pivoted['day_ratio'] = pivoted['daytime_abs'] / (pivoted['daytime_abs'] + pivoted['nighttime_abs'])
#     pivoted['night_ratio'] = pivoted['nighttime_abs'] / (pivoted['daytime_abs'] + pivoted['nighttime_abs'])
#
#     cfm_day_ratio = pivoted['day_ratio'].mean()
#     cfm_night_ratio = pivoted['night_ratio'].mean()
#
#     print(f"[{df_name}] Average CFM1990 day ratio: {cfm_day_ratio:.3f}")
#     print(f"[{df_name}] Average CFM1990 night ratio: {cfm_night_ratio:.3f}")
#
#     # gtf_df = pd.concat([day_df, night_df])
#     # gtf_df = gtf_df[gtf_df['engine_display'] == 'GTF2035WI - 100']  # Filter for GTF engine
#     #
#     # # Pivot table to get daytime and nighttime values side-by-side for each trajectory-season
#     # pivoted_gtf = gtf_df.pivot_table(
#     #     index=['trajectory', 'season'],
#     #     columns='diurnal',
#     #     values='contrail_atr20_cocip_sum'
#     # )
#     #
#     # # Count all pairs
#     # total_pairs_gtf = len(pivoted_gtf)
#     #
#     # # Drop rows where either day or night is missing
#     # pivoted_gtf = pivoted_gtf.dropna(subset=['daytime', 'nighttime'])
#     # valid_pairs_gtf = len(pivoted_gtf)
#     #
#     # print(f"[GTF] Total possible trajectory-season pairs: {total_pairs_gtf}")
#     # print(f"[GTF] Valid day-night pairs (used): {valid_pairs_gtf}")
#     # print(f"[GTF] Discarded due to missing day or night: {total_pairs_gtf - valid_pairs_gtf}")
#     #
#     # # Calculate absolute values and ratios
#     # pivoted_gtf['daytime_abs'] = np.abs(pivoted_gtf['daytime'])
#     # pivoted_gtf['nighttime_abs'] = np.abs(pivoted_gtf['nighttime'])
#     #
#     # pivoted_gtf['day_ratio'] = pivoted_gtf['daytime_abs'] / (pivoted_gtf['daytime_abs'] + pivoted_gtf['nighttime_abs'])
#     # pivoted_gtf['night_ratio'] = pivoted_gtf['nighttime_abs'] / (
#     #             pivoted_gtf['daytime_abs'] + pivoted_gtf['nighttime_abs'])
#     #
#     # # Average across all trajectory-season pairs
#     # gtf_day_ratio = pivoted_gtf['day_ratio'].mean()
#     # gtf_night_ratio = pivoted_gtf['night_ratio'].mean()
#     #
#     # print(f"[GTF] Average GTF day ratio: {gtf_day_ratio:.3f}")
#     # print(f"[GTF] Average GTF night ratio: {gtf_night_ratio:.3f}")
#
#     # ------------------- Step 2: Compute mean RASD for each engine ------------------- #
#     day_df = day_df[day_df['engine_display'].isin(engine_order)]
#     night_df = night_df[night_df['engine_display'].isin(engine_order)]
#
#     day_grouped = day_df.groupby("engine_display")[metric].mean().reset_index()
#     night_grouped = night_df.groupby("engine_display")[metric].mean().reset_index()
#
#     merged_df = pd.merge(day_grouped, night_grouped, on='engine_display', suffixes=('_day', '_night'))
#
#     # ------------------- Step 3: Apply RASD to % Conversion ------------------- #
#     for col in [f"{metric}_day", f"{metric}_night"]:
#         merged_df[col] = (2 * merged_df[col]) / (1 - merged_df[col]) * 100
#         merged_df[col] = merged_df[col] + 100
#
#     # ------------------- Step 4: Compute stacked bars ------------------- #
#     merged_df['stacked_day'] = merged_df[f"{metric}_day"] * cfm_day_ratio
#     merged_df['stacked_night'] = merged_df[f"{metric}_night"] * cfm_night_ratio
#
#     # ------------------- Step 5: Sort for consistent plotting ------------------- #
#     merged_df = merged_df.set_index("engine_display").reindex(engine_order).reset_index()
#     x = np.arange(len(merged_df))
#     width = 0.6
#
#     # ------------------- Step 6: Plotting ------------------- #
#     plt.figure(figsize=(12, 6))
#     plt.bar(x, merged_df['stacked_day'], width=width, label="Daytime", alpha=0.8)
#     plt.bar(x, merged_df['stacked_night'], bottom=merged_df['stacked_day'], width=width, label="Nighttime", alpha=0.8)
#
#     plt.ylabel("Stacked Climate Impact (%)")
#     plt.title(f"{metric}: Day vs Night (Stacked, Based on CFM1990 Split)")
#     plt.xticks(x, [engine_labels[eng] for eng in merged_df['engine_display']], rotation=0, ha="center")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.5)
#
#     filename = f"results_report/barplot/day_night_barplot_stacked_{metric}_{df_name}.png".replace(" ", "_")
#     # plt.savefig(filename, dpi=300, bbox_inches="tight")
#     print(f"Saved plot as: {filename}")



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

    # Take the mean RASD per engine first
    winter_df = winter_df.groupby("engine_display")[metric].mean().reset_index()
    spring_df = spring_df.groupby("engine_display")[metric].mean().reset_index()
    summer_df = summer_df.groupby("engine_display")[metric].mean().reset_index()
    autumn_df = autumn_df.groupby("engine_display")[metric].mean().reset_index()

    # Merge seasonal data based on engine_display
    merged_df = pd.merge(winter_df, spring_df, on='engine_display', suffixes=('_winter', '_spring'))
    merged_df = pd.merge(merged_df, summer_df, on='engine_display')
    merged_df = pd.merge(merged_df, autumn_df, on='engine_display', suffixes=('_summer', '_autumn'))

    # Rename columns to be clear
    merged_df = merged_df.rename(columns={
        metric: f"{metric}_summer",
        f"{metric}_autumn": f"{metric}_autumn"
    })

    # Apply RASD to RAD conversion and shift to relative scale
    for col in [f"{metric}_winter", f"{metric}_spring", f"{metric}_summer", f"{metric}_autumn"]:
        merged_df[col] = (2 * merged_df[col]) / (1 - merged_df[col]) * 100 + 100

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
    plt.title(f"Contrail Climate Impact: Seasonal Effect Relative to CFM1990")
    plt.xticks(x, [engine_labels[eng] for eng in merged_df['engine_display']], rotation=0, ha="center")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save figure
    filename = f"results_report/barplot/seasonal/seasonal_barplot_{metric}_{df_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot as: {filename}")

def compute_seasonal_stack_ratios(df, engine_name, df_label):
    """
    Compute average seasonal ratios (winter, spring, summer, autumn) for a given engine.
    This works like the diurnal stacking logic, but across seasons, and handles day/night separately.

    Parameters:
        df (DataFrame): Combined seasonal DataFrame (e.g., concat of winter/spring/summer/autumn)
        engine_name (str): Engine type to filter on (e.g., 'GTF1990')
        df_label (str): Label for printing debug info

    Returns:
        Dict with average ratios per season: {'winter': ..., 'spring': ..., 'summer': ..., 'autumn': ...}
    """

    # Filter for the specified engine
    df = df[df['engine_display'] == engine_name].copy()

    # Count basic stats
    unique_pairs = df[['trajectory', 'diurnal']].drop_duplicates()
    # Seasonal presence per (trajectory, diurnal) combination
    season_presence_table = df[['trajectory', 'diurnal', 'season_astro']].drop_duplicates()

    season_presence_table = season_presence_table.pivot_table(
        index=['trajectory', 'diurnal'],
        columns='season_astro',
        aggfunc='size',
        fill_value=0
    )

    # Optional: enforce season column order
    season_presence_table = season_presence_table.reindex(columns=['winter', 'spring', 'summer', 'autumn'],
                                                          fill_value=0)

    season_presence_table = season_presence_table.sort_values(by=['trajectory', 'diurnal']).reset_index()

    print(f"[{df_label}] All unique (trajectory, diurnal) combinations with seasonal data presence:")
    print(season_presence_table)

    day_count = (df['diurnal'] == 'daytime').sum()
    night_count = (df['diurnal'] == 'nighttime').sum()

    print(f"[{df_label}] --- INPUT DATA STATS FOR {engine_name} ---")
    print(f"[{df_label}] Unique (trajectory, diurnal) combinations: {len(unique_pairs)}")
    print(f"[{df_label}] # of daytime entries: {day_count}")
    print(f"[{df_label}] # of nighttime entries: {night_count}")
    print(f"[{df_label}] Total entries: {len(df)}")

    # Take absolute contrail sum
    df['abs_contrail'] = df['contrail_atr20_cocip_sum'].abs()

    # Pivot so each row is a trajectory + diurnal combo, with columns = seasons
    pivoted = df.pivot_table(
        index=['trajectory', 'diurnal'],
        columns='season_astro',
        values='abs_contrail',
        aggfunc='first'  # safe since we assume one row per (trajectory, season, diurnal)
    ).dropna(axis=0)  # Keep only rows with all 4 seasons

    print(f"[{df_label}] Valid rows with all 4 seasons: {len(pivoted)}")
    print(pivoted)
    # Normalize each row to get seasonal ratios per trajectory+diurnal
    row_sums = pivoted.sum(axis=1)
    seasonal_ratios = pivoted.div(row_sums, axis=0)

    # Average across all trajectory-diurnal pairs
    average_ratios = seasonal_ratios.mean().to_dict()

    # Print results
    print(f"[{df_label}] --- AVERAGE SEASONAL RATIOS FOR {engine_name} ---")
    for season in ['winter', 'spring', 'summer', 'autumn']:
        print(f"[{df_label}] {season.capitalize()} ratio: {average_ratios.get(season, 0):.3f}")

    return average_ratios

#
#
#
# Load CSV files


# Call day/night barplot function
# plot_day_night_barplot(day_df, night_df, 'contrails_yes_cocip', metric="contrail_atr20_cocip_sum_relative_change")
# plot_day_night_barplot_stacked(day_df, night_df, 'contrails_yes_cocip', metric="contrail_atr20_cocip_sum_relative_change")
# plot_day_night_barplot(day_df, night_df, 'contrails_yes', metric="NOx")
# plot_day_night_barplot(day_df, night_df, 'contrails_yes', metric="Non-CO2")
# plot_day_night_barplot(day_df, night_df, 'contrails_yes', metric="Total Climate Impact Conservative")
# plot_day_night_barplot(day_df, night_df, 'contrails_yes', metric="Total Climate Impact Optimistic")



# Call seasonal barplot function
plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'contrails_yes_cocip',metric="contrail_atr20_cocip_sum_relative_change")

seasonal_df = pd.concat([
    winter_df.assign(season_astro='winter'),
    spring_df.assign(season_astro='spring'),
    summer_df.assign(season_astro='summer'),
    autumn_df.assign(season_astro='autumn'),
])
ratios_gtf1990 = compute_seasonal_stack_ratios(seasonal_df, engine_name='GTF2035', df_label='contrails_yes_cocip')

# def plot_seasonal_barplot_stacked(winter_df, spring_df, summer_df, autumn_df, df_name, metric='climate_total_cons_sum_relative_change'):
#     """
#     Creates a stacked bar plot showing seasonal breakdown of climate impact per engine,
#     normalized using base engine's seasonal contrail distribution.
#
#     Parameters:
#         winter_df, spring_df, summer_df, autumn_df: DataFrames for each season.
#         df_name (str): Name for the dataset context.
#         metric (str): Column used to compute relative change.
#     """
#
#     # Combine full DataFrame for stacking base calculation
#     full_df = pd.concat([winter_df, spring_df, summer_df, autumn_df], ignore_index=True)
#
#     # Compute seasonal distribution from base engine (e.g., GTF1990)
#     base_engine = 'CFM1990'
#     seasonal_ratios = compute_seasonal_stack_ratios(full_df, base_engine, df_label=df_name)
#
#     # Step 1: Filter for engines of interest
#     def process_season(df, season):
#         df = df[df['engine_display'].isin(engine_order)].copy()
#         df = df.groupby("engine_display")[metric].mean().reset_index()
#         df.rename(columns={metric: f"{metric}_{season}"}, inplace=True)
#         return df
#
#     winter = process_season(winter_df, "winter")
#     spring = process_season(spring_df, "spring")
#     summer = process_season(summer_df, "summer")
#     autumn = process_season(autumn_df, "autumn")
#
#     # Step 2: Merge all seasonal means into one DataFrame
#     merged = winter.merge(spring, on="engine_display")
#     merged = merged.merge(summer, on="engine_display")
#     merged = merged.merge(autumn, on="engine_display")
#
#     # Step 3: Convert RASD to Relative Climate Impact (%)
#     for season in ['winter', 'spring', 'summer', 'autumn']:
#         col = f"{metric}_{season}"
#         merged[col] = (2 * merged[col]) / (1 - merged[col]) * 100
#         merged[col] = merged[col] + 100  # Add 100 baseline
#
#     # Step 4: Apply seasonal ratios to build stacked values
#     for season in ['winter', 'spring', 'summer', 'autumn']:
#         merged[f"{season}_stack"] = seasonal_ratios[season] * merged[f"{metric}_{season}"]
#
#     # Sort by engine display order
#     merged = merged.set_index("engine_display").reindex(engine_order).reset_index()
#
#     # Step 5: Plot
#     x = np.arange(len(merged))
#     width = 0.6
#
#     plt.figure(figsize=(12, 6))
#     bottom = np.zeros(len(merged))
#     for season in ['winter', 'spring', 'summer', 'autumn']:
#         bar_values = merged[f"{season}_stack"]
#         plt.bar(x, bar_values, bottom=bottom, width=width, label=season.capitalize(), alpha=0.8)
#         bottom += bar_values
#
#     plt.ylabel("Relative Climate Impact (%)")
#     plt.title(f"{metric}: Seasonal Stacked Climate Impact")
#     plt.xticks(x, [engine_labels[eng] for eng in merged['engine_display']], rotation=0, ha="center")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.5)
#
#     filename = f"results_report/barplot/seasonal/seasonal_stacked_barplot_{metric}_{df_name}.png".replace(" ", "_")
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     print(f"Saved plot as: {filename}")

# plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'contrails_yes',metric="Total Climate Impact Conservative")
# plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'contrails_yes',metric="Total Climate Impact Optimistic")
# plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df,'contrails_yes', metric="NOx")
# plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df,'contrails_yes', metric="Non-CO2")
# plot_seasonal_barplot(winter_df, spring_df, summer_df, autumn_df, 'contrails_yes',metric="Contrail")

# plot_seasonal_barplot_stacked(
#     winter_df, spring_df, summer_df, autumn_df,
#     df_name='contrails_yes_cocip',
#     metric="contrail_atr20_cocip_sum_relative_change"
# )

def investigate_full_coverage(df, df_label):
    """
    Investigates and prints (trajectory, diurnal) pairs that have entries across all seasons
    for all engine types, and further filters those that contain only non-zero contrail values.
    Returns two DataFrames: all valid full-coverage, and those with strictly non-zero contributions.
    """

    expected_engines = set(df['engine_display'].unique())
    expected_seasons = {'winter', 'spring', 'summer', 'autumn'}

    # Count entries per trajectory + diurnal + engine + season
    grouped = df.groupby(['trajectory', 'diurnal', 'engine_display', 'season_astro']).size().reset_index(name='count')

    # Pivot for season completeness check
    pivoted = grouped.pivot_table(
        index=['trajectory', 'diurnal', 'engine_display'],
        columns='season_astro',
        values='count',
        fill_value=0
    )

    def has_all_seasons(row):
        return all(row.get(season, 0) > 0 for season in expected_seasons)

    pivoted['has_all_seasons'] = pivoted.apply(has_all_seasons, axis=1)

    # Step 1: Find full season coverage for all engines
    complete_engines = pivoted[pivoted['has_all_seasons']].reset_index()
    full_coverage_counts = complete_engines.groupby(['trajectory', 'diurnal'])['engine_display'].nunique().reset_index(name='engine_count')
    fully_covered = full_coverage_counts[full_coverage_counts['engine_count'] == len(expected_engines)]

    print(f"\n[{df_label}] (trajectory, diurnal) combinations with ALL engines having ALL 4 seasons:")
    if fully_covered.empty:
        print(f"[{df_label}] None found.")
        return pd.DataFrame(), pd.DataFrame()

    print(fully_covered.sort_values(by=['trajectory', 'diurnal']).reset_index(drop=True))

    # Step 2: Filter those where all values are also non-zero
    valid_rows = []
    for _, row in fully_covered.iterrows():
        traj = row['trajectory']
        diurnal = row['diurnal']
        subset = df[(df['trajectory'] == traj) & (df['diurnal'] == diurnal)]

        pivot_check = subset.pivot_table(
            index='engine_display',
            columns='season_astro',
            values='contrail_atr20_cocip_sum',
            aggfunc='sum'  # assume unique rows per engine+season already
        )

        # Take abs and check if all values exist and are > 0
        pivot_check = pivot_check.abs()
        if (pivot_check > 0).all().all():
            valid_rows.append((traj, diurnal))

    # Report valid ones
    print(f"\n[{df_label}] STRICTLY NON-ZERO combinations (for plotting):")
    strictly_valid = pd.DataFrame(valid_rows, columns=['trajectory', 'diurnal'])
    print(strictly_valid if not strictly_valid.empty else f"[{df_label}] None with non-zero values only.")

    # Step 3: Optional print for review
    print(f"\n[{df_label}] Full rows for strictly valid combinations:\n")
    for traj, diurnal in valid_rows:
        subset = df[(df['trajectory'] == traj) & (df['diurnal'] == diurnal)]
        display_cols = ['trajectory', 'diurnal', 'engine_display', 'season_astro', 'contrail_atr20_cocip_sum']
        print(f"\n--- {traj} | {diurnal} ---")
        print(subset[display_cols].sort_values(by=['engine_display', 'season_astro']).reset_index(drop=True))

    return fully_covered, strictly_valid


combined_seasonal_df = pd.concat([winter_df, spring_df, summer_df, autumn_df])
all_covered, valid_nonzero = investigate_full_coverage(combined_seasonal_df, "contrails_yes_cocip")


# Convert strictly valid result to tuple list
target_trajectories = list(valid_nonzero.itertuples(index=False, name=None))


def plot_stacked_contrail_bars_relative(df, target_trajectories, df_label):
    """
    Plot stacked bar charts normalized to CFM1990 = 100%, preserving seasonal distribution.
    Each bar represents an engine, scaled relative to CFM1990's total contrail for that mission.
    """

    for traj, diurnal in target_trajectories:
        subset = df[(df['trajectory'] == traj) & (df['diurnal'] == diurnal)].copy()
        subset['abs_contrail'] = subset['contrail_atr20_cocip_sum'].abs()

        # Pivot: index = engine, columns = season
        pivoted = subset.pivot_table(
            index='engine_display',
            columns='season_astro',
            values='abs_contrail',
            aggfunc='sum'
        ).reindex(engine_order).dropna()

        # Ensure all engines have all 4 seasons and non-zero values
        pivoted = pivoted[(pivoted > 0).all(axis=1)]
        if pivoted.empty or "CFM1990" not in pivoted.index:
            print(f"[{df_label}] Skipping {traj} | {diurnal}: No valid CFM1990 or incomplete data")
            continue

        # Normalize total CFM1990 to 100
        cfm1990_total = pivoted.loc["CFM1990"].sum()
        rel_scaled = pivoted.div(cfm1990_total).mul(100)

        # Plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(rel_scaled))
        bottom = np.zeros(len(rel_scaled))

        for season in ['winter', 'spring', 'summer', 'autumn']:
            values = rel_scaled[season].values
            plt.bar(x, values, bottom=bottom, label=season.capitalize(), alpha=0.8)
            bottom += values

        # X-axis setup
        xtick_labels = [engine_labels.get(eng, eng) for eng in rel_scaled.index]
        plt.xticks(x, xtick_labels, rotation=0, ha="center")

        # Title formatting
        formatted_traj = traj.replace("_", "-").upper()
        formatted_diurnal = diurnal.capitalize()

        plt.ylabel("Relative Climate Impact (%)")
        plt.title(f"Contrail Climate Impact: Seasonal Contributions for {formatted_traj} | {formatted_diurnal} Mission")
        plt.legend(title="Season")
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)

        filename = f"results_report/barplot/seasonal/stacked_contrail_relative_{traj}_{diurnal}.png".replace(" ", "_")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved plot as: {filename}")
        # plt.show()


# Plot stacked bar charts for those combinations
plot_stacked_contrail_bars_relative(combined_seasonal_df, target_trajectories, "contrails_yes_cocip")


# def plot_day_night_barplot_stacked_weighted(day_df, night_df, df_name, metric='contrail_atr20_cocip_sum'):
#     """
#     Plots a stacked bar chart for day and night climate impact contributions,
#     weighted by actual contrail generation proportions and using RASD-derived relative impact.
#
#     Parameters:
#         day_df (DataFrame): Data for daytime missions.
#         night_df (DataFrame): Data for nighttime missions.
#         df_name (str): Label for saving plots or debugging.
#         metric (str): Column with RASD (relative avoided signal difference) per mission.
#     """
#
#     # ------------------- Step 1: Filter to valid day-night pairs for each engine ------------------- #
#     combined = pd.concat([day_df.assign(diurnal='daytime'), night_df.assign(diurnal='nighttime')])
#     all_engines = combined['engine_display'].unique()
#
#     all_results = []
#
#     for engine in engine_order:
#         df_engine = combined[combined['engine_display'] == engine].copy()
#
#         # Pivot to get day/night side-by-side per (trajectory, season)
#         pivot = df_engine.pivot_table(
#             index=['trajectory', 'season'],
#             columns='diurnal',
#             values=['contrail_atr20_cocip_sum', metric],
#             aggfunc='first'  # assume one row per mission
#         )
#
#         # Drop incomplete pairs
#         pivot = pivot.dropna(subset=[('contrail_atr20_cocip_sum', 'daytime'), ('contrail_atr20_cocip_sum', 'nighttime')])
#
#         if pivot.empty:
#             print(f"[{df_name}] Skipping {engine}: no valid day-night pairs")
#             continue
#
#         # Get day/night contribution ratios (absolute contrail sums)
#         pivot['day_abs'] = np.abs(pivot[('contrail_atr20_cocip_sum', 'daytime')])
#         pivot['night_abs'] = np.abs(pivot[('contrail_atr20_cocip_sum', 'nighttime')])
#         pivot['day_ratio'] = pivot['day_abs'] / (pivot['day_abs'] + pivot['night_abs'])
#         pivot['night_ratio'] = pivot['night_abs'] / (pivot['day_abs'] + pivot['night_abs'])
#
#         day_ratio = pivot['day_ratio'].mean()
#         night_ratio = pivot['night_ratio'].mean()
#
#         # Compute mean RASD for day and night
#         mean_rasd_day = pivot[(metric, 'daytime')].mean()
#         mean_rasd_night = pivot[(metric, 'nighttime')].mean()
#
#         # # Convert RASD to relative impact
#         # rad_day = (2 * mean_rasd_day) / (1 - mean_rasd_day) * 100 + 100
#         # rad_night = (2 * mean_rasd_night) / (1 - mean_rasd_night) * 100 + 100
#         #
#         # # Combine with day/night ratios for final stacked height
#         # total_day = rad_day * day_ratio
#         # total_night = rad_night * night_ratio
#
#         # Compute overall RASD mean (combined day + night)
#         all_rasd_values = pivot[(metric, 'daytime')].dropna().tolist() + pivot[(metric, 'nighttime')].dropna().tolist()
#         mean_rasd = np.mean(all_rasd_values)
#
#         # Convert to RAD (%)
#         rad_total = (2 * mean_rasd) / (1 - mean_rasd) * 100 + 100
#
#         # Split into day/night based on actual contrail proportions
#         total_day = rad_total * day_ratio
#         total_night = rad_total * night_ratio
#
#         all_results.append({
#             'engine_display': engine,
#             'stacked_day': total_day,
#             'stacked_night': total_night
#         })
#
#         print(f"[{df_name}] --- {engine} ---")
#         print(f"  Avg RASD: Day={mean_rasd_day:.3f}, Night={mean_rasd_night:.3f}")
#         # print(f"  RAD%: Day={rad_day:.1f}%, Night={rad_night:.1f}%")
#         print(rad_total)
#         print(f"  Ratios: Day={day_ratio:.3f}, Night={night_ratio:.3f}")
#         print(f"  Stacked total: {total_day + total_night:.1f}%\n")
#
#     # ------------------- Step 2: Plot ------------------- #
#     df_plot = pd.DataFrame(all_results)
#     df_plot = df_plot.set_index('engine_display').reindex(engine_order).reset_index()
#
#     x = np.arange(len(df_plot))
#     width = 0.6
#
#     plt.figure(figsize=(12, 6))
#     plt.bar(x, df_plot['stacked_day'], width=width, label="Daytime", alpha=0.8)
#     plt.bar(x, df_plot['stacked_night'], width=width, bottom=df_plot['stacked_day'], label="Nighttime", alpha=0.8)
#
#     plt.xticks(x, [engine_labels[eng] for eng in df_plot['engine_display']], rotation=0, ha='center')
#     plt.ylabel("Relative Climate Impact (%)")
#     plt.title(f"Contrail Climate Impact: Diurnal Contributions")
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#
#     filename = f"results_report/barplot/diurnal/day_night_barplot_stacked_weighted_{df_name}.png".replace(" ", "_")
#     plt.savefig(filename, dpi=300)
#     print(f"Saved plot as: {filename}")
#     # plt.show()
#
# plot_day_night_barplot_stacked_weighted(
#     day_df,
#     night_df,
#     df_name="contrails_yes_cocip",
#     metric="contrail_atr20_cocip_sum_relative_change"
# )

def plot_day_night_barplot_stacked_weighted_also_zero(day_df, night_df, df_name, metric='contrail_atr20_cocip_sum'):
    """
    Stacked bar chart for day/night climate impact, allowing missing day or night entries (set to zero).
    Uses actual contrail proportions and RASD-derived relative impact.
    """

    combined = pd.concat([day_df.assign(diurnal='daytime'), night_df.assign(diurnal='nighttime')])
    all_results = []

    for engine in engine_order:
        df_engine = combined[combined['engine_display'] == engine].copy()

        pivot = df_engine.pivot_table(
            index=['trajectory', 'season'],
            columns='diurnal',
            values=['contrail_atr20_cocip_sum', metric],
            aggfunc='first'
        )

        # Fill missing day/night with 0 instead of dropping
        for col in [('contrail_atr20_cocip_sum', 'daytime'), ('contrail_atr20_cocip_sum', 'nighttime'),
                    (metric, 'daytime'), (metric, 'nighttime')]:
            if col not in pivot.columns:
                pivot[col] = 0.0
        pivot = pivot.fillna(0.0)

        # Compute contrail ratios
        pivot['day_abs'] = np.abs(pivot[('contrail_atr20_cocip_sum', 'daytime')])
        pivot['night_abs'] = np.abs(pivot[('contrail_atr20_cocip_sum', 'nighttime')])
        total_contrail = pivot['day_abs'] + pivot['night_abs']

        # Avoid division by zero (if both day and night are zero, skip)
        pivot = pivot[total_contrail > 0]
        if pivot.empty:
            print(f"[{df_name}] Skipping {engine}: no contrail data at all")
            continue

        pivot['day_ratio'] = pivot['day_abs'] / total_contrail
        pivot['night_ratio'] = pivot['night_abs'] / total_contrail

        # Get average ratios
        day_ratio = pivot['day_ratio'].mean()
        night_ratio = pivot['night_ratio'].mean()

        df_engine = combined[combined['engine_display'] == engine]
        mean_rasd = df_engine[metric].mean()

        # Convert to relative climate impact (RAD %)
        rad_total = (2 * mean_rasd) / (1 - mean_rasd) * 100 + 100
        total_day = rad_total * day_ratio
        total_night = rad_total * night_ratio

        all_results.append({
            'engine_display': engine,
            'stacked_day': total_day,
            'stacked_night': total_night
        })

        print(f"[{df_name}] --- {engine} ---")
        # print(f"  Mean RASD: Day={mean_rasd_day:.3f}, Night={mean_rasd_night:.3f}")
        print(f"  Contrail Ratio: Day={day_ratio:.3f}, Night={night_ratio:.3f}")
        print(f"  Final Impact: Day={total_day:.1f}%, Night={total_night:.1f}%, Total={total_day + total_night:.1f}%\n")
        print(rad_total)
    # Plot
    df_plot = pd.DataFrame(all_results)
    df_plot = df_plot.set_index('engine_display').reindex(engine_order).reset_index()

    x = np.arange(len(df_plot))
    width = 0.6

    plt.figure(figsize=(12, 6))
    plt.bar(x, df_plot['stacked_day'], width=width, label="Daytime", alpha=1.0)
    plt.bar(x, df_plot['stacked_night'], bottom=df_plot['stacked_day'], width=width, label="Nighttime", alpha=1.0)
    # Annotate bars with value and ratio: e.g., "23.3% / 0.43"
    # Get baseline values for CFM1990
    cfm_row = df_plot[df_plot['engine_display'] == 'CFM1990']
    cfm_day = cfm_row['stacked_day'].values[0]
    cfm_night = cfm_row['stacked_night'].values[0]

    # Annotate bars
    for i, row in df_plot.iterrows():
        x_pos = i
        y_day = row['stacked_day']
        y_night = row['stacked_night']
        y_total = y_day + y_night
        is_baseline = row['engine_display'] == "CFM1990"

        # --- Day Annotation ---
        if y_day > 4:
            if is_baseline:
                # Single white label + smaller baseline
                plt.text(
                    x_pos,
                    y_day * 0.5 + 0.6,
                    f"{y_day:.1f}%",
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=9
                )
                plt.text(
                    x_pos,
                    y_day * 0.5 - 2.1,
                    "(baseline)",
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=7
                )
            else:
                reduction_day = 100 * (1 - y_day / cfm_day)
                plt.text(
                    x_pos,
                    y_day * 0.5 + 0.6,
                    f"{y_day:.1f}%",
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=9
                )
                plt.text(
                    x_pos,
                    y_day * 0.5 - 2.1,
                    f"(-{reduction_day:.1f}%)",
                    ha='center',
                    va='center',
                    color='lime',
                    fontsize=7
                )

        # --- Night Annotation ---
        if y_night > 4:
            if is_baseline:
                plt.text(
                    x_pos,
                    y_day + y_night * 0.5 + 0.6,
                    f"{y_night:.1f}%",
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=9
                )
                plt.text(
                    x_pos,
                    y_day + y_night * 0.5 - 2.1,
                    "(baseline)",
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=7
                )
            else:
                reduction_night = 100 * (1 - y_night / cfm_night)
                plt.text(
                    x_pos,
                    y_day + y_night * 0.5 + 0.6,
                    f"{y_night:.1f}%",
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=9
                )
                plt.text(
                    x_pos,
                    y_day + y_night * 0.5 - 2.1,
                    f"(-{reduction_night:.1f}%)",
                    ha='center',
                    va='center',
                    color='lime',
                    fontsize=7
                )

        # # --- Total above bar ---
        # if y_total > 2:
        #     plt.text(
        #         x_pos,
        #         y_total + 0.5,
        #         f"{y_total:.1f}%",
        #         ha='center',
        #         va='bottom',
        #         color='black',
        #         fontsize=9
        #     )
    plt.xticks(x, [engine_labels[eng] for eng in df_plot['engine_display']], rotation=0, ha='center')
    plt.ylabel("Relative Climate Impact (%)")
    plt.title(f"Contrail Climate Impact: Diurnal Effect")
    plt.legend()
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"results_report/barplot/diurnal/day_night_barplot_stacked_final_{df_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as: {filename}")

plot_day_night_barplot_stacked_weighted_also_zero(
    day_df,
    night_df,
    df_name="contrails_yes_cocip",
    metric="contrail_atr20_cocip_sum_relative_change"
)

def plot_seasonal_barplot_stacked_weighted_also_zero(winter_df, spring_df, summer_df, autumn_df, df_name, metric='contrail_atr20_cocip_sum_relative_change'):
    """
    Stacked seasonal bar chart: each bar = total mean RAD per engine,
    stack segments weighted by CFM1990's seasonal contrail distribution.
    """

    # Combine all seasons
    full_df = pd.concat([
        winter_df.assign(season_astro='winter'),
        spring_df.assign(season_astro='spring'),
        summer_df.assign(season_astro='summer'),
        autumn_df.assign(season_astro='autumn')
    ])

    # Get seasonal stack ratios from CFM1990
    base_engine = 'CFM1990'
    seasonal_ratios = compute_seasonal_stack_ratios(full_df, base_engine, df_label=df_name)

    # Compute mean RAD per engine across all data
    combined = full_df.copy()
    combined = combined[combined['engine_display'].isin(engine_order)]

    mean_rads = combined.groupby('engine_display')[metric].mean().reset_index()
    mean_rads['rad_total'] = (2 * mean_rads[metric]) / (1 - mean_rads[metric]) * 100 + 100

    # Apply stacking weights
    for season in ['winter', 'spring', 'summer', 'autumn']:
        mean_rads[f"{season}_stack"] = mean_rads['rad_total'] * seasonal_ratios.get(season, 0)

    # Sort for plotting
    mean_rads = mean_rads.set_index("engine_display").reindex(engine_order).reset_index()

    # Plotting
    x = np.arange(len(mean_rads))
    width = 0.6
    plt.figure(figsize=(12, 6))

    bottom = np.zeros(len(mean_rads))
    for season in ['winter', 'spring', 'summer', 'autumn']:
        heights = mean_rads[f"{season}_stack"].values
        plt.bar(x, heights, bottom=bottom, width=width, label=season.capitalize(), alpha=0.9)
        bottom += heights

    # Axis & labels
    plt.xticks(x, [engine_labels[eng] for eng in mean_rads['engine_display']], rotation=0, ha='center')
    plt.ylabel("Relative Climate Impact (%)")
    plt.title(f"Contrail Climate Impact: Seasonal Stacked Contributions")
    plt.legend()
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"results_report/barplot/seasonal/seasonal_barplot_stacked_weighted_{df_name}.png".replace(" ", "_")
    # plt.savefig(filename, dpi=300)
    print(f"Saved plot as: {filename}")



plot_seasonal_barplot_stacked_weighted_also_zero(
    winter_df, spring_df, summer_df, autumn_df,
    df_name="contrails_yes_cocip",
    metric="contrail_atr20_cocip_sum_relative_change"
)





def print_contrail_counts_per_engine_per_season(df, contrail_column='contrail_atr20_cocip_sum'):
    """
    Prints how many times each engine formed a contrail per season (non-zero entries).

    Parameters:
        df (DataFrame): Data containing 'engine_display', 'season_astro', and the contrail column.
        contrail_column (str): The column representing contrail magnitude.
    """
    print("\nContrail Counts Per Engine Per Season (non-zero entries):\n")

    # Filter for non-zero contrail values
    df_nonzero = df[df[contrail_column].abs() > 0]

    # Group and count
    counts = df_nonzero.groupby(['engine_display', 'season_astro']).size().unstack(fill_value=0)

    # Optional: Sort by engine order if needed
    if 'engine_order' in globals():
        counts = counts.reindex(engine_order)

    print(counts)

# Usage example:
print_contrail_counts_per_engine_per_season(combined_seasonal_df)

def compute_engine_seasonal_ratios(df, engine_name, contrail_col='contrail_atr20_cocip_sum'):
    """
    Compute average seasonal ratios (winter, spring, summer, autumn) for a given engine,
    based on trajectory + diurnal level seasonal contributions.

    Parameters:
        df (DataFrame): Input seasonal dataframe with columns:
                        ['engine_display', 'trajectory', 'diurnal', 'season_astro', contrail_col]
        engine_name (str): The engine to filter for.
        contrail_col (str): Name of the contrail column (default = 'contrail_atr20_cocip_sum').

    Returns:
        Dict with average seasonal ratios for that engine.
    """

    season_order = ['winter', 'spring', 'summer', 'autumn']

    # Filter only this engine
    df_engine = df[df['engine_display'] == engine_name].copy()

    # Absolute contrail values
    df_engine['abs_contrail'] = df_engine[contrail_col].abs()

    # Pivot to get seasons side-by-side for each trajectory + diurnal
    pivot = df_engine.pivot_table(
        index=['trajectory', 'diurnal'],
        columns='season_astro',
        values='abs_contrail',
        aggfunc='first'
    )

    # Fill missing seasons with 0
    pivot = pivot.reindex(columns=season_order, fill_value=0)

    # Filter out rows where total = 0 (no contrail at all for that pair)
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot[pivot['total'] > 0]

    # Normalize row-wise to get seasonal ratios
    seasonal_ratios = pivot[season_order].div(pivot['total'], axis=0)

    # Average across all rows
    average_ratios = seasonal_ratios.fillna(0).mean().to_dict()

    return average_ratios

engine_seasonal_ratios = {}

for engine in engine_order:
    ratios = compute_engine_seasonal_ratios(combined_seasonal_df, engine)
    engine_seasonal_ratios[engine] = ratios
    print(f"{engine}: {ratios}")

def plot_seasonal_barplot_stacked_weighted_engine_specific(winter_df, spring_df, summer_df, autumn_df, df_name,
                                                           metric='contrail_atr20_cocip_sum_relative_change'):
    """
    Stacked seasonal climate impact bars per engine, with annotations for winter and autumn.
    Each engine's seasonal ratios are computed from its own contrail distribution.
    Missing seasons are treated as zero.
    """

    import numpy as np

    # Combine all data with season labels
    full_df = pd.concat([
        winter_df.assign(season_astro='winter'),
        spring_df.assign(season_astro='spring'),
        summer_df.assign(season_astro='summer'),
        autumn_df.assign(season_astro='autumn')
    ])

    full_df = full_df[full_df['engine_display'].isin(engine_order)]
    all_results = []

    season_list = ['winter', 'spring', 'summer', 'autumn']

    for engine in engine_order:
        df_engine = full_df[full_df['engine_display'] == engine].copy()
        mean_rasd = df_engine[metric].mean()
        rad_total = (2 * mean_rasd) / (1 - mean_rasd) * 100 + 100

        ratios = engine_seasonal_ratios.get(engine, {s: 0 for s in season_list})

        row = {'engine_display': engine, 'rad_total': rad_total}
        for season in season_list:
            row[f'{season}_stack'] = rad_total * ratios[season]

        all_results.append(row)

    # Plot
    df_plot = pd.DataFrame(all_results)
    df_plot = df_plot.set_index("engine_display").reindex(engine_order).reset_index()

    x = np.arange(len(df_plot))
    width = 0.6
    plt.figure(figsize=(12, 6))

    bottom = np.zeros(len(df_plot))

    # Get baseline values for reductions
    cfm_row = df_plot[df_plot['engine_display'] == 'CFM1990'].iloc[0]

    for season in season_list:
        bar_values = df_plot[f"{season}_stack"].values
        bars = plt.bar(x, bar_values, bottom=bottom, width=width, label=season.capitalize(), alpha=0.9)

        for i, (bar_val, btm) in enumerate(zip(bar_values, bottom)):
            if bar_val > 2:
                y_mid = btm + bar_val * 0.5
                engine = df_plot.loc[i, 'engine_display']
                is_baseline = engine == "CFM1990"

                # Base font sizes
                if season in ['winter', 'autumn']:
                    fs_main = 9
                    fs_sub = 7
                else:
                    fs_main = 8
                    fs_sub = 6

                if engine == "CFM2008":
                    fs_main -= 1
                    fs_sub -= 1

                # Dynamic spacing
                offset_up = fs_main * 0.07
                offset_down = fs_sub * 0.3

                if is_baseline:
                    plt.text(
                        x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                        ha='center', va='center', color='white', fontsize=fs_main
                    )
                    plt.text(
                        x[i], y_mid - offset_down, "(baseline)",
                        ha='center', va='center', color='white', fontsize=fs_sub
                    )

                else:
                    cfm_val = cfm_row[f"{season}_stack"]
                    if cfm_val > 0:
                        diff = 100 * (1 - bar_val / cfm_val)
                        if diff >= 0:
                            label_text = f"(-{diff:.1f}%)"
                            color = 'lime'
                        else:
                            label_text = f"(+{abs(diff):.1f})%"
                            color = 'red'
                    else:
                        label_text = ""
                        color = 'black'

                    plt.text(
                        x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                        ha='center', va='center', color='white', fontsize=fs_main
                    )
                    if label_text:
                        plt.text(
                            x[i], y_mid - offset_down, label_text,
                            ha='center', va='center', color=color, fontsize=fs_sub
                        )

        bottom += bar_values

    # Total height label (optional, uncomment if desired)
    # for i, total in enumerate(bottom):
    #     if total > 2:
    #         plt.text(x[i], total + 0.5, f"{total:.1f}%", ha='center', va='bottom', color='black', fontsize=9)

    plt.xticks(x, [engine_labels.get(eng, eng) for eng in df_plot['engine_display']], rotation=0, ha="center")
    plt.ylabel("Relative Climate Impact (%)")
    plt.title(f"Contrail Climate Impact: Seasonal Effect")
    plt.legend(title="Season")
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"results_report/barplot/seasonal/seasonal_barplot_stacked_final_{df_name}.png".replace(
        " ", "_")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as: {filename}")


plot_seasonal_barplot_stacked_weighted_engine_specific(
    winter_df, spring_df, summer_df, autumn_df,
    df_name="contrails_yes_cocip",
    metric="contrail_atr20_cocip_sum_relative_change"
)



plt.show()