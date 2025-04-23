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
    'ei_nox_sum',   'ei_nvpm_num_sum', 'co2_conservative_sum', 'co2_optimistic_sum', 'nox_sum', 'nvpm_num_sum','nox_impact_sum', 'h2o_impact_sum'
]

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

results_df = generate_engine_display(results_df)

# Helper function to calculate percentage changes and average by season
def calculate_seasonal_changes_and_return(baseline_engine, baseline_saf=0):
    baseline_df = results_df[(results_df['engine'] == baseline_engine) & (results_df['saf_level'] == baseline_saf)]
    merged_df = results_df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

    for metric in metrics_to_compare:
        merged_df[f'{metric}_change'] = 100 * (merged_df[metric] - merged_df[f'{metric}_baseline']) / merged_df[f'{metric}_baseline']

    columns_to_drop = [col for col in merged_df.columns if '_baseline' in col]
    merged_df = merged_df.drop(columns=columns_to_drop)

    # General seasons
    general_seasons_df = merged_df.copy()
    seasonal_dfs = {}
    for season in ['winter', 'spring', 'summer', 'autumn']:
        season_df = general_seasons_df[general_seasons_df['season_adjusted'] == season]
        avg_df = season_df.groupby(['engine', 'saf_level', 'water_injection'])[
            [f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index().round(1)

        avg_df = generate_engine_display(avg_df)
        avg_df = avg_df.rename(columns={
            "fuel_kg_sum_change": "Fuel Flow",
            "nox_sum_change": "NOx",
            "nvpm_num_sum_change": "nvPM Number",
            "nox_impact_sum_change": "NOx Climate Impact"
        })

        seasonal_dfs[season] = avg_df

    return seasonal_dfs

    # # SIN -> MAA specific seasons
    # sinmaa_df = merged_df[merged_df['trajectory'] == 'sin_maa']
    # for season in ['winter', 'summer', 'monsoon', 'postmonsoon']:
    #     season_df = sinmaa_df[sinmaa_df['season_adjusted'] == season]
    #     average_df = season_df.groupby(['engine', 'saf_level', 'water_injection'])[[f'{metric}_change' for metric in metrics_to_compare]].mean().reset_index()
    #     average_df = average_df.round(1)
    #     baseline_label = baseline_engine.lower()
    #     average_df.to_csv(f'results_report/emissions/seasonal/season_sinmaa_{season}_vs_{baseline_label}.csv', index=False)

# # Calculate for both GTF1990 and GTF baselines
# calculate_seasonal_changes('GTF1990')
# calculate_seasonal_changes('GTF')
# print(results_df[['trajectory', 'season', 'season_adjusted']].drop_duplicates())
#
# print("Saved all seasonal CSVs successfully!")
#
# ### open csv's
#
# winter_df = pd.read_csv("results_report/emissions/seasonal/season_winter_vs_gtf1990.csv")
# autumn_df = pd.read_csv("results_report/emissions/seasonal/season_autumn_vs_gtf1990.csv")
# spring_df = pd.read_csv("results_report/emissions/seasonal/season_spring_vs_gtf1990.csv")
# summer_df = pd.read_csv("results_report/emissions/seasonal/season_summer_vs_gtf1990.csv")
seasonal_dfs = calculate_seasonal_changes_and_return('GTF1990')
winter_df = seasonal_dfs['winter']
spring_df = seasonal_dfs['spring']
summer_df = seasonal_dfs['summer']
autumn_df = seasonal_dfs['autumn']


# winter_df = generate_engine_display(winter_df)
# autumn_df = generate_engine_display(autumn_df)
# spring_df = generate_engine_display(spring_df)
# summer_df = generate_engine_display(summer_df)

winter_df = winter_df.rename(columns={
    "fuel_kg_sum_change": "Fuel Flow",
    "nox_sum_change": "NOx",
    "nox_impact_sum_change": "NOx Climate Impact",
    "nvpm_num_sum_change": "nvPM Number"
})
spring_df = spring_df.rename(columns={
    "fuel_kg_sum_change": "Fuel Flow",
    "nox_sum_change": "NOx",
    "nox_impact_sum_change": "NOx Climate Impact",
    "nvpm_num_sum_change": "nvPM Number"
})
summer_df = summer_df.rename(columns={
    "fuel_kg_sum_change": "Fuel Flow",
    "nox_sum_change": "NOx",
    "nox_impact_sum_change": "NOx Climate Impact",
    "nvpm_num_sum_change": "nvPM Number"
})
autumn_df = autumn_df.rename(columns={
    "fuel_kg_sum_change": "Fuel Flow",
    "nox_sum_change": "NOx",
    "nox_impact_sum_change": "NOx Climate Impact",
    "nvpm_num_sum_change": "nvPM Number"
})


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

def compute_seasonal_ratios(results_df, metric):
    """
    Compute average seasonal ratios (season / total) for each engine config.
    """
    group_cols = ['engine', 'saf_level', 'water_injection', 'trajectory', 'diurnal']

    # Group and pivot using the adjusted season
    seasonal_totals = results_df.groupby(group_cols + ['season_adjusted'])[metric].sum().unstack('season_adjusted')

    # Ensure all seasons exist as columns
    for season in ['winter', 'spring', 'summer', 'autumn']:
        if season not in seasonal_totals.columns:
            seasonal_totals[season] = 0

    # Compute total and fractions
    seasonal_totals['total'] = seasonal_totals[['winter', 'spring', 'summer', 'autumn']].sum(axis=1)
    for season in ['winter', 'spring', 'summer', 'autumn']:
        seasonal_totals[f'{season}_frac'] = seasonal_totals[season] / seasonal_totals['total']

    # Reset index to access grouping columns
    seasonal_ratios = seasonal_totals.reset_index()

    # Only keep relevant numeric columns for averaging
    keep_cols = ['engine', 'saf_level', 'water_injection'] + [f'{s}_frac' for s in ['winter', 'spring', 'summer', 'autumn']]
    seasonal_ratios = seasonal_ratios[keep_cols]

    # Group by engine config and average only numeric columns
    ratio_df = seasonal_ratios.groupby(['engine', 'saf_level', 'water_injection'])[[
        f'{s}_frac' for s in ['winter', 'spring', 'summer', 'autumn']
    ]].mean().reset_index()

    return ratio_df


def prepare_stacked_seasonal_bar_data(winter_df, spring_df, summer_df, autumn_df, results_df, metric, metric_display_name):
    """
    Prepares stacked bar data for seasonal emissions split into 4 seasons.
    """
    # Compute seasonal fractions
    ratio_df = compute_seasonal_ratios(results_df, metric)

    # Merge seasonal emission values
    merged = winter_df[['engine', 'saf_level', 'water_injection', metric_display_name]].rename(columns={metric_display_name: 'winter_val'})
    merged = pd.merge(merged, spring_df[['engine', 'saf_level', 'water_injection', metric_display_name]].rename(columns={metric_display_name: 'spring_val'}), on=['engine', 'saf_level', 'water_injection'])
    merged = pd.merge(merged, summer_df[['engine', 'saf_level', 'water_injection', metric_display_name]].rename(columns={metric_display_name: 'summer_val'}), on=['engine', 'saf_level', 'water_injection'])
    merged = pd.merge(merged, autumn_df[['engine', 'saf_level', 'water_injection', metric_display_name]].rename(columns={metric_display_name: 'autumn_val'}), on=['engine', 'saf_level', 'water_injection'])

    # Convert relative to actual bar heights
    for season in ['winter_val', 'spring_val', 'summer_val', 'autumn_val']:
        merged[season] += 100

    # Compute average total
    merged['total_val'] = merged[[f'{s}_val' for s in ['winter', 'spring', 'summer', 'autumn']]].mean(axis=1)

    # Merge with seasonal fractions
    merged = pd.merge(merged, ratio_df, on=['engine', 'saf_level', 'water_injection'])

    # Compute stacked bar components
    for season in ['winter', 'spring', 'summer', 'autumn']:
        merged[f'stacked_{season}'] = merged['total_val'] * merged[f'{season}_frac']

    merged = generate_engine_display(merged)
    return merged

def plot_stacked_seasonal_bars(stacked_df, winter_df, spring_df, summer_df, autumn_df, metric_display_name, df_name):
    """
    Plots:
    - Left: Stacked seasonal total bar (striped by season)
    - Right 4 bars: Individual seasons (solid)
    """
    # Align engine order
    stacked_df = stacked_df[stacked_df['engine_display'].isin(engine_order)].set_index('engine_display').reindex(engine_order).reset_index()
    winter_df = winter_df.set_index('engine_display').reindex(engine_order).reset_index()
    spring_df = spring_df.set_index('engine_display').reindex(engine_order).reset_index()
    summer_df = summer_df.set_index('engine_display').reindex(engine_order).reset_index()
    autumn_df = autumn_df.set_index('engine_display').reindex(engine_order).reset_index()

    width = 0.15
    x = np.arange(len(stacked_df))

    # Colors
    colors = {
        "winter": "#1f77b4",   # blue
        "spring": "#2ca02c",   # green
        "summer": "#d62728",   # red
        "autumn": "#ff7f0e"    # orange
    }
    hatches = {
        "winter": '\\',
        "spring": '\\',
        "summer": '\\',
        "autumn": '\\'
    }

    plt.figure(figsize=(16, 6))

    # Plot stacked seasonal total
    bottoms = np.zeros(len(stacked_df))
    bar_handles = []
    for season in ['winter', 'spring', 'summer', 'autumn']:
        bar = plt.bar(
            x - 2*width, stacked_df[f'stacked_{season}'], bottom=bottoms,
            width=width, color=colors[season], hatch=hatches[season], alpha=0.7,
            edgecolor="white", label=f"{season.capitalize()} Total"
        )
        bar_handles.append(bar[0])
        bottoms += stacked_df[f'stacked_{season}']

    # Individual season bars (solid)
    plt.bar(x - width, winter_df[metric_display_name] + 100, width=width, color=colors['winter'], alpha=0.7, label="Winter")
    plt.bar(x,         spring_df[metric_display_name] + 100, width=width, color=colors['spring'], alpha=0.7, label="Spring")
    plt.bar(x + width, summer_df[metric_display_name] + 100, width=width, color=colors['summer'], alpha=0.7, label="Summer")
    plt.bar(x + 2*width, autumn_df[metric_display_name] + 100, width=width, color=colors['autumn'], alpha=0.7, label="Autumn")

    plt.ylabel("Relative Emissions (%)")
    plt.title(f"{metric_display_name}: Total vs Seasonal Breakdown")
    plt.xticks(x, [engine_labels[eng] for eng in stacked_df['engine_display']], rotation=0)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Custom legend
    solid_handles = [plt.Rectangle((0,0),1,1,color=colors[s], alpha=0.7) for s in ['winter', 'spring', 'summer', 'autumn']]
    solid_labels = ['Winter', 'Spring', 'Summer', 'Autumn']
    plt.legend(bar_handles + solid_handles, [f"{s} Contribution To Total" for s in solid_labels] + solid_labels, ncol=2)

    filename = f"results_report/emissions/seasonal/stacked_seasonal_barplot_{metric_display_name}_{df_name}.png".replace(" ", "_")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved seasonal stacked bar plot: {filename}")

stacked_seasonal_fuel = prepare_stacked_seasonal_bar_data(
    winter_df, spring_df, summer_df, autumn_df,
    results_df, metric='fuel_kg_sum', metric_display_name='Fuel Flow'
)
plot_stacked_seasonal_bars(stacked_seasonal_fuel, winter_df, spring_df, summer_df, autumn_df, 'Fuel Flow', '1990_emissions')


# NOx
stacked_seasonal_nox = prepare_stacked_seasonal_bar_data(
    winter_df, spring_df, summer_df, autumn_df,
    results_df, metric='nox_sum', metric_display_name='NOx'
)
plot_stacked_seasonal_bars(stacked_seasonal_nox, winter_df, spring_df, summer_df, autumn_df, 'NOx', '1990_emissions')

# nvPM Number
stacked_seasonal_nvpm = prepare_stacked_seasonal_bar_data(
    winter_df, spring_df, summer_df, autumn_df,
    results_df, metric='nvpm_num_sum', metric_display_name='nvPM Number'
)
plot_stacked_seasonal_bars(stacked_seasonal_nvpm, winter_df, spring_df, summer_df, autumn_df, 'nvPM Number', '1990_emissions')

stacked_seasonal_nox_impact = prepare_stacked_seasonal_bar_data(
    winter_df, spring_df, summer_df, autumn_df,
    results_df, metric='nox_impact_sum', metric_display_name='NOx Climate Impact'
)
plot_stacked_seasonal_bars(
    stacked_seasonal_nox_impact, winter_df, spring_df, summer_df, autumn_df,
    'NOx Climate Impact', '1990_emissions'
)

def compute_engine_seasonal_ratios(df, engine_name, value_col='nox_impact_sum'):
    df_engine = df[df['engine_display'] == engine_name].copy()
    df_engine['abs_val'] = df_engine[value_col].abs()

    duplicates = df_engine.groupby(['trajectory', 'diurnal', 'season_adjusted']).size()
    if (duplicates > 1).any():
        print(f"⚠️ Warning: Found duplicates for {engine_name}")
        print(duplicates[duplicates > 1])

    pivot = df_engine.pivot_table(
        index=['trajectory', 'diurnal'],
        columns='season_adjusted',
        values='abs_val',
        aggfunc='sum'
    )

    pivot = pivot.reindex(columns=['winter', 'spring', 'summer', 'autumn'], fill_value=0)
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot[pivot['total'] > 0]
    seasonal_ratios = pivot[['winter', 'spring', 'summer', 'autumn']].div(pivot['total'], axis=0)
    return seasonal_ratios.mean().to_dict()

def plot_seasonal_barplot_stacked_nox_climate_v2(
    winter_df, spring_df, summer_df, autumn_df, results_df, df_name="nox_climate_final"
):
    """
    Creates a stacked seasonal NOₓ climate impact plot with proper annotations.
    Each engine's seasonal ratios are computed from its own NOₓ distribution.
    """

    season_order = ["winter", "spring", "summer", "autumn"]
    full_df = pd.concat([
        winter_df.assign(season_astro='winter'),
        spring_df.assign(season_astro='spring'),
        summer_df.assign(season_astro='summer'),
        autumn_df.assign(season_astro='autumn')
    ])
    full_df = generate_engine_display(full_df)
    full_df = full_df[full_df['engine_display'].isin(engine_order)]

    all_results = []

    for engine in engine_order:
        try:
            winter_val = winter_df.loc[winter_df['engine_display'] == engine, 'NOx Climate Impact'].values[0] + 100
            spring_val = spring_df.loc[spring_df['engine_display'] == engine, 'NOx Climate Impact'].values[0] + 100
            summer_val = summer_df.loc[summer_df['engine_display'] == engine, 'NOx Climate Impact'].values[0] + 100
            autumn_val = autumn_df.loc[autumn_df['engine_display'] == engine, 'NOx Climate Impact'].values[0] + 100
        except IndexError:
            print(f"Skipping {engine}: missing seasonal NOₓ climate data")
            continue

        rad_total = np.mean([winter_val, spring_val, summer_val, autumn_val])
        ratios = compute_engine_seasonal_ratios(results_df, engine, value_col='nox_impact_sum')

        row = {'engine_display': engine, 'rad_total': rad_total}
        for season in season_order:
            row[f'{season}_stack'] = rad_total * ratios.get(season, 0.0)
        all_results.append(row)

    df_plot = pd.DataFrame(all_results)
    df_plot = df_plot.set_index("engine_display").reindex(engine_order).reset_index()

    x = np.arange(len(df_plot))
    width = 0.6
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(df_plot))

    colors = {
        "winter": "#1f77b4", "spring": "#2ca02c",
        "summer": "#d62728", "autumn": "#ff7f0e"
    }

    cfm_row = df_plot[df_plot['engine_display'] == 'CFM1990'].iloc[0]

    for season in season_order:
        bar_values = df_plot[f'{season}_stack'].values
        bars = plt.bar(x, bar_values, bottom=bottom, width=width, label=season.capitalize(), color=colors[season], alpha=0.9)

        for i, (bar_val, btm) in enumerate(zip(bar_values, bottom)):
            if bar_val > 2:
                y_mid = btm + bar_val * 0.5
                engine = df_plot.loc[i, 'engine_display']
                is_baseline = engine == "CFM1990"

                if season in ['winter', 'autumn']:
                    fs_main = 9
                    fs_sub = 7
                else:
                    fs_main = 9
                    fs_sub = 7

                # if engine == "CFM2008":
                #     fs_main -= 1
                #     fs_sub -= 1

                offset_up = fs_main * 0.07
                offset_down = fs_sub * 0.3

                # if is_baseline:
                #     plt.text(x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                #              ha='center', va='center', color='white', fontsize=fs_main)
                #     plt.text(x[i], y_mid - offset_down, "(baseline)",
                #              ha='center', va='center', color='white', fontsize=fs_sub)
                if is_baseline:
                    # Get mean value of actual h2o_impact_sum for this season for CFM1990
                    offset_up = fs_main * 0.07
                    offset_down = fs_sub * 0.3
                    season_data = results_df[
                        (results_df['engine_display'] == "CFM1990") &
                        (results_df['season_adjusted'] == season)
                        ]
                    season_mean_val = season_data['nox_impact_sum'].sum()

                    plt.text(x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                             ha='center', va='center', color='white', fontsize=fs_main)
                    plt.text(x[i], y_mid - offset_down, "(baseline)",
                             ha='center', va='center', color='white', fontsize=fs_sub)
                    # if not np.isnan(season_mean_val):
                    #     avg_label = f"(sum:\n{season_mean_val:.2e} K)"
                    #     plt.text(x[i], y_mid - offset_down - fs_sub * 0.6,
                    #              avg_label, ha='center', va='center',
                    #              color='white', fontsize=fs_sub, linespacing=1.3)
                else:
                    cfm_val = cfm_row[f"{season}_stack"]
                    if cfm_val > 0:
                        diff = 100 * (1 - bar_val / cfm_val)
                        if abs(diff) < 0.05:
                            label_text = "(0.0%)"
                            color = 'white'
                        elif diff > 0:
                            label_text = f"(-{diff:.1f}%)"
                            color = 'lime'
                        else:
                            label_text = f"(+{abs(diff):.1f}%)"
                            color = 'red'

                    else:
                        label_text = ""
                        color = 'black'

                    plt.text(x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                             ha='center', va='center', color='white', fontsize=fs_main)
                    if label_text:
                        plt.text(x[i], y_mid - offset_down, label_text,
                                 ha='center', va='center', color=color, fontsize=fs_sub)

        bottom += bar_values

    plt.xticks(x, [engine_labels.get(eng, eng) for eng in df_plot['engine_display']], rotation=0, ha="center")
    plt.ylabel("Relative Climate Impact (%)")
    plt.title("NOₓ Climate Impact: Seasonal Effect")
    plt.legend(title="Season")
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"results_report/emissions/seasonal/seasonal_barplot_stacked_final_{df_name}.png".replace(" ", "_")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as: {filename}")




plot_seasonal_barplot_stacked_nox_climate_v2(winter_df, spring_df, summer_df, autumn_df, results_df, df_name='nox_climate_final')


def plot_seasonal_barplot_stacked_h2o_climate_v2(
    winter_df, spring_df, summer_df, autumn_df, results_df, df_name="h2o_climate_final"
):
    """
    Creates a stacked seasonal h2o climate impact plot with proper annotations.
    Each engine's seasonal ratios are computed from its own NOₓ distribution.
    """

    season_order = ["winter", "spring", "summer", "autumn"]
    full_df = pd.concat([
        winter_df.assign(season_astro='winter'),
        spring_df.assign(season_astro='spring'),
        summer_df.assign(season_astro='summer'),
        autumn_df.assign(season_astro='autumn')
    ])
    full_df = generate_engine_display(full_df)
    full_df = full_df[full_df['engine_display'].isin(engine_order)]

    all_results = []

    for engine in engine_order:
        try:
            winter_val = winter_df.loc[winter_df['engine_display'] == engine, 'h2o_impact_sum_change'].values[0] + 100
            spring_val = spring_df.loc[spring_df['engine_display'] == engine, 'h2o_impact_sum_change'].values[0] + 100
            summer_val = summer_df.loc[summer_df['engine_display'] == engine, 'h2o_impact_sum_change'].values[0] + 100
            autumn_val = autumn_df.loc[autumn_df['engine_display'] == engine, 'h2o_impact_sum_change'].values[0] + 100
        except IndexError:
            print(f"Skipping {engine}: missing seasonal h2o climate data")
            continue

        rad_total = np.mean([winter_val, spring_val, summer_val, autumn_val])
        ratios = compute_engine_seasonal_ratios(results_df, engine, value_col='h2o_impact_sum')

        row = {'engine_display': engine, 'rad_total': rad_total}
        for season in season_order:
            row[f'{season}_stack'] = rad_total * ratios.get(season, 0.0)
        all_results.append(row)

    df_plot = pd.DataFrame(all_results)
    df_plot = df_plot.set_index("engine_display").reindex(engine_order).reset_index()

    x = np.arange(len(df_plot))
    width = 0.6
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(df_plot))

    colors = {
        "winter": "#1f77b4", "spring": "#2ca02c",
        "summer": "#d62728", "autumn": "#ff7f0e"
    }

    cfm_row = df_plot[df_plot['engine_display'] == 'CFM1990'].iloc[0]

    for season in season_order:
        bar_values = df_plot[f'{season}_stack'].values
        bars = plt.bar(x, bar_values, bottom=bottom, width=width, label=season.capitalize(), color=colors[season], alpha=0.9)

        for i, (bar_val, btm) in enumerate(zip(bar_values, bottom)):
            if bar_val > 2:
                y_mid = btm + bar_val * 0.5
                engine = df_plot.loc[i, 'engine_display']
                is_baseline = engine == "CFM1990"

                if season in ['winter', 'autumn']:
                    fs_main = 9
                    fs_sub = 7
                else:
                    fs_main = 9
                    fs_sub = 7

                # if engine == "CFM2008":
                #     fs_main -= 1
                #     fs_sub -= 1

                offset_up = fs_main * 0.07
                offset_down = fs_sub * 0.3

                # if is_baseline:
                #     plt.text(x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                #              ha='center', va='center', color='white', fontsize=fs_main)
                #     plt.text(x[i], y_mid - offset_down, "(baseline)",
                #              ha='center', va='center', color='white', fontsize=fs_sub)
                if is_baseline:
                    # Get mean value of actual h2o_impact_sum for this season for CFM1990
                    offset_up = fs_main * 0.07
                    offset_down = fs_sub * 0.3
                    season_data = results_df[
                        (results_df['engine_display'] == "CFM1990") &
                        (results_df['season_adjusted'] == season)
                        ]
                    season_mean_val = season_data['h2o_impact_sum'].sum()

                    plt.text(x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                             ha='center', va='center', color='white', fontsize=fs_main)
                    plt.text(x[i], y_mid - offset_down, "(baseline)",
                             ha='center', va='center', color='white', fontsize=fs_sub)
                    # if not np.isnan(season_mean_val):
                    #     avg_label = f"(sum:\n{season_mean_val:.2e} K)"
                    #     plt.text(x[i], y_mid - offset_down - fs_sub * 0.6,
                    #              avg_label, ha='center', va='center',
                    #              color='white', fontsize=fs_sub, linespacing=1.3)
                else:
                    cfm_val = cfm_row[f"{season}_stack"]
                    if cfm_val > 0:
                        diff = 100 * (1 - bar_val / cfm_val)
                        if abs(diff) < 0.05:
                            label_text = "(0.0%)"
                            color = 'white'
                        elif diff > 0:
                            label_text = f"(-{diff:.1f}%)"
                            color = 'lime'
                        else:
                            label_text = f"(+{abs(diff):.1f}%)"
                            color = 'red'

                    else:
                        label_text = ""
                        color = 'black'

                    plt.text(x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                             ha='center', va='center', color='white', fontsize=fs_main)
                    if label_text:
                        plt.text(x[i], y_mid - offset_down, label_text,
                                 ha='center', va='center', color=color, fontsize=fs_sub)

        bottom += bar_values

    plt.xticks(x, [engine_labels.get(eng, eng) for eng in df_plot['engine_display']], rotation=0, ha="center")
    plt.ylabel("Relative Climate Impact (%)")
    plt.title("H₂O Climate Impact: Seasonal Effect")
    plt.legend(title="Season")
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"results_report/emissions/seasonal/seasonal_barplot_stacked_final_{df_name}.png".replace(" ", "_")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as: {filename}")
plot_seasonal_barplot_stacked_h2o_climate_v2(winter_df, spring_df, summer_df, autumn_df, results_df, df_name='h2o_climate_final')


def plot_seasonal_barplot_stacked_co2_climate_v2(
    winter_df, spring_df, summer_df, autumn_df, results_df, df_name="co2_climate_final"
):
    """
    Creates a stacked seasonal h2o climate impact plot with proper annotations.
    Each engine's seasonal ratios are computed from its own NOₓ distribution.
    """

    season_order = ["winter", "spring", "summer", "autumn"]
    full_df = pd.concat([
        winter_df.assign(season_astro='winter'),
        spring_df.assign(season_astro='spring'),
        summer_df.assign(season_astro='summer'),
        autumn_df.assign(season_astro='autumn')
    ])
    full_df = generate_engine_display(full_df)
    full_df = full_df[full_df['engine_display'].isin(engine_order)]

    all_results = []

    for engine in engine_order:
        try:
            winter_val = winter_df.loc[winter_df['engine_display'] == engine, 'co2_conservative_sum_change'].values[0] + 100
            spring_val = spring_df.loc[spring_df['engine_display'] == engine, 'co2_conservative_sum_change'].values[0] + 100
            summer_val = summer_df.loc[summer_df['engine_display'] == engine, 'co2_conservative_sum_change'].values[0] + 100
            autumn_val = autumn_df.loc[autumn_df['engine_display'] == engine, 'co2_conservative_sum_change'].values[0] + 100
        except IndexError:
            print(f"Skipping {engine}: missing seasonal h2o climate data")
            continue

        rad_total = np.mean([winter_val, spring_val, summer_val, autumn_val])
        ratios = compute_engine_seasonal_ratios(results_df, engine, value_col='co2_conservative_sum')

        row = {'engine_display': engine, 'rad_total': rad_total}
        for season in season_order:
            row[f'{season}_stack'] = rad_total * ratios.get(season, 0.0)
        all_results.append(row)

    df_plot = pd.DataFrame(all_results)
    df_plot = df_plot.set_index("engine_display").reindex(engine_order).reset_index()

    x = np.arange(len(df_plot))
    width = 0.6
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(df_plot))

    colors = {
        "winter": "#1f77b4", "spring": "#2ca02c",
        "summer": "#d62728", "autumn": "#ff7f0e"
    }

    cfm_row = df_plot[df_plot['engine_display'] == 'CFM1990'].iloc[0]

    for season in season_order:
        bar_values = df_plot[f'{season}_stack'].values
        bars = plt.bar(x, bar_values, bottom=bottom, width=width, label=season.capitalize(), color=colors[season], alpha=0.9)

        for i, (bar_val, btm) in enumerate(zip(bar_values, bottom)):
            if bar_val > 2:
                y_mid = btm + bar_val * 0.5
                engine = df_plot.loc[i, 'engine_display']
                is_baseline = engine == "CFM1990"

                if season in ['winter', 'autumn']:
                    fs_main = 9
                    fs_sub = 7
                else:
                    fs_main = 9
                    fs_sub = 7

                # if engine == "CFM2008":
                #     fs_main -= 1
                #     fs_sub -= 1

                offset_up = fs_main * 0.07
                offset_down = fs_sub * 0.3

                # if is_baseline:
                #     plt.text(x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                #              ha='center', va='center', color='white', fontsize=fs_main)
                #     plt.text(x[i], y_mid - offset_down, "(baseline)",
                #              ha='center', va='center', color='white', fontsize=fs_sub)
                if is_baseline:
                    # Get mean value of actual h2o_impact_sum for this season for CFM1990
                    offset_up = fs_main * 0.07
                    offset_down = fs_sub * 0.3
                    season_data = results_df[
                        (results_df['engine_display'] == "CFM1990") &
                        (results_df['season_adjusted'] == season)
                        ]
                    season_mean_val = season_data['co2_conservative_sum'].sum()

                    plt.text(x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                             ha='center', va='center', color='white', fontsize=fs_main)
                    plt.text(x[i], y_mid - offset_down, "(baseline)",
                             ha='center', va='center', color='white', fontsize=fs_sub)
                    # if not np.isnan(season_mean_val):
                    #     avg_label = f"(sum:\n{season_mean_val:.2e} K)"
                    #     plt.text(x[i], y_mid - offset_down - fs_sub * 0.6,
                    #              avg_label, ha='center', va='center',
                    #              color='white', fontsize=fs_sub, linespacing=1.3)
                else:
                    cfm_val = cfm_row[f"{season}_stack"]
                    if cfm_val > 0:
                        diff = 100 * (1 - bar_val / cfm_val)
                        if abs(diff) < 0.05:
                            label_text = "(0.0%)"
                            color = 'white'
                        elif diff > 0:
                            label_text = f"(-{diff:.1f}%)"
                            color = 'lime'
                        else:
                            label_text = f"(+{abs(diff):.1f}%)"
                            color = 'red'

                    else:
                        label_text = ""
                        color = 'black'

                    plt.text(x[i], y_mid + offset_up, f"{bar_val:.1f}%",
                             ha='center', va='center', color='white', fontsize=fs_main)
                    if label_text:
                        plt.text(x[i], y_mid - offset_down, label_text,
                                 ha='center', va='center', color=color, fontsize=fs_sub)

        bottom += bar_values

    plt.xticks(x, [engine_labels.get(eng, eng) for eng in df_plot['engine_display']], rotation=0, ha="center")
    plt.ylabel("Relative Climate Impact (%)")
    plt.title("CO₂ Climate Impact: Seasonal Effect")
    plt.legend(title="Season")
    plt.gca().set_axisbelow(True)
    plt.grid(True, linestyle='--', alpha=0.5)

    filename = f"results_report/emissions/seasonal/seasonal_barplot_stacked_final_{df_name}.png".replace(" ", "_")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as: {filename}")
plot_seasonal_barplot_stacked_co2_climate_v2(winter_df, spring_df, summer_df, autumn_df, results_df, df_name='co2_climate_final')
plt.show()