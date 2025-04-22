import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.colors import to_rgba
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

axis_titles = {
    'climate_non_co2': 'Mission Non-CO2 Climate Impact (K)',
    'climate_non_co2_cocip': 'Mission Non-CO2 Climate Impact (Contrail CoCiP) (K)',
    'climate_non_co2_accf_cocip_pcfa': 'Mission Non-CO2 Climate Impact (Contrail aCCF) (K)',
    'climate_co2': 'Mission CO2 Climate Impact (K)',
    'nox_impact_sum': 'Mission NOx Climate Impact (K)',
    'co2_impact_cons_sum': 'Mission CO2 Climate Impact (conservative) (K)',
    'co2_impact_opti_sum': 'Mission CO2 Climate Impact (optimistic) (K)',
    'co2_impact_sum': 'Mission CO2 Climate Impact (K)',
    'contrail_atr20_cocip_sum': 'Mission Contrail Climate Impact (CoCiP) (K)',
    'contrail_atr20_accf_cocip_pcfa_sum': 'Mission Contrail Climate Impact (aCCF) (K)',
    'contrail_atr20_cocip_sum_abs_change': 'Mission Contrail Climate Impact Factor Compared to 1990 (-) old',
    'nox_impact_sum_abs_change': 'Mission NOx Climate Impact Factor Compared to 1990 (-) old',
    'co2_impact_cons_sum_abs_change': 'Mission CO2 Climate Impact (conservative) Factor Compared to 1990 (-) old',
    'co2_impact_opti_sum_abs_change': 'Mission CO2 Climate Impact (optimistic) Factor Compared to 1990 (-) old',
    'co2_impact_sum_abs_change': 'Mission CO2 Climate Impact Factor Compared to 1990 (-) old',
    'climate_non_co2_abs_change': 'Mission Non-CO2 Climate Impact Factor Compared to 1990 (-) old',
    'contrail_atr20_cocip_sum_relative_change': 'Mission Contrail Climate Impact Factor (CoCiP) (RASD) Compared to 1990 (-)',
    'contrail_atr20_accf_cocip_pcfa_sum_relative_change': 'Mission Contrail Climate Impact Factor (aCCF) (RASD) Compared to 1990 (-)',
    'nox_impact_sum_relative_change': 'Mission NOx Climate Impact Factor (RASD) Compared to 1990 (-)',
    'co2_impact_cons_sum_relative_change': 'Mission CO2 Climate Impact (conservative) Factor (RASD) Compared to 1990 (-)',
    'co2_impact_opti_sum_relative_change': 'Mission CO2 Climate Impact (optimistic) Factor (RASD) Compared to 1990 (-)',
    'co2_impact_sum_relative_change': 'Mission CO2 Climate Impact Factor (RASD) Compared to 1990 (-)',
    'climate_non_co2_relative_change': 'Mission Non-CO2 Climate Impact Factor (RASD) Compared to 1990 (-)',
    'climate_non_co2_cocip_relative_change': 'Mission Non-CO2 Climate Impact Factor (Contrail CoCiP) (RASD) Compared to 1990 (-)',
    'climate_non_co2_accf_cocip_pcfa_relative_change': 'Mission Non-CO2 Climate Impact Factor (Contrail aCCF) (RASD) Compared to 1990 (-)'
}

def get_short_label(column_name):
    """ Returns a short, clean version of the column name for the title. """
    if "nox" in column_name:
        return "NOx"
    elif "non_co2_cocip" in column_name:
        return "Non-CO2 (Contrail CoCiP)"
    elif "non_co2_accf_cocip_pcfa" in column_name:
        return "Non-CO2 (Contrail aCCF)"
    elif "co2" in column_name:
        return "CO2"
    elif "contrail_atr20_cocip" in column_name:
        return "Contrail (CoCiP)"
    elif "contrail_atr20_accf_cocip_pcfa" in column_name:
        return "Contrail (aCCF)"
    else:
        return column_name  # Fallback for unknown columns


def scatter_plot(data, engines, x_col, y_col, saf_levels=None, filter_contrails=False, filter_no_contrails=False, filter_daytime=False, filter_nighttime=False, effect=None, save_fig=False):
    """
    Generates a scatter plot based on the specified parameters.
    """
    df_filtered = data[data['engine'].isin(engines)]

    if filter_contrails and filter_no_contrails:
        raise ValueError("Cannot filter both contrail-forming and non-contrail-forming flights at the same time!")
    if filter_daytime and filter_nighttime:
        raise ValueError("Cannot filter both daytime and nighttime flights at the same time!")

    if filter_contrails:
        df_filtered = df_filtered[df_filtered['contrail_atr20_cocip_sum'] != 0]
    if filter_no_contrails:
        df_filtered = df_filtered[df_filtered['contrail_atr20_cocip_sum'] == 0]
    if filter_daytime:
        df_filtered = df_filtered[df_filtered['diurnal'] == 'daytime']
    if filter_nighttime:
        df_filtered = df_filtered[df_filtered['diurnal'] == 'nighttime']
    if saf_levels is not None:
        df_filtered = df_filtered[df_filtered['saf_level'].isin(saf_levels)]

    plt.figure(figsize=(10, 6))

    for engine in engines:
        engine_subset = df_filtered[df_filtered['engine'] == engine]
        if effect == 'diurnal':
            for label, color in diurnal_colors.items():
                subset = engine_subset[engine_subset['diurnal'] == label]
                plt.scatter(subset[x_col], subset[y_col],
                            label=f"{engine_display_names[engine]} - {label.capitalize()}", color=color,
                            marker=engine_groups[engine]['marker'], s=40, alpha=0.5)
        elif effect == 'season':
            for label, color in season_colors.items():
                subset = engine_subset[engine_subset['season'] == label]
                plt.scatter(subset[x_col], subset[y_col], label=f"{engine_display_names[engine]} - {label}",
                            color=color, marker=engine_groups[engine]['marker'], s=40, alpha=0.5)

        elif effect == 'contrails':
            for label, color in contrail_colors.items():
                subset = engine_subset[engine_subset['contrail_atr20_cocip_sum'] != 0] if label == 'formed' else \
                engine_subset[engine_subset['contrail_atr20_cocip_sum'] == 0]
                plt.scatter(subset[x_col], subset[y_col],
                            label=f"{engine_display_names[engine]} - {label.replace('_', ' ').capitalize()}",
                            color=color, marker=engine_groups[engine]['marker'], s=40, alpha=0.5)

        elif saf_levels is not None:
            for saf_level in saf_levels:
                for label, color in diurnal_colors.items() if effect == 'diurnal' else [(None, None)]:
                    subset = engine_subset[(engine_subset['saf_level'] == saf_level) & (
                        engine_subset['diurnal'] == label if label else True)]
                    label_text = f"{engine_display_names[engine]}" if saf_level == 0 else f"{engine_display_names[engine]}-{saf_level}"
                    if label:
                        label_text += f" - {label.capitalize()}"
                    plt.scatter(subset[x_col], subset[y_col], label=label_text,
                                color=saf_colors.get((engine, saf_level), 'black') if not label else color,
                                marker=engine_groups[engine]['marker'], s=40, alpha=0.5)
        else:
            plt.scatter(engine_subset[x_col], engine_subset[y_col], label=f"{engine_display_names.get(engine, engine)}",
                        color=engine_groups[engine]['color'], marker=engine_groups[engine]['marker'], s=40, alpha=0.5)

    if effect == 'diurnal':
        for label, color in diurnal_colors.items():
            subset = df_filtered[df_filtered['diurnal'] == label]
            if not subset.empty:
                x_mean = subset[x_col].mean()
                y_mean = subset[y_col].mean()

                # Plot vertical and horizontal lines
                plt.axvline(x=x_mean, color=color, linestyle='-', linewidth=1.5)#, label=f"{label.capitalize()} Avg X")
                plt.axhline(y=y_mean, color=color, linestyle='--', linewidth=1.5)#, label=f"{label.capitalize()} Avg Y")

    elif effect == 'season':
        for label, color in season_colors.items():
            subset = df_filtered[df_filtered['season'] == label]
            if not subset.empty:
                x_mean = subset[x_col].mean()
                y_mean = subset[y_col].mean()
                plt.axvline(x=x_mean, color=color, linestyle='-', linewidth=1.5)#, label=f"{label} Avg X")
                plt.axhline(y=y_mean, color=color, linestyle='--', linewidth=1.5)#, label=f"{label} Avg Y")

    elif effect == 'contrails':
        for label, color in contrail_colors.items():
            subset = df_filtered[df_filtered['contrail_atr20_cocip_sum'] != 0] if label == 'formed' else df_filtered[
                df_filtered['contrail_atr20_cocip_sum'] == 0]
            if not subset.empty:
                x_mean = subset[x_col].mean()
                y_mean = subset[y_col].mean()
                plt.axvline(x=x_mean, color=color, linestyle='-', linewidth=1.5)#,
                           # label=f"{label.replace('_', ' ').capitalize()} Avg X")
                plt.axhline(y=y_mean, color=color, linestyle='--', linewidth=1.5)#,
                            #label=f"{label.replace('_', ' ').capitalize()} Avg Y")


    else:
        saf_present = saf_levels is not None  # Check if SAF levels are part of the filtering
        for engine in engines:
            if saf_present:
                # Loop over selected SAF levels and calculate separate averages
                for saf in saf_levels:
                    subset = df_filtered[(df_filtered['engine'] == engine) & (df_filtered['saf_level'] == saf)]
                    if not subset.empty:
                        x_mean = subset[x_col].mean()
                        y_mean = subset[y_col].mean()
                        plt.axvline(x=x_mean, color=saf_colors.get((engine, saf), 'black'), linestyle='-',
                                    linewidth=1.5)#,
                                    #label=f"{engine_display_names[engine]}-{saf} Avg X")
                        plt.axhline(y=y_mean, color=saf_colors.get((engine, saf), 'black'), linestyle='--',
                                    linewidth=1.5)#,
                                    #label=f"{engine_display_names[engine]}-{saf} Avg Y")
            else:
                # Only use SAF=0 data when no specific SAF filtering is applied
                subset = df_filtered[(df_filtered['engine'] == engine) & (df_filtered['saf_level'] == 0)]
                if not subset.empty:
                    x_mean = subset[x_col].mean()
                    y_mean = subset[y_col].mean()
                    plt.axvline(x=x_mean, color=engine_groups[engine]['color'], linestyle='-', linewidth=1.5)#,
                                #label=f"{engine_display_names[engine]} Avg X")
                    plt.axhline(y=y_mean, color=engine_groups[engine]['color'], linestyle='--', linewidth=1.5)#,
                                #label=f"{engine_display_names[engine]} Avg Y")

    y_label = axis_titles[y_col]
    if saf_levels is not None and set(saf_levels) == {0} and set(engines).issubset(
            {'GTF1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi'}) and (y_col == 'co2_impact_cons_sum' or y_col == 'co2_impact_opti_sum'):
        y_label = axis_titles['co2_impact_sum']
    if saf_levels is not None and set(saf_levels) == {0} and set(engines).issubset(
            {'GTF1990', 'GTF2000', 'GTF', 'GTF2035', 'GTF2035_wi'}) and (y_col == 'co2_impact_cons_sum_abs_change' or y_col == 'co2_impact_opti_sum_abs_change'):
        y_label = axis_titles['co2_impact_sum_abs_change']

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

    # Set custom axis limits for specific columns
    contrail_cols = {'contrail_atr20_cocip_sum', 'contrail_atr20_accf_cocip_pcfa_sum'}
    axis_min, axis_max = -0.4e-9, 1.4e-9
    if x_col in contrail_cols:
        plt.xlim(axis_min, axis_max)
    if y_col in contrail_cols:
        plt.ylim(axis_min, axis_max)

    plt.xlabel(axis_titles.get(x_col, x_col))
    plt.ylabel(y_label)
    # Modify title based on filters
    title_effect = f"({effect.capitalize()} Effect)" if effect else ""
    title_contrails = "(Only Flights with Contrail Formation)" if filter_contrails else \
        "(Only Flights without Contrail Formation)" if filter_no_contrails else ""

    title_diurnal = "(Only Daytime Flights)" if filter_daytime else \
        "(Only Nighttime Flights)" if filter_nighttime else ""

    # **Generate Cleaned Title**
    x_label_short = get_short_label(x_col)
    y_label_short = get_short_label(y_col)
    title_text = f"{y_label_short} vs {x_label_short} Climate Impact"
    # plt.title(title_text)

    plt.title(f"{title_text} {title_effect} {title_contrails} {title_diurnal}")
    plt.legend()

    # **Save figure with a compact filename**
    if save_fig:
        filename_parts = [
            f"Engines-{'_'.join(engines)}",
            f"X-{x_col}_Y-{y_col}",
            f"SAF{','.join(map(str, saf_levels))}" if saf_levels else "",
            "Diurnal" if effect == "diurnal" else "",
            "Seasonal" if effect == "season" else "",
            "Contrails" if filter_contrails else "NoContrails" if filter_no_contrails else "",
            "Day" if filter_daytime else "Night" if filter_nighttime else ""
        ]
        filename = "_".join(filter(None, filename_parts))  # Remove empty parts
        filename = filename[:100]  # Limit to 100 characters
        filepath = f"results_report/scatter/{filename}.png"

        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Figure saved as: {filepath}")
    # plt.show()


scatter_plot(results_df, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='contrail_atr20_cocip_sum', y_col='nox_impact_sum',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)

scatter_plot(results_df, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='contrail_atr20_accf_cocip_pcfa_sum', y_col='nox_impact_sum',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)
#
# scatter_plot(results_df, engines=['GTF2035','GTF2035_wi'], x_col='contrail_atr20_cocip_sum', y_col='nox_impact_sum',
#              saf_levels=[0,20,100], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)
#
scatter_plot(results_df, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='climate_non_co2_cocip', y_col='co2_impact_cons_sum',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)

scatter_plot(results_df, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='climate_non_co2_accf_cocip_pcfa', y_col='co2_impact_cons_sum',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)

scatter_plot(results_df, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='contrail_atr20_cocip_sum', y_col='contrail_atr20_accf_cocip_pcfa_sum',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)
#
# scatter_plot(results_df, engines=['GTF2035','GTF2035_wi'], x_col='climate_non_co2', y_col='co2_impact_cons_sum',
#              saf_levels=[0,20,100], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)
#
# scatter_plot(results_df, engines=['GTF2035','GTF2035_wi'], x_col='climate_non_co2', y_col='co2_impact_opti_sum',
#              saf_levels=[0,20,100], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)

plt.show()


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

# gtf_df = results_df[
#     (results_df['engine'] == 'GTF') &
#     (results_df['contrail_atr20_cocip_sum'] != 0)
# ].copy()
# # Compute absolute values
# gtf_df['abs_cocip'] = gtf_df['contrail_atr20_cocip_sum'].abs()
# gtf_df['abs_accf'] = gtf_df['contrail_atr20_accf_sum'].abs()
# gtf_df['abs_nox'] = gtf_df['nox_impact_sum'].abs()
#
# # Split by diurnal
# night_df = gtf_df[gtf_df['diurnal'] == 'nighttime']
# day_df = gtf_df[gtf_df['diurnal'] == 'daytime']
#
# # Compute means of absolute values
# night_mean_cocip = night_df['abs_cocip'].mean()
# night_mean_accf = night_df['abs_accf'].mean()
# night_mean_nox = night_df['abs_nox'].mean()
#
# day_mean_cocip = day_df['abs_cocip'].mean()
# day_mean_accf = day_df['abs_accf'].mean()
# day_mean_nox = day_df['abs_nox'].mean()
#
# # Print results
# print("== Mean ABS Values (GTF engine only) ==")
# print(f"Nighttime — CoCiP: {night_mean_cocip:.6e}, ACCF: {night_mean_accf:.6e}, NOx Impact: {night_mean_nox:.6e}")
# print(f"Daytime   — CoCiP: {day_mean_cocip:.6e}, ACCF: {day_mean_accf:.6e}, NOx Impact: {day_mean_nox:.6e}")
#
# Baseline: GTF1990, saf_level = 0
baseline_df = results_df[(results_df['engine'] == 'GTF1990') & (results_df['saf_level'] == 0)]
#
# Define metrics for comparison
common_metrics = [
    'nox_impact_sum', 'co2_impact_cons_sum', 'co2_impact_opti_sum', 'h2o_impact_sum',
    'climate_non_co2_cocip', 'climate_non_co2_accf', 'climate_non_co2_accf_cocip_pcfa',
    'climate_total_cons_cocip', 'climate_total_opti_cocip', 'climate_total_cons_accf', 'climate_total_opti_accf',
    'climate_total_cons_accf_cocip_pcfa', 'climate_total_opti_accf_cocip_pcfa'
]
contrail_metrics = ['contrail_atr20_cocip_sum', 'contrail_atr20_accf_sum', 'contrail_atr20_accf_cocip_pcfa_sum'] + common_metrics

# # Helper function to calculate absolute percentage changes
# def calculate_absolute_changes(df, metrics):
#     merged_df = df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))
#     problematic_rows = []
#
#     for metric in metrics:
#         baseline_metric = np.abs(merged_df[f'{metric}_baseline'])  # Take absolute values
#         new_metric = np.abs(merged_df[metric])
#
#         # Avoid division by zero by marking problematic rows
#         zero_baseline_mask = baseline_metric == 0
#         if zero_baseline_mask.any():
#             problematic_entries = merged_df.loc[zero_baseline_mask, ['trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection']]
#             problematic_rows.extend(problematic_entries.to_dict('records'))
#
#         # Compute absolute percentage change
#         merged_df[f'{metric}_abs_change'] = np.where(
#             zero_baseline_mask,
#             np.nan,  # Avoids infinite values
#             (new_metric - baseline_metric) / baseline_metric
#         )
#
#     # Collect entries with any NaN values in the change columns
#     nan_rows = merged_df.loc[merged_df[[f'{metric}_abs_change' for metric in metrics]].isna().any(axis=1),
#                              ['trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection']]
#
#     if not nan_rows.empty:
#         print("Rows where baseline impact was zero (division by zero avoided):")
#         print(nan_rows)
#
#     # Drop baseline columns and compute averages
#     columns_to_drop = [col for col in merged_df.columns if '_baseline' in col]
#     merged_df = merged_df.drop(columns=columns_to_drop)
#     flight_level_df = merged_df[['trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection'] +
#                                 [f'{metric}_abs_change' for metric in metrics]]
#
#     return flight_level_df, problematic_rows
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
    flight_level_df = merged_df#[['trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection'] +
                                #[f'{metric}_relative_change' for metric in metrics]]

    return flight_level_df

# Compute absolute changes for both contrail-forming and non-contrail flights
# contrail_no_changes_abs, _= calculate_absolute_changes(contrail_no_df, common_metrics)
# contrail_yes_changes_abs, _ = calculate_absolute_changes(contrail_yes_df, contrail_metrics)
contrail_no_cocip_changes= calculate_relative_changes(contrail_no_df_cocip, common_metrics)
contrail_yes_cocip_changes = calculate_relative_changes(contrail_yes_df_cocip, contrail_metrics)
contrail_yes_all_cocip_changes = calculate_relative_changes(contrail_yes_all_df_cocip, contrail_metrics)
contrail_no_accf_changes= calculate_relative_changes(contrail_no_df_accf, common_metrics)
contrail_yes_accf_changes = calculate_relative_changes(contrail_yes_df_accf, contrail_metrics)
contrail_yes_all_accf_changes = calculate_relative_changes(contrail_yes_all_df_accf, contrail_metrics)
#
#
#
scatter_plot(contrail_yes_cocip_changes, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='contrail_atr20_cocip_sum_relative_change', y_col='nox_impact_sum_relative_change',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)
scatter_plot(contrail_yes_accf_changes, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='contrail_atr20_accf_cocip_pcfa_sum_relative_change', y_col='nox_impact_sum_relative_change',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)
# #
nox_2000_1990 = contrail_yes_cocip_changes
nox_2000_1990.loc[nox_2000_1990['engine'] == 'GTF2000', 'nox_impact_sum_relative_change'] = 0.0
# min_value = nox_2000_1990['contrail_atr20_cocip_sum_relative_change'].min()
# print("Minimum contrail ATR20 CoCiP sum relative change:", min_value)
#
# filtered_df = nox_2000_1990[
#     (nox_2000_1990['engine'] == 'GTF2000') &
#     (nox_2000_1990['contrail_atr20_cocip_sum_relative_change'] > 0)
# ]

# # Select and print the relevant columns: 'trajectory', 'season', and 'diurnal'
# print(filtered_df[['trajectory', 'season', 'diurnal']])

scatter_plot(nox_2000_1990, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='contrail_atr20_cocip_sum_relative_change', y_col='nox_impact_sum_relative_change',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)


nox_2000_1990_accf = contrail_yes_accf_changes
nox_2000_1990_accf.loc[nox_2000_1990_accf['engine'] == 'GTF2000', 'nox_impact_sum_relative_change'] = 0.0
nox_2000_1990_accf.loc[nox_2000_1990_accf['engine'] == 'GTF2000', 'climate_non_co2_accf_cocip_pcfa_relative_change'] = 0.0
# min_value = nox_2000_1990_accf['contrail_atr20_cocip_sum_relative_change'].min()
# print("Minimum contrail ATR20 CoCiP sum relative change:", min_value)

# filtered_df = nox_2000_1990_accf[
#     (nox_2000_1990_accf['engine'] == 'GTF2000') &
#     (nox_2000_1990_accf['contrail_atr20_cocip_sum_relative_change'] > 0)
# ]
#
# # Select and print the relevant columns: 'trajectory', 'season', and 'diurnal'
# print(filtered_df[['trajectory', 'season', 'diurnal']])

scatter_plot(nox_2000_1990_accf, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='contrail_atr20_accf_cocip_pcfa_sum_relative_change', y_col='nox_impact_sum_relative_change',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)

scatter_plot(contrail_yes_cocip_changes, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='climate_non_co2_cocip_relative_change', y_col='co2_impact_cons_sum_relative_change',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)

scatter_plot(nox_2000_1990_accf, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='climate_non_co2_accf_cocip_pcfa_relative_change', y_col='co2_impact_cons_sum_relative_change',
             saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)

# scatter_plot(contrail_yes_changes, engines=['GTF2035', 'GTF2035_wi'], x_col='contrail_atr20_cocip_sum_relative_change', y_col='nox_impact_sum_relative_change',
#              saf_levels=[0,20,100], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)
# #
# scatter_plot(contrail_yes_changes, engines=['GTF2035', 'GTF2035_wi'], x_col='climate_non_co2_relative_change', y_col='co2_impact_cons_sum_relative_change',
#              saf_levels=[0,20,100], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)
#
# scatter_plot(contrail_yes_changes, engines=['GTF2035', 'GTF2035_wi'], x_col='climate_non_co2_relative_change', y_col='co2_impact_opti_sum_relative_change',
#              saf_levels=[0,20,100], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False, save_fig=True)
#
# scatter_plot(contrail_yes_changes, engines=['GTF'], x_col='contrail_atr20_cocip_sum_relative_change', y_col='nox_impact_sum_relative_change',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect='diurnal', filter_daytime=False, save_fig=True)
#
# scatter_plot(contrail_yes_changes, engines=['GTF'], x_col='climate_non_co2_relative_change', y_col='co2_impact_cons_sum_relative_change',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect='diurnal', filter_daytime=False, save_fig=True)
#
# scatter_plot(contrail_yes_changes, engines=['GTF2035'], x_col='contrail_atr20_cocip_sum_relative_change', y_col='nox_impact_sum_relative_change',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect='diurnal', filter_daytime=False)
#
# scatter_plot(contrail_yes_changes, engines=['GTF2035'], x_col='climate_non_co2_relative_change', y_col='co2_impact_cons_sum_relative_change',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect='diurnal', filter_daytime=False)
#
# scatter_plot(contrail_yes_changes, engines=['GTF'], x_col='contrail_atr20_cocip_sum', y_col='nox_impact_sum',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect='diurnal', filter_daytime=False)
#
# scatter_plot(contrail_yes_changes, engines=['GTF2035'], x_col='contrail_atr20_cocip_sum', y_col='nox_impact_sum',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect='diurnal', filter_daytime=False)
#
# # scatter_plot(contrail_no_changes, engines=['GTF1990','GTF2000','GTF', 'GTF2035'], x_col='climate_non_co2_relative_change', y_col='co2_impact_cons_sum_relative_change',
# #              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=False)
# scatter_plot(contrail_yes_changes, engines=['GTF'], x_col='contrail_atr20_cocip_sum', y_col='nox_impact_sum',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=True)
#
# scatter_plot(contrail_yes_changes, engines=['GTF'], x_col='contrail_atr20_cocip_sum', y_col='nox_impact_sum',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_nighttime=True)
#
# scatter_plot(contrail_yes_changes, engines=['GTF2035'], x_col='contrail_atr20_cocip_sum', y_col='nox_impact_sum',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_daytime=True)
#
# scatter_plot(contrail_yes_changes, engines=['GTF2035'], x_col='contrail_atr20_cocip_sum', y_col='nox_impact_sum',
#              saf_levels=[0], filter_contrails=False, filter_no_contrails=False,effect=None, filter_nighttime=True)
#
# plt.show()
#
def plot_climate_impact_pies(df, engines, saf_levels, df_name, daytime_filter=False, nighttime_filter=False,
                             season_filter=None, save_fig=False):
    """
    Generates pie charts for each engine configuration, showing climate impact contributions.
    """

    species_colors = {
        'CO₂': 'tab:orange',
        'NOx': 'tab:blue',
        'Contrails (CoCiP)': 'tab:red',
        'Contrails (aCCF)': 'tab:green',
        'Water Vapour': 'tab:grey'
    }

    impact_columns = {
        'CO₂ (Conservative)': 'co2_impact_cons_sum',
        'CO₂ (Optimistic)': 'co2_impact_opti_sum',
        'NOx': 'nox_impact_sum',
        'Contrails (CoCiP)': 'contrail_atr20_cocip_sum',
        'Contrails (aCCF)': 'contrail_atr20_accf_cocip_pcfa_sum',
        'Water Vapour': 'h2o_impact_sum'
    }

    if '_accf' in df_name:
        contrail_label = 'Contrails (aCCF)'
    else:
        contrail_label = 'Contrails (CoCiP)'

    impact_cols_used = {
        'CO₂ (Conservative)': impact_columns['CO₂ (Conservative)'],
        'CO₂ (Optimistic)': impact_columns['CO₂ (Optimistic)'],
        'NOx': impact_columns['NOx'],
        contrail_label: impact_columns[contrail_label],  # Key is now 'Contrails (aCCF)' or 'Contrails (CoCiP)'
        'Water Vapour': impact_columns['Water Vapour']
    }

    if season_filter:
        df = df[df['season'] == season_filter]
        filter_label = season_filter
    elif daytime_filter and not nighttime_filter:
        df = df[df['diurnal'] == "daytime"]
        filter_label = "Daytime"
    elif nighttime_filter and not daytime_filter:
        df = df[df['diurnal'] == "nighttime"]
        filter_label = "Nighttime"
    else:
        filter_label = ""

    # if '_accf' in df_name and 'GTF1990' in df['engine'].unique() and 'GTF2000' in df['engine'].unique():
    #     gtf1990_data = df[df['engine'] == 'GTF1990'].copy()
    #     gtf2000_copy = gtf1990_data.copy()
    #     gtf2000_copy['engine'] = 'GTF2000'
    #     df = pd.concat([df[df['engine'] != 'GTF2000'], gtf2000_copy], ignore_index=True)

    num_pies = sum(2 if saf in [20, 100] else 1 for saf in saf_levels for _ in engines)
    ncols = 1 if num_pies == 1 else 2 if num_pies == 2 else 3
    nrows = math.ceil(num_pies / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    pie_index = 0
    for engine in engines:
        for saf in saf_levels:
            df_filtered = df[(df['engine'] == engine) & (df['saf_level'] == saf)]
            if df_filtered.empty:
                continue

            df_filtered = df_filtered.copy()
            df_filtered[list(impact_columns.values())] = df_filtered[list(impact_columns.values())].abs()

            contrail_col = impact_cols_used[contrail_label]

            if saf in [20, 100]:
                df_filtered["total_impact_cons"] = (
                        df_filtered["co2_impact_cons_sum"] +
                        df_filtered["nox_impact_sum"] +
                        df_filtered[contrail_col] +
                        df_filtered["h2o_impact_sum"]
                )
                df_filtered["total_impact_opti"] = (
                        df_filtered["co2_impact_opti_sum"] +
                        df_filtered["nox_impact_sum"] +
                        df_filtered[contrail_col] +
                        df_filtered["h2o_impact_sum"]
                )
            else:
                df_filtered["total_impact"] = (
                        df_filtered["co2_impact_cons_sum"] +
                        df_filtered["nox_impact_sum"] +
                        df_filtered[contrail_col] +
                        df_filtered["h2o_impact_sum"]
                )


            if saf in [20, 100]:
                impact_values_cons = {
                    label: (df_filtered[column] / df_filtered["total_impact_cons"]).mean()
                    for label, column in impact_cols_used.items()
                    if "Optimistic" not in label
                }
                impact_values_opti = {
                    label: (df_filtered[column] / df_filtered["total_impact_opti"]).mean()
                    for label, column in impact_cols_used.items()
                    if "Conservative" not in label
                }
            else:
                impact_values = {
                    label: (df_filtered[column] / df_filtered["total_impact"]).mean()
                    for label, column in impact_cols_used.items()
                    if "Optimistic" not in label
                }

            def filter_nonzero(data):
                labels, values, colors = [], [], []
                for label, value in data.items():
                    if value > 0:
                        clean_label = "CO₂" if "CO₂" in label else label
                        labels.append(clean_label)
                        values.append(value)
                        colors.append(to_rgba(species_colors[clean_label], alpha=0.7))
                return labels, values, colors

            if saf in [20, 100]:
                cons_labels, cons_values, cons_colors = filter_nonzero(impact_values_cons)
                opti_labels, opti_values, opti_colors = filter_nonzero(impact_values_opti)
            else:
                cons_labels, cons_values, cons_colors = filter_nonzero(impact_values)

            if not cons_values and (not opti_values if saf in [20, 100] else True):
                continue

            if engine == "GTF2035_wi":
                engine_title = "GTF2035WI"
            elif engine == "GTF1990":
                engine_title = "CFM1990"
            elif engine == "GTF2000":
                engine_title = "CFM2008"
            else:
                engine_title = engine
            saf_label = f" SAF {saf}" if saf != 0 else ""
            plot_title = f"{engine_title}{saf_label} ({filter_label})" if filter_label else f"{engine_title}{saf_label}"

            def plot_pie(ax, values, labels, colors, title):
                def autopct_func(pct):
                    return f'{pct:.1f}%' if pct >= 2 else ''

                wedges, texts, autotexts = ax.pie(
                    values, labels=labels, autopct=autopct_func,
                    colors=colors, startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 2},
                    textprops={'color': 'black', 'fontsize': 12}
                )

                # Adjust font size and color for percentage text
                for autotext in autotexts:
                    try:
                        pct = float(autotext.get_text().strip('%'))
                        autotext.set_fontsize(4 if pct < 4.1 else 7 if pct < 7 else 9)
                        autotext.set_color('white')
                    except ValueError:
                        autotext.set_text('')

                # Identify and conditionally adjust CO₂ and Water Vapour labels if they are close and one is <2%
                label_positions = {}
                for text, label in zip(texts, labels):
                    label_positions[label] = text.get_position()

                if 'CO₂' in label_positions and 'Water Vapour' in label_positions:
                    y_co2 = label_positions['CO₂'][1]
                    y_h2o = label_positions['Water Vapour'][1]
                    dist = abs(y_co2 - y_h2o)

                    value_dict = dict(zip(labels, values))
                    small_co2 = value_dict.get('CO₂', 1) < 0.02
                    small_h2o = value_dict.get('Water Vapour', 1) < 0.02

                    if dist < 0.1 and (small_co2 or small_h2o):
                        for text, label in zip(texts, labels):
                            x, y = text.get_position()
                            if label == 'CO₂':
                                text.set_position((x, y - 0.05))
                            elif label == 'Water Vapour':
                                text.set_position((x, y + 0.05))

                ax.set_title(title)

            if saf in [20, 100]:
                plot_pie(axes[pie_index], cons_values, cons_labels, cons_colors, f"{plot_title} (Conservative)")
                pie_index += 1
                plot_pie(axes[pie_index], opti_values, opti_labels, opti_colors, f"{plot_title} (Optimistic)")
            else:
                plot_pie(axes[pie_index], cons_values, cons_labels, cons_colors, plot_title)

            pie_index += 1

    plt.tight_layout()
    if save_fig:
        filename = f"results_report/portions/pie_chart_{df_name.replace(' ', '_')}_{'_'.join(engines)}_SAF{'_'.join(map(str, saf_levels))}{'_' + filter_label if filter_label else ''}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {filename}")
#
# # # For contrail_yes_changes
# # yes_climb = contrail_yes_changes[contrail_yes_changes['flight_phase'] == 'climb']
# # yes_cruise = contrail_yes_changes[contrail_yes_changes['flight_phase'] == 'cruise']
# # yes_descent = contrail_yes_changes[contrail_yes_changes['flight_phase'] == 'descent']
# #
# # # For contrail_no_changes
# # no_climb = contrail_no_changes[contrail_no_changes['flight_phase'] == 'climb']
# # no_cruise = contrail_no_changes[contrail_no_changes['flight_phase'] == 'cruise']
# # no_descent = contrail_no_changes[contrail_no_changes['flight_phase'] == 'descent']
# # plot_climate_impact_pies(yes_climb,
# #                          engines=['GTF'],
# #                          saf_levels=[0], save_fig=True, df_name='yes_climb')
# # plot_climate_impact_pies(yes_cruise,
# #                          engines=['GTF'],
# #                          saf_levels=[0], save_fig=True, df_name='yes_cruise')
# # plot_climate_impact_pies(yes_descent,
# #                          engines=['GTF'],
# #                          saf_levels=[0], save_fig=True, df_name='yes_descent')
# #
# # plot_climate_impact_pies(no_climb,
# #                          engines=['GTF'],
# #                          saf_levels=[0], save_fig=True, df_name='no_climb')
# # plot_climate_impact_pies(no_cruise,
# #                          engines=['GTF'],
# #                          saf_levels=[0], save_fig=True, df_name='no_cruise')
# # plot_climate_impact_pies(no_descent,
# #                          engines=['GTF'],
# #                          saf_levels=[0], save_fig=True, df_name='no_descent')
#
plot_climate_impact_pies(contrail_yes_cocip_changes,
                         engines=['GTF1990', 'GTF2000'],
                         saf_levels=[0], save_fig=True, df_name='contrail_yes_cocip_changes')

plot_climate_impact_pies(contrail_yes_all_cocip_changes,
                         engines=['GTF1990', 'GTF2000'],
                         saf_levels=[0], save_fig=True, df_name='contrail_yes_all_cocip_changes')

plot_climate_impact_pies(contrail_yes_cocip_changes,
                         engines=['GTF', 'GTF2035', 'GTF2035_wi'],
                         saf_levels=[0], save_fig=True, df_name='contrail_yes_cocip_changes')

plot_climate_impact_pies(contrail_yes_all_cocip_changes,
                         engines=['GTF', 'GTF2035', 'GTF2035_wi'],
                         saf_levels=[0], save_fig=True, df_name='contrail_yes_all_cocip_changes')

plot_climate_impact_pies(contrail_yes_cocip_changes,
                         engines=['GTF2035'],
                         saf_levels=[20], save_fig=True, df_name='contrail_yes_cocip_changes')

plot_climate_impact_pies(contrail_yes_cocip_changes,
                         engines=['GTF2035'],
                         saf_levels=[100], save_fig=True, df_name='contrail_yes_cocip_changes')

plot_climate_impact_pies(contrail_yes_cocip_changes,
                         engines=['GTF2035_wi'],
                         saf_levels=[20], save_fig=True, df_name='contrail_yes_cocip_changes')

plot_climate_impact_pies(contrail_yes_cocip_changes,
                         engines=['GTF2035_wi'],
                         saf_levels=[100], save_fig=True, df_name='contrail_yes_cocip_changes')

plot_climate_impact_pies(contrail_yes_accf_changes,
                         engines=['GTF1990', 'GTF2000'],
                         saf_levels=[0], save_fig=True, df_name='contrail_yes_accf_changes')
plot_climate_impact_pies(contrail_yes_all_accf_changes,
                         engines=['GTF1990', 'GTF2000'],
                         saf_levels=[0], save_fig=True, df_name='contrail_yes_all_accf_changes')

plot_climate_impact_pies(contrail_yes_accf_changes,
                         engines=['GTF', 'GTF2035', 'GTF2035_wi'],
                         saf_levels=[0], save_fig=True, df_name='contrail_yes_accf_changes')

plot_climate_impact_pies(contrail_yes_all_accf_changes,
                         engines=['GTF', 'GTF2035', 'GTF2035_wi'],
                         saf_levels=[0], save_fig=True, df_name='contrail_yes_all_accf_changes')

plot_climate_impact_pies(contrail_yes_accf_changes,
                         engines=['GTF2035'],
                         saf_levels=[20], save_fig=True, df_name='contrail_yes_accf_changes')

plot_climate_impact_pies(contrail_yes_accf_changes,
                         engines=['GTF2035'],
                         saf_levels=[100], save_fig=True, df_name='contrail_yes_accf_changes')

plot_climate_impact_pies(contrail_yes_accf_changes,
                         engines=['GTF2035_wi'],
                         saf_levels=[20], save_fig=True, df_name='contrail_yes_accf_changes')

plot_climate_impact_pies(contrail_yes_accf_changes,
                         engines=['GTF2035_wi'],
                         saf_levels=[100], save_fig=True, df_name='contrail_yes_accf_changes')
#
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF1990', 'GTF2000'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_changes', daytime_filter=True)
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF1990', 'GTF2000'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_changes', nighttime_filter=True)
#
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF', 'GTF2035', 'GTF2035_wi'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_changes', daytime_filter=True)
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF', 'GTF2035', 'GTF2035_wi'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_changes', nighttime_filter=True)
#
# plot_climate_impact_pies(contrail_yes_all_df,
#                          engines=['GTF', 'GTF2035', 'GTF2035_wi'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_all', nighttime_filter=True)
#
# plot_climate_impact_pies(gtf_df,
#                          engines=['GTF'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_all_gtf', nighttime_filter=True)
#
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF2035'],
#                          saf_levels=[20], save_fig=True, df_name='contrail_yes_changes', daytime_filter=True)
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF2035'],
#                          saf_levels=[20], save_fig=True, df_name='contrail_yes_changes', nighttime_filter=True)
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF2035'],
#                          saf_levels=[100], save_fig=True, df_name='contrail_yes_changes', daytime_filter=True)
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF2035'],
#                          saf_levels=[100], save_fig=True, df_name='contrail_yes_changes', nighttime_filter=True)
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF2035_wi'],
#                          saf_levels=[20], save_fig=True, df_name='contrail_yes_changes', daytime_filter=True)
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF2035_wi'],
#                          saf_levels=[100], save_fig=True, df_name='contrail_yes_changes', daytime_filter=True)
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF2035_wi'],
#                          saf_levels=[20], save_fig=True, df_name='contrail_yes_changes', nighttime_filter=True)
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF2035_wi'],
#                          saf_levels=[100], save_fig=True, df_name='contrail_yes_changes', nighttime_filter=True)
#
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_changes', season_filter='2023-02-06')
#
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_changes', season_filter='2023-05-05')
#
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_changes', season_filter='2023-08-06')
#
# plot_climate_impact_pies(contrail_yes_changes,
#                          engines=['GTF'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_yes_changes', season_filter='2023-11-06')
#
#
# """no contrails"""
# """general"""
plot_climate_impact_pies(contrail_no_cocip_changes,
                         engines=['GTF1990', 'GTF2000'],
                         saf_levels=[0], save_fig=True, df_name='contrail_no_cocip_changes')

plot_climate_impact_pies(contrail_no_cocip_changes,
                         engines=['GTF', 'GTF2035', 'GTF2035_wi'],
                         saf_levels=[0], save_fig=True, df_name='contrail_no_cocip_changes')

plot_climate_impact_pies(contrail_no_cocip_changes,
                         engines=['GTF2035'],
                         saf_levels=[20], save_fig=True, df_name='contrail_no_cocip_changes')

plot_climate_impact_pies(contrail_no_cocip_changes,
                         engines=['GTF2035'],
                         saf_levels=[100], save_fig=True, df_name='contrail_no_cocip_changes')

plot_climate_impact_pies(contrail_no_cocip_changes,
                         engines=['GTF2035_wi'],
                         saf_levels=[20], save_fig=True, df_name='contrail_no_cocip_changes')

plot_climate_impact_pies(contrail_no_cocip_changes,
                         engines=['GTF2035_wi'],
                         saf_levels=[100], save_fig=True, df_name='contrail_no_cocip_changes')

plot_climate_impact_pies(contrail_no_accf_changes,
                         engines=['GTF1990', 'GTF2000'],
                         saf_levels=[0], save_fig=True, df_name='contrail_no_accf_changes')

plot_climate_impact_pies(contrail_no_accf_changes,
                         engines=['GTF', 'GTF2035', 'GTF2035_wi'],
                         saf_levels=[0], save_fig=True, df_name='contrail_no_accf_changes')

plot_climate_impact_pies(contrail_no_accf_changes,
                         engines=['GTF2035'],
                         saf_levels=[20], save_fig=True, df_name='contrail_no_accf_changes')

plot_climate_impact_pies(contrail_no_accf_changes,
                         engines=['GTF2035'],
                         saf_levels=[100], save_fig=True, df_name='contrail_no_accf_changes')

plot_climate_impact_pies(contrail_no_accf_changes,
                         engines=['GTF2035_wi'],
                         saf_levels=[20], save_fig=True, df_name='contrail_no_accf_changes')

plot_climate_impact_pies(contrail_no_accf_changes,
                         engines=['GTF2035_wi'],
                         saf_levels=[100], save_fig=True, df_name='contrail_no_accf_changes')

# """diurnal"""
# plot_climate_impact_pies(contrail_no_changes,
#                          engines=['GTF', 'GTF2035', 'GTF2035_wi'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_no_changes', daytime_filter=True)
# plot_climate_impact_pies(contrail_no_changes,
#                          engines=['GTF', 'GTF2035', 'GTF2035_wi'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_no_changes', nighttime_filter=True)
#
# """seasonal"""
# plot_climate_impact_pies(contrail_no_changes,
#                          engines=['GTF'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_no_changes', season_filter='2023-02-06')
#
# plot_climate_impact_pies(contrail_no_changes,
#                          engines=['GTF'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_no_changes', season_filter='2023-05-05')
#
# plot_climate_impact_pies(contrail_no_changes,
#                          engines=['GTF'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_no_changes', season_filter='2023-08-06')
#
# plot_climate_impact_pies(contrail_no_changes,
#                          engines=['GTF'],
#                          saf_levels=[0], save_fig=True, df_name='contrail_no_changes', season_filter='2023-11-06')
# # plt.show()
#
# # Total counts
# print('Total flights:', contrail_yes_changes.shape[0] + contrail_no_changes.shape[0])
# print('Contrail-forming flights:', contrail_yes_changes.shape[0])
# print('Non-contrail-forming flights:', contrail_no_changes.shape[0])
#
# # Diurnal breakdown
# print('Daytime (contrail-forming):', contrail_yes_changes[contrail_yes_changes['diurnal'] == 'daytime'].shape[0])
# print('Nighttime (contrail-forming):', contrail_yes_changes[contrail_yes_changes['diurnal'] == 'nighttime'].shape[0])
# print('Daytime (non-contrail-forming):', contrail_no_changes[contrail_no_changes['diurnal'] == 'daytime'].shape[0])
# print('Nighttime (non-contrail-forming):', contrail_no_changes[contrail_no_changes['diurnal'] == 'nighttime'].shape[0])
#
# # Seasonal mapping
# season_mapping = {
#     '2023-02-06': 'Winter',
#     '2023-05-05': 'Spring',
#     '2023-08-06': 'Summer',
#     '2023-11-06': 'Autumn'
# }
#
# # Replace dates with season names
# contrail_yes_changes['season_label'] = contrail_yes_changes['season'].map(season_mapping)
# contrail_no_changes['season_label'] = contrail_no_changes['season'].map(season_mapping)
#
# # Seasonal breakdown (contrail-forming)
# for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
#     count = contrail_yes_changes[contrail_yes_changes['season_label'] == season].shape[0]
#     print(f'{season} (contrail-forming):', count)
#
# # Seasonal breakdown (non-contrail-forming)
# for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
#     count = contrail_no_changes[contrail_no_changes['season_label'] == season].shape[0]
#     print(f'{season} (non-contrail-forming):', count)