import pandas as pd

# Load the results CSV
results_df = pd.read_csv('results_main_simulations.csv')

# Filter out special cases
excluded_categories = ['special cases']
filtered_df = results_df[~results_df['contrail_category'].isin(excluded_categories)]

# Split into no contrail, warming, cooling
no_contrail_df = filtered_df[filtered_df['contrail_category'] == 'no contrail']
warming_df = filtered_df[filtered_df['contrail_category'] == 'warming']
cooling_df = filtered_df[filtered_df['contrail_category'] == 'cooling']

# Helper function to compute fraction per flight
def compute_fractions(df):
    df = df.copy()
    df['co2_impact_avg'] = (df['co2_impact_cons_sum'] + df['co2_impact_opti_sum']) / 2

    df['climate_total_avg_sum'] = (df['climate_total_cons_sum'] + df['climate_total_opti_sum']) / 2

    df['fraction_h2o'] = 100 * df['h2o_impact_sum'] / df['climate_total_avg_sum']
    df['fraction_co2'] = 100 * df['co2_impact_avg'] / df['climate_total_avg_sum']
    df['fraction_nox'] = 100 * df['nox_impact_sum'] / df['climate_total_avg_sum']

    if 'contrail_atr20_cocip_sum' in df.columns:
        df['fraction_contrails'] = 100 * df['contrail_atr20_cocip_sum'] / df['climate_total_avg_sum']
    else:
        df['fraction_contrails'] = 0.0

    return df

# Apply fraction calculation to each category
no_contrail_df = compute_fractions(no_contrail_df)
warming_df = compute_fractions(warming_df)
cooling_df = compute_fractions(cooling_df)

# Aggregate averages by engine, SAF, WAR
fraction_columns = ['fraction_h2o', 'fraction_co2', 'fraction_nox', 'fraction_contrails']

def average_fractions(df):
    return df.groupby(['engine', 'saf_level', 'water_injection'])[fraction_columns].mean().reset_index().round(1)

no_contrail_fractions = average_fractions(no_contrail_df)
warming_fractions = average_fractions(warming_df)
cooling_fractions = average_fractions(cooling_df)

# Save outputs
no_contrail_fractions.to_csv('results_report/climate/fractions_no_contrail.csv', index=False)
warming_fractions.to_csv('results_report/climate/fractions_warming.csv', index=False)
cooling_fractions.to_csv('results_report/climate/fractions_cooling.csv', index=False)

print("Saved climate impact fractions for no contrail, warming, and cooling cases successfully!")
