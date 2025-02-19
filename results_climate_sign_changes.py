import pandas as pd
import numpy as np

# Load your main results dataframe
results_df = pd.read_csv('results_main_simulations.csv')


# Function to check sign differences against a baseline engine
def check_sign_changes(baseline_engine, baseline_saf=0):
    # Exclude GTF1990 and GTF2000 when using GTF as baseline
    if baseline_engine == 'GTF':
        comparison_df = results_df[~results_df['engine'].isin(['GTF1990', 'GTF2000'])]
    else:
        comparison_df = results_df

    # Filter the baseline engine data (same trajectory, season, diurnal)
    baseline_df = comparison_df[(comparison_df['engine'] == baseline_engine) & (comparison_df['saf_level'] == baseline_saf)]

    merged_df = comparison_df.merge(baseline_df, on=['trajectory', 'season', 'diurnal'], suffixes=('', '_baseline'))

    # Compare contrail signs
    merged_df['cocip_sign_change'] = (
        np.sign(merged_df['contrail_atr20_cocip_sum']) != np.sign(merged_df['contrail_atr20_cocip_sum_baseline'])
    ) & (merged_df['contrail_atr20_cocip_sum'] != 0) & (merged_df['contrail_atr20_cocip_sum_baseline'] != 0)

    # Extract rows with sign change
    sign_changes_df = merged_df[merged_df['cocip_sign_change']]

    # Optional: Display relevant information
    sign_changes_relevant = sign_changes_df[[
        'trajectory', 'season', 'diurnal', 'engine', 'saf_level', 'water_injection',
        'contrail_atr20_cocip_sum', 'contrail_atr20_cocip_sum_baseline'
    ]]

    return sign_changes_relevant


# Check sign changes with GTF1990 as baseline
gtf1990_sign_changes = check_sign_changes('GTF1990')

# Check sign changes with GTF as baseline
gtf_sign_changes = check_sign_changes('GTF')

# Display results
print(f"Sign changes compared to GTF1990 baseline:\n{gtf1990_sign_changes}")
print(f"\nSign changes compared to GTF baseline:\n{gtf_sign_changes}")

# Optionally save the results
gtf1990_sign_changes.to_csv('contrail_sign_changes_vs_gtf1990.csv', index=False)
gtf_sign_changes.to_csv('contrail_sign_changes_vs_gtf.csv', index=False)

print(f"\nNumber of sign changes vs GTF1990: {len(gtf1990_sign_changes)}")
print(f"Number of sign changes vs GTF: {len(gtf_sign_changes)}")
