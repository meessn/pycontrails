import pandas as pd
import os
import re

# 📌 Load final corrected fuel flow data
final_corrected_fuel_flow_path = "results_df_corrected_fuel_emissions_v2.csv"
final_df = pd.read_csv(final_corrected_fuel_flow_path)

# 📌 Base directory for emissions CSVs
base_dir = "main_results_figures/results/"

# 🔍 Pattern to extract metadata from emissions filenames
pattern = r"([^/]+)/\1_(\d{4}-\d{2}-\d{2})_(daytime|nighttime)/emissions/(GTF(?:2035_wi|2035)?)_SAF_(\d+)_A20N_full_WAR_\d+_\d+_\d+\.csv"

# 📌 Process only emissions files
for root, _, files in os.walk(base_dir):
    if "emissions" not in root:  # 🚨 Skip all non-emissions folders
        continue

    for file in files:
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)

        # 🔍 Extract metadata from filename
        match = re.search(pattern, file_path.replace("\\", "/"))  # Convert to Unix-style path for regex
        if match:
            trajectory, season, diurnal, engine, saf_level = match.groups()
            saf_level = int(saf_level)  # Convert to integer
            print(f"\n📌 Processing: {engine} | SAF: {saf_level} | {trajectory} | {season} | {diurnal}")

            # 🚨 Skip GTF1990 & GTF2000
            if engine in ["GTF1990", "GTF2000"]:
                print("⛔ Skipping GTF1990 and GTF2000!")
                continue

            # 📌 Load emissions CSV
            emissions_df = pd.read_csv(file_path)

            # 📌 Filter `final_df` for matching rows
            filtered_final_df = final_df[
                (final_df['trajectory'] == trajectory) &
                (final_df['season'] == season) &
                (final_df['diurnal'] == diurnal) &
                (final_df['engine'] == engine) &
                (final_df['saf_level'] == saf_level)
            ].copy()

            # 🚨 Check if `final_df` has matching rows
            if filtered_final_df.empty:
                print("⚠️ No matching rows found in final_df!")
                continue

            # ✅ Merge based on `index` (and `time` if available)
            merge_cols = ['index']
            if 'time' in emissions_df.columns and 'time' in filtered_final_df.columns:
                merge_cols.append('time')

            merged_df = emissions_df.merge(
                filtered_final_df[['index', 'time', 'fuel_flow_corrected']],
                on=merge_cols,
                how="left"
            )

            # 🔍 Fill missing values: If `fuel_flow_corrected` is NaN, use `fuel_flow` from emissions_df
            merged_df = merged_df.assign(fuel_flow_corrected=merged_df['fuel_flow_corrected'].fillna(merged_df['fuel_flow']))

            # ✅ Overwrite the original CSV file
            merged_df.to_csv(file_path, index=False)
            print(f"✅ Updated {file_path} with `fuel_flow_corrected`!")
        else:
            print(f"🚨 Skipping non-emissions file: {file_path}")  # No more errors for climate files
