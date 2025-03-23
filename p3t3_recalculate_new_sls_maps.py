import os
import pandas as pd
import warnings
import pickle
from emission_index import thrust_setting, p3t3_nox, p3t3_nvpm_meem, p3t3_nvpm_meem_mass
from scipy.interpolate import interp1d
# Path to results directory
base_dir = 'main_results_figures/results/malaga/malaga'

# Loop through flight trajectories
for root, dirs, files in os.walk(base_dir):
    if 'emissions' in root:
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                print(f"Processing: {filepath}")

                # Load CSV
                df_gsp = pd.read_csv(filepath)

                # Extract engine model from filename
                filename_parts = file.split('_')
                engine_model = filename_parts[0]

                if engine_model in ('GTF'):
                    with open('p3t3_graphs_sls_gtf_final.pkl', 'rb') as f:
                        loaded_functions = pickle.load(f)
                elif engine_model in ('GTF2035', 'GTF2035_wi'):
                    with open('p3t3_graphs_sls_gtf2035_final.pkl', 'rb') as f:
                        loaded_functions = pickle.load(f)
                elif engine_model in ('GTF1990', 'GTF2000'):
                    with open('p3t3_graphs_sls_1990_2000_final.pkl', 'rb') as f:
                        loaded_functions = pickle.load(f)
                else:
                    raise ValueError(f"Unsupported engine_model: {engine_model}.")

                interp_func_far = loaded_functions['interp_func_far']
                interp_func_pt3 = loaded_functions['interp_func_pt3']
                interp_func_fgr = loaded_functions['interp_func_fgr']
                # Get interpolation function bounds
                x_min, x_max = interp_func_far.x[0], interp_func_far.x[-1]

                # Get min and max TT3 from df_gsp
                tt3_min = df_gsp['TT3'].min()
                tt3_max = df_gsp['TT3'].max()

                # Identify out-of-bounds values
                out_of_bounds_mask = (df_gsp['TT3'] < x_min) | (df_gsp['TT3'] > x_max)
                out_of_bounds_values = df_gsp.loc[out_of_bounds_mask, 'TT3']

                if not out_of_bounds_values.empty:
                    warnings.warn(f"TT3 values in df_gsp are outside the interpolation range ({x_min}, {x_max}). "
                                  f"Min TT3: {tt3_min}, Max TT3: {tt3_max}. Extreme values are clipped.")

                    print(f"Number of TT3 values out of bounds: {out_of_bounds_values.shape[0]}")
                    print("Out-of-bounds TT3 values:", out_of_bounds_values.tolist())

                    # Clamp values to stay within bounds
                    df_gsp['TT3'] = df_gsp['TT3'].clip(lower=x_min, upper=x_max)

                df_gsp['thrust_setting_meem'] = df_gsp.apply(
                    lambda row: thrust_setting(
                        engine_model,
                        row['TT3'],
                        interp_func_pt3,
                        interp_func_fgr
                    ),
                    axis=1
                )

                """NOx p3t3"""
                df_gsp['ei_nox_p3t3'] = df_gsp.apply(
                    lambda row: p3t3_nox(
                        row['PT3'],
                        row['TT3'],
                        interp_func_far,
                        interp_func_pt3,
                        row['specific_humidity'],
                        row['WAR_gsp'],
                        engine_model
                    ),
                    axis=1
                )
                #

                """P3T3 _MEEM"""
                df_gsp['ei_nvpm_number_p3t3_meem'] = df_gsp.apply(
                    lambda row: p3t3_nvpm_meem(
                        row['PT3'],
                        row['TT3'],
                        row['FAR'],
                        interp_func_far,
                        interp_func_pt3,
                        row['SAF'],
                        row['thrust_setting_meem'],
                        engine_model
                    ),
                    axis=1
                )

                df_gsp['ei_nvpm_mass_p3t3_meem'] = df_gsp.apply(
                    lambda row: p3t3_nvpm_meem_mass(
                        row['PT3'],
                        row['TT3'],
                        row['FAR'],
                        interp_func_far,
                        interp_func_pt3,
                        row['SAF'],
                        row['thrust_setting_meem'],
                        engine_model
                    ),
                    axis=1
                )

                df_gsp.to_csv(filepath, index=False)
