import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
# Load your GSP CSV data (adjust path)
df_gsp = pd.read_csv('main_results_figures/results/malaga/malaga/emissions/GTF_SAF_0_A20N_full_WAR_0_0_0.csv')

# Engine model (adjust as needed)
engine_model = 'GTF'

# ICAO LTO thrust settings
thrust_setting_icao = [0.07, 0.3, 0.85, 1]

# Engine-specific nvPM emissions indices (SLS)
if engine_model in ['GTF', 'GTF2035', 'GTF2035_wi']:
    EI_mass_icao_sl = [7.8, 0.6, 26.3, 36.3]
    EI_number_icao_sl = [5.78e15, 3.85e14, 1.60e15, 1.45e15]
elif engine_model == 'GTF1990':
    EI_mass_icao_sl = [30.6, 58.2, 92.3, 102]
    EI_number_icao_sl = [4.43e15, 9.03e15, 2.53e15, 1.62e15]
elif engine_model == 'GTF2000':
    EI_mass_icao_sl = [5.5, 3.13, 50.8, 64]
    EI_number_icao_sl = [7.98e14, 4.85e14, 1.39e15, 1.02e15]
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}.")

# Interpolate SLS values based on thrust setting per row
df_gsp['ei_nvpm_mass_sls'] = df_gsp['thrust_setting_meem'].apply(
    lambda thrust: np.interp(thrust, thrust_setting_icao, EI_mass_icao_sl)
)
df_gsp['ei_nvpm_number_sls'] = df_gsp['thrust_setting_meem'].apply(
    lambda thrust: np.interp(thrust, thrust_setting_icao, EI_number_icao_sl)
)

# Plot A: EI_nvPM_mass (PyContrails vs SLS)
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_py'], label='Pycontrails', linestyle='-')
plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_sls'], label='SLS - MEEM', linestyle='-', color='tab:red')
plt.title('EI_nvPM_mass')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvPM_mass (mg / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig('main_results_figures/figures/malaga/malaga/emissions/t4t2_vs_p3t3meemslsmass.png', format='png')
# plt.show()

# Plot B: EI_nvPM_number (PyContrails vs SLS)
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_py'], label='Pycontrails', linestyle='-')
plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_sls'], label='SLS - MEEM', linestyle='-', color='tab:red')
plt.title('EI_nvPM_number')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvPM_number (# / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig('main_results_figures/figures/malaga/malaga/emissions/t4t2_vs_p3t3meemslsnumber.png', format='png')
# plt.show()


"""other way around"""
# Load your GSP CSV data (adjust path if needed)
df_gsp = pd.read_csv('main_results_figures/results/malaga/malaga/emissions/GTF_SAF_0_A20N_full_WAR_0_0_0.csv')

# Engine model and SAF fraction (optional, adjust as needed)
engine_model = 'GTF'  # or 'GTF2035', 'GTF1990', etc.
SAF = 0  # Adjust if needed

# Load the correct interpolation functions for SLS PT3 and FAR based on engine model
if engine_model in ('GTF', 'GTF2035', 'GTF2035_wi'):
    with open('p3t3_graphs_sls.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
elif engine_model in ('GTF1990', 'GTF2000'):
    with open('p3t3_graphs_sls_1990_2000.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}.")

interp_func_far = loaded_functions['interp_func_far']
interp_func_pt3 = loaded_functions['interp_func_pt3']

# ICAO LTO thrust settings
thrust_setting_icao = [0.07, 0.3, 0.85, 1]

# Engine-specific nvPM emissions indices (SLS)
if engine_model in ['GTF', 'GTF2035', 'GTF2035_wi']:
    EI_mass_icao_sl = [7.8, 0.6, 26.3, 36.3]
    EI_number_icao_sl = [5.78e15, 3.85e14, 1.60e15, 1.45e15]
elif engine_model == 'GTF1990':
    EI_mass_icao_sl = [30.6, 58.2, 92.3, 102]
    EI_number_icao_sl = [4.43e15, 9.03e15, 2.53e15, 1.62e15]
elif engine_model == 'GTF2000':
    EI_mass_icao_sl = [5.5, 3.13, 50.8, 64]
    EI_number_icao_sl = [7.98e14, 4.85e14, 1.39e15, 1.02e15]
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}.")

# P3T3 Correction for SAF (based on thrust setting) – dummy function
def saf_correction_number(saf, thrust_setting):
    # Add your proper SAF correction logic if needed
    return 0


# Function to compute altitude-corrected PyContrails and MEEM values
def compute_altitude_correction(row):
    TT3 = row['TT3']
    PT3 = row['PT3']
    FAR = row['FAR']
    thrust_setting = row['thrust_setting_meem']
    saf = row.get('SAF', 0)

    # SLS interpolated FAR and PT3 from TT3
    far_sls = interp_func_far(TT3)
    pt3_sls = interp_func_pt3(TT3)

    # ICAO interpolated mass and number SLS values from thrust setting
    ei_nvpm_mass_sls = np.interp(thrust_setting, thrust_setting_icao, EI_mass_icao_sl)
    ei_nvpm_number_sls = np.interp(thrust_setting, thrust_setting_icao, EI_number_icao_sl)

    # PyContrails values (before correction)
    ei_nvpm_mass_py = row['ei_nvpm_mass_py']
    ei_nvpm_number_py = row['ei_nvpm_number_py']

    # Altitude correction (P3T3) for PyContrails outputs
    ei_nvpm_mass_py_corrected = ei_nvpm_mass_py * (PT3 / pt3_sls) ** 1.35 * (FAR / far_sls) ** 2.5
    ei_nvpm_number_py_corrected = ei_nvpm_mass_py_corrected * (ei_nvpm_number_py / ei_nvpm_mass_py)

    # MEEM P3T3 mass correction
    ei_nvpm_mass_p3t3 = ei_nvpm_mass_sls * (PT3 / pt3_sls) ** 1.35 * (FAR / far_sls) ** 2.5

    # MEEM P3T3 number correction (scaled from mass)
    ei_nvpm_number_p3t3 = ei_nvpm_mass_p3t3 * (ei_nvpm_number_sls / ei_nvpm_mass_sls)

    if saf != 0:
        del_saf = saf_correction_number(saf, thrust_setting)
        ei_nvpm_number_p3t3 *= 1.0 + del_saf / 100.0

    return pd.Series([
        ei_nvpm_mass_py_corrected, ei_nvpm_number_py_corrected,
        ei_nvpm_mass_p3t3, ei_nvpm_number_p3t3
    ])


# Apply corrections
df_gsp[['ei_nvpm_mass_py_corrected', 'ei_nvpm_number_py_corrected',
        'ei_nvpm_mass_p3t3_meem', 'ei_nvpm_number_p3t3_meem']] = df_gsp.apply(compute_altitude_correction, axis=1)

# Plot A: EI_nvPM_mass comparison
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_py_corrected'], label='Pycontrails (altitude corrected)', linestyle='-')
plt.plot(df_gsp.index, df_gsp['ei_nvpm_mass_p3t3_meem'], label='P3T3 - MEEM', linestyle='-', color='tab:red')
plt.title('EI_nvPM_mass (Altitude Corrected)')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvPM_mass (mg / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig('main_results_figures/figures/malaga/malaga/emissions/t4t2_vs_p3t3meemmass.png', format='png')

# Plot B: EI_nvPM_number comparison
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_py_corrected'], label='Pycontrails (altitude corrected)', linestyle='-')
plt.plot(df_gsp.index, df_gsp['ei_nvpm_number_p3t3_meem'], label='P3T3 - MEEM', linestyle='-', color='tab:red')
plt.title('EI_nvPM_number (Altitude Corrected)')
plt.xlabel('Time in minutes')
plt.ylabel('EI_nvPM_number (# / kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig('main_results_figures/figures/malaga/malaga/emissions/t4t2_vs_p3t3meemnumber.png', format='png')

plt.show()


