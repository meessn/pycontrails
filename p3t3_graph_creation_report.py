import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from emission_index import p3t3_nox, p3t3_nvpm_meem, p3t3_nvpm_meem_mass, thrust_setting
import pickle
def ei_nox_sls(tt3, pt3, far):
    return 1.4094 * pt3** 0.1703 * np.exp(0.0011 * tt3) * 12.2308 ** (16.4302 * far)


"""PW1127G"""
# Read the file
file_path = 'P3T3_SLS_GRAPHS_PW1127G_corr.csv'
data = pd.read_csv(file_path, delimiter=';', decimal=',')
data = data.drop(index=0).reset_index(drop=True)
data['TT3'] = pd.to_numeric(data['TT3'].str.replace(',', '.'))
data = data[data['TT3'] <= 900]
# Convert columns to numeric after replacing ',' with '.'
# data['TT3'] = pd.to_numeric(data['TT3'].str.replace(',', '.'))
data['FAR'] = pd.to_numeric(data['FAR'].str.replace(',', '.'))
data['PT3'] = pd.to_numeric(data['PT3'].str.replace(',', '.'))

# Extract the 'TT3', 'FAR', and 'PT3' columns
TT3 = data['TT3'].values
FAR = data['FAR'].values
PT3 = data['PT3'].values
# Apply the ei_nox_sls function to each row
data['EI_NOx_SLS'] = data.apply(lambda row: ei_nox_sls(row['TT3'], row['PT3'], row['FAR']), axis=1)
NOX = data['EI_NOx_SLS'].values

# Create interpolation functions
interp_func_FAR = interp1d(TT3, FAR, kind='linear')
interp_func_PT3 = interp1d(TT3, PT3, kind='linear')
interp_func_NOX = interp1d(TT3, NOX, kind='linear')
# Define points for red dots
highlight_tt3 = [835.38, 805.03, 643.29, 485.36]
# 835,38
# 805,03
# 643,29
# 485,36

# Interpolated values for red dots
highlight_FAR = interp_func_FAR(highlight_tt3)
highlight_PT3 = interp_func_PT3(highlight_tt3)
highlight_NOX = interp_func_NOX(highlight_tt3)
print(highlight_NOX)

"""GTF2035"""
file_path_1 = 'P3T3_SLS_GRAPHS_GTF2035.csv'
data_2035 = pd.read_csv(file_path_1, delimiter=';', decimal=',')
data_2035 = data_2035.drop(index=0).reset_index(drop=True)
data_2035['TT3'] = pd.to_numeric(data_2035['TT3'].str.replace(',', '.'))
data_2035 = data_2035[data_2035['TT3'] <= 900]
# Convert columns to numeric after replacing ',' with '.'
# data['TT3'] = pd.to_numeric(data['TT3'].str.replace(',', '.'))
data_2035['FAR'] = pd.to_numeric(data_2035['FAR'].str.replace(',', '.'))
data_2035['PT3'] = pd.to_numeric(data_2035['PT3'].str.replace(',', '.'))

# Extract the 'TT3', 'FAR', and 'PT3' columns
TT3_2035 = data_2035['TT3'].values
FAR_2035 = data_2035['FAR'].values
PT3_2035 = data_2035['PT3'].values
# Apply the ei_nox_sls function to each row
data_2035['EI_NOx_SLS'] = data_2035.apply(lambda row: ei_nox_sls(row['TT3'], row['PT3'], row['FAR']), axis=1)
NOX_2035 = data_2035['EI_NOx_SLS'].values

# Create interpolation functions
interp_func_FAR_2035 = interp1d(TT3_2035, FAR_2035, kind='linear')
interp_func_PT3_2035 = interp1d(TT3_2035, PT3_2035, kind='linear')
interp_func_NOX_2035 = interp1d(TT3_2035, NOX_2035, kind='linear')
# Define points for red dots
highlight_tt3_2035 = [875.67, 843.57, 672.96, 498.62]
# 835,38
# 805,03
# 643,29
# 485,36

# Interpolated values for red dots
highlight_FAR_2035 = interp_func_FAR_2035(highlight_tt3_2035)
highlight_PT3_2035 = interp_func_PT3_2035(highlight_tt3_2035)
highlight_NOX_2035 = interp_func_NOX_2035(highlight_tt3_2035)
print(highlight_NOX_2035)


data_1124 = {
    "LTOstage": ["T/O", "CLIMB", "APPR", "IDLE"],
    "TT3": [810.26, 780.38, 629.21, 466.85],
    "PT3": [29.05609, 25.08446, 9.43599, 3.50655],
    "FAR": [2.52E-02, 2.36E-02, 2.00E-02, 1.48E-02],
    "TT4": [1636.67, 1565.47, 1331.05, 1023.14]
}

df_1124 = pd.DataFrame(data_1124)

data_1129 = {
    "LTOstage": ["T/O", "CLIMB", "APPR", "IDLE"],
    "TT3": [852.04, 819.51, 656.8, 488.04],
    "PT3": [34.70164, 29.84731, 10.88136, 3.88328],
    "FAR": [2.86E-02, 2.68E-02, 2.25E-02, 1.65E-02],
    "TT4": [1767.34, 1688.55, 1429.96, 1096.17]
}

df_1129 = pd.DataFrame(data_1129)

# Data for 1133
data_1133 = {
    "LTOstage": ["T/O", "CLIMB", "APPR", "IDLE"],
    "TT3": [877.87, 843.15, 662.78, 492.97],
    "PT3": [38.97456, 33.48126, 12.63606, 4.10652],
    "FAR": [3.14E-02, 2.88E-02, 2.03E-02, 1.60E-02],
    "TT4": [1862.62, 1764.29, 1365.43, 1086.04]
}

df_1133 = pd.DataFrame(data_1133)

df_1124['EI_NOx_SLS'] = df_1124.apply(lambda row: ei_nox_sls(row['TT3'], row['PT3'], row['FAR']), axis=1)
df_1129['EI_NOx_SLS'] = df_1129.apply(lambda row: ei_nox_sls(row['TT3'], row['PT3'], row['FAR']), axis=1)
df_1133['EI_NOx_SLS'] = df_1133.apply(lambda row: ei_nox_sls(row['TT3'], row['PT3'], row['FAR']), axis=1)

# Plot 1: PT3 as a function of TT3
plt.figure(figsize=(10, 6))
plt.plot(TT3, PT3, label='GTF, OPR t/o: 31.7')
plt.scatter(highlight_tt3, highlight_PT3, color='tab:blue', label='GTF LTO stages', zorder=5)
plt.plot(TT3_2035, PT3_2035, label='GTF2035, OPR t/o: 39.93')
plt.scatter(highlight_tt3_2035, highlight_PT3_2035, color='tab:orange', label='GTF2035 LTO stages', zorder=5)
plt.scatter(df_1124['TT3'], df_1124['PT3'], label='PW1124G, OPR t/o: 28.8', color='tab:red')
plt.scatter(df_1129['TT3'], df_1129['PT3'], label='PW1129G, OPR t/o: 34.0', color='tab:green')
plt.scatter(df_1133['TT3'], df_1133['PT3'], label='PW1133G, OPR t/o: 38.2', color='tab:purple')
plt.title(r'$P_{t3,GR}$ versus $T_{t3}$ Map')
plt.xlabel(r'$T_{t3}$ (k)')
plt.ylabel(r'$P_{t3,GR}$ (bar)')
plt.legend()
plt.grid(True)
# plt.xlim(0, 900)
plt.savefig('results_report/p3t3_graph/pt3_tt3_sls.png', format='png')

# Plot 2: FAR as a function of TT3
plt.figure(figsize=(10, 6))
plt.plot(TT3, FAR, label='GTF, OPR t/o: 31.7')
plt.scatter(highlight_tt3, highlight_FAR, color='tab:blue', label='GTF LTO stages', zorder=5)
plt.plot(TT3_2035, FAR_2035, label='GTF2035, OPR t/o: 39.93')
plt.scatter(highlight_tt3_2035, highlight_FAR_2035, color='tab:orange', label='GTF2035 LTO stages', zorder=5)
plt.scatter(df_1124['TT3'], df_1124['FAR'], label='PW1124G, OPR t/o: 28.8', color='tab:red')
plt.scatter(df_1129['TT3'], df_1129['FAR'], label='PW1129G, OPR t/o: 34.0', color='tab:green')
plt.scatter(df_1133['TT3'], df_1133['FAR'], label='PW1133G, OPR t/o: 38.2', color='tab:purple')
plt.title(r'$FAR_{GR}$ versus $T_{t3}$ Map')
plt.xlabel(r'$T_{t3}$ (k)')
plt.ylabel(r'$FAR_{GR}$ (-)')
plt.legend()
plt.grid(True)
# plt.xlim(0, 900)
plt.savefig('results_report/p3t3_graph/far_tt3_sls.png', format='png')

# Plot 3: ei nox as a function of TT3
plt.figure(figsize=(10, 6))
plt.plot(TT3, NOX, label='GTF, OPR t/o: 31.7')
plt.scatter(highlight_tt3, highlight_NOX, color='tab:blue', label='GTF LTO stages', zorder=5)
plt.plot(TT3_2035, NOX_2035, label='GTF2035, OPR t/o: 39.93')
plt.scatter(highlight_tt3_2035, highlight_NOX_2035, color='tab:orange', label='GTF LTO stages', zorder=5)
plt.scatter(df_1124['TT3'], df_1124['EI_NOx_SLS'], label='PW1124G, OPR t/o: 28.8', color='tab:red')
plt.scatter(df_1129['TT3'], df_1129['EI_NOx_SLS'], label='PW1129G, OPR t/o: 34.0', color='tab:green')
plt.scatter(df_1133['TT3'], df_1133['EI_NOx_SLS'], label='PW1133G, OPR t/o: 38.2', color='tab:purple')
plt.title(r'$EI_{NOx,sls}$ versus $T_{t3}$ Map')
plt.xlabel(r'$T_{t3}$ (k)')
plt.ylabel(r'$EI_{NOx,GR}$ (g / kg Fuel)')
plt.legend()
plt.grid(True)
# plt.xlim(0, 900)
plt.savefig('results_report/p3t3_graph/nox_tt3_sls.png', format='png')

file_path_3 = f'main_results_figures/results/malaga/malaga/emissions/GTF2035_SAF_0_A20N_full_WAR_0_0_0.csv'
df = pd.read_csv(file_path_3)


engine_model = 'GTF2035'
if engine_model in ('GTF'):
    with open('p3t3_graphs_sls_gtf_corr.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
elif engine_model in ('GTF2035', 'GTF2035_wi'):
    print('yes')
    with open('p3t3_graphs_sls_gtf_2035.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
elif engine_model in ('GTF1990', 'GTF2000'):
    with open('p3t3_graphs_sls_1990_2000.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}.")

interp_func_far = loaded_functions['interp_func_far']
interp_func_pt3 = loaded_functions['interp_func_pt3']

df['thrust_setting_meem_2035_gr'] = df.apply(
        lambda row: thrust_setting(
            engine_model,
            row['TT3'],
            interp_func_pt3
        ),
        axis=1
    )


"""NOx p3t3"""
df['ei_nox_p3t3_2035_gr'] = df.apply(
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
df['ei_nvpm_number_p3t3_meem_2035_gr'] = df.apply(
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

df['ei_nvpm_mass_p3t3_meem_2035_gr'] = df.apply(
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

file_path_gtf = f'main_results_figures/results/malaga/malaga/emissions/GTF_SAF_0_A20N_full_WAR_0_0_0.csv'
df_gtf = pd.read_csv(file_path_gtf)

# Plot EI NOx
plt.figure(figsize=(10, 6))
plt.plot(df_gtf.index, df_gtf['ei_nox_p3t3'], label='GTF')
# plt.plot(df.index, df['ei_nox_p3t3'], label='GTF2035 (GTF P3T3 SLS Maps)')
plt.plot(df.index, df['ei_nox_p3t3_2035_gr'], label='GTF2035 (GTF2035 P3T3 SLS Maps)')
plt.xlabel("Time (minutes)")
plt.ylabel(f"$EI_{{\\mathrm{{NOx}}}}$ (g / kg fuel)")
plt.title(f"$EI_{{\\mathrm{{NOx}}}}$ for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig(f'../results_report/performance_emissions_chapter/EI_nox_engines_gtf_gtf2035.png', format='png')

# Plot EI nvPM Number
plt.figure(figsize=(10, 6))
plt.plot(df_gtf.index, df_gtf['ei_nvpm_number_p3t3_meem'], label='GTF')
# plt.plot(df.index, df['ei_nvpm_number_p3t3_meem'], label='GTF2035 (GTF P3T3 SLS Maps)')
plt.plot(df.index, df['ei_nvpm_number_p3t3_meem_2035_gr'], label='GTF2035 (GTF2035 P3T3 SLS Maps)')
plt.xlabel("Time (minutes)")
plt.ylabel(f'$EI_{{\\mathrm{{nvPM,number}}}}$ (# / kg Fuel)')
plt.title(f"$EI_{{\\mathrm{{nvPM,number}}}}$ for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig(f'../results_report/performance_emissions_chapter/EI_nvpm_number_engines_gtf_gtf2035.png', format='png')

# Plot EI nvPM Mass
plt.figure(figsize=(10, 6))
plt.plot(df_gtf.index, df_gtf['ei_nvpm_mass_p3t3_meem'], label='GTF')
# plt.plot(df.index, df['ei_nvpm_mass_p3t3_meem'], label='GTF2035 (GTF P3T3 SLS Maps)')
plt.plot(df.index, df['ei_nvpm_mass_p3t3_meem_2035_gr'], label='GTF2035 (GTF2035 P3T3 SLS Maps)')
plt.xlabel("Time (minutes)")
plt.ylabel(f'$EI_{{\\mathrm{{nvPM,mass}}}}$ (mg / kg Fuel)')
plt.title(f"$EI_{{\\mathrm{{nvPM,mass}}}}$ for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig(f'../results_report/performance_emissions_chapter/EI_nvpm_mass_engines_gtf_gtf2035.png', format='png')
plt.show()