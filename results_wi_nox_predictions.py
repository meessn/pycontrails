import pandas as pd
import matplotlib.pyplot as plt
from emission_index import p3t3_nox_xue
import pickle
with open('p3t3_graphs_sls.pkl', 'rb') as f:
    loaded_functions = pickle.load(f)

interp_func_far = loaded_functions['interp_func_far']
interp_func_pt3 = loaded_functions['interp_func_pt3']

# Parameters
flight = 'malaga' # Replace with your flight identifier
engine = 'GTF2035'  # Replace with your engine model

file_name = f'results/{flight}/{flight}_model_{engine}_SAF_0_aircraft_A20N_full_WAR_5_5_5.csv'

df_gsp = pd.read_csv(file_name)

# Plot A: EI_NOx
plt.figure(figsize=(10, 6))
plt.plot(df_gsp.index, df_gsp['EI_nox_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nox_p3t3'], label='P3T3', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nox_p3t3_xue'], label='P3T3', linestyle='-', marker='o', markersize=2.5)
# plt.plot(df_gsp.index, df_gsp['EI_nox_boer'], label='Boer', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nox_kaiser'], label='Kaiser', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['EI_nox_kypriandis'], label='Kypriandis', linestyle='-', marker='o', markersize=2.5)

plt.title('EI_NOx predictions Steam Injection 5% during entire flight')
plt.xlabel('Time in minutes')
plt.ylabel('EI_NOx (g/ kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'figures/{flight}/water_injection/ei_nox_correlations.png', format='png')