import pandas as pd
import matplotlib.pyplot as plt
from emission_index import p3t3_nox_xue
import pickle
# with open('p3t3_graphs_sls.pkl', 'rb') as f:
#     loaded_functions = pickle.load(f)
#
# interp_func_far = loaded_functions['interp_func_far']
# interp_func_pt3 = loaded_functions['interp_func_pt3']

# Parameters
flight = 'malaga' # Replace with your flight identifier
engine = 'GTF2035'  # Replace with your engine model

file_name = f'../main_results_figures/results/malaga/malaga/emissions/{engine}_SAF_0_A20N_full_WAR_5_5_5.csv'

df_gsp = pd.read_csv(file_name)

# Plot A: EI_NOx
plt.figure(figsize=(10, 6))
# plt.plot(df_gsp.index, df_gsp['EI_nox_py'], label='Pycontrails', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3'], label='P3T3 Kaiser Correction', linestyle='-')
plt.plot(df_gsp.index, df_gsp['ei_nox_p3t3_xue'], label='P3T3 Xue Correction', linestyle='-')
# plt.plot(df_gsp.index, df_gsp['EI_nox_boer'], label='Boer', linestyle='-', marker='o', markersize=2.5)
plt.plot(df_gsp.index, df_gsp['ei_nox_kaiser'], label='Kaiser', linestyle='-')
plt.plot(df_gsp.index, df_gsp['ei_nox_kypriandis'], label='Kyprianidis', linestyle='-')

plt.title(f'$EI_{{\\mathrm{{NOx}}}}$ predictions Steam Injection 5% during entire flight, GTF2035')
plt.xlabel('Time (Minutes)')
plt.ylabel(f'$EI_{{\\mathrm{{NOx}}}}$ (g/ kg Fuel)')
plt.legend()
plt.grid(True)
plt.savefig(f'../results_report/performance_emissions_chapter/ei_nox_correlations_5.png', format='png')