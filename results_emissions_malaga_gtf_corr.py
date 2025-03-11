import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# === Optimized Cruise Fit Parameters ===
# Break Point (x_break): 0.59865
# Slope before breakpoint (m1): 1.37738
# Intercept before breakpoint (b1): -0.10068
# Slope after breakpoint (m2): 0.27877
#
# === Optimized Climb Fit Parameters ===
# a (Quadratic term): -0.37603
# b (Linear term): 1.48060
# c (Intercept): -0.05849

# # Define the piecewise linear function with an optimized breakpoint
# def piecewise_linear(x, x_break, m1, b1, m2):
#     """
#     Piecewise linear function with optimized breakpoint.
#
#     Parameters:
#     x       - Input fuel flow data (GSP fuel flow)
#     x_break - The breakpoint where the slope changes
#     m1      - Slope of the first segment (before breakpoint)
#     b1      - Intercept of the first segment
#     m2      - Slope of the second segment (after breakpoint)
#
#     Returns:
#     y       - Output fuel flow (PyContrails fuel flow)
#     """
#     x = np.asarray(x)  # Ensure x is a NumPy array
#     return np.where(x < x_break, m1 * x + b1, m2 * (x - x_break) + (m1 * x_break + b1))
#
#
# # Define a polynomial function (second-degree)
# def poly_fit(x, a, b, c):
#     return a * x**2 + b * x + c


# Define the piecewise linear function for cruise phase
def piecewise_linear(x, x_break=0.59865, m1=1.37738, b1=-0.10068, m2=0.27877):
    return np.where(x < x_break, m1 * x + b1, m2 * (x - x_break) + (m1 * x_break + b1))

# Define the polynomial function for climb phase
def poly_fit(x, a=-0.37603, b=1.48060, c=-0.05849):
    return a * x**2 + b * x + c

# Load the dataset
file_path = "main_results_figures/results/malaga/malaga/emissions/GTF_SAF_0_A20N_full_WAR_0_0_0.csv"
df = pd.read_csv(file_path)

file_path_gtf_corr = "main_results_figures/results/malaga/malaga/emissions/GTF_corr_SAF_0_A20N_full_WAR_0_0_0.csv"
df_corr = pd.read_csv(file_path_gtf_corr)
# Ensure the necessary columns exist
if 'fuel_flow_gsp' not in df.columns or 'flight_phase' not in df.columns:
    raise ValueError("CSV file must contain 'fuel_flow_gsp' and 'flight_phase' columns.")

df['fuel_flow_2_engines_gsp'] = df['fuel_flow_gsp']*2

# Apply corrections based on flight phase
def apply_fuel_correction(row):
    if row['flight_phase'] == 'cruise':
        return piecewise_linear(row['fuel_flow_2_engines_gsp'])
    elif row['flight_phase'] == 'climb':
        return poly_fit(row['fuel_flow_2_engines_gsp'])
    elif row['flight_phase'] == 'descent':
        return row['fuel_flow_2_engines_gsp']  # No correction for descent
    else:
        return np.nan  # Handle unexpected phases

df['fuel_flow_corrected'] = df.apply(apply_fuel_correction, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(df['fuel_flow_per_engine'], label="pycontrails")
# plt.plot(df['fuel_flow_gsp'], label="GTF")
plt.plot(df['fuel_flow_corrected']/2, label="GTF Corrected")
plt.plot(df_corr['fuel_flow_gsp'], label="GTF Improved GSP")
plt.xlabel("Time (m)")
plt.ylabel("Fuel flow [kg/s]")
plt.title("Fuel flow for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig('results_report/improved_gsp_vs_corrected/fuel_flow_gsp_improved_corrected.png', format='png')

plt.figure(figsize=(10, 6))
plt.plot(df['ei_nox_p3t3'], label="GTF", color='tab:orange')
plt.plot(df_corr['ei_nox_p3t3'], label="GTF Improved GSP", color='tab:green')
plt.xlabel("Time (m)")
plt.ylabel("EI_NOx")
plt.title("P3T3 EI_NOx for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig('results_report/improved_gsp_vs_corrected/ei_nox_gsp_improved_corrected.png', format='png')

plt.figure(figsize=(10, 6))
plt.plot(df['ei_nvpm_number_p3t3_meem'], label="GTF", color='tab:orange')
plt.plot(df_corr['ei_nvpm_number_p3t3_meem'], label="GTF Improved GSP", color='tab:green')
plt.xlabel("Time (m)")
plt.ylabel("EI_nvpm_number")
plt.title("P3T3-MEEM EI_nvPM_number for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig('results_report/improved_gsp_vs_corrected/ei_nvpm_gsp_improved_corrected.png', format='png')

plt.figure(figsize=(10, 6))
plt.plot(df['ei_nox_p3t3']*(df['fuel_flow_corrected']/2), label="GTF Corrected", color='tab:orange')
plt.plot(df_corr['ei_nox_p3t3']*df_corr['fuel_flow_gsp'], label="GTF Improved GSP", color='tab:green')
plt.xlabel("Time (m)")
plt.ylabel("NOx")
plt.title("P3T3 NOx for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig('results_report/improved_gsp_vs_corrected/nox_gsp_improved_corrected.png', format='png')

plt.figure(figsize=(10, 6))
plt.plot(df['ei_nvpm_number_p3t3_meem']*(df['fuel_flow_corrected']/2), label="GTF Corrected", color='tab:orange')
plt.plot(df_corr['ei_nvpm_number_p3t3_meem']*df_corr['fuel_flow_gsp'], label="GTF Improved GSP", color='tab:green')
plt.xlabel("Time (m)")
plt.ylabel("nvPM number")
plt.title("P3T3-MEEM nvPM number for different engines")
plt.legend(title="Engine")
plt.grid()
# plt.savefig('results_report/improved_gsp_vs_corrected/nvpm_gsp_improved_corrected.png', format='png')
plt.show()

# output_file = "main_results_figures/results/malaga/malaga/emissions/GTF_curve_corr_SAF_0_A20N_full_WAR_0_0_0.csv"
# df.to_csv(output_file, index=False)

# Compute total sums for each variable in both datasets
total_fuel_flow_corrected = df['fuel_flow_corrected'].sum() / 2  # Per engine
total_fuel_flow_gsp_improved = df_corr['fuel_flow_gsp'].sum()

total_ei_nox_corrected = df['ei_nox_p3t3'].sum()
total_ei_nox_gsp_improved = df_corr['ei_nox_p3t3'].sum()

total_ei_nvpm_corrected = df['ei_nvpm_number_p3t3_meem'].sum()
total_ei_nvpm_gsp_improved = df_corr['ei_nvpm_number_p3t3_meem'].sum()

total_nox_corrected = (df['ei_nox_p3t3'] * df['fuel_flow_corrected'] / 2).sum()
total_nox_gsp_improved = (df_corr['ei_nox_p3t3'] * df_corr['fuel_flow_gsp']).sum()

total_nvpm_corrected = (df['ei_nvpm_number_p3t3_meem'] * (df['fuel_flow_corrected'] / 2)).sum()
total_nvpm_gsp_improved = (df_corr['ei_nvpm_number_p3t3_meem'] * df_corr['fuel_flow_gsp']).sum()

# Compute percentage differences
percentage_differences = {
    'Fuel Flow': ((total_fuel_flow_gsp_improved - total_fuel_flow_corrected) / total_fuel_flow_corrected) * 100,
    'EI NOx': ((total_ei_nox_gsp_improved - total_ei_nox_corrected) / total_ei_nox_corrected) * 100,
    'NOx': ((total_nox_gsp_improved - total_nox_corrected) / total_nox_corrected) * 100,
    'EI nvPM': ((total_ei_nvpm_gsp_improved - total_ei_nvpm_corrected) / total_ei_nvpm_corrected) * 100,
    'nvPM': ((total_nvpm_gsp_improved - total_nvpm_corrected) / total_nvpm_corrected) * 100
}

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(percentage_differences.keys(), percentage_differences.values())
plt.xlabel("Metric")
plt.ylabel("Percentage Difference (%)")
plt.title("Percentage Difference of GTF Improved GSP Compared to GTF Corrected")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


file_path_cl = "main_results_figures/results/malaga/malaga/climate/mees/era5model/GTF_curve_corr_SAF_0_A20N_full_WAR_0_climate.csv"


file_path_gtf_corr_cl = "main_results_figures/results/malaga/malaga/climate/mees/era5model/GTF_corr_SAF_0_A20N_full_WAR_0_climate.csv"

df_climate = pd.read_csv(file_path_cl)
df_corr_climate = pd.read_csv(file_path_gtf_corr_cl)

total_co2_impact_corrected = df_climate['accf_sac_co2_impact'].sum()
total_co2_impact_gsp_improved = df_corr_climate['accf_sac_co2_impact'].sum()

total_nox_impact_corrected = df_climate['accf_sac_nox_impact'].sum()
total_nox_impact_gsp_improved = df_corr_climate['accf_sac_nox_impact'].sum()
print(total_nox_impact_corrected)
print(total_nox_impact_gsp_improved)
total_cocip_atr20_impact_corrected = df_climate['cocip_atr20'].sum()
total_cocip_atr20_impact_gsp_improved = df_corr_climate['cocip_atr20'].sum()

total_non_co2_impact_corrected = df_climate['accf_sac_nox_impact'].sum()+df_climate['cocip_atr20'].sum()
total_non_co2_impact_gsp_improved = df_corr_climate['accf_sac_nox_impact'].sum()+df_corr_climate['cocip_atr20'].sum()

total_impact_corrected =df_climate['accf_sac_nox_impact'].sum()+df_climate['cocip_atr20'].sum()+df_climate['accf_sac_co2_impact'].sum()
total_impact_gsp_improved =df_corr_climate['accf_sac_nox_impact'].sum()+df_corr_climate['cocip_atr20'].sum()+df_corr_climate['accf_sac_co2_impact'].sum()

impact_labels = ['CO2', 'NOx', 'Contrails', 'Non-CO2', 'Total Climate Impact']
percentage_climate_differences = [
    ((total_co2_impact_gsp_improved - total_co2_impact_corrected) / total_co2_impact_corrected) * 100,
    ((total_nox_impact_gsp_improved - total_nox_impact_corrected) / total_nox_impact_corrected) * 100,
    ((total_cocip_atr20_impact_gsp_improved - total_cocip_atr20_impact_corrected) / total_cocip_atr20_impact_corrected) * 100,
    ((total_non_co2_impact_gsp_improved - total_non_co2_impact_corrected) / total_non_co2_impact_corrected) * 100,
    ((total_impact_gsp_improved - total_impact_corrected) / total_impact_corrected) * 100
]

plt.figure(figsize=(8, 6))
plt.bar(impact_labels, percentage_climate_differences)
plt.xlabel("Metric")
plt.ylabel("Percentage Difference (%)")
plt.title("Percentage Difference of GTF Improved GSP Compared to GTF Corrected")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()