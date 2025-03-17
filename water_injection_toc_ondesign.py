import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import pickle
import warnings
from emission_index import thrust_setting, p3t3_nox,p3t3_nvpm_meem
import matplotlib.ticker as ticker
# Load DataFrame from CSV (Modify the filename)
from matplotlib.lines import Line2D
engine_model = 'GTF2035_wi'
file_path = "water_injection_optimization_1.csv"  # Change to your actual file path
df = pd.read_csv(file_path, delimiter=";", decimal=",")
df = df.rename(columns={"Tt3 [K]": "TT3", "Pt3 [bar]": "PT3"})

print(df.columns)
if engine_model in ('GTF', 'GTF2035', 'GTF2035_wi'):
    with open('p3t3_graphs_sls_gtf_corr.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
elif engine_model in ('GTF1990', 'GTF2000'):
    with open('p3t3_graphs_sls_1990_2000.pkl', 'rb') as f:
        loaded_functions = pickle.load(f)
else:
    raise ValueError(f"Unsupported engine_model: {engine_model}.")

interp_func_far = loaded_functions['interp_func_far']
interp_func_pt3 = loaded_functions['interp_func_pt3']

# Get interpolation function bounds
x_min, x_max = interp_func_far.x[0], interp_func_far.x[-1]

# Get min and max TT3 from df
tt3_min = df['TT3'].min()
tt3_max = df['TT3'].max()

# Identify out-of-bounds values
out_of_bounds_mask = (df['TT3'] < x_min) | (df['TT3'] > x_max)
out_of_bounds_values = df.loc[out_of_bounds_mask, 'TT3']

df['specific_humidity'] = 0
df['SAF'] = 0
df['WAR'] = df['WAR']*100
if not out_of_bounds_values.empty:
    warnings.warn(f"TT3 values in df are outside the interpolation range ({x_min}, {x_max}). "
                  f"Min TT3: {tt3_min}, Max TT3: {tt3_max}. Extreme values are clipped.")

    print(f"Number of TT3 values out of bounds: {out_of_bounds_values.shape[0]}")
    print("Out-of-bounds TT3 values:", out_of_bounds_values.tolist())

    # Clamp values to stay within bounds
    df['TT3'] = df['TT3'].clip(lower=x_min, upper=x_max)

df['thrust_setting_meem'] = df.apply(
        lambda row: thrust_setting(
            engine_model,
            row['TT3'],
            interp_func_pt3
        ),
        axis=1
    )

"""NOx p3t3"""
df['ei_nox_p3t3'] = df.apply(
    lambda row: p3t3_nox(
        row['PT3'],
        row['TT3'],
        interp_func_far,
        interp_func_pt3,
        row['specific_humidity'],
        row['WAR'],
        engine_model
    ),
    axis=1
)
#

"""P3T3 _MEEM"""
df['ei_nvpm_number_p3t3_meem'] = df.apply(
    lambda row: p3t3_nvpm_meem(
        row['PT3'],
        row['TT3'],
        row['FAR '],
        interp_func_far,
        interp_func_pt3,
        row['SAF'],
        row['thrust_setting_meem'],
        engine_model
    ),
    axis=1
)

# Extract relevant data
# ei_nox = df["ei_nox_p3t3"]
# ei_nvpm_number = df["ei_nvpm_number_p3t3_meem"]
# tsfc = df["TSFC "]
# war = df["WAR"]
# tt4 = df["Tt4 [K]"]
# opr = df["OPR [-]"]
df = df.rename(columns={
    "Tt3 [K]": "TT3",
    "Pt3 [bar]": "PT3",
    "TSFC ": "TSFC",
    "Tt4 [K]": "TT4",
    "OPR [-]": "OPR",
    "FAR ": "FAR"  # Fix potential extra space
})
# print(ei_nvpm_number.max())
# Normalize EI nvPM Number for interpolation (to avoid precision errors)
df["ei_nvpm_number_norm"] = df["ei_nvpm_number_p3t3_meem"] / 1e14
ei_nvpm_norm = df["ei_nvpm_number_norm"]
ei_nvpm = df["ei_nvpm_number_p3t3_meem"]
ei_nox = df["ei_nox_p3t3"]
tsfc = df["TSFC"]
war = df["WAR"]
tt4 = df["TT4"]
opr = df["OPR"]

# Create grid for contour plot
xi = np.linspace(tsfc.min() , tsfc.max() , 100)  # Slightly zoomed out
yi = np.linspace(ei_nox.min() , ei_nox.max(), 100)
X, Y = np.meshgrid(xi, yi)

# Interpolated WAR values
Z = griddata((tsfc, ei_nox), war, (X, Y), method='linear')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Filled contour plot for WAR
c = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
cbar = plt.colorbar(c, label='WAR [%]')

# Extract unique OPR and TT4 values
op_levels = np.sort(df["OPR"].unique())  # Unique OPR levels
print(op_levels)
tt4_levels = [1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800]

# OPR Contours (Black solid lines)
op_contours = ax.contour(
    X, Y, griddata((tsfc, ei_nox), opr, (X, Y), method='linear'),
    levels=op_levels, colors='black'
)
ax.clabel(op_contours, fmt='%0.1f', colors='black')

# TT4 Contours (Red dashed lines)
tt4_contours = ax.contour(
    X, Y, griddata((tsfc, ei_nox), tt4, (X, Y), method='linear'),
    levels=tt4_levels, colors='red', linestyles='dashed'
)

ax.clabel(tt4_contours, fmt='%d', colors='red')

# Labels and title
ax.set_xlabel("TSFC [g/kNs]")
ax.set_ylabel("EI[NOx] [g/kg]")
ax.set_title("EI NOx vs TSFC with OPR and TT4 Contours")
# Add Legend for Contours
legend_lines = [
    Line2D([0], [0], color='black', linestyle='solid', label='OPR Contours'),
    Line2D([0], [0], color='red', linestyle='dashed', label='TT4 Contours')
]
ax.legend(handles=legend_lines, loc='lower right')



"""nvPM"""
# Create grid for contour plot
xi = np.linspace(tsfc.min() , tsfc.max() , 100)  # Slightly zoomed out
yi = np.linspace(ei_nvpm_norm.min() , ei_nvpm_norm.max(), 100)
X, Y = np.meshgrid(xi, yi)

# Interpolated WAR values
Z = griddata((tsfc, ei_nvpm_norm), war, (X, Y), method='linear')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Filled contour plot for WAR
c = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
cbar = plt.colorbar(c, label='WAR [%]')

# Extract unique OPR and TT4 values
op_levels = np.sort(df["OPR"].unique())  # Unique OPR levels
print(op_levels)
tt4_levels = [1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800]

# OPR Contours (Black solid lines)
op_contours = ax.contour(
    X, Y, griddata((tsfc, ei_nvpm_norm), opr, (X, Y), method='linear'),
    levels=op_levels, colors='black'
)
ax.clabel(op_contours, fmt='%0.1f', colors='black')

# TT4 Contours (Red dashed lines)
tt4_contours = ax.contour(
    X, Y, griddata((tsfc, ei_nvpm_norm), tt4, (X, Y), method='linear'),
    levels=tt4_levels, colors='red', linestyles='dashed'
)

ax.clabel(tt4_contours, fmt='%d', colors='red')

# Labels and title
ax.set_xlabel("TSFC [g/kNs]")
ax.set_ylabel("EI nvPM number [g/kg]")
ax.set_title("EI nvPM number vs TSFC with OPR and TT4 Contours")
# Add Legend for Contours
legend_lines = [
    Line2D([0], [0], color='black', linestyle='solid', label='OPR Contours'),
    Line2D([0], [0], color='red', linestyle='dashed', label='TT4 Contours')
]
ax.legend(handles=legend_lines, loc='lower right')

# Show plot
plt.show()