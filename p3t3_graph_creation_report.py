import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Read the file
file_path = 'P3T3_SLS_GRAPHS_PW1127G_V2.csv'
data = pd.read_csv(file_path, delimiter=';', decimal=',')
data = data.drop(index=0).reset_index(drop=True)

# Convert columns to numeric after replacing ',' with '.'
data['TT3'] = pd.to_numeric(data['TT3'].str.replace(',', '.'))
data['FAR'] = pd.to_numeric(data['FAR'].str.replace(',', '.'))
data['PT3'] = pd.to_numeric(data['PT3'].str.replace(',', '.'))

# Extract the 'TT3', 'FAR', and 'PT3' columns
TT3 = data['TT3'].values
FAR = data['FAR'].values
PT3 = data['PT3'].values

# Create interpolation functions
interp_func_FAR = interp1d(TT3, FAR, kind='linear')
interp_func_PT3 = interp1d(TT3, PT3, kind='linear')

# Define points for red dots
highlight_tt3 = [869.99, 836.98, 664.19, 498.41]

# Interpolated values for red dots
highlight_FAR = interp_func_FAR(highlight_tt3)
highlight_PT3 = interp_func_PT3(highlight_tt3)

# Plot 1: PT3 as a function of TT3
plt.figure(figsize=(10, 6))
plt.plot(TT3, PT3, label='$P_{t3,GR}$ vs $T_{t3}$', color='#00C4FC', linewidth=1.5)
plt.scatter(highlight_tt3, highlight_PT3, color='grey', label='LTO stages', zorder=5)
plt.title(r'$P_{t3,GR}$ versus $T_{t3}$ Map')
plt.xlabel(r'$T_{t3}$ [K]')
plt.ylabel(r'$P_{t3}$ [bar]')
plt.legend()
plt.grid(True)
plt.savefig('figures/powerpoint/pt3_tt3_map')

# Plot 2: FAR as a function of TT3
plt.figure(figsize=(10, 6))
plt.plot(TT3, FAR, label='$FAR_{GR}$ vs $T_{t3}$', color='#00C4FC', linewidth=1.5)
plt.scatter(highlight_tt3, highlight_FAR, color='grey', label='LTO stages', zorder=5)
plt.title(r'$FAR_{GR}$ versus $T_{t3}$ Map')
plt.xlabel(r'$T_{t3}$ [K]')
plt.ylabel(r'$FAR_{GR}$ [-]')
plt.legend()
plt.grid(True)
plt.savefig('figures/powerpoint/far_tt3_map')
