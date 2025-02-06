import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle

# Load the CSV file (adjust the path to your CSV file)
file_path = 'P3T3_SLS_GRAPHS_PW1127G_V3.csv'  # Replace with the actual file path

# Read the CSV file with the specified delimiter and decimal format
data = pd.read_csv(file_path, delimiter=';', decimal=',')
data = data.drop(index=0).reset_index(drop=True)

# Display the loaded data to verify
print(data.head())

# Convert columns to numeric after replacing ',' with '.'
data['TT3'] = pd.to_numeric(data['TT3'].str.replace(',', '.'))
data['FAR'] = pd.to_numeric(data['FAR'].str.replace(',', '.'))
data['PT3'] = pd.to_numeric(data['PT3'].str.replace(',', '.'))


# Extract the 'TT3' and 'FAR' columns
TT3 = data['TT3'].values
FAR = data['FAR'].values
PT3 = data['PT3'].values
# Create the interpolation function
interp_func_FAR = interp1d(TT3, FAR, kind='linear')
interp_func_PT3 = interp1d(TT3, PT3, kind='linear')

# Example usage: Get an interpolated FAR value for a specific TT3
TT3_query = 450  # Replace with your desired TT3 value
FAR_value = interp_func_FAR(TT3_query)
PT3_value = interp_func_PT3(TT3_query)
print(f"Interpolated FAR value at TT3={TT3_query}: {FAR_value}")
print(f"Interpolated PT3 value at TT3={TT3_query}: {PT3_value}")
# Optional: Plot the data and the interpolation
plt.plot(TT3, FAR, 'o', label='Measured data')
plt.xlabel('TT3 (K)')
plt.ylabel('FAR (-)')
plt.legend()
plt.show()

functions = {'interp_func_far': interp_func_FAR, 'interp_func_pt3': interp_func_PT3}
#save interpolation functions
with open('p3t3_graphs_sls.pkl', 'wb') as f:
    pickle.dump(functions, f)

