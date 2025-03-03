import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Generate random numbers for x (new) and y (old), ensuring they are never 0
np.random.seed(42)  # For reproducibility
num_samples = 1000

x_values = np.random.uniform(-10, 10, num_samples)
y_values = np.random.uniform(-10, 10, num_samples)

# Ensure that none of the values are exactly zero
x_values[x_values == 0] = np.random.choice([-1, 1]) * np.random.uniform(0.1, 10)
y_values[y_values == 0] = np.random.choice([-1, 1]) * np.random.uniform(0.1, 10)

# Compute both formulas
percent_change_old = (np.abs(x_values) - np.abs(y_values)) / np.abs(y_values)
normalized_relative_diff = (np.abs(x_values) - np.abs(y_values)) / (np.abs(x_values) + np.abs(y_values))

# Transform NRD using the formula 2a / (1 - a)
transformed_nrd = (2 * normalized_relative_diff) / (1 - normalized_relative_diff)

# Create DataFrame to compare results
df = pd.DataFrame({
    "New (x)": x_values,
    "Old (y)": y_values,
    "(|x|-|y|)/|y|": percent_change_old,
    "(|x|-|y|)/(|x|+|y|)": normalized_relative_diff,
    "Transformed NRD (2a/(1-a))": transformed_nrd,
    "Difference (Should be ~0)": transformed_nrd - percent_change_old
})

plt.figure(figsize=(10, 6))
plt.plot(df.index, df["Difference (Should be ~0)"])
plt.title("Plot of Differences (Should be Centered at 0)")
# plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(df.index, df["(|x|-|y|)/|y|"])
plt.plot(df.index, df["(|x|-|y|)/(|x|+|y|)"])
plt.title("Plot of Differences (Should be Centered at 0)")

plt.figure(figsize=(10, 6))
# plt.plot(df.index, df["(|x|-|y|)/|y|"])
plt.plot(df.index, df["(|x|-|y|)/(|x|+|y|)"])
plt.title("Plot of Differences (Should be Centered at 0)")
plt.show()

non_zero_rows = df[df["Difference (Should be ~0)"] != 0]
print(non_zero_rows[["(|x|-|y|)/|y|", "Transformed NRD (2a/(1-a))"]])


