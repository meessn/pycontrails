import pandas as pd
import matplotlib.pyplot as plt

# Parameters
flight = "malaga"  # Replace with your flight identifier
engine_model = "GTF"  # Replace with your engine model
SAF_values = [0, 20, 100]  # List of SAF configurations
nvPM_number_data = {}  # To store data for plotting
nvPM_mass_data = {}
# Loop through SAF values and read corresponding files
for SAF in SAF_values:
    file_name = f"results/{flight}_model_{engine_model}_SAF_{SAF}.csv"
    try:
        # Read the CSV file
        df = pd.read_csv(file_name)

        # Extract the nvPM_p3t3_meem column and store it
        nvPM_number_data[SAF] = df['EI_nvpm_number_p3t3_meem']
        nvPM_mass_data[SAF] = df['EI_nvpm_mass_p3t3_meem']

    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except KeyError:
        print(f"'nvPM_p3t3_meem' column not found in file: {file_name}")

# Plot the data
plt.figure(figsize=(10, 6))
for SAF, nvPM_number in nvPM_number_data.items():
    plt.plot(nvPM_number, label=f"SAF {SAF}%")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_nvpm_number")
plt.title("P3T3-MEEM nvPM number for different SAF blends")
plt.legend(title="SAF blends")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
for SAF, nvPM_mass in nvPM_mass_data.items():
    plt.plot(nvPM_mass, label=f"SAF {SAF}%")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_nvpm_mass")
plt.title("P3T3-MEEM nvPM mass for different SAF blends")
plt.legend(title="SAF blends")
plt.grid()
plt.show()