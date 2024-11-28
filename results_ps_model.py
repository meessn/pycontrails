import pandas as pd
import matplotlib.pyplot as plt

# Define file paths (update the paths if necessary)
files = {
    "A20N": "results/malaga_model_GTF_SAF_0_aircraft_A20N.csv",
    "A20N wf opr": "results/malaga_model_GTF_SAF_0_aircraft_A20N_wf_opr.csv",
    "A20N Full": "results/malaga_model_GTF_SAF_0_aircraft_A20N_full.csv"

}

# Initialize data structure for plotting
data = {}

# Process each file and calculate percentage difference
for label, file in files.items():
    try:
        # Load the CSV
        df = pd.read_csv(file)

        # Ensure the required columns exist
        if 'fuel_flow_per_engine' in df.columns and 'fuel_flow_gsp' in df.columns:
            # Calculate percentage difference
            df['percent_diff'] = 100 * (df['fuel_flow_gsp'] - df['fuel_flow_per_engine']) / df['fuel_flow_per_engine']
            data[label] = df['percent_diff']
        else:
            print(f"Columns 'fuel_flow' and 'fuel_flow_gsp' not found in {file}")
    except FileNotFoundError:
        print(f"File not found: {file}")
    except Exception as e:
        print(f"An error occurred while processing {file}: {e}")

# Plot the data
plt.figure(figsize=(10, 6))

for label, percent_diff in data.items():
    plt.plot(percent_diff, label=label)

plt.title("Fuel flow gsp difference from fuel flow PS model")
plt.xlabel("Index")
plt.ylabel("Percentage Difference (%)")
plt.legend()
plt.grid()
plt.savefig("figures/malaga/fuelflow_ps_model.png", format='png')