import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Define file paths (update the paths if necessary)
files = {
    "A20N": "results/malaga_model_GTF_SAF_0_aircraft_A20N.csv",
    "A20N wf opr": "results/malaga_model_GTF_SAF_0_aircraft_A20N_wf_opr.csv",
    "A20N Full": "results/malaga_model_GTF_SAF_0_aircraft_A20N_full.csv"

}

# Initialize data structure for plotting
data = {}
total_sums = {}
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

            # Calculate total sums and percentage difference for the entire mission
            total_fuel_flow_per_engine = df['fuel_flow_per_engine'].sum()
            total_fuel_flow_gsp = df['fuel_flow_gsp'].sum()
            percent_diff_total = 100 * (total_fuel_flow_gsp - total_fuel_flow_per_engine) / total_fuel_flow_per_engine

            # Store results
            total_sums[label] = {
                'Total Fuel Flow per Engine': total_fuel_flow_per_engine,
                'Total Fuel Flow GSP': total_fuel_flow_gsp,
                'Percentage Difference': percent_diff_total
            }
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


# Prepare data for the bar chart
labels = list(total_sums.keys())
ps_model_values = [total_sums[label]['Total Fuel Flow per Engine'] for label in labels]
gsp_values = [total_sums[label]['Total Fuel Flow GSP'] for label in labels]
percentage_diffs = [total_sums[label]['Percentage Difference'] for label in labels]

# Bar chart positions
x = np.arange(len(labels))
width = 0.25  # Width of the bars

# Create the bar chart
plt.figure(figsize=(12, 7))
plt.bar(x - width, ps_model_values, width, label='PS Model (Fuel Flow per Engine)')
plt.bar(x, gsp_values, width, label='GSP (Fuel Flow)')
plt.bar(x + width, percentage_diffs, width, label='Percentage Difference (%)')

plt.title("Total mission fuel flow")
plt.xlabel("Files")
plt.ylabel("Values")
plt.xticks(x, labels)
plt.legend()
plt.grid(axis='y')
# bar_chart_path = os.path.join(output_dir, "fuelflow_ps_model_bar.png")
plt.savefig("figures/malaga/fuelflow_ps_model_sums.png", format='png')
# print(f"Bar chart saved to {bar_chart_path}")