import pandas as pd

# Load the CSV file (update 'your_file.csv' with the actual filename)
prediction = "mees"
file_path = f"main_results_figures/results/malaga/malaga/climate/{prediction}/era5model/GTF_SAF_0_A20N_full_WAR_0_cli_cont.csv"
df = pd.read_csv(file_path)

# Ensure the file contains data
if df.empty:
    print("The CSV file is empty.")
else:
    # Extract column names and values from the first row
    for column in df.columns:
        value = df[column].iloc[0]  # Get the first value for each column
        print(f"{column}: {value}")