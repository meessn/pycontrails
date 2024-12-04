import pandas as pd

# # Loop through each file from 0 to 9
# for i in range(9):
#     # Define the input file name
#     input_file = f"water_injection_optimized/point_{i}_output.csv"
#
#     # Define the output file name
#     output_file = f"water_injection_optimized/point_{i}_output_processed.csv"
#
#     # Load the CSV file into a pandas DataFrame
#     df = pd.read_csv(input_file)
#
#     # Save the DataFrame to a new CSV file with the required formatting
#     df.to_csv(output_file, sep=';', decimal=',', index=False)
#
#     print(f"Processed and saved: {output_file}")

# Define the input file name
input_file = f"results/malaga/malaga_model_GTF2035_SAF_0_aircraft_A20N_full_WAR_0_0_0.csv"

# Define the output file name
output_file = f"results/malaga/malaga_model_GTF2035_SAF_0_aircraft_A20N_full_WAR_0_0_0_processed.csv"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Save the DataFrame to a new CSV file with the required formatting
df.to_csv(output_file, sep=';', decimal=',', index=False)