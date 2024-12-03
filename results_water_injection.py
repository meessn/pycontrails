import pandas as pd
import matplotlib.pyplot as plt

"""FLIGHT PARAMETERS"""
engine_model = 'GTF2035'        # GTF , GTF2035
water_injection = [0, 0, 0]     # WAR climb cruise approach/descent
SAF = 0                         # 0, 20, 100 unit = %
flight = 'malaga'
aircraft = 'A20N_full'        # A20N ps model, A20N_wf is change in Thrust and t/o and idle fuel flows
                            # A20N_wf_opr is with changed nominal opr and bpr
                            # A20N_full has also the eta 1 and 2 and psi_0



# Define the two water injection options
water_injection_options = [
    [0, 0, 0],
    [17, 14, 16]
]
fuel_flow_gsp_data = {}
nox_p3t3_data = {}

# Loop through SAF values and read corresponding files
for water_injection in water_injection_options:
    # Format the water injection values into strings
    formatted_values = [str(value).replace('.', '_') for value in water_injection]
    formatted_war = "_".join(formatted_values)

    # Construct the file name
    file_name = f'results/{flight}/{flight}_model_{engine_model}_SAF_{SAF}_aircraft_{aircraft}_WAR_{formatted_war}.csv'

    try:
        # Read the CSV file
        df = pd.read_csv(file_name)

        # Use tuple as key for dictionary
        water_injection_key = tuple(water_injection)

        # Extract the fuel flow and EI_NOx data
        fuel_flow_gsp_data[water_injection_key] = df['fuel_flow_gsp']
        nox_p3t3_data[water_injection_key] = df['EI_nox_p3t3']
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except KeyError as e:
        print(f"Column {e} not found in file: {file_name}")




# Plot the data
plt.figure(figsize=(10, 6))
for war, fuel in fuel_flow_gsp_data.items():
    plt.plot(fuel, label=f"WAR {war}")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("Fuel flow [kg/s]")
plt.title("Fuel flow")
plt.legend(title="WAR")
plt.grid()
plt.savefig(f'figures/{flight}/fuel_flow_water_injection.png', format='png')

plt.figure(figsize=(10, 6))
for war, nox in nox_p3t3_data.items():
    plt.plot(nox, label=f"WAR {war}")

plt.xlabel("Index")  # Or replace "Index" with a relevant x-axis label if needed
plt.ylabel("EI_NOx")
plt.title("P3T3 EI_NOx")
plt.legend(title="WAR")
plt.grid()
plt.savefig(f'figures/{flight}/EI_nox_water_injection.png', format='png')

