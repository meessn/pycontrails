import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV file
df_luft = pd.read_csv("20250219_Selected_Reference_Aircraft_Missions_luftbauhaus.csv", delimiter=";", decimal=',')  # Adjust delimiter if needed

# Convert duration to cumulative time in minutes
df_luft["Cumulative Time (min)"] = df_luft["Duration [s]"].cumsum() / 60

# Generate full-minute timestamps
time_min = np.arange(0, int(np.ceil(df_luft["Cumulative Time (min)"].max())) + 1, 1)

# Interpolate values for the new DataFrame
df_luft_interp = pd.DataFrame({"Cumulative Time (min)": time_min})
df_luft_interp["Fuel Flow per Engine [kg/s]"] = np.interp(time_min, df_luft["Cumulative Time (min)"], df_luft["Fuel Flow per Engine [kg/s]"])
df_luft_interp["Total Aircraft Thrust [N]"] = np.interp(time_min, df_luft["Cumulative Time (min)"], df_luft["Total Aircraft Thrust [N]"])
df_luft_interp["Altitude [m]"] = np.interp(time_min, df_luft["Cumulative Time (min)"], df_luft["Altitude [m]"])
df_luft_interp["Mach [-]"] = np.interp(time_min, df_luft["Cumulative Time (min)"], df_luft["Mach [-]"])

# Compute Engine Thrust (kN)
df_luft_interp["Engine Thrust [kN]"] = (0.5 * df_luft_interp["Total Aircraft Thrust [N]"]) / 1000


df_malaga_luft = pd.read_csv("main_results_figures/results/malaga/malaga/emissions/GTF_SAF_0_A20N_full_WAR_0_0_0.csv")


# Find the starting altitude in df2
start_altitude = df_malaga_luft.iloc[0]["altitude"]

# Find the first index in df1 where altitude matches the start altitude of df2
start_index = (df_luft_interp["Altitude [m]"] - start_altitude).abs().idxmin()

# Get the corresponding cumulative time in df1
start_time = df_luft_interp.loc[start_index, "Cumulative Time (min)"]

# Assign cumulative time in df2 starting from start_time
df_malaga_luft["Cumulative Time (min)"] = np.arange(start_time, start_time + len(df_malaga_luft))

df_piano_luft = pd.read_csv(f"pianoX_malaga.csv", delimiter=';', decimal=',', index_col='index')

# Find the first index in df_malaga_luft
start_index_malaga = df_malaga_luft.index.min()  # Smallest index in df_malaga_luft
start_index_piano = df_piano_luft.index.min()  # Smallest index in df_piano_luft (36 in this case)

# Find the corresponding cumulative time for df_piano_luft's first index in df_malaga_luft
start_time_piano = df_malaga_luft.loc[start_index_piano, "Cumulative Time (min)"]

# Assign cumulative time to df_piano_luft based on df_malaga_luft's timeline
df_piano_luft["Cumulative Time (min)"] = np.arange(start_time_piano, start_time_piano + len(df_piano_luft))

# Plot Fuel Flow per Engine vs. Time
plt.figure(figsize=(10, 6))
plt.plot(df_luft_interp["Cumulative Time (min)"], df_luft_interp["Fuel Flow per Engine [kg/s]"], label="Reference Flight", linestyle="-", marker="o", markersize=2.5, color='tab:purple')
plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["fuel_flow_per_engine"], label="pycontrails", linestyle="-", marker="o", markersize=2.5, color='tab:blue')
plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["fuel_flow_gsp"], label="GSP", linestyle="-", marker="o", markersize=2.5, color='tab:orange')
plt.plot(df_piano_luft["Cumulative Time (min)"], df_piano_luft['fuel_flow_piano'], label='PianoX', linestyle='-', marker='o', markersize=2.5, color='tab:green')
plt.title("Fuel Flow Over Time")
plt.xlabel("Time (minutes)")
plt.ylabel("Fuel Flow (kg/s)")
plt.legend()
plt.grid(True)
# plt.show()

# Plot Thrust per Engine vs. Time
plt.figure(figsize=(10, 6))
plt.plot(df_luft_interp["Cumulative Time (min)"], df_luft_interp["Engine Thrust [kN]"], label="Reference Flight", linestyle="-", marker="o", markersize=2.5, color='tab:purple')
plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["thrust_per_engine"], label="pycontrails", linestyle="-", marker="o", markersize=2.5, color='tab:blue')
# plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["thrust_gsp"], label="GSP", linestyle="-", marker="o", markersize=2.5, color='tab:orange')
plt.title("Thrust per Engine Over Time")
plt.xlabel("Time (minutes)")
plt.ylabel("Thrust (kN)")
plt.legend()
plt.grid(True)
# plt.show()

# Plot Altitude vs. Time
plt.figure(figsize=(10, 6))
plt.plot(df_luft_interp["Cumulative Time (min)"], df_luft_interp["Altitude [m]"], label="Reference Flight", linestyle="-", marker="o", markersize=2.5, color='tab:purple')
plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["altitude"], label="pycontrails", linestyle="-", marker="o", markersize=2.5, color='tab:blue')
plt.title("Altitude Over Time")
plt.xlabel("Time (minutes)")
plt.ylabel("Altitude (m)")
plt.legend()
plt.grid(True)
# plt.show()

# Plot Mach vs. Time
plt.figure(figsize=(10, 6))
plt.plot(df_luft_interp["Cumulative Time (min)"], df_luft_interp["Mach [-]"], label="Reference Flight", linestyle="-", marker="o", markersize=2.5, color='tab:purple')
plt.plot(df_malaga_luft["Cumulative Time (min)"], df_malaga_luft["mach"], label="pycontrails", linestyle="-", marker="o", markersize=2.5, color='tab:blue')
plt.title("Mach Over Time")
plt.xlabel("Time (minutes)")
plt.ylabel(f"Mach $EI_{{\\mathrm{{NOx}}}}$")
plt.legend()
plt.grid(True)
plt.show()