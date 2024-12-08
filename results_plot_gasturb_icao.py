import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    "Stage": ["T/O", "Climb", "Approach", "Idle"],
    "ICAO": [0.800, 0.67, 0.232, 0.08],
    "GasTurb": [0.8218, 0.6745, 0.2243, 0.0848],
    "% diff": [3, 1, -3, 6]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Adjusting the plot to make fuel flow bars smaller, let all three bars touch, and center x-axis labels
fig, ax1 = plt.subplots(figsize=(10, 6))

bar_width = 0.2
x = range(len(df["Stage"]))

# Plot ICAO and GasTurb fuel flows
ax1.bar([p - bar_width for p in x], df["ICAO"], width=bar_width, label="ICAO Fuel Flow", align='center', color='#00A6D6')
ax1.bar(x, df["GasTurb"], width=bar_width, label="GasTurb Fuel Flow", align='center', color='#0C2340')
ax1.set_xlabel("LTO Stage")
ax1.set_ylabel("Fuel Flow (kg/s)")
ax1.set_xticks(x)
ax1.set_xticklabels(df["Stage"])
ax1.set_ylim(-0.4, 0.9)
ax1.legend(loc='lower left')

# Add percentage difference bars to the second axis
ax2 = ax1.twinx()
ax2.bar([p + bar_width for p in x], df["% diff"], width=bar_width, label="% Difference", color='#BFBFBF', alpha=0.6)
ax2.set_ylabel("Percentage Difference (%)")
ax2.set_ylim(-4, 9)  # Secondary axis now has its own independent scale
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Align zero lines
ax2.legend(loc='upper right')

# Add title and layout adjustments
plt.title("Comparison of ICAO and GasTurb Fuel Flows for LTO stages")
plt.tight_layout()
plt.savefig(f'figures/powerpoint/icao_gasturb.png', format='png')
