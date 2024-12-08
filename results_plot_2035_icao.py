import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    "Stage": ["T/O", "Climb", "Approach", "Idle"],
    "GTF": [0.8218, 0.6745, 0.2243, 0.0848],
    "GTF2035": [0.7285, 0.6021, 0.2012, 0.0693],
}

# Create a DataFrame
df = pd.DataFrame(data)
df['% diff'] = ((df['GTF2035'] - df['GTF'])/df['GTF'])*100
# Adjusting the plot to make fuel flow bars smaller, let all three bars touch, and center x-axis labels
fig, ax1 = plt.subplots(figsize=(10, 6))

bar_width = 0.3
x = range(len(df["Stage"]))

# Plot ICAO and GasTurb fuel flows
ax1.bar(x, df["GTF"], width=bar_width, label="GTF", align='center', color='#00A6D6')
ax1.bar([p + bar_width for p in x], df["GTF2035"], width=bar_width, label="GTF2035", align='center', color='#0C2340')
ax1.set_xlabel("LTO Stage")
ax1.set_ylabel("Fuel Flow (kg/s)")
ax1.set_xticks([p + bar_width / 2 for p in x])
ax1.set_xticklabels(df["Stage"])
ax1.set_ylim(0, 0.9)
ax1.legend(loc='upper right')

# # Add percentage difference bars to the second axis
# ax2 = ax1.twinx()
# ax2.bar([p + bar_width for p in x], df["% diff"], width=bar_width, label="% Difference", color='#BFBFBF', alpha=0.6)
# ax2.set_ylabel("Percentage Difference (%)")
# ax2.set_ylim(-2, 9)  # Secondary axis now has its own independent scale
# ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Align zero lines
# ax2.legend(loc='upper right')

# Add title and layout adjustments
plt.title("Comparison of GTF and GTF2035 for LTO stages")
plt.tight_layout()
plt.savefig(f'figures/powerpoint/gtf_2035_lto.png', format='png')