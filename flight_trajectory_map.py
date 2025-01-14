import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Define flight trajectories
flight_trajectories = {
    "HEL → KEF": [(24.9633, 60.3172), (-22.6056, 63.9850)],  # Helsinki to Reykjavik
    "DUS → TOS": [(6.7668, 51.2895), (18.9189, 69.6833)],    # Dusseldorf to Tromso
    "LHR → IST": [(-0.4543, 51.4700), (28.9760, 41.0082)],    # London to Istanbul
    "CTS → TPE": [(141.6703, 43.0642), (121.2322, 25.0777)],  # Sapporo to Taiwan
    "BOS → FLL": [(-71.0052, 42.3656), (-80.2906, 25.7933)],  # Boston to Miami
    "SFO → DFW": [(-122.3790, 37.6213), (-97.0381, 32.8998)], # San Francisco to Dallas
    "SIN → MAA": [(103.9915, 1.3644), (80.1700, 13.0827)],    # Singapore to Chennai
    "GRU → LIM": [(-46.6292, -23.5505), (-77.1143, -12.0219)] # Sao Paulo to Lima
}

# Create the map
fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Add gridlines
gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()

# Add climate region boundaries
# Polar regions
ax.plot([-180, 180], [66.5, 66.5], color='blue', linestyle='--', linewidth=1, transform=ccrs.PlateCarree())
ax.plot([-180, 180], [-66.5, -66.5], color='blue', linestyle='--', linewidth=1, transform=ccrs.PlateCarree())
# Tropical region
ax.plot([-180, 180], [23.5, 23.5], color='red', linestyle='--', linewidth=1, transform=ccrs.PlateCarree())
ax.plot([-180, 180], [-23.5, -23.5], color='red', linestyle='--', linewidth=1, transform=ccrs.PlateCarree())

# Plot flight trajectories with geodesic curves
for route, coords in flight_trajectories.items():
    lons, lats = zip(*coords)
    ax.plot(lons, lats, marker='o', label=route, transform=ccrs.Geodetic())

# Add labels for key airports
airport_labels = {
    "HEL": (24.9633, 60.3172),
    "KEF": (-22.6056, 63.9850),
    "DUS": (6.7668, 51.2895),
    "TOS": (18.9189, 69.6833),
    "LHR": (-0.4543, 51.4700),
    "IST": (28.9760, 41.0082),
    "CTS": (141.6703, 43.0642),
    "TPE": (121.2322, 25.0777),
    "BOS": (-71.0052, 42.3656),
    "FLL": (-80.2906, 25.7933),
    "SFO": (-122.3790, 37.6213),
    "DFW": (-97.0381, 32.8998),
    "SIN": (103.9915, 1.3644),
    "MAA": (80.1700, 13.0827),
    "GRU": (-46.6292, -23.5505),
    "LIM": (-77.1143, -12.0219)
}
for code, (lon, lat) in airport_labels.items():
    if code == "LHR":
        x_offset, y_offset = (-16, -2)  # Move LHR label to avoid overlap
    elif code == "DUS":
        x_offset, y_offset = (4, 0)  # Move DUS label to avoid overlap
    else:
        x_offset, y_offset = (2, 2)
    ax.text(lon + x_offset, lat + y_offset, code, transform=ccrs.PlateCarree(), fontsize=10, color="black", weight="bold",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))


# Add legend and title
ax.legend(loc='lower left', fontsize='medium', title="Flight Routes")
ax.set_title("Selected Flight Trajectories", fontsize=16, weight="bold")

plt.savefig(f'figures/powerpoint/flight_trajectories.png', format='png')
