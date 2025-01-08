import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Define flight trajectories
flight_trajectories = {
    "YFB → YAB": [(-68.5568, 63.7506), (-85.1511, 73.0056)],  # Iqaluit to Arctic Bay
    "JAV → NAQ": [(-51.0694, 69.2433), (-69.3856, 77.4889)],  # Ilulissat to Qaanaaq
    "LHR → IST": [(-0.4543, 51.4700), (28.9760, 41.0082)],    # London to Istanbul
    "CTS → PVG": [(141.6703, 43.0642), (121.8052, 31.1434)],  # Sapporo to Shanghai
    "SIN → CMB": [(103.9915, 1.3644), (79.8840, 6.9271)],    # Singapore to Colombo
    "GRU → MAO": [(-46.6292, -23.5505), (-60.0217, -3.1019)], # São Paulo to Manaus
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
    "YFB": (-68.5568, 63.7506),
    "YAB": (-85.1511, 73.0056),
    "JAV": (-51.0694, 69.2433),
    "NAQ": (-69.3856, 77.4889),
    "LHR": (-0.4543, 51.4700),
    "IST": (28.9760, 41.0082),
    "CTS": (141.6703, 43.0642),
    "PVG": (121.8052, 31.1434),
    "SIN": (103.9915, 1.3644),
    "CMB": (79.8840, 6.9271),
    "GRU": (-46.6292, -23.5505),
    "MAO": (-60.0217, -3.1019)
}
for code, (lon, lat) in airport_labels.items():
    ax.text(lon + 2, lat + 2, code, transform=ccrs.PlateCarree(), fontsize=10, color="black", weight="bold",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

# Add legend and title
ax.legend(loc='lower left', fontsize='medium', title="Flight Routes")
ax.set_title("Flight Trajectories", fontsize=16, weight="bold")

# Show the map
plt.savefig(f'figures/powerpoint/flight_trajectories.png', format='png')