"""Retrieve flights from OpenSky"""
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from traffic.data import opensky
# flight = opensky.history(
#     "2024-06-07 9:20",
#     stop="2024-06-07 11:50",
#     icao24="34610D",
#     # returns a Flight instead of a Traffic
#     return_flight=True
# )
#
# columns_pycontrail = ['timestamp', 'icao24', 'callsign', 'latitude', 'longitude', 'groundspeed', 'geoaltitude']
# flight_pycontrails = flight.data[columns_pycontrail]
# flight_pycontrails.to_csv('malaga_flight_test.csv')

flight = opensky.history(
    "2025-01-10 7:00",
    stop="2025-01-10 13:00",
    # departure_airport="EGLL",
    # arrival_airport="VCBI",
    # callsign = "H25563",
    icao24 = "E80338",
    # returns a Flight instead of a Traffic
    return_flight=True
)


# Columns for export
columns_pycontrail = ['timestamp', 'icao24', 'callsign', 'latitude', 'longitude', 'groundspeed', 'geoaltitude']
flight_pycontrails = flight.data[columns_pycontrail]

# Filter by a specific callsign
# specific_callsign = "HO1384"  # Replace with the callsign you want to filter
# filtered_flight = flight_pycontrails[flight_pycontrails['callsign'] == specific_callsign]
# flight_pycontrails = flight_pycontrails[flight_pycontrails['icao24'] == "781a94"]

# Save filtered flight to CSV
flight_pycontrails.to_csv('flight_trajectories/gru_lim.csv', index=False)

# Plot the flight on a map
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

# Add features to the map
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.set_extent([-20, 50, 30, 70])  # Focus on Europe
ax.set_extent([-90, -30, -60, 15])  # Focus on South America

flight_pycontrails = flight_pycontrails.dropna(subset=['latitude'])
# Plot the flight trajectory
plt.plot(
    flight_pycontrails['longitude'],
    flight_pycontrails['latitude'],
    linestyle='-', marker='o', color='blue',
    transform=ccrs.Geodetic(),
    # label=f"Flight: {specific_callsign}"
)

# Add annotations
start_point = flight_pycontrails.iloc[0]
end_point = flight_pycontrails.iloc[-1]
ax.text(start_point['longitude'], start_point['latitude'], 'Start', transform=ccrs.Geodetic(), color='green')
ax.text(end_point['longitude'], end_point['latitude'], 'End', transform=ccrs.Geodetic(), color='red')

# Add title and legend
plt.title(f"Flight Path")
plt.legend()
plt.show()