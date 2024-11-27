"""Retrieve flights from OpenSky"""
import traffic
from traffic.data import opensky

flight = opensky.history(
    "2024-06-07 9:20",
    stop="2024-06-07 11:50",
    icao24="34610D",
    # returns a Flight instead of a Traffic
    return_flight=True
)

columns_pycontrail = ['timestamp', 'icao24', 'callsign', 'latitude', 'longitude', 'groundspeed', 'geoaltitude']
flight_pycontrails = flight.data[columns_pycontrail]
flight_pycontrails.to_csv('malaga_flight_1.csv')
