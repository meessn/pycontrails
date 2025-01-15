from datetime import datetime, timedelta
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
import pytz
from pycontrails import Flight
import os
from pycontrails.physics.geo import cosine_solar_zenith_angle, orbital_position
# Define the list of airports and their details

def process_flight(trajectory, flight_results):
    # Construct file name from trajectory details
    file_name = f"{trajectory['departure_airport'].lower()}_{trajectory['arrival_airport'].lower()}.csv"

    # Read and preprocess the flight data
    df = pd.read_csv(f"flight_trajectories/{file_name}")
    df = df.rename(columns={'geoaltitude': 'altitude', 'groundspeed': 'groundspeed', 'timestamp': 'time'})
    df['altitude'] = df['altitude'] * 0.3048  # foot to meters
    df['groundspeed'] = df['groundspeed'] * 0.514444444
    df['time'] = pd.to_datetime(df['time'])
    df = df.dropna(subset=['latitude', 'longitude', 'altitude'])

    # Resample using PyContrails Flight
    fl = Flight(df)
    fl = fl.resample_and_fill(freq="60s", drop=False)
    # print(fl.dataframe['altitude'])
    df_resampled = fl.dataframe  # Confirm this property is correct

    # Adjust times for each date and save
    for date in dates_of_interest:
        for period in ['Daytime', 'Nighttime']:
            # Extract departure time from flight_results
            departure_time_col = f"{date.strftime('%Y-%m-%d')} {period} Departure (UTC)"
            departure_time = pd.to_datetime(
                flight_results.loc[flight_results['Flight'] == trajectory['flight'], departure_time_col].values[0])

            # Adjust timestamps
            time_deltas = df_resampled['time'].diff().fillna(pd.Timedelta(0))
            # print(time_deltas)
            df_resampled['time'] = departure_time + time_deltas.cumsum()
            # df_resampled['time'] = df_resampled['time'].dt.tz_localize('UTC')
            # df_resampled['time'] = df_resampled['time'].dt.strftime('%Y-%m-%d %H:%M:%S') + '+00:00'
            # Create output folder structure inside flight_trajectories

            # Calculate cosine of the solar zenith angle
            df_resampled['cos_sza'] = cosine_solar_zenith_angle(
                longitude=df_resampled['longitude'].values,
                latitude=df_resampled['latitude'].values,
                time=df_resampled['time'].values.astype('datetime64[ns]'),
                theta_rad=orbital_position(df_resampled['time'].values.astype('datetime64[ns]'))
            )

            # Determine if the sun is present
            sun_present = df_resampled['cos_sza'] > 0

            # Compute the fraction of the trajectory in daylight
            sunlit_fraction = sun_present.mean()

            # Print summary for the flight
            print(f"Flight {trajectory['flight']} on {date.strftime('%Y-%m-%d')} ({period}):")
            print(f"  Sunlit fraction: {sunlit_fraction:.2%}")
            if sunlit_fraction > 0.5:
                print(f"  Status: Sun is mostly present during the {period.lower()} flight.")
            else:
                print(f"  Status: Sun is mostly absent during the {period.lower()} flight.")
            print("-" * 50)

            folder = os.path.join("flight_trajectories", "processed_flights",
                                  f"{trajectory['departure_airport'].lower()}_{trajectory['arrival_airport'].lower()}")
            os.makedirs(folder, exist_ok=True)

            # Build output file name with flight, date, and period in lowercase
            output_file = os.path.join(folder,
                                       f"{trajectory['departure_airport'].lower()}_{trajectory['arrival_airport'].lower()}_{date.strftime('%Y-%m-%d')}_{period.lower()}.csv")

            # Save to CSV
            df_resampled.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


airports = [
    {"airport": "HEL", "city": "Helsinki, Finland", "latitude": 60.1695, "longitude": 24.9354, "timezone": "Europe/Helsinki"},
    {"airport": "KEF", "city": "Reykjavik, Iceland", "latitude": 64.1355, "longitude": -21.8954, "timezone": "Atlantic/Reykjavik"},
    {"airport": "DUS", "city": "Dusseldorf, Germany", "latitude": 51.2277, "longitude": 6.7735, "timezone": "Europe/Berlin"},
    {"airport": "TOS", "city": "Tromso, Norway", "latitude": 69.6496, "longitude": 18.9560, "timezone": "Europe/Oslo"},
    {"airport": "LHR", "city": "London, UK", "latitude": 51.4700, "longitude": -0.4543, "timezone": "Europe/London"},
    {"airport": "IST", "city": "Istanbul, Turkey", "latitude": 41.0082, "longitude": 28.9784, "timezone": "Europe/Istanbul"},
    {"airport": "CTS", "city": "Sapporo, Japan", "latitude": 43.0642, "longitude": 141.3469, "timezone": "Asia/Tokyo"},
    {"airport": "TPE", "city": "Taoyuan, Taiwan", "latitude": 25.0777, "longitude": 121.2322, "timezone": "Asia/Taipei"},
    {"airport": "BOS", "city": "Boston, USA", "latitude": 42.3601, "longitude": -71.0589, "timezone": "America/New_York"},
    {"airport": "FLL", "city": "Miami, USA", "latitude": 25.7617, "longitude": -80.1918, "timezone": "America/New_York"},
    {"airport": "SFO", "city": "San Francisco, USA", "latitude": 37.7749, "longitude": -122.4194, "timezone": "America/Los_Angeles"},
    {"airport": "DFW", "city": "Dallas, USA", "latitude": 32.7767, "longitude": -96.7970, "timezone": "America/Chicago"},
    {"airport": "SIN", "city": "Singapore, Singapore", "latitude": 1.3521, "longitude": 103.8198, "timezone": "Asia/Singapore"},
    {"airport": "MAA", "city": "Chennai, India", "latitude": 13.0827, "longitude": 80.2707, "timezone": "Asia/Kolkata"},
    {"airport": "GRU", "city": "Sao Paulo, Brazil", "latitude": -23.5505, "longitude": -46.6333, "timezone": "America/Sao_Paulo"},
    {"airport": "LIM", "city": "Lima, Peru", "latitude": -12.0464, "longitude": -77.0428, "timezone": "America/Lima"}
]

# Define dates of interest
dates_of_interest = [
    datetime(2023, 2, 6),
    datetime(2023, 5, 5),
    datetime(2023, 8, 6),
    datetime(2023, 11, 6)
]

# Initialize a dictionary for the reformatted results
reformatted_results = {"Airport": []}
reformatted_results_utc = {"Airport": []}

# Add columns for each date
for date in dates_of_interest:
    reformatted_results[date.strftime("%Y-%m-%d") + " Sunrise"] = []
    reformatted_results[date.strftime("%Y-%m-%d") + " Sunset"] = []
    reformatted_results_utc[date.strftime("%Y-%m-%d") + " Sunrise (UTC)"] = []
    reformatted_results_utc[date.strftime("%Y-%m-%d") + " Sunset (UTC)"] = []

# Populate the reformatted results
for airport in airports:
    reformatted_results["Airport"].append(airport["airport"])
    reformatted_results_utc["Airport"].append(airport["airport"])
    location = LocationInfo(airport["city"], "Country", airport["timezone"], airport["latitude"], airport["longitude"])
    for date in dates_of_interest:
        try:
            s = sun(location.observer, date=date, tzinfo=pytz.timezone(airport["timezone"]))
            reformatted_results[date.strftime("%Y-%m-%d") + " Sunrise"].append(s["sunrise"].strftime("%Y-%m-%d %H:%M"))
            reformatted_results[date.strftime("%Y-%m-%d") + " Sunset"].append(s["sunset"].strftime("%Y-%m-%d %H:%M"))
            reformatted_results_utc[date.strftime("%Y-%m-%d") + " Sunrise (UTC)"].append(s["sunrise"].astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M"))
            reformatted_results_utc[date.strftime("%Y-%m-%d") + " Sunset (UTC)"].append(s["sunset"].astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M"))
        except ValueError:
            # Handle cases for continuous daylight or darkness
            reformatted_results[date.strftime("%Y-%m-%d") + " Sunrise"].append("N/A")
            reformatted_results[date.strftime("%Y-%m-%d") + " Sunset"].append("N/A")
            reformatted_results_utc[date.strftime("%Y-%m-%d") + " Sunrise (UTC)"].append("N/A")
            reformatted_results_utc[date.strftime("%Y-%m-%d") + " Sunset (UTC)"].append("N/A")

# Convert results to DataFrames
df_reformatted = pd.DataFrame(reformatted_results)
df_reformatted_utc = pd.DataFrame(reformatted_results_utc)

# Define flights and their departure times and durations
flights = [
    {"flight": "HEL-KEF", "departure_airport": "HEL", "arrival_airport": "KEF", "duration": timedelta(hours=3, minutes=40)},
    {"flight": "DUS-TOS", "departure_airport": "DUS", "arrival_airport": "TOS", "duration": timedelta(hours=3, minutes=25)},
    {"flight": "LHR-IST", "departure_airport": "LHR", "arrival_airport": "IST", "duration": timedelta(hours=4)},
    {"flight": "CTS-TPE", "departure_airport": "CTS", "arrival_airport": "TPE", "duration": timedelta(hours=4, minutes=40)},
    {"flight": "BOS-FLL", "departure_airport": "BOS", "arrival_airport": "FLL", "duration": timedelta(hours=3, minutes=40)},
    {"flight": "SFO-DFW", "departure_airport": "SFO", "arrival_airport": "DFW", "duration": timedelta(hours=3, minutes=40)},
    {"flight": "SIN-MAA", "departure_airport": "SIN", "arrival_airport": "MAA", "duration": timedelta(hours=4, minutes=15)},
    {"flight": "GRU-LIM", "departure_airport": "GRU", "arrival_airport": "LIM", "duration": timedelta(hours=5)}
]

# Initialize results for flight departure and arrival times
flight_results = {"Flight": []}
for date in dates_of_interest:
    flight_results[date.strftime("%Y-%m-%d") + " Daytime Departure (UTC)"] = []
    flight_results[date.strftime("%Y-%m-%d") + " Daytime Arrival (UTC)"] = []
    flight_results[date.strftime("%Y-%m-%d") + " Nighttime Departure (UTC)"] = []
    flight_results[date.strftime("%Y-%m-%d") + " Nighttime Arrival (UTC)"] = []

# Calculate departure and arrival times in UTC
for flight in flights:
    flight_results["Flight"].append(flight["flight"])
    departure_airport = flight["departure_airport"]
    duration = flight["duration"]
    airport_info = next(a for a in airports if a["airport"] == departure_airport)
    airport_timezone = pytz.timezone(airport_info["timezone"])
    for date in dates_of_interest:
        # Daytime departure and arrival
        daytime_local = airport_timezone.localize(datetime.combine(date, datetime.strptime("10:00", "%H:%M").time()))
        daytime_utc = daytime_local.astimezone(pytz.utc)
        daytime_arrival_utc = daytime_utc + duration
        flight_results[date.strftime("%Y-%m-%d") + " Daytime Departure (UTC)"].append(daytime_utc.strftime("%Y-%m-%d %H:%M"))
        flight_results[date.strftime("%Y-%m-%d") + " Daytime Arrival (UTC)"].append(daytime_arrival_utc.strftime("%Y-%m-%d %H:%M"))

        # Nighttime departure and arrival
        nighttime_local = airport_timezone.localize(datetime.combine(date, datetime.strptime("00:00", "%H:%M").time()))
        nighttime_utc = nighttime_local.astimezone(pytz.utc)
        nighttime_arrival_utc = nighttime_utc + duration
        flight_results[date.strftime("%Y-%m-%d") + " Nighttime Departure (UTC)"].append(nighttime_utc.strftime("%Y-%m-%d %H:%M"))
        flight_results[date.strftime("%Y-%m-%d") + " Nighttime Arrival (UTC)"].append(nighttime_arrival_utc.strftime("%Y-%m-%d %H:%M"))

# Convert flight results to a DataFrame
df_flight_results = pd.DataFrame(flight_results)

# Save all DataFrames to CSV
df_reformatted.to_csv("flight_trajectories/sunrise_sunset_selected_dates.csv", index=False, decimal=",", sep=";")
df_reformatted_utc.to_csv("flight_trajectories/sunrise_sunset_selected_dates_utc.csv", index=False, decimal=",", sep=";")
df_flight_results.to_csv("flight_trajectories/flight_results.csv", index=False, decimal=",", sep=";")

# print("All tables have been successfully saved.")
for flight in flights:
    process_flight(flight, df_flight_results)





