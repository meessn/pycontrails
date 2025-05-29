from datetime import datetime
import pytz
import os
from pycontrails.physics.geo import cosine_solar_zenith_angle, orbital_position
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timezonefinder import TimezoneFinder
from geopy.distance import geodesic
import re
# Define the list of airports and their details

# Load the global airport dataset
global_airports = pd.read_csv("flight_trajectories/airports.csv")
global_airports = global_airports[global_airports["iata_code"].notna()]

# Function to find nearest airport from a coordinate
def find_nearest_airport(lat, lon):
    def distance_to(row):
        return geodesic((lat, lon), (row["latitude_deg"], row["longitude_deg"])).kilometers
    global_airports["distance"] = global_airports.apply(distance_to, axis=1)
    nearest = global_airports.loc[global_airports["distance"].idxmin()]
    return {
        "icao": nearest["ident"],
        "iata": nearest["iata_code"],
        "name": nearest["name"],
        "lat": nearest["latitude_deg"],
        "lon": nearest["longitude_deg"]
    }

# Get timezone from coordinates
def get_timezone(lat, lon):
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    return pytz.timezone(tz_name) if tz_name else pytz.utc


def remove_static_rows(df):
    # Identify and remove rows where latitude and longitude remain the same as the previous row
    df["prev_latitude"] = df["latitude"].shift(1)
    df["prev_longitude"] = df["longitude"].shift(1)
    df = df[~((df["latitude"] == df["prev_latitude"]) & (df["longitude"] == df["prev_longitude"]))]
    df = df.drop(columns=["prev_latitude", "prev_longitude"]).reset_index(drop=True)
    return df


def create_full_timeline(df):
    # Create a full second-by-second timeline
    start_time = df["time"].min()
    end_time = df["time"].max()
    full_time_range = pd.date_range(start=start_time, end=end_time, freq="1S")
    return pd.DataFrame({"time": full_time_range})


def merge_and_interpolate(df_full_time, df_cleaned):
    # Merge flight data into the full timeline
    df_merged = df_full_time.merge(df_cleaned, on="time", how="left")

    # Interpolate missing values for latitude, longitude, and geoaltitude
    df_merged["latitude"] = df_merged["latitude"].interpolate(method="linear")
    df_merged["longitude"] = df_merged["longitude"].interpolate(method="linear")
    df_merged["baroaltitude"] = df_merged["baroaltitude"].interpolate(method="linear")

    return df_merged


def remove_outliers(df, column, threshold):
    df = df.copy()
    df["diff"] = df[column].diff().abs()
    df.loc[df["diff"] > threshold, column] = np.nan
    df[column] = df[column].interpolate(method="linear")
    df = df.drop(columns=["diff"])
    return df


def compute_segment_lengths(df):
    segment_lengths = []
    timestamps = []
    for i in range(1, len(df)):
        p1 = df.iloc[i - 1]
        p2 = df.iloc[i]
        horizontal_distance = geodesic((p1["latitude"], p1["longitude"]), (p2["latitude"], p2["longitude"])).meters
        vertical_distance = abs(p2["baroaltitude"] - p1["baroaltitude"]) if not pd.isna(
            p1["baroaltitude"]) and not pd.isna(p2["baroaltitude"]) else 0
        total_distance = np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)
        segment_lengths.append(total_distance)
        timestamps.append(p2["time"])
    return timestamps, segment_lengths


def resample_to_60s(df):
    return df.set_index("time").resample("60S").first().reset_index()


def plot_data(df, timestamps, segment_lengths):
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["baroaltitude"], marker="o", linestyle="-", alpha=0.7,
             label="Altitude Over Time (60s Resampled)")
    plt.title("Altitude vs. Time After 60s Resampling")
    plt.xlabel("Time")
    plt.ylabel("Altitude (meters)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, segment_lengths, marker="o", linestyle="-", alpha=0.7,
             label="Segment Length (3D) - 60s Resampled")
    plt.axhline(y=np.mean(segment_lengths), color='r', linestyle="--",
                label=f"Mean Segment Length: {np.mean(segment_lengths):.2f} m")
    plt.legend()
    plt.title("Segment Length Over Time After 60s Resampling")
    plt.xlabel("Timestamp")
    plt.ylabel("Segment Length (meters)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()






def infer_airports_from_trajectory(df, airports):
    start_point = (df.iloc[0]['latitude'], df.iloc[0]['longitude'])
    end_point = (df.iloc[-1]['latitude'], df.iloc[-1]['longitude'])

    def closest_airport(point):
        return min(
            airports,
            key=lambda a: geodesic(point, (a['latitude'], a['longitude'])).meters
        )

    departure_airport = closest_airport(start_point)['airport']
    arrival_airport = closest_airport(end_point)['airport']
    return departure_airport, arrival_airport

def process_flight(file_path, airports):
    # Construct file name from trajectory details
    # file_name = f"{trajectory['departure_airport'].lower()}_{trajectory['arrival_airport'].lower()}.csv"

    df = pd.read_csv(file_path)

    # Rename to standard format
    df = df.rename(columns={
        'lat': 'latitude',
        'lon': 'longitude',
        'time': 'time'
    })

    # Convert time column
    df['time'] = pd.to_datetime(df['time'])

    # Drop rows missing essential data
    df = df.dropna(subset=['latitude', 'longitude', 'baroaltitude'])

    # Keep only the relevant columns
    required_columns = ['time', 'icao24', 'callsign', 'latitude', 'longitude', 'velocity', 'baroaltitude']
    df = df[required_columns]

    # Rename velocity to match expected column name
    df = df.rename(columns={'velocity': 'groundspeed'})

    # Infer airports
    departure_code, arrival_code = infer_airports_from_trajectory(df, airports)

    # Special case: remove geoaltitude outlier for main_idx_2
    if "main_idx_2" in file_path:
        upper_threshold = df["baroaltitude"].quantile(0.99)
        df = df[df["baroaltitude"] <= upper_threshold]

    df_cleaned = remove_static_rows(df)

    df_full_time = create_full_timeline(df_cleaned)
    df_interpolated = merge_and_interpolate(df_full_time, df_cleaned)
    df_interpolated = remove_outliers(df_interpolated, "baroaltitude", 200)
    df_interpolated = remove_outliers(df_interpolated, "latitude", 0.01)
    df_interpolated = remove_outliers(df_interpolated, "longitude", 0.01)
    df_resampled_60 = resample_to_60s(df_interpolated)
    timestamps, segment_lengths = compute_segment_lengths(df_resampled_60)
    plot_data(df_resampled_60, timestamps, segment_lengths)
    df_resampled_60 = df_resampled_60.rename(columns={'baroaltitude': 'altitude'})
    # Resample using PyContrails Flight
    # fl = Flight(df_interpolated)
    # fl = fl.resample_and_fill(freq="60s", drop=False)
    # print(fl.dataframe['altitude'])
    # Step 5: Plot segment lengths to check for peaks

    df_resampled = df_resampled_60#fl.dataframe  # Confirm this property is correct

    # Get first trajectory point
    start_lat = df_resampled.iloc[0]["latitude"]
    start_lon = df_resampled.iloc[0]["longitude"]

    # Look up airport and timezone
    nearest_airport = find_nearest_airport(start_lat, start_lon)
    departure_code = nearest_airport["iata"]  # or ICAO if needed
    local_tz = get_timezone(start_lat, start_lon)

    # Only process the February 6th daytime flight
    for date in dates_of_interest:
        if date.month == 2 and date.day == 6:
            period = 'Daytime'
            local_departure_time = datetime(2023, 2, 6, 10, 0)
            local_departure = local_tz.localize(local_departure_time)
            departure_utc = local_departure.astimezone(pytz.utc)

            time_deltas = df_resampled['time'].diff().fillna(pd.Timedelta(seconds=0))
            df_resampled['time'] = departure_utc + time_deltas.cumsum()

            # Solar calculations
            df_resampled['cos_sza'] = cosine_solar_zenith_angle(
                longitude=df_resampled['longitude'].values,
                latitude=df_resampled['latitude'].values,
                time=df_resampled['time'].values.astype('datetime64[ns]'),
                theta_rad=orbital_position(df_resampled['time'].values.astype('datetime64[ns]'))
            )

            sun_present = df_resampled['cos_sza'] > 0
            sunlit_fraction = sun_present.mean()

            print(f"Flight {departure_code}-{arrival_code} on {date.strftime('%Y-%m-%d')} ({period}):")
            print(f"  Sunlit fraction: {sunlit_fraction:.2%}")
            print(f"  Status: {'Mostly sunlit' if sunlit_fraction > 0.5 else 'Mostly dark'}")
            print("-" * 50)

            folder = os.path.join("flight_trajectories", "extra_trajectories_yin", "processed_flights",
                                  f"{departure_code.lower()}_{arrival_code.lower()}")
            os.makedirs(folder, exist_ok=True)

            match = re.search(r'(main_idx_\d+)', os.path.basename(file_path))
            idx_tag = f"_{match.group(1)}" if match else ""

            output_file = os.path.join(
                folder,
                f"{departure_code.lower()}_{arrival_code.lower()}{idx_tag}.csv"
            )

            df_resampled.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


dates_of_interest = [
    datetime(2023, 2, 6),
    datetime(2023, 5, 5),
    datetime(2023, 8, 6),
    datetime(2023, 11, 6)
]

airports = [
    {
        "airport": row["iata_code"],
        "latitude": row["latitude_deg"],
        "longitude": row["longitude_deg"]
    }
    for _, row in global_airports.iterrows()
]

# Then loop through files to process them
input_folder = "flight_trajectories/extra_trajectories_yin"

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        print(f"Processing: {file_path}")
        process_flight(file_path, airports)





