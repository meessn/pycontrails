import pandas as pd
from pycontrails import Flight, MetDataset
from matplotlib import pyplot as plt
flight = 'malaga'
aircraft = 'A20N_full'

df = pd.read_csv(f"{flight}.csv")
df = df.rename(columns={'geoaltitude': 'altitude', 'groundspeed': 'groundspeed', 'timestamp':'time'})
callsign = df['callsign'].dropna().unique()[0]
df['altitude'] = df['altitude']*0.3048 #foot to meters
df['groundspeed'] = df['groundspeed']*0.514444444
attrs = {
    "flight_id" : f"{callsign}",
    "aircraft_type": f"{aircraft}",
    "engine_uid": "01P22PW163"
}
df= df.dropna(subset=['latitude', 'longitude', 'altitude'])
fl = Flight(df, attrs=attrs)

fl.plot_profile(kind="scatter", s=5, figsize=(10, 6))


fl.plot(kind="scatter", s=5, figsize=(10, 6))



fl = fl.resample_and_fill(freq="60s", drop=False)

fl.plot_profile(kind="scatter", s=5, figsize=(10, 6))

fl.plot(kind="scatter", s=5, figsize=(10, 6))


plt.show()