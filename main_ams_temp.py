from main_emissions import run_emissions
from main_climate import run_climate

trajectory = "ams_bcn"
flight_path = "flight_trajectories/extra_trajectories_yin/processed_flights/ams_bcn/ams_bcn_main_idx_12.csv"

engine_model = "GTF"
water_injection = [0, 0, 0]
SAF = 0
aircraft = "A20N_full"
time_bounds = ("2023-02-05 14:00", "2023-02-07 11:00")  # for Feb 6
prediction = "mees"
weather_model = "era5model"
accuracy = None
diurnal = "day"

# Run emissions
print(f"Running emissions for {flight_path}")
run_emissions(
    trajectory=trajectory,
    flight_path=flight_path,
    engine_model=engine_model,
    water_injection=water_injection,
    SAF=SAF,
    aircraft=aircraft,
    time_bounds=time_bounds
)

# Run climate
print(f"Running climate model for {flight_path}")
run_climate(
    trajectory=trajectory,
    flight_path=flight_path,
    engine_model=engine_model,
    water_injection=water_injection,
    SAF=SAF,
    aircraft=aircraft,
    time_bounds=time_bounds,
    prediction=prediction,
    diurnal=diurnal,
    weather_model=weather_model,
    accuracy=accuracy
)
