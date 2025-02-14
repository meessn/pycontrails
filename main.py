from main_emissions import run_emissions
from main_climate import run_climate
import os

import threading
import time
import win32gui
import win32con
import win32api
import win32process
import pyautogui


def close_error_window():
    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if "Fout" in window_title:
                print(f"[Error Handler] Found error window: {window_title}")
                bring_window_to_front(hwnd)

                time.sleep(0.5)  # Let it gain focus

                # Click center or press Enter (choose what works better)
                pyautogui.press('enter')  # or click if needed
                print(f"[Error Handler] Pressed Enter to close: {window_title}")

    win32gui.EnumWindows(callback, None)


def bring_window_to_front(hwnd):
    try:
        foreground_hwnd = win32gui.GetForegroundWindow()
        current_thread_id = win32api.GetCurrentThreadId()
        target_thread_id, _ = win32process.GetWindowThreadProcessId(hwnd)
        foreground_thread_id, _ = win32process.GetWindowThreadProcessId(foreground_hwnd)

        if current_thread_id != target_thread_id:
            win32process.AttachThreadInput(current_thread_id, target_thread_id, True)
        if foreground_thread_id != target_thread_id:
            win32process.AttachThreadInput(foreground_thread_id, target_thread_id, True)

        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)

        # Detach input threads again
        if current_thread_id != target_thread_id:
            win32process.AttachThreadInput(current_thread_id, target_thread_id, False)
        if foreground_thread_id != target_thread_id:
            win32process.AttachThreadInput(foreground_thread_id, target_thread_id, False)

        print(f"[Window Handler] Forced window '{win32gui.GetWindowText(hwnd)}' to front.")

    except Exception as e:
        print(f"[Window Handler] Failed to bring window to front: {e}")


def monitor_error_window():
    while True:
        close_error_window()
        time.sleep(2)


# Start this in the background
error_monitor_thread = threading.Thread(target=monitor_error_window, daemon=True)
error_monitor_thread.start()

print("Background error monitor started.")


# Root directory containing flight trajectories
root_dir = "flight_trajectories/processed_flights"
malaga_flight_path = "malaga.csv"

# Select which trajectories to simulate
flight_trajectories_to_simulate = {
    "bos_fll": True,  # Example of processing other flights
    "cts_tpe": True,
    "dus_tos": True,
    "gru_lim": True,
    "hel_kef": True,
    "lhr_ist": True,
    "sfo_dfw": True,
    "sin_maa": True,
    "malaga": False
}

# Debug flag: Set to True to process only **one** flight for testing
process_one_flight_only = False

# Time bounds for different flight dates
time_bounds_dict = {
    "2023-02-06": ("2023-02-05 14:00", "2023-02-07 11:00"),
    "2023-05-05": ("2023-05-04 14:00", "2023-05-06 11:00"),
    "2023-08-06": ("2023-08-05 14:00", "2023-08-07 11:00"),
    "2023-11-06": ("2023-11-05 14:00", "2023-11-07 11:00"),
    "malaga": ("2024-06-07 9:00", "2024-06-08 02:00")
}

# Engine models to run
engine_models = {
    "GTF1990": False,
    "GTF2000": False,
    "GTF": False,
    "GTF2035": False,
    "GTF2035_wi": True
}

# SAF values based on engine model
saf_dict = {
    "SAF20": True,
    "SAF100": True
}

prediction = "mees"
weather_model = "era5model"


# Function to process a flight file
def process_flight(trajectory, flight_file, flight_path):
    file_parts = flight_file.split("_")

    # Handle Malaga separately
    if trajectory == "malaga":
        flight_date = "malaga"
        diurnal = "day"  # Assuming it's a daytime flight, adjust if needed
    else:
        flight_date = file_parts[2]  # Extract YYYY-MM-DD
        flight_time = file_parts[3].split(".")[0]  # Extract daytime or nighttime
        diurnal = "day" if flight_time == "daytime" else "night"

    time_bounds = time_bounds_dict.get(flight_date, ("Unknown Start", "Unknown End"))

    # Run for each engine model
    for engine_model, run_engine in engine_models.items():
        if not run_engine:
            continue

        # Determine SAF values
        saf_values = [0] # hier normaal 0!!!!!
        if engine_model in ("GTF2035", "GTF2035_wi"):
            if saf_dict["SAF20"]:
                saf_values.append(20)
            if saf_dict["SAF100"]:
                saf_values.append(100)

        # Determine water injection values
        water_injection = [0, 0, 0]
        if engine_model == "GTF2035_wi":
            water_injection = [15, 15, 15]

        for SAF in saf_values:
            print(f"Running emissions for: {flight_file}, Engine: {engine_model}, SAF: {SAF}")
            run_emissions(trajectory, flight_path, engine_model, water_injection, SAF, aircraft="A20N_full",
                          time_bounds=time_bounds)

            print(f"Running climate model for: {flight_file}, Engine: {engine_model}, SAF: {SAF}")
            run_climate(trajectory, flight_path, engine_model, water_injection, SAF, aircraft="A20N_full",
                        time_bounds=time_bounds, prediction=prediction, diurnal=diurnal, weather_model=weather_model)


# Process standard flight directories
for trajectory, should_simulate in flight_trajectories_to_simulate.items():
    if not should_simulate or trajectory == "malaga":
        continue

    trajectory_path = os.path.join(root_dir, trajectory)

    if not os.path.exists(trajectory_path):
        print(f"Warning: {trajectory_path} does not exist. Skipping.")
        continue

    flight_files = [f for f in os.listdir(trajectory_path) if f.endswith(".csv")]

    if process_one_flight_only:
        flight_files = flight_files[:1]  # Take only the first flight file :1

    for flight_file in flight_files:
        process_flight(trajectory, flight_file, os.path.join(trajectory_path, flight_file))

    if process_one_flight_only:
        print("Processed one flight only. Exiting for debug mode.")
        break


if flight_trajectories_to_simulate["malaga"] and os.path.exists(malaga_flight_path):
    print(f"Processing Malaga flight: {malaga_flight_path}")
    process_flight("malaga", "malaga.csv", malaga_flight_path)
