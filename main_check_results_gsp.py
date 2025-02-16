import os
import re
import threading
import pandas as pd
import time
import win32gui
import win32con
import win32api
import win32process
import pyautogui
from main_emissions import run_emissions
from main_climate import run_climate


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
# Define the root directory containing the emissions outputs
root_results_dir = 'main_results_figures/results/'

# Columns that indicate a faulty run if they contain NaN values
critical_columns = ['PT3', 'TT3', 'FAR', 'specific_humidity_gsp', 'fuel_flow_gsp', 'thrust_gsp', 'W3']

completely_faulty_files = []

# Walk through the results directory
for root, dirs, files in os.walk(root_results_dir):
    for file in files:
        if file.endswith('.csv') and 'emissions' in root:
            csv_path = os.path.join(root, file)

            try:
                df = pd.read_csv(csv_path)

                # Check if all values in any of the critical columns are NaN
                all_nan_columns = df[critical_columns].isna().all()
                if all_nan_columns.any():
                    faulty_cols = all_nan_columns[all_nan_columns].index.tolist()
                    print(f"Completely faulty file detected: {csv_path}, Faulty columns: {faulty_cols}")
                    completely_faulty_files.append(csv_path)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

print("\n=== Summary of Completely Faulty Files ===")
print(f"Total completely faulty files found: {len(completely_faulty_files)}")

if completely_faulty_files:
    for file in completely_faulty_files:
        print(file)

# Time bounds for different flight dates
time_bounds_dict = {
    "2023-02-06": ("2023-02-05 14:00", "2023-02-07 11:00"),
    "2023-05-05": ("2023-05-04 14:00", "2023-05-06 11:00"),
    "2023-08-06": ("2023-08-05 14:00", "2023-08-07 11:00"),
    "2023-11-06": ("2023-11-05 14:00", "2023-11-07 11:00"),
    "malaga": ("2024-06-07 9:00", "2024-06-08 02:00")
}
# Rerun emissions and climate for faulty files
prediction = "mees"
weather_model = "era5model"

for file_path in completely_faulty_files:
    normalized_path = file_path.replace("\\", "/")  # Normalize slashes for regex
    match = re.search(r'results/(.+?)/(.+?)/emissions/(.+?)_SAF_(\d+)_A20N_full_WAR_(\d+)_(\d+)_(\d+).csv', normalized_path)

    if match:
        trajectory = match.group(1)
        flight = match.group(2)
        engine_model = match.group(3)
        SAF = int(match.group(4))
        WAR_values = [int(match.group(5)), int(match.group(6)), int(match.group(7))]
        aircraft = "A20N_full"

        flight_path = f"flight_trajectories/processed_flights/{trajectory}/{flight}.csv"

        flight_date_match = re.search(r'_(\d{4}-\d{2}-\d{2})_', flight)
        if flight_date_match:
            flight_date = flight_date_match.group(1)
            time_bounds = time_bounds_dict.get(flight_date)
        else:
            print(f"[Warning] Could not extract date from flight: {flight}, skipping...")
            continue

        diurnal = "day" if "daytime" in flight else "night"

        print(f"\n--- Rerunning: {trajectory}/{flight} | Engine: {engine_model} | SAF: {SAF} | WAR: {WAR_values} ---")
        print(f"Flight Path: {flight_path}")
        print(f"Time Bounds: {time_bounds}")
        print(f"Diurnal: {diurnal}")

        run_emissions(trajectory, flight_path, engine_model, WAR_values, SAF, aircraft, time_bounds)
        run_climate(
            trajectory, flight_path, engine_model, WAR_values, SAF, aircraft,
            time_bounds, prediction, diurnal, weather_model
        )
    else:
        print(f"[Error] Could not parse file path: {file_path}")