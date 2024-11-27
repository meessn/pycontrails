import sys
import csv
from gsp_api_functions import gsp_api_initialize, gsp_api_close, process_single_row_direct

if __name__ == "__main__":
    input_csv_path = sys.argv[1]
    output_csv_path = sys.argv[2]

    # Read input CSV file with the first column as the index
    with open(input_csv_path, mode='r') as input_file:
        reader = csv.DictReader(input_file)
        rows = [row for row in reader]

    if rows:
        engine_model = rows[1]['engine_model']  # Assuming the column is named 'engine_model'
        print(f"Engine Model: {engine_model}")
    else:
        print("No rows found in the input CSV.")

    # Initialize the DLL
    gspdll = gsp_api_initialize(engine_model)

    # Prepare to write the output CSV
    output_headers = ['index', 'PT3', 'TT3', 'TT4', 'specific_humidity_gsp', 'FAR', 'fuel_flow_gsp', 'thrust_gsp']
    with open(output_csv_path, mode='w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(output_headers)  # Write the header row

        # Process each row
        try:
            for row in rows:
                # Extract the index and inputs
                index = row['index']  # Unnamed column for the index
                mach = float(row['mach'])
                specific_humidity = float(row['specific_humidity'])
                air_temperature = float(row['air_temperature'])
                air_pressure = float(row['air_pressure'])
                thrust_per_engine = float(row['thrust_per_engine'])
                war = float(row['WAR'])
                # Process the row using the DLL
                try:
                    output_values = process_single_row_direct(
                        gspdll, mach, specific_humidity, air_temperature, air_pressure, thrust_per_engine, war
                    )
                except Exception as e:
                    print(f"Error processing row with index {index}: {e}")
                    output_values = [None] * 7  # Placeholder for failed rows

                # Write the result to the output CSV
                writer.writerow([index] + output_values)
        finally:
            # Ensure the DLL is properly closed
            gsp_api_close(gspdll)
