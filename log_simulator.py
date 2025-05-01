# log_simulator.py (Place in the root zero_day_detector folder)

import pandas as pd
import requests # To send HTTP requests
import json
import time
import random
import numpy as np # Import numpy

# URL of your running Flask application's API endpoint
FLASK_API_URL = "http://127.0.0.1:5000/api/submit_log" # Default Flask dev server address

# --- Configuration ---
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# !!! ACTION REQUIRED: Replace 'cicids2017_preprocessed.csv' with the exact name
#                      of your dataset file in the 'data' folder.
LOG_DATA_PATH = "data/cicids2017_preprocessed.csv"
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# How many logs to send in each batch
BATCH_SIZE = 15 # Increase/decrease as needed

# Delay between sending batches (in seconds)
SEND_INTERVAL = 3 # Increase/decrease as needed

def simulate_log_stream():
    print(f"--- Log Simulator ---")
    print(f"Attempting to read log data from: {LOG_DATA_PATH}")
    try:
        # Read the entire CSV. For very large files (> RAM),
        # you might read in chunks, but this is usually okay for CIC-IDS2017.
        df = pd.read_csv(LOG_DATA_PATH)

        # IMPORTANT: Handle problematic values BEFORE converting to JSON
        # Replace infinite values with NaN first
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Convert entire DataFrame rows to list of dictionaries
        # Using fillna('') converts remaining NaN to empty string for better JSON compatibility.
        # If your detector handles None better, use df.fillna(None) instead.
        print("Converting DataFrame to list of dictionaries (this may take a moment)...")
        logs = df.fillna('').to_dict(orient='records')
        print(f"Loaded and converted {len(logs)} log entries.")
        if not logs:
             print("Warning: No logs loaded. Check the CSV file path and content.")
             return

    except FileNotFoundError:
        print(f"Error: Log data file not found at {LOG_DATA_PATH}")
        print("Please ensure the path and filename are correct.")
        return
    except Exception as e:
        print(f"Error reading or converting log data: {e}")
        import traceback
        print(traceback.format_exc())
        return

    print(f"Starting simulation...")
    print(f"Target API Endpoint: {FLASK_API_URL}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Send Interval: {SEND_INTERVAL} seconds")
    print("--------------------------------------------------")

    index = 0
    total_sent = 0
    while True:
        # Get a batch of logs
        batch = []
        start_index = index
        for _ in range(BATCH_SIZE):
            if index >= len(logs):
                print("End of log data reached. Restarting from beginning...")
                index = 0 # Loop back to the beginning
            log_entry = logs[index]
            batch.append(log_entry)
            index += 1

        if not batch:
            print("Warning: Batch is empty. Waiting...")
            time.sleep(SEND_INTERVAL)
            continue

        # Send the batch to the Flask API
        try:
            print(f"Sending batch of {len(batch)} logs (Indices {start_index}-{index-1})...")
            # Send as JSON payload, set timeout for the request
            response = requests.post(FLASK_API_URL, json=batch, timeout=20) # Increased timeout
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            total_sent += len(batch)
            print(f"Batch sent successfully. Response: {response.json()}. Total sent: {total_sent}")

        except requests.exceptions.Timeout:
             print("Error: Request timed out. Server might be busy or unresponsive.")
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to Flask app at {FLASK_API_URL}.")
            print(f"Is the Flask app (main.py) running?")
            print(f"Will retry in {SEND_INTERVAL * 2} seconds...")
            time.sleep(SEND_INTERVAL * 2) # Wait longer if connection fails
            continue # Skip the regular wait and try connecting again
        except requests.exceptions.RequestException as e:
            print(f"Error sending logs: {e}")
            # Consider adding more robust error handling/retry logic here
            # If the server returns an error (like 400 or 500), it will be caught here too
            try:
                 print(f"Server Response Content: {response.text}") # Try to print server error message
            except NameError:
                 pass # response object might not exist if connection failed early
            print(f"Will retry in {SEND_INTERVAL} seconds...")


        # Wait before sending the next batch
        time.sleep(SEND_INTERVAL)


if __name__ == "__main__":
    # Simple prompt to ensure Flask app is running before starting
    print("--- Before starting the simulator ---")
    print("1. Make sure your Flask application ('python main.py') is running in a separate terminal.")
    print(f"2. Make sure the Flask app is accessible at {FLASK_API_URL}")
    input("Press Enter to start the log simulator...")
    simulate_log_stream()