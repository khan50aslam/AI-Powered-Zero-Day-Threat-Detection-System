# In app/routes.py
from flask import Blueprint, render_template, jsonify, request # Added request
# Import your actual detector function
from .detector import detect_threats
# Import pandas if needed for any data handling (optional here, detector handles it)
# import pandas as pd
# Use deque for efficient fixed-size storage of recent logs
from collections import deque
import datetime # Keep datetime if needed, maybe for logging received time

app = Blueprint('routes', __name__)

# Store the most recent logs processed by the detector (adjust size as needed)
# This acts as a simple in-memory store for the dashboard
MAX_LOGS_DISPLAYED = 100
# deque automatically removes oldest items when maxlen is reached
detected_logs = deque(maxlen=MAX_LOGS_DISPLAYED)

# --- REMOVE the old simulated data function ---
# def generate_sample_data(): ... (Delete this whole function)

# --- NEW Endpoint to RECEIVE data ---
@app.route('/api/submit_log', methods=['POST'])
def submit_log():
    """
    API endpoint to receive new log data (e.g., from a simulator).
    Expects JSON data in the format: [{"col1": val1, ...}, {"col1": val2, ...}]
    """
    # Log received time
    received_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{received_time}] Received request at /api/submit_log")

    if not request.is_json:
        print("Error: Request received is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        log_data = request.get_json() # Gets the list of dicts
        print(f"Received {len(log_data)} log entries in JSON payload.")

        if not isinstance(log_data, list):
             # If a single log entry dict is sent, wrap it in a list for the detector
             if isinstance(log_data, dict):
                 log_data = [log_data]
                 print("Single log entry received, wrapped in list.")
             else:
                 print("Error: JSON payload is not a list or dictionary.")
                 return jsonify({"error": "JSON payload must be a list of log entries (dictionaries)"}), 400

        # --- Call your actual detection function ---
        # This is where the magic happens!
        print("Calling threat detector...")
        results = detect_threats(log_data) # Process the submitted logs

        # Add the processed results (including scores) to our stored deque
        # extendleft adds multiple items to the left (beginning) efficiently
        if results:
             print(f"Detector returned {len(results)} processed entries. Adding to display queue.")
             detected_logs.extendleft(results)
        else:
             print("Detector returned no results (or an error occurred).")


        return jsonify({"status": "success", "processed_count": len(results)}), 200

    except Exception as e:
        # Catch potential errors during JSON parsing or other issues
        import traceback
        print(f"Error processing /api/submit_log request:")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error processing request."}), 500


# --- MODIFIED Dashboard Route ---
@app.route('/')
def index():
    """
    Displays the dashboard with the latest detected logs from our deque.
    """
    # Pass the current list of detected logs (from newest to oldest) to the template
    # Convert deque to simple list for rendering
    current_logs_list = list(detected_logs)
    print(f"Serving dashboard with {len(current_logs_list)} logs.")
    return render_template('dashboard.html', logs=current_logs_list)


# --- MODIFIED API Route (Optional but good practice) ---
@app.route('/api/threats')
def api_threats():
    """
    Returns the latest detected logs as JSON.
    """
    # Convert deque to list for JSON serialization
    current_logs_list = list(detected_logs)
    print(f"Serving API /api/threats with {len(current_logs_list)} logs.")
    return jsonify(current_logs_list)