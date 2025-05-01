# In app/detector.py
from pyod.models.iforest import IForest
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np # Import numpy

# --- Configuration ---
# Use the paths for the model/scaler trained on CIC-IDS2017
MODEL_PATH = 'models/trained_model_cicids.pkl'
SCALER_PATH = 'models/scaler_cicids.pkl'

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# !!! ACTION REQUIRED: Replace the list below with the EXACT list you finalized
#                      and used successfully in train_model.py!
FEATURE_COLUMNS = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Length of Fwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'FIN Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'Average Packet Size',
    # 'Fwd Header Length.1', # Often a duplicate, check if needed/exists
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward',
    'Active Mean',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Max',
    'Idle Min'
    # NOTE: Bulk rate/bytes/packets columns are often zero, you might exclude them later if they don't help.
] 
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# !!! ACTION REQUIRED: Customize this list with columns from your dataset
#                      that you want to SEE on the dashboard.
# Look at the 'Available columns' printed when you ran train_model.py.
# Choose columns like IPs, Ports, Protocol, Timestamp, etc.
DISPLAY_COLUMNS = [
    'Destination Port'
    # Add any other identifiers you want to display, like 'Flow ID'
    # Ensure these names EXACTLY match columns in your CSV data!
]   
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


def detect_threats(data):
    """
    Detects threats in new incoming data (expected to match CIC-IDS2017 structure).
    Args:
        data (list of dicts): List containing dictionaries, where each dict represents a log entry.
    Returns:
        list of dicts: Processed data including selected display columns,
                         'anomaly_score', and 'is_threat'. Returns empty list on error.
    """
    if not data:
        print("Detector received no data.")
        return []

    print(f"Detector received {len(data)} log entries.")

    try:
        df = pd.DataFrame(data)
        print(f"DataFrame created with columns: {df.columns.tolist()}")


        # --- Preprocessing ---
        # 1. Select display columns (handle missing ones gracefully)
        original_data_for_display = {}
        found_display_cols = []
        for col in DISPLAY_COLUMNS:
            if col in df.columns:
                original_data_for_display[col] = df[col]
                found_display_cols.append(col)
            else:
                print(f"Warning: Display column '{col}' not found in incoming data. Skipping.")
                # Optionally add placeholder: original_data_for_display[col] = pd.Series([None] * len(df))
        print(f"Selected display columns: {found_display_cols}")


        # 2. Select and order FEATURES for the model
        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            print(f"Error: Missing required feature columns in input data: {missing_features}")
            print(f"Available columns in input: {df.columns.tolist()}")
            return []

        features_df = df[FEATURE_COLUMNS].copy() # Ensure correct order
        print(f"Selected {len(FEATURE_COLUMNS)} features for model.")

        # 3. Handle potential non-numeric issues and infinite values BEFORE scaling
        print("Converting features to numeric for detection...")
        for col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if features_df.isnull().values.any():
            print("Warning: NaN values found in incoming features. Filling with 0.")
            # Using 0 is simple; using the training median/mean saved from scaler would be more robust
            features_df.fillna(0, inplace=True)


        # --- Load Scaler and Model ---
        print(f"Loading scaler from {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        print(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

        # --- Scale and Predict ---
        # 4. Scale the new data using the LOADED scaler
        print("Scaling incoming features...")
        features_scaled = scaler.transform(features_df)

        # 5. Make predictions and get scores
        print("Predicting anomalies and calculating scores...")
        predictions = model.predict(features_scaled)       # 0 for normal, 1 for anomaly
        scores = model.decision_function(features_scaled)  # Raw anomaly scores

        # --- Combine Results ---
        print("Combining results...")
        # Create results DataFrame starting with display columns
        results_df = pd.DataFrame(original_data_for_display)

        # Add anomaly scores and prediction results
        results_df['anomaly_score'] = scores
        results_df['is_threat'] = predictions

        # Convert back to list of dictionaries for the API/template
        final_results = results_df.to_dict(orient='records')
        print(f"Detection complete. Returning {len(final_results)} results.")
        return final_results

    except FileNotFoundError:
        print(f"FATAL ERROR: Model or Scaler file not found.")
        print(f"Looked for: {MODEL_PATH} and {SCALER_PATH}")
        return []
    except KeyError as e:
        # This error might occur if expected columns are missing during DataFrame creation/selection
        print(f"Error processing data: Missing expected column {e}.")
        print(f"Columns available in input data: {df.columns.tolist()}")
        return []
    except ValueError as e:
         # This might happen if data types are wrong or NaNs weren't handled properly before scaling/prediction
         print(f"Error during scaling or prediction. Check data consistency/types: {e}")
         return []
    except Exception as e:
        # Catch other potential errors
        import traceback
        print(f"An unexpected error occurred during threat detection:")
        print(traceback.format_exc()) # Print detailed traceback
        return []