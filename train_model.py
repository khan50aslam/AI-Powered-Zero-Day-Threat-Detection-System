# In train_model.py

import pandas as pd
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np # Import numpy

# --- Configuration ---
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# !!! ACTION REQUIRED: Replace 'cicids2017_preprocessed.csv' with the exact name
#                      of the dataset file you put in the 'data' folder.
DATA_PATH = 'data/cicids2017_preprocessed.csv'
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# Using new names to avoid overwriting models trained on sample data
MODEL_SAVE_PATH = 'models/trained_model_cicids.pkl'
SCALER_SAVE_PATH = 'models/scaler_cicids.pkl'

# --- Load Data ---
print(f"Loading data from {DATA_PATH}...")
try:
    # You might need to add options if your CSV is structured differently
    # e.g., pd.read_csv(DATA_PATH, sep=',')
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    # Print all column names to help you choose features
    print("--------------------------------------------------")
    print("Available columns in the dataset:")
    print(df.columns.tolist())
    print("--------------------------------------------------")
    print("Dataset shape (rows, columns):", df.shape)
    print("--------------------------------------------------")
except FileNotFoundError:
    print(f"Error: Dataset file not found at {DATA_PATH}")
    print("Please make sure the filename is correct and it's inside the 'data' folder.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- Feature Selection ---
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# !!! ACTION REQUIRED: EDIT THIS LIST CAREFULLY!
# Choose the numerical features from YOUR CIC-IDS2017 dataset you want to use.
# Look at the 'Available columns' printed above.
# - Include only columns with numbers.
# - Exclude identifier columns (like Flow ID, Source IP, Destination IP, Port numbers, Timestamp).
# - Exclude any text-based columns.
# - CRITICAL: Exclude the 'Label' column if it exists (this tells the answer!).
# The example list below is based on common CIC-IDS2017 features,
# YOU MUST ADJUST IT TO MATCH YOUR SPECIFIC FILE.
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

print(f"Attempting to use the following {len(FEATURE_COLUMNS)} features:")
print(FEATURE_COLUMNS)
print("--------------------------------------------------")

# --- Preprocessing ---
# Select only the chosen features
try:
    X = df[FEATURE_COLUMNS].copy()
    print(f"Successfully selected {X.shape[1]} features.")
except KeyError as e:
    print(f"Error selecting features: One or more columns listed in FEATURE_COLUMNS were not found in the dataset.")
    print(f"Missing column(s): {e}")
    print("Please check the spelling and capitalization in the FEATURE_COLUMNS list against the 'Available columns' printed above.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during feature selection: {e}")
    exit()

# Handle potential non-numeric issues and infinite values BEFORE scaling
# Convert all selected columns to numeric, coercing errors to NaN
print("Converting features to numeric type...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Replace infinite values (positive or negative) with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check for and handle any remaining NaN values (missing data)
if X.isnull().values.any():
    print("Warning: Missing values (NaN) found in features after conversion/inf replacement.")
    # Option 1: Fill with median (often better for skewed data)
    print("Filling missing values with the median of each column.")
    X.fillna(X.median(), inplace=True)
    # Option 2: Fill with mean
    # print("Filling missing values with the mean of each column.")
    # X.fillna(X.mean(), inplace=True)
    # Option 3: Drop rows with NaNs (can lose a lot of data)
    # print("Dropping rows with missing values.")
    # X.dropna(inplace=True)

    # Verify NaNs are handled
    if X.isnull().values.any():
         print("Error: Still found NaN values after attempting to fill. Check data.")
         exit()
    else:
         print("Missing values handled.")
else:
    print("No missing values found in selected features.")

print("--------------------------------------------------")

# Scale the features
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled successfully.")
print("--------------------------------------------------")

# --- Model Training ---
print("Training Isolation Forest model...")
# Adjust model parameters if needed:
# - contamination: Expected proportion of anomalies ('auto' or float like 0.01). 'auto' is often a good start.
# - n_estimators: Number of trees (e.g., 100, 200). More trees can be better but slower.
# - max_samples: Number/fraction of samples to draw for each tree.
# - random_state: For reproducibility.
# - n_jobs=-1: Use all available CPU cores for faster training.
model = IForest(contamination=0.1, random_state=42, n_jobs=-1)
model.fit(X_scaled)
print("Model training complete.")
print("--------------------------------------------------")

# --- Save Model and Scaler ---
print(f"Saving model to {MODEL_SAVE_PATH}")
joblib.dump(model, MODEL_SAVE_PATH)

print(f"Saving scaler to {SCALER_SAVE_PATH}")
joblib.dump(scaler, SCALER_SAVE_PATH)

print("--------------------------------------------------")
print("Script finished. Model and Scaler saved successfully.")
print("--------------------------------------------------")