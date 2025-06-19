import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.train_lstm import train_lstm
from utils.train_ml_model import train_random_forest, train_xgboost

# Define model training jobs
model_jobs = [
    # ---- 5–30 Minutes ----
    ("data/XAU_1m_data.csv", "models/1m_LSTM.h5", "LSTM"),
    ("data/XAU_1m_data.csv", "models/1m_RF.pkl", "RF"),
    ("data/XAU_1m_data.csv", "models/1m_XGB.pkl", "XGB"),

    ("data/XAU_5m_data.csv", "models/5m_LSTM.h5", "LSTM"),
    ("data/XAU_5m_data.csv", "models/5m_RF.pkl", "RF"),
    ("data/XAU_5m_data.csv", "models/5m_XGB.pkl", "XGB"),

    # ---- 1–4 Hours ----
    ("data/XAU_1h_data.csv", "models/1h_LSTM.h5", "LSTM"),
    ("data/XAU_1h_data.csv", "models/1h_RF.pkl", "RF"),
    ("data/XAU_1h_data.csv", "models/1h_XGB.pkl", "XGB"),

    ("data/XAU_4h_data.csv", "models/4h_LSTM.h5", "LSTM"),
    ("data/XAU_4h_data.csv", "models/4h_RF.pkl", "RF"),
    ("data/XAU_4h_data.csv", "models/4h_XGB.pkl", "XGB"),

    # ---- Tomorrow ----
    ("data/XAU_1d_data.csv", "models/1d_LSTM.h5", "LSTM"),
    ("data/XAU_1d_data.csv", "models/1d_RF.pkl", "RF"),
    ("data/XAU_1d_data.csv", "models/1d_XGB.pkl", "XGB"),

    # ---- Next Week ----
    ("data/XAU_1w_data.csv", "models/1w_LSTM.h5", "LSTM"),
    ("data/XAU_1w_data.csv", "models/1w_RF.pkl", "RF"),
    ("data/XAU_1w_data.csv", "models/1w_XGB.pkl", "XGB"),

    # ---- Next Month ----
    ("data/XAU_1Month_data.csv", "models/1mo_LSTM.h5", "LSTM"),
    ("data/XAU_1Month_data.csv", "models/1mo_RF.pkl", "RF"),
    ("data/XAU_1Month_data.csv", "models/1mo_XGB.pkl", "XGB"),
]
# Loop and train each model
for csv_path, model_path, model_type in model_jobs:
    print(f"\n🔧 Training {model_type} on {csv_path}...")
    try:
        if model_type == "LSTM":
            train_lstm(csv_path, model_path)
        elif model_type == "RF":
            train_random_forest(csv_path, model_path)
        elif model_type == "XGB":
            train_xgboost(csv_path, model_path)
    except Exception as e:
        print(f"❌ Failed to train {model_type} on {csv_path}: {e}")
