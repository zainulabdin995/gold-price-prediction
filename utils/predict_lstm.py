import pandas as pd
import numpy as np
from tensorflow.keras.models import model_from_json
import joblib
import h5py

# ✅ Custom model loader to avoid `time_major=False` config issue
def load_legacy_lstm_model(h5_path):
    with h5py.File(h5_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError("Model config not found in the H5 file.")
        model = model_from_json(model_config if isinstance(model_config, str) else model_config.decode('utf-8'))
        model.load_weights(h5_path)
    return model

def predict_lstm(csv_path, model_path, scaler_path, steps=1, time_steps=60):
    # Load and preprocess CSV
    df = pd.read_csv(csv_path, sep=';')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df = df.sort_values('Date' if 'Date' in df.columns else df.columns[0])

    # Load scaler and model
    scaler = joblib.load(scaler_path)
    model = load_legacy_lstm_model(model_path)

    # Scale all 5 features for proper inverse transformation
    scaled_full = scaler.transform(df)

    if len(scaled_full) < time_steps:
        raise ValueError(f"Require at least {time_steps} rows, got {len(scaled_full)}.")

    # Use only 4 input features: Open, High, Low, Volume (not Close)
    input_indices = [0, 1, 2, 4]
    last_sequence = scaled_full[-time_steps:, input_indices]
    last_sequence = last_sequence.reshape(1, time_steps, 4)

    predictions = []

    for _ in range(steps):
        pred_scaled_close = model.predict(last_sequence, verbose=0)[0][0]

        last_input_row = last_sequence[0, -1]
        new_scaled_row = np.zeros(5)
        new_scaled_row[input_indices[0]] = last_input_row[0]  # Open
        new_scaled_row[input_indices[1]] = last_input_row[1]  # High
        new_scaled_row[input_indices[2]] = last_input_row[2]  # Low
        new_scaled_row[3] = pred_scaled_close                 # Close
        new_scaled_row[input_indices[3]] = last_input_row[3]  # Volume

        # Inverse scale to get actual Close price
        unscaled_row = scaler.inverse_transform([new_scaled_row])[0]
        predictions.append(unscaled_row[3])  # Close

        # Update last_sequence with new input
        next_input = new_scaled_row[input_indices]
        last_sequence = np.append(last_sequence[:, 1:, :], [[next_input]], axis=1)

    return predictions
