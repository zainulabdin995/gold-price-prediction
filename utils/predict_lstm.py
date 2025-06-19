import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def predict_lstm(csv_path, model_path, scaler_path, steps=1, time_steps=60):
    df = pd.read_csv(csv_path, sep=';')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df = df.sort_values('Date' if 'Date' in df.columns else df.columns[0])

    scaler = joblib.load(scaler_path)
    
    # ✅ FIX: Avoid loading incompatible configs like time_major
    model = load_model(model_path, compile=False)

    # Scale using all 5 columns for proper inverse transformation
    scaled_full = scaler.transform(df)

    if len(scaled_full) < time_steps:
        raise ValueError(f"Require at least {time_steps} rows.")

    # Slice only required 4 input features for prediction: ['Open', 'High', 'Low', 'Volume']
    input_indices = [0, 1, 2, 4]  # Ignore 'Close'
    last_sequence = scaled_full[-time_steps:, input_indices]
    last_sequence = last_sequence.reshape(1, time_steps, 4)

    predictions = []

    for _ in range(steps):
        # Predict next 'Close' in scaled form
        pred_scaled_close = model.predict(last_sequence, verbose=0)[0][0]

        # Build full 5-feature scaled row (needed to inverse transform)
        last_input_row = last_sequence[0, -1]
        new_scaled_row = np.zeros(5)
        new_scaled_row[input_indices[0]] = last_input_row[0]  # Open
        new_scaled_row[input_indices[1]] = last_input_row[1]  # High
        new_scaled_row[input_indices[2]] = last_input_row[2]  # Low
        new_scaled_row[3] = pred_scaled_close                 # Close
        new_scaled_row[input_indices[3]] = last_input_row[3]  # Volume

        # Inverse transform to get actual Close price
        unscaled_row = scaler.inverse_transform([new_scaled_row])[0]
        predictions.append(unscaled_row[3])  # Real 'Close' price

        # Prepare next input (only 4 features again)
        next_input = new_scaled_row[input_indices]
        last_sequence = np.append(last_sequence[:, 1:, :], [[next_input]], axis=1)

    return predictions
