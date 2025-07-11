import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import joblib

# Custom LSTM to ignore time_major
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super(CustomLSTM, self).__init__(*args, **kwargs)

def predict_lstm(csv_path, model_path, scaler_path, steps=1, time_steps=60):
    df = pd.read_csv(csv_path, sep=';')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df = df.sort_values('Date' if 'Date' in df.columns else df.columns[0])

    scaler = joblib.load(scaler_path)
    model = load_model(model_path, custom_objects={'LSTM': CustomLSTM})

    # Scale using all 5 columns
    scaled_full = scaler.transform(df)

    if len(scaled_full) < time_steps:
        raise ValueError(f"Require at least {time_steps} rows.")

    # Slice 4 input features: Open, High, Low, Volume
    input_indices = [0, 1, 2, 4]
    last_sequence = scaled_full[-time_steps:, input_indices]
    last_sequence = last_sequence.reshape(1, time_steps, 4)

    predictions = []
    for _ in range(steps):
        pred_scaled_close = model.predict(last_sequence, verbose=0)[0][0]

        # Build full 5-feature scaled row for inverse transform
        last_input_row = last_sequence[0, -1]
        new_scaled_row = np.zeros(5)
        new_scaled_row[0] = last_input_row[0]  # Open
        new_scaled_row[1] = last_input_row[1]  # High
        new_scaled_row[2] = last_input_row[2]  # Low
        new_scaled_row[3] = pred_scaled_close  # Close
        new_scaled_row[4] = last_input_row[3]  # Volume

        # Inverse transform
        unscaled_row = scaler.inverse_transform([new_scaled_row])[0]
        predictions.append(unscaled_row[3])

        # Prepare next input (4 features)
        next_input = new_scaled_row[input_indices]
        last_sequence = np.append(last_sequence[:, 1:, :], [[next_input]], axis=1)

    return predictions