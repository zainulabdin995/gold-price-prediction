import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

def predict_lstm(model_path, scaler_path, csv_path, steps=1):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(csv_path, sep=';')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Include all 5 features
    df = df.dropna()

    # Scale using the full 5 features
    scaled = scaler.transform(df.values)

    # Use last 60 timesteps (adjust if your model uses different input length)
    last_sequence = scaled[-60:].reshape(1, 60, 5)

    predictions = []

    for _ in range(steps):
        pred_scaled = model.predict(last_sequence, verbose=0)[0][0]
        predictions.append(pred_scaled)

        # Append predicted Close to sequence with placeholder values
        last_row = last_sequence[0, -1]
        next_input = np.array([[
            pred_scaled,  # Open
            pred_scaled,  # High
            pred_scaled,  # Low
            pred_scaled,  # Close
            last_row[4]   # Volume (reuse last volume)
        ]])

        last_sequence = np.append(last_sequence[:, 1:, :], [next_input], axis=1)

    # Inverse scale only the predicted Close column
    dummy_input = np.zeros((len(predictions), 5))
    dummy_input[:, 3] = predictions  # Set to 'Close' column
    inversed = scaler.inverse_transform(dummy_input)

    return inversed[:, 3].tolist()
