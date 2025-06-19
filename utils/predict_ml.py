import pandas as pd
import numpy as np
import joblib

def predict_ml(csv_path, model_path, scaler_path, steps=1):
    df = pd.read_csv(csv_path, sep=';')
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    predictions = []

    # Start from the last row
    last_row = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1]

    for _ in range(steps):
        # Create input using only the 4 features used for training
        X_input = pd.DataFrame([{
            'Open': last_row['Open'],
            'High': last_row['High'],
            'Low': last_row['Low'],
            'Volume': last_row['Volume']
        }])

        X_scaled = scaler.transform(X_input)
        pred_close = model.predict(X_scaled)[0]
        predictions.append(pred_close)

        # Simulate new row with predicted 'Close'
        last_row['Open'] = last_row['High'] = last_row['Low'] = pred_close
        last_row['Close'] = pred_close  # Update this so future inputs have context

    return predictions
