import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib

def preprocess_ml_data(csv_path, use_last_n_rows=100_000):
    print(f"🔄 Reading: {csv_path}")
    df = pd.read_csv(csv_path, sep=';')
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Use only last 100k rows if available
    if len(df) > use_last_n_rows:
        df = df.tail(use_last_n_rows)
        print(f"✅ Using last {use_last_n_rows} rows.")
    else:
        print(f"⚠️ Only {len(df)} rows available. Using all.")

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler

def train_random_forest(csv_path, model_path):
    X, y, scaler = preprocess_ml_data(csv_path)

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)

    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"🌲 RandomForest RMSE: {rmse:.4f}")

    joblib.dump(model, model_path)
    joblib.dump(scaler, model_path.replace(".pkl", "_scaler.pkl"))
    print(f"✅ Saved: {model_path}")

def train_xgboost(csv_path, model_path):
    X, y, scaler = preprocess_ml_data(csv_path)

    model = XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    model.fit(X, y)

    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"⚡ XGBoost RMSE: {rmse:.4f}")

    joblib.dump(model, model_path)
    joblib.dump(scaler, model_path.replace(".pkl", "_scaler.pkl"))
    print(f"✅ Saved: {model_path}")
