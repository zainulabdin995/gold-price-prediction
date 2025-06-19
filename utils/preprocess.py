import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if 'Close' not in df.columns:
        raise ValueError("❌ 'Close' column not found in the data.")

    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df = df.dropna()

    X = df[['MA_5', 'MA_10']]
    y = df['Close']
    return X, y
