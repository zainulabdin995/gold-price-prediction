import os
import sys
# Add root path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.predict_ml import predict_ml
from utils.predict_lstm import predict_lstm


# Cache configuration
st.set_page_config(page_title="Gold Price Dashboard", layout="wide")

# Cache data loading
@st.cache_data
def load_data(csv_path, rows_to_display):
    df = pd.read_csv(csv_path, sep=';', usecols=['Date', 'Close', 'High', 'Low', 'Volume'], 
                     parse_dates=['Date'], engine='c')
    return df.sort_values('Date').tail(rows_to_display)

# Cache model predictions
@st.cache_resource
def get_predictions(model_path, scaler_path, csv_path, steps, model_type):
    if model_type == 'LSTM':
        return predict_lstm(model_path=model_path, scaler_path=scaler_path, 
                           csv_path=csv_path, steps=steps)
    return predict_ml(model_path=model_path, scaler_path=scaler_path, 
                     csv_path=csv_path, steps=steps)

# Configuration
TIMEFRAMES = {
    '1 Minute': '1m', '5 Minutes': '5m', '1 Hour': '1h', 
    '4 Hours': '4h', '1 Day': '1d', '1 Week': '1w', '1 Month': '1mo'
}

TIME_INCREMENTS = {
    '1m': timedelta(minutes=1), '5m': timedelta(minutes=5), '1h': timedelta(hours=1),
    '4h': timedelta(hours=4), '1d': timedelta(days=1), '1w': timedelta(weeks=1),
    '1mo': timedelta(days=30)
}

MODEL_TYPES = {
    'LSTM': {'ext': 'h5', 'suffix': 'LSTM'},
    'RandomForest': {'ext': 'pkl', 'suffix': 'RF'},
    'XGBoost': {'ext': 'pkl', 'suffix': 'XGB'}
}

# UI Components
st.title("📈 Gold Price Prediction Dashboard")
st.subheader("📊 Historical Trend and Forecast")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    timeframe = st.selectbox("Select Timeframe", list(TIMEFRAMES.keys()))
with col2:
    model_type = st.selectbox("Select Model Type", list(MODEL_TYPES.keys()))
with col3:
    pred_steps = st.slider("Future Predictions", min_value=1, max_value=30, value=1)

suffix = TIMEFRAMES[timeframe]
model_info = MODEL_TYPES[model_type]
model_path = f"models/{suffix}_{model_info['suffix']}.{model_info['ext']}"
scaler_path = f"models/{suffix}_{model_info['suffix']}_scaler.pkl"
csv_path = f"data/XAU_{suffix}_data.csv"

# Data loading
rows_to_display = st.slider("Recent Data Points to Display", min_value=20, max_value=200, value=100, step=10)
try:
    df = load_data(csv_path, rows_to_display)
    previous_close = df['Close'].iloc[-1]
    st.session_state['df'] = df
except Exception as e:
    st.warning(f"Could not load data: {e}")
    df = None
    previous_close = None

# Plotting
if df is not None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines+markers', 
                            name='Close Price', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['High'] - df['Low'], mode='lines', 
                            name='Volatility (High-Low)', line=dict(color='lightblue')))
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], 
                        name='Volume', marker=dict(color='lightgray'), yaxis='y2'))

    fig.update_layout(
        height=700,
        title=f"Gold Price Chart ({timeframe}) with Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis2=dict(title="Volume", overlaying='y', side='right', showgrid=False),
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        template='plotly_white'
    )

# Predictions
predictions = []
pred_times = []
if st.button("Predict"):
    try:
        predictions = get_predictions(model_path, scaler_path, csv_path, pred_steps, model_type)
        
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
        
        if previous_close and df is not None:
            last_date = df['Date'].iloc[-1]
            interval = TIME_INCREMENTS[suffix]
            pred_times = [last_date + interval * (i + 1) for i in range(len(predictions))]
            
            trend = "📈 Up" if predictions[-1] > previous_close else "📉 Down"
            st.success(f"Predicted Close Price (Next {pred_steps} steps): ${predictions[-1]:.2f}")
            st.info(f"Trend vs Last Close (${previous_close:.2f}): {trend}")
            st.write("📌 Prediction Timestamp:", pred_times[-1].strftime('%Y-%m-%d %H:%M:%S'))
            
            # Display predictions
            st.dataframe(pd.DataFrame({
                'Timestamp': pred_times,
                'Predicted Close': predictions
            }), use_container_width=True)
            
            # Log predictions
            log_data = pd.DataFrame([[datetime.now(), timeframe, model_type, predictions[-1]]],
                                    columns=['Timestamp', 'Timeframe', 'Model', 'PredictedClose'])
            os.makedirs("logs", exist_ok=True)
            log_path = "logs/prediction_logs.csv"
            log_data.to_csv(log_path, mode='a', 
                           header=not os.path.exists(log_path), index=False)
            
            # Add forecast to plot
            forecast_dates = [df['Date'].iloc[-1]] + pred_times
            forecast_prices = [df['Close'].iloc[-1]] + predictions
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, mode='lines+markers',
                                    name='Forecast', line=dict(color='red', dash='dash')))
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# Display chart
if df is not None:
    st.plotly_chart(fig, use_container_width=True)

# Raw data expander
with st.expander("📄 Show Raw Data"):
    if df is not None:
        st.dataframe(df.tail(200), use_container_width=True)
    else:
        st.warning("Data not available.")