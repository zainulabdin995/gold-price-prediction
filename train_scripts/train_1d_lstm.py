# train_1d_lstm.py
import os
import sys

# 👇 Dynamically add the root folder to sys.path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from utils.train_lstm import train_lstm  # ✅ Now should work!

# 🔁 Update with your actual file path
csv_path = "data/XAU_1d_data.csv"
model_path = "models/1d_LSTM.h5"

train_lstm(csv_path, model_path)
