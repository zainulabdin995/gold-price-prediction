# Training script 
import os
import sys

# 👇 Dynamically add the root folder to sys.path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)


from utils.train_lstm import train_lstm

csv_path = "data/XAU_1w_data.csv"
model_path = "models/1w_LSTM.h5"

train_lstm(csv_path, model_path)
