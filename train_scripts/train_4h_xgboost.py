# Training script 
import os
import sys

# 👇 Dynamically add the root folder to sys.path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)


from utils.train_ml_model import train_ml_model

csv_path = "data/XAU_4h_data.csv"
model_path = "models/4h_XGBoost.pkl"

train_ml_model("xgboost", csv_path, model_path)
