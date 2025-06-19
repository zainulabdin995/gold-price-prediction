# Training script 
import os
import sys

# 👇 Dynamically add the root folder to sys.path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)


from utils.train_ml_model import train_ml_model

csv_path = "data/XAU_1h_data.csv"
model_path = "models/1h_RandomForest.pkl"

train_ml_model("random_forest", csv_path, model_path)
