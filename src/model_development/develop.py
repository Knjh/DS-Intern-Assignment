import pandas as pd
import numpy as np
import os
import pickle
import yaml
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

def load_config(path="src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_model(df):
    # Drop NA rows
    df = df.dropna()

    # Convert timestamp to datetime if not already
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Separate features and target
    X = df.drop(columns=['Unnamed: 0', 'timestamp', 'year', 'equipment_energy_consumption', 'season'], errors='ignore')
    y = df['equipment_energy_consumption']

    # Progressive time-based split
    train_size = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    print(f"Train period: {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[train_size].date()}")
    print(f"Test period: {df['timestamp'].iloc[train_size + 1].date()} to {df['timestamp'].iloc[-1].date()}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model initialization and training
    model = DecisionTreeRegressor(
        max_depth=9,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=None
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return model, scaler

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def run_training():
    config = load_config()["model_development"]
    input_path = config["input_path"]
    model_path = config["model_output_path"]
    scaler_path = config["scaler_output_path"]

    df = pd.read_csv(input_path)
    model, scaler = train_model(df)

    save_pickle(model, model_path)
    save_pickle(scaler, scaler_path)

    print(f"Model and scaler saved at:\n   - Model: {model_path}\n   - Scaler: {scaler_path}")

if __name__ == "__main__":
    run_training()
