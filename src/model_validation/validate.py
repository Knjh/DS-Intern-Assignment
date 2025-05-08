import pandas as pd
import pickle
import yaml
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

def load_config(path="src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def validate_model(df, model, scaler):
    df = df.dropna()

    # Convert timestamp to datetime if not already
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Features and target
    X = df.drop(columns=['Unnamed: 0', 'timestamp', 'year', 'equipment_energy_consumption', 'season'], errors='ignore')
    y = df['equipment_energy_consumption']

    # Progressive time split
    train_size = int(len(df) * 0.7)
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    # Scale test set
    X_test_scaled = scaler.transform(X_test)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R² score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

def run_validation():
    config = load_config()["model_validation"]
    df = pd.read_csv(config["input_path"])
    model = load_pickle(config["model_path"])
    scaler = load_pickle(config["scaler_path"])

    validate_model(df, model, scaler)

if __name__ == "__main__":
    run_validation()
