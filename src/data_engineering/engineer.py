import pandas as pd
import numpy as np
import os
import yaml

def load_config(path="src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

def engineer_features(df):
    df_engineered = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_engineered['timestamp']):
        df_engineered['timestamp'] = pd.to_datetime(df_engineered['timestamp'])

    df_engineered = df_engineered.set_index('timestamp')

    # Indoor-Outdoor temperature difference
    for i in range(1, 10):
        col_name = f'zone{i}_temperature'
        if col_name in df_engineered.columns and 'outdoor_temperature' in df_engineered.columns:
            df_engineered[f'zone{i}_temp_diff'] = df_engineered[col_name] - df_engineered['outdoor_temperature']

    # Time-based features
    df_engineered['is_weekend'] = df_engineered.index.weekday.isin([5, 6]).astype(int)
    df_engineered['is_business_hours'] = ((df_engineered.index.hour >= 8) & (df_engineered.index.hour < 18)).astype(int)
    df_engineered['season'] = df_engineered.index.month.map(get_season)
    df_engineered = pd.get_dummies(df_engineered, columns=['season'], prefix='season')

    # Rolling averages
    df_engineered['equipment_energy_24h_avg'] = df_engineered['equipment_energy_consumption'].rolling(window=24).mean()
    df_engineered['lighting_energy_24h_avg'] = df_engineered['lighting_energy'].rolling(window=24).mean()

    # Zone averages
    temp_cols = [f'zone{i}_temperature' for i in range(1, 10) if f'zone{i}_temperature' in df_engineered.columns]
    humid_cols = [f'zone{i}_humidity' for i in range(1, 10) if f'zone{i}_humidity' in df_engineered.columns]
    if temp_cols:
        df_engineered['avg_zone_temperature'] = df_engineered[temp_cols].mean(axis=1)
    if humid_cols:
        df_engineered['avg_zone_humidity'] = df_engineered[humid_cols].mean(axis=1)

    # Lag features
    for lag in [1, 2, 3, 24]:
        df_engineered[f'energy_lag_{lag}'] = df_engineered['equipment_energy_consumption'].shift(lag)

    # Rolling statistics
    df_engineered['energy_rolling_3h_mean'] = df_engineered['equipment_energy_consumption'].rolling(3).mean()
    df_engineered['energy_rolling_24h_std'] = df_engineered['equipment_energy_consumption'].rolling(24).std()

    # Zone interaction features
    if 'zone1_temperature' in df.columns and 'zone1_humidity' in df.columns:
        df_engineered['production_heat_index'] = 0.5 * (df_engineered['zone1_temperature'] + df_engineered['zone1_humidity'])
    if 'zone5_temperature' in df_engineered.columns and 'zone6_temperature' in df_engineered.columns:
        df_engineered['storage_temp_gradient'] = df_engineered['zone5_temperature'] - df_engineered['zone6_temperature']

    df_engineered = df_engineered.bfill().dropna()
    df_engineered = df_engineered.reset_index()
    return df_engineered

def run_engineering():
    config = load_config()["data_engineering"]
    input_path = config["input_path"]
    output_path = config["output_path"]

    df = pd.read_csv(input_path)
    df_engineered = engineer_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_engineered.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Data saved at: {output_path}")

if __name__ == "__main__":
    run_engineering()
