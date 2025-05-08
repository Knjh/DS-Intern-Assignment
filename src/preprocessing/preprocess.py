import pandas as pd
import numpy as np
import os
import yaml

def load_config(path="src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_negative_to_nan(series):
    return series.where(series >= 0, np.nan)

def handle_outliers(df, column, strategy='winsorize', lower_bound=None, upper_bound=None, iqr_multiplier=1.5, create_indicator=True):
    df_result = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    if strategy == 'winsorize':
        lower = q1 - (iqr_multiplier * iqr)
        upper = q3 + (iqr_multiplier * iqr)
    elif strategy == 'cap':
        assert lower_bound is not None and upper_bound is not None, "Bounds must be specified"
        lower = lower_bound
        upper = upper_bound
    else:
        lower = q1 - (iqr_multiplier * iqr)
        upper = q3 + (iqr_multiplier * iqr)

    if create_indicator:
        df_result[f'{column}_outlier'] = ((df_result[column] < lower) | (df_result[column] > upper)).astype(int)

    if strategy == 'winsorize' or strategy == 'cap':
        df_result[column] = df_result[column].clip(lower, upper)
    elif strategy == 'remove':
        df_result = df_result[(df_result[column] >= lower) & (df_result[column] <= upper)]

    return df_result

def run_preprocessing():
    config = load_config()["preprocessing"]
    input_path = config["input_path"]
    output_path = config["output_path"]

    df = pd.read_csv(input_path)

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Fix numeric columns stored as objects
    object_cols = [
        'equipment_energy_consumption', 'lighting_energy',
        'zone1_temperature', 'zone1_humidity', 'zone2_temperature'
    ]
    for col in object_cols:
        df[col] = df[col].astype(str).str.replace('[^\d.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace negative values
    columns_to_clean = [col for col in df.columns if 'humidity' in col.lower()] + ['wind_speed', 'visibility_index']
    df[columns_to_clean] = df[columns_to_clean].apply(set_negative_to_nan)

    # Apply outlier handling
    df = handle_outliers(df, 'equipment_energy_consumption', strategy='winsorize', iqr_multiplier=3)

    for zone in range(1, 7):
        df = handle_outliers(df, f'zone{zone}_temperature', strategy='cap', lower_bound=-5.0, upper_bound=50.0)
        df = handle_outliers(df, f'zone{zone}_humidity', strategy='cap', lower_bound=0.0, upper_bound=100.0)

    df = handle_outliers(df, 'visibility_index', strategy='winsorize', iqr_multiplier=2.0)
    df = handle_outliers(df, 'atmospheric_pressure', strategy='winsorize', iqr_multiplier=1.5)

    # Forward/backward fill for time series
    for zone in range(1, 10):
        for t_col in [f'zone{zone}_temperature', f'zone{zone}_humidity']:
            if t_col in df.columns:
                df[t_col] = df[t_col].ffill().bfill()

    num_cols = ['lighting_energy', 'atmospheric_pressure', 'wind_speed', 'dew_point', 'visibility_index']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in ['outdoor_temperature', 'outdoor_humidity']:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Drop unnecessary columns
    drop_cols = [c for c in df.columns if 'outlier' in c or 'random_variable' in c]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Cleaned data saved at: {output_path}")

if __name__ == "__main__":
    preprocess()
