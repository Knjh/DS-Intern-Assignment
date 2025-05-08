import os
import pandas as pd
import yaml

def load_config(path="src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_ingestion():
    config = load_config()["data_ingestion"]
    source_path = config["source_path"]
    output_dir = config["output_dir"]
    output_filename = config["output_filename"]

    # Read the dataset
    df = pd.read_csv(source_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to output path
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    print(f"Data successfully ingested and saved to: {output_path}")

if __name__ == "__main__":
    run_ingestion()
