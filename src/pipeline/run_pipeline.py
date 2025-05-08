import os
from src.data_ingestion.ingest import run_ingestion
from src.preprocessing.preprocess import run_preprocessing
from src.data_engineering.engineer import run_engineering
from src.model_development.develop import run_training
from src.model_validation.validate import run_validation

def create_artifact_dirs():
    os.makedirs("artifacts/raw", exist_ok=True)
    os.makedirs("artifacts/preprocessed", exist_ok=True)
    os.makedirs("artifacts/engineered_data", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)

def main():
    print("Starting Energy Consumption Prediction Pipeline...\n")

    create_artifact_dirs()

    print("Step 1: Data Ingestion")
    run_ingestion()

    print("\nStep 2: Data Preprocessing")
    run_preprocessing()

    print("\nStep 3: Feature Engineering")
    run_engineering()

    print("\nStep 4: Model Development")
    run_training()

    print("\nStep 5: Model Validation")
    run_validation()

    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()
