import os
import pandas as pd

# Notice we import from 'src' because this file lives in the root directory!
from src.data_prep import load_data_efficiently, preprocess_and_split
from src.resampling import balance_training_data
from src.model import train_models, make_predictions, save_model
from src.evaluate import evaluate_predictions
from src.visualize import generate_performance_plots

# --- CONFIGURATION ---
DATA_PATH = "data/raw/creditcard.csv"
MODEL_SAVE_NAME = "random_forest_fraud_model"
MODEL_LOAD_PATH = f"models/{MODEL_SAVE_NAME}.joblib"
PLOT_SAVE_NAME = "rf_performance.png"

# Set to True for fast testing, False for the real deal
PROTOTYPE_MODE = False


def main():
    print("==============================================")
    print("   CREDIT CARD FRAUD DETECTION PIPELINE")
    print("==============================================")

    # 1. Data Ingestion & Preprocessing
    print("\n[STEP 1] Loading and Preprocessing Data...")
    df = load_data_efficiently(DATA_PATH, is_prototyping=PROTOTYPE_MODE)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    # 2. Resampling (Addressing Imbalance)
    print("\n[STEP 2] Balancing the Training Data...")
    X_train_res, y_train_res = balance_training_data(X_train, y_train)

    # 3. Model Training
    print("\n[STEP 3] Training Machine Learning Models...")
    models = train_models(X_train_res, y_train_res)
    log_reg, rf_model = models

    # 4. Evaluation via Terminal
    print("\n[STEP 4] Evaluating Models...")
    log_preds, rf_preds = make_predictions(models, X_test)
    evaluate_predictions(y_test, log_preds, "Logistic Regression")
    evaluate_predictions(y_test, rf_preds, "Random Forest")

    # 5. Saving Artifacts
    print("\n[STEP 5] Saving Best Model (Random Forest)...")
    save_model(rf_model, MODEL_SAVE_NAME)

    # 6. Visualization
    print("\n[STEP 6] Generating Final Reports...")
    generate_performance_plots(MODEL_LOAD_PATH, X_test, y_test, PLOT_SAVE_NAME)

    print("\n==============================================")
    print("   PIPELINE EXECUTION COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()
