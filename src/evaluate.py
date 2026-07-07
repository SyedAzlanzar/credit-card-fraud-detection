import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_predictions(y_test: pd.Series, predictions: np.ndarray, model_name: str):
    """
    Evaluates model predictions using a Confusion Matrix and Classification Report.
    """
    print(f"\n=========================================")
    print(f"      EVALUATION: {model_name.upper()}")
    print(f"=========================================")
    
    # 1. Confusion Matrix
    print("\n1. Confusion Matrix:")
    # Format: 
    # [True Negatives (Normal right)  , False Positives (Normal wrong)]
    # [False Negatives (Fraud missed) , True Positives (Fraud caught)]
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    # 2. Classification Report (Precision, Recall, F1)
    print("\n2. Classification Report:")
    # target_names makes the output easy to read instead of just "0" and "1"
    report = classification_report(y_test, predictions, target_names=['Normal (0)', 'Fraud (1)'])
    print(report)

# Master Execution Block
if __name__ == "__main__":
    # Import our pipeline functions
    from src.data_prep import load_data_efficiently, preprocess_and_split
    from src.resampling import balance_training_data
    from src.model import train_models, make_predictions
    
    FILE_PATH = "data/raw/creditcard.csv" 
    
    # --- STEP 1: Data Prep ---
    df = load_data_efficiently(FILE_PATH, is_prototyping=False)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    
    # --- STEP 2: Resample ---
    X_train_res, y_train_res = balance_training_data(X_train, y_train)
    
    # --- STEP 3: Train & Predict ---
    models = train_models(X_train_res, y_train_res)
    log_preds, rf_preds = make_predictions(models, X_test)
    
    # --- STEP 4: Evaluate ---
    evaluate_predictions(y_test, log_preds, "Logistic Regression")
    evaluate_predictions(y_test, rf_preds, "Random Forest")