import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from src.data_prep import load_data_efficiently, preprocess_and_split

def balance_training_data(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Balances the training data using a combination of Undersampling and SMOTE.
    Optimized for memory efficiency.
    """
    print("\n--- Starting Resampling Process ---")
    original_fraud = y_train.sum()
    original_normal = len(y_train) - original_fraud
    print(f"Original Normal transactions: {original_normal}")
    print(f"Original Fraud transactions: {original_fraud}")
    
    # Undersample the majority class (Normal)
    # sampling_strategy=0.1 means: shrink Normal transactions until Frauds equal 10% of the Normal count.
    under = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    
    # Oversample the minority class (Fraud) using SMOTE
    # sampling_strategy=1.0 means: generate synthetic Frauds until they equal 100% of the Normal count.
    over = SMOTE(sampling_strategy=1.0, random_state=42)
    
    # Chain them together in an Imblearn Pipeline
    steps = [('undersample', under), ('smote', over)]
    pipeline = Pipeline(steps=steps)
    
    print("Applying Undersampling and SMOTE...")
    # Apply ONLY to the training data
    X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)
    
    new_fraud = y_train_resampled.sum()
    new_normal = len(y_train_resampled) - new_fraud
    
    print("\n--- Resampling Complete ---")
    print(f"New Normal transactions: {new_normal}")
    print(f"New Fraud transactions: {new_fraud}")
    print(f"Resampled Training data shape: {X_train_resampled.shape}")
    
    return X_train_resampled, y_train_resampled

if __name__ == "__main__":
    FILE_PATH = "data/raw/creditcard.csv" 
    
    # 1. Load & Split
    df = load_data_efficiently(FILE_PATH, is_prototyping=True)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    
    # 2. Resample
    X_train_res, y_train_res = balance_training_data(X_train, y_train)