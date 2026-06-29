import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_data_efficiently(file_path: str, is_prototyping: bool = True) -> pd.DataFrame:
    """Loads the Kaggle dataset with memory-efficient data types."""
    dtypes = {'Time': 'float32', 'Amount': 'float32', 'Class': 'int8'}
    for i in range(1, 29):
        dtypes[f'V{i}'] = 'float32'
    
    if is_prototyping:
        print("Prototyping Mode: Loading the first 30,000 rows...")
        df = pd.read_csv(file_path, dtype=dtypes, nrows=30000)
    else:
        print("Full Run Mode: Loading the entire dataset...")
        df = pd.read_csv(file_path, dtype=dtypes)
        
    return df

def preprocess_and_split(df: pd.DataFrame):
    """Splits the data and scales the Time and Amount columns."""
    print("Splitting data into features (X) and target (y)...")
    
    # 1. Separate Features (X) and Target (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 2. Train/Test Split (80% training, 20% testing)
    # stratify=y ensures the 20% test set has the exact same ratio of frauds as the training set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling Time and Amount columns with RobustScaler...")
    # 3. Scale the features (Fit ONLY on training data to prevent leakage)
    scaler = RobustScaler()
    
    # We only scale Time and Amount. The other columns are already PCA transformed.
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    
    return X_train, X_test, y_train, y_test

# Quick test block
if __name__ == "__main__":
    FILE_PATH = "data/raw/creditcard.csv" 
    
    try:
        # Load the data
        df = load_data_efficiently(FILE_PATH, is_prototyping=True)
        
        # Split and Scale
        X_train, X_test, y_train, y_test = preprocess_and_split(df)
        
        print("\n--- Preprocessing Complete ---")
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        print(f"Number of frauds in Training set: {y_train.sum()}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {FILE_PATH}.")