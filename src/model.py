import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import our previous pipelines
from src.data_prep import load_data_efficiently, preprocess_and_split
from src.resampling import balance_training_data

def train_models(X_train_resampled, y_train_resampled):
    """Trains Logistic Regression and Random Forest models on the balanced data."""
    print("\n--- Training Models ---")
    
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_resampled, y_train_resampled)
    
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
       n_estimators=100,         # Tuned
        max_depth=10,             # Tuned
        min_samples_split=5,      # Tuned
        min_samples_leaf=1,       # Tuned
        random_state=42, 
        n_jobs=-1      
    )
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    print("Models trained successfully!")
    return log_reg, rf_model

def make_predictions(models, X_test):
    """Generates predictions using the trained models on the UNTOUCHED test data."""
    log_reg, rf_model = models
    
    print("\n--- Making Predictions on Test Data ---")
    log_preds = log_reg.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    
    print("Predictions generated successfully!")
    return log_preds, rf_preds

def save_model(model, filename: str):
    """
    Saves the trained model to the models/ directory.
    """
    os.makedirs("models", exist_ok=True)
    
    file_path = f"models/{filename}.joblib"
    joblib.dump(model, file_path)
    print(f"Model successfully saved to: {file_path}")

if __name__ == "__main__":
    FILE_PATH = "data/raw/creditcard.csv" 
    
    # 1. Load, Split, and Resample (Prototyping mode for speed)
    df = load_data_efficiently(FILE_PATH, is_prototyping=True)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    X_train_res, y_train_res = balance_training_data(X_train, y_train)
    
    # 2. Train the models
    models = train_models(X_train_res, y_train_res)
    log_reg, rf_model = models
    
    # 3. Save the model (Random Forest)
    print("\n--- Saving Artifacts ---")
    save_model(rf_model, "random_forest_fraud_model")