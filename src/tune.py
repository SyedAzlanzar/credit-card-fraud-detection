from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Import our pipeline functions
from src.data_prep import load_data_efficiently, preprocess_and_split
from src.resampling import balance_training_data

def tune_random_forest(X_train_resampled, y_train_resampled):
    """
    Uses RandomizedSearchCV to find the best hyperparameters for Random Forest.
    Optimized for laptop processing by limiting n_iter.
    """
    print("\n--- Starting Hyperparameter Tuning ---")
    
    # 1. Define the base model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 2. Define the "Grid" of settings to test
    param_distributions = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Testing combinations of the following settings:")
    print(param_distributions)
    
    # 3. Set up the Randomized Search
    # n_iter=10 means it will randomly pick 10 combinations to test (saves time)
    # cv=3 means 3-fold cross-validation (tests each combination 3 times to be sure)
    # scoring='f1' tells it to pick the model that best balances Precision and Recall
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=10, 
        cv=3,
        scoring='f1',
        n_jobs=-1, # Use all CPU cores
        random_state=42,
        verbose=2  # Prints progress to the terminal
    )
    
    # 4. Run the search on the training data
    print("\nRunning Randomized Search (This might take a minute or two...)")
    random_search.fit(X_train_resampled, y_train_resampled)
    
    # 5. Output the results
    print("\n--- Tuning Complete! ---")
    print(f"Best F1 Score Achieved: {random_search.best_score_:.4f}")
    print("Best Parameters Found:")
    for param, value in random_search.best_params_.items():
        print(f" - {param}: {value}")
        
    return random_search.best_estimator_

if __name__ == "__main__":
    FILE_PATH = "data/raw/creditcard.csv" 
    
    # 1. Load and prepare a small chunk of data for speed testing
    df = load_data_efficiently(FILE_PATH, is_prototyping=True)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    X_train_res, y_train_res = balance_training_data(X_train, y_train)
    
    # 2. Run the tuner!
    best_rf_model = tune_random_forest(X_train_res, y_train_res)