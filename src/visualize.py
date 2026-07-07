import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay

from src.data_prep import load_data_efficiently, preprocess_and_split


def generate_performance_plots(model_path: str, X_test, y_test, output_filename: str):
    """
    Loads a saved model, generates evaluation plots, and saves them as an image.
    """
    print(f"Loading model from: {model_path}...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("Error: Could not find the saved model. Did you run model.py first?")
        return

    print("Generating plots...")
    # Create a wide figure to hold two side-by-side graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Plot the Precision-Recall Curve
    display_pr = PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test, ax=ax1, color="darkorange", name="Random Forest"
    )
    ax1.set_title("Precision-Recall Curve")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Plot the Confusion Matrix
    display_cm = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax2,
        cmap="Blues",
        display_labels=["Normal (0)", "Fraud (1)"],
    )
    ax2.set_title("Confusion Matrix")

    plt.tight_layout()

    # Save the plot to your reports folder
    save_path = f"reports/{output_filename}"
    plt.savefig(save_path, dpi=300)  # dpi=300 makes it high-resolution
    print(f"Success! Visualizations saved to: {save_path}")


if __name__ == "__main__":
    FILE_PATH = "data/raw/creditcard.csv"
    MODEL_PATH = "models/random_forest_fraud_model.joblib"

    df = load_data_efficiently(FILE_PATH, is_prototyping=True)
    _, X_test, _, y_test = preprocess_and_split(df)

    print("\n--- Starting Visualization Phase ---")
    generate_performance_plots(MODEL_PATH, X_test, y_test, "rf_performance.png")
