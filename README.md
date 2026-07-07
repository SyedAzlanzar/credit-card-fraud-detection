# Credit Card Fraud Detection: Resampling & Processing Engine
> *This repository is currently under active development. Data ingestion, resampling pipelines, and baseline machine learning classifications are complete. Currently staging for hyperparameter tuning and model evaluation.*

A modular, memory-efficient Python pipeline engineered to process and balance highly skewed financial datasets. Designed specifically for low-resource hardware constraints, this project implements robust data ingestion, feature scaling, advanced algorithmic resampling (SMOTE), and baseline classification to detect fraudulent credit card transactions.

---
## 📅 Project Roadmap
This project is being built progressively to ensure high code quality and mathematical rigor at each step.

* [x] **Day 1: System Architecture & Ingestion** — Structuring the repository, optimizing memory usage (`float64` to `float32`), and implementing leakage-proof Train/Test splits.
* [x] **Day 2: The Equalizer** — Handling the 319:1 class imbalance using Imbalanced-Learn pipelines (Targeted Undersampling + SMOTE).
* [x] **Day 3: Baseline Modeling (Current)** — Building, training, and serializing Logistic Regression and Random Forest classifiers on the balanced data.
* [x] **Day 4: Evaluation & Tuning (Current)** — Hyperparameter tuning via Randomized Search and generating Matplotlib Precision-Recall matrices.
* [ ] **Day 5: Orchestration & Deployment** — Finalizing the `main.py` pipeline and deploying the saved `.joblib` model via a lightweight API.

---
## 🔗 System Components
* 💾 **[Data Prep Module](src/data_prep.py)**: Memory-optimized data loader, train/test allocator, and outlier-resistant feature scaler.
* ⚖️ **[Resampling Module](src/resampling.py)**: Mathematical class equalizer utilizing synthetic generation and targeted downsampling.
* 🧠 **[Model Engine](src/model.py)**: The core machine learning training, prediction, and serialization module.
* ⚙️ **[Optimization & Tuning](src/tune.py)**: Automated Randomized Search to mathematically optimize tree algorithms.
* 📦 **[Requirements](requirements.txt)**: Strict dependency tracker ensuring environment reproducibility.

---
## 🏗️ System Architecture Flow (Current State)

```mermaid
flowchart TD
    User((Data Engineer)) -->|Executes Scripts| Pipeline[Processing Pipeline]
    
    subgraph Data Ingestion Layer [src/data_prep.py]
        Pipeline -->|Reads CSV via Pandas| RawData[Kaggle Dataset: 284k rows]
        RawData -->|Memory Optimization| MemOpt[Downcast to float32]
        MemOpt -->|Train/Test Split| Split[80/20 Separation]
        Split -->|Isolate test data| TestSet[(X_test, y_test: Locked)]
        Split -->|RobustScaler| Scaler[Scaled Training Data]
    end
    
    subgraph Resampling Engine [src/resampling.py]
        Scaler -->|Task: Balance Classes| Imbalance{319:1 Skew}
        Imbalance -->|Reduce Majority Class| Under[Random Undersampling]
        Imbalance -->|Synthesize Minority Class| Over[SMOTE Oversampling]
    end

    subgraph Machine Learning Engine [src/model.py]
        Under & Over -->|Balanced Data| ML[Classifier Training]
        ML --> Tune[Randomized Search CV]
        Tune -->|Best Params| RF[Optimized Random Forest]
        RF -->|Serialize State| Joblib[(random_forest_fraud_model.joblib)]
    end

    subgraph Evaluation & Reporting
        Joblib & TestSet --> Eval[Scikit-Learn Metrics]
        Joblib & TestSet --> Viz[Matplotlib Engine]
        Eval --> Output1[\Terminal: Recall, Precision, F1/]
        Viz --> Output2[(reports/rf_performance.png)]
    end

```

---

## ✨ Key Features (Implemented)

* **Memory-Optimized Ingestion:** Programmatically downcasts natively heavy 64-bit float arrays to 32-bit and 8-bit integers, effectively cutting RAM consumption in half to support laptop-grade computation.
* **Leakage-Proof Architecture:** Enforces strict separation of training and testing data matrices *prior* to any scaling or resampling, completely preventing future target leakage.
* **Hybrid Resampling Strategy:** Combines targeted deletion of majority "Normal" transactions with SMOTE (Synthetic Minority Over-sampling) to artificially generate highly realistic "Fraud" points, perfectly balancing a skewed dataset.
* **Automated Hyperparameter Tuning: Implements Scikit-Learn's RandomizedSearchCV to automatically find the mathematical "sweet spot" for ensemble depth and splitting rules, optimized specifically for F1-Score.
* **Business-Centric Evaluation:** Abandons standard "Accuracy" in favor of Precision, Recall, and Confusion Matrices to directly measure the real-world trade-off between catching fraud and minimizing customer false alarms.
* **Visual Reporting:** Automatically plots and saves high-resolution Precision-Recall curves and Confusion Matrices using Matplotlib.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Numerical Processing:** NumPy & Pandas
* **Preprocessing & Modeling:** Scikit-Learn (Algorithms, Scaling, Splitting)
* **Class Balancing:** Imbalanced-Learn (SMOTE & Undersampling)
* **Artifact Persistence:** Joblib
* **Visualization:** Matplotlib

---

## 🚀 Getting Started

**1. Clone the repository:**

```bash
git clone <your-repository-url>
cd FRAUD-DETECTION

```

**2. Configure Environment & Dependencies:**
Ensure you have a virtual environment (`.venv`) activated, then install dependencies:

```bash
pip install -r requirements.txt

```

**3. Acquire the Raw Data:**
Download the "Credit Card Fraud Detection" dataset from Kaggle and place `creditcard.csv` directly into the `data/raw/` directory.

**4. Run Commands (CLI Execution):**

*Task 1 - Memory-Efficient Data Loading & Scaling:*

```bash
python src/data_prep.py

```

*Task 2 - Class Balancing (SMOTE & Undersampling):*

```bash
python src/resampling.py

```

*Task 3 - Train Classifiers and Save Model Artifacts:*

```bash
python src/model.py

```
*Task 4 - Tune, Evaluate, and Visualize:*

```bash
python -m src.tune
python src/evaluate.py
python src/visualize.py

```

---

## 🧩 Core Architecture Modules

* **`models/`**: Dedicated local repository for saved `.joblib` binary artifacts. *(Ignored via .gitignore)*
* **`reports/`**: Output directory for generated `.png` graphs and visualizations.
* **`src/tune.py`**: Executes Randomized Search Cross-Validation to mathematically extract the optimal hyperparameter dictionary.
* **`src/evaluate.py`**: Cross-references isolated test data against model predictions to generate terminal-based Precision/Recall matrices.
* **`src/visualize.py`**: Loads serialized `.joblib` models to generate exportable Matplotlib visualizations.