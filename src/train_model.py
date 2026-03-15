import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Import our custom functions from data_processing.py
from data_processing import load_data, optimize_memory, preprocess

def train_and_select_best_model():
    # 1. Load and prepare the data
    print("Loading and optimizing data...")
    df = load_data('data/heart_failure_clinical_records_dataset.csv')
    df = optimize_memory(df)
    
    print("\nPreprocessing data (Splitting and applying SMOTE)...")
    X_train, X_test, y_train, y_test = preprocess(df)

    # 2. Initialize the models required by the project
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    best_score = 0
    best_model = None
    best_name = ""

    # 3. Train each model and find the best one using ROC-AUC
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # We use predict_proba for ROC-AUC to get probability scores
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"Trained {name} - Validation AUC: {auc:.4f}")

        # Keep track of the best model
        if auc > best_score:
            best_score = auc
            best_model = model
            best_name = name
    
    print(f"  BEST MODEL: {best_name}")
    print(f"   AUC Score: {best_score:.4f}")

    return best_model, X_train, X_test, y_train, y_test

    

if __name__ == "__main__":
    train_and_select_best_model()