from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import sys

sys.path.append('src')

from data_processing import load_data, preprocess
from train_model import train_and_select_best_model

def evaluate_saved_model():
    print("Training model and getting test data...")
    
    # Get model AND test data directly from function — no pkl needed
    model, X_train, X_test, y_train, y_test = train_and_select_best_model()
    
    print("\n--- Final Model Evaluation Metrics ---")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate all required metrics
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print results
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Return metrics so other files can use them
    return {
        'model': model,
        'AUC': auc,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1
    }

if __name__ == "__main__":
    evaluate_saved_model()