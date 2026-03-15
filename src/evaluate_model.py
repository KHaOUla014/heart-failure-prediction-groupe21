from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import shap
import os
import sys

sys.path.append('src')

from data_processing import load_data, preprocess
from train_model import train_and_select_best_model

def evaluate_saved_model():
    print("Training model and getting test data...")
    
    # Get model AND test data directly from function
    model, X_train, X_test, y_train, y_test = train_and_select_best_model()
    
    # Create reports folder if it doesn't exist
    os.makedirs('reports', exist_ok=True)

    print("\n--- Final Model Evaluation Metrics ---")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate all metrics
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

    # ROC CURVE — saved to reports/
    print("\nGenerating ROC curve...")
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f'ROC Curve (AUC = {auc:.4f})')
    plt.tight_layout()
    plt.savefig('reports/roc_curve.png')
    plt.clf()
    print("ROC curve saved to reports/roc_curve.png ")

    # SHAP SUMMARY PLOT — saved to reports/
    print("\nGenerating SHAP summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, show=False)
    else:
        shap.summary_plot(shap_values, X_test, show=False)

    plt.tight_layout()
    plt.savefig('reports/shap_summary_plot.png')
    plt.clf()
    print("SHAP summary plot saved to reports/shap_summary_plot.png ")

    # Return metrics
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
