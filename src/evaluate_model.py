from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix,
                             RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import shap
import os
import sys
import pandas as pd

sys.path.append('src')

from data_processing import load_data, optimize_memory, preprocess
from train_model import train_and_select_best_model

def evaluate_all_models():
    """
    Trains ALL models, evaluates each one individually,
    generates confusion matrices, ROC curves, and SHAP plots.
    Saves everything to reports/.
    """

    print("Loading and preparing data...")
    df = load_data('data/heart_failure_clinical_records_dataset.csv')
    df = optimize_memory(df)
    X_train, X_test, y_train, y_test = preprocess(df)

    os.makedirs('reports', exist_ok=True)


    # ─────────────────────────────────────
    # SMOTE IMPACT COMPARISON
    # ─────────────────────────────────────
    print("\n--- SMOTE Impact Analysis ---")
    from sklearn.model_selection import train_test_split
    
    X_raw = df.drop('DEATH_EVENT', axis=1)
    y_raw = df['DEATH_EVENT']
    X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    xgb_no_smote = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_no_smote.fit(X_tr_raw, y_tr_raw)
    y_pred_no  = xgb_no_smote.predict(X_te_raw)
    y_proba_no = xgb_no_smote.predict_proba(X_te_raw)[:, 1]

    xgb_smote = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_smote.fit(X_train, y_train)   # ✅ X_train exists now
    y_pred_sm  = xgb_smote.predict(X_test)
    y_proba_sm = xgb_smote.predict_proba(X_test)[:, 1]

    smote_comparison = pd.DataFrame({
        'Metric': ['Recall', 'F1-Score', 'ROC-AUC'],
        'Without SMOTE': [
            round(recall_score(y_te_raw, y_pred_no), 4),
            round(f1_score(y_te_raw, y_pred_no), 4),
            round(roc_auc_score(y_te_raw, y_proba_no), 4),
        ],
        'With SMOTE': [
            round(recall_score(y_test, y_pred_sm), 4),
            round(f1_score(y_test, y_pred_sm), 4),
            round(roc_auc_score(y_test, y_proba_sm), 4),
        ]
    })

    print(smote_comparison.to_string(index=False))
    smote_comparison.to_csv('reports/smote_comparison.csv', index=False)
    print("SMOTE comparison saved ✅")

    # ─────────────────────────────────────
    # EVALUATE EACH MODEL INDIVIDUALLY
    # ─────────────────────────────────────
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest':       RandomForestClassifier(random_state=42),
        'LightGBM':            LGBMClassifier(random_state=42, verbose=-1),
        'XGBoost':             XGBClassifier(random_state=42, eval_metric='logloss'),
    }

    results = []
    best_auc   = 0
    best_model = None
    best_name  = ""

    print("\n--- Evaluating All Models ---")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc  = roc_auc_score(y_test, y_proba)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        results.append({
            'Model':     name,
            'ROC-AUC':   round(auc,  4),
            'Accuracy':  round(acc,  4),
            'Precision': round(prec, 4),
            'Recall':    round(rec,  4),
            'F1-Score':  round(f1,   4),
        })

        print(f"\n{name}:")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        # Confusion Matrix per model
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=['Survived','Died']).plot(ax=ax, colorbar=False)
        ax.set_title(f'Confusion Matrix — {name}')
        plt.tight_layout()
        filename = name.lower().replace(' ', '_')
        plt.savefig(f'reports/confusion_matrix_{filename}.png')
        plt.clf()
        print(f"  Confusion matrix saved ✅")

        if auc > best_auc:
            best_auc   = auc
            best_model = model
            best_name  = name

    # ─────────────────────────────────────
    # SAVE RESULTS TABLE TO CSV
    # ─────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/model_comparison.csv', index=False)
    print("\n--- Model Comparison ---")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name} (AUC={best_auc:.4f})")
    print("Results saved to reports/model_comparison.csv ✅")

    # ─────────────────────────────────────
    # ROC CURVE — ALL MODELS ON ONE CHART
    # ─────────────────────────────────────
    print("\nGenerating ROC curves...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Retrain to plot all on same chart
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)

    ax.set_title('ROC Curves — All Models')
    ax.plot([0,1],[0,1],'k--', label='Random baseline')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('reports/roc_curves_all_models.png')
    plt.clf()
    print("ROC curves saved to reports/roc_curves_all_models.png ✅")

    # ─────────────────────────────────────
    # SHAP SUMMARY — BEST MODEL
    # ─────────────────────────────────────
    print("\nGenerating SHAP summary plot...")
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, show=False)
    else:
        shap.summary_plot(shap_values, X_test, show=False)

    plt.title(f'SHAP Summary — {best_name}')
    plt.tight_layout()
    plt.savefig('reports/shap_summary_plot.png')
    plt.clf()
    print("SHAP summary saved to reports/shap_summary_plot.png ✅")

    return {
        'best_model': best_model,
        'best_name':  best_name,
        'results':    results_df,
    }

    
if __name__ == "__main__":
    evaluate_all_models()
