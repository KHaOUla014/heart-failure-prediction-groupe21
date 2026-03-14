import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib
import sys
sys.path.append('src')
from data_processing import load_data, optimize_memory, preprocess

df = load_data('data/heart_failure_clinical_records_dataset.xls')
df = optimize_memory(df)
X_train, X_test, y_train, y_test = preprocess(df)

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

best_score = 0
best_model = None
best_name = ''

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"{name}: AUC={auc:.3f} | Accuracy={acc:.3f} | F1={f1:.3f} | Precision={precision:.3f} | Recall={recall:.3f}")

    if auc > best_score:
        best_score = auc
        best_model = model
        best_name = name

joblib.dump(best_model, 'src/best_model.pkl')
print(f"\nMeilleur modèle : {best_name} (AUC={best_score:.3f})")
print("Modèle sauvegardé !")