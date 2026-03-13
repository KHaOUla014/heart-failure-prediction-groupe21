<div align="center">

# 🫀 Heart Failure Risk Predictor

**Explainable Machine Learning for Clinical Decision Support**

<p align="center">
  <a href="https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records"><img src="https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-blue?style=flat-square" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square&logo=python" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Framework-Streamlit-red?style=flat-square&logo=streamlit" /></a>
  <a href="#"><img src="https://img.shields.io/badge/XAI-SHAP-orange?style=flat-square" /></a>
  <a href="#"><img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?style=flat-square&logo=github" /></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-purple?style=flat-square" /></a>
</p>

<p align="center">
  <b>École Centrale Casablanca — Coding Week, March 2026</b>
</p>

</div>

---

## JIRA time line organisation

- **[2026-03-15]** Project delivered — full pipeline with SHAP explainability and Streamlit UI.
- **[2026-03-14]** CI/CD pipeline live on GitHub Actions — automated testing on every push.
- **[2026-03-10]** EDA complete — class imbalance handled with SMOTE; XGBoost selected as best model.
- **[2026-03-09]** project comprehension and sharing tasks .
- **[2026-03-09]** Github - get acquainted and expolring the envirement.


---

## 📖 Overview

This project is an advanced **clinical decision-support tool** that helps physicians predict the risk of death from heart failure using patient clinical records. It combines state-of-the-art gradient-boosted classifiers with **SHAP (SHapley Additive exPlanations)** to provide transparent, interpretable predictions at the patient level.

| Feature | Detail |
|---|---|
| 📦 Dataset | 299 patients · 12 clinical features · binary target |
| 🤖 Models | Random Forest · XGBoost · LightGBM · Logistic Regression |
| 🏆 Best Model | **XGBoost** (ROC-AUC ~0.92) |
| 🔍 Explainability | SHAP summary & per-patient waterfall charts |
| 🖥️ Interface | Streamlit web application |
| ⚙️ CI/CD | GitHub Actions (auto-train + pytest on every push) |

---

## 🗂️ Repository Structure

```
heart-failure-project/
├── data/
│   └── heart_failure.csv           # UCI clinical records dataset
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
├── src/
│   ├── data_processing.py          # Loading, cleaning, SMOTE, memory optimisation
│   ├── train_model.py              # Multi-model training & selection
│   └── evaluate_model.py           # Metrics, ROC curve, SHAP summary
├── app/
│   └── app.py                      # Streamlit physician interface
├── tests/
│   └── test_data_processing.py     # 8 automated pytest tests
├── models/                         # Saved model + scaler (generated after training)
├── reports/                        # Plots (generated after evaluation)
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions CI pipeline
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train_model.py
```

> This trains all 4 classifiers, prints a comparison table, and saves the best model to `models/`.

### 3. Evaluate & Generate SHAP Plots

```bash
python src/evaluate_model.py
```

> Outputs confusion matrix, ROC curve, and SHAP summary plot to `reports/`.

### 4. Launch the Web Application

```bash
streamlit run app/app.py
```

> Open **http://localhost:8501** in your browser.

### 5. Run Automated Tests

```bash
pytest tests/ -v
```

### 6. Docker (Optional)

```bash
docker build -t heart-failure-app .
docker run -p 8501:8501 heart-failure-app
```

---

## 📊 Model Performance

All models evaluated on a stratified held-out test set (80/20 split) after SMOTE oversampling on the training set.

| Model | ROC-AUC | F1 (Deceased) | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | ~0.84 | ~0.76 | ~0.78 | ~0.74 | ~0.80 |
| Random Forest | ~0.89 | ~0.82 | ~0.84 | ~0.80 | ~0.85 |
| LightGBM | ~0.91 | ~0.84 | ~0.85 | ~0.83 | ~0.87 |
| **XGBoost** ✅ | **~0.92** | **~0.85** | **~0.86** | **~0.84** | **~0.88** |

> **XGBoost** was selected as the final model for its highest ROC-AUC and F1 on the minority (deceased) class. In a clinical setting, minimizing false negatives (missed deaths) is critical — XGBoost's `scale_pos_weight` parameter combined with SMOTE gave the best balance between precision and recall.

---

## 🔬 SHAP Explainability

SHAP values reveal **which features drive each individual prediction**, making the model transparent and trustworthy for clinicians.

**Top features by global SHAP importance:**

| Rank | Feature | Clinical Interpretation |
|---|---|---|
| 1 | `time` | Shorter follow-up period → higher risk |
| 2 | `serum_creatinine` | Elevated creatinine → kidney stress → increased risk |
| 3 | `ejection_fraction` | Lower ejection % → weaker heart pump → higher risk |
| 4 | `age` | Older patients at greater risk |
| 5 | `serum_sodium` | Low sodium (hyponatremia) → increased mortality risk |

> These results align with established cardiology literature, validating the model's clinical plausibility.

---

## ⚖️ Handling Class Imbalance

The dataset is **imbalanced**: ~68% survived (class 0), ~32% deceased (class 1).

**Strategy chosen: SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE generates synthetic minority-class samples by interpolating between existing ones in feature space. It is applied **exclusively on the training set** to prevent data leakage into the test set.

**Impact of SMOTE:**

| Metric | Without SMOTE | With SMOTE |
|---|---|---|
| Recall (Deceased) | ~0.65 | ~0.84 |
| F1 (Deceased) | ~0.71 | ~0.85 |
| ROC-AUC | ~0.88 | ~0.92 |

> False negatives (predicting a patient survives when they don't) are clinically dangerous. SMOTE significantly improved recall for the deceased class.

---

## 🧠 Memory Optimisation

The `optimize_memory(df)` function in `src/data_processing.py` reduces DataFrame memory usage by downcasting numeric types:

```python
def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    # float64  →  float32
    # int64    →  int32
    # binary   →  int8
```

**Result on this dataset:**

```
Before optimisation : 37.84 KB
After  optimisation : 17.62 KB
Memory saved        : 20.22 KB  (53.4% reduction)
```

---

## 🤖 Prompt Engineering Documentation

**Task selected:** Generating the `optimize_memory(df)` function.

**Prompt used:**
```
I have a pandas DataFrame loaded from a CSV with clinical records.
Some columns are float64 and int64 but could be stored more efficiently.
Write a Python function called optimize_memory(df) that:
1. Converts float64 columns to float32
2. Converts int64 columns to int32
3. Converts binary integer columns (only 0/1 values) to int8
4. Returns the optimised DataFrame without modifying the original
5. Includes clear docstrings
```

**Result:** The AI produced a complete, correct function in one pass — correctly handling all three dtype cases and using `.copy()` to avoid mutating the original input.

**Effectiveness:** The prompt was highly effective because it was specific (named exact dtypes), structured (numbered requirements), and included the non-obvious binary → int8 edge case. A vague prompt like *"optimise this dataframe"* returned an incomplete result. **Potential improvement:** Including an expected before/after memory output example would prompt the AI to also generate a demonstration snippet automatically.

---

## 🧪 Automated Tests

Tests in `tests/test_data_processing.py` — run automatically via GitHub Actions on every push to `main` or `dev`.

| Test | Description |
|---|---|
| `test_no_missing_values` | Verifies the dataset has zero missing values |
| `test_preprocess_handles_injected_nulls` | Pipeline survives rows with injected NaNs |
| `test_optimize_memory_reduces_size` | Memory must decrease after optimisation |
| `test_optimize_memory_preserves_shape` | Shape identical before and after |
| `test_optimize_memory_no_float64` | No float64 columns remain |
| `test_preprocess_output_shapes` | Correct number of features in train/test sets |
| `test_preprocess_smote_balances_classes` | SMOTE produces equal class counts |
| `test_model_predicts` | Loaded model returns valid binary predictions |

---

## 📦 Dataset

- **Source:** [UCI ML Repository — Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- **Citation:** Davide Chicco, Giuseppe Jurman (2020). *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.* BMC Medical Informatics and Decision Making.
- **Samples:** 299 patients · **Features:** 12 clinical + 1 binary target (`DEATH_EVENT`) · **Missing values:** None

---

## 👥 Team
AAzizi Hajar
STITOU Amal
Tayyeb Idriss
ZERZBANE Khawla
ZHIRI Ahmed

## 👥 Supervising teacher
ZERHOUNI Kawtar 

## 👥 Expression of gratitude.
We are eternally grateful to M.NASSIH Rim for her support and help during this week, she helped us a lot, thank you.

<div align="center">

**École Centrale Casablanca — Coding Week, 09–15 March 2026**


</div>