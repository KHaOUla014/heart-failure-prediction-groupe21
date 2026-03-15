<div align="center">

<img src="assets/logo.jpg" width="1000"/>

#  Heart Failure Risk Predictor

**Explainable Machine Learning for Clinical Decision Support**

<p align="center">
  <a href="https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records"><img src="https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-blue?style=flat-square" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square&logo=python" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Framework-Streamlit-red?style=flat-square&logo=streamlit" /></a>
  <a href="#"><img src="https://img.shields.io/badge/XAI-SHAP-orange?style=flat-square" /></a>
  <a href="#"><img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?style=flat-square&logo=github" /></a>
  <a href="https://zhiria09-1773220288884.atlassian.net/jira/software/projects/KAN/list?jql=project+%3D+KAN+ORDER+BY+created+DESC&atlOrigin=eyJpIjoiZGMwNTAxMzVhOTg3NDI2MDhmMDA3ZmM2YWM2ZDhjZTMiLCJwIjoiaiJ9"><img src="https://img.shields.io/badge/Project%20Board-Jira-0052CC?style=flat-square&logo=jira" /></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-purple?style=flat-square" /></a>
</p>

<p align="center">
  <b>École Centrale Casablanca — Coding Week, March 2026</b>
</p>

</div>

---

## 🗓️ JIRA Timeline & Task Organisation

> 📋 **Full project board:** [View on Jira](https://zhiria09-1773220288884.atlassian.net/jira/software/projects/KAN/list?jql=project+%3D+KAN+ORDER+BY+created+DESC&atlOrigin=eyJpIjoiZGMwNTAxMzVhOTg3NDI2MDhmMDA3ZmM2YWM2ZDhjZTMiLCJwIjoiaiJ9)

| Date | Milestone |
|---|---|
| 2026-03-09 | 🔍 Project comprehension and task distribution among team members |
| 2026-03-09 | 🌐 GitHub onboarding — exploring the environment and setting up the repository |
| 2026-03-10 | 📊 EDA complete — class imbalance handled with SMOTE; XGBoost selected as best model |
| 2026-03-14 | ⚙️ CI/CD pipeline live on GitHub Actions — automated testing on every push |
| 2026-03-15 | 🎉 Project delivered — full pipeline with SHAP explainability and Streamlit UI |

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
│   ├── heart_failure_clinical_records_dataset.csv   # UCI clinical records dataset
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
├── src/
│   ├── data_processing.py          # Loading, cleaning, SMOTE, memory optimisation
│   ├── train_model.py              # Multi-model training & selection
│   └── evaluate_model.py           # Metrics, ROC curve, SHAP summary
├── app/
│   └── app.py                      # Streamlit physician interface
├── tests/
│   └── test_data_processing.py     # Automated pytest tests
├── assets/
│   ├── heart_logo.jpg              # Project logo (displayed in README header)
│   └── output.gif                  # Application demo animation
├── models/                         # Saved model + scaler (generated after training)
├── reports/                        # Plots (generated after evaluation)
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions CI pipeline
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 💻 Code Explanation

### `src/data_processing.py`

This file is the **data foundation** of the entire pipeline. It handles everything from loading to preprocessing.

```python
def load_data(filepath):
    """
    Loads the dataset from either CSV or Excel format.
    Automatically detects the file type from the extension.
    Supports: .csv, .xls, .xlsx
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath, engine='xlrd')
```

> 📌 **Why auto-detect?** The project uses `.xls` locally but `.csv` in CI/CD (GitHub Actions). This function makes the code work in both environments without modification.

```python
def optimize_memory(df):
    """
    Reduces memory footprint by converting:
      float64 → float32  (saves 50% per float column)
      int64   → int32    (saves 50% per int column)
    Critical for scalability when dataset grows larger.
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
```

> 📌 **Result:** Memory reduced from 31.228 KB to 15.68 KB — a **49.8% reduction** with no loss of information.

```python
def preprocess(df):
    """
    Full preprocessing pipeline:
      1. Separates features (X) from target (y = DEATH_EVENT)
      2. Splits into train (80%) / test (20%) with stratification
      3. Applies SMOTE ONLY on training data — never on test data
    
    ⚠️ CRITICAL: SMOTE must be applied AFTER the split to avoid
    data leakage (synthetic samples in test set = invalid evaluation).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

---

### `src/train_model.py`

This file trains all four required classifiers and returns the best one automatically.

```python
def get_data_path():
    """
    Resolves the dataset path dynamically.
    Priority: .xls (local dev) → .csv (CI/CD fallback)
    Raises FileNotFoundError if neither is found.
    """
```

```python
models = {
    'Random Forest':       RandomForestClassifier(random_state=42),
    'XGBoost':             XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM':            LGBMClassifier(random_state=42, verbose=-1),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}
```

> 📌 **Why these 4 models?** They cover the full spectrum: linear baseline (Logistic Regression), ensemble bagging (Random Forest), and two gradient boosting implementations (XGBoost, LightGBM) — allowing a fair comparison across model families.

```python
for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    # Track the model with the highest AUC
    if auc > best_score:
        best_score = auc
        best_model = model
```

> 📌 **Why ROC-AUC for selection?** AUC measures the model's ability to rank patients correctly regardless of classification threshold — more robust than accuracy for imbalanced datasets.

---

### `src/evaluate_model.py`

Loads the best model and computes the full evaluation report.

```python
def evaluate_saved_model():
    """
    Runs the best model on the held-out test set and reports:
      - ROC-AUC  : overall discrimination ability
      - Accuracy : overall correct predictions
      - Precision: of predicted deaths, how many were real
      - Recall   : of actual deaths, how many were caught  ← most critical clinically
      - F1-Score : harmonic mean of precision and recall
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
```

> ⚕️ **Clinical priority:** In a medical context, **Recall** is the most important metric. A missed death (false negative) is far more dangerous than a false alarm (false positive). This is why SMOTE — which significantly boosts recall — was the right choice.

---

### `app/app.py`

The **Streamlit physician interface** — the user-facing layer of the entire project.

```python
@st.cache_resource
def load_model():
    """
    Trains the model once on startup and caches it in memory.
    @st.cache_resource ensures the model is NOT retrained on every
    user interaction — critical for application performance.
    """
    model, X_train, X_test, y_train, y_test = train_and_select_best_model()
    return model, X_train, X_test, y_train, y_test
```

```python
# SHAP waterfall plot for the individual patient
explainer = shap.TreeExplainer(model)
shap_values_obj = explainer(patient_data)

# Handle both binary and multi-output SHAP formats
if len(shap_values_obj.shape) == 3:
    shap.plots.waterfall(shap_values_obj[0, :, 1], show=False)
else:
    shap.plots.waterfall(shap_values_obj[0], show=False)
```

> 📌 **Why the shape check?** Different versions of SHAP and different tree models return SHAP values in different array shapes. This guard makes the code robust across environments.

---

### `tests/test_data_processing.py`

10 automated tests that verify every critical component of the pipeline.

```python
def test_preprocess_smote_balances_classes(df):
    """
    After SMOTE the two classes in y_train must be perfectly equal.
    This test ensures the imbalance fix is actually working,
    not just assumed to work.
    """
    X_train, X_test, y_train, y_test = preprocess(df)
    counts = pd.Series(y_train).value_counts()
    assert counts[0] == counts[1]
```

```python
def test_optimize_memory_no_float64(df):
    """
    Verifies that NO float64 columns remain after optimization.
    Without this test, a silent dtype regression could go unnoticed.
    """
    opt = optimize_memory(df.copy())
    float64_cols = [c for c in opt.columns if opt[c].dtype == np.float64]
    assert float64_cols == []
```

---

### `.github/workflows/ci.yml`

```yaml
on:
  push:
    branches: [ main, dev ]   # triggers on every push to main or dev
  pull_request:
    branches: [ main ]        # triggers on every pull request to main

jobs:
  test:
    steps:
      - pip install -r requirements.txt   # install all dependencies
      - pytest tests/ -v --tb=short       # run all 10 tests automatically
```

> 📌 **Why CI/CD matters:** It guarantees that broken code can never silently reach the main branch. If a teammate pushes code that breaks a test, GitHub immediately flags it before it affects anyone else.

---

## 🎬 Application Demo

<div align="center">
  <img src="assets/output.gif" alt="Heart Failure Risk Predictor Demo" width="700"/>
</div>

The demo above shows the full prediction workflow:

| Step | Description |
|---|---|
| 1️⃣ Launch | Starting the Streamlit app locally |
| 2️⃣ Input | Entering patient clinical data in the sidebar |
| 3️⃣ Predict | Clicking **PREDICT RISK** and reading the HIGH / LOW result |
| 4️⃣ SHAP | Interpreting the waterfall chart for the individual patient |
| 5️⃣ Summary | Reading the patient data summary table |

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

### 3. Evaluate & Generate SHAP Plots
```bash
python src/evaluate_model.py
```

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

| Model | ROC-AUC | F1 (Deceased) | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | ~0.84 | ~0.76 | ~0.78 | ~0.74 | ~0.80 |
| Random Forest | ~0.89 | ~0.82 | ~0.84 | ~0.80 | ~0.85 |
| LightGBM | ~0.91 | ~0.84 | ~0.85 | ~0.83 | ~0.87 |
| **XGBoost** ✅ | **~0.92** | **~0.85** | **~0.86** | **~0.84** | **~0.88** |

---

## 🔬 SHAP Explainability

| Rank | Feature | Clinical Interpretation |
|---|---|---|
| 1 | `time` | Shorter follow-up period → higher risk |
| 2 | `serum_creatinine` | Elevated creatinine → kidney stress → increased risk |
| 3 | `ejection_fraction` | Lower ejection % → weaker heart pump → higher risk |
| 4 | `age` | Older patients at greater risk |
| 5 | `serum_sodium` | Low sodium (hyponatremia) → increased mortality risk |

---

## ⚖️ Handling Class Imbalance

**Strategy: SMOTE** applied exclusively on the training set.

| Metric | Without SMOTE | With SMOTE |
|---|---|---|
| Recall (Deceased) | ~0.65 | ~0.84 |
| F1 (Deceased) | ~0.71 | ~0.85 |
| ROC-AUC | ~0.88 | ~0.92 |

---

## 🧠 Memory Optimisation

```
Before optimisation : 37.84 KB
After  optimisation : 17.62 KB
Memory saved        : 20.22 KB  (53.4% reduction)
```

---

## 🤖 Prompt Engineering Documentation

**Task selected:** Selecting and justifying the best ML model for heart failure prediction.

**Prompt used:**
```
I am building a binary classification model to predict heart failure death risk.
My dataset has 299 patients, is imbalanced (68% survived / 32% deceased),
and has 12 clinical features. I have trained these 4 models and obtained
the following ROC-AUC scores:
  - Logistic Regression : 0.84
  - Random Forest       : 0.89
  - LightGBM            : 0.91
  - XGBoost             : 0.92

Given the medical context where false negatives (missed deaths) are dangerous,
which model should I select as my final model and why?
Justify your answer using clinical and technical criteria.
```

**AI Response (summary):** The AI recommended **XGBoost** as the final model, justifying the choice on three grounds: (1) highest ROC-AUC indicating best overall discrimination, (2) best F1 on the minority class showing balanced precision/recall, and (3) the availability of `scale_pos_weight` as an additional imbalance correction layer on top of SMOTE. The AI also noted that in clinical settings, the cost asymmetry between false negatives and false positives makes maximising recall critical.

**Effectiveness:** Highly effective — by providing the concrete scores AND the clinical context in the prompt, the AI was able to reason from both a statistical and medical perspective simultaneously. A prompt without the clinical context would have selected the model on AUC alone without addressing the false-negative risk.

**Lesson learned:** Context is everything in prompt engineering. The same question with different context yields fundamentally different (and more useful) answers.

---

## 📚 What We Learned This Coding Week

### 🤖 Machine Learning
- Training and comparing multiple classifiers: Random Forest, XGBoost, LightGBM, Logistic Regression
- Understanding evaluation metrics beyond accuracy: ROC-AUC, F1, Precision, Recall
- Handling **class imbalance** with SMOTE and understanding the risk of data leakage
- Model selection based on clinically relevant criteria — prioritising recall over accuracy

### 🔍 Explainable AI (XAI)
- Discovering **SHAP** as a tool for model transparency in medical applications
- Generating global summary plots and per-patient waterfall charts
- Understanding why explainability is non-negotiable in clinical AI

### 🛠️ Software Engineering
- Structuring a Python ML project professionally (src/, tests/, app/, notebooks/)
- Writing **clean, modular, documented code** with docstrings
- Optimising memory usage through dtype downcasting
- Writing **automated tests** with `pytest`

### 🌐 Web Development
- Building an interactive clinical UI with **Streamlit**
- Integrating ML predictions and SHAP visualisations in real-time
- Caching expensive computations with `@st.cache_resource`

### ⚙️ DevOps & CI/CD
- Creating a **GitHub repository** with a professional branching strategy
- Writing a **GitHub Actions** workflow for automated CI/CD
- Containerising the app with **Docker**

### 📋 Project Management
- Organising and tracking tasks with **Jira** (To Do → In Progress → Review → Done)
- Distributing responsibilities across team members
- Collaborating on a shared GitHub repository with branches and pull requests

### 🤝 Prompt Engineering
- Using AI assistants as coding and reasoning partners
- Writing specific, context-rich prompts to get high-quality outputs
- Understanding that vague prompts produce generic results

---

## 🚧 Challenges & Problems Encountered

This section honestly documents the real difficulties the team faced — and how we solved them.

### 📋 1. Task Management & Coordination
At the start of the week, dividing work among 5 people on a single codebase was harder than expected. Two team members worked on the same file simultaneously and created conflicting versions. **Solution:** We adopted a strict GitHub branching strategy (one branch per feature/task) and used Jira to assign ownership clearly — no two people working on the same file at the same time.

### 💡 2. Changing Ideas Mid-Way
The team initially planned to use a Flask interface, but midway through decided to switch to Streamlit for faster development. This required rewriting the entire `app.py` from scratch. **Lesson:** Deciding on the tech stack early and sticking to it saves significant time.

### 🐛 3. Debugging & Silent Errors
The most time-consuming bug was related to SHAP output shapes. The `shap_values` array returned different shapes depending on the model type and SHAP version — causing the waterfall plot to crash silently for some models. The fix required adding a shape check:
```python
if len(shap_values_obj.shape) == 3:
    shap.plots.waterfall(shap_values_obj[0, :, 1], show=False)
else:
    shap.plots.waterfall(shap_values_obj[0], show=False)
```

### 📦 4. Missing Modules & Environment Issues
Several `ModuleNotFoundError` errors appeared when running the project on different machines because the `requirements.txt` was incomplete. `xlrd`, `openpyxl`, and `joblib` were missing. **Fix:** We systematically ran the project on a clean environment and added every missing package. The final `requirements.txt` was validated on three different machines before submission.

### 🤖 5. Correcting & Adapting AI-Generated Code
AI tools (Claude, ChatGPT) were extremely helpful but never 100% ready to use. Key adaptations required:
- The AI-generated `optimize_memory()` function did not handle binary columns (`int8` optimization) — we added that case manually
- The AI-generated `train_model.py` hardcoded the path to `.xls` and failed in CI where only `.csv` was available — we rewrote `get_data_path()` to auto-detect both formats
- The AI's initial SHAP integration assumed a single-output model — we added the shape guard for binary classification compatibility

> 💡 **Key takeaway:** AI-generated code should always be treated as a **first draft** — it accelerates development significantly, but human review, testing, and adaptation are always necessary.

---

## 📦 Dataset

- **Source:** [UCI ML Repository — Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- **Citation:** Davide Chicco, Giuseppe Jurman (2020). *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.* BMC Medical Informatics and Decision Making.
- **Samples:** 299 patients · **Features:** 12 clinical + 1 binary target (`DEATH_EVENT`) · **Missing values:** None

---

## 👥 Team

| Name 
|---
| Azizi Hajar 
| Stitou Amal 
| Tayyeb Idriss 
| Zerzbane Khaoula 
| Zhiri Ahmed 
## 🎓 Supervising Teachers

Dr. Zerhouni Kawtar · Dr. Nassih Rym · Pr. Hermann Agossou · Pr. Kourouma Nouhan · Pr. Mehdi Soufiane

## 💙 Expression of Gratitude

We are eternally grateful to **Dr. Nassih Rym** for her constant support and guidance throughout this week. Her help was invaluable — thank you sincerely and all the coding week team.

---

<div align="center">

**École Centrale Casablanca — Coding Week, 09–15 March 2026**


</div>
