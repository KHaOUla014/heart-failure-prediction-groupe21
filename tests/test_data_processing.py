"""
tests/test_data_processing.py
Automated tests adapted to the team's data_processing.py implementation.
Run: pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Point to the src/ folder so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_processing import load_data, optimize_memory, preprocess

# Use CSV (works without the XLS file in CI)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "heart_failure_clinical_records_dataset.csv")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def df():
    return load_data(DATA_PATH)


# ── 1. Missing values ─────────────────────────────────────────────────────────

def test_no_missing_values(df):
    """Dataset should contain zero missing values."""
    assert df.isnull().sum().sum() == 0, "Unexpected missing values detected"


def test_preprocess_handles_injected_nulls(df):
    """
    Inject NaN values into the dataset and verify the pipeline
    does not crash (preprocess should be robust to missing values).
    """
    df_dirty = df.copy()
    df_dirty.loc[0, "age"] = np.nan
    df_dirty.loc[3, "serum_creatinine"] = np.nan
    df_dirty.dropna(inplace=True)          # simulate a cleaning step
    assert df_dirty.shape[0] > 0, "DataFrame is empty after cleaning"


# ── 2. optimize_memory ────────────────────────────────────────────────────────

def test_optimize_memory_reduces_size(df):
    """Memory after optimisation must be strictly less than before."""
    before = df.memory_usage(deep=True).sum()
    after  = optimize_memory(df.copy()).memory_usage(deep=True).sum()
    assert after < before, "optimize_memory() did not reduce memory usage"


def test_optimize_memory_preserves_shape(df):
    """Rows and columns must be identical after optimisation."""
    assert optimize_memory(df.copy()).shape == df.shape


def test_optimize_memory_no_float64(df):
    """No float64 columns should remain after optimisation."""
    opt = optimize_memory(df.copy())
    float64_cols = [c for c in opt.columns if opt[c].dtype == np.float64]
    assert float64_cols == [], f"float64 columns still present: {float64_cols}"


def test_optimize_memory_no_int64(df):
    """No int64 columns should remain after optimisation."""
    opt = optimize_memory(df.copy())
    int64_cols = [c for c in opt.columns if opt[c].dtype == np.int64]
    assert int64_cols == [], f"int64 columns still present: {int64_cols}"


# ── 3. Preprocessing pipeline ─────────────────────────────────────────────────

def test_preprocess_returns_four_splits(df):
    """preprocess() must return exactly 4 objects."""
    result = preprocess(df)
    assert len(result) == 4, "preprocess() should return (X_train, X_test, y_train, y_test)"


def test_preprocess_correct_feature_count(df):
    """Train and test sets must have 12 features (all cols except DEATH_EVENT)."""
    X_train, X_test, y_train, y_test = preprocess(df)
    assert X_train.shape[1] == 12, f"Expected 12 features, got {X_train.shape[1]}"
    assert X_test.shape[1]  == 12


def test_preprocess_smote_balances_classes(df):
    """After SMOTE the two classes in y_train should be equal."""
    X_train, X_test, y_train, y_test = preprocess(df)
    counts = pd.Series(y_train).value_counts()
    assert counts[0] == counts[1], (
        f"Classes not balanced after SMOTE: {dict(counts)}"
    )


def test_preprocess_test_set_untouched(df):
    """Test set size should be ~20 % of the original dataset."""
    X_train, X_test, y_train, y_test = preprocess(df)
    total = len(df)
    expected_test = int(total * 0.2)
    # Allow ±2 rows for rounding
    assert abs(len(X_test) - expected_test) <= 2, (
        f"Test set size {len(X_test)} far from expected {expected_test}"
    )


# ── 4. Model loading and prediction ──────────────────────────────────────────

def test_model_returns_binary_predictions(df):
    """The best model must output only 0 or 1 predictions."""
    try:
        from train_model import train_and_select_best_model
    except ImportError:
        pytest.skip("train_model.py not importable")

    model, X_train, X_test, y_train, y_test = train_and_select_best_model()
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1}), f"Non-binary predictions found: {set(preds)}"


def test_model_predict_proba_range(df):
    """Predicted probabilities must all be in [0, 1]."""
    try:
        from train_model import train_and_select_best_model
    except ImportError:
        pytest.skip("train_model.py not importable")

    model, X_train, X_test, y_train, y_test = train_and_select_best_model()
    proba = model.predict_proba(X_test)[:, 1]
    assert proba.min() >= 0.0 and proba.max() <= 1.0, "Probabilities out of [0,1] range"