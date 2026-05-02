# ==========================================
# Issue #13 – XGBoost Imbalance-Aware Trainer
# Author: Shaik Irfan Basha
# ==========================================

"""
Train XGBoost with imbalance-aware settings and stable configuration.

Usage:
    python -m credit_default_prediction.model_training.xgboost_trainer

Outputs:
    models/xgboost_imbalanced.pkl     – trained model artifact
    reports/xgboost_results.json      – evaluation metrics
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ..config import DATA_DIR, MODELS_DIR, PROJECT_ROOT, REPORTS_DIR

# ==========================================
# Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==========================================
# Constants
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "SeriousDlqin2yrs"

MODEL_SAVE_PATH = MODELS_DIR / "xgboost_imbalanced.pkl"
RESULTS_SAVE_PATH = REPORTS_DIR / "xgboost_results.json"


# ==========================================
# 1. Load Data
# ==========================================
def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the GMSC benchmark CSV."""

    if path is None:
        path = DATA_DIR / "cs-training.csv"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    log.info("Loading data from %s", path)
    df = pd.read_csv(path)

    # Drop unnamed index column if present
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    log.info("Loaded %d rows × %d columns", *df.shape)
    return df


# ==========================================
# 2. Preprocess
# ==========================================
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Handle missing values and split features/target."""

    log.info("Preprocessing...")

    # Fill missing values with median (stable for skewed financial data)
    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    if "NumberOfDependents" in df.columns:
        df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
            df["NumberOfDependents"].median()
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    missing_total = int(X.isna().sum().sum())
    log.info("Features: %d | Remaining missing cells: %d", X.shape[1], missing_total)

    return X, y


# ==========================================
# 3. Build XGBoost with imbalance-aware config
# ==========================================
def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    """
    Create an XGBClassifier with imbalance-aware and stable settings.

    Imbalance-aware:
        - scale_pos_weight: auto-computed ratio of negatives to positives
    Stability:
        - early_stopping_rounds via fit() to prevent overfitting
        - L1 + L2 regularization
        - conservative max_depth and min_child_weight
        - row + column subsampling for robustness
        - fixed random seed
    """

    model = XGBClassifier(
        # --- Core ---
        n_estimators=500,           # ceiling, early stopping will pick the best
        learning_rate=0.05,
        max_depth=5,

        # --- Imbalance ---
        scale_pos_weight=scale_pos_weight,

        # --- Regularization ---
        reg_alpha=0.1,              # L1
        reg_lambda=1.0,             # L2
        min_child_weight=5,         # prevents splits on tiny leaf groups

        # --- Subsampling ---
        subsample=0.8,
        colsample_bytree=0.8,

        # --- Stability ---
        eval_metric="auc",
        early_stopping_rounds=50,   # stop if val AUC doesn't improve for 50 rounds
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=1,
    )

    log.info("XGBoost configured  |  scale_pos_weight=%.2f", scale_pos_weight)
    return model


# ==========================================
# 4. Train
# ==========================================
def train(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    """Fit with early stopping on the validation set."""

    log.info("Training XGBoost  (%d train / %d val)...", len(X_train), len(X_val))

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,  # print every 50 rounds
    )

    best_iteration = getattr(model, "best_iteration", model.n_estimators)
    log.info("Training complete  |  best iteration: %d", best_iteration)

    return model


# ==========================================
# 5. Evaluate
# ==========================================
def evaluate(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Evaluate on holdout set with ROC-AUC as the primary metric."""

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    roc_auc = roc_auc_score(y_val, y_proba)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
    }

    log.info("=" * 50)
    log.info("  EVALUATION RESULTS")
    log.info("=" * 50)
    log.info("  ROC-AUC   : %.4f", roc_auc)
    log.info("  Recall    : %.4f", recall)
    log.info("  Precision : %.4f", precision)
    log.info("  F1 Score  : %.4f", f1)
    log.info("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["No Default", "Default"]))

    return metrics


# ==========================================
# 6. Save outputs
# ==========================================
def save_outputs(model: XGBClassifier, metrics: dict) -> None:
    """Persist the trained model and evaluation metrics."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_SAVE_PATH)
    log.info("Model saved  → %s", MODEL_SAVE_PATH)

    with open(RESULTS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("Results saved → %s", RESULTS_SAVE_PATH)


# ==========================================
# Main Pipeline
# ==========================================
def main(data_path: str | Path | None = None) -> dict:
    """End-to-end XGBoost training pipeline."""

    log.info("=" * 50)
    log.info("  ISSUE #13 – XGBoost Imbalance-Aware Trainer")
    log.info("=" * 50)

    # Load & preprocess
    df = load_data(data_path)
    X, y = preprocess(df)

    # Compute class imbalance ratio
    n_negative = int((y == 0).sum())
    n_positive = int((y == 1).sum())
    scale_pos_weight = n_negative / n_positive
    log.info(
        "Class distribution  |  0: %d (%.1f%%)  |  1: %d (%.1f%%)  |  ratio: %.2f",
        n_negative,
        100 * n_negative / len(y),
        n_positive,
        100 * n_positive / len(y),
        scale_pos_weight,
    )

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Build, train, evaluate
    model = build_xgboost(scale_pos_weight)
    model = train(model, X_train, y_train, X_val, y_val)
    metrics = evaluate(model, X_val, y_val)

    # Save
    save_outputs(model, metrics)

    log.info("Done ✅")
    return metrics


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    main()
