# ==========================================
# Credit Risk Model Training Script
# ==========================================

import pandas as pd
import numpy as np
import os
import logging
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==========================================
# Logging Setup
# ==========================================
logging.basicConfig(level=logging.INFO)


# ==========================================
# 1. Load Data
# ==========================================
def load_data(path):
    logging.info("Loading data...")
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


# ==========================================
# 2. Preprocessing
# ==========================================
def preprocess(df):
    logging.info("Preprocessing data...")

    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]

    return X, y


# ==========================================
# 3. Train-Test Split
# ==========================================
def split_data(X, y):
    logging.info("Splitting data...")
    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


# ==========================================
# 4. Cross Validation
# ==========================================
def cross_validation(model, X, y):
    logging.info("Running Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]

        scores.append(roc_auc_score(y_val, y_pred))

    logging.info(f"CV ROC-AUC: {np.mean(scores):.4f}")


# ==========================================
# 5. Train Models
# ==========================================
def train_models(X_train, y_train, scale_pos_weight):

    logging.info("Training models...")
    models = {}

    models["Logistic"] = LogisticRegression(class_weight='balanced', max_iter=1000)
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1
    )
    models["XGBoost"] = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        scale_pos_weight=scale_pos_weight, eval_metric='logloss',
        use_label_encoder=False, random_state=42
    )
    models["LightGBM"] = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        num_leaves=31, scale_pos_weight=scale_pos_weight, random_state=42
    )

    # Train all
    for name, model in models.items():
        model.fit(X_train, y_train)
        logging.info(f"{name} trained.")

    return models


# ==========================================
# 6. Evaluation
# ==========================================
def evaluate_models(models, X_val, y_val):
    logging.info("Evaluating models...")

    results = {}
    predictions = {}

    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)

        results[name] = auc
        predictions[name] = y_pred_proba

        print(f"{name} ROC-AUC: {auc:.4f}")

    return results, predictions


# ==========================================
# 7. Threshold Optimization
# ==========================================
def find_best_threshold(y_true, y_pred):
    best_thresh = 0.5
    best_score = 0

    for t in np.arange(0.1, 0.9, 0.05):
        y_bin = (y_pred > t).astype(int)
        score = roc_auc_score(y_true, y_bin)

        if score > best_score:
            best_score = score
            best_thresh = t

    logging.info(f"Best Threshold: {best_thresh}")
    return best_thresh


# ==========================================
# Save All Models
# ==========================================
def save_models(models):
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        file_path = f"models/{name.lower()}_model.pkl"
        joblib.dump(model, file_path)
        logging.info(f"{name} model saved at {file_path}")


# ==========================================
# Main Pipeline
# ==========================================
def main():

    df = load_data("data/cs-training.csv")
    X, y = preprocess(df)

    X_train, X_val, y_train, y_val = split_data(X, y)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    # Cross-validation
    base_model = LGBMClassifier(scale_pos_weight=scale_pos_weight)
    cross_validation(base_model, X, y)

    # Train
    models = train_models(X_train, y_train, scale_pos_weight)

    # Evaluate
    results, predictions = evaluate_models(models, X_val, y_val)

    # Best model
    best_model = models["LightGBM"]
    best_pred = predictions["LightGBM"]

    # Threshold tuning
    best_thresh = find_best_threshold(y_val, best_pred)
    y_pred_final = (best_pred > best_thresh).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_final))

    # Save models
    save_models(models)
    joblib.dump(best_model, "models/best_model.pkl")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(list(results.items()), columns=["Model", "ROC-AUC"])\
        .to_csv("outputs/model_results.csv", index=False)

    logging.info("Training Completed Successfully!")


# ==========================================
# Run
# ==========================================
if __name__ == "__main__":
    main()