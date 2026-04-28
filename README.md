# Explainable Credit Default Prediction

Built by Evoastra Innovation Systems.

This repository is the execution scaffold for the 20-day research project on explainable credit default prediction using XGBoost and SHAP. The initial focus is the Viraj setup task: environment setup, reproducible execution control, and a clean repository structure for the downstream ML teams.

## What is included

- A Python package scaffold under `src/credit_default_prediction`
- A runnable bootstrap CLI for project initialization and readiness checks
- A dependency set for modeling, explainability, evaluation, and reporting
- A repository layout aligned to the project roadmap and dual-dataset workflow

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m credit_default_prediction init
python -m credit_default_prediction check
python -m credit_default_prediction benchmark
```

If you are starting from the supplied benchmark file, place `cs-training (1).csv` in `data/raw/` and rename it to `cs-training.csv` so downstream scripts can use the standard project path.

For a one-command Windows bootstrap, run `.\scripts\bootstrap.ps1` from the repository root.

## Repository layout

- `data/raw` for source inputs
- `data/processed` for engineered datasets
- `models` for trained artifacts
- `reports` for charts, tables, and executive outputs
- `notebooks` for analysis notebooks
- `src/credit_default_prediction` for the package code

## Dataset rule

For the first 19 days, use the training data with an `80/20` `train_test_split` so ROC-AUC, calibration, and SHAP charts can be generated locally. Ignore the official Kaggle test files until the final submission day.

## Next implementation blocks

- Data loading and validation
- Missing value and outlier handling
- Baseline models and metric reporting
- XGBoost training and SHAP explainability
- Governance review and compliance mapping in [reports/governance_review.md](reports/governance_review.md)
