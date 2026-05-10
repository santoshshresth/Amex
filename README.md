# Explainable Credit Default Prediction

**Built by Evoastra Innovation Systems.**

This repository contains an end-to-end, production-ready Machine Learning pipeline for predicting credit card defaults. Built to handle massive, highly imbalanced, and anonymized datasets (5.5 million records), the project demonstrates enterprise-grade ML practices including GPU-accelerated training, Bayesian hyperparameter tuning, Explainable AI (XAI), and rigorous algorithmic fairness auditing.

## 📊 Project Results & Business Impact

Our Champion **XGBoost Model** achieved an elite **0.961 ROC-AUC**, drastically outperforming standard industry baselines. 

* **High Capture Rate (0.95 Recall):** Successfully identifies and flags over 95% of actual defaulters, minimizing catastrophic financial loss.
* **Safe Approvals (0.98 Precision):** When the model approves a customer, there is a 98% statistical probability they will pay their bill, securing corporate revenue.
* **Fully Explainable (GDPR Compliant):** Utilizes SHAP Waterfalls to generate mathematically perfect, human-readable "receipts" for every credit denial, fulfilling the Right to Explanation.
* **Fairness Audited (ECOA / FHA Compliant):** Benchmarked against a secondary dataset to proactively identify, isolate, and mitigate demographic biases (e.g., Age) before production deployment.

---

## 🗂️ Repository Structure

The project is organized sequentially to support reproducibility and clear hand-offs to downstream MLOps teams.

```text
📁 Amex-Credit-Risk-Prediction
│
├── 📁 data/               
│   ├── 📁 raw/            # Source Kaggle datasets & benchmark CSVs (git-ignored)
│   └── 📁 processed/      # Engineered stages (imputed, outliers_handled, ml_ready.parquet)
│
├── 📁 models/             # Saved model artifacts (e.g., xgboost_optimized_final.joblib)
│
├── 📁 notebooks/          # Step-by-step Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_outlier.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_aggregation_encoding.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_explainability_global_SHAP_LIME.ipynb
│   ├── 08_metric_review_calibration.ipynb
│   └── 09_benchmark_evaluation.ipynb
│
├── 📁 reports/            # Executive summaries, SHAP receipts, and LIME governance charts
│
├── 📁 src/                # Core Python package scaffold
│
├── 📄 requirements.txt    # Project dependencies
└── 📄 README.md           # Project documentation
```
🛠️ Major Packages Used
This pipeline relies on a modern data science stack optimized for speed, scalability, and transparency:

xgboost: Core gradient boosting framework (compiled with GPU support) for champion model training.

shap: Game-theoretic framework used for global portfolio drivers and generating localized, GDPR-compliant "Waterfall" denial receipts.

lime: Local Interpretable Model-agnostic Explanations used specifically for human-readable rule extraction and ECOA fairness benchmarking.

hyperopt: Bayesian optimization library used for efficient hyperparameter tuning over massive parameter spaces.

pandas & numpy: Core data manipulation, utilizing .parquet file formats to prevent RAM bottlenecking on 5.5M+ rows.

scikit-learn: Baseline modeling, metric calculation (Precision, Recall, Brier Score, MCC), and calibration displays.

⚙️ Installation & Setup
To reproduce this environment and run the notebooks, please install the required packages using the provided requirements.txt file in the root directory.

Prerequisites
Python 3.9+

(Optional but recommended) NVIDIA GPU with CUDA drivers installed for accelerated XGBoost training.

Setup Instructions
1. Clone the repository
```Bash
git clone [https://github.com/your-username/Amex-Credit-Risk-Prediction.git](https://github.com/your-username/Amex-Credit-Risk-Prediction.git)
cd Amex-Credit-Risk-Prediction
```
2. Create and activate a virtual environment
Windows (PowerShell):
```PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
macOS / Linux:
```Bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Install dependencies from requirements.txt
```Bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
4. Launch Jupyter Notebook
```Bash
jupyter notebook
```
Navigate to the notebooks/ folder and execute the pipeline sequentially starting from 01_data_preparation.ipynb.

(Note: Raw dataset files must be placed in data folder prior to running the pipeline. Due to size and licensing, Amex data is not tracked in version control).