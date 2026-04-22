# Domain Understanding: Credit Risk in AMEX Default Prediction

## 1. Problem Overview

This project is based on the American Express Default Prediction dataset, where the objective is to predict whether a customer will default in the future.

The target variable:
- `default = 1` → Customer will default
- `default = 0` → Customer will not default

The goal is to estimate the **Probability of Default (PD)** for each customer.

---

## 2. Nature of the Dataset

Unlike traditional tabular datasets, this dataset contains:

- **Time-series behavioral data**
- Multiple records per customer (`customer_ID`)
- Monthly snapshots of financial activity

This means:
- Customer behavior over time is more important than single-point values
- Trends (e.g., worsening payment behavior) are critical

---

## 3. Feature Categories

The dataset contains anonymized features grouped as:

### Delinquency Variables (D)
- Capture late payments or missed obligations
- Strong indicators of financial stress

### Balance Variables (B)
- Represent outstanding balances or utilization
- High values may indicate over-leverage

### Payment Variables (P)
- Reflect repayment behavior
- Consistency and trends matter

### Spending Variables (S)
- Represent transaction or usage patterns

### Risk Variables (R)
- Engineered risk indicators

---

## 4. Business Interpretation

The model simulates a real-world credit decision system:

- Input: Customer historical financial behavior
- Output: Risk score (probability of default)
- Decision: Approve, reject, or adjust credit terms

---

## 5. Temporal Behavior Matters

Since data is sequential:
- Increasing delinquency over time → higher risk
- Decreasing payment amounts → warning signal
- Volatility in balances → instability

> Models should consider **customer-level aggregation or sequence patterns**.

---

## 6. Business Cost of Errors

### False Negatives (Critical)
- Predicting "non-default" for a risky customer
- Leads to financial loss

### False Positives
- Rejecting a good customer
- Leads to lost revenue

> In this dataset, **detecting defaulters is the priority**.

---

## 7. Key Challenges in This Dataset

- Severe class imbalance (~3–4% defaults)
- High dimensionality (hundreds of features)
- Time-series structure
- Anonymized features (limited domain labeling)

---

## 8. Guidance for Modeling Team

- Aggregate features at customer level (mean, max, trend)
- Capture temporal patterns where possible
- Focus on delinquency and payment signals
- Avoid relying purely on static snapshots

---

## 9. Summary

This dataset represents a **real-world credit risk system** where:
- Behavior over time defines risk
- Rare default events carry high importance
- Models must balance predictive power with interpretability
