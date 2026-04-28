# Class Imbalance in AMEX Default Prediction

## 1. Overview

The AMEX dataset exhibits **severe class imbalance**, with only "~3–4%" of customers labeled as defaulters.

- Majority Class: Non-default (≈ 96–97%)
- Minority Class: Default (≈ 3–4%)

---

## 2. Why This Happens

This reflects real-world financial systems:
- Most customers repay loans
- Defaults are rare but high-impact events

---

## 3. Implications for Modeling

A naive model can:
- Predict all customers as non-default
- Achieve ~96% accuracy
- Still be completely useless

---

## 4. Metric Selection is Critical

### Accuracy → Misleading
- Dominated by majority class

### Recall (for default class) → Critical
- Measures ability to detect risky customers

### Precision → Important
- Avoids unnecessary rejection of good customers

### ROC-AUC → Overall performance

---

## 5. AMEX-Specific Evaluation Insight

The competition itself emphasizes:
- Ranking quality (who is riskier)
- Not just classification accuracy

---

## 6. Techniques to Handle Imbalance

### Data-Level
- Oversampling minority class
- Undersampling majority class
- SMOTE (synthetic generation)

### Algorithm-Level
- Class weights
- Cost-sensitive learning

---

## 7. Risks of Poor Handling

- Missing defaulters → financial loss
- Biased predictions → unreliable system

---

## 8. Guidance for Team (Data Engineering & Model Evaluation Team)
  
- Do NOT rely on accuracy
- Monitor recall for default class
- Use weighted or imbalance-aware models
- Evaluate using ROC-AUC and recall together

---

## 9. Summary

Class imbalance is a defining characteristic of this dataset.

Proper handling is essential to:
- Capture risk effectively
- Build a usable credit scoring model
