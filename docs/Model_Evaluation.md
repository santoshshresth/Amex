# Model Evaluation Strategy for AMEX Dataset

## 1. Overview

Evaluation must reflect:
- Class imbalance
- Business risk priorities

---

## 2. Why Accuracy is Not Suitable

With ~96% non-defaults:
- Accuracy can be misleadingly high
- Does not reflect risk detection ability

---

## 3. Key Metrics

### Recall (Default Class)
- Detect as many defaulters as possible

### Precision
- Avoid excessive false alarms

### ROC-AUC
- Measures ranking performance

---

## 4. Business Trade-offs

- High Recall → safer but more false positives
- High Precision → efficient but riskier

---

## 5. Threshold Strategy

- Avoid fixed threshold (0.5)
- Tune threshold based on business risk tolerance

---

## 6. AMEX Competition Insight

The competition focuses on:
- Ranking customers by risk
- Identifying top-risk segments

---

## 7. Guidance for Team

- Prioritize recall for minority class
- Use ROC-AUC as main benchmark
- Evaluate trade-offs explicitly

---

## 8. Summary

Evaluation should align with:
- Risk detection
- Business impact
- Real-world deployment considerations
