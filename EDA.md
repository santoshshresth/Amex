# Exploratory Data Analysis (EDA) – AMEX Dataset  



##  Overview  
This analysis explores the AMEX dataset to understand feature distributions, identify data patterns, and examine relationships between customer behavior and default risk.

The goal is to extract insights that guide feature engineering and model development.

---

##  Target Distribution  

The dataset is imbalanced, with a higher proportion of non-default customers compared to defaulters.

> This reflects real-world financial systems where defaults are relatively rare but critical.

---

##  Feature Distribution Analysis  

###  P_2 (Payment Behavior)  
- Values concentrated between 0.6–0.9  
- Slight skewness with few extreme values  

**Insight:**  
Most customers show stable payment behavior, but lower values indicate potential risk.

---

###  D_39 (Delinquency Indicator)  
- Highly right-skewed  
- Majority near 0 with long tail  

**Insight:**  
Most customers have low delinquency, but high values indicate severe financial stress.

---

###  B_1 (Balance Feature)  
- Strong right skew  
- Majority near 0 with extreme values  

**Insight:**  
Higher balances suggest over-leverage and increased default risk.

---

###  B_2 (Balance / Utilization)  
- Distinct distribution patterns  
- Peaks at higher values  

**Insight:**  
Indicates different customer usage patterns, separating risk profiles.

---

##  Feature vs Target Analysis  

###  P_2 vs Target  
- Default customers show lower values  

**Insight:**  
Lower payment-related values indicate weaker repayment behavior.

---

###  D_39 vs Target  
- Default customers have higher spread and values  

**Insight:**  
Higher delinquency strongly correlates with default risk.

---

###  B_1 vs Target  
- Default customers have higher balances  

**Insight:**  
Higher balances indicate financial stress and risk.

---

###  B_2 vs Target  
- Clear separation between classes  

**Insight:**  
This feature is highly predictive of default behavior.

---

##  Outlier Analysis  

- Significant outliers present across multiple features  
- Especially in delinquency and balance-related variables  

**Insight:**  
Outliers represent extreme financial behavior and are important for identifying high-risk customers.

---

##  Time-Series Insight  

- Each customer has ~13 records on average  

**Insight:**  
The dataset has a time-series structure, making customer behavior over time critical for prediction.

---

##  Domain Insights  

- Financial behavior varies significantly across customers  
- Payment, balance, and delinquency features capture risk patterns  
- Behavioral trends over time are key indicators of default  

---

##  Model Evaluation Insight  

Due to class imbalance:  
- Accuracy is not reliable  
- ROC-AUC and Recall are preferred  

**Insight:**  
Evaluation should focus on identifying high-risk customers effectively.

---

##  Explainability Insight  

- Features are anonymized  
- Distribution analysis helps identify important variables  
---

##  Key Findings  

1. Features are highly skewed with long tails  
2. Clear differences between default and non-default customers  
3. Outliers represent high-risk behavior  
4. Payment and delinquency features are strong predictors  
5. Dataset is time-series and requires aggregation  

---

##  Conclusion  

EDA reveals strong behavioral patterns in the dataset, with clear distinctions between default and non-default customers.  

The presence of skewness, outliers, and temporal structure highlights the need for:
- Feature transformation  
- Behavioral feature engineering  
- Customer-level aggregation  


---
