#  Domain Understanding – AMEX Regulatory Expectations  

---

##  Overview  
This document outlines the key financial domain concepts and regulatory expectations relevant to building a credit risk prediction model using the AMEX dataset.

The goal is to ensure that the model is not only accurate but also **fair, interpretable, and compliant with real-world financial standards**.

---

##  Problem Context  
Credit risk modeling aims to predict whether a customer will default on their financial obligations based on historical behavioral data.

This is a critical function in financial institutions for:
- Credit approval decisions  
- Risk assessment  
- Customer profiling  

---

##  Regulatory Considerations  

###  Fairness  
The model must avoid bias and should not discriminate based on sensitive attributes such as gender, race, or ethnicity.

---

###  Explainability  
Model predictions must be interpretable. Financial institutions should be able to explain:
> Why a customer was classified as high-risk.

Techniques:
- SHAP  
- LIME  

---

###  Data Privacy  
Customer data must be handled securely, ensuring no exposure of sensitive or personal information.

---

###  No Data Leakage  
Only past and present data should be used. Future information must not influence model predictions.

---

###  Handling Class Imbalance  
Default cases are relatively rare. Therefore:
- Accuracy is not a reliable metric  
- ROC-AUC is preferred for evaluation  

---

##  Dataset Understanding  

- Time-series dataset with multiple records per customer  
- Target variable defined at customer level  
- Features represent anonymized financial behavior  

### Key Insight  
> Recent customer behavior is more predictive of default than historical averages.

---

##  Role & Contribution  

As part of Team 2, the focus was on:

- Aligning feature engineering with domain knowledge  
- Ensuring no biased or irrelevant features are introduced  
- Supporting explain ability for downstream modeling  
- Maintaining compliance with financial regulations  

---
