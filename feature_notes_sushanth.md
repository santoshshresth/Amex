                                                       ###Feature Engineering 

## Objective
The goal of this task was to create debt capacity and utilization signals to help identify risky customers.

## Approach
Since the dataset contains anonymized feature names, proxy features were created instead of relying on exact financial meanings.

## Features Created

### 1. Utilization Features
Ratios between related variables were created to represent usage relative to capacity.
Example:
- util_1 = B_1 / B_2
- util_2 = D_39 / P_2

### 2. Capacity Features
Differences between variables were used to represent remaining capacity.
Example:
- capacity_left = B_2 - B_1

### 3. Behavioral Features
Changes over time were captured using difference features.
Example:
- util_1_diff

### 4. Aggregation
All features were aggregated at the customer level using mean, max, and min to represent overall behavior.

## Conclusion
These features help capture customer risk based on how much they use, how much capacity remains, and how their behavior changes over time.
