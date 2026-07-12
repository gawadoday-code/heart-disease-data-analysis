# Data Dictionary

| Column | Type | Description | Use |
|---|---|---|---|
| id | Identifier | Record identifier | Excluded from analysis/modeling |
| age | Numeric | Age in years | Analysis and modeling |
| sex | Categorical | Male/Female | Analysis and modeling |
| dataset | Categorical | Source hospital/dataset | Stratification and validation |
| cp | Categorical | Chest pain type | Analysis and modeling |
| trestbps | Numeric | Resting blood pressure | Analysis and modeling with caution |
| chol | Numeric | Serum cholesterol | Source-dependent missingness |
| fbs | Categorical | Fasting blood sugar indicator | Analysis and modeling |
| restecg | Categorical | Resting ECG category | Analysis and modeling |
| thalch | Numeric | Maximum heart rate achieved | Analysis and modeling |
| exang | Categorical | Exercise-induced angina | Analysis and modeling |
| oldpeak | Numeric | ST depression induced by exercise | Analysis and modeling |
| slope | Categorical | Slope of peak exercise ST segment | Analysis and modeling with caution |
| ca | Numeric/categorical | Number of major vessels | Secondary analysis only |
| thal | Categorical | Thallium test result | Secondary analysis only |
| num | Target source | Original disease coding 0–4 | Target only; excluded from predictors |
| disease_present | Binary target | `Yes` when `num > 0` | Modeling target |
