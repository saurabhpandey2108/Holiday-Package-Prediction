# Holiday Package Prediction

## Overview
Predict customer purchases of holiday packages using machine learning, enabling businesses to target the right customers efficiently and reduce marketing costs.

## Dataset & Features
List and explain features like Age, CityTier, ProductPitched, Passport, MonthlyIncome, etc. Describe the target variable `ProdTaken`.

## Smart Pipeline
1. Data Cleaning & Preprocessing  
2. Feature Selection & Engineering  
3. Modeling: Logistic Regression, Random Forest, XGBoost, etc.  
4. Evaluation: Accuracy, Recall, Precision, F1-score, and Gain/Lift Analysis  

## Results
| Model         | Recall | Precision | Accuracy |
|---------------|--------|-----------|----------|
| XGBoost (best)| 0.85   | 0.60      | 0.75     |

Top features: Passport, Product Pitched, CityTier, Marital Status (Single), etc.

## Business Impact
- Reached 80% of potential buyers by targeting top 20%.
- Lowered CAC from $5.31 to $1.25, improving ROI.

## Recommendations
- Focus on high-potential segments (passport holders, tier-3 city, single, basic pitch).  
- Develop unique campaigns for low-potential groups.  
- Consider expanding features, testing advanced models, and deploying via web app.

## Installation
```bash
pip install -r requirements.txt
# or
pip install -e .
