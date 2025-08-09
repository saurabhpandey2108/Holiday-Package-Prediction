# Holiday Package Prediction

## 1. Overview
A machine learning pipeline to predict customer purchase decisions for holiday packages using demographic and behavioral data. This helps businesses optimize marketing campaigns and increase conversions.

## 2. Project Structure
- `notebook/`: Exploratory data analysis and model development notebooks  
- `src/`: Core scripts for preprocessing, modeling, evaluation  
- `app.py` / `appliccation.py`: Deployable web interface (Flask)  
- `templates/`: Front-end templates for the web app  
- `.ebextensions/`: AWS Elastic Beanstalk deployment configs  
- `requirements.txt` & `setup.py`: Dependency and package management 

## 3. Dataset Details
- **Source**: (Specify the origin or Kaggle link if applicable)
- **Size**: X samples, Y features + 1 target
- **Features**:
  - Age (years)  
  - Gender (categorical)  
  - Annual Income (₹ or $)  
  - Spending Score (1–100)  
  - Preferred Destination (categorical)  
  - Mode of Transport (categorical)  
  - Past Travel Experience (numeric/year range)  
- **Target**: Purchase Decision (Yes/No)

## 4. Preprocessing
- Handle missing data using (e.g., mean/mode imputation or drop)
- Encode categorical features with (e.g., one-hot, label encoding)
- Scaling applied to numeric features (MinMaxScaler / StandardScaler)
- Data split: Train (70%), Validation (15%), Test (15%)

## 5. Modeling Pipeline
- Algorithms compared:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - (If applicable) XGBoost / CatBoost  
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Best model: **Model X**, with parameters: `…`

## 6. Evaluation Metrics
- **Accuracy**: e.g., 82%  
- **Precision**: 78%  
- **Recall**: 85%  
- **F1-score**: 81%  
- ROC-AUC: 0.90  
- Include confusion matrix and ROC curve visuals

## 7. Feature Importance & Insights
- Most influential features: e.g., Annual Income, Spending Score, Past Travel Experience
- Interpretation: high-income customers with previous travel history are more likely to buy

## 8. Business Impact
- Targeting top 20% scored customers captured ~80% of buyers — optimizing campaign ROI
- Reduced Customer Acquisition Cost (CAC) from $X to $Y

## 9. Deployment
- Run the app:
  ```bash
  pip install -r requirements.txt
  python app.py
