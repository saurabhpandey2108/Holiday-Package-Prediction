import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
	AdaBoostClassifier,
	GradientBoostingClassifier,
	RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
	trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
	def __init__(self):
		self.model_trainer_config=ModelTrainerConfig()


	def initiate_model_trainer(self,train_array,test_array):
		try:
			logging.info("Split training and test input data")
			X_train,y_train,X_test,y_test=(
				train_array[:,:-1],
				train_array[:,-1],
				test_array[:,:-1],
				test_array[:,-1]
			)
			models = {
				"Logistic Regression": LogisticRegression(max_iter=1000),
				"Random Forest": RandomForestClassifier(),
				"Decision Tree": DecisionTreeClassifier(),
				"Gradient Boosting": GradientBoostingClassifier(),
				"XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
				"AdaBoost Classifier": AdaBoostClassifier(),
			}
			params={
				"Logistic Regression": {
					'C': [0.1, 1.0, 10.0],
					'solver': ['lbfgs', 'liblinear']
				},
				"Random Forest":{
					'n_estimators': [200, 500, 1000],
					'min_samples_split': [2, 5, 10]
				},
				"Decision Tree": {
					'criterion':['gini', 'entropy', 'log_loss'],
				},
				"Gradient Boosting":{
					'learning_rate':[.1,.01,.05,.001],
					'n_estimators': [64,128,256]
				},
				"XGBClassifier":{
					'learning_rate':[.1,.01,.05,.001],
					'n_estimators': [64,128,256]
				},
				"AdaBoost Classifier":{
					'learning_rate':[.1,.01,0.5,.001],
					'n_estimators': [64,128,256]
				}
			}

			model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
											 models=models,param=params, classification=True)
			
			## To get best model score from dict
			best_model_score = max(sorted(model_report.values()))

			## To get best model name from dict

			best_model_name = list(model_report.keys())[ 
				list(model_report.values()).index(best_model_score)
			]
			best_model = models[best_model_name]

			if best_model_score<0.6:
				raise CustomException("No best model found")
			logging.info(f"Best found model on both training and testing dataset")

			save_object(
				file_path=self.model_trainer_config.trained_model_file_path,
				obj=best_model
			)

			predicted=best_model.predict(X_test)

			score = f1_score(y_test, predicted)
			return score
			
		except Exception as e:
			raise CustomException(e,sys)