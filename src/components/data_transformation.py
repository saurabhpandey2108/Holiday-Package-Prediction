import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer ##for missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
	preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
	def __init__(self):
		self.data_transformation_config=DataTransformationConfig()

	def get_data_transformer_object(self, train_df: pd.DataFrame):
		'''
		Build preprocessing transformer based on dataset schema from notebook:
		- OneHotEncoder(drop='first') for categorical
		- StandardScaler for numeric
		- Median imputation for numeric, Most frequent for categorical
		'''
		try:
			# derive features excluding target
			target_column_name = "ProdTaken"
			feature_df = train_df.drop(columns=[target_column_name], errors='ignore')
			categorical_columns = feature_df.select_dtypes(include="object").columns.tolist()
			numerical_columns = feature_df.select_dtypes(exclude="object").columns.tolist()

			num_pipeline= Pipeline(
				steps=[
				("imputer",SimpleImputer(strategy="median")),
				("scaler",StandardScaler())

				]
			)

			cat_pipeline=Pipeline(

				steps=[
				("imputer",SimpleImputer(strategy="most_frequent")),
				("one_hot_encoder",OneHotEncoder(drop='first', handle_unknown='ignore')),
				("scaler",StandardScaler(with_mean=False))
				]

			)

			logging.info(f"Categorical columns: {categorical_columns}")
			logging.info(f"Numerical columns: {numerical_columns}")

			preprocessor=ColumnTransformer(
				[
				("num_pipeline",num_pipeline,numerical_columns),
				("cat_pipelines",cat_pipeline,categorical_columns)

				]


			)

			return preprocessor
		
		except Exception as e:
			raise CustomException(e,sys)
		
	def initiate_data_transformation(self,train_path,test_path):

		try:
			train_df=pd.read_csv(train_path)
			test_df=pd.read_csv(test_path)

			logging.info("Read train and test data completed")

			# Feature engineering to match notebook: create TotalVisiting and drop components
			for df in (train_df, test_df):
				if 'NumberOfPersonVisiting' in df.columns and 'NumberOfChildrenVisiting' in df.columns:
					df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
					df.drop(columns=['NumberOfPersonVisiting','NumberOfChildrenVisiting'], inplace=True)

			logging.info("Obtaining preprocessing object")

			preprocessing_obj=self.get_data_transformer_object(train_df)

			target_column_name="ProdTaken" ##dependent variable

			if target_column_name not in train_df.columns:
				raise CustomException(f"Target column '{target_column_name}' not found in training data", sys)

			input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
			target_feature_train_df=train_df[target_column_name]

			input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
			target_feature_test_df=test_df[target_column_name]

			logging.info(
				f"Applying preprocessing object on training dataframe and testing dataframe."
			)

			input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
			input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

			train_arr = np.c_[
				input_feature_train_arr, np.array(target_feature_train_df)
			]
			test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

			logging.info(f"Saved preprocessing object.")

			save_object(

				file_path=self.data_transformation_config.preprocessor_obj_file_path,
				obj=preprocessing_obj

			)

			return (
				train_arr,
				test_arr,
				self.data_transformation_config.preprocessor_obj_file_path,
			)
		except Exception as e:
			raise CustomException(e,sys)
