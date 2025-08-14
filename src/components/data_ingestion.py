import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
	train_data_path: str=os.path.join('artifacts',"train.csv")
	test_data_path: str=os.path.join('artifacts',"test.csv")
	raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
	def __init__(self):
		self.ingestion_config=DataIngestionConfig()

	def _read_source_dataframe(self) -> pd.DataFrame:
		"""Read source data from notebook data folder (supports xls/xlsx/csv).
		Falls back to CSV parser if Excel engine fails (handles mislabelled .xls files)."""
		possible_paths = [
			os.path.join('notebook', 'data', 'Travel.xls'),
			os.path.join('notebook', 'data', 'Travel.xlsx'),
			os.path.join('notebook', 'data', 'Travel.csv'),
			os.path.join('notebook', 'Travel.csv'),
		]
		for path in possible_paths:
			if os.path.exists(path):
				logging.info(f"Loading dataset from {path}")
				lower = path.lower()
				try:
					if lower.endswith('.xls'):
						try:
							return pd.read_excel(path, engine='xlrd')
						except Exception:
							# Fallback: try as CSV if file is not a real XLS
							logging.info("Reading .xls with csv fallback")
							return pd.read_csv(path)
					elif lower.endswith('.xlsx'):
						try:
							return pd.read_excel(path, engine='openpyxl')
						except Exception:
							logging.info("Reading .xlsx with csv fallback")
							return pd.read_csv(path)
					else:
						return pd.read_csv(path)
				except ImportError as ie:
					raise CustomException(
						f"Missing Excel engine for '{path}'. Install required package (e.g., pip install xlrd for .xls or openpyxl for .xlsx). Original error: {ie}",
						sys,
					)
				except Exception as e:
					raise CustomException(e, sys)
		raise CustomException(f"Source dataset not found in expected locations: {possible_paths}", sys)

	def initiate_data_ingestion(self):
		logging.info("Entered the data ingestion component")
		try:
			df=self._read_source_dataframe()
			logging.info('Read the dataset as dataframe')

			os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

			df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

			logging.info("Train-test split initiated")
			train_set,test_set=train_test_split(df,test_size=0.2,random_state=42,stratify=df['ProdTaken'] if 'ProdTaken' in df.columns else None)

			train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

			test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

			logging.info("Ingestion of the data is completed")

			return(
				self.ingestion_config.train_data_path,
				self.ingestion_config.test_data_path

			)
		except Exception as e:
			raise CustomException(e,sys)
		
if __name__=="__main__":
	obj=DataIngestion()
	train_data,test_data=obj.initiate_data_ingestion()

	data_transformation=DataTransformation()
	train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

	modeltrainer=ModelTrainer()
	print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



