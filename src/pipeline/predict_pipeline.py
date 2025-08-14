import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
	def __init__(self):
		pass

	def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Mirror training feature engineering for inference."""
		df = df.copy()
		if 'NumberOfPersonVisiting' in df.columns and 'NumberOfChildrenVisiting' in df.columns:
			df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
			df.drop(columns=['NumberOfPersonVisiting','NumberOfChildrenVisiting'], inplace=True)
		return df

	def predict(self,features):
		try:
			model_path=os.path.join("artifacts","model.pkl")
			preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
			model=load_object(file_path=model_path)
			preprocessor=load_object(file_path=preprocessor_path)
			features = self._apply_feature_engineering(features)
			data_scaled=preprocessor.transform(features)
			preds=model.predict(data_scaled)
			return preds
		
		except Exception as e:
			raise CustomException(e,sys)



class CustomData:
	def __init__(	self,
		Age: int,
		TypeofContact: str,
		CityTier: int,
		DurationOfPitch: float,
		Occupation: str,
		Gender: str,
		NumberOfPersonVisiting: int,
		NumberOfFollowups: float,
		ProductPitched: str,
		PreferredPropertyStar: float,
		MaritalStatus: str,
		NumberOfTrips: float,
		Passport: int,
		PitchSatisfactionScore: int,
		OwnCar: int,
		NumberOfChildrenVisiting: float,
		Designation: str,
		MonthlyIncome: float,
	):

		self.Age = Age
		self.TypeofContact = TypeofContact
		self.CityTier = CityTier
		self.DurationOfPitch = DurationOfPitch
		self.Occupation = Occupation
		self.Gender = Gender
		self.NumberOfPersonVisiting = NumberOfPersonVisiting
		self.NumberOfFollowups = NumberOfFollowups
		self.ProductPitched = ProductPitched
		self.PreferredPropertyStar = PreferredPropertyStar
		self.MaritalStatus = MaritalStatus
		self.NumberOfTrips = NumberOfTrips
		self.Passport = Passport
		self.PitchSatisfactionScore = PitchSatisfactionScore
		self.OwnCar = OwnCar
		self.NumberOfChildrenVisiting = NumberOfChildrenVisiting
		self.Designation = Designation
		self.MonthlyIncome = MonthlyIncome

	def get_data_as_data_frame(self):
		try:
			custom_data_input_dict = {
				"Age": [self.Age],
				"TypeofContact": [self.TypeofContact],
				"CityTier": [self.CityTier],
				"DurationOfPitch": [self.DurationOfPitch],
				"Occupation": [self.Occupation],
				"Gender": [self.Gender],
				"NumberOfPersonVisiting": [self.NumberOfPersonVisiting],
				"NumberOfFollowups": [self.NumberOfFollowups],
				"ProductPitched": [self.ProductPitched],
				"PreferredPropertyStar": [self.PreferredPropertyStar],
				"MaritalStatus": [self.MaritalStatus],
				"NumberOfTrips": [self.NumberOfTrips],
				"Passport": [self.Passport],
				"PitchSatisfactionScore": [self.PitchSatisfactionScore],
				"OwnCar": [self.OwnCar],
				"NumberOfChildrenVisiting": [self.NumberOfChildrenVisiting],
				"Designation": [self.Designation],
				"MonthlyIncome": [self.MonthlyIncome],
			}

			return pd.DataFrame(custom_data_input_dict)

		except Exception as e:
			raise CustomException(e, sys)

