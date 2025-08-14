import os
import sys
import json
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_object


@dataclass
class SegmentationConfig:
	segmenter_path: str = os.path.join('artifacts', 'segmenter.pkl')
	segmenter_meta_path: str = os.path.join('artifacts', 'segmenter_meta.json')


class CustomerSegmentationTrainer:
	def __init__(self):
		self.config = SegmentationConfig()

	def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Mirror transformation step: create TotalVisiting and drop its components if present."""
		df = df.copy()
		if 'NumberOfPersonVisiting' in df.columns and 'NumberOfChildrenVisiting' in df.columns:
			df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
			df.drop(columns=['NumberOfPersonVisiting','NumberOfChildrenVisiting'], inplace=True)
		return df

	def train(self, train_csv_path: str, test_csv_path: str, preprocessor_path: str) -> dict:
		try:
			# Load datasets
			train_df = pd.read_csv(train_csv_path)
			test_df = pd.read_csv(test_csv_path)
			full_df = pd.concat([train_df, test_df], ignore_index=True)

			# Separate features and apply same preprocessing
			target = 'ProdTaken'
			X_full = full_df.drop(columns=[target], errors='ignore')
			X_full = self._apply_feature_engineering(X_full)
			preprocessor = load_object(preprocessor_path)
			X_full_transformed = preprocessor.transform(X_full)

			# Try different cluster counts and pick by silhouette score
			best_k = None
			best_score = -1.0
			best_model = None
			for k in [3, 4, 5, 6]:
				km = KMeans(n_clusters=k, n_init=10, random_state=42)
				labels = km.fit_predict(X_full_transformed)
				score = silhouette_score(X_full_transformed, labels)
				logging.info(f"KMeans(k={k}) silhouette: {score:.4f}")
				if score > best_score:
					best_score = score
					best_k = k
					best_model = km

			# Persist best model
			save_object(self.config.segmenter_path, best_model)

			# Build human-readable profiles on original feature space
			labels = best_model.predict(X_full_transformed)
			profile_df = X_full.copy()
			profile_df['segment'] = labels
			# Simple numeric means for quick profiling
			numeric_cols = profile_df.select_dtypes(exclude='object').columns.tolist()
			profiles_df = profile_df.groupby('segment')[numeric_cols].mean()

			# Simple strategy suggestions per segment (based on income and trips)
			segment_summaries = {}
			for seg, row in profiles_df.iterrows():
				mi = float(row.get('MonthlyIncome', 0.0))
				tr = float(row.get('NumberOfTrips', 0.0))
				pps = float(row.get('PreferredPropertyStar', 0.0))
				if mi >= float(profiles_df.get('MonthlyIncome', pd.Series([mi])).median()):
					strategy = 'Premium bundle offer (Deluxe/King), loyalty perks, concierge call'
				elif tr >= float(profiles_df.get('NumberOfTrips', pd.Series([tr])).median()):
					strategy = 'Frequent-traveler discount, upsell to higher star property'
				else:
					strategy = 'Entry-level Basic/Standard with limited-time discount via email/SMS'
				segment_summaries[str(int(seg))] = {
					'avg_monthly_income': round(mi, 2),
					'avg_trips': round(tr, 2),
					'avg_pref_star': round(pps, 2),
					'strategy': strategy,
				}

			meta = {
				'best_k': int(best_k),
				'silhouette': float(best_score),
				'profiles': segment_summaries,
			}
			with open(self.config.segmenter_meta_path, 'w', encoding='utf-8') as f:
				json.dump(meta, f, indent=2)

			logging.info(f"Saved segmenter (k={best_k}) and meta")
			return meta
		except Exception as e:
			raise CustomException(e, sys) 