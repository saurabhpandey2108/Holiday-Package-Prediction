from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import json
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
	return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
	if request.method=='GET':
		return render_template('home.html')
	else:
		data=CustomData(
			Age=int(request.form.get('Age')),
			TypeofContact=request.form.get('TypeofContact'),
			CityTier=int(request.form.get('CityTier')),
			DurationOfPitch=float(request.form.get('DurationOfPitch')),
			Occupation=request.form.get('Occupation'),
			Gender=request.form.get('Gender'),
			NumberOfPersonVisiting=int(request.form.get('NumberOfPersonVisiting')),
			NumberOfFollowups=float(request.form.get('NumberOfFollowups')),
			ProductPitched=request.form.get('ProductPitched'),
			PreferredPropertyStar=float(request.form.get('PreferredPropertyStar')),
			MaritalStatus=request.form.get('MaritalStatus'),
			NumberOfTrips=float(request.form.get('NumberOfTrips')),
			Passport=int(request.form.get('Passport')),
			PitchSatisfactionScore=int(request.form.get('PitchSatisfactionScore')),
			OwnCar=int(request.form.get('OwnCar')),
			NumberOfChildrenVisiting=float(request.form.get('NumberOfChildrenVisiting')),
			Designation=request.form.get('Designation'),
			MonthlyIncome=float(request.form.get('MonthlyIncome')),
		)
		
		pred_df=data.get_data_as_data_frame()
		print(pred_df)

		predict_pipeline=PredictPipeline()
		results=predict_pipeline.predict(pred_df)
		label = 'Will Purchase' if int(results[0])==1 else 'Will Not Purchase'
		return render_template('home.html',results=label)

@app.route('/segments')
def view_segments():
	meta_path = os.path.join('artifacts','segmenter_meta.json')
	if not os.path.exists(meta_path):
		return render_template('segments.html', meta=None, profiles=None)
	with open(meta_path, 'r', encoding='utf-8') as f:
		meta = json.load(f)
	profiles = meta.get('profiles', {})
	return render_template('segments.html', meta=meta, profiles=profiles)


if __name__=="__main__":      
	app.run(host="0.0.0.0",debug=True)        


