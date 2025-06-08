from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score

app = Flask(__name__)

preprocessor = joblib.load("../artifacts/preprocessor.pkl")
default_model = joblib.load("../artifacts/model.pkl")
tabnet_model = joblib.load("../artifacts/tabnet_model.pkl")

data = pd.read_csv("../artifacts/data.csv")
X = data.drop(columns=["Habitable", "Area Name"])
y = data["Habitable"]
X_transformed = preprocessor.transform(X)

try:
    default_model_pred = default_model.predict(X_transformed)
    default_model_accuracy = accuracy_score(y, default_model_pred)
except:
    default_model_accuracy = 0.0

try:
    tabnet_model_pred = tabnet_model.predict(X_transformed)
    tabnet_model_accuracy = accuracy_score(y, tabnet_model_pred)
except:
    tabnet_model_accuracy = 0.0

model = tabnet_model if tabnet_model_accuracy > default_model_accuracy else default_model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    natural_resource_score = (
        data['forested_area'] +
        data['water_bodies'] +
        data['crop_area'] +
        data['graze_land_area']
    )

    humidity_temp = data['relative_humidity'] * data['daily_avg_temp']

    features_df = pd.DataFrame([{
        'Area Name': 'Area 1',
        'Daily Min Temp': data['daily_min_temp'],
        'Daily Max Temp': data['daily_max_temp'],
        'Daily Avg Temp': data['daily_avg_temp'],
        'Total Precipitation': data['total_precipitation'],
        'Relative Humidity': data['relative_humidity'],
        'Water Bodies': data['water_bodies'],
        'Population Density': data['population_density'],
        'Day Length': data['day_length'],
        'Urban / Rural Area': data['urban_rural_area'],
        'Forested Area': data['forested_area'],
        'Crop Area': data['crop_area'],
        'Graze Land Area': data['graze_land_area'],
        'Habitable': 1,
        'Natural Resource Score': natural_resource_score,
        'Humidity_Temp': humidity_temp
    }])

    features_transformed = preprocessor.transform(features_df)
    prediction = model.predict(features_transformed)[0]

    return jsonify({'prediction': float(prediction)})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback_data = {k: data[k] for k in data if k != 'city_name'}

    feedback_df = pd.DataFrame([feedback_data])
    file_path = "feedback/feedback.csv"

    if not os.path.exists(file_path):
        feedback_df.to_csv(file_path, index=False)
    else:
        feedback_df.to_csv(file_path, mode='a', header=False, index=False)

    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
