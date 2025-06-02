from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

preprocessor = joblib.load("../artifacts/preprocessor.pkl")
model = joblib.load("../artifacts/model.pkl")

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

if __name__ == '__main__':
    app.run(debug=True)
