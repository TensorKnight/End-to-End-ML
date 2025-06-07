import pandas as pd
import joblib
from pathlib import Path

class TabNetPredictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "tabnet_model.pkl")
        self.scaler = joblib.load(self.model_dir / "tabnet_scaler.pkl")
        self.accuracy = joblib.load(self.model_dir / "tabnet_accuracy.pkl")

    def _engineer_features(self, raw_data):
        natural_resource_score = (
            raw_data['Forested Area'] +
            raw_data['Water Bodies'] +
            raw_data['Crop Area'] +
            raw_data['Graze Land Area']
        )
        humidity_temp = raw_data['Relative Humidity'] * raw_data['Daily Avg Temp']

        raw_data['Natural Resource Score'] = natural_resource_score
        raw_data['Humidity_Temp'] = humidity_temp

        return raw_data

    def predict(self, input_dict):
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.drop(columns=["Area Name"])
        input_df = self._engineer_features(input_df)
        input_scaled = self.scaler.transform(input_df.values)
        prediction = self.model.predict(input_scaled)
        return {
            "prediction": int(prediction[0]),
            "accuracy": round(float(self.accuracy), 4)
        }


if __name__ == "__main__":
    sample_input = {
        "Area Name": "Area X",
        "Daily Min Temp": 24.5,
        "Daily Max Temp": 36.1,
        "Daily Avg Temp": 30.3,
        "Total Precipitation": 15.0,
        "Relative Humidity": 60.5,
        "Water Bodies": 1,
        "Population Density": 5000,
        "Day Length": 13.2,
        "Urban / Rural Area": 0,
        "Forested Area": 1,
        "Crop Area": 1,
        "Graze Land Area": 0
    }

    predictor = TabNetPredictor(model_dir="../../artifacts")
    result = predictor.predict(sample_input)
    print(result)
