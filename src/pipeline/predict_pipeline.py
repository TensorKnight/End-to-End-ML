import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 area_name: str,
                 daily_min_temp: float,
                 daily_max_temp: float,
                 daily_avg_temp: float,
                 total_precipitation: float,
                 relative_humidity: float,
                 water_bodies: int,
                 population_density: int,
                 day_length: float,
                 urban_rural_area: int,
                 forested_area: int,
                 crop_area: int,
                 graze_land_area: int,
                 habitable: int,
                 natural_resource_score: int,
                 humidity_temp: float):

        self.area_name = area_name
        self.daily_min_temp = daily_min_temp
        self.daily_max_temp = daily_max_temp
        self.daily_avg_temp = daily_avg_temp
        self.total_precipitation = total_precipitation
        self.relative_humidity = relative_humidity
        self.water_bodies = water_bodies
        self.population_density = population_density
        self.day_length = day_length
        self.urban_rural_area = urban_rural_area
        self.forested_area = forested_area
        self.crop_area = crop_area
        self.graze_land_area = graze_land_area
        self.habitable = habitable
        self.natural_resource_score = natural_resource_score
        self.humidity_temp = humidity_temp

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Area Name": [self.area_name],
                "Daily Min Temp": [self.daily_min_temp],
                "Daily Max Temp": [self.daily_max_temp],
                "Daily Avg Temp": [self.daily_avg_temp],
                "Total Precipitation": [self.total_precipitation],
                "Relative Humidity": [self.relative_humidity],
                "Water Bodies": [self.water_bodies],
                "Population Density": [self.population_density],
                "Day Length": [self.day_length],
                "Urban / Rural Area": [self.urban_rural_area],
                "Forested Area": [self.forested_area],
                "Crop Area": [self.crop_area],
                "Graze Land Area": [self.graze_land_area],
                "Habitable": [self.habitable],
                "Natural Resource Score": [self.natural_resource_score],
                "Humidity_Temp": [self.humidity_temp],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
