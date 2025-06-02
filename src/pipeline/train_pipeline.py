import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

def main():
    try:
        logging.info("Starting the training pipeline")

        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")

        transformer = DataTransformation()
        train_array, test_array, _ = transformer.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data transformation completed")

        trainer = ModelTrainer()
        f1 = trainer.initiate_model_trainer(train_array, test_array)
        logging.info(f"Model training completed with F1 Score: {f1}")

    except Exception as e:
        logging.error("Exception occurred in the training pipeline")
        raise CustomException(e, sys)

