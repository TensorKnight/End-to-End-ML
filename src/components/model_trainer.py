import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "LGBM": LGBMClassifier(verbose=-1, n_estimators=100),
                "XGBoost": XGBClassifier(verbosity=0, n_estimators=100, use_label_encoder=False),
                "CatBoost": CatBoostClassifier(verbose=0, n_estimators=100),
                "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
                "RandomForest": RandomForestClassifier(n_estimators=100),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=100),
                "LogisticRegression": LogisticRegression(max_iter=5000),
                "NaiveBayes": GaussianNB(),
                "SVM": SVC(kernel='linear', probability=True),
                "LDA": LinearDiscriminantAnalysis(),
                "RidgeClassifier": RidgeClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "AdaBoost": AdaBoostClassifier(n_estimators=100)
            }

        
            params = {
                 "LGBM": {"learning_rate": [0.01, 0.1], "num_leaves": [31, 50]},
                 "XGBoost": {"learning_rate": [0.01, 0.1], "max_depth": [3, 6]},
                 "CatBoost": {"learning_rate": [0.01, 0.1], "depth": [6, 10]},
                 "GradientBoosting": {"learning_rate": [0.01, 0.1]},
                 "RandomForest": {"n_estimators": [100, 200]},
                 "ExtraTrees": {"n_estimators": [100, 200]},
                 "LogisticRegression": {"C": [1.0, 0.1]},
                 "NaiveBayes": {},
                 "SVM": {"C": [0.1, 1.0]},
                 "LDA": {},
                 "RidgeClassifier": {"alpha": [1.0, 0.5]},
                 "DecisionTree": {"max_depth": [None, 10]},
                 "KNN": {"n_neighbors": [5, 10]},
                 "AdaBoost": {"learning_rate": [0.01, 0.1]}
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            # model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            f1 = f1_score(y_test, predicted, average='weighted')
            return f1

        except Exception as e:
            raise CustomException(e, sys)
