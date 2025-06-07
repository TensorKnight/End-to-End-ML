import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
from pathlib import Path


class TabNetTrainer:
    def __init__(self, data_path, model_dir):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = TabNetClassifier(verbose=1)
        self.scaler = StandardScaler()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        self.X = df.drop(columns=["Area Name", "Habitable"])
        self.y = df["Habitable"]
        if self.y.dtype != 'int':
            self.y = LabelEncoder().fit_transform(self.y)

    def preprocess(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X.values, self.y, test_size=0.2, random_state=42
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def train(self):
        self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess()
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], patience=10)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.save_artifacts(acc)
        print(f"Model accuracy: {acc:.4f}")

    def save_artifacts(self, accuracy):
        joblib.dump(self.model, self.model_dir / "tabnet_model.pkl")
        joblib.dump(self.scaler, self.model_dir / "tabnet_scaler.pkl")
        joblib.dump(accuracy, self.model_dir / "tabnet_accuracy.pkl")


if __name__ == "__main__":
    trainer = TabNetTrainer(
        data_path="../../notebooks/datasets/updated_df.csv",
        model_dir="../../artifacts"
    )
    trainer.train()
