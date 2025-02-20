# app/models/autism_predictor.py
import pickle
import numpy as np

class AutismPredictor:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    def load_model(self, path: str):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Model not found at: {path}")

    def predict(self, data):
        data_array = np.array(data).reshape(1, -1)
        prediction = self.model.predict(data_array)[0]
        confidence = max(self.model.predict_proba(data_array)[0])
        return prediction, confidence
