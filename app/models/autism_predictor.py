import pickle
import numpy as np

class AutismPredictor:
    def __init__(self, model_path="app/ml_model/logistic_regression_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, data):
        data_array = np.array(data).reshape(1, -1)
        prediction = self.model.predict(data_array)[0]
        confidence = max(self.model.predict_proba(data_array)[0])  # Get max probability score
        return prediction, confidence
