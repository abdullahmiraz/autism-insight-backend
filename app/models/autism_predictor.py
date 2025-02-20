# app/models/autism_predictor.py
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

photo_size = 224  # Ensure this matches your training setup


# 🔹 Existing Class for Question-Based Prediction
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


class ImageAutismPredictor:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    def load_model(self, path: str):
        """Load the trained .pkl model."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Model not found at: {path}")

    def preprocess_image(self, image_path: str):
        """Preprocess image for prediction."""
        img = load_img(image_path, target_size=(photo_size, photo_size))
        img_array = img_to_array(img) / 255.0  # Normalize
        return img_array.flatten().reshape(1, -1)  # Flatten and reshape

    def predict_image(self, image_path: str):
        """Predict autism from a preprocessed image."""
        try:
            processed_image = self.preprocess_image(image_path)
            prediction = self.model.predict(processed_image)[0]
            confidence = max(self.model.predict_proba(processed_image)[0])
            label = "autistic" if prediction == 1 else "non_autistic"
            return label, confidence
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
