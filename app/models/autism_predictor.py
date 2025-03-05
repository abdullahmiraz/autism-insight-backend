import tensorflow as tf
import numpy as np
import cv2
import os
import h5py
from collections import Counter  # For majority vote
from PIL import Image
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths for models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "../ml_model/autism_detection_model.h5")  # Trained model path

IMAGE_SIZE = (224, 224)  # Consistent image size
NUM_FRAMES = 10  # Number of frames to extract from video

def create_model(image_size):
    """Creates the EfficientNetB7 model."""
    base_model = EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_shape=(image_size[0], image_size[1], 3)
    )

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x) # Add BatchNormalization
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

    model = Model(inputs=base_model.input, outputs=predictions)
    return model
# Load the trained image model

image_model = create_model(IMAGE_SIZE)

# Load weights *only*
try:
    image_model.load_weights(IMAGE_MODEL_PATH)
    print(f"✅ Successfully loaded model weights from {IMAGE_MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model weights. Error: {e}")
    exit()

print('Successfully loaded the model')

def preprocess_image(image_path):
    """Preprocess an image for model prediction."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"❌ Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def predict_image(image_path):
    """Predict autism from an image."""
    try:
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return {"error": "Failed to preprocess image"}

        prediction = image_model.predict(processed_image)
        confidence = float(prediction[0][0])  # Sigmoid output
        return {"prediction": int(confidence > 0.5), "confidence": confidence}  # Simple threshold
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        return {"error": str(e)}

def preprocess_video(video_path):
    """Extract frames for model inference and process each one."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)  # Evenly spaced

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, IMAGE_SIZE)
                frame = tf.keras.applications.efficientnet.preprocess_input(frame)
                frames.append(frame)
            else:
                print(f"⚠️ Warning: Could not read frame {idx}. Skipping.")

        cap.release()

        if not frames:
            raise ValueError(f"❌ No valid frames extracted from video: {video_path}")

        return np.array(frames)

    except Exception as e:
        print(f"❌ Error preprocessing video: {e}")
        return None

def predict_video(video_path):
    """Predict autism from video using image model and majority vote."""
    try:
        frames = preprocess_video(video_path)
        if frames is None:
            return {"error": "Failed to preprocess video"}

        predictions = image_model.predict(frames)  # Use image_model (EfficientNetB7)
        binary_predictions = (predictions > 0.5).astype(int)  # Threshold at 0.5

        # Majority Vote
        counts = Counter(binary_predictions.flatten())  # Count 0's and 1's
        majority_prediction = counts.most_common(1)[0][0]  # 0 or 1
        confidence = counts[majority_prediction] / len(binary_predictions)  # Fraction of votes

        return {"prediction": int(majority_prediction), "confidence": float(confidence)}

    except Exception as e:
        print(f"❌ Error predicting video: {e}")
        return {"error": str(e)}