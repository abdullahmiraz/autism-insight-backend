import tensorflow as tf
import numpy as np
import cv2
import os
import h5py

# Paths for models
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "../ml_model/efficient_net_B7_model.h5")
VIDEO_MODEL_PATH = os.path.join(BASE_DIR, "../ml_model/autism-S-224-89.33.h5")

def load_trained_model(model_path):
    """Load the full model if possible, otherwise load weights into a compatible architecture."""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Successfully loaded full model from {model_path}")
        return model
    except Exception as e:
        print(f"⚠️ Failed to load full model. Trying to load weights. Error: {e}")

        with h5py.File(model_path, "r") as f:
            layers = list(f["model_weights"].keys())
            print("🔍 Available layers in saved weights:", layers)

        base_model = tf.keras.applications.EfficientNetB7(weights=None, include_top=True)
        base_model.load_weights(model_path, by_name=True, skip_mismatch=True)
        print("✅ Weights loaded successfully with layer mismatch handling.")
        return base_model

# Load models safely
image_model = load_trained_model(IMAGE_MODEL_PATH)
video_model = load_trained_model(VIDEO_MODEL_PATH)

def preprocess_image(image_path, target_size=(600, 600)):
    """Preprocess an image for model prediction."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size) / 255.0
    return np.expand_dims(image, axis=0)



# def preprocess_image(image_path, target_size=(224, 224)):
#     """Preprocess an image for model prediction."""
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"❌ Image not found: {image_path}")

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, target_size) / 255.0
#     return np.expand_dims(image, axis=0)



def predict_image(image_path):
    """Predict autism from an image."""
    try:
        processed_image = preprocess_image(image_path)
        prediction = image_model.predict(processed_image)
        confidence = float(prediction[0][0])
        return {"prediction": int(confidence > 0.5), "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

def preprocess_video(video_path, target_size=(600, 600), num_frames=10):
    """Extract frames for model inference and process each one to 600x600 size."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size) / 255.0
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"❌ No valid frames extracted from video: {video_path}")

    return np.array(frames)

def predict_video(video_path):
    """Predict autism from video using resized frames."""
    try:
        frames = preprocess_video(video_path)
        predictions = image_model.predict(frames)

        confidence = float(np.mean(predictions))
        return {"prediction": int(confidence > 0.5), "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}
