# app/routes/detection.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.autism_predictor import AutismPredictor, ImageAutismPredictor
from app.schemas import AutismPredictionRequest, AutismPredictionResponse
from config import MODEL_PATHS
import shutil
import os

router = APIRouter()

# Ensure upload directories exist
UPLOAD_DIR = "uploads"
os.makedirs(f"{UPLOAD_DIR}/videos", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/images", exist_ok=True)

# Load models using centralized config
predictor = AutismPredictor(MODEL_PATHS["question_model"])
photo_predictor = ImageAutismPredictor(MODEL_PATHS["photo_model"])
video_predictor = AutismPredictor(MODEL_PATHS["video_model"])


# 1️⃣ Autism Detection from Questionnaire
@router.post("/predict", response_model=AutismPredictionResponse)
async def predict_autism(request: AutismPredictionRequest):
    try:
        # Convert request data to list of integers
        data = [getattr(request, field) for field in request.__fields__]
        prediction, confidence = predictor.predict(data)
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# 2️⃣ Video Prediction
@router.post("/predict-video", response_model=AutismPredictionResponse)
async def predict_from_video(video: UploadFile = File(...)):
    try:
        video_path = f"{UPLOAD_DIR}/videos/{video.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Mock video prediction (replace with actual video analysis)
        prediction, confidence = video_predictor.predict([1] * 10)
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process video: {str(e)}"
        )


# 3️⃣ Image Prediction (Multiple)
@router.post("/predict-images", response_model=AutismPredictionResponse)
async def predict_from_images(images: list[UploadFile] = File(...)):
    try:
        results = []
        for image in images:
            image_path = f"{UPLOAD_DIR}/images/{image.filename}"
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            # Predict from the image using the ImageAutismPredictor
            label, confidence = photo_predictor.predict_image(image_path)
            results.append({"label": label, "confidence": confidence})

        return {"prediction": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process images: {(e)}"
        )


# # 3️⃣ Image Prediction (Multiple)
# @router.post("/predict-images", response_model=AutismPredictionResponse)
# async def predict_from_images(images: list[UploadFile] = File(...)):
#     try:
#         for image in images:
#             image_path = f"{UPLOAD_DIR}/images/{image.filename}"
#             with open(image_path, "wb") as buffer:
#                 shutil.copyfileobj(image.file, buffer)

#         # Mock image prediction (replace with actual image analysis)
#         classes, [] = photo_predictor.predict([0, 1] * 5)
#         return {"prediction": classes, "confidence": []}
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Failed to process images: {str(e)}"
#         )
