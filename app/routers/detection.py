from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List  #Import List for correct type hinting
from app.models.autism_predictor import predict_image, predict_video
from app.schemas import AutismPredictionRequest, AutismPredictionResponse
import shutil
import os

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(f"{UPLOAD_DIR}/videos", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/images", exist_ok=True)


# ! Question Prediction (Mock - Replace with Real Logic) ###########################

@router.post("/predict", response_model=AutismPredictionResponse)
async def predict_autism(request: AutismPredictionRequest):
    """
    Mock endpoint for Autism prediction based on request data.
    REPLACE this with your actual autism prediction logic.
    """
    try:
        data = [getattr(request, field) for field in request.__fields__]
        prediction = int(sum(data) > len(data) / 2)
        confidence = sum(data) / len(data)
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ! Video Prediction ###########################

@router.post("/predict-video", response_model=AutismPredictionResponse)
async def predict_from_video(video: UploadFile = File(...)):
    """
    Endpoint to predict autism from an uploaded video.
    Uses the predict_video function from the autism_predictor module.
    """
    try:
        video_path = os.path.join(UPLOAD_DIR, "videos", video.filename)  # Secure file path
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)  # Efficient file copy

        result = predict_video(video_path)
        os.remove(video_path)  #Cleanup after use

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return AutismPredictionResponse(prediction=result["prediction"], confidence=result["confidence"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ! Image Prediction (Multiple Images) ###########################

@router.post("/predict-images", response_model=List[AutismPredictionResponse])
async def predict_from_images(images: List[UploadFile] = File(...)):
    """
    Endpoint to predict autism from a list of uploaded images.
    Uses the predict_image function from the autism_predictor module.
    """
    try:
        if not images:
            raise HTTPException(status_code=400, detail="No images uploaded for prediction.")

        predictions: List[AutismPredictionResponse] = [] #Explicit Type Hint

        for image in images:
            image_path = os.path.join(UPLOAD_DIR, "images", image.filename) #Secure path
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)  #Copy image

            result = predict_image(image_path) #Make Prediction
            os.remove(image_path)  #Clean UP

            if "error" in result:
                print(f"Warning: Skipping image {image.filename} due to error: {result['error']}")
                continue  # Skip if there's an error

            predictions.append(AutismPredictionResponse(prediction=result["prediction"], confidence=result["confidence"])) #Append

        if not predictions:
            raise HTTPException(status_code=500, detail="Failed to predict any images.")

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))