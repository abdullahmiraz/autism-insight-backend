from fastapi import APIRouter, HTTPException, UploadFile, File
# from app.models.autism_predictor import predict_image
from app.models.autism_predictor import predict_image, predict_video
from app.schemas import AutismPredictionRequest, AutismPredictionResponse
import shutil
import os

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(f"{UPLOAD_DIR}/videos", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/images", exist_ok=True)


# ! Question Prediction ###########################

@router.post("/predict", response_model=AutismPredictionResponse)
async def predict_autism(request: AutismPredictionRequest):
    try:
        data = [getattr(request, field) for field in request.__fields__]
        prediction = int(sum(data) > len(data) / 2)
        confidence = sum(data) / len(data)
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# !video prediction ###########################

@router.post("/predict-video", response_model=AutismPredictionResponse)
async def predict_from_video(video: UploadFile = File(...)):
    try:
        video_path = f"{UPLOAD_DIR}/videos/{video.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        result = predict_video(video_path)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


 # ! image prediction ###########################

@router.post("/predict-images", response_model=list[AutismPredictionResponse])
async def predict_from_images(images: list[UploadFile] = File(...)):
    try:
        if not images:
            raise HTTPException(status_code=400, detail="No images uploaded for prediction.")

        predictions = []

        for image in images:
            image_path = f"{UPLOAD_DIR}/images/{image.filename}"
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            result = predict_image(image_path)

            if "error" in result:
                continue  # Skip if there's an error with an image

            # Append each result in the desired format
            predictions.append(AutismPredictionResponse(
                prediction=result.get("prediction", 0),
                confidence=result.get("confidence", 0.0)
            ))

        if not predictions:
            raise HTTPException(status_code=500, detail="Failed to predict any images.")

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# single image prediction working fine below this
# @router.post("/predict-images", response_model=dict)
# async def predict_from_images(images: list[UploadFile] = File(...)):
#     try:
#         if len(images) != 1:
#             raise HTTPException(status_code=400, detail="Please upload exactly one image for prediction.")

#         image = images[0]
#         image_path = f"{UPLOAD_DIR}/images/{image.filename}"
#         with open(image_path, "wb") as buffer:
#             shutil.copyfileobj(image.file, buffer)

#         result = predict_image(image_path)

#         if "error" in result:
#             raise HTTPException(status_code=500, detail=result["error"])

#         # Return a dictionary matching the expected format
#         return {
#             "prediction": result.get("prediction", 0),
#             "confidence": result.get("confidence", 0.0)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))




# @router.post("/predict-images", response_model=dict)
# async def predict_from_images(images: list[UploadFile] = File(...)):
#     try:
#         results = []
#         for image in images:
#             image_path = f"{UPLOAD_DIR}/images/{image.filename}"
#             with open(image_path, "wb") as buffer:
#                 shutil.copyfileobj(image.file, buffer)

#             result = predict_image(image_path)
#             if "error" in result:
#                 continue
#             results.append(result)

#         return {"predictions": results}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
