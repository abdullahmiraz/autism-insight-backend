from fastapi import APIRouter, HTTPException
from app.models.autism_predictor import AutismPredictor
from app.schemas import AutismPredictionRequest, AutismPredictionResponse

router = APIRouter()

# Load the model
predictor = AutismPredictor("app/ml_model/logistic_regression_model.pkl")


@router.post("/predict", response_model=AutismPredictionResponse)
async def predict_autism(request: AutismPredictionRequest):
    try:
        data = [
            request.A1,
            request.A2,
            request.A3,
            request.A4,
            request.A5,
            request.A6,
            request.A7,
            request.A8,
            request.A9,
            request.A10,
            request.age_mons,  # Age in months
            request.qchat_10_score,  # Qchat-10 Score
            request.sex,  # Sex
            request.ethnicity,  # Ethnicity
            request.jaundice,  # Jaundice history
            request.family_mem_with_asd,  # Family member with ASD
            request.who_completed_test,  # Who completed the test
        ]
        prediction, confidence = predictor.predict(data)
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# import pickle
# from fastapi import APIRouter

# # Load the ML model
# with open("app/ml_model/autism_model.pkl", "rb") as f:
#     model = pickle.load(f)

# router = APIRouter()

# @router.post("/predict")
# async def predict_symptoms(data: dict):
#     # Sample input data preprocessing (you'll adapt this)
#     features = [data["feature_1"], data["feature_2"], data["feature_3"]]

#     # Use the loaded model to make a prediction
#     prediction = model.predict([features])

#     return {"prediction": prediction.tolist()}
