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
            int(request.A1),  # Ensure these are integers
            int(request.A2),
            int(request.A3),
            int(request.A4),
            int(request.A5),
            int(request.A6),
            int(request.A7),
            int(request.A8),
            int(request.A9),
            int(request.A10),
            int(request.Age_Mons),  # Convert Age_Mons to int if needed
            int(request.Qchat_10_Score),  # Ensure it's an integer
            str(request.Sex),  # Sex can remain a string
            str(request.Ethnicity),  # Ethnicity should be a string
            str(request.Jaundice),  # Jaundice should be a string
            str(request.Family_mem_with_ASD),  # Same for Family_mem_with_ASD
            str(request.Who_completed_test),  # Same for Who_completed_test
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
