import pickle
from fastapi import APIRouter

# Load the ML model
with open("app/ml_model/autism_model.pkl", "rb") as f:
    model = pickle.load(f)

router = APIRouter()

@router.post("/predict")
async def predict_symptoms(data: dict):
    # Sample input data preprocessing (you'll adapt this)
    features = [data["feature_1"], data["feature_2"], data["feature_3"]]

    # Use the loaded model to make a prediction
    prediction = model.predict([features])

    return {"prediction": prediction.tolist()}
