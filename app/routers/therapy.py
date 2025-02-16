from fastapi import APIRouter
from app.db import db

router = APIRouter()

@router.get("/recommend-therapy")
async def recommend_therapy():
    # Logic for recommending therapy, for example, based on user inputs
    therapies = [
        {"name": "ABA Therapy", "description": "Applied Behavior Analysis"},
        {"name": "Speech Therapy", "description": "Speech development interventions"},
        {"name": "Occupational Therapy", "description": "Occupational skills development"}
    ]
    return {"recommended_therapies": therapies}
