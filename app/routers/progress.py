from fastapi import APIRouter
from app.schemas import ProgressInput
from app.db import db

router = APIRouter()

@router.post("/track-progress")
async def track_progress(input_data: ProgressInput):
    # Insert progress data into MongoDB
    progress_collection = db["progress"]
    progress_data = input_data.dict()
    progress_collection.insert_one(progress_data)

    return {"message": "Progress tracked successfully"}
