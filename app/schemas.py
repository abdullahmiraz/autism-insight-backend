from pydantic import BaseModel
from typing import Optional

class AutismPredictionRequest(BaseModel):
    A1: int
    A2: int
    A3: int
    A4: int
    A5: int
    A6: int
    A7: int
    A8: int
    A9: int
    A10: int
    Age_Mons: int
    Sex: str
    Ethnicity: str
    Jaundice: str
    Family_mem_with_ASD: str

class AutismPredictionResponse(BaseModel):
    prediction: str
    confidence: float
