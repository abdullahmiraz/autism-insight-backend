from pydantic import BaseModel
from typing import List

class DetectionInput(BaseModel):
    q1: str
    q2: str
    q3: str
    q4: str
    q5: str
    q6: str
    q7: str
    q8: str
    q9: str
    q10: str

class ProgressInput(BaseModel):
    task_id: str
    progress_percentage: int
    feedback: str
