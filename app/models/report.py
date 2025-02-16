from pydantic import BaseModel
from typing import List

class Report(BaseModel):
    user_id: str
    progress: List[str]
    recommendations: List[str]
