from pydantic import BaseModel
from typing import List, Optional

class RecItem(BaseModel):
    job_id: int
    title: str
    company: Optional[str] = None
    score: float
    why: List[str]

class RecommendResponse(BaseModel):
    items: List[RecItem]
