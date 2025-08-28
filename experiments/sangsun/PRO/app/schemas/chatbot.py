from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class ChatRequest(BaseModel):
    user_id: int
    task: str = Field(pattern="^(review_summary|employee_interview|company_analysis|strategy_report)$")
    input: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    citations: list[str] = []
    token_usage: dict | None = None
