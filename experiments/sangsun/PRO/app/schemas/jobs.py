from typing import Optional, List
from pydantic import BaseModel, Field


class SyncRequest(BaseModel):
    since: Optional[str] = None  # ISO8601


class SyncResponse(BaseModel):
    fetched: int
    inserted: int
    updated: int
    skipped: int


class JobOut(BaseModel):
    id: int
    source: str
    source_id: str
    company: Optional[str] = None
    title: str
    description: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    salary: Optional[str] = None
    posted_at: Optional[str] = None
    deadline_at: Optional[str] = None
    url: Optional[str] = None
    # 가변 기본값 안전하게: default_factory
    skills: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class JobsPage(BaseModel):
    items: List[JobOut]
    next_cursor: Optional[str] = None
