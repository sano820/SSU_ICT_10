from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.services.recommend import recommend_jobs
from app.schemas.recommend import RecommendResponse, RecItem

router = APIRouter(prefix="/jobs", tags=["Recommend"])


@router.get("/recommend", response_model=RecommendResponse)
async def recommend(
    user_id: int = Query(..., ge=1),
    top_k: int = Query(30, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    items = await recommend_jobs(db, user_id=user_id, top_k=top_k)
    return {"items": [RecItem(**x) for x in items]}
