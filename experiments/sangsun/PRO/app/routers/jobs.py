from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.jobs import SyncRequest, SyncResponse, JobsPage, JobOut
from app.services.job_sync import sync_worknet
from app.services.jobs_query import list_jobs_by_company
from app.db.session import get_db

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("/sync", response_model=SyncResponse)
async def sync_jobs(req: SyncRequest | None = None):
    result = await sync_worknet(since=req.since if req else None)
    return result


@router.get("/by-company", response_model=JobsPage)
async def jobs_by_company(
    user_id: int = Query(..., ge=1),
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    page = await list_jobs_by_company(db, user_id=user_id, limit=limit, cursor=cursor)
    # pydantic 모델로 매핑
    items = [JobOut(**it) for it in page["items"]]
    return {"items": items, "next_cursor": page["next_cursor"]}
