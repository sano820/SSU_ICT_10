from fastapi import FastAPI
from zoneinfo import ZoneInfo
from app.core.config import get_settings
from app.routers import jobs as jobs_router

# Optional scheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from app.services.job_sync import sync_worknet

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)


@app.get("/health")
async def health():
    return {"ok": True, "env": settings.APP_ENV}


# v1 라우트 등록
app.include_router(jobs_router.router, prefix="/v1")


# 내부 APScheduler (운영에서 외부 크론만 쓸 거면 .env에서 ENABLE_INTERNAL_SCHEDULER=false)
@app.on_event("startup")
async def start_scheduler():
    if not settings.ENABLE_INTERNAL_SCHEDULER:
        return
    tz = ZoneInfo(settings.TZ)
    scheduler = AsyncIOScheduler(timezone=tz)

    # 매일 10:00, 14:00, 19:00 KST
    trigger = CronTrigger(hour="10,14,19", minute=0, second=0, timezone=tz)
    scheduler.add_job(
        sync_worknet,
        trigger=trigger,
        id="worknet_sync",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,  # 10분 유예
    )
    scheduler.start()

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
