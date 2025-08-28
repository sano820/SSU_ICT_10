from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func
from app.db.models import Job, Company, UserInterestCompany, JobSkillMap, Skill
from app.utils.paging import encode_cursor, decode_cursor
from app.utils.datetime import to_iso


async def list_jobs_by_company(
    db: AsyncSession, user_id: int, limit: int = 50, cursor: Optional[str] = None
) -> Dict[str, Any]:
    # 관심 회사 id들
    q_comp = await db.execute(
        select(UserInterestCompany.company_id).where(UserInterestCompany.user_id == user_id)
    )
    company_ids = [row[0] for row in q_comp.all()]
    if not company_ids:
        return {"items": [], "next_cursor": None}

    # 정렬 기준: coalesce(posted_at, created_at) DESC, id DESC
    sort_dt = func.coalesce(Job.posted_at, Job.created_at)

    conds = [Job.company_id.in_(company_ids)]
    if cursor:
        ts, pk = decode_cursor(cursor)
        conds.append(
            or_(
                sort_dt < ts,
                and_(sort_dt == ts, Job.id < pk),
            )
        )

    q = (
        select(Job, Company.name, sort_dt.label("sort_dt"))
        .join(Company, Company.id == Job.company_id, isouter=True)
        .where(*conds)
        .order_by(sort_dt.desc(), Job.id.desc())
        .limit(limit + 1)
    )

    rows = (await db.execute(q)).all()

    # next_cursor 계산
    has_more = len(rows) > limit
    rows = rows[:limit]

    job_ids = [r[0].id for r in rows]
    skills_map: dict[int, list[str]] = {}
    if job_ids:
        srows = (
            await db.execute(
                select(JobSkillMap.job_id, Skill.name)
                .join(Skill, Skill.id == JobSkillMap.skill_id)
                .where(JobSkillMap.job_id.in_(job_ids))
            )
        ).all()
        for jid, sname in srows:
            skills_map.setdefault(jid, []).append(sname)

    items: List[Dict[str, Any]] = []
    for job, company_name, sdt in rows:
        items.append(
            {
                "id": job.id,
                "source": job.source,
                "source_id": job.source_id,
                "company": company_name or job.company_name_raw,
                "title": job.title,
                "description": job.description,
                "location": job.location,
                "employment_type": job.employment_type,
                "salary": job.salary,
                "posted_at": to_iso(job.posted_at),
                "deadline_at": to_iso(job.deadline_at),
                "url": job.url,
                "skills": skills_map.get(job.id, []),
                "created_at": to_iso(job.created_at),
                "updated_at": to_iso(job.updated_at),
            }
        )

    next_cursor = None
    if has_more and rows:
        last_job, _, sdt = rows[-1]
        next_cursor = encode_cursor(sdt, last_job.id)

    return {"items": items, "next_cursor": next_cursor}
