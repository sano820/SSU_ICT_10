from typing import Any, Dict, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.dialects.mysql import insert as mysql_insert

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.db.models.company import Company
from app.db.models.job import Job
from app.db.models.skill import Skill, JobSkillMap
from app.services.worknet_client import WorknetClient
from app.utils.text import normalize_whitespace, sha256_text

settings = get_settings()


def _extract_skills(text: str) -> List[str]:
    """
    아주 단순한 키워드 매칭(1차 스켈레톤).
    추후 사전/정규식/토큰화/임베딩으로 고도화하세요.
    """
    skills = []
    text_low = text.lower()
    candidates = [
        "python", "fastapi", "django", "flask",
        "sql", "mysql", "postgresql",
        "docker", "kubernetes",
        "aws", "gcp", "azure",
        "pandas", "numpy",
        "java", "spring",
        "javascript", "node", "react",
    ]
    for c in candidates:
        if c in text_low:
            skills.append(c.capitalize() if c.isalpha() else c)
    return list(dict.fromkeys(skills))  # unique order-preserving


def _normalize(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    title = normalize_whitespace(raw.get("title"))
    desc = normalize_whitespace(raw.get("description"))
    salary = normalize_whitespace(raw.get("salary"))
    content_hash = sha256_text(f"{title}|{desc}|{salary}")

    data = {
        "source": "worknet",
        "source_id": raw["source_id"],
        "company_name_raw": normalize_whitespace(raw.get("company_name")),
        "title": title,
        "description": desc,
        "location": normalize_whitespace(raw.get("location")),
        "employment_type": normalize_whitespace(raw.get("employment_type")),
        "salary": salary,
        "posted_at": raw.get("posted_at"),
        "deadline_at": raw.get("deadline_at"),
        "url": raw.get("url"),
        "hash": content_hash,
    }

    skills = _extract_skills(f"{title}\n{desc}")
    return data, skills


async def _get_or_create_company(db: AsyncSession, name: str | None) -> Company | None:
    if not name:
        return None
    q = await db.execute(select(Company).where(Company.name == name))
    comp = q.scalar_one_or_none()
    if comp:
        return comp
    comp = Company(name=name)
    db.add(comp)
    await db.flush()
    return comp


async def _get_or_create_skills(db: AsyncSession, names: List[str]) -> List[Skill]:
    if not names:
        return []
    existing = (
        await db.execute(select(Skill).where(Skill.name.in_(names)))
    ).scalars().all()
    exist_map = {s.name: s for s in existing}
    created: List[Skill] = []
    for n in names:
        if n not in exist_map:
            s = Skill(name=n)
            db.add(s)
            created.append(s)
    if created:
        await db.flush()
    # requery all to ensure ids
    allrows = (
        await db.execute(select(Skill).where(Skill.name.in_(names)))
    ).scalars().all()
    return allrows


async def _upsert_job(db: AsyncSession, payload: Dict[str, Any], company_id: int | None) -> int:
    """
    MySQL on-duplicate-key upsert.
    Returns job id (selected after upsert).
    """
    if company_id is not None:
        payload = {**payload, "company_id": company_id}

    stmt = mysql_insert(Job).values(**payload)
    update_cols = {
        # id/created_at 제외
        "company_id": stmt.inserted.company_id,
        "company_name_raw": stmt.inserted.company_name_raw,
        "title": stmt.inserted.title,
        "description": stmt.inserted.description,
        "location": stmt.inserted.location,
        "employment_type": stmt.inserted.employment_type,
        "salary": stmt.inserted.salary,
        "posted_at": stmt.inserted.posted_at,
        "deadline_at": stmt.inserted.deadline_at,
        "url": stmt.inserted.url,
        "hash": stmt.inserted.hash,
        "updated_at": func.now(),
    }
    stmt = stmt.on_duplicate_key_update(**update_cols)
    await db.execute(stmt)

    # upsert 후 id 조회
    q = await db.execute(
        select(Job.id).where(
            Job.source == payload["source"],
            Job.source_id == payload["source_id"],
        )
    )
    job_id = q.scalar_one()
    return job_id


async def _upsert_job_skills(db: AsyncSession, job_id: int, skills: List[Skill]) -> None:
    if not skills:
        return
    # 단순히 (job_id, skill_id) 존재 보장: 중복은 PK로 무시됨
    for s in skills:
        stmt = mysql_insert(JobSkillMap).values(job_id=job_id, skill_id=s.id, weight=1.0)
        stmt = stmt.on_duplicate_key_update(weight=stmt.inserted.weight)
        await db.execute(stmt)


async def sync_worknet(since: str | None = None) -> Dict[str, int]:
    """
    외부 스케줄러 혹은 내부 APScheduler가 호출.
    """
    client = WorknetClient(api_key=settings.WORKNET_API_KEY)
    raws = await client.fetch(since=since)

    inserted = updated = skipped = 0
    async with AsyncSessionLocal() as db:
        for r in raws:
            data, skill_names = _normalize(r)
            company = await _get_or_create_company(db, data.get("company_name_raw"))

            # 현재 저장된 row의 hash와 비교해 변화 여부 판단을 원하면
            # 먼저 조회 후 분기할 수도 있으나, 여기서는 upsert로 단순화
            job_id = await _upsert_job(db, data, company.id if company else None)

            # 스킬 upsert
            skills = await _get_or_create_skills(db, skill_names)
            await _upsert_job_skills(db, job_id, skills)

            # inserted/updated/skip 통계는 실제 구현에서
            # 사전 조회 + hash 비교 or ROW_COUNT() 활용 등으로 정교화 가능
            inserted += 1  # 스켈레톤: 우선 inserted에 누적

        await db.commit()

    return {"fetched": len(raws), "inserted": inserted, "updated": updated, "skipped": skipped}
