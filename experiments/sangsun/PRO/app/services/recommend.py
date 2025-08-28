import re
from typing import Any, Dict, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from rank_bm25 import BM25Okapi

from app.db.models import (
    Job,
    Company,
    JobSkillMap,
    Skill,
    UserPreference,
    Portfolio,
    PortfolioSkillMap,
)

TOKEN_RE = re.compile(r"[A-Za-z가-힣0-9]+")


def tokenize(s: str) -> List[str]:
    if not s:
        return []
    return [t.lower() for t in TOKEN_RE.findall(s)]


async def _load_user_profile(db: AsyncSession, user_id: int) -> Tuple[List[str], List[str], List[str]]:
    # target roles/locations
    q = await db.execute(select(UserPreference).where(UserPreference.user_id == user_id))
    pref = q.scalar_one_or_none()
    roles = (pref.target_roles or []) if pref else []
    locations = (pref.target_locations or []) if pref else []

    # portfolio skills
    q2 = await db.execute(
        select(Skill.name)
        .join(PortfolioSkillMap, PortfolioSkillMap.skill_id == Skill.id)
        .join(Portfolio, Portfolio.id == PortfolioSkillMap.portfolio_id)
        .where(Portfolio.user_id == user_id)
    )
    p_skills = [row[0] for row in q2.all()]

    # to strings
    r_roles = [str(x) for x in roles]
    r_locs = [str(x) for x in locations]
    r_skills = [str(x) for x in p_skills]
    return r_roles, r_locs, r_skills


async def recommend_jobs(db: AsyncSession, user_id: int, top_k: int = 30) -> List[Dict[str, Any]]:
    roles, locations, pskills = await _load_user_profile(db, user_id)

    # 후보군: 최근 공고 300건 (필요 시 조건 필터)
    sort_dt = func.coalesce(Job.posted_at, Job.created_at)
    conds = []
    if locations:
        # 매우 단순한 location like 필터 (고도화 여지)
        like_conds = [Job.location.ilike(f"%{loc}%") for loc in locations]
        conds.append(or_(*like_conds))

    q = (
        select(Job, Company.name)
        .join(Company, Company.id == Job.company_id, isouter=True)
        .where(*conds) if conds else select(Job, Company.name).join(Company, Company.id == Job.company_id, isouter=True)
    )
    q = q.order_by(sort_dt.desc()).limit(300)
    rows = (await db.execute(q)).all()

    if not rows:
        return []

    job_ids = [r[0].id for r in rows]
    # skill map
    srows = (
        await db.execute(
            select(JobSkillMap.job_id, Skill.name)
            .join(Skill, Skill.id == JobSkillMap.skill_id)
            .where(JobSkillMap.job_id.in_(job_ids))
        )
    ).all()
    skills_map: dict[int, list[str]] = {}
    for jid, sname in srows:
        skills_map.setdefault(jid, []).append(sname)

    # 문서 생성
    docs: List[str] = []
    meta: List[Tuple[Job, str | None]] = []
    for job, cname in rows:
        text = " ".join(
            [
                job.title or "",
                job.description or "",
                " ".join(skills_map.get(job.id, [])),
            ]
        )
        docs.append(text)
        meta.append((job, cname))

    # BM25
    corpus_tokens = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus_tokens)

    # 쿼리 토큰: roles + portfolio skills 중심
    query_tokens = tokenize(" ".join(roles + pskills)) or tokenize("engineer developer 백엔드 데이터")
    scores = bm25.get_scores(query_tokens)

    # 규칙 보정
    results: List[Tuple[float, Job, str | None, List[str]]] = []
    s_min = min(scores) if len(scores) else 0.0
    s_max = max(scores) if len(scores) else 1.0
    span = (s_max - s_min) or 1.0

    for i, (job, cname) in enumerate(meta):
        base = (scores[i] - s_min) / span  # 0~1 정규화
        why: List[str] = []
        bonus = 0.0

        # 위치 보정
        if locations and job.location:
            hit_locs = [loc for loc in locations if loc.lower() in job.location.lower()]
            if hit_locs:
                bonus += 0.1
                why.append(f"선호 지역: {', '.join(hit_locs)} 일치")

        # 역할(직무) 보정
        if roles:
            hit_roles = [r for r in roles if r.lower() in f"{job.title} {job.description}".lower()]
            if hit_roles:
                bonus += 0.15
                why.append(f"직무 키워드: {', '.join(hit_roles)} 매칭")

        # 스킬 매칭 보정
        jskills = [s.lower() for s in skills_map.get(job.id, [])]
        if pskills and jskills:
            matched = [s for s in pskills if s.lower() in jskills]
            if matched:
                bonus += min(0.25, 0.05 * len(matched))
                why.append(f"포트폴리오 스킬: {', '.join(matched[:5])}")

        final_score = round(float(base + bonus), 4)
        if not why:
            why.append("키워드 유사도 상위")
        results.append((final_score, job, cname, why))

    # 정렬 및 상위 K 추출
    results.sort(key=lambda x: x[0], reverse=True)
    items: List[Dict[str, Any]] = []
    for score, job, cname, why in results[:top_k]:
        items.append(
            {
                "job_id": job.id,
                "title": job.title,
                "company": cname or job.company_name_raw,
                "score": score,
                "why": why,
            }
        )
    return items
