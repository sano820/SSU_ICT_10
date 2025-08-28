from typing import Any, Dict, Optional, List
from google import genai
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.core.config import get_settings
from app.db.models import Job, Company

_settings = get_settings()

_client = genai.Client(api_key=_settings.GEMINI_API_KEY or _settings.GOOGLE_API_KEY)

SYSTEM_BASE = """You are a job-search assistant specializing in Korean tech hiring.
Write clear, structured Korean unless asked otherwise.
Be honest about uncertainty and avoid fabrications.
"""

TEMPLATES: dict[str, str] = {
    "review_summary": """다음 합격 후기를 5~7개의 핵심 bullet로 요약해줘.
- 포지션/기술스택/난이도/면접질문/합격 포인트 중심
후기:
{content}""",
    "employee_interview": """아래 재직자 인터뷰/기사에서 ‘업무 실상/필요 역량/커리어 성장/문화’를 정리하고,
지원자를 위한 follow-up 질문 8개를 만들어줘.
자료:
{content}""",
    "company_analysis": """회사명: {company}
역할: {role}
요청: 공개 자료(너의 사전 지식 범위 내)만 바탕으로 기업 개요/주요 사업/최근 이슈/채용 포인트/면접 포인트를 정리하고,
초·중급 지원자가 준비해야 할 체크리스트 10개를 제시해줘.
불확실한 정보는 ‘추정/불명확’으로 표시.""",
    "strategy_report": """사용자 프로필(요약)과 목표 역할을 바탕으로 2주 ‘집중 취업 전략 보고서’를 작성해줘.
- ①핵심 강점/약점 ②GitHub/포트폴리오 보완 TODO ③기업/직무 타겟 리스트(이유) ④모의면접 질문셋 ⑤학습 로드맵
프로필:
{profile}
목표 역할: {role}
""",
}


async def _build_context(db: AsyncSession, ctx: Optional[Dict[str, Any]]) -> str:
    """
    간단한 RAG-lite: 회사나 역할이 있으면 최근 공고 3개를 컨텍스트로 주입
    """
    if not ctx:
        return ""
    company = ctx.get("company")
    role = ctx.get("role")

    sort_dt = func.coalesce(Job.posted_at, Job.created_at)
    q = select(Job, Company.name).join(Company, Company.id == Job.company_id, isouter=True)

    if company:
        q = q.where(Company.name == company)
    elif role:
        q = q.where(Job.title.ilike(f"%{role}%"))

    q = q.order_by(sort_dt.desc()).limit(3)
    rows = (await db.execute(q)).all()

    if not rows:
        return ""

    lines: List[str] = ["<CONTEXT> 최신 관련 공고 요약"]
    for job, cname in rows:
        lines.append(f"- [{cname or job.company_name_raw}] {job.title} @ {job.location or '-'} | {job.url or ''}")
    return "\n".join(lines)


async def run_chat_task(
    db: AsyncSession,
    task: str,
    user_input: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sys = SYSTEM_BASE
    tmpl = TEMPLATES[task]

    # 컨텍스트 주입
    ctx_text = await _build_context(db, context)
    if task == "strategy_report":
        prompt = tmpl.format(profile=user_input, role=(context or {}).get("role", ""))
    elif task == "company_analysis":
        prompt = tmpl.format(company=(context or {}).get("company", ""), role=(context or {}).get("role", ""))
    else:
        prompt = tmpl.format(content=user_input)

    full = sys + "\n\n" + (ctx_text + "\n\n" if ctx_text else "") + prompt
    res = _client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[{"role": "user", "parts": [full]}],
    )
    out = {
        "answer": getattr(res, "text", ""),
        "citations": [],
        "token_usage": getattr(res, "usage_metadata", None),
    }
    return out
