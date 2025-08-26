# filename: llm_summary_agent.py

import json
import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import config

try:
    genai.configure(api_key=config.GEMINI_API_KEY)
    print("✅ Gemini API가 성공적으로 설정되었습니다.")   # api 설정됐는지 확인
except Exception as e:
    print(f"❌ Gemini API 설정 실패: {e}")
    exit()

def _fmt_period(start_date: str | None, end_date: str | None) -> str:   # 채용일자 통일화/표준화
    """YYYYMMDD 또는 YYYY-MM-DD -> 'YYYY-MM-DD ~ YYYY-MM-DD' 로 표준화"""
    def norm(s):
        if not s: return None
        s = s.strip()
        if re.fullmatch(r"\d{8}", s): return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s): return s
        return None
    sd = norm(start_date)
    ed = norm(end_date)
    if sd and ed: return f"{sd} ~ {ed}"
    if sd: return f"{sd} ~"
    if ed: return f"~ {ed}"
    return "-"

def _safe(val):
    return val.strip() if isinstance(val, str) and val.strip() else "-"

# 카테고리 크게 분류
CATEGORY_RULES = [
    (r"반도체|웨이퍼|공정|소자|패키지|foundry|fab|etch|litho", "반도체"),
    (r"백엔드|backend|server|api|java|spring|node|golang|go|django|flask|nest", "IT"),
    (r"프론트엔드|frontend|react|vue|angular|next|nuxt|typescript|javascript", "IT"),
    (r"모바일|android|ios|swift|kotlin|flutter|react native", "IT"),
    (r"데이터|data|ml|machine learning|ai|딥러닝|머신러닝|모델|python|pytorch|tensorflow", "IT"),
    (r"devops|infra|sre|쿠버네티스|k8s|docker|aws|gcp|azure|클라우드", "IT"),
    (r"보안|security|siem|soc|모의해킹|암호|iam", "IT/보안"),
    (r"게임|game|unity|unreal|게임기획|레벨디자인", "게임"),
    (r"연구|r&d|lab|실험|분석|분체|소재|화학|폴리머|촉매", "연구개발"),
    (r"금융|은행|증권|자산|IB|트레이딩|리스크|보험", "금융"),
    (r"제조|생산|품질|qc|qa|공장|공정개선|설비", "제조"),
    (r"마케팅|브랜딩|광고|캠페인|퍼포먼스|crm|콘텐츠", "마케팅"),
]

def _local_category(company: str, title: str) -> str:
    base = f"{company} {title}".lower()
    for pat, lab in CATEGORY_RULES:
        if re.search(pat, base, flags=re.IGNORECASE): return lab
    if re.search(r"하이닉스|samsung|삼성전자|sk hynix|반도체", base, re.I): return "반도체"
    if re.search(r"게임|ncsoft|넥슨|netmarble|크래프톤|스마일게이트|pearl|pearlabyss", base, re.I): return "게임"
    return "IT"

#LLM이 처리하기 좋은 깔끔한 구조로 가공
def build_structured_summaries(job_data: list[dict]) -> list[dict]:
    """
    수집된 raw 데이터를 LLM이 처리하기 좋은 구조로 변환합니다.
    """
    results = []
    for it in job_data:
        company = _safe(it.get("company_name"))
        title = _safe(it.get("job_title"))
        emp = _safe(it.get("employment_type") or it.get("employement_type"))
        period = _fmt_period(
            it.get("start_date") or (it.get("period") or {}).get("start_date"),
            it.get("end_date") or (it.get("period") or {}).get("end_date"),
        )
        link = _safe(it.get("apply_link"))
        cat = _local_category(company, title)
        results.append({
            "company_name": company,
            "job_title": title,
            "employment_type": emp,
            "period": period,
            "apply_link": link,
            "category_hint": cat,
        })
    return results



SYSTEM_RULES = """
너는 채용 공고를 **한국어 챗봇 대화체**로만 요약한다.

[출력 형식 (각 공고당 6줄)]
1) 🔔 알림: 관심기업 {회사명}에서 새로운 채용 공고가 등록되었어요!
2) 🤖 에이전트: 모집 직무는 ‘{직무명(한국어로 표현)}’ 입니다.
3) 🏷️ 직무 분야: {IT, 반도체, 금융, 제조, 게임, 연구개발, IT/보안, 마케팅 등으로 간단 요약}
4) • 고용형태: {고용형태 또는 '-'}
5) • 채용일자: {YYYY-MM-DD ~ YYYY-MM-DD 또는 '-' }
6) 🧷 지원 링크: {URL}

[규칙]
- 절대 다른 설명/머리말/꼬리말 금지. 반드시 위 6줄 형식만 출력.
- JSON 키 이름(company_name 등) 같은 변수명은 절대 출력하지 말 것.
- 직무명은 영어라도 자연스럽게 한국어로 번역해 표현할 것.
- 직무 분야는 회사 업종과 직무명을 함께 보고 가장 적절한 한두 단어로만 요약.
- 공고 블록 사이에는 빈 줄 1줄만 둘 것.
"""

def render_chat_with_gemini(structured: list[dict]) -> str | None:
    """
    수집된 공고 데이터를 LLM에 전달하여 챗봇 대화체로 요약합니다.
    """
    if not structured:
        return None
        
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    
    prompt = SYSTEM_RULES + "\n\n아래 JSON 데이터를 위 형식으로만 요약하라:\n" + \
        json.dumps(structured, ensure_ascii=False, indent=2)

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("LLM이 빈 응답을 반환했습니다.")
        return text
    except Exception as e:
        print(f" Gemini API 호출 실패: {e}")
        return None



SYSTEM_PROMPT_TEMPLATE = """
너는 채용 공고를 **한국어 챗봇 대화체**로만 요약한다.

[출력 형식 (각 공고당 6줄)]
1) 🔔 알림: 관심기업 {회사명}에서 새로운 채용 공고가 등록되었어요!
2) 🤖 에이전트: 모집 직무는 ‘{직무명(한국어로 표현)}’ 입니다.
3) 🏷️ 직무 분야: {IT, 반도체, 금융, 제조, 게임, 연구개발, IT/보안, 마케팅 등으로 간단 요약}
4) • 고용형태: {고용형태 또는 '-'}
5) • 채용일자: {YYYY-MM-DD ~ YYYY-MM-DD 또는 '-' }
6) 🧷 지원 링크: {URL}

[규칙]
- 절대 다른 설명/머리말/꼬리말 금지. 반드시 위 6줄 형식만 출력.
- JSON 키 이름(company_name 등) 같은 변수명은 절대 출력하지 말 것.
- 직무명은 영어라도 자연스럽게 한국어로 번역해 표현할 것.
- 직무 분야는 회사 업종과 직무명을 함께 보고 가장 적절한 한두 단어로만 요약.
- 공고 블록 사이에는 빈 줄 1줄만 둘 것.

---
아래 JSON 데이터를 위 형식으로만 요약하라:
{job_data_json}
"""

def generate_chat_summary(job_data: list) -> str | None:
    """수집된 채용 공고 데이터로 챗봇 메시지를 생성합니다."""
    if not job_data:
        print("💡 요약할 채용 공고 데이터가 없습니다.")
        return None
    
    # 1. LLM 초기화
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                 temperature=0,
                                 google_api_key=config.GEMINI_API_KEY)
    
    # 2. LangChain 체인 구성
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    # 3. 데이터 준비 및 체인 실행
    structured_jobs = build_structured_summaries(job_data)
    job_data_json_str = json.dumps(structured_jobs, ensure_ascii=False, indent=2)

    try:
        chat_summary_text = chain.invoke({"job_data_json": job_data_json_str})
        print("✅ 대화체 요약 생성 완료.")
        return chat_summary_text.strip()
    except Exception as e:
        print(f"❌ LLM 호출 실패로 인해 대화체 요약을 생성하지 못했습니다: {e}")
        return None
    
