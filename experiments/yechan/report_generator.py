import json
from datetime import datetime, timedelta

# OpenAI & LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# LangChain Community Tools & Loaders
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import ArxivQueryRun, TavilySearchResults
from langchain_community.document_loaders import ArxivLoader
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# LangGraph
from langgraph.graph import StateGraph, END

# Typing
from typing import TypedDict, List, Dict, Any, Optional, Annotated, Type, Literal
import operator

# Tools
from tools import analyze_video_content, search_naver_news, search_global_news, search_arxiv_papers, tavily_web_search, find_videos_with_transcripts, analyze_youtube_topic

# Prompt
from prompts import INTENT_CLASSIFIER_PROMPT, USER_PROFILING_PROMPT, DOMESTIC_JOB_ANALYSIS_PROMPT, GLOBAL_TREND_ANALYSIS_PROMPT, GAP_ANALYSIS_PROMPTS, LLM_ROUTER_PROMPT, RECOMMEND_LEARNING, RECOMMEND_STORYTELLING_PROMPT, FINAL_REPORT_PROMPT

# API키
import config 

class AgentState(TypedDict):
    """
    에이전트의 전체 실행 과정에서 모든 상태를 저장하는 최종 TypedDict
    """
    # === 1. 초기 입력 및 공용 객체 ===
    user_profile_raw: Dict[str, Any]
    api_keys: Any
    youtube_service: Any
    llm_creative: Any
    llm_structured_analyzer: Any
    llm_fast_classifier: Any
    llm_final_analyzer : Any
    tools: Dict[str, Any]

    # === 2. 전처리 및 의도 분류 결과 ===
    intent_classification: str
    user_profile_structured: Dict[str, Any]
    target_job: List[str]
    target_company: List[str]
    user_questions: str

    # === 3. 국내 시장 분석 결과 (병렬 처리) ===
    postings_analysis: Dict[str, Any]
    reviews_analysis: Dict[str, Any]
    interviews_analysis: Dict[str, Any]

    # === 4. 국내 시장 분석 종합 및 키워드 추출 결과 ===
    domestic_analysis_components: Dict[str, Any]
    domestic_keywords: Dict[str, Any]

    # === 5. 글로벌 트렌드 분석 결과 (병렬 처리) ===
    tech_trends_raw: str # Raw text from Tavily
    market_trends_raw: str # Raw text from News API
    leaders_vision_raw: str # Raw text from leader analysis
    global_trends: Dict[str, Any]

    # === 6. 갭 분석 결과 ===
    gap_analysis: Dict[str, Any]

    # === 7. 라우터 결정 결과 ===
    next_action: str
    routing_reason_narrative: str
    routing_reason_structured: Dict[str, Any]

    # === 8. 최종 추천 결과 ===
    learning_recommendations: Dict[str, Any]
    storytelling_recommendations: Dict[str, Any]

    # === 9. 사용자에게 보여줄 스트리밍 메시지 모음 ===
    streaming_intent: str
    streaming_user_profile: str
    streaming_domestic_analysis: str
    streaming_global_analysis: str
    streaming_gap_analysis: str
    streaming_route: str
    streaming_study_recommend: str
    streaming_story_recommend: str

def extract_structured_data_flexible(
    prompt_inputs: Dict[str, Any],
    extraction_prompt_template: str,
    pydantic_model: Type[BaseModel],
    llm: Any,
    log_message: str
) -> Dict[str, Any]:
    """
    주어진 '어떤' 입력 변수(prompt_inputs)와 Pydantic 모델을 사용하여
    구조화된 데이터를 추출하는 더 유연한 범용 헬퍼 함수.
    """
    print(log_message)

    try:
        parser = PydanticOutputParser(pydantic_object=pydantic_model)

        prompt = ChatPromptTemplate.from_template(
            template=extraction_prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        extractor_chain = prompt | llm | parser
        # [수정] 인자로 받은 prompt_inputs 딕셔너리를 그대로 invoke에 전달
        extracted_data = extractor_chain.invoke(prompt_inputs)
        return extracted_data.model_dump()

    except Exception as e:
        print(f"-> 구조화된 데이터 추출 실패 (오류: {e}), 빈 딕셔너리를 반환합니다.")
        return {}

class IntentClassificationOutput(BaseModel):
    """
    사용자 질문의 의도를 분류한 결과를 담는 모델
    """
    intent: Literal["portfolio_analysis", "irrelevant"] = Field(
        description="질문의 의도. 커리어/진로/스펙/포트폴리오 관련이면 'portfolio_analysis', 관련 없으면 'irrelevant'로 분류."
    )
    reason: str = Field(
        description="왜 그렇게 의도를 분류했는지에 대한 간단한 설명."
    )

def intent_classifier_node(state: AgentState) -> dict:
    """
    사용자의 초기 질문 의도를 분석하고, 그에 맞는 스트리밍 메시지를 함께 반환합니다.
    """
    print("\n--- [Step 0] 사용자 질문 의도 분류 노드 실행 ---")

    # --- 1. 사전 준비 ---
    user_questions = state.get("user_profile_raw", {}).get("고민 또는 궁금한 점", "")
    llm = state['llm_fast_classifier']

    # [수정] 기본 스트리밍 메시지 및 의도 설정
    intent = "portfolio_analysis"
    message = "안녕하세요! 입력해주신 질문을 확인했습니다. 먼저 사용자님의 프로필을 분석하여 더 정확한 답변을 준비하겠습니다."

    if user_questions:
        # print("-> 사용자 질문 의도 분석 중...")
        # --- 2. 의도 분류를 위한 프롬프트 ---
        intent_prompt_template = INTENT_CLASSIFIER_PROMPT['intent_prompt']
        # --- 3. 헬퍼 함수 호출 ---
        prompt_variable_inputs = { "question": user_questions }
        classification_result = extract_structured_data_flexible(
            prompt_inputs=prompt_variable_inputs,
            extraction_prompt_template=intent_prompt_template,
            pydantic_model=IntentClassificationOutput,
            llm=llm,
            log_message="-> Helper: 사용자 질문 의도 분석 중..."
        )
        intent = classification_result.get("intent", "irrelevant")

        # [신규] 의도 분류 결과에 따라 메시지 변경
        if intent == "irrelevant":
            message = "안녕하세요! 저는 개인 맞춤형 포트폴리오 분석 에이전트입니다. 아쉽게도 입력해주신 질문은 분석 가능한 주제와 관련이 없어 분석을 진행하지 않습니다."

    print(f"-> 분석된 의도: {intent}")
    print(f"-> 생성된 메시지: {message}")

    # [수정] 의도와 메시지를 함께 반환
    return {
        "intent_classification": intent,
        "streaming_intent": message
    }

class UserProfile(BaseModel):
    """사용자 프로필 구조화 모델"""
    academic_year: Optional[str] = Field(description="사용자의 현재 학년 또는 상태 (예: '4학년 1학기', '석사 졸업')")
    major: Optional[str] = Field(description="사용자의 전공 및 복수/부전공")
    skills_and_certs: Optional[str] = Field(description="사용자가 보유한 기술 스택 및 자격증 요약")
    experience_specs: Optional[str] = Field(description="사용자의 핵심 관련 경험 및 프로젝트 요약")
    goals: Optional[str] = Field(description="사용자의 관심 분야 및 커리어 목표 요약")
    narrative_summary: str = Field(description="위 모든 정보를 종합하여, 사용자가 어떤 사람인지 1~2문장으로 요약")

class RefinedJobs(BaseModel):
    """직무명 구체화 모델"""
    refined_jobs: List[str] = Field(description="구체화된 직무명 리스트")

class RefinedCompanies(BaseModel):
    """기업명 정제 모델"""
    companies: List[str] = Field(description="정제된 개별 공식 기업명 리스트")


def user_profiling_node(state: AgentState) -> Dict[str, Any]:
    """
    사용자 정보 구조화 후, 다음 단계 진행 전 사용자에게 보여줄
    확인 메시지를 함께 생성하여 반환합니다.
    """
    print("\n--- [Step 1] 사용자 정보 구조화 노드 실행 ---")

    # --- 1. state에서 필요한 객체 가져오기 ---
    user_profile_raw = state["user_profile_raw"]
    analyzer_llm = state["llm_structured_analyzer"]
    creative_llm = state["llm_creative"]
    fast_llm = state["llm_fast_classifier"]

    # --- 2. 프로필 텍스트 생성 ---
    key_to_label = { "목표 직무": "목표 직무", "희망 기업": "희망 기업", "학년/학기": "학년/학기", "재학 여부": "재학 여부", "전공 및 복수(부)전공": "전공", "보유 기술 및 자격증": "보유 기술 및 자격증", "관련 경험 및 스펙" : "관련 경험 및 스펙", "관심 분야 및 목표": "관심 분야 및 목표", "고민 또는 궁금한 점": "고민 또는 궁금한 점" }
    target_job = user_profile_raw.get("목표 직무", "지정되지 않음")
    profile_parts = [f"- {label}: {user_profile_raw.get(key)}" for key, label in key_to_label.items() if user_profile_raw.get(key)]
    profile_text = "\n".join(profile_parts)

    # --- 3. LLM을 이용한 핵심 작업 수행 ---
    
    # 3-1. 사용자 프로필 구조화 (analyzer_llm 사용)
    try:
        structured_llm = analyzer_llm.with_structured_output(UserProfile)
        prompt = ChatPromptTemplate.from_template(USER_PROFILING_PROMPT["user_info_extract"])
        profiling_chain = prompt | structured_llm
        
        # [수정 2] Pydantic 모델 객체를 먼저 받고, 그 다음에 dict로 변환합니다.
        result_model = profiling_chain.invoke({
            "profile_text": profile_text,
            "target_job": target_job
        })
        structured_profile = result_model.dict()
    except Exception as e:
        print(f"-> 프로필 구조화 실패: {e}")
        structured_profile = {} # 실패 시에는 Pydantic 모델이 아니므로 .dict()를 호출하지 않습니다.

    # 3-2. 목표 직무 구체화 (analyzer_llm 사용)
    original_target_job = user_profile_raw.get("목표 직무", "")
    refined_jobs = [original_target_job] if original_target_job else []
    if original_target_job:
        try:
            structured_llm = analyzer_llm.with_structured_output(RefinedJobs)
            prompt = ChatPromptTemplate.from_template(USER_PROFILING_PROMPT["job_refinement"])
            refinement_chain = prompt | structured_llm
            result_model = refinement_chain.invoke({"original_job": original_target_job})
            
            # [수정 3] Pydantic 모델의 필드명('refined_jobs')으로 리스트에 접근합니다.
            refined_jobs = result_model.refined_jobs
            print(f"-> 구체화된 직무: {refined_jobs}")
        except Exception as e:
            print(f"-> 직무 구체화 실패: {e}")

    # 3-3. 희망 기업명 정제 (fast_llm 사용)
    original_target_companies = user_profile_raw.get("희망 기업", [])
    refined_companies = original_target_companies
    if original_target_companies:
        try:
            structured_llm = fast_llm.with_structured_output(RefinedCompanies)
            prompt = ChatPromptTemplate.from_template(USER_PROFILING_PROMPT["company_refinement"])
            refinement_chain = prompt | structured_llm
            result_model = refinement_chain.invoke({"company_list_str": ", ".join(original_target_companies)})
            refined_companies = result_model.companies
            print(f"-> 정제된 기업: {refined_companies}")
        except Exception as e:
            print(f"-> 기업명 정제 실패: {e}")

    # 3-4. 사용자 질문 추출
    user_questions = user_profile_raw.get("고민 또는 궁금한 점", "")
    print(f"-> 추출된 사용자 질문: '{user_questions}'")

    # [수정 1] 이 노드에 맞는 올바른 프롬프트 템플릿을 정의합니다.
    confirmation_prompt_template = """
    당신은 친절한 AI 커리어 어시스턴트입니다.
    아래 [정리된 프로필]을 바탕으로, 사용자에게 분석을 시작하기 전 내용을 확인시켜주는 안내 메시지를 생성해주세요.

    [지시사항]:
    - 사용자가 자신의 정보가 잘 입력되었는지 확인할 수 있게, 프로필 내용을 간결하게 포함하여 안내해주세요.
    - 메시지 마지막에는, 이 정보를 바탕으로 심층 분석을 시작하겠다는 내용을 포함시켜주세요.
    - 과하게 친절하거나 부자연스러운 말투는 피해주세요.

    ---
    [정리된 프로필]
    {profile_summary}
    ---
    """

    # 구조화된 프로필을 예쁜 문자열로 변환 (Markdown 활용)
    profile_summary_for_display = f"""
- **목표 직무:** {', '.join(refined_jobs)}
- **희망 기업:** {', '.join(refined_companies)}
- **현재 상태:** {structured_profile.get('academic_year', '')}, {structured_profile.get('major', '')}
- **보유 역량:** {structured_profile.get('skills_and_certs', '')}
- **주요 경험:** {structured_profile.get('experience_specs', '')}
    """

    confirmation_chain = ChatPromptTemplate.from_template(confirmation_prompt_template) | creative_llm | StrOutputParser()

    # [수정 2] 템플릿 변수명과 일치하는 'profile_summary' 키를 사용합니다.
    llm_generated_message = confirmation_chain.invoke({
        "profile_summary": profile_summary_for_display
    })

    additional_instruction = "\n\n* 추가/수정하고 싶은 부분이 있으면 회원정보 수정 후 다시 수행해주세요."
    final_confirmation_message = llm_generated_message + additional_instruction

    # --- 5. 다음 노드로 전달할 상태를 정확히 반환 ---
    return {
        "user_profile_structured": structured_profile,
        "target_job": refined_jobs,
        "target_company": refined_companies,
        "user_questions": user_questions,
        "streaming_user_profile": final_confirmation_message
    }

class PostingAnalysisOutput(BaseModel):
    """채용 공고 분석 결과 모델"""
    role_goal: str = Field(description="이 직무가 달성해야 할 정량적/정성적 목표 요약")
    key_responsibilities: List[str] = Field(description="목표 달성을 위한 구체적인 주요 책임(업무) 리스트")
    hard_skills: List[str] = Field(description="필수적으로 요구되는 툴과 기술(프로그래밍 언어, 프레임워크 등) 리스트")
    collaboration_process: str = Field(description="어떤 동료(기획자, 디자이너 등)와 어떤 방식으로 협업하는지에 대한 설명")
    preferred_experiences: List[str] = Field(description="필수는 아니지만 우대받는 구체적인 경험 리스트")

class ReviewAnalysisOutput(BaseModel):
    """
    모든 직군의 '합격 후기'에서 범용적으로 사용할 수 있는 데이터 추출 모델.
    '어떻게 합격했는가?'라는 채용 프로세스 단계별 핵심 전략에 초점을 맞춥니다.
    """
    document_strategy: List[str] = Field(
        description="서류 전형(자기소개서, 이력서) 통과를 위해 합격자가 강조했던 핵심 경험, 역량, 또는 작성 전략"
    )
    test_strategy: List[str] = Field(
        description="인적성, NCS, 논술, 코딩 테스트, 과제 등 필기/과제 전형의 종류와 구체적인 준비 방법 또는 팁"
    )
    job_interview_strategy: List[str] = Field(
        description="1차 면접(실무진 면접)에서 직무 역량을 어필하기 위해 합격자가 사용한 전략이나 받았던 핵심 질문"
    )
    final_interview_strategy: List[str] = Field(
        description="최종 면접(임원 면접)에서 인성이나 조직 적합성을 어필하기 위해 합격자가 사용한 전략이나 받았던 핵심 질문"
    )
    critical_success_factor: str = Field(
        description="합격자가 스스로 생각하는, 합격에 가장 결정적이었던 자신만의 차별화 포인트 (가장 중요한 것 한 가지)"
    )

class InterviewAnalysisOutput(BaseModel):
    """
    모든 직군의 현직자 인터뷰에서 범용적으로 사용할 수 있는 데이터 추출 모델
    """
    core_competencies_and_tools: List[str] = Field(
        description="현직자가 강조하는 해당 직무의 핵심 역량과, 업무에 실제로 사용하는 필수 툴(예: MS Office, Slack, Jira, Salesforce, Figma, Adobe Photoshop) 등"
    )
    team_culture_and_workflow: str = Field(
        description="팀의 의사소통 방식, 회의 문화, 협업 프로세스, 성과 평가 방식 등 구체적인 업무 스타일과 조직 문화"
    )
    growth_and_career_path: str = Field(
        description="회사에서 제공하는 교육, 멘토링 제도나 현직자가 말하는 현실적인 커리어 성장 경로"
    )
    advice_for_applicants: List[str] = Field(
        description="현직자가 해당 직무 지원자에게 '이것만은 꼭 준비하라'고 조언하는 구체적인 경험이나 역량"
    )

class GlobalSearchKeywords(BaseModel):
    """
    글로벌 트렌드를 다각적으로 검색하기 위한 키워드 모델
    """
    core_technologies: List[str] = Field(
        description="분석 결과의 핵심이 되는 주요 기술 키워드 (예: Deep Learning, Computer Vision)"
    )
    business_domains: List[str] = Field(
        description="언급된 주요 비즈니스 및 산업 도메인 키워드 (예: Fintech, Robotics)"
    )
    emerging_roles: List[str] = Field(
        description="새롭게 부상하거나 중요도가 높아지는 직무명 (예: Prompt Engineer, AI Ethicist, MLOps Specialist)"
    )
    problem_solution_keywords: List[str] = Field(
        description="기술이 해결하려는 구체적인 문제나 솔루션 키워드 (예: Fraud Detection, Hyper-personalization, Digital Twin, ESG)"
    )

def analyze_postings_node(state: AgentState) -> Dict[str, Any]:
    """[병렬] 국내 채용 공고를 분석합니다."""
    print("-> [병렬 실행] 1. 채용 공고 분석 중...")

    # 사전 준비
    target_jobs_list = state["target_job"]
    refined_companies = state["target_company"]
    target_job_title = ", ".join(target_jobs_list)
    job_query = f'"{" OR ".join(target_jobs_list)}"' if target_jobs_list else ""
    company_query = f'"{" OR ".join(refined_companies)}"' if refined_companies else ""
    api_keys = state['api_keys']
    analyzer_llm = state["llm_structured_analyzer"]

    # 데이터 수집 및 분석
    print("\n1. 채용 공고 분석 중...")
    postings_web_results = tavily_web_search.invoke(
        " ".join(part for part in [company_query, job_query, "신입 채용 공고 자격요건 우대사항"] if part)
    )

    postings_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['posting_analysis']
    prompt_variable_inputs = {
        "target_job_title": target_job_title,
        "search_results": postings_web_results
    }
    postings_analysis = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=postings_prompt,
        pydantic_model=PostingAnalysisOutput,
        llm=analyzer_llm,
        log_message="-> Helper: 채용 공고 분석 및 구조화 실행..."
    )

    return {"postings_analysis": postings_analysis}

def analyze_reviews_node(state: AgentState) -> Dict[str, Any]:
    """[병렬] 합격 후기를 분석합니다."""
    print("-> [병렬 실행] 2. 합격 후기 분석 중...")

    # 사전 준비
    target_jobs_list = state["target_job"]
    refined_companies = state["target_company"]
    target_job_title = ", ".join(target_jobs_list)
    job_query = f'"{" OR ".join(target_jobs_list)}"' if target_jobs_list else ""
    company_query = f'"{" OR ".join(refined_companies)}"' if refined_companies else ""
    api_keys = state['api_keys']
    analyzer_llm = state["llm_structured_analyzer"]

    current_year = datetime.now().year
    years_query = " OR ".join(str(y) for y in range(current_year, current_year - 3, -1))

    # 데이터 수집 (tavily)
    reviews_web_results = tavily_web_search.invoke(
        f'{company_query} {job_query} 신입 합격 OR 면접 후기 ({years_query}) site:velog.io OR site:tistory.com OR site:brunch.co.kr'
    )
    combined_reviews  = reviews_web_results

    # 분석
    reviews_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['web_review_summary']
    prompt_variable_inputs = {
        "target_job_title": target_job_title,
        "search_results": combined_reviews
    }
    reviews_analysis = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=reviews_prompt,
        pydantic_model=ReviewAnalysisOutput,
        llm=analyzer_llm,
        log_message="-> Helper: 합격 후기 분석 및 구조화 실행..."
    )

    return {"reviews_analysis": reviews_analysis}

def analyze_interviews_node(state: AgentState) -> Dict[str, Any]:
    """[병렬] 현직자 인터뷰를 분석합니다."""
    print("-> [병렬 실행] 3. 현직자 인터뷰 분석 중...")

    # 사전 준비
    target_jobs_list = state["target_job"]
    refined_companies = state["target_company"]
    api_keys = state['api_keys']

    fast_llm = state["llm_fast_classifier"]
    analyzer_llm = state["llm_structured_analyzer"]

    # --- [수정] 1. 핵심 정보 선택 ---
    # 가장 대표적인 회사 1개만 선택 (없으면 빈 문자열)
    main_company = refined_companies[0] if refined_companies else ""
    # 전체 직무명을 자연스러운 구(phrase)로 사용
    target_job_title = ", ".join(target_jobs_list)

    # --- [수정] 2. LLM으로 자연스러운 검색어 생성 ---
    query_generation_prompt = f"""
    아래 정보를 바탕으로, 현직자 인터뷰나 팀 문화에 대한 블로그 글을 찾기 위한 가장 효과적인 단일 검색어(single search query)를 생성해줘.

    - 핵심 기업: {main_company}
    - 핵심 직무: {target_job_title}
    
    검색어 예시:
    - "네이버 AI 엔지니어 팀 문화"
    - "카카오 백엔드 개발자 일하는 방식"
    - "삼성전자 반도체 연구원 커리어"

    가장 효과적인 검색어 하나만 따옴표 없이 출력해줘.
    """
    
    # LLM을 호출하여 최적의 검색어 생성
    optimized_query = fast_llm.invoke(query_generation_prompt).content.strip()
    print(f"-> LLM으로 생성된 최적 검색어: {optimized_query}")


    # --- [수정] 3. 최종 검색 실행 ---
    # 생성된 최적 검색어와 고정 키워드를 조합
    final_query = f'{optimized_query} "현직자 인터뷰" OR "팀 문화" site:tistory.com OR site:brunch.co.kr'
    
    interviews_web_results = tavily_web_search.invoke(final_query)

    # 분석
    interviews_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['web_interview_summary']
    prompt_variable_inputs = {
        "target_job_title": target_job_title,
        "search_results": interviews_web_results
    }
    interviews_analysis = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=interviews_prompt,
        pydantic_model=InterviewAnalysisOutput,
        llm=analyzer_llm,
        log_message="-> Helper: 현직자 인터뷰 분석 및 구조화 실행..."
    )

    return {"interviews_analysis": interviews_analysis}

def combine_domestic_analysis_node(state: AgentState) -> Dict[str, Any]:
    """병렬 분석 결과를 종합하고, 최종 키워드와 사용자용 요약 메시지를 생성합니다."""
    print("-> [취합] 병렬 분석 결과 종합 및 키워드/요약 생성 중...")

    # --- 1. 사전 준비 ---
    postings_analysis = state.get("postings_analysis", {})
    reviews_analysis = state.get("reviews_analysis", {})
    interviews_analysis = state.get("interviews_analysis", {})
    # 역할에 맞는 LLM들을 state에서 가져옵니다.
    fast_llm = state["llm_fast_classifier"]
    creative_llm = state["llm_creative"]

    # --- 2. 결과들을 하나의 딕셔너리로 병합 ---
    combined_components = {
        "postings_analysis": postings_analysis,
        "reviews_analysis": reviews_analysis,
        "interviews_analysis": interviews_analysis,
    }
    combined_analysis_str = json.dumps(combined_components, ensure_ascii=False, indent=2)

    # --- 3. 다음 단계용 키워드 추출 (기존 로직) ---
    keyword_extract_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['domestic_keyword_extract']
    domestic_keywords = extract_structured_data_flexible(
        prompt_inputs={"market_analysis_json": combined_analysis_str},
        extraction_prompt_template=keyword_extract_prompt,
        pydantic_model=GlobalSearchKeywords,
        llm=fast_llm,
        log_message="-> Helper: Global Trend 검색용 키워드 추출 실행..."
    )

    # --- 4. [신규] 사용자에게 보여줄 자연어 요약 생성 ---
    print("-> 사용자용 국내 동향 분석 요약 메시지 생성 중...")
    
    # 요약 생성을 위한 프롬프트 템플릿
    summary_generation_prompt = """
    당신은 수석 커리어 애널리스트입니다.
    아래 [국내 채용 시장 분석 결과]를 바탕으로, 사용자가 지원할 직무에 대한 핵심 동향을 이해하기 쉬운 2~3 문단의 자연스러운 글로 요약해주세요.

    [지시사항]:
    - 채용 공고에서 나타난 핵심 기술 역량(hard_skills)을 반드시 언급해주세요.
    - 합격 후기와 인터뷰에서 공통적으로 나타난 기업 문화나 면접 특징이 있다면 포함해주세요.
    - 사용자가 자신감을 얻을 수 있도록 긍정적이고 전문적인 톤을 유지해주세요.

    ---
    [국내 채용 시장 분석 결과]
    {analysis_json}
    ---
    """
    
    # 체인 생성 및 실행
    summary_chain = (
        ChatPromptTemplate.from_template(summary_generation_prompt)
        | creative_llm
        | StrOutputParser()
    )
    streaming_summary = summary_chain.invoke({"analysis_json": combined_analysis_str})

    print("streaming_summary", streaming_summary)

    # --- 5. [수정] 최종 결과 반환 ---
    return {
        "domestic_analysis_components": combined_components,
        "domestic_keywords": domestic_keywords,
        "streaming_domestic_analysis": streaming_summary # <-- 사용자용 요약 추가
    }

class GlobalTrendsOutput(BaseModel):
    """
    글로벌 트렌드 분석 결과를 구조화하기 위한 모델
    """
    future_outlook: str = Field(
        description="이 직무의 역할과 중요성에 대한 미래 전망 종합 분석 (1-2 문장 요약)"
    )
    key_technology_shifts: List[str] = Field(
        description="현재 떠오르거나, 앞으로 필수가 될 구체적인 최신 기술, 툴, 플랫폼 키워드 리스트"
    )
    changing_market_demands: List[str] = Field(
        description="기업들이 이 직무에 대해 새롭게 요구하기 시작한 구체적인 역량이나 경험 리스트"
    )
    key_messages_from_leaders: str = Field(description="업계 리더들이 공통적으로 강조하는 가장 핵심적인 메시지나 인용구 (1-2개)")
  
class TechSearchQueries(BaseModel):
    """기술 트렌드 검색어 생성 모델"""
    search_queries: List[str] = Field(description="생성된 구체적인 영문 검색어 리스트")


# --- 2. analyze_tech_trends_node 함수 내부 로직 수정 ---

def analyze_tech_trends_node(state: AgentState) -> Dict[str, Any]:
    """[병렬] 국내 키워드를 바탕으로 글로벌 기술 트렌드를 분석합니다."""
    print("-> [병렬 실행] 1. 글로벌 기술 트렌드 분석 중...")
    
    # --- 사전 준비 ---
    domestic_keywords = state["domestic_keywords"]
    target_job_title = ", ".join(state["target_job"])
    creative_llm = state["llm_creative"]
    tavily_web_search = state["tools"]["tavily"]
    one_year_ago_str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # --- 데이터 수집 (기술 트렌드) ---
    print("\n1. 기술 트렌드 데이터 수집 중...")
    tech_trends_results = ""
    web_search_query_prompt = GLOBAL_TREND_ANALYSIS_PROMPT['web_search_query']
    
    try:
        # [수정 1] Pydantic 모델을 사용하여 출력을 구조화하고 강제합니다.
        structured_llm = creative_llm.with_structured_output(TechSearchQueries)
        search_query_chain = ChatPromptTemplate.from_template(web_search_query_prompt) | structured_llm
        
        result_model = search_query_chain.invoke({
            "keywords_str": json.dumps(domestic_keywords, ensure_ascii=False),
            "job_title": target_job_title
        })

        # [수정 2] Pydantic 모델의 'search_queries' 속성에서 리스트를 안전하게 가져옵니다.
        search_queries_list = result_model.search_queries
        
        if not search_queries_list:
            raise ValueError("LLM이 검색어를 생성하지 못했습니다.")

        timed_search_queries = [f"{q} after:{one_year_ago_str}" for q in search_queries_list]
        print(f"-> 생성된 기술 트렌드 검색어: {timed_search_queries}")

        tech_trends_results_list = tavily_web_search.batch(timed_search_queries)
        tech_trends_results = "\n\n---\n\n".join([str(result) for result in tech_trends_results_list])

    except Exception as e:
        print(f"-> 기술 트렌드 검색 실패 ({e}).")
        tech_trends_results = "기술 트렌드 검색에 실패했습니다."

    return {"tech_trends_raw": tech_trends_results}

def analyze_market_trends_node(state: AgentState) -> Dict[str, Any]:
    """[병렬] 타겟 기업/산업의 글로벌 시장 트렌드를 분석합니다."""
    print("-> [병렬 실행] 2. 글로벌 시장 트렌드 분석 중...")
    # ... (기존 2b. 시장/기업 트렌드 (News API) 데이터 수집 로직)
    # 산업 추론 -> 직무 번역 -> News API 검색어 생성 -> 뉴스 검색
    # 2b. 시장/기업 트렌드 (News API)
    print("\n2. 시장/기업 트렌드 데이터 수집 중...")

    domestic_keywords = state["domestic_keywords"]
    target_job_title = ", ".join(state["target_job"])
    target_companies = state["target_company"]
    api_keys = state["api_keys"]
    fast_llm = state["llm_fast_classifier"]
    one_year_ago_str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    market_trends_results = ""
    if target_companies:
        try:
            # --- [수정] 1. LLM으로 뉴스 검색에 최적화된 단일 쿼리 생성 ---
            main_company = target_companies[0]
            
            news_query_generation_prompt = f"""
            당신은 해외 뉴스 기사를 검색하는 전문가입니다.
            아래 정보를 바탕으로, 관련 산업 및 채용 트렌드에 대한 뉴스 기사를 찾기 위한 가장 효과적인 영어 검색어(query)를 생성해줘.

            [지시사항]
            - '{target_job_title}' 같은 매우 상세한 직무명은 그대로 사용하지 말고, 'AI engineer', 'researcher', 'software developer' 처럼 더 넓고 일반적인 영어 직무명으로 바꿔서 사용해줘.
            - 검색어는 AND, OR 연산자를 사용해서 자연스러운 형태로 만들어줘.

            - 주요 회사: {main_company}
            - 관련 직무: {target_job_title}
            - 찾고 싶은 주제: 채용, 필요한 기술, 미래 트렌드

            검색어 예시:
            - '"Samsung Electronics" AND ("AI expert" OR "researcher") AND (hiring OR trend)'
            - '"Lunit" AND "Medical AI engineer" AND (recruitment OR technology)'

            가장 효과적인 검색어 하나만 따옴표 없이 출력해줘.
            """

            news_query = fast_llm.invoke(news_query_generation_prompt).content.strip()
            print(f"-> [수정된] LLM으로 생성된 News API 검색어: {news_query}")
            

            # --- 2. 생성된 쿼리로 뉴스 검색 ---
            market_trends_results = search_global_news.invoke({
                "query": news_query,
                "from_date": one_year_ago_str
            })
            
        except Exception as e:
            print(f"-> 시장/기업 트렌드 검색 실패 ({e}).")
            market_trends_results = "시장/기업 트렌드 검색에 실패했습니다."
    else:
        print("-> 희망 기업이 지정되지 않아 시장/기업 트렌드 분석을 건너뜁니다.")

    return {"market_trends_raw": market_trends_results}

def analyze_leaders_vision_node(state: AgentState) -> Dict[str, Any]:
    """[병렬] Tavily 웹 검색을 사용하여 업계 리더들의 비전과 철학을 분석합니다."""
    print("-> [병렬 실행] 3. 글로벌 리더 비전 분석 중...")

    # --- 1. 사전 준비 ---
    target_job_title = ", ".join(state["target_job"])
    target_companies = state.get("target_company", [])
    analyzer_llm = state["llm_structured_analyzer"]
    creative_llm = state["llm_creative"]
    tavily_tool = state.get("tools", {}).get("tavily")

    if not target_companies:
        print("-> 희망 기업이 지정되지 않아 리더십 트렌드 분석을 건너뜁니다.")
        return {"leaders_vision_raw": "Not analyzed (no target company)."}

    leaders_vision_summary = "리더 비전 분석에 실패했습니다."
    try:
        # --- 2. LLM으로 분석할 주요 인물 리스트 생성 ---
        main_company = target_companies[0]
        key_figures_prompt = GLOBAL_TREND_ANALYSIS_PROMPT['leader_name_extractor']

        key_figures_chain = ChatPromptTemplate.from_template(key_figures_prompt) | analyzer_llm | JsonOutputParser()
        
        # [수정 1] 변수명을 key_figures_dict로 변경하여 딕셔너리임을 명확히 함
        key_figures_dict = key_figures_chain.invoke({
            "main_company": main_company,
            "target_job_title": target_job_title
        })
        print(f"-> 분석할 주요 인물 (Raw): {key_figures_dict}")

        # [수정 2] 딕셔너리에서 'leaders' 키로 실제 리스트를 추출
        # .get()을 사용하여 키가 없어도 오류가 나지 않도록 함
        key_figures_list = key_figures_dict.get("leaders", [])

        # --- 3. Tavily를 사용하여 각 인물에 대한 인터뷰/강연 검색 ---
        # [수정 3] 이제 정상적으로 리스트를 슬라이싱할 수 있음
        search_queries = [f'"{figure_name}" interview on {target_job_title} future vision' for figure_name in key_figures_list[:3]]
        
        print(f"-> 생성된 리더 비전 검색어: {search_queries}")

        search_results_list = tavily_tool.batch(search_queries)
        search_results_text = "\n\n---\n\n".join([str(result) for result in search_results_list])

        # --- 4. [수정] LLM으로 모든 검색 결과를 한 번에 요약 ---
        summary_prompt_template = """
        당신은 기술 분야의 전문 애널리스트입니다.
        아래 [검색된 자료]를 종합하여, '{target_job_title}' 직무와 관련된 미래 비전, 기술 철학, 그리고 업계에 던지는 핵심 메시지를 2~3 문단으로 요약해주세요.
        여러 리더의 의견에서 공통적으로 나타나는 핵심적인 비전을 중심으로 정리해야 합니다.

        ---
        [검색된 자료]
        {search_results}
        ---
        """
        summary_chain = ChatPromptTemplate.from_template(summary_prompt_template) | creative_llm  | StrOutputParser()
        leaders_vision_summary = summary_chain.invoke({
            "target_job_title": target_job_title,
            "search_results": search_results_text
        })

    except Exception as e:
        print(f"-> ⚠️ 권위자 비전 분석 실패 ({e}).")
        # 실패 시에도 기본값을 유지

    return {"leaders_vision_raw": leaders_vision_summary}

def combine_global_trends_node(state: AgentState) -> Dict[str, Any]:
    """병렬 분석된 글로벌 트렌드 Raw 데이터를 종합하여 최종 분석 및 사용자용 요약을 수행합니다."""
    print("-> [취합] 글로벌 트렌드 종합 및 최종 분석/요약 중...")

    # --- 1. 사전 준비 ---
    tech_trends = state.get("tech_trends_raw", "")
    market_trends = state.get("market_trends_raw", "")
    leaders_vision = state.get("leaders_vision_raw", "")
    target_job_title = ", ".join(state["target_job"])
    
    # 역할에 맞는 LLM들을 state에서 가져옵니다.
    analyzer_llm = state["llm_final_analyzer"]
    creative_llm = state["llm_creative"]

    # --- 2. 결과들을 하나의 컨텍스트로 병합 ---
    combined_results = f"""
    --- Technical Trends ---
    {tech_trends}

    --- Market & Industry Trends ---
    {market_trends}

    --- Vision from Industry Leaders ---
    {leaders_vision}
    """

    # --- 3. 다음 단계용 최종 분석 (기존 로직) ---
    prompt_template = GLOBAL_TREND_ANALYSIS_PROMPT['total_global_trend_summary']
    prompt_inputs = {
        "target_job_title": target_job_title,
        "search_results": combined_results
    }

    global_trends_analysis = extract_structured_data_flexible(
        prompt_inputs=prompt_inputs,
        extraction_prompt_template=prompt_template,
        pydantic_model=GlobalTrendsOutput,
        llm=analyzer_llm,
        log_message="-> Helper: 최종 글로벌 트렌드 분석 및 구조화 실행..."
    )

    # --- 4. [신규] 사용자에게 보여줄 자연어 요약 생성 ---
    print("-> 사용자용 글로벌 동향 분석 요약 메시지 생성 중...")

    # 구조화된 분석 결과를 LLM에 전달할 문자열로 변환
    analysis_str = json.dumps(global_trends_analysis, ensure_ascii=False, indent=2)

    # 요약 생성을 위한 프롬프트 템플릿
    summary_generation_prompt = """
    당신은 글로벌 기술 시장을 분석하는 전문 애널리스트입니다.
    아래 [글로벌 트렌드 분석 결과]를 바탕으로, 사용자가 지원할 직무와 관련된 핵심적인 해외 동향을 2~3 문단의 자연스러운 글로 요약해주세요.

    [지시사항]:
    - 주목해야 할 최신 기술(emerging_technologies)과 시장의 요구사항(market_demands)을 중심으로 설명해주세요.
    - 업계 리더들이 강조하는 미래 비전이나 인재상이 있다면 함께 언급해주세요.
    - 사용자가 글로벌 시장의 큰 그림을 이해하고 동기부여를 얻을 수 있도록 작성해주세요.

    ---
    [글로벌 트렌드 분석 결과]
    {analysis_json}
    ---
    """

    # 체인 생성 및 실행
    summary_chain = (
        ChatPromptTemplate.from_template(summary_generation_prompt)
        | creative_llm
        | StrOutputParser()
    )
    streaming_summary = summary_chain.invoke({"analysis_json": analysis_str})

    # --- 5. [수정] 최종 결과 반환 ---
    return {
        "global_trends": global_trends_analysis,
        "streaming_global_analysis": streaming_summary # <-- 사용자용 요약 추가
    }

class GapAnalysisOutput(BaseModel):
    """
    사용자 프로필과 시장 분석 결과를 비교하여 강점, 약점, 기회를 진단하는 모델
    """
    strengths: List[str] = Field(
        description="사용자가 현재 보유한 역량/경험 중, 시장의 요구사항 및 트렌드와 일치하는 명확한 강점 리스트"
    )
    weaknesses: List[str] = Field(
        description="시장의 요구사항 및 트렌드에 비해 사용자가 명백히 부족하거나 없는 역량/경험 리스트 (보완점)"
    )
    opportunities: List[str] = Field(
        description="사용자의 강점을 글로벌 트렌드와 연결하여, 경쟁력을 한 단계 더 높일 수 있는 기회 영역 리스트"
    )
    summary: str = Field(
        description="사용자의 현재 상태에 대한 종합적인 평가 및 다음 단계에 대한 방향성 요약 (2-3 문장)"
    )

    # [신규] 사용자에게 스트리밍으로 보여줄 자연어 요약 메시지
    summary_narrative: str = Field(
        description="시장 분석 데이터를 근거로 강점/약점을 설명하는, 사용자에게 보여주기 위한 친절한 자연어 요약 메시지"
    )


def gap_analysis_node(state: AgentState) -> dict:
    """
    사용자의 강점/약점/기회를 분석하고, 그 근거를 포함한
    자연어 요약 메시지를 한 번에 생성합니다.
    """
    print("\n--- [Step 4] 사용자 프로필 및 시장 요구사항 갭 분석 노드 실행 ---")

    # --- 1. 사전 준비 ---
    user_profile = state["user_profile_structured"]
    domestic_analysis = state["domestic_analysis_components"]
    global_trends = state["global_trends"]
    analyzer_llm = state["llm_final_analyzer"]

    user_profile_str = json.dumps(user_profile, ensure_ascii=False, indent=2)
    market_analysis_str = json.dumps({
        "domestic_analysis": domestic_analysis,
        "global_trends": global_trends
    }, ensure_ascii=False, indent=2)

    # --- 2. 헬퍼 함수 호출 (분석과 요약을 한 번에) ---
    # prompts.py에서 템플릿을 가져옵니다.
    gap_analysis_prompt_template = GAP_ANALYSIS_PROMPTS['gap_analysis']

    prompt_variable_inputs = {
        "user_profile_str": user_profile_str,
        "market_analysis_str": market_analysis_str
    }

    gap_analysis_full_result = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=gap_analysis_prompt_template,
        pydantic_model=GapAnalysisOutput, # summary_narrative 필드가 포함된 모델
        llm=analyzer_llm,
        log_message="-> Helper: 사용자 진단 및 요약 메시지 동시 생성 중..."
    )

    # --- 3. 결과 분리 및 최종 반환 ---
    gap_analysis_dict = gap_analysis_full_result.copy()

    narrative_summary = gap_analysis_dict.pop("summary_narrative", "요약 메시지 생성에 실패했습니다.")

    final_narrative = narrative_summary + "\n\n다음 단계에서는 이러한 분석결과를 기반으로 맞춤형 추천을 제공할 예정입니다."

    return {
        "gap_analysis": gap_analysis_dict,
        "streaming_gap_analysis": final_narrative
    }

class StructuredReasoning(BaseModel):
    """
    다음 노드로 전달할 구조화된 판단 근거 모델
    """
    based_on_strengths: List[str] = Field(description="판단의 근거가 된 사용자의 핵심 강점")
    based_on_weaknesses: List[str] = Field(description="판단의 근거가 된 사용자의 핵심 약점")
    based_on_questions: str = Field(description="판단의 근거가 된 사용자의 질문 요약")

class RouterOutput(BaseModel):
    """
    다음 행동과 그 이유만을 포함하도록 수정한 모델
    """
    next_action: Literal["recommend_learning", "recommend_storytelling"]
    streaming_route: str
    reasoning_structured: StructuredReasoning

class StructuredReasoning(BaseModel):
    """
    다음 노드로 전달할 구조화된 판단 근거 모델
    """
    based_on_strengths: List[str] = Field(description="판단의 근거가 된 사용자의 핵심 강점")
    based_on_weaknesses: List[str] = Field(description="판단의 근거가 된 사용자의 핵심 약점")
    based_on_questions: str = Field(description="판단의 근거가 된 사용자의 질문 요약")

class RouterOutput(BaseModel):
    """
    다음 행동과 그 이유만을 포함하도록 수정한 모델
    """
    next_action: Literal["recommend_learning", "recommend_storytelling"]
    streaming_route: str
    reasoning_structured: StructuredReasoning

def llm_router_node(state: AgentState) -> dict:
    """
    LLM을 사용하여 다음 행동을 결정하고,
    그 이유를 '자연어'와 '구조화된 데이터' 두 가지 형태로 반환합니다.
    """
    print("\n--- [Step 5] LLM 기반 다음 행동 결정 (Router) ---")

    # 1. state에서 필요한 데이터를 가져옵니다.
    user_questions = state.get("user_questions", "")
    gap_analysis = state.get("gap_analysis", {})
    analyzer_llm = state["llm_final_analyzer"]

    # 2. LLM에 전달할 컨텍스트를 생성합니다.
    routing_context_str = json.dumps({
        "user_questions": user_questions,
        "gap_analysis": gap_analysis
    }, ensure_ascii=False, indent=2)

    # 3. 프롬프트 템플릿 및 헬퍼 함수 호출
    routing_prompt = LLM_ROUTER_PROMPT['routing']
    prompt_variable_inputs = {
        "routing_context": routing_context_str
    }

    router_result = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=routing_prompt,
        pydantic_model=RouterOutput,
        llm=analyzer_llm,
        log_message="-> Helper: 다음 행동 및 이유 분석 중..."
    )

    # 4. 두 가지 형태의 이유를 모두 state에 저장할 수 있도록 반환
    # [수정] 4가지 결과물을 모두 추출
    next_action = router_result.get("next_action", "recommend_storytelling")
    reasoning_structured = router_result.get("reasoning_structured", {})
    streaming_route = router_result.get("streaming_route", "곧 알려드릴게요.")

    print(f"-> [LLM 라우터] 결정: {next_action}")
    print(f"-> [LLM 라우터] 이유 (시스템용): {reasoning_structured}")

    # [수정] state에 streaming_message도 함께 반환
    return {
        "next_action": next_action,
        "routing_reason_structured": reasoning_structured,
        "streaming_route": streaming_route # <-- 스트리밍 메시지 추가
    }

class Resource(BaseModel):
    """
    개별 학습 자료(아티클, 논문 등)를 위한 모델
    """
    title: str = Field(description="자료의 제목")
    url: str = Field(description="자료로 바로 연결되는 URL")

class RecommendedTopic(BaseModel):
    """
    하나의 완성된 학습 주제 추천 세트를 위한 모델
    """
    topic_name: str = Field(description="추천하는 학습 주제에 대한 동기부여가 되는 이름 (예: MLOps 파이프라인 구축 첫걸음)")
    relevance_summary: str = Field(description="왜 지금 이 주제를 공부해야 하는지, 사용자의 약점과 연결하여 설명하는 요약 (1-2 문장)")
    foundational_resources: List[Resource] = Field(description="Tavily 웹 검색을 통해 찾은, 개념 학습 및 실습에 좋은 기초/실용 자료 리스트 (블로그, 튜토리얼 등)")
    deep_dive_topics: List[Resource] = Field(description="arXiv 검색을 통해 찾은, 더 깊은 학술적 탐구를 위한 심화 주제 리스트 (최신 논문 등)")

class LearningRecommendationOutput(BaseModel):
    """
    최종 학습 주제 추천 결과 모델
    """
    recommendations: List[RecommendedTopic] = Field(description="사용자에게 추천하는 맞춤형 학습 주제 리스트 (최대 2-3개)")

def recommend_learning_node(state: AgentState) -> dict:
    """
    사용자의 약점/강점을 기반으로 학습 로드맵을 구조화하고,
    사용자에게 보여줄 최종 자연어 보고서를 생성합니다.
    """
    print("\n--- [Step 6-A] 학습 주제 추천 노드 실행 (최종 보고) ---")

    # --- 1. 역할에 맞는 LLM 및 도구 가져오기 ---
    gap_analysis = state.get("gap_analysis", {})
    weaknesses = gap_analysis.get("weaknesses", [])
    strengths = gap_analysis.get("strengths", [])
    opportunities = gap_analysis.get("opportunities", [])
    
    final_analyzer_llm = state["llm_final_analyzer"]
    creative_llm = state["llm_creative"]
    tools = state.get("tools", {})
    tavily_tool = tools.get("tavily")
    arxiv_tool = tools.get("arxiv")

    # --- 2. 약점 유무에 따른 분기 처리 ---
    if weaknesses:
        print("-> 분석된 약점을 보완하는 방향으로 학습 주제를 추천합니다.")
        keyword_context = {"부족한 역량": weaknesses}
        recommendation_context_type = "약점 보완"
        recommendation_goal = "사용자의 약점을 보완하고 시장의 요구사항을 충족"
    else:
        print("-> 분석된 약점이 없습니다. 강점 강화 방향으로 학습 주제를 추천합니다.")
        if not strengths:
            print("-> 분석된 강점도 없어 추천을 종료합니다.")
            return {
                "learning_recommendations": {"recommendations": []},
                "streaming_study_recommend": "# 학습 추천\n\n분석된 강점 또는 약점이 없어 추천을 생성할 수 없습니다."
            }
        keyword_context = {"보유한 강점": strengths, "미래 기회": opportunities}
        recommendation_context_type = "강점 강화"
        recommendation_goal = "사용자의 강점을 독보적인 수준으로 발전시켜 미래 기회를 선점"

    # --- 3. [수정] '창의적 LLM'으로 검색 키워드 생성 ---
    # 검색어 브레인스토밍은 창의성이 필요하므로 creative_llm 사용
    keyword_generation_prompt = f"""
    주어진 '{recommendation_context_type}' 컨텍스트를 바탕으로, 학습 자료를 찾기 위한 구체적인 영문 검색 키워드를 3개 생성해주세요.
    - 컨텍스트: {keyword_context}
    - 출력 형식: JSON string array
    """
    search_keywords = (creative_llm | JsonOutputParser()).invoke(keyword_generation_prompt)
    print(f"-> 생성된 검색 키워드 ({recommendation_context_type}): {search_keywords}")

    # --- 4. Tavily와 arXiv에서 자료 검색 ---
    all_resources_text = ""
    try:
        print("-> Tavily 웹 검색 실행 (기초/실용 자료 수집)...")
        tavily_results = tavily_tool.batch(search_keywords)
        all_resources_text += f"--- Web Search Tutorials & Articles ---\n{tavily_results}\n\n"
        
        print("-> arXiv 논문 검색 실행 (심화 자료 수집)...")
        arxiv_query = next((kw for kw in search_keywords if "paper" in kw or "research" in kw), search_keywords[-1])
        arxiv_results = arxiv_tool.run(arxiv_query)
        all_resources_text += f"--- arXiv Research Papers ---\n{arxiv_results}"
    except Exception as e:
        print(f"-> 자료 검색 중 오류 발생: {e}")

    # --- 5. [수정] '분석용 LLM'으로 구조화된 학습 계획 생성 ---
    recommendation_prompt_template = RECOMMEND_LEARNING["study_plan_suggestion"]
    prompt_variable_inputs = {
        "recommendation_goal": recommendation_goal,
        "recommendation_context_type": recommendation_context_type,
        "keyword_context": str(keyword_context),
        "all_resources_text": all_resources_text
    }
    learning_recommendations = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=recommendation_prompt_template,
        pydantic_model=LearningRecommendationOutput,
        llm=final_analyzer_llm, # <-- 최고 성능 분석 LLM 사용
        log_message="-> Helper: 최종 학습 계획 및 자료 추천 생성 중..."
    )

    # --- 6. [수정] '창의적 LLM'으로 최종 자연어 보고서 생성 ---
    print("-> 최종 추천 보고서(스트리밍 메시지) 생성 중...")
    recommendations_str = json.dumps(learning_recommendations, ensure_ascii=False, indent=2)
    final_report_prompt = RECOMMEND_LEARNING["study_final_report"]
    
    report_chain = ChatPromptTemplate.from_template(final_report_prompt) | creative_llm | StrOutputParser() # <-- 창의적 LLM 사용
    streaming_report = report_chain.invoke({"recommendations_json": recommendations_str})

    # --- 7. 최종 반환 ---
    return {
        "learning_recommendations": learning_recommendations,
        "streaming_study_recommend": streaming_report
    }

class StorytellingTip(BaseModel):
    """
    개별 스토리텔링 팁과 경험 제안을 구조화하기 위한 모델
    """
    original_experience: str = Field(description="사용자가 이미 보유한 핵심 경험 또는 강점")
    suggested_story: str = Field(description="기존 경험을 시장의 트렌드나 요구사항과 연결하여, 면접이나 자기소개서에서 사용할 수 있는 구체적인 스토리텔링 스크립트")
    bridging_experience_idea: str = Field(description="기존 경험을 한 단계 발전시키기 위해 단기간에 실행할 수 있는 '연결고리' 토이 프로젝트 또는 학습 활동 아이디어")

class StorytellingRecommendationOutput(BaseModel):
    """
    최종 스토리텔링 추천 결과 모델
    """
    recommendations: List[StorytellingTip] = Field(description="사용자의 강점을 극대화하기 위한 맞춤형 스토리텔링 팁 리스트 (최대 2-3개)")


def recommend_storytelling_node(state: AgentState) -> dict:
    """
    사용자의 강점 유무에 따라 '강점 활용' 또는 '경험 생성' 스토리텔링을 제안하고,
    사용자에게 보여줄 최종 자연어 보고서를 생성합니다.
    """
    print("\n--- [Step 6-B] 스토리텔링 추천 노드 실행 (최종 보고) ---")

    # --- 1. 역할에 맞는 LLM 및 도구 가져오기 ---
    gap_analysis = state.get("gap_analysis", {})
    strengths = gap_analysis.get("strengths", [])
    opportunities = gap_analysis.get("opportunities", [])
    user_experience = state.get("user_profile_structured", {}).get("experience_specs", "")
    
    final_analyzer_llm = state["llm_final_analyzer"]
    creative_llm = state["llm_creative"]
    fast_llm = state["llm_fast_classifier"]
    tavily_tool = state.get("tools", {}).get("tavily")

    recommendation_prompt_template = ""
    prompt_variable_inputs = {}

    # --- 2. 강점 유무에 따른 분기 처리 ---
    if strengths:
        # --- 2-A. 강점이 있을 경우: 강점 활용 로직 ---
        print("-> 분석된 강점을 활용하는 방향으로 스토리텔링을 추천합니다.")

        # [수정] LLM으로 컨텍스트 검색 키워드 생성
        try:
            keyword_prompt = f"'{', '.join(opportunities)}'와 관련된 최신 시장 트렌드를 찾기 위한 영어 검색 키워드를 2개 생성해줘. JSON string array 형식으로."
            search_keywords = (fast_llm | JsonOutputParser()).invoke(keyword_prompt)
            print(f"-> 생성된 컨텍스트 검색 키워드: {search_keywords}")
            
            context_text = ""
            if tavily_tool and search_keywords:
                tavily_results = tavily_tool.batch(search_keywords)
                context_text = f"--- Latest Market Trends & Context ---\n{tavily_results}\n\n"
        except Exception as e:
            print(f"-> ⚠️ 컨텍스트 검색 중 오류 발생: {e}")
            context_text = ""

        recommendation_prompt_template = RECOMMEND_STORYTELLING_PROMPT['story_plan_suggestion']
        prompt_variable_inputs = {
            "strengths_str": str(strengths),
            "opportunities_str": str(opportunities),
            "user_experience_str": user_experience,
            "context_text": context_text
        }
    else:
        # --- 2-B. 강점이 없을 경우: 경험 생성 제안 로직 ---
        print("-> 분석된 강점이 없습니다. '스토리텔링을 위한 경험 생성'을 제안합니다.")
        recommendation_prompt_template = RECOMMEND_STORYTELLING_PROMPT['experience_suggestion']
        prompt_variable_inputs = {
            "opportunities_str": str(opportunities),
            "user_experience_str": user_experience,
        }

    # --- 3. [수정] '분석용 LLM'으로 구조화된 스토리텔링 전략 생성 ---
    storytelling_recommendations = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=recommendation_prompt_template,
        pydantic_model=StorytellingRecommendationOutput,
        llm=final_analyzer_llm, # <-- 최고 성능 분석 LLM 사용
        log_message="-> Helper: 최종 스토리텔링 전략 및 경험 제안 생성 중..."
    )

    # --- 4. [수정] '창의적 LLM'으로 최종 자연어 보고서 생성 ---
    print("-> 최종 추천 보고서(스트리밍 메시지) 생성 중...")
    recommendations_str = json.dumps(storytelling_recommendations, ensure_ascii=False, indent=2)
    final_report_prompt = RECOMMEND_STORYTELLING_PROMPT['story_final_report']

    report_chain = ChatPromptTemplate.from_template(final_report_prompt) | creative_llm | StrOutputParser() # <-- 창의적 LLM 사용
    streaming_report = report_chain.invoke({"recommendations_json": recommendations_str})
    
    # --- 5. 최종 반환 ---
    return {
        "storytelling_recommendations": storytelling_recommendations,
        "streaming_story_recommend": streaming_report
    }


def run_graph_analysis(user_profile_raw: Dict[str, Any]) -> str:
    """LangGraph를 실행하여 최종 분석 리포트를 생성합니다."""
    print("🚀 LangGraph 전체 분석 시작...")
    initial_state = {"user_profile_raw": user_profile_raw}
    
    final_state = app.invoke(initial_state)
    final_report = final_state.get("final_report", "리포트 생성에 실패했습니다.")
    
    print("✅ LangGraph 전체 분석 완료.")
    return final_report
