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
from prompts import USER_PROFILING_PROMPT, DOMESTIC_JOB_ANALYSIS_PROMPT, GLOBAL_TREND_ANALYSIS_PROMPT, GAP_ANALYSIS_PROMPT, LLM_ROUTER_PROMPT, RECOMMEND_LEARNING, RECOMMEND_STORYTELLING_PROMPT, FINAL_REPORT_PROMPT

# API키
import config 

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", 
                             temperature=0,
                             google_api_key=config.GEMINI_API_KEY)

class AgentState(TypedDict):
    user_profile_raw: Dict[str, Any]
    youtube_service: Any
    user_profile_structured: Dict[str, Any]
    target_job: List[str]
    target_company: List[str]
    user_questions: str
    domestic_analysis_components: Dict[str, Any]
    domestic_keywords: Dict[str, Any]
    global_trends: Dict[str, Any]
    gap_analysis: Dict[str, Any]
    next_action: str
    routing_reason_narrative: str
    routing_reason_structured: Dict[str, Any]
    learning_recommendations: Dict[str, Any]
    storytelling_recommendations: Dict[str, Any]
    final_report: str

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

class UserProfile(TypedDict):
    academic_year: Optional[str]
    major: Optional[str]
    skills_and_certs: Optional[str]
    experience_specs: Optional[str]
    goals: Optional[str]
    narrative_summary: str

class RefinedCompanies(BaseModel):
    companies: List[str] = Field(description="정제된 개별 공식 기업명 리스트")

def user_profiling_node(state: AgentState) -> Dict[str, Any]:
    """
    사용자가 입력한 원본 정보를 정제/구조화하고,
    '고민 또는 궁금한 점'을 별도의 키로 추출하여 반환합니다.
    """
    print("\n--- [Step 0] 사용자 정보 구조화 노드 실행 ---")

    user_info = state["user_profile_raw"]

    key_to_label = {
        "목표 직무": "목표 직무",
        "희망 기업": "희망 기업",
        "학년/학기": "학년/학기",
        "재학 여부": "재학 여부",
        "전공 및 복수(부)전공": "전공",
        "보유 기술 및 자격증": "보유 기술 및 자격증",
        "관련 경험 및 스펙" : "관련 경험 및 스펙",
        "관심 분야 및 목표": "관심 분야 및 목표",
        "고민 또는 궁금한 점": "고민 또는 궁금한 점"
    }
    target_job = user_info.get("목표 직무", "지정되지 않음")
    target_company = user_info.get("희망 기업", [])
    profile_parts = [f"- {label}: {user_info.get(key)}" for key, label in key_to_label.items() if user_info.get(key) and str(user_info.get(key)).strip()]
    profile_text = "\n".join(profile_parts) if profile_parts else "입력된 사용자 정보가 없습니다."

    user_info_extract_prompt = USER_PROFILING_PROMPT["user_info_extract"]

    parser = JsonOutputParser(pydantic_object=UserProfile)
    prompt = ChatPromptTemplate.from_template(template=user_info_extract_prompt)
    profiling_chain = prompt | llm | parser

    try:
        structured_profile = profiling_chain.invoke({
            "profile_text": profile_text,
            "target_job": target_job
        })
        print("-> 구조화된 사용자 프로필 생성 완료.")
    except Exception as e:
        print(f"-> 프로필 생성 체인 실행 실패: {e}\n-> 대체 프로필을 생성합니다.")
        structured_profile = {"narrative_summary": "사용자 정보를 분석하는 데 실패했습니다."}

    # 사용자의 핵심 질문을 별도로 추출
    user_questions = user_info.get("고민 또는 궁금한 점", "")
    print(f"-> 추출된 사용자 질문: '{user_questions}'")

    original_target_job = user_info.get("목표 직무", "")
    refined_jobs = [original_target_job] if original_target_job else [] # 기본값은 원본값을 담은 리스트

    # LLM을 이용한 직무명 구체화 로직 ---
    if original_target_job:
        print(f"-> 목표 직무 구체화 시작 (입력: '{original_target_job}')")

        # 1. 직무명 구체화를 위한 프롬프트와 체인 정의
        job_refinement_prompt = USER_PROFILING_PROMPT["job_refinement"]

        try:
            # 체인 구성 및 호출
            parser = JsonOutputParser()
            prompt = ChatPromptTemplate.from_template(job_refinement_prompt)
            refinement_chain = prompt | llm | parser

            refined_jobs = refinement_chain.invoke({"original_job": original_target_job})

            print(f"-> 구체화된 직무 목록: {refined_jobs}")

        except Exception as e:
            # LLM 파싱 실패 또는 오류 발생 시 처리
            print(f"-> 직무명 구체화 실패 (오류: {e}), 원본 값을 사용합니다.")
            refined_jobs = [original_target_job]
    else:
        # 목표 직무를 입력하지 않은 경우 처리
        print("-> 목표 직무가 입력되지 않았습니다.")
        refined_jobs = []

    original_target_companies = user_info.get("희망 기업", [])
    refined_companies = []
    # LLM을 이용한 기업명 정제 로직 ---
    if original_target_companies:
        print(f"-> 희망 기업 목록 정제 시작 (입력: {original_target_companies})")

        # 1. 함수 내에서 프롬프트와 체인을 직접 정의
        company_refinement_prompt = USER_PROFILING_PROMPT["company_refinement"]

        try:
            # 2. LLM이 Pydantic 모델에 맞춰 출력하도록 강제합니다.
            structured_llm = llm.with_structured_output(RefinedCompanies)
            prompt = ChatPromptTemplate.from_template(company_refinement_prompt)
            refinement_chain = prompt | structured_llm

            company_list_str = ", ".join(original_target_companies)

            # 3. 체인을 호출하면 Pydantic 모델 객체가 반환됩니다.
            result_model = refinement_chain.invoke({"company_list_str": company_list_str})
            refined_companies = result_model.companies # .companies 속성으로 리스트에 접근

            print(f"-> 정제된 기업 목록: {refined_companies}")

        except Exception as e:
            print(f"-> 기업명 정제 실패 (오류: {e}), 원본 값을 사용합니다.")
            refined_companies = original_target_companies
    else:
        # 4. 희망 기업을 입력하지 않은 경우 처리
        print("-> 희망 기업이 입력되지 않았습니다.")

    return {
        "user_profile_structured": structured_profile,
        "target_job": refined_jobs,
        "target_company": refined_companies,
        "user_questions": user_questions
    }

class PostingAnalysisOutput(BaseModel):
    """채용 공고 분석 결과 모델"""
    role_goal: str = Field(description="이 직무가 달성해야 할 정량적/정성적 목표")
    key_responsibilities: List[str] = Field(description="목표 달성을 위한 구체적인 주요 책임 리스트")
    hard_skills: List[str] = Field(description="필수적인 툴과 기술명 리스트")
    collaboration_process: str = Field(description="어떤 동료와 어떤 방식으로 협업하는지에 대한 구체적인 프로세스")
    preferred_experiences: List[str] = Field(description="선호하는 구체적인 경험 리스트")

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

def domestic_job_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    정제된 직무와 기업명을 바탕으로 국내 채용 시장의 각 측면을 분석하고 종합합니다.
    """
    print("\n--- [Step 2] 국내 채용 시장 분석 노드 실행 ---")

    # --- 1. 사전 준비 ---
    target_jobs_list = state["target_job"]
    refined_companies = state["target_company"]
    target_job_title = ", ".join(target_jobs_list)
    job_query = f'"{" OR ".join(target_jobs_list)}"' if target_jobs_list else ""
    company_query = f'"{" OR ".join(refined_companies)}"' if refined_companies else ""
    api_keys = state['api_keys']

    # --- 2. 채용 공고 분석 ---
    print("\n1. 채용 공고 분석 중...")
    postings_web_results = tavily_web_search.invoke(
        " ".join(part for part in [company_query, job_query, "신입 채용 공고 자격요건 우대사항"] if part)
    )

    postings_prompt = prompts.DOMESTIC_JOB_ANALYSIS_PROMPT['posting_analysis']
    prompt_variable_inputs = {
    "target_job_title": target_job_title,
    "search_results": postings_web_results
    }

    postings_analysis = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=postings_prompt,
        pydantic_model=PostingAnalysisOutput,
        llm=llm,
        log_message="-> Helper: 채용 공고 분석 및 구조화 실행..."
    )


    # --- 3. 합격 후기 분석 ---
    print("\n2. 합격 후기 분석 중...")
    current_year = datetime.now().year
    years_query = " OR ".join(str(y) for y in range(current_year, current_year - 3, -1))

    reviews_web_results = tavily_web_search.invoke(
        f'{company_query} {job_query} 신입 합격 OR 면접 후기 ({years_query}) site:velog.io OR site:tistory.com OR site:brunch.co.kr'
    )

    youtube_review_topic_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['youtube_review_topic']
    youtube_review_topic_chain = ChatPromptTemplate.from_template(youtube_review_topic_prompt) | llm | StrOutputParser()

    youtube_review_topic = youtube_review_topic_chain.invoke({
        "companies": refined_companies,
        "jobs": target_jobs_list
    })
    youtube_review_summary_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['youtube_review_summary']

    youtube_summary = analyze_youtube_topic(
        topic=youtube_review_topic,
        analysis_prompt=youtube_review_summary_prompt,
        num_to_analyze=2,
        transcripts_only=True,
        api_key=config.youtube_api_key
    )


    # 웹 검색 결과와 유튜브 분석 결과를 합쳐서 최종 분석
    # --- 4b. 수집된 데이터 종합 및 헬퍼 함수 호출 ---
    combined_reviews  = f"--- 웹 검색 결과 (최근 3년) ---\n{reviews_web_results}\n\n--- 유튜브 영상 분석 (최근 3년) ---\n{youtube_summary}"

    total_review_summary_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['web_youtube_review_total_summary']
    prompt_variable_inputs = {
    "target_job_title": target_job_title,
    "search_results": combined_reviews
    }

    reviews_analysis = extract_structured_data_flexible(
    prompt_inputs=prompt_variable_inputs,
    extraction_prompt_template=total_review_summary_prompt,
    pydantic_model=ReviewAnalysisOutput,
    llm=llm,
    log_message="-> Helper: 합격 후기 분석 및 구조화 실행..."
    )

    # --- 4. 현직자 인터뷰 분석 ---
    print("\n3. 현직자 인터뷰 분석 중...")
    interviews_web_results = tavily_web_search.invoke(
        f'{company_query} {job_query} "현직자 인터뷰" OR "직장인 브이로그" OR "일하는 방식" OR "팀 문화" OR "커리어" site:tistory.com OR site:brunch.co.kr'
    )

    # LLM을 사용하여 최적의 유튜브 검색 주제어를 생성합니다.
    youtube_interview_topic_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['youtube_interview_topic']
    topic_generation_chain = ChatPromptTemplate.from_template(youtube_interview_topic_prompt) | llm | StrOutputParser()

    interview_topic = topic_generation_chain.invoke({
        "companies": refined_companies, # 정제된 기업명 리스트
        "jobs": target_jobs_list      # 정제된 직무명 리스트
    })
    interview_prompt = f"'{target_job_title}' 직무 현직자로서 일하는 방식, 조직 문화, 신입에게 필요한 역량에 대해 말하는 부분을 핵심만 요약해줘."
    youtube_interview_summary = analyze_youtube_topic(
        topic=interview_topic,
        analysis_prompt=interview_prompt,
        num_to_analyze=1,
        transcripts_only=True,
        api_key=config.youtube_api_key
    )

    # --- 5b. 수집된 데이터 종합 및 헬퍼 함수 호출 ---
    print("-> 텍스트와 유튜브 분석 결과를 종합하여 최종 요약 생성 중...")
    combined_interviews = f"""
    --- Text-based Interview Search Results ---
    {interviews_web_results}

    --- YouTube Interview Analysis Summary ---
    {youtube_interview_summary}
    """

    youtube_interview_summary_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['youtube_interview_summary']
    prompt_variable_inputs = {
    "target_job_title": target_job_title,
    "search_results": combined_interviews
    }

    interviews_analysis = extract_structured_data_flexible(
    prompt_inputs=prompt_variable_inputs,
    extraction_prompt_template=youtube_interview_summary_prompt,
    pydantic_model=InterviewAnalysisOutput,
    llm=llm,
    log_message="-> Helper: 현직자 인터뷰 분석 및 구조화 실행..."
    )

    print("-> 다음 노드로 전달할 핵심 키워드 추출 중...")

    # --- 6. 모든 분석 결과 종합 및 키워드 추출 (헬퍼 함수 호출) ---
    combined_analysis_dict = {
        "postings_analysis": postings_analysis,
        "reviews_analysis": reviews_analysis,
        # "interviews_analysis": interviews_analysis
    }

    combined_analysis_str = json.dumps(combined_analysis_dict, ensure_ascii=False, indent=2)
    prompt_variable_inputs = {
    "market_analysis_json": combined_analysis_str
    }
    domestic_keyword_extract_prompt = DOMESTIC_JOB_ANALYSIS_PROMPT['domestic_keyword_extract']

    # [수정] 범용 함수를 사용하여 Global Trend 검색용 키워드 추출
    global_search_keywords = extract_structured_data_flexible(
    prompt_inputs=prompt_variable_inputs,
    extraction_prompt_template=domestic_keyword_extract_prompt,
    pydantic_model=GlobalSearchKeywords,
    llm=llm,
    log_message="-> Helper: Global Trend 검색용 키워드 추출 실행..."
    )

    # --- 6. 최종 반환 ---
    print("--- 국내 채용 시장 분석 완료 ---")
    return {
        "domestic_analysis_components": {
            "postings_analysis": postings_analysis,
            "reviews_analysis": reviews_analysis,
            "interviews_analysis": interviews_analysis
        },
        "domestic_keywords": global_search_keywords
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

def global_trend_analysis_node(state: AgentState) -> dict:
    """
    국내 분석 키워드와 산업 분야를 바탕으로,
    기술/시장/리더십 트렌드를 분석하여 종합적인 글로벌 동향을 도출합니다.
    """
    print("\n--- [Step 3] 글로벌 트렌드 분석 노드 실행 ---")

    # --- 1. 사전 준비 ---
    domestic_keywords = state["domestic_keywords"]
    target_job_title = ", ".join(state["target_job"])
    target_companies = state["target_company"]
    api_keys = state["api_keys"]
    one_year_ago_str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # ----------------------------------------------------------------------------------

    # --- 2. 데이터 수집 (기술, 시장, 리더십) ---

    # 2a. 기술 트렌드 (Tavily)
    print("\n1. 기술 트렌드 데이터 수집 중...")
    tech_trends_results = ""
    web_search_query_prompt = GLOBAL_TREND_ANALYSIS_PROMPT['web_search_query']
    try:
        search_query_chain = ChatPromptTemplate.from_template(web_search_query_prompt) | llm | JsonOutputParser()
        search_queries = search_query_chain.invoke({
            "keywords_str": json.dumps(domestic_keywords, ensure_ascii=False),
            "job_title": target_job_title
        })

        # [수정] 각 검색어에 'after:'를 추가하여 1년 이내로 시간 제한
        timed_search_queries = [f"{q} after:{one_year_ago_str}" for q in search_queries]
        print(f"-> 생성된 기술 트렌드 검색어: {timed_search_queries}")

        tech_trends_results_list = tavily_web_search.batch(timed_search_queries)
        tech_trends_results = "\n\n---\n\n".join([str(result) for result in tech_trends_results_list])
    except Exception as e:
        print(f"-> 기술 트렌드 검색 실패 ({e}).")
        tech_trends_results = "기술 트렌드 검색에 실패했습니다."

    # ----------------------------------------------------------------------------------

    # 2b. 시장/기업 트렌드 (News API)
    print("\n2. 시장/기업 트렌드 데이터 수집 중...")
    market_trends_results = ""
    if target_companies:
        try:
            # 산업 분야 추론 (기존과 동일)
            main_company = target_companies[0]
            industry_name_prompt = f"The company '{main_company}' primarily operates in which industry? Provide a concise, common English industry name (e.g., 'E-commerce', 'Semiconductor', 'Financial Services')."
            industry_name = llm.invoke(industry_name_prompt).content.strip()
            print(f"-> '{main_company}'의 핵심 산업 분야 추론: {industry_name}")

            # 한글 직무명을 영어로 번역 (기존과 동일)
            korean_job_title = ", ".join(state["target_job"])
            translation_prompt = f"Translate the following Korean job titles into a concise, comma-separated English string: '{korean_job_title}'"
            english_job_titles_str = llm.invoke(translation_prompt).content.strip()
            print(f"-> 직무명 영문 번역 완료: {english_job_titles_str}")

            # 1. 각 직무명의 양쪽에 큰따옴표(")를 추가합니다.
            # -> ["Machine Learning Engineer", "Data Scientist"]
            job_titles_list = [job.strip() for job in english_job_titles_str.split(',')]
            quoted_job_titles = [f'"{job}"' for job in job_titles_list]

            # 2. 큰따옴표로 묶인 직무명들을 " OR "로 연결합니다.
            # -> "Machine Learning Engineer" OR "Data Scientist"
            or_separated_jobs = " OR ".join(quoted_job_titles)

            # 3. 최종적으로 양쪽을 괄호()로 감싸줍니다.
            # -> ("Machine Learning Engineer" OR "Data Scientist")
            job_titles_query_part = f'({or_separated_jobs})'

            news_query = f'"{industry_name}" AND {job_titles_query_part} AND (hiring OR skill OR future OR trend)'
            print(f"-> News API 검색어: {news_query}")

            market_trends_results = search_global_news.invoke({
                "query": news_query,
                "from_date": one_year_ago_str
            })
        except Exception as e:
            print(f"-> 시장/기업 트렌드 검색 실패 ({e}).")
            market_trends_results = "시장/기업 트렌드 검색에 실패했습니다."
    else:
        print("-> 희망 기업이 지정되지 않아 시장/기업 트렌드 분석을 건너뜁니다.")

    # ----------------------------------------------------------------------------------

    # 2c. 리더십 트렌드 (YouTube)
    print("\n3. 리더십(권위자 비전) 트렌드 데이터 수집 중...")
    vision_analysis_summary = ""

    if target_companies:
        try :
            main_company = target_companies[0]
            youtube_conference_keyword_extractor_prompt = GLOBAL_TREND_ANALYSIS_PROMPT['youtube_conference_keyword_extractor']
            key_figures_chain = ChatPromptTemplate.from_template(youtube_conference_keyword_extractor_prompt) | llm | JsonOutputParser()
            key_figures_list = key_figures_chain.invoke({
            "main_company": main_company,
            "target_job_title": target_job_title
            })

            analysis_results = []
            for figure_name in key_figures_list:
                # 각 인물에 대한 검색 주제와 분석 프롬프트를 생성
                vision_topic = f'"{figure_name}" conference OR interview'
                vision_prompt = f"'{figure_name}'의 강연 내용을 바탕으로, '{target_job_title}' 직무와 관련된 미래 비전, 기술 철학, 그리고 업계에 던지는 핵심 메시지를 요약해줘."

                # 각 인물별로 영상 1개만 분석하여 결과 리스트에 추가
                summary = analyze_youtube_topic(
                    topic=vision_topic,
                    analysis_prompt=vision_prompt,
                    api_key=config.youtube_api_key,
                    lang_code='en',
                    max_results=20,  # 인물당 3개 후보 검색
                    num_to_analyze=1, # 그 중 1개만 분석
                    transcripts_only=True
                )
                analysis_results.append(summary)

            # 개별 분석 결과를 최종적으로 하나로 합침
            vision_analysis_summary = "\n\n---\n\n".join(analysis_results)

        except Exception as e:
            print(f"-> 권위자 비전 분석 실패 ({e}).")
            vision_analysis_summary = "권위자 비전 분석에 실패했습니다."
    else:
        vision_analysis_summary = "희망 기업이 지정되지 않아 리더십 트렌드 분석을 건너뜁니다."

    # --- 3. 모든 정보 종합 및 최종 분석 ---
    print("\n4. 모든 정보 종합 및 최종 트렌드 분석 중...")
    combined_results = f"""
    --- Technical Trends (from Keyword Search) ---
    {tech_trends_results}

    --- Market & Industry Trends (from News Search) ---
    {market_trends_results}

    --- Vision from Industry Leader (from YouTube) ---
    {vision_analysis_summary}
    """

    total_golbal_trend_summary_prompt = GLOBAL_TREND_ANALYSIS_PROMPT['total_golbal_trend_summary']

    # [수정] 새로운 프롬프트와 Pydantic 모델을 사용하여 extract_structured_data_flexible 호출
    prompt_variable_inputs = {
        "target_job_title": target_job_title,
        "search_results": combined_results
    }

    global_trends_analysis = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=total_golbal_trend_summary_prompt,
        pydantic_model=GlobalTrendsOutput,
        llm=llm,
        log_message="-> Helper: 최종 글로벌 트렌드 분석 및 구조화 실행..."
    )

    return {
        "global_trends": global_trends_analysis
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

def gap_analysis_node(state: AgentState) -> dict:
    """
    사용자 프로필과 국내외 시장 분석 결과를 종합하여
    사용자의 강점, 약점, 기회를 분석합니다. (사용자 프로필 처리 로직 강화)
    """

    print("\n--- [Step 4] 사용자 프로필 및 시장 요구사항 갭 분석 노드 실행 ---")
    print(state)
    # --- 1. 사전 준비 ---
    user_profile = state["user_profile_structured"]
    domestic_analysis = state["domestic_analysis_components"]
    global_trends = state["global_trends"]
    llm = state["llm"]

    user_profile_str = json.dumps(user_profile, ensure_ascii=False, indent=2)
    market_analysis_str = json.dumps({
        "domestic_analysis": domestic_analysis,
        "global_trends": global_trends
    }, ensure_ascii=False, indent=2)

    # --- 2. 프롬프트 엔지니어링 및 헬퍼 함수 호출 ---
    gap_analysis_prompt = GAP_ANALYSIS_PROMPT['gap_analysis']
    prompt_variable_inputs = {
        "user_profile_str": user_profile_str,
        "market_analysis_str": market_analysis_str
    }

    gap_analysis_result = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=gap_analysis_prompt,
        pydantic_model=GapAnalysisOutput,
        llm=llm,
        log_message="-> Helper: 사용자 강점, 약점, 기회 분석 실행..."
    )

    return {
        "gap_analysis": gap_analysis_result
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
    다음 행동과 그에 대한 이유(자연어/구조화)를 포함하는 모델
    """
    next_action: Literal["recommend_learning", "recommend_storytelling"] = Field(
        description="다음에 실행할 노드의 이름. 반드시 'recommend_learning' 또는 'recommend_storytelling' 중 하나여야 함."
    )
    reasoning_narrative: str = Field(
        description="사용자에게 보여줄, 친절하고 명확한 판단 이유 (자연어, 2-3 문장)."
    )
    reasoning_structured: StructuredReasoning = Field(
        description="다음 노드로 전달할, 판단의 핵심 근거가 되는 구조화된 데이터."
    )

def llm_router_node(state: AgentState) -> dict:
    """
    LLM을 사용하여 다음 행동을 결정하고,
    그 이유를 '자연어'와 '구조화된 데이터' 두 가지 형태로 반환합니다.
    """
    print("\n--- [Step 5] LLM 기반 다음 행동 결정 (Router) ---")

    # 1. state에서 필요한 데이터를 가져옵니다.
    user_questions = state.get("user_questions", "")
    gap_analysis = state.get("gap_analysis", {})
    llm = state.get("llm")

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
        llm=llm,
        log_message="-> Helper: 다음 행동 및 이유 분석 중..."
    )

    # 4. 두 가지 형태의 이유를 모두 state에 저장할 수 있도록 반환
    next_action = router_result.get("next_action", "recommend_storytelling")
    reasoning_narrative = router_result.get("reasoning_narrative", "판단 이유(자연어) 생성에 실패했습니다.")
    reasoning_structured = router_result.get("reasoning_structured", {})

    print(f"-> [LLM 라우터] 결정: {next_action}")
    print(f"-> [LLM 라우터] 이유 (사용자용): {reasoning_narrative}")
    print(f"-> [LLM 라우터] 이유 (시스템용): {reasoning_structured}")

    return {
        "next_action": next_action,
        "routing_reason_narrative": reasoning_narrative,
        "routing_reason_structured": reasoning_structured
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
    사용자의 약점을 기반으로 Tavily와 arXiv에서 자료를 검색하고,
    구체적인 학습 주제와 로드맵을 추천합니다.
    """
    print("\n--- [Step 6-A] 학습 주제 추천 노드 실행 ---")

    # --- 1. 사전 준비 ---
    gap_analysis = state.get("gap_analysis", {})
    routing_reason = state.get("routing_reason_structured", {})
    weaknesses = gap_analysis.get("weaknesses", [])
    strengths = gap_analysis.get("strengths", [])
    llm = state.get("llm")

    if not weaknesses:
        print("-> 분석된 약점이 없어 학습 주제 추천을 건너뜁니다.")
        return {"learning_recommendations": {"recommendations": []}}

    # --- 2. LLM을 사용하여 검색 키워드 생성 ---
    study_keyword_generator_prompt = RECOMMEND_LEARNING['study_keyword_generator']
    search_keywords = (llm | JsonOutputParser()).invoke(study_keyword_generator_prompt)
    print(f"-> 생성된 학습 자료 검색 키워드: {search_keywords}")

    # --- 3. Tavily와 arXiv에서 자료 검색 ---
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

    # --- 4. LLM을 사용하여 최종 추천 생성 ---
    study_recommend_summary_prompt = RECOMMEND_LEARNING['study_recommend_summary']

    prompt_variable_inputs = {
        "strengths_str": str(strengths),
        "weaknesses_str": str(weaknesses),
        "reasoning_str": str(routing_reason),
        "resources_text": all_resources_text
    }

    learning_recommendations = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=study_recommend_summary_prompt,
        pydantic_model=LearningRecommendationOutput,
        llm=llm,
        log_message="-> Helper: 최종 학습 계획 및 자료 추천 생성 중..."
    )

    return {
        "learning_recommendations": learning_recommendations
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
    사용자의 강점을 시장의 기회와 연결하고, '연결고리 경험'을 제안하여
    매력적인 포트폴리오 스토리를 완성합니다.
    """
    print("\n--- [Step 6-B] 스토리텔링 추천 노드 실행 ---")

    # --- 1. 사전 준비 ---
    gap_analysis = state.get("gap_analysis", {})
    strengths = gap_analysis.get("strengths", [])
    opportunities = gap_analysis.get("opportunities", [])
    user_experience = state.get("user_profile_structured", {}).get("experience_specs", "")
    llm = state.get("llm")
    tools = state.get("tools", {})
    tavily_tool = tools.get("tavily")

    if not strengths:
        print("-> 분석된 강점이 없어 스토리텔링 추천을 건너뜁니다.")
        return {"storytelling_recommendations": {"recommendations": []}}

    # --- 2. Tavily를 사용하여 최신 트렌드 및 컨텍스트 검색 ---
    # 기회(Opportunities)를 검색어로 활용하여 관련 최신 정보를 수집
    search_keywords = [f"{opp} latest trends" for opp in opportunities[:2]] # 최대 2개의 기회에 대해 검색
    print(f"-> 생성된 컨텍스트 검색 키워드: {search_keywords}")

    context_text = ""
    try:
        if tavily_tool and search_keywords:
            tavily_results = tavily_tool.batch(search_keywords)
            context_text = f"--- Latest Market Trends & Context ---\n{tavily_results}\n\n"
    except Exception as e:
        print(f"-> 컨텍스트 검색 중 오류 발생: {e}")

    # --- 3. LLM을 사용하여 최종 스토리텔링 추천 생성 ---
    recommend_storytelling_summary_prompt = RECOMMEND_STORYTELLING_PROMPT['recommend_storytelling_summary']

    prompt_variable_inputs = {
        "strengths_str": str(strengths),
        "opportunities_str": str(opportunities),
        "user_experience_str": user_experience,
        "context_text": context_text
    }

    storytelling_recommendations = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=recommend_storytelling_summary_prompt,
        pydantic_model=StorytellingRecommendationOutput,
        llm=llm,
        log_message="-> Helper: 최종 스토리텔링 전략 및 경험 제안 생성 중..."
    )

    return {
        "storytelling_recommendations": storytelling_recommendations
    }

class FinalReportOutput(BaseModel):
    """
    최종 보고서 텍스트를 담기 위한 모델
    """
    report: str = Field(description="모든 분석 결과를 종합하여 Markdown 형식으로 작성된 최종 보고서")

def final_report_node(state: AgentState) -> dict:
    """
    지금까지의 모든 분석 결과를 종합하여, 사용자에게 제공할
    최종적인 자연어 보고서를 생성합니다.
    """
    print("\n--- [Step 7] 최종 보고서 생성 노드 실행 ---")

    # --- 1. 사전 준비 ---
    # 보고서 작성에 필요한 모든 재료를 state에서 가져옵니다.
    user_profile = state.get("user_profile_structured", {})
    gap_analysis = state.get("gap_analysis", {})
    routing_reason = state.get("routing_reason_narrative", "")
    learning_rec = state.get("learning_recommendations", {})
    storytelling_rec = state.get("storytelling_recommendations", {})
    llm = state.get("llm")

    # 추천 결과를 선택 (라우팅 결과에 따라)
    recommendations = learning_rec if learning_rec.get("recommendations") else storytelling_rec

    # --- 2. LLM에 전달할 컨텍스트 생성 ---
    report_context_str = json.dumps({
        "user_profile": user_profile,
        "gap_analysis": gap_analysis,
        "routing_reason": routing_reason,
        "recommendations": recommendations
    }, ensure_ascii=False, indent=2)

    # --- 3. 최종 보고서 생성을 위한 프롬프트 ---
    report_summary_prompt = FINAL_REPORT_PROMPT['report_summary']

    prompt_variable_inputs = {
        "report_context": report_context_str
    }

    final_report_result = extract_structured_data_flexible(
        prompt_inputs=prompt_variable_inputs,
        extraction_prompt_template=report_summary_prompt,
        pydantic_model=FinalReportOutput,
        llm=llm,
        log_message="-> Helper: 최종 보고서 생성 중..."
    )

    report_text = final_report_result.get("report", "최종 보고서를 생성하는 데 실패했습니다.")

    return {
        "final_report": report_text
    }


def run_graph_analysis(user_profile_raw: Dict[str, Any]) -> str:
    """LangGraph를 실행하여 최종 분석 리포트를 생성합니다."""
    print("🚀 LangGraph 전체 분석 시작...")
    initial_state = {"user_profile_raw": user_profile_raw}
    
    final_state = app.invoke(initial_state)
    final_report = final_state.get("final_report", "리포트 생성에 실패했습니다.")
    
    print("✅ LangGraph 전체 분석 완료.")
    return final_report