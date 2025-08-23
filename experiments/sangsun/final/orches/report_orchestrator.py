import requests
import json
from datetime import datetime, timedelta  # 날짜 계산을 위해 추가

# OpenAI & LangChain
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# LangChain Community Tools & Loaders
from langchain_community.tools.youtube.search import YouTubeSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.retrievers import TavilySearchAPIRetriever

# External libs
from pytube import YouTube  # pytube 직접 사용

# LangGraph
from langgraph.graph import StateGraph, END

# Typing
from typing import TypedDict, List, Dict, Optional

# Tools
from tools import find_youtube_videos, analyze_video_content, search_naver_news, search_global_news, search_arxiv_papers, tavily_web_search


# 재구성된 도구와 에이전트 로직을 가져옵니다.
from tools import market_research

class AgentState(TypedDict):
    # 필수 입력값
    target_job: str
    target_company: List[str]

    # 노드를 거치며 채워지는 값들
    final_report: Optional[str]
    domestic_analysis: Optional[DomesticAnalysis]
    global_trends: GlobalTrends
    academic_research: AcademicResearch

class DomesticAnalysis(TypedDict):
    final_summary: str
    distilled_summary: str

class GlobalTrends(TypedDict):
    generated_keywords: str
    prediction: str
    distilled_prediction: str

class AcademicResearch(TypedDict):
    summary: str
    distilled_summary: str




# --- 노드 함수 정의 ---
def domestic_job_analysis_node(state: AgentState) -> Dict[str, DomesticAnalysis]:
    """
    국내 채용 시장을 3가지 관점(채용 공고, 합격 후기, 현직자 인터뷰)에서 분석합니다.
    """
    print("--- 국내 채용 시장 분석 노드 실행 ---")
    target_job = state["target_job"]
    target_companies = state["target_company"]
    company_query = " OR ".join(target_companies) # 여러 기업을 OR로 묶어 검색

    # 1. 채용 공고 분석
    print("1. 채용 공고 분석 중...")
    postings_results = tavily_web_search.invoke(
        f'"{company_query}" {target_job} 신입 채용 공고 자격요건 우대사항'
    )
    postings_summary = analyzer_chain.invoke({
        "target_job": target_job,
        "search_results": postings_results,
        "request": f"""
                  '{company_query}'의 '{target_job}' 신입 채용 공고를 분석하여, 아래 구조에 맞춰 **'측정 가능하고 구체적인'** 정보만 추출해줘.

                  - **[직무 목표(Role Goal)]**: 이 직무가 달성해야 할 **정량적/정성적 목표**는 무엇인가? (예: MAU 10만 달성, 리텐션 5% 개선, 신규 피처 성공적 런칭)
                  - **[주요 책임(Key Responsibilities)]**: 위 목표 달성을 위한 구체적인 액션은 무엇인가?
                  - **[핵심 역량(Core Competencies)]**:
                      - **(Hard Skills)**: 필수적인 **툴과 기술명**을 모두 나열해줘. (예: Python, AWS, Figma, SQL, GA)
                      - **(협업 방식)**: 어떤 동료와 어떤 방식으로 협업하는지 **구체적인 프로세스**를 명시해줘. (예: "PO, 디자이너와 슬랙으로 소통", "Git-flow 기반 코드 리뷰", "Zeplin으로 디자인 가이드 공유")
                  - **[우대 사항(Preferred)]**: 어떤 **구체적인 경험**을 가진 지원자를 선호하는가? (예: "실제 서비스 출시 경험", "오픈소스 기여 경험", "데이터 기반 의사결정 경험")

                  **※ '도전 정신', '열정' 등 추상적인 키워드는 분석에서 제외할 것.**
                  """
    }).content

    # 2. 합격 후기 분석
    print("2. 합격 후기 분석 중 (최근 3년 정보)...")

    # 최근 3년 연도 쿼리 생성
    current_year = datetime.now().year
    years_query = " OR ".join(str(y) for y in range(current_year, current_year - 3, -1))

    # Tavily 웹 검색 (최근 3년)
    reviews_web_results = tavily_web_search.invoke(
        f'"{company_query}" {target_job} 신입 합격 후기 ({years_query}) site:velog.io OR site:tistory.com OR site:brunch.co.kr'
    )

    # 유튜브 영상 검색 (최근 3년)
    youtube_summary = ""
    try:
        # 여기에 사용자의 유튜브 도구를 호출하는 코드를 넣습니다.

        video_urls_str = find_youtube_videos.invoke({
            "topic": f'"{company_query}" {target_job} 신입 합격 후기',
            "language": "korean",
            "time_filter": "3 years"
        })
        if video_urls_str and "Error" not in video_urls_str:
            first_video_url = video_urls_str.splitlines()[0].strip()
            print(f"합격 후기 영상 분석 중: {first_video_url}")

            youtube_summary = analyze_video_content({
                "video_url" : first_video_url,
                "question" : f"""
                            '{company_query}'의 '{target_job}' 직무 신입 합격 후기 정보들을 종합하여, **'검증 가능한 사실'** 기반의 합격 전략을 아래 구조로 정리해줘.

                            - **[합격자 프로필(Profile)]**: 합격자들의 공통적인 **구체적 스펙**은 무엇인가? (예: 전공, 학점, 인턴 경험 횟수, 프로젝트 개수)
                            - **[채용 프로세스별 준비 사항(Process Prep)]**: 각 단계별로 **무엇을, 어떻게** 준비했는가?
                                - **(서류/과제)**: 어떤 **프로젝트**를 강조했고, 과제는 **어떤 기술/방법론**을 사용해 해결했는가?
                                - **(면접)**: 어떤 **프로젝트 경험**에 대한 질문을 받았고, 어떻게 답변했는가?
                            - **[결정적 합격 증거(Actionable Evidence)]**: 합격자들이 제시하는 자신의 **가장 강력한 경쟁력(정량적 성과 또는 구체적 경험)**은 무엇인가?
                                (예: "MSA 구조로 전환하여 서버 비용 20% 절감 프로젝트", "공모전 2회 수상", "Kaggle 상위 5% 달성", "GAU 1만명 앱 출시 경험")

                            **※ '노력', '끈기' 같은 추상적인 답변은 제외하고, 반드시 구체적인 사례를 기반으로 작성해줘.**
                            """
            })
    except Exception as e:
        print(f"유튜브 합격 후기 분석 중 오류: {e}")
        youtube_summary = "유튜브 영상 분석 중 오류가 발생했습니다."

    # 웹 검색 결과와 유튜브 분석 결과를 합쳐서 최종 분석
    combined_results = f"--- 웹 검색 결과 (최근 3년) ---\n{reviews_web_results}\n\n--- 유튜브 영상 분석 (최근 3년) ---\n{youtube_summary}"

    reviews_summary = analyzer_chain.invoke({
        "target_job": target_job,
        "search_results": combined_results,
        "request": f"""
                  '{company_query}'의 '{target_job}' 현직자 인터뷰 내용을 바탕으로, **'실천 가능한(Actionable)'** 정보만 아래 구조로 분석해줘.

                  - **[주니어의 첫 1년(First Year)]**: 입사 1년차에게 기대하는 **구체적인 역할과 성과 수준**은 무엇인가?
                  - **[성과 측정 방식(Performance Metric)]**: 이 직무의 성과는 **어떤 지표(KPI)**로 측정되는가? (예: 개발 속도, 버그 발생률, 유저 리텐션 기여도)
                  - **[구체적인 성장 조언(Actionable Advice)]**: 현직자들이 취준생에게 '뜬구름 잡는 소리 말고' 실질적으로 준비해야 한다고 강조하는 **구체적인 기술, 경험, 공부 방법**은 무엇인가?
                      (예: "CS 기초, 특히 네트워크와 OS를 깊게 파라", "단순 클론 코딩 말고, 트래픽을 고려한 설계를 해봐라", "실제 사용자 피드백을 받아 서비스를 개선해본 경험이 중요하다")

                  **※ '성실함', '열정'과 같은 추상적인 조언은 배제하고, 행동으로 옮길 수 있는 조언만 요약해줘.**
                  """
    }).content

    # 3. 현직자 인터뷰 분석 (기존과 동일)
    print("3. 현직자 인터뷰 분석 중...")
    interviews_results = tavily_web_search.invoke(
        f'"{company_query}" {target_job} 현직자 인터뷰 "일하는 방식" "조직 문화"'
    )
    interview_summary = analyzer_chain.invoke({
        "target_job": target_job,
        "search_results": interviews_results,
        "request": f"'{company_query}'의 '{target_job}' 현직자 인터뷰 내용을 바탕으로, 신입이 미리 알고 있으면 좋은 점과 면접관에게 어필할 수 있을만한 경험이나 스펙을 정리해주세요"
    }).content

    # 최종 결과 종합 (기존과 동일)
    print("종합 분석 결과 생성 중...")
    final_summary = f"""
                    ### **{', '.join(target_companies)} {target_job} 직무 국내 시장 분석**

                    #### **1. 채용 공고에서 드러난 공식 요구사항**
                    {postings_summary}

                    #### **2. 합격자들이 말하는 실제 취업 준비 과정 (최근 3년)**
                    {reviews_summary}

                    #### **3. 현직자가 말하는 실제 업무 환경과 성장 포인트**
                    {interview_summary}
                    """

    # 다음 노드로 전달할 핵심 요약 생성
    print("-> 다음 노드로 전달할 핵심 요약 생성 중...")
    distillation_prompt = f"""
    아래의 상세 분석 보고서에서, '{target_job}' 직무의 국내 시장 현황에 대한 가장 핵심적인 내용만 한 문장으로 요약해줘.
    ---
    {final_summary}
    """
    distilled_summary = llm.invoke(distillation_prompt).content

    # 이 요약이 비어있거나 이상하게 나오는지 확인
    print(f"-> [domestic] Distilled Summary (다음 노드로 전달될 값): '{distilled_summary}'")


    # AgentState의 DomesticAnalysis 구조와 정확히 일치하는 딕셔너리 반환
    return {
            "domestic_analysis": {
            "final_summary": final_summary,
            "distilled_summary": distilled_summary
            }
    }

def global_tech_trend_node(state: AgentState) -> Dict:
    """
    모든 직무에 대해 글로벌 트렌드를 다각적으로 분석하고, 각 정보를 연결하여 미래를 예측합니다.
    Tavily(실무)와 News API(시장)를 함께 사용합니다.
    """
    print("\n--- 글로벌 동적 트렌드 분석 노드 실행 ---")
    target_job = state["target_job"]
    target_company = state["target_company"][0]
    domestic_summary = state["domestic_analysis"]["distilled_summary"]

    # --- 1. 직무 맞춤형 검색 키워드 생성 ---
    print("1. 직무 맞춤형 검색 키워드 생성 중...")
    keyword_generation_prompt = f"'{target_job}' 직무의 미래 동향을 파악하기 위한 영문 검색 키워드를 5개 생성해주세요."
    search_keywords = llm.invoke(keyword_generation_prompt).content
    print(f"-> 생성된 검색 키워드: {search_keywords}")

    # --- 2. [⭐수정⭐] 'News API'로 거시 환경 트렌드 분석 ---
    print("\n2. News API로 거시 환경(시장, 기업) 트렌드 분석 중...")
    industry_name_prompt = f"'{target_company}' 회사가 속한 핵심 산업 분야를 한 단어의 영문으로 알려줘."
    industry_name = llm.invoke(industry_name_prompt).content.strip()

    # News API에 더 적합한 쿼리로 수정
    news_query = f'"{target_company}" AND ("{industry_name}" OR business strategy OR market trend OR investment)'

    # [⭐수정⭐] tavily_web_search 대신 search_global_news 도구 사용
    macro_results = search_global_news.invoke(news_query)

    macro_trends_summary = analyzer_chain.invoke({
        "target_job": target_job,
        "search_results": macro_results,
        "request": f"""
        '{industry_name}' 산업과 '{target_company}'에 대한 최신 뉴스 기사를 바탕으로, 이 산업에 영향을 미치는 가장 중요한 **시장 동향이나 비즈니스 변화**를 객관적으로 요약해줘.
        """
    }).content

    # --- 3. 'Tavily'로 실무자 트렌드 분석 ---
    print("\n3. Tavily로 글로벌 실무자 트렌드 분석 중...")
    practitioner_query = f'"{target_job}" AND ({search_keywords}) case study OR best practices OR industry report'
    practitioner_results = tavily_web_search.invoke(practitioner_query) # 여기서는 Tavily 사용
    practitioner_trends_summary = analyzer_chain.invoke({
        "target_job": target_job,
        "search_results": practitioner_results,
        "request": "최신 글로벌 성공 사례나 전문가 리포트를 분석하여, 현재 가장 주목받는 새로운 업무 방식이나 성공 방정식을 요약해줘."
    }).content

    # --- 4. 미래 신호 분석 (YouTube 등) ---
    print("\n4. 미래 신호(컨퍼런스 등) 분석 중...")
    future_signals_summary = "관련 미래 신호 정보를 찾을 수 없습니다." # (기존 로직 유지)

    # --- 5. 최종 종합 및 '추론적' 미래 예측 ---
    print("\n5. 종합 분석 및 '추론적' 미래 예측 중...")
    combined_trends = f"""
    --- [시장 관점] 주요 뉴스 분석 (News API 기반) ---
    {macro_trends_summary}

    --- [실무자 관점] 성공 방정식 분석 (Tavily 기반) ---
    {practitioner_trends_summary}

    --- [미래 신호 관점] 다가올 패러다임 (컨퍼런스 등) ---
    {future_signals_summary}
    """

    # (prediction_prompt 및 나머지 로직은 이전과 동일)
    prediction_prompt = f"""
    당신은 최고의 전략 컨설턴트입니다. 아래 정보들을 '연결'하고 '추론'하여 최종 보고서를 작성해주세요.
    [국내 현황]: {domestic_summary}
    [글로벌 동향]: {combined_trends}
    [보고서 작성 지시]
    1. [미래 격차 분석]: ...
    2. [기회의 창]: ...
    3. [액션 플랜]: ...
    """
    prediction = llm.invoke(prediction_prompt).content
    distilled_prediction = llm.invoke(f"다음 보고서의 핵심 결론을 한 문장으로 요약해줘.\n---\n{prediction}").content

    # [⭐수정⭐] AgentState 구조에 맞춰 generated_keywords도 함께 반환
    return {
        "global_trends": {
            "generated_keywords": search_keywords,
            "prediction": prediction,
            "distilled_prediction": distilled_prediction
        }
    }

# 직업의 특성 파악해 논문 서칭 여부 결정

def should_research_papers(state: AgentState) -> str:
    """
    target_job의 특성을 분석하여 논문 분석이 필요한지 여부를 결정하는 라우터입니다.
    """
    print("\n--- [Router] 논문 분석 필요 여부 판단 중... ---")
    target_job = state["target_job"]

    prompt = f"""
    사용자의 목표 직무는 '{target_job}'입니다.
    이 직무는 최신 학술 연구 논문을 깊이 있게 분석하는 것이 취업 준비에 결정적으로 중요한, R&D 또는 딥테크 분야에 해당합니까?
    오직 'yes' 또는 'no'로만 답변해주세요.
    """

    response = llm.invoke(prompt).content.strip().lower()
    print(f"-> LLM의 판단: {response}")

    if "yes" in response:
        return "research_papers" # 'yes'일 경우, academic_research_node로 가는 경로 이름
    else:
        return "skip_research"   # 'no'일 경우, 바로 다음 단계로 가는 경로 이름

def academic_research_node(state: AgentState) -> Dict[str, Dict[str, str]]:
    """
    이전 노드의 분석 결과를 바탕으로 'search_arxiv_papers' 도구를 사용하여
    관련 최신 논문을 동적으로 검색하고, 심층 학습 방향을 제시합니다.
    """
    print("\n--- 학술 연구 분석 노드 실행 (arXiv) ---")
    target_job = state["target_job"]
    global_trends = state.get("global_trends", {})

    # 1. 이전 노드에서 생성된 동적 키워드로 검색어 생성
    search_keywords = global_trends.get("generated_keywords", target_job)
    query = f'"{target_job}" OR ({search_keywords})'
    print(f"arXiv 동적 검색 쿼리: {query}")

    # 2. ArxivLoader 로직 대신 새로 만든 도구 호출
    paper_abstracts = search_arxiv_papers.invoke({
        "query": query,
        "load_max_docs": 2
    })

    # 3. 분석 및 조언 생성
    global_prediction = global_trends.get("prediction", "알 수 없음")

    prompt_template = """
    당신은 IT 기술 분야의 수석 연구원이자 친절한 멘토입니다.
    현재 산업계에서는 "{global_prediction}"와 같은 미래가 예측되고 있습니다.
    이러한 산업계의 예측을 염두에 두고, 아래 최신 학술 연구 논문들을 분석하여 학생에게 조언해주세요.

    --- 논문 내용 ---
    {abstracts}

    ---
    [분석 요청]
    학생의 눈높이에서, 산업계 예측과 학계 연구를 연결하여 다음 구체적인 조언을 해주세요.
    1. **핵심 기반 지식**: 이 연구들을 이해하기 위해 학생이 공부해야 할 핵심 이론은 무엇인가요?
    2. **미래 역량**: 이 논문들이 암시하는 미래의 {target_job}에게 중요해질 새로운 기술 역량은 무엇인가요?
    3. **학습 방향**: 이 개념들을 경험해볼 수 있는 간단한 토이 프로젝트 아이디어를 제안해주세요.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    research_analyzer_chain = prompt | llm

    summary = research_analyzer_chain.invoke({
        "target_job": target_job,
        "global_prediction": global_prediction,
        "abstracts": paper_abstracts
    }).content

    print("학술 연구 분석 완료.")

    # 다음 노드로 전달할 핵심 요약 생성
    distillation_prompt = f"다음 학술 분석 보고서의 핵심 결론만 한 문장으로 요약해줘.\n---\n{summary}"
    distilled_summary = llm.invoke(distillation_prompt).content
    print(f"-> 학술 연구 핵심 요약 생성 완료: '{distilled_summary}'")

    # AgentState 구조에 맞춰 두 가지 요약본을 모두 반환
    return {
        "academic_research": {
            "summary": summary,
            "distilled_summary": distilled_summary
        }
    }

def generate_final_report(state: AgentState) -> Dict:
    print("\n--- [Final Step] 최종 보고서 생성 ---")

    domestic_summary = state["domestic_analysis"]["distilled_summary"]
    global_prediction = state["global_trends"]["distilled_prediction"]


    academic_summary = ""
    if state.get("academic_research"):
        academic_summary = state["academic_research"]["summary"]

    academic_section = f"\n[학술 연구 동향]: {academic_summary}" if academic_summary else ""

    final_report_prompt = f"""
    아래의 핵심 요약 정보들을 바탕으로, '{state['target_job']}' 취업 준비생을 위한 최종 커리어 로드맵을 완성도 높게 작성해줘.

    [국내 현황]: {domestic_summary}
    [글로벌 예측]: {global_prediction}{academic_section}
    """
    final_report = llm.invoke(final_report_prompt).content

    return {"final_report": final_report}



# LLM과 Analyzer Chain
llm = ChatOpenAI(model="gpt-4o", temperature=0)

ANALYSIS_PROMPT_TEMPLATE = """
당신은 {target_job} 분야 전문 커리어 애널리스트입니다. 주어진 검색 결과를 바탕으로 요청사항을 분석하고 핵심 내용을 한국어로 요약해주세요.
--- 검색 결과 ---
{search_results}
--- 분석 요청 ---
{request}
"""

prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)
analyzer_chain = prompt | llm

# --- 노드 함수 정의 (기존 함수들을 에이전트 호출 방식으로 변경) ---
# 이 부분은 CompanyAnalyst, PortfolioStrategist 클래스를 만들어 더 깔끔하게 구성할 수 있으나,
# 기존 코드 구조를 최대한 유지하기 위해 함수 형태로 작성합니다.



# --- LangGraph 워크플로우 구성 ---
workflow = StateGraph(AgentState)
workflow.add_node("domestic_analysis", domestic_job_analysis_node)
workflow.add_node("global_trends", global_tech_trend_node)
workflow.add_node("academic_research", academic_research_node)
workflow.add_node("generate_final_report", generate_final_report)

workflow.set_entry_point("domestic_analysis")
workflow.add_edge("domestic_analysis", "global_trends")
workflow.add_conditional_edges(
    "global_trends",
    should_research_papers,
    {
        "research_papers": "academic_research",
        "skip_research": "generate_final_report"
    }
)
workflow.add_edge("academic_research", "generate_final_report")
workflow.add_edge("generate_final_report", END)

app = workflow.compile()

def run_analysis_report(target_job: str, target_company: list):
    """기업 분석 및 포트폴리오 제안 보고서를 생성합니다."""
    print(f"--- {target_company} {target_job} 직무 분석 보고서 생성 시작 ---")
    initial_state = {
        "target_job": target_job,
        "target_company": target_company
    }
    final_state = app.invoke(initial_state)
    report = final_state.get("final_report", "보고서 생성에 실패했습니다.")
    
    print("\n--- 최종 결과 ---")
    print(report)
    return report