import pprint
import json
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools import ArxivQueryRun, TavilySearchResults
from googleapiclient.discovery import build
from IPython.display import display, Markdown

# (AgentState, 모든 Pydantic 모델, 모든 노드 및 헬퍼 함수가 정의되어 있다고 가정합니다.)

if __name__ == '__main__':
    # --- 1. 준비 단계 ---
    try:
        api_keys = AgentAPIs()
        youtube_service = build('youtube', 'v3', developerKey=api_keys.youtube_api_key)
        print("✅ API 및 서비스 객체 생성 완료.")
    except Exception as e:
        print(f"🔴 API 키 또는 서비스 객체 생성 실패: {e}")
        exit()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tavily_tool = TavilySearchResults(max_results=3)
    arxiv_tool = ArxivQueryRun()

    # --- 2. LangGraph 워크플로우 생성 및 구성 ---
    # --- 2. LangGraph 워크플로우 생성 및 구성 ---
    workflow = StateGraph(AgentState)

    # --- 2a. 모든 노드 정의 ---
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("user_profiling", user_profiling_node)
    # 국내 분석 (병렬)
    workflow.add_node("analyze_postings", analyze_postings_node)
    workflow.add_node("analyze_reviews", analyze_reviews_node)
    workflow.add_node("analyze_interviews", analyze_interviews_node)
    workflow.add_node("combine_domestic", combine_domestic_analysis_node)
    # 글로벌 트렌드 분석 (병렬)
    workflow.add_node("analyze_tech_trends", analyze_tech_trends_node)
    workflow.add_node("analyze_market_trends", analyze_market_trends_node)
    workflow.add_node("analyze_leaders_vision", analyze_leaders_vision_node)
    workflow.add_node("combine_global", combine_global_trends_node)
    # 갭 분석 및 라우터
    workflow.add_node("gap_analysis", gap_analysis_node)
    workflow.add_node("router", llm_router_node)
    # 최종 추천 노드
    workflow.add_node("recommend_learning", recommend_learning_node)
    workflow.add_node("recommend_storytelling", recommend_storytelling_node) # <-- 스토리텔링 노드 추가


    # --- 2b. 엣지(연결선) 정의 ---
    workflow.set_entry_point("intent_classifier")

    # 의도 분류 결과에 따라 분기
    workflow.add_conditional_edges(
        "intent_classifier",
        lambda state: state["intent_classification"],
        {
            "portfolio_analysis": "user_profiling",
            "irrelevant": END
        }
    )

    # 국내 분석 파이프라인 (병렬)
    workflow.add_edge("user_profiling", "analyze_postings")
    workflow.add_edge("user_profiling", "analyze_reviews")
    workflow.add_edge("user_profiling", "analyze_interviews")
    workflow.add_edge("analyze_postings", "combine_domestic")
    workflow.add_edge("analyze_reviews", "combine_domestic")
    workflow.add_edge("analyze_interviews", "combine_domestic")

    # 글로벌 트렌드 분석 파이프라인 (병렬)
    workflow.add_edge("combine_domestic", "analyze_tech_trends")
    workflow.add_edge("combine_domestic", "analyze_market_trends")
    workflow.add_edge("combine_domestic", "analyze_leaders_vision")
    workflow.add_edge("analyze_tech_trends", "combine_global")
    workflow.add_edge("analyze_market_trends", "combine_global")
    workflow.add_edge("analyze_leaders_vision", "combine_global")
    
    # 갭 분석 및 라우터 연결
    workflow.add_edge("combine_global", "gap_analysis")
    workflow.add_edge("gap_analysis", "router")

    # [수정] 라우터의 결정에 따라 추천 노드로 분기
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_action"],
        {
            "recommend_learning": "recommend_learning",
            "recommend_storytelling": "recommend_storytelling" # <-- 스토리텔링 노드로 연결
        }
    )

    # [수정] 각 추천 노드가 실행된 후 그래프 종료
    workflow.add_edge("recommend_learning", END)
    workflow.add_edge("recommend_storytelling", END) # <-- 스토리텔링 노드도 종료로 연결
    
    app = workflow.compile()
    print("✅ Workflow compiled successfully!")

    # --- 3. 테스트를 위한 초기 데이터 정의 ---
    """
    user_input = {
        "목표 직무": "AI 엔지니어",
        "희망 기업": ["네이버", "카카오"],
        "학년/학기": "4학년 1학기",
        "전공 및 복수(부)전공": "컴퓨터공학과",
        "보유 기술 및 자격증": "Python, SQL, PyTorch, AWS S3/EC2 기본 사용 경험, 정보처리기사",
        "관련 경험 및 스펙" : "캡스톤 디자인 프로젝트 (PyTorch 기반 이미지 분류 모델 개발 및 배포 시도)",
        "고민 또는 궁금한 점": "MLOps 분야로 전문성을 키우고 싶은데, 어떤 기술을 더 공부해야 할까요?"
    }

    """

    user_input = {
        "목표 직무": "AI 모델 최적화 엔지니어 또는 경량화 연구원",
        "희망 기업": ["삼성전자", "SKT", "Lunit"],
        "학년/학기": "석사 졸업 후 취업준비생",
        "전공 및 복수(부)전공": "전자공학과 석사 졸업",
        "보유 기술 및 자격증": "Python, C++, Linux, PyTorch, TensorFlow, ONNX, 모델 경량화 (Pruning, Quantization), CUDA 프로그래밍 기본",
        "관련 경험 및 스펙" : "석사 졸업 논문: 'Transformer 기반 언어 모델의 Knowledge Distillation을 통한 경량화 연구', 자율주행 관련 학회에서 논문 포스터 발표 경험",
        "고민 또는 궁금한 점": "제 석사 연구 경험이 실제 산업 현장에서 어떻게 어필될 수 있을지, 면접에서 어떻게 설명해야 다른 지원자들과 차별화될 수 있을지 궁금합니다."
    }

    initial_state = {
        "user_profile_raw": user_input,
        "api_keys": api_keys,
        "youtube_service": youtube_service,
        "llm": llm,
        "tools": {"tavily": tavily_tool, "arxiv": arxiv_tool}
    }

# --- 4. 그래프 실행 (백엔드 역할) ---
print("\n🚀 전체 에이전트 실행 시작 (백엔드 역할)")
print("="*80)

execution_log = {}
final_state = None  # <-- [추가] 최종 상태를 저장할 변수

# stream을 통해 각 노드의 실행 결과를 실시간으로 확인하고 로그에 기록
for state_update in app.stream(initial_state):
    node_name = list(state_update.keys())[0]
    node_output = state_update[node_name]

    # 개발자 확인용 내부 로그 출력
    print(f"\n--- 📌 [노드: {node_name}] 실행 완료 (내부 데이터) ---")
    pprint.pprint(node_output)

    # 실행 로그에 현재 노드의 모든 출력값을 업데이트
    if node_output:
        execution_log.update(node_output)
    
    final_state = state_update # <-- [추가] 매번 마지막 상태를 덮어쓰기

print("\n\n✅ 전체 에이전트 실행 완료! 결과가 execution_log와 final_state에 저장되었습니다.")
print("다음 셀에서 최종 결과물을 확인하세요.")

# --- 5. 최종 결과물 출력 (프론트엔드 역할) ---

from IPython.display import display, Markdown

print("="*80)
print("✨ 포트폴리오 분석 결과 (사용자 화면) ✨")
print("="*80)

# 1. 의도 분석 메시지 출력
intent_message = execution_log.get("streaming_intent", "")
if intent_message:
    display(Markdown(f"### 🔍 초기 분석"))
    display(Markdown(intent_message))

# 2. 프로필 분석 메시지 출력
profile_message = execution_log.get("streaming_user_profile", "")
if profile_message:
    display(Markdown(f"\n### 👤 프로필 요약"))
    display(Markdown(profile_message))

# 3. 갭 분석 메시지 출력
gap_message = execution_log.get("streaming_gap_analysis", "")
if gap_message:
    display(Markdown(f"\n### 📊 역량 진단"))
    display(Markdown(gap_message))

# 4. 라우터의 추천 방향 미리보기 메시지 출력
router_message = execution_log.get("streaming_route", "") # 'streaming_route' -> 'streaming_message'
if router_message:
    display(Markdown(f"\n### 🧭 추천 방향 미리보기"))
    display(Markdown(router_message))

# 5. 최종 추천 보고서 출력 (학습 또는 스토리텔링)
learning_report_text = execution_log.get("streaming_study_recommend", "")
story_report_text = execution_log.get("streaming_story_recommend", "")

if learning_report_text:
    display(Markdown(f"\n### 📚 맞춤형 학습 로드맵"))
    display(Markdown(learning_report_text))
elif story_report_text:
    display(Markdown(f"\n### 🎙️ 맞춤형 스토리텔링 가이드"))
    display(Markdown(story_report_text))

print("="*80)
