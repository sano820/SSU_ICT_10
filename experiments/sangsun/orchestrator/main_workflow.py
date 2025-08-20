# langgraph_orchestrator/graphs/main_workflow.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Agent 및 모델 임포트
from agents.duplicate_removal_agent import DuplicateRemovalAgent
from agents.summarization_agent import JobSummarizationAgent
from agents.company_analysis_agent import CompanyAnalysisAgent
from models.data_models import JobPostingRaw, JobPostingProcessed, FinalPosting
from tools.db_tools import db_tool # db 연결 종료를 위해 임포트

# LangGraph의 상태(State) 정의
# 워크플로우의 각 단계를 거치며 이 State 객체가 업데이트됩니다.
class GraphState(TypedDict):
    raw_postings: List[JobPostingRaw]           # 크롤링 후 초기 입력
    unique_postings: List[JobPostingRaw]        # 중복 제거 후
    processed_postings: List[JobPostingProcessed] # 요약 완료 후
    final_postings: List[FinalPosting]          # 최종 분석 완료 후

def build_graph():
    """LangGraph 워크플로우를 빌드하고 컴파일합니다."""
    
    # 각 역할을 수행할 에이전트 인스턴스 생성
    duplicate_remover = DuplicateRemovalAgent()
    summarizer = JobSummarizationAgent()
    company_analyzer = CompanyAnalysisAgent()
    
    # 워크플로우(그래프) 객체 생성
    workflow = StateGraph(GraphState)
    
    # 1. 노드(Node) 추가: 각 노드는 그래프의 작업 단위를 의미합니다.
    #    각 노드는 에이전트의 run 메서드를 실행합니다.
    workflow.add_node("duplicate_removal", duplicate_remover.run)
    workflow.add_node("job_summarization", summarizer.run)
    workflow.add_node("company_analysis", company_analyzer.run)
    
    # 2. 엣지(Edge) 추가: 노드 간의 데이터 흐름을 정의합니다.
    workflow.add_edge("duplicate_removal", "job_summarization")
    workflow.add_edge("job_summarization", "company_analysis")
    
    # 3. 진입점(Entry Point) 및 종료점(End Point) 설정
    workflow.set_entry_point("duplicate_removal")
    workflow.add_edge("company_analysis", END)
    
    # 4. 그래프 컴파일
    #    컴파일된 'app' 객체를 통해 워크플로우를 실행할 수 있습니다.
    app = workflow.compile()
    
    return app

if __name__ == "__main__":
    # --- 워크플로우 실행 테스트 ---
    
    # 1. 그래프 빌드
    graph_app = build_graph()

    # 2. 테스트용 초기 데이터 생성 (실제로는 data_agent/job_crawler.py의 결과물)
    initial_state = {
        "raw_postings": [
            JobPostingRaw(
                company="네이버", 
                title="AI 기술 플랫폼 개발자", 
                raw_description="하이퍼클로바X 기반의 차세대 AI 플랫폼을 함께 만들어갈 개발자를 모집합니다. 주요 업무는 대규모 분산 시스템 설계 및 개발이며, Python, C++, k8s 사용 경험이 필수입니다...",
                source_url="https://recruit.navercorp.com/..."
            ),
            JobPostingRaw(
                company="카카오", 
                title="클라우드 백엔드 개발자", 
                raw_description="카카오 클라우드 서비스의 백엔드 시스템을 개발합니다. Java/Kotlin, Spring Boot, MSA 환경에 대한 깊은 이해가 필요하며, 대용량 트래픽 처리 경험을 우대합니다.",
                source_url="https://careers.kakao.com/..."
            ),
        ]
    }
    
    print("🚀 LangGraph 워크플로우 실행을 시작합니다.")
    print(f"초기 입력 데이터: {len(initial_state['raw_postings'])}개")
    
    # 3. 그래프 실행 (invoke)
    #    .invoke() 메서드는 워크플로우가 끝날 때까지 모든 단계를 동기적으로 실행합니다.
    final_state = graph_app.invoke(initial_state)
    
    # 4. 최종 결과 출력
    print("\n🏁 LangGraph 워크플로우 실행 완료!")
    
    if final_state.get("final_postings"):
        print(f"총 {len(final_state['final_postings'])}개의 최종 결과물이 생성되었습니다.")
        
        # 첫 번째 결과 상세 출력 예시
        first_result = final_state["final_postings"][0]
        print("\n--- 최종 결과물 예시 ---")
        print(f"**회사명:** {first_result.posting_data.company}")
        print(f"**공고명:** {first_result.posting_data.title}")
        print("\n**[요약 정보]**")
        print(first_result.posting_data.summary)
        print("\n**[기업 분석 리포트]**")
        print(f"- 기업 개요: {first_result.analysis_report.company_overview}")
        print(f"- 기술 스택: {first_result.analysis_report.tech_stack}")
        print(f"- 경쟁사 분석: {first_result.analysis_report.competitor_analysis}")
        print(f"- 포트폴리오 제안: {first_result.analysis_report.portfolio_suggestions}")
        print("--------------------")
    else:
        print("처리된 결과물이 없습니다. 모든 공고가 중복되었을 수 있습니다.")

    # 5. DB 연결 종료
    #    프로세스가 끝날 때 DB 커넥션을 닫아줍니다.
    db_tool.close_connection()