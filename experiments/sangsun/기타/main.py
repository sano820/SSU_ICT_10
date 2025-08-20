
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# .env 파일에서 환경 변수를 로드합니다. (OPENAI_API_KEY)
load_dotenv()

# 프로젝트의 다른 모듈들을 임포트합니다.
from experiments.sangsun.memory import AgentState
from experiments.sangsun.planning_agent import planner_node
from experiments.sangsun.execution_agent import execution_node

# 그래프 워크플로우를 정의합니다.
workflow = StateGraph(AgentState)

# 노드들을 그래프에 추가합니다.
workflow.add_node("planner", planner_node)
workflow.add_node("executor", execution_node)

# 그래프의 흐름(엣지)을 정의합니다.
workflow.set_entry_point("planner")  # 시작점은 planner
workflow.add_edge("planner", "executor")  # planner 다음은 executor

# 조건부 엣지를 정의합니다.
# executor 노드 실행 후, 남은 plan이 있는지 확인하고 분기합니다.
def should_continue(state: AgentState) -> str:
    """
    실행 후 다음 상태를 결정하는 조건부 엣지 함수입니다.
    남은 계획이 있으면 'executor'로, 없으면 'END'로 분기합니다.
    """
    if not state['plan']:
        print("---모든 계획 완료, 종료---")
        return END
    else:
        print("---남은 계획 존재, 실행 계속---")
        return "executor"

workflow.add_conditional_edges("executor", should_continue)

# 그래프를 컴파일하여 실행 가능한 객체로 만듭니다.
app = workflow.compile()


# 메인 실행 블록
if __name__ == "__main__":
    # 사용자의 질문을 정의합니다.
    question = "현재 대한민국 대통령은 누구인지 검색하고, 그 사람에 대한 위키피디아 정보를 요약해줘."

    # 초기 상태를 설정하고 그래프를 실행합니다.
    initial_state = {"question": question, "plan": [], "past_steps": [], "response": ""}
    final_state = app.invoke(initial_state)

    # 최종 결과 출력
    print("\n\n===== 최종 실행 결과 =====")
    for i, (step, result) in enumerate(final_state['past_steps']):
        print(f"단계 {i+1}: {step}")
        print(f"결과: {result}\n")