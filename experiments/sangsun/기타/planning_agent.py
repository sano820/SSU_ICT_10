# agents/planning_agent.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from experiments.sangsun.memory import AgentState

# 시스템 프롬프트를 통해 Planner Agent의 역할과 목표를 명확하게 정의합니다.
PLANNER_PROMPT = """
당신은 계획 수립 전문가 AI입니다. 사용자의 질문을 해결하기 위한 명확하고 간결한 단계별 계획을 세워야 합니다.
각 단계는 실행 가능한 하나의 액션이어야 합니다. 예를 들어, "A에 대해 검색하고 B 파일을 읽어 요약해줘" 라는 질문이 들어오면,
1. A에 대해 검색한다.
2. B 파일을 읽는다.
3. 검색 결과와 파일 내용을 종합하여 요약한다.
와 같이 단계를 나누어야 합니다.
각 단계는 번호와 점(.)으로 시작해주세요. (예: 1., 2., 3.)
---
사용자 질문: {question}
"""

def planner_node(state: AgentState) -> dict:
    """
    사용자의 질문을 받아 계획을 수립하는 노드입니다.

    Args:
        state (AgentState): 현재 그래프의 상태

    Returns:
        dict: 업데이트할 상태 값을 담은 딕셔너리. 여기서는 'plan'을 업데이트합니다.
    """
    print("---계획 수립 시작---")
    
    # LLM 모델을 초기화합니다.
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # 프롬프트 템플릿을 생성합니다.
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    
    # LLM 체인을 구성합니다.
    planner = prompt | llm
    
    # 사용자의 질문을 기반으로 계획을 생성합니다.
    plan_string = planner.invoke({"question": state['question']})
    
    # 생성된 계획 문자열을 리스트 형태로 파싱합니다.
    plan = [f"{i+1}. {step}" for i, step in enumerate(plan_string.content.split('\n')) if step]
    
    for step in plan:
        print(step)
        
    print("---계획 수립 완료---")
    
    return {"plan": plan}