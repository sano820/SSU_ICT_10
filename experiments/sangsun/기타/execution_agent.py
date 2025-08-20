# agents/execution_agent.py

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from experiments.sangsun.memory import AgentState
from experiments.sangsun.tools import tool_list

# 실행 에이전트의 역할을 정의하는 시스템 프롬프트입니다.
EXECUTION_PROMPT = """
당신은 주어진 도구를 사용하여 주어진 임무를 완수하는 실행 전문가 AI입니다.
당신은 오직 하나의 임무(task)만 받게 되며, 그 임무를 해결하기 위해 어떤 도구를 사용해야 할지 결정하고 실행해야 합니다.
임무 수행 결과를 간결하게 요약하여 반환해주세요.
---
임무: {input}
"""

def execution_node(state: AgentState) -> dict:
    """
    계획의 각 단계를 실제로 실행하는 노드입니다.

    Args:
        state (AgentState): 현재 그래프의 상태

    Returns:
        dict: 업데이트할 상태 값을 담은 딕셔너리. 여기서는 'past_steps'를 업데이트합니다.
    """
    print("---작업 실행 시작---")
    
    # 현재 실행해야 할 계획 단계를 가져옵니다.
    plan = state['plan']
    current_task = plan[0]
    print(f"현재 작업: {current_task}")
    
    # LLM과 프롬프트를 설정하여 에이전트를 생성합니다.
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXECUTION_PROMPT),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        HumanMessage(content="{input}"),
    ])
    
    # LangChain에서 제공하는 기본 에이전트 생성 함수를 사용합니다.
    agent = create_openai_functions_agent(llm, tool_list, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)

    # 현재 작업을 실행하고 결과를 받습니다.
    result = agent_executor.invoke({"input": current_task})
    
    # 실행 결과를 상태에 기록합니다. (현재 작업, 결과) 튜플 형태
    updated_past_steps = (current_task, result['output'])
    
    print(f"작업 결과: {result['output']}")
    print("---작업 실행 완료---")
    
    return {
        "past_steps": [updated_past_steps],  # 리스트에 담아 반환해야 operator.add가 동작
        "plan": plan[1:]  # 실행 완료된 계획은 리스트에서 제거
    }