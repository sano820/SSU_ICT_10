# src/components/memory.py

from typing import List, Tuple, Annotated
from typing_extensions import TypedDict
import operator

# LangGraph의 상태(State)를 정의하는 부분입니다.
# 그래프의 각 노드를 거치면서 이 State 객체가 계속 업데이트됩니다.
class AgentState(TypedDict):
    """
    AgentState는 워크플로우의 상태를 나타냅니다.

    Attributes:
        question (str): 사용자의 초기 질문
        plan (List[str]): Planner에 의해 생성된 단계별 계획
        past_steps (Annotated[List[Tuple], operator.add]): 실행된 단계와 그 결과의 기록
        response (str): 최종 생성된 응답
    """
    question: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str