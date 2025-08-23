from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 도구들을 tools 패키지에서 가져옵니다.
from tools import market_research

class PortfolioStrategistAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    def should_research_papers(self, target_job: str) -> bool:
        """
        직무 특성을 분석하여 논문 분석이 필요한지 여부를 판단합니다.
        """
        print("\n--- [Router] 논문 분석 필요 여부 판단 중... ---")
        prompt = f"""사용자의 목표 직무는 '{target_job}'입니다. 이 직무는 최신 학술 연구 논문을 깊이 있게 분석하는 것이 취업 준비에 결정적으로 중요한 R&D 또는 딥테크 분야에 해당합니까? 오직 'yes' 또는 'no'로만 답변해주세요."""
        response = self.llm.invoke(prompt).content.strip().lower()
        print(f"-> LLM의 판단: {response}")
        return "yes" in response

    def analyze_academic_research(self, target_job: str, global_trends: dict) -> dict:
        """
        arXiv에서 관련 최신 논문을 검색하고, 심층 학습 방향을 제시합니다.
        """
        print("\n--- 학술 연구 분석 노드 실행 (arXiv) ---")
        search_keywords = global_trends.get("generated_keywords", target_job)
        query = f'"{target_job}" OR ({search_keywords})'
        print(f"arXiv 동적 검색 쿼리: {query}")

        paper_abstracts = market_research.search_arxiv_papers(query=query, load_max_docs=2)
        
        prompt_template = """당신은 IT 기술 분야의 수석 연구원이자 친절한 멘토입니다.
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
        research_analyzer_chain = prompt | self.llm
        
        summary = research_analyzer_chain.invoke({
            "target_job": target_job,
            "global_prediction": global_trends.get("prediction", "알 수 없음"),
            "abstracts": paper_abstracts
        }).content
        
        distilled_summary = self.llm.invoke(f"다음 학술 분석 보고서의 핵심 결론만 한 문장으로 요약해줘.\n---\n{summary}").content
        
        return {
            "summary": summary,
            "distilled_summary": distilled_summary
        }

    def generate_final_report(self, state: dict) -> dict:
        """
        모든 분석 결과를 종합하여 최종 커리어 로드맵을 생성합니다.
        """
        print("\n--- [Final Step] 최종 보고서 생성 ---")
        
        domestic_summary = state["domestic_analysis"]["distilled_summary"]
        global_prediction = state["global_trends"]["distilled_prediction"]
        
        academic_summary = ""
        # 학술 연구 노드를 거쳤을 경우에만 해당 정보를 포함
        if state.get("academic_research"):
            academic_summary = state["academic_research"]["distilled_summary"]
        
        academic_section = f"\n[최신 연구 동향]: {academic_summary}" if academic_summary else ""

        final_report_prompt = f"""당신은 최고의 커리어 코치입니다. 한 명의 취업 준비생을 위해, 아래의 핵심 정보들을 종합하여 최종적인 '개인 맞춤형 커리어 로드맵'을 작성해주세요. 보고서는 아래 구조를 반드시 따라야 합니다.

**[분석 요약]**
- **국내 현황:** {domestic_summary}
- **글로벌 예측:** {global_prediction}{academic_section}

---

**[최종 커리어 로드맵]**

**1. 목표 설정 (Goal Setting):**
   - 분석 결과를 바탕으로, 지금 시점에서 가장 유망하고 현실적인 커리어 목표를 한 문장으로 정의해주세요. (예: '데이터 기반 의사결정 역량을 갖춘 서비스 백엔드 전문가')

**2. 🗺️ 3단계 로드맵 (3-Step Roadmap):**
   - **(1단계: 기반 다지기 - 1개월)**: 목표 달성을 위해 가장 먼저 학습하고 준비해야 할 필수 기초 지식과 기술은 무엇인가요?
   - **(2단계: 차별화 전략 - 3개월)**: 다른 지원자와 차별화될 수 있는 자신만의 강력한 무기는 무엇이며, 이것을 어떻게 증명할 수 있을까요? (포트폴리오 프로젝트 아이디어 포함)
   - **(3단계: 실전 대비 - 1개월)**: 서류와 면접에서 위 내용을 어떻게 효과적으로 어필할 수 있을지 구체적인 전략을 제시해주세요.

**3. 💡 최종 조언 (Final Advice):**
   - 이 모든 과정을 준비하는 지원자에게 해줄 수 있는 가장 핵심적인 조언 한마디를 남겨주세요.
"""
        final_report = self.llm.invoke(final_report_prompt).content

        return {"final_report": final_report}