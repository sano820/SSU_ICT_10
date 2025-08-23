from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# 도구들을 tools 패키지에서 가져옵니다.
from tools import market_research

class CompanyAnalystAgent:
    def __init__(self):
        """
        에이전트 초기화 시 LLM과 분석 체인을 설정합니다.
        """
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        ANALYSIS_PROMPT_TEMPLATE = """
당신은 {target_job} 분야 전문 커리어 애널리스트입니다. 주어진 검색 결과를 바탕으로 요청사항을 분석하고 핵심 내용을 한국어로 요약해주세요.
--- 검색 결과 ---
{search_results}
--- 분석 요청 ---
{request}
"""
        prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)
        self.analyzer_chain = prompt | self.llm

    def analyze_domestic_market(self, target_job: str, target_companies: list[str]) -> dict:
        """
        국내 채용 시장을 채용 공고, 합격 후기, 현직자 인터뷰 관점에서 분석합니다.
        """
        print("--- 국내 채용 시장 분석 실행 ---")
        company_query = " OR ".join(target_companies)

        # 1. 채용 공고 분석
        print("1. 채용 공고 분석 중...")
        postings_results = market_research.tavily_web_search(
            f'"{company_query}" {target_job} 신입 채용 공고 자격요건 우대사항'
        )
        postings_summary = self.analyzer_chain.invoke({
            "target_job": target_job,
            "search_results": postings_results,
            "request": f"""'{company_query}'의 '{target_job}' 신입 채용 공고를 분석하여, [직무 목표], [주요 책임], [핵심 역량(Hard Skills, 협업 방식)], [우대 사항] 구조에 맞춰 측정 가능하고 구체적인 정보만 추출해줘. 추상적인 키워드는 제외할 것."""
        }).content

        # 2. 합격 후기 분석 (최근 3년)
        print("2. 합격 후기 분석 중...")
        current_year = datetime.now().year
        years_query = " OR ".join(str(y) for y in range(current_year, current_year - 3, -1))
        reviews_web_results = market_research.tavily_web_search(
            f'"{company_query}" {target_job} 신입 합격 후기 ({years_query}) site:velog.io OR site:tistory.com OR site:brunch.co.kr'
        )
        
        youtube_summary = ""
        try:
            video_urls_str = market_research.find_youtube_videos(
                topic=f'"{company_query}" {target_job} 신입 합격 후기',
                language="korean",
                time_filter="3 years"
            )
            if video_urls_str and "Error" not in video_urls_str:
                first_video_url = video_urls_str.splitlines()[0].strip()
                print(f"합격 후기 영상 분석 중: {first_video_url}")
                youtube_summary = market_research.analyze_video_content(
                    video_url=first_video_url,
                    question=f"'{company_query}'의 '{target_job}' 직무 신입 합격 후기 정보들을 종합하여, [합격자 프로필], [채용 프로세스별 준비 사항], [결정적 합격 증거] 구조로 검증 가능한 사실 기반의 합격 전략을 정리해줘."
                )
        except Exception as e:
            print(f"유튜브 합격 후기 분석 중 오류: {e}")
            youtube_summary = "유튜브 영상 분석 중 오류가 발생했습니다."

        reviews_summary = self.analyzer_chain.invoke({
            "target_job": target_job,
            "search_results": f"{reviews_web_results}\n\n{youtube_summary}",
            "request": "웹 검색 결과와 유튜브 영상 분석 결과를 종합하여 합격자들의 프로필, 준비 과정, 핵심 경쟁력을 요약해줘."
        }).content

        # 3. 현직자 인터뷰 분석
        print("3. 현직자 인터뷰 분석 중...")
        interviews_results = market_research.tavily_web_search(
            f'"{company_query}" {target_job} 현직자 인터뷰 "일하는 방식" "조직 문화"'
        )
        interview_summary = self.analyzer_chain.invoke({
            "target_job": target_job,
            "search_results": interviews_results,
            "request": f"'{company_query}'의 '{target_job}' 현직자 인터뷰 내용을 바탕으로, [주니어의 첫 1년], [성과 측정 방식], [구체적인 성장 조언] 구조로 실천 가능한 정보만 분석해줘. 추상적인 조언은 배제할 것."
        }).content

        # 최종 결과 종합
        final_summary = f"""### **{', '.join(target_companies)} {target_job} 직무 국내 시장 분석**

#### **1. 채용 공고에서 드러난 공식 요구사항**
{postings_summary}

#### **2. 합격자들이 말하는 실제 취업 준비 과정 (최근 3년)**
{reviews_summary}

#### **3. 현직자가 말하는 실제 업무 환경과 성장 포인트**
{interview_summary}
"""
        distilled_summary = self.llm.invoke(f"다음 상세 분석 보고서에서, '{target_job}' 직무의 국내 시장 현황에 대한 가장 핵심적인 내용만 한 문장으로 요약해줘.\n---\n{final_summary}").content
        
        return {
            "final_summary": final_summary,
            "distilled_summary": distilled_summary
        }

    def analyze_global_trends(self, target_job: str, target_company: str, domestic_summary: str) -> dict:
        """
        글로벌 기술 및 시장 동향을 분석하고 미래를 예측합니다.
        """
        print("\n--- 글로벌 동적 트렌드 분석 실행 ---")
        
        # 1. 검색 키워드 생성
        keyword_prompt = f"'{target_job}' 직무의 미래 동향을 파악하기 위한 영문 검색 키워드를 5개 생성해주세요."
        search_keywords = self.llm.invoke(keyword_prompt).content
        print(f"-> 생성된 검색 키워드: {search_keywords}")

        # 2. 거시 환경(시장) 트렌드 분석
        industry_name_prompt = f"'{target_company}' 회사가 속한 핵심 산업 분야를 한 단어의 영문으로 알려줘."
        industry_name = self.llm.invoke(industry_name_prompt).content.strip()
        news_query = f'"{target_company}" AND ("{industry_name}" OR business strategy OR market trend OR investment)'
        macro_results = market_research.search_global_news(news_query)
        macro_trends_summary = self.analyzer_chain.invoke({
            "target_job": target_job, "search_results": macro_results,
            "request": f"'{industry_name}' 산업과 '{target_company}'에 대한 최신 뉴스 기사를 바탕으로, 이 산업에 영향을 미치는 가장 중요한 시장 동향이나 비즈니스 변화를 객관적으로 요약해줘."
        }).content

        # 3. 실무자 트렌드 분석
        practitioner_query = f'"{target_job}" AND ({search_keywords}) case study OR best practices OR industry report'
        practitioner_results = market_research.tavily_web_search(practitioner_query)
        practitioner_trends_summary = self.analyzer_chain.invoke({
            "target_job": target_job, "search_results": practitioner_results,
            "request": "최신 글로벌 성공 사례나 전문가 리포트를 분석하여, 현재 가장 주목받는 새로운 업무 방식이나 성공 방정식을 요약해줘."
        }).content

        # 4. 종합 및 미래 예측
        combined_trends = f"--- [시장 관점] 주요 뉴스 분석 ---\n{macro_trends_summary}\n\n--- [실무자 관점] 성공 방정식 분석 ---\n{practitioner_trends_summary}"
        prediction_prompt = f"""당신은 최고의 전략 컨설턴트입니다. 아래 정보들을 '연결'하고 '추론'하여 최종 보고서를 작성해주세요.
[국내 현황]: {domestic_summary}
[글로벌 동향]: {combined_trends}

[보고서 작성 지시]
1. [미래 격차 분석]: 국내 현황과 글로벌 동향 사이의 격차는 무엇이며, 이 격차가 '{target_job}' 직무에 어떤 위기나 기회를 가져올지 분석해줘.
2. [기회의 창]: 이 격차를 파고들어 남들보다 앞서나갈 수 있는 구체적인 '기회의 창'은 무엇인지 정의해줘.
3. [액션 플랜]: 그 기회를 잡기 위해 지금 당장 시작해야 할 가장 중요한 액션 플랜 3가지를 제시해줘.
"""
        prediction = self.llm.invoke(prediction_prompt).content
        distilled_prediction = self.llm.invoke(f"다음 보고서의 핵심 결론을 한 문장으로 요약해줘.\n---\n{prediction}").content
        
        return {
            "generated_keywords": search_keywords,
            "prediction": prediction,
            "distilled_prediction": distilled_prediction
        }