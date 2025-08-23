from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class MatchingRankingAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = ChatPromptTemplate.from_template("""
당신은 최고의 채용 매니저입니다. 아래 사용자 프로필과 채용 공고 목록을 비교하여, 사용자에게 가장 적합한 공고 3개를 추천해주세요.

각 추천 공고에 대해, 어떤 점(기술, 경험, 키워드)이 사용자 프로필과 잘 맞는지 **'추천 이유'**를 명확하게 한 문장으로 설명해야 합니다.

--- 사용자 프로필 ---
{user_profile}

--- 채용 공고 목록 ---
{job_postings}

--- 출력 형식 ---
1. [회사명] - [공고 제목]
   - 추천 이유:
   - 상세 링크:
2. ...
""")
        self.chain = self.prompt | self.llm

    def rank_jobs(self, user_profile: dict, job_postings: list) -> str:
        """사용자 프로필과 가장 잘 맞는 채용 공고를 순위 매겨 반환합니다."""
        # LLM 처리를 위해 데이터를 문자열로 변환
        profile_str = "\n".join([f"- {k}: {v}" for k, v in user_profile.items()])
        jobs_str = "\n\n".join([
            f"회사: {j.get('company', 'N/A')}\n공고명: {j.get('title', 'N/A')}\n요구사항: {j.get('career', 'N/A')} 경력, {j.get('minEdubg', 'N/A')}\n링크: {j.get('detailUrl', 'N/A')}"
            for j in job_postings
        ])
        
        response = self.chain.invoke({
            "user_profile": profile_str,
            "job_postings": jobs_str
        })
        return response.content