from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class UserProfilerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 HR 전문가입니다. 주어진 이력서 텍스트를 분석하여 아래 JSON 형식에 맞춰 핵심 내용을 추출해주세요.
             - skills: 주요 기술 스택 (프로그래밍 언어, 프레임워크, 툴 등)
             - experience_years: 총 경력 (신입은 0)
             - desired_job: 희망 직무
             - keywords: 프로필을 대표하는 핵심 키워드 3가지
             
             {format_instructions}"""),
            ("human", "이력서 내용:\n{resume_text}")
        ])
        self.parser = JsonOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def create_profile(self, resume_text: str) -> dict:
        """이력서 텍스트로부터 사용자 프로필 JSON을 생성합니다."""
        return self.chain.invoke({
            "resume_text": resume_text,
            "format_instructions": self.parser.get_format_instructions()
        })