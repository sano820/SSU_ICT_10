# langgraph_orchestrator/tools/web_search_tool.py
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# .env 파일에서 환경 변수 로드
# .env 파일 내용 예시:
# TAVILY_API_KEY="your_tavily_api_key"
load_dotenv()

class WebSearchTool:
    def __init__(self):
        """Tavily API 클라이언트를 초기화합니다."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("❌ TAVILY_API_KEY environment variable not set.")
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> str:
        """
        주어진 쿼리로 웹 검색을 수행하고, 결과를 요약된 문자열로 반환합니다.
        """
        try:
            print(f"🔎 Searching for: {query}")
            response = self.client.search(query=query, search_depth="advanced", max_results=max_results)
            # 검색 결과를 하나의 문자열로 합쳐서 반환
            return "\n".join([f"- {obj['content']}" for obj in response['results']])
        except Exception as e:
            print(f"❌ An error occurred during web search: {e}")
            return "검색 중 오류가 발생했습니다."

# 전역 인스턴스로 생성하여 사용
web_search_tool = WebSearchTool()