# src/components/tools.py

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# LangChain에서 기본으로 제공하는 검색 도구를 초기화합니다.
# 이 도구는 웹 검색을 수행하여 최신 정보를 가져오는 데 사용됩니다.
search_tool = DuckDuckGoSearchRun()

@tool
def file_reader_tool(file_path: str) -> str:
    """
    지정된 경로의 파일을 읽어 그 내용을 문자열로 반환합니다.
    에이전트가 로컬 파일에 접근해야 할 때 사용됩니다.
    예시: file_reader_tool(file_path='./data/some_data.txt')
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"파일 읽기 중 오류 발생: {e}"

# 실행 에이전트가 사용할 도구들을 리스트 형태로 묶습니다.
# 여기에 필요한 커스텀 도구들을 추가할 수 있습니다.
tool_list = [search_tool, file_reader_tool]