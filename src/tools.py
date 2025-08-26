import os
import requests
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict, Any
import glob
import config

import os # os import 추가
from langchain_core.tools import tool
from googleapiclient.discovery import build

from langchain.tools import tool
from langchain_community.tools.youtube.search import YouTubeSearchTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

"""유튜브 데이터 크롤링"""
def analyze_youtube_topic(
    topic: str,
    analysis_prompt: str,
    api_key: str,
    lang_code: str = "ko",
    max_results: int = 5,
    num_to_analyze: int = 2,
    transcripts_only: bool = True  # <-- [수정] 자막 필터링 여부를 인자로 받도록 추가
) -> str:
    """
    주어진 주제(topic)로 유튜브 영상을 검색하고, 주어진 프롬프트(analysis_prompt)로 내용을 분석하여 요약합니다.
    """
    print(f"\n-> 유튜브 '{topic}' 주제 분석 시작...")
    try:
        # [수정] 💡 인자로 받은 transcripts_only 값을 사용합니다.
        videos = find_videos_with_transcripts.invoke({
            "topic": topic,
            "api_key": api_key,
            "lang_code": lang_code,
            "transcripts_only": transcripts_only, # <-- 수정된 부분
            "max_results": max_results
        })

        # --- 이하 함수의 나머지 부분은 모두 동일합니다 ---
        if not videos:
            print(f"-> 분석할 '{topic}' 관련 영상이 없습니다.")
            return f"관련 유튜브 영상 없음."
            
        videos_to_analyze = videos[:num_to_analyze]
        print(f"-> 총 {len(videos)}개의 영상을 찾았으며, 상위 {len(videos_to_analyze)}개를 분석합니다:")
        for video in videos_to_analyze:
            print(f"  - {video.get('title', '제목 없음')}")

        analysis_tasks = [
            {
                "video_url": video['url'],
                "question": analysis_prompt
            }
            for video in videos_to_analyze
        ]
        
        analysis_results = analyze_video_content.batch(analysis_tasks)
        
        return "\n\n---\n\n".join(analysis_results)

    except Exception as e:
        print(f"-> ⚠️ 유튜브 '{topic}' 분석 중 오류: {e}")
        return f"유튜브 '{topic}' 분석 중 오류 발생."

@tool
def find_videos_with_transcripts(
    topic: str,
    api_key: str,
    max_results: int = 10,
    lang_code: str = 'en',
    transcripts_only: bool = True
) -> List[Dict[str, Any]]:
    """
    주어진 주제로 유튜브를 검색하여 영상의 상세 정보 리스트를 반환합니다.
    """
    print(f"-> '{topic}' 주제로 영상 검색 시작 (자막 필터링: {transcripts_only}, 언어: {lang_code})")
    
    try:
        if not api_key:
            raise ValueError("API 키가 전달되지 않았습니다.")
        youtube_service = build('youtube', 'v3', developerKey=api_key)
    except Exception as e:
        print(f"-> ⚠️ YouTube API 서비스 생성 실패. API 키를 확인하세요. (오류: {e})")
        return []

    try:
        search_response = youtube_service.search().list(
            q=topic,
            part='snippet',
            type='video',
            order='relevance',
            maxResults=max_results
        ).execute()
        
        video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
        if not video_ids:
            return []

        video_details_response = youtube_service.videos().list(
            part='snippet,statistics',
            id=','.join(video_ids)
        ).execute()
        
        all_video_details = []
        for item in video_details_response.get('items', []):
            all_video_details.append({
                "title": item['snippet']['title'],
                "url": f"https://www.youtube.com/watch?v={item['id']}",
                "view_count": int(item['statistics'].get('viewCount', 0)),
                "video_id": item['id']
            })
            
        if not transcripts_only:
            return all_video_details

        final_videos = []
        for video in all_video_details:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video['video_id'])
                transcript_list.find_transcript([lang_code])
                final_videos.append(video)
            except Exception:
                continue
                
        return final_videos

    except Exception as e:
        print(f"-> 유튜브 상세 정보 검색 중 오류 발생: {e}")
        return []


@tool
def find_youtube_videos(topic: str, language: str, time_filter: str = "3 years") -> str:
    """
    특정 주제(topic)의 YouTube 영상을 지정된 언어(language)와 기간(time_filter) 조건으로 검색합니다.
    예: topic="데이터 분석", language="korean", time_filter="1 year"
    """
    # 검색어를 조합하여 더 정확한 검색을 유도합니다.
    query = f"{topic} {language} within {time_filter}"
    print(f"Executing search with query: '{query}'") # 어떤 검색어로 실행되는지 확인

    # max_results를 2로 고정하여 2개씩 찾도록 합니다.
    tool_instance = YouTubeSearchTool(max_results=2)
    return tool_instance.run(query)

# 도구 2: [FINAL & CORRECTED] GenericLoader와 Parser를 사용한 영상 내용 분석 도구
@tool
def analyze_video_content(video_url: str, question: str) -> str:
    """
    특정 YouTube 영상의 오디오를 추출하고 텍스트로 변환하여, 주어진 질문에 답변합니다.
    영상 URL과 분석할 질문, 두 가지를 반드시 입력해야 합니다.
    """
    save_dir = "./temp_audio"
    try:
        # 1. [수정] GenericLoader를 사용하여 로더와 파서를 결합합니다.
        # 1-1. 재료 준비 담당: 유튜브 오디오를 가져올 로더 설정
        blob_loader = YoutubeAudioLoader([video_url], save_dir)

        # 1-2. 요리사: 오디오를 텍스트로 변환할 파서 설정
        parser = OpenAIWhisperParser()

        # 1-3. 주방 시스템: 로더와 파서를 GenericLoader로 연결
        loader = GenericLoader(blob_loader, parser)

        # 2. [수정] GenericLoader를 실행하여 문서(텍스트)를 가져옵니다.
        docs = loader.load()

        if not docs:
            return "영상 내용을 분석할 수 없습니다. (오디오 추출 또는 변환 실패)"

        transcript_text = docs[0].page_content

        # 3. 변환된 텍스트를 기반으로 질문에 답변합니다.
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", 
                             temperature=0,
                             google_api_key=config.GEMINI_API_KEY)
        prompt = f"""
        아래는 YouTube 영상의 전체 음성 내용을 텍스트로 변환한 것입니다.
        이 내용을 바탕으로 다음 질문에 대해 상세하게 답변해주세요.

        --- 변환된 텍스트 내용 ---
        {transcript_text[:4000]}

        --- 질문 ---
        {question}
        """
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"영상 분석 중 오류 발생: {str(e)}"
    finally:
        # 4. 작업이 끝나면 다운로드했던 임시 오디오 파일을 모두 삭제합니다.
        if os.path.exists(save_dir):
            files = glob.glob(os.path.join(save_dir, '*'))
            for f in files:
                os.remove(f)

"""네이버 뉴스 데이터 크롤링"""

@tool
def search_naver_news(company_name: str, display_count: int = 5) -> str:
    """
    네이버 뉴스에서 특정 회사(company_name)에 대한 최신 뉴스를 검색합니다.
    검색할 결과 개수(display_count)를 지정할 수 있습니다.
    """

    # 2. API 요청 정보 설정
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": os.environ['NAVER_CLIENT_ID'],
        "X-Naver-Client-Secret": os.environ['NAVER_CLIENT_SECRET']
    }
    params = {
        "query": company_name,
        "display": display_count,
        "sort": "date"  # 최신순으로 정렬
    }

    # 3. API 호출
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # 오류가 발생하면 예외를 발생시킴

        # 4. 결과 파싱 및 정리
        data = response.json()
        items = data.get("items", [])

        if not items:
            return f"'{company_name}'에 대한 네이버 뉴스 검색 결과가 없습니다."

        # 결과를 LLM이 이해하기 좋은 형태로 가공
        formatted_results = []
        for item in items:
            title = item.get("title", "").replace("<b>", "").replace("</b>", "").replace("&quot;", '"')
            description = item.get("description", "").replace("<b>", "").replace("</b>", "").replace("&quot;", '"')
            link = item.get("link", "")
            pub_date = item.get("pubDate", "")

            formatted_results.append(
                f"- 제목: {title}\n"
                f"  - 날짜: {pub_date}\n"
                f"  - 요약: {description}\n"
                f"  - 링크: {link}"
            )

        return "\n\n".join(formatted_results)

    except requests.exceptions.RequestException as e:
        return f"API 요청 중 오류가 발생했습니다: {e}"
    except Exception as e:
        return f"오류가 발생했습니다: {e}"

"""해외 뉴스 데이터 크롤링"""

@tool
def search_global_news(query: str, days_ago: int = 30) -> str:
    """
    해외 주요 언론사를 대상으로 특정 주제(query)에 대한 뉴스를 검색합니다.
    최근 며칠(days_ago) 내의 뉴스를 검색할지 지정할 수 있습니다. (기본값: 30일)
    외국계 기업을 조사하거나, 국내 기업의 해외 반응을 볼 때 유용합니다.
    """

    # 2. API 요청 정보 설정
    url = "https://newsapi.org/v2/everything"

    # 검색 시작 날짜 계산
    from_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

    params = {
        "q": query,
        "apiKey": os.environ['NEWS_API_KEY'],
        "from": from_date,
        "sortBy": "relevancy",  # 관련성 순으로 정렬
        "language": "en",       # 영어 뉴스 우선 검색
        "pageSize": 5           # 최대 5개의 결과만 가져옴
    }

    # 3. API 호출
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        # 4. 결과 파싱 및 정리
        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return f"'{query}'에 대한 해외 뉴스 검색 결과가 없습니다."

        formatted_results = []
        for article in articles:
            title = article.get("title", "")
            source = article.get("source", {}).get("name", "")
            description = article.get("description", "")
            url = article.get("url", "")
            pub_date = article.get("publishedAt", "").split("T")[0] # 날짜만 표시

            formatted_results.append(
                f"- 제목: {title}\n"
                f"  - 출처: {source} ({pub_date})\n"
                f"  - 요약: {description}\n"
                f"  - 링크: {url}"
            )

        return "\n\n".join(formatted_results)

    except requests.exceptions.RequestException as e:
        return f"API 요청 중 오류가 발생했습니다: {e}"
    except Exception as e:
        return f"오류가 발생했습니다: {e}"

"""Arxiv 논문 크롤링"""

@tool
def search_arxiv_papers(query: str, load_max_docs: int = 2) -> str:
    """
    주어진 쿼리(query)로 arXiv에서 최신 논문을 검색하고, 각 논문 초록의 길이를
    안전하게 잘라서(2500자) 너무 길지 않게 정리한 후 최종 결과를 반환합니다.
    """
    print(f"--> (ArXiv 도구) 논문 검색 실행: {query}")
    try:
        loader = ArxivLoader(
            query=query,
            load_max_docs=load_max_docs,
            sort_by="submittedDate"
        )
        docs = loader.load()

        if not docs:
            return "관련 최신 논문을 찾을 수 없습니다."

        summaries = []
        for doc in docs:
            # [⭐핵심⭐] 각 논문 초록의 내용을 최대 2500자로 제한합니다.
            truncated_abstract = doc.page_content[:2500]
            summary = f"논문 제목: {doc.metadata['Title']}\n초록: {truncated_abstract}"
            summaries.append(summary)

        final_summary = "\n\n---\n\n".join(summaries)
        print(f"-> ArXiv 검색 완료. (최종 텍스트 길이: {len(final_summary)})")
        return final_summary

    except Exception as e:
        print(f"arXiv 검색 중 오류 발생: {e}")
        return "arXiv에서 논문을 검색하는 중 오류가 발생했습니다."

"""웹 서칭 크롤링"""

@tool
def tavily_web_search(query: str) -> str:
    """
    주어진 쿼리(query)로 웹을 검색하고, 각 검색 결과의 내용을
    일정한 길이(3000자)로 잘라서 너무 길지 않게 정리한 후 최종 결과를 반환합니다.
    """
    print(f"--> (스마트 검색 도구) 실행: {query[:50]}...")

    # 1. k=3: 최대 3개의 가장 관련성 높은 웹페이지만 가져오도록 설정
    retriever = TavilySearchAPIRetriever(k=3)

    # 2. retriever를 실행하여 Document 객체 리스트를 받음
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        return f"웹 검색 중 오류가 발생했습니다: {e}"

    # 3. 각 Document의 내용을 잘라서 하나의 문자열로 합치기
    summaries = []
    for doc in docs:
        # [⭐핵심⭐] 각 문서의 내용을 최대 3000자로 제한합니다.
        # 이렇게 하면 LLM의 컨텍스트 창을 넘을 가능성이 거의 없습니다.
        truncated_content = doc.page_content[:3000]
        summary = f"--- 검색 결과 출처: {doc.metadata.get('source', 'N/A')} ---\n{truncated_content}"
        summaries.append(summary)

    final_summary = "\n\n".join(summaries)
    print(f"-> 스마트 검색 완료. (최종 텍스트 길이: {len(final_summary)})")

    return final_summary
