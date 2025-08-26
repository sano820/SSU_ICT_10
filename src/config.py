import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
WORKNET_API_KEY = os.getenv("WORKNET_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL")