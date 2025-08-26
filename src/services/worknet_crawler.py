from sqlalchemy.orm import Session
from database import User, CompanyInterest
import config
import mysql.connector
import requests
import xml.etree.ElementTree as ET
import json
import re
from dotenv import load_dotenv



# ==========================
#  문자열 정규화 (기업명 비교용)
# ==========================
def _normalize(s: str) -> str:
    """기업명/별칭을 단순화하여 비교 가능하게 정규화"""
    if not s:
        return ""
    s = re.sub(r"\(주\)", "", s)         # '(주)' 제거
    s = re.sub(r"[\s·•\-_/]", "", s)     # 공백 및 특수구분자 제거
    s = re.sub(r"[^\w가-힣]", "", s)     # 한글/영문/숫자 외 제거
    return s.lower()

# ==========================
#  관심 기업 세트 구축
# ==========================
def build_interest_set(companies, aliases):
    """DB에서 가져온 기업명 + 별칭을 정규화하여 Set으로 만듦"""
    norm_set = set()
    for c in companies:
        norm_set.add(_normalize(c))
    for a in aliases:
        if a:
            norm_set.add(_normalize(a))
    return norm_set

# ==========================
#  관심 기업 매칭 함수
# ==========================
def is_interest_company(company_name: str, interest_norm_set) -> bool:
    """채용공고 기업명이 관심기업 리스트에 해당하는지 검사"""
    n = _normalize(company_name)
    if not n:
        return False
    # ① 정확 일치
    if n in interest_norm_set:
        return True
    # ② 부분 포함 (앞뒤 문자열 포함)
    for target in interest_norm_set:
        if target in n or n in target:
            return True
    return False


# ==========================
#  DB에서 관심 기업 불러오기
# ==========================
def fetch_companies_from_db(db: Session, user_id: int) -> (list, list):
    """DB에서 특정 사용자의 관심 기업 목록을 불러옵니다."""
    print(f"🔍 DB에서 사용자(id={user_id})의 관심 기업 불러오기")

    # user_id를 기반으로 해당 사용자의 관심 기업만 조회
    interests = db.query(CompanyInterest).filter(CompanyInterest.user_id == user_id).all()
    
    if not interests:
        print(f" 사용자(id={user_id})에게 등록된 관심 기업이 없습니다.")
        return [], []

    companies = [item.company_name for item in interests]
    
    print(f"✅ 관심 기업 {len(companies)}개 불러옴: {companies}")
    return companies, []

# ==========================
#  API 호출 및 관심기업 필터링
# ==========================
def fetch_and_filter_jobs(companies, aliases):
    if not companies:
        print(" 관심 기업이 없습니다.")
        return []

    url = "https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo210L21.do"
    all_raw_jobs = []
    interest_norm_set = build_interest_set(companies, aliases)

    print("\n📦 API 채용공고 수집 시작\n")

    # 페이지 루프 (최대 10페이지만 예시로 설정)
    for page in range(1, 11):
        params = {
            "authKey": config.WORKNET_API_KEY,   
            "callTp": "L",
            "returnType": "XML",
            "startPage": str(page),
            "display": "100"
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f" API 요청 실패 (페이지 {page})")
                break

            root = ET.fromstring(response.content)

            for job_item in root.findall(".//dhsOpenEmpInfo"):
                company_name = job_item.findtext("empBusiNm") or ""

                # 관심기업만 필터링
                if not is_interest_company(company_name, interest_norm_set):
                    continue
                
                print(f" 매칭된 기업: {company_name}")  # 로그 출력 확인용

                job_data = {  ## 워크넷 출력예시에 있는  파라미터들
                    "company_name": company_name,
                    "job_title": job_item.findtext("empWantedTitle"),
                    "employment_type": job_item.findtext("empWantedTypeNm"),
                    "start_date": job_item.findtext("empWantedStdt"),
                    "end_date": job_item.findtext("empWantedEndt"),
                    "company_type": job_item.findtext("coClcdNm"),
                    "company_logo": job_item.findtext("regLogImgNm"),
                    "apply_link": job_item.findtext("empWantedHomepgDetail"),
                }
                all_raw_jobs.append(job_data)

        except Exception as e:
            print(f" API 오류 (페이지 {page}): {e}")
            continue

    print(f"\n 관심기업 공고 수집 결과: {len(all_raw_jobs)}건")
    return all_raw_jobs



def fetch_companies_from_db(db: Session, user_id: int) -> (list, list):
    """DB에서 특정 사용자의 관심 기업 목록을 불러옵니다."""
    print(f"🔍 DB에서 사용자(id={user_id})의 관심 기업 불러오기")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        print(f"❌ 사용자(id={user_id})를 찾을 수 없습니다.")
        return [], []
        
    interests = db.query(CompanyInterest).filter(CompanyInterest.user_id == user_id).all()
    companies = [item.company_name for item in interests]
    # 별칭(alias) 기능은 단순화를 위해 이번 리팩토링에서는 제외
    print(f"✅ 관심 기업 {len(companies)}개 불러옴: {companies}")
    return companies, []


def run_worknet_crawling(db: Session, user_id: int) -> list:
    """한 명의 사용자에 대한 워크넷 크롤링 전체 과정을 실행합니다."""
    companies, aliases = fetch_companies_from_db(db, user_id)
    if not companies:
        return []
    raw_jobs = fetch_and_filter_jobs(companies, aliases)
    return raw_jobs

