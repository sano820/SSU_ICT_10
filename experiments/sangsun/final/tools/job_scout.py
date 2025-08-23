import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from config import WORKNET_API_KEY

def search_worknet_jobs(keyword: str, num_results: int = 10) -> list:
    """
    고용24(워크넷) Open API를 사용해 최신 채용 공고를 검색합니다.
    """
    if not WORKNET_API_KEY:
        return [{"error": "워크넷 API 키가 설정되지 않았습니다."}]

    url = "http://openapi.work.go.kr/opi/opi/opia/wantedApi.do"
    today = datetime.now()
    start_date = today - timedelta(days=7) # 최근 7일간의 공고 검색

    params = {
        'authKey': WORKNET_API_KEY,
        'callTp': 'L', # 목록 A, 내용 D
        'returnType': 'XML', # XML, JSON
        'startPage': '1',
        'display': str(num_results),
        'regDate': 'D', # D 등록일, E 마감일
        'startDate': start_date.strftime('%Y%m%d'),
        'endDate': today.strftime('%Y%m%d'),
        'keyword': keyword
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        jobs = []
        for wanted in root.findall('wanted'):
            job = {
                'company': wanted.findtext('company'),
                'title': wanted.findtext('title'),
                'salTpNm': wanted.findtext('salTpNm'), # 급여 형태
                'sal': wanted.findtext('sal'), # 급여
                'region': wanted.findtext('region'),
                'holidayTpNm': wanted.findtext('holidayTpNm'), # 근무형태
                'minEdubg': wanted.findtext('minEdubg'), # 최소학력
                'career': wanted.findtext('career'), # 경력
                'closeDt': wanted.findtext('closeDt'), # 마감일
                'wantedAuthNo': wanted.findtext('wantedAuthNo'),
                'detailUrl': f"https://www.work.go.kr/empInfo/empInfoSrch/detail/empDetailAuthView.do?wantedAuthNo={wanted.findtext('wantedAuthNo')}"
            }
            jobs.append(job)
        return jobs

    except requests.exceptions.RequestException as e:
        return [{"error": f"API 요청 중 오류 발생: {e}"}]
    except ET.ParseError as e:
        return [{"error": f"XML 파싱 중 오류 발생: {e}"}]