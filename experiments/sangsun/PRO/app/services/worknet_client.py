from typing import Any, List, Dict, Optional

# 실제 Worknet API 연동 전까지의 스텁 클라이언트
# 추후 httpx.AsyncClient로 교체하여 페이징/재시도/쿼터 처리 권장


class WorknetClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    async def fetch(self, since: str | None = None) -> List[Dict[str, Any]]:
        """
        TODO:
        - httpx.AsyncClient로 Worknet API 호출
        - since(ISO8601) 기준 필터
        - 페이지네이션 처리
        - 실패/재시도(tenacity)
        반환 스키마(예시):
        [
          {
            "source_id": "W123",
            "company_name": "Acme",
            "title": "백엔드 엔지니어",
            "description": "...",
            "location": "Seoul",
            "employment_type": "full-time",
            "salary": "면접 후 협의",
            "posted_at": "2025-08-20T03:00:00Z",
            "deadline_at": None,
            "url": "https://worknet.example/job/W123"
          },
          ...
        ]
        """
        # 지금은 빈 리스트 반환 (스켈레톤)
        return []
