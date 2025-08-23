from agents.user_profiler import UserProfilerAgent
from agents.matching_agent import MatchingRankingAgent
from tools.job_scout import search_worknet_jobs

# 이 부분은 실제 서비스에서 DB나 파일에서 사용자 정보를 가져와야 합니다.
DUMMY_USER_DATA = {
    "user_123": {
        "name": "김개발",
        "resume_text": "Python과 Django, AWS 사용에 능숙한 3년차 백엔드 개발자입니다. 최근 MSA 구조에 관심이 많아 FastAPI와 Docker를 학습하고 있습니다. 희망 직무는 백엔드 개발자입니다."
    }
}

def run_daily_job_recommendation(user_id: str):
    """지정된 사용자를 위해 일일 채용 공고 추천 작업을 실행합니다."""
    print(f"--- {user_id} 사용자를 위한 일일 채용 추천 시작 ---")

    # 1. 사용자 정보 가져오기
    user_info = DUMMY_USER_DATA.get(user_id)
    if not user_info:
        return "사용자 정보를 찾을 수 없습니다."

    # 2. 사용자 프로파일러 에이전트 실행
    profiler = UserProfilerAgent()
    user_profile = profiler.create_profile(user_info["resume_text"])
    print(f"✅ 사용자 프로필 생성 완료: {user_profile}")

    # 3. 채용 공고 스카우터 실행 (프로필의 키워드 사용)
    search_keyword = f"{user_profile.get('desired_job', '')} {user_profile.get('skills', [])[0]}"
    job_postings = search_worknet_jobs(keyword=search_keyword)
    print(f"✅ 채용 공고 {len(job_postings)}건 수집 완료")
    if not job_postings or 'error' in job_postings[0]:
        return "채용 공고를 가져오는 데 실패했습니다."

    # 4. 매칭 및 랭킹 에이전트 실행
    matcher = MatchingRankingAgent()
    recommendations = matcher.rank_jobs(user_profile, job_postings)
    print("✅ 맞춤 공고 추천 완료")

    # 5. 최종 결과 생성 및 알림 (실제로는 Slack, 카톡 등으로 전송)
    final_message = f"🌟 {user_info['name']}님을 위한 오늘의 맞춤 채용 공고!\n\n{recommendations}"
    print("\n--- 최종 결과 ---")
    print(final_message)
    
    return final_message