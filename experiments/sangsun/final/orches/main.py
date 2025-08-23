import os
import config # API 키 로드를 위해 import
from orchestrators.daily_job_orchestrator import run_daily_job_recommendation
from orchestrators.report_orchestrator import run_analysis_report

def main():
    print("채용 모니터링 챗봇 오케스트라에 오신 것을 환영합니다!")
    print("="*50)

    while True:
        print("\n실행할 시나리오를 선택해주세요:")
        print("1. 일일 채용 공고 추천 (시나리오 1)")
        print("2. 기업 분석 및 포트폴리오 제안 (시나리오 2)")
        print("3. 종료")
        choice = input(">> ")

        if choice == '1':
            # 시나리오 1 실행
            # 실제 서비스에서는 로그인된 사용자 ID를 사용해야 합니다.
            run_daily_job_recommendation(user_id="user_123")

        elif choice == '2':
            # 시나리오 2 실행
            job = input("분석할 직무를 입력하세요 (예: 백엔드 개발자): ")
            companies_str = input("분석할 기업들을 콤마(,)로 구분하여 입력하세요 (예: 카카오,네이버): ")
            companies = [c.strip() for c in companies_str.split(',')]
            run_analysis_report(target_job=job, target_company=companies)

        elif choice == '3':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 다시 입력해주세요.")

if __name__ == '__main__':
    main()