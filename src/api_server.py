import uvicorn
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from database import init_db, get_db, User, CompanyInterest, AnalysisReport
from orchestrator import run_daily_job_recommendation, run_analysis_report_task

# FastAPI 앱 생성
app = FastAPI(title="채용 모니터링 챗봇 API", version="1.0")

@app.on_event("startup")
def on_startup():
    """서버 시작 시 DB 테이블을 생성합니다."""
    print("데이터베이스를 초기화합니다...")
    init_db()
    print("데이터베이스 초기화 완료.")

# --- API 엔드포인트 정의 ---

@app.post("/users/", response_model=dict)
def create_user(username: str, db: Session = Depends(get_db)):
    """테스트용 사용자 생성"""
    db_user = User(username=username)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"id": db_user.id, "username": db_user.username}

@app.post("/users/{user_id}/interests/", response_model=dict)
def add_company_interest(user_id: int, company_name: str, db: Session = Depends(get_db)):
    """사용자 관심 기업 추가"""
    interest = CompanyInterest(user_id=user_id, company_name=company_name)
    db.add(interest)
    db.commit()
    return {"status": "success"}

@app.post("/tasks/daily-recommendation/{user_id}", response_model=dict)
def trigger_daily_recommendation(user_id: int, db: Session = Depends(get_db)):
    """[시나리오 1] 특정 사용자의 일일 채용 공고 추천을 실행합니다."""
    run_daily_job_recommendation(db, user_id)
    return {"message": f"사용자(id={user_id})의 일일 채용 공고 추천 작업이 실행되었습니다."}


@app.post("/tasks/analysis-report/", response_model=dict)
def request_analysis_report(
    user_id: int,
    target_job: str,
    target_company_str: str, # "카카오,네이버" 형태의 문자열로 받음
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """[시나리오 2] 분석 리포트 생성을 요청합니다. (비동기 처리)"""
    companies = [c.strip() for c in target_company_str.split(',')]
    
    # 1. DB에 리포트 생성 요청 기록
    new_report = AnalysisReport(
        user_id=user_id,
        target_job=target_job,
        target_company=", ".join(companies), # DB에는 문자열로 저장
        status="PENDING"
    )
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    
    # 2. 백그라운드에서 리포트 생성 작업 실행
    background_tasks.add_task(
        run_analysis_report_task, 
        new_report.id, 
        target_job, 
        companies, 
        db=next(get_db()) # 백그라운드 태스크에 새 DB 세션을 전달
    )
    
    return {"message": "리포트 생성이 시작되었습니다.", "report_id": new_report.id}


@app.get("/reports/{report_id}", response_model=dict)
def get_report_status(report_id: int, db: Session = Depends(get_db)):
    """생성된 리포트의 상태와 결과를 조회합니다."""
    report = db.query(AnalysisReport).filter(AnalysisReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="리포트를 찾을 수 없습니다.")
    
    return {
        "report_id": report.id,
        "status": report.status,
        "target_job": report.target_job,
        "target_company": report.target_company,
        "created_at": report.created_at,
        "content": report.content,
        "error_message": report.error_message
    }


if __name__ == "__main__":
    # 서버 실행 (터미널에서 uvicorn api_server:app --reload)
    uvicorn.run(app, host="0.0.0.0", port=8000)