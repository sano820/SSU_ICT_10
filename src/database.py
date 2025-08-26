from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func
import config

# SQLAlchemy 설정
engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DB 모델 정의 ---
class User(Base):
    """사용자 정보 모델"""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    # user_profiling 결과를 저장할 필드들
    narrative_summary = Column(Text, nullable=True)
    skills_and_certs = Column(Text, nullable=True)
    experience_specs = Column(Text, nullable=True)
    
    interests = relationship("CompanyInterest", back_populates="user")
    reports = relationship("AnalysisReport", back_populates="user")

class CompanyInterest(Base):
    """사용자 관심 기업 모델"""
    __tablename__ = "company_interests"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    company_name = Column(String, index=True)
    
    user = relationship("User", back_populates="interests")

class AnalysisReport(Base):
    """분석 리포트 저장 모델"""
    __tablename__ = "analysis_reports"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    target_job = Column(String)
    target_company = Column(String)
    status = Column(String, default="PENDING") # PENDING, RUNNING, COMPLETED, FAILED
    content = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="reports")

# --- DB 초기화 함수 ---
def init_db():
    """데이터베이스 테이블 생성"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """DB 세션 제공 함수"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()