# app/db/models/user_pref.py
from sqlalchemy import String, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import DATETIME as MySQLDateTime
from app.db.models.base import Base


class UserPreference(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    # JSON 컬럼: 리스트/딕셔너리 모두 허용
    target_roles: Mapped[dict | list | None] = mapped_column(JSON)       # ex) ["백엔드","데이터 엔지니어"]
    target_locations: Mapped[dict | list | None] = mapped_column(JSON)   # ex) ["Seoul","Remote"]
    seniority: Mapped[str | None] = mapped_column(String(64))


class UserInterestCompany(Base):
    __tablename__ = "user_interest_companies"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), primary_key=True)
    created_at: Mapped["MySQLDateTime"] = mapped_column(
        MySQLDateTime(fsp=6), server_default=func.now(), nullable=False
    )
