from sqlalchemy import String, ForeignKey, Float
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import DATETIME as MySQLDateTime
from app.db.models.base import Base

class Portfolio(Base):
    __tablename__ = "portfolios"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str | None] = mapped_column(String(255))
    summary: Mapped[str | None] = mapped_column(String(2000))
    url: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped["MySQLDateTime"] = mapped_column(
        MySQLDateTime(fsp=6), server_default=func.now(), nullable=False
    )

class PortfolioSkillMap(Base):
    __tablename__ = "portfolio_skill_map"
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id"), primary_key=True)
    level: Mapped[float | None] = mapped_column(Float)
