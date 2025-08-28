from sqlalchemy import String, Float, ForeignKey
from sqlalchemy.dialects.mysql import DATETIME as MySQLDateTime
from sqlalchemy.orm import Mapped, mapped_column, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)


class JobSkillMap(Base):
    __tablename__ = "job_skill_map"

    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id"), primary_key=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    created_at: Mapped["MySQLDateTime"] = mapped_column(
        MySQLDateTime(fsp=6), server_default=func.now(), nullable=False
    )
