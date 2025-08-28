from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import DATETIME as MySQLDateTime
from app.db.models.base import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped["MySQLDateTime"] = mapped_column(
        MySQLDateTime(fsp=6), server_default=func.now(), nullable=False
    )
