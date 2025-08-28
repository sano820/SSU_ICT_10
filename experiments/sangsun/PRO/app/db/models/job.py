from sqlalchemy import (
    String,
    Text,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import DATETIME as MySQLDateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uniq_source_sourceid"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # e.g., worknet
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)

    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"))
    company_name_raw: Mapped[str | None] = mapped_column(String(255))

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text())
    location: Mapped[str | None] = mapped_column(String(255))
    employment_type: Mapped[str | None] = mapped_column(String(64))
    salary: Mapped[str | None] = mapped_column(String(255))

    posted_at: Mapped["MySQLDateTime | None"] = mapped_column(MySQLDateTime(fsp=6))
    deadline_at: Mapped["MySQLDateTime | None"] = mapped_column(MySQLDateTime(fsp=6), nullable=True)

    url: Mapped[str | None] = mapped_column(String(1024))
    hash: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped["MySQLDateTime"] = mapped_column(
        MySQLDateTime(fsp=6), server_default=func.now(), nullable=False
    )
    updated_at: Mapped["MySQLDateTime"] = mapped_column(
        MySQLDateTime(fsp=6),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # relationships (optional at this stage)
    company = relationship("Company", lazy="joined")
