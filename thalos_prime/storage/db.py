from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session

from thalos_prime.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


class RawDocument(Base):
    __tablename__ = "raw_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(32), default="web")
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    robots_allowed: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(32), default="raw")

    normalized_docs: Mapped[list["NormalizedDocument"]] = relationship(
        "NormalizedDocument", back_populates="raw_doc", cascade="all, delete-orphan"
    )


class NormalizedDocument(Base):
    __tablename__ = "normalized_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    raw_doc_id: Mapped[str] = mapped_column(String(36), ForeignKey("raw_documents.id"), index=True)
    language: Mapped[str] = mapped_column(String(16), default="en")
    original_text: Mapped[str] = mapped_column(Text)
    translated_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    normalized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    word_count: Mapped[int] = mapped_column(default=0)

    raw_doc: Mapped["RawDocument"] = relationship("RawDocument", back_populates="normalized_docs")
    claims: Mapped[list["Claim"]] = relationship(
        "Claim", back_populates="norm_doc", cascade="all, delete-orphan"
    )


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    norm_doc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("normalized_documents.id"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    coordinate: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    language: Mapped[str] = mapped_column(String(16), default="en")
    entities_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    norm_doc: Mapped["NormalizedDocument"] = relationship(
        "NormalizedDocument", back_populates="claims"
    )
    evidence_links: Mapped[list["EvidenceLink"]] = relationship(
        "EvidenceLink",
        back_populates="claim",
        cascade="all, delete-orphan",
        foreign_keys="EvidenceLink.claim_id",
    )
    verification_record: Mapped[Optional["VerificationRecord"]] = relationship(
        "VerificationRecord", back_populates="claim", cascade="all, delete-orphan", uselist=False
    )


class EvidenceLink(Base):
    __tablename__ = "evidence_links"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_id: Mapped[str] = mapped_column(String(36), ForeignKey("claims.id"), index=True)
    evidence_text: Mapped[str] = mapped_column(Text)
    evidence_doc_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("raw_documents.id"), nullable=True
    )
    bm25_score: Mapped[float] = mapped_column(Float, default=0.0)
    semantic_score: Mapped[float] = mapped_column(Float, default=0.0)
    link_type: Mapped[str] = mapped_column(String(16), default="neutral")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    claim: Mapped["Claim"] = relationship(
        "Claim", back_populates="evidence_links", foreign_keys=[claim_id]
    )
    evidence_doc: Mapped[Optional["RawDocument"]] = relationship("RawDocument")


class ContradictionPair(Base):
    __tablename__ = "contradiction_pairs"
    __table_args__ = (UniqueConstraint("claim_a_id", "claim_b_id", name="uq_contradiction_pair"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_a_id: Mapped[str] = mapped_column(String(36), ForeignKey("claims.id"), index=True)
    claim_b_id: Mapped[str] = mapped_column(String(36), ForeignKey("claims.id"), index=True)
    nli_score: Mapped[float] = mapped_column(Float, default=0.0)
    label: Mapped[str] = mapped_column(String(16), default="neutral")
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    claim_a: Mapped["Claim"] = relationship("Claim", foreign_keys=[claim_a_id])
    claim_b: Mapped["Claim"] = relationship("Claim", foreign_keys=[claim_b_id])


class VerificationRecord(Base):
    __tablename__ = "verification_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("claims.id"), unique=True, index=True
    )
    composite_score: Mapped[float] = mapped_column(Float, default=0.0)
    signals_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    promoted_to: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    claim: Mapped["Claim"] = relationship("Claim", back_populates="verification_record")


class Feed(Base):
    __tablename__ = "feeds"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    url: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    feed_type: Mapped[str] = mapped_column(String(32), default="rss")
    last_fetched: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True)


_engines: dict[str, object] = {}


def get_engine(database_url: Optional[str] = None):
    settings = get_settings()
    url = database_url or settings.database_url
    if url not in _engines:
        kwargs = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}

        engine = create_engine(url, **kwargs)

        if url.startswith("sqlite"):
            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        _engines[url] = engine
    return _engines[url]


@contextmanager
def get_session(database_url: Optional[str] = None) -> Generator[Session, None, None]:
    engine = get_engine(database_url)
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def init_db(database_url: Optional[str] = None) -> None:
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
