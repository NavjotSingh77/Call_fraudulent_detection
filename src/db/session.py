"""
Database engine and session management.

Supports SQLite by default and PostgreSQL (or other SQL backends) via DATABASE_URL
without altering any business or ML logic.
"""

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

db_url = settings.database_url
connect_args = {}
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables."""
    from src.db.base import Base
    import src.db.models  # Ensure models are imported
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized at: %s", db_url)
    except Exception as exc:
        logger.error("Failed to initialize database: %s", exc)


def get_db() -> Generator[Session, None, None]:
    """Dependency injection helper to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
