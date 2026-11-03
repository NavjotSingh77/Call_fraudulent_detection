"""
FastAPI dependency injection providers.
"""

from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session
from src.db.session import get_db
from src.repositories.call_repository import CallRepository
from src.services.model_service import ModelService, get_model_service
from src.config.settings import get_settings

settings = get_settings()


def get_call_repo(db: Session = Depends(get_db)) -> CallRepository:
    """Provide a scoped CallRepository instance."""
    return CallRepository(db)


def get_model() -> ModelService:
    """Provide active ModelService instance."""
    return get_model_service(model_dir=settings.model_dir)
