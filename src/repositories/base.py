"""
Generic abstract repository interface.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar
from abc import ABC, abstractmethod

T = TypeVar("T")


class IRepository(ABC, Generic[T]):
    """Abstract interface defining standard CRUD operations."""

    @abstractmethod
    def get(self, id: Any) -> Optional[T]:
        pass

    @abstractmethod
    def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        pass

    @abstractmethod
    def create(self, entity: T) -> T:
        pass

    @abstractmethod
    def update(self, entity: T) -> T:
        pass

    @abstractmethod
    def delete(self, id: Any) -> bool:
        pass
