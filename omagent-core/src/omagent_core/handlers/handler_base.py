# handler_base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


from sqlmodel import Column, DateTime, Field, SQLModel, func

class VectorDBHandler(ABC):
    @abstractmethod
    def create_index(self, index_id: str, mapping: Dict[str, Any]):
        pass

    @abstractmethod
    def delete_index(self, index_id: str):
        pass

    @abstractmethod
    def add_vectors(self, index_id: str, vectors: List[Dict[str, Any]]) -> List[str]:
        pass

    @abstractmethod
    def search_vectors(
        self,
        index_id: str,
        vector: List[float],
        top_k: int,
        threshold: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_vectors(self, index_id: str, ids: List[str]):
        pass


class DataHandler(ABC):
    @abstractmethod
    def execute_sql(self, sql_query: str):
        pass

    @abstractmethod
    def simple_get(self, bot_id: str, number: int = None) -> List[Any]:
        pass

    @abstractmethod
    def simple_add(self, data: List[Union[Any, Dict]]) -> List[str]:
        pass

    @abstractmethod
    def simple_update(self, id: int, key: str, value: Any):
        pass

    @abstractmethod
    def simple_delete(self, id: int):
        pass

    @abstractmethod
    def init_table(self):
        pass

    @abstractmethod
    def get_by_id(self, bot_id: str, id: int) -> Optional[Any]:
        pass


class BaseTable(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    bot_id: str = Field(index=True, nullable=False)

    create_time: Optional[datetime] = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )

    update_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
        )
    )

    deleted: int = 0
