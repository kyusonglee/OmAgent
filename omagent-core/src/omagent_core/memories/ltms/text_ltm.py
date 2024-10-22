# text_ltm.py

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from .text_ltm_base import TextLTMBase
from .handler_base import DataHandler
from .utils.error import VQLError
from .utils.logger import logging


class TextLTM(TextLTMBase, BaseModel):
    data_handler: Optional[DataHandler] = None
    table: Optional[Any] = None  # Should be a SQLModel subclass
    engine: Optional[Any] = None  # SQLAlchemy engine

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def handler_register(self, name: str, handler):
        setattr(self, name, handler)
        if isinstance(handler, DataHandler):
            self.data_handler = handler

    def add(
        self,
        data: List[Union[Any, Dict]],
    ):
        if not self.data_handler:
            raise VQLError(500, detail='DataHandler must be registered before adding data')

        data_instances = [
            item if hasattr(item, "__table__") else self.table(**item) for item in data
        ]

        ids = self.data_handler.simple_add(data_instances)
        return ids

    def match(
        self,
        query_data: str,
        bot_id: str,
        size: int = 1,
    ) -> List[Dict[str, Any]]:
        if not self.data_handler:
            raise VQLError(500, detail='DataHandler must be registered before matching data')

        # Implement text search logic using SQL full-text search or LIKE queries
        results = self.data_handler.text_search(
            query=query_data,
            bot_id=bot_id,
            limit=size,
        )

        output = []
        for result in results:
            output.append({"data": result})
        return output

    def delete_data(self, ids: List[str]):
        if not self.data_handler:
            raise VQLError(500, detail='DataHandler must be registered before deleting data')
        for id in ids:
            self.data_handler.simple_delete(id)

    def init_knowledge(self) -> int:
        if not self.data_handler:
            raise VQLError(500, detail='DataHandler must be registered before initializing knowledge')
        return self.data_handler.init_table()
