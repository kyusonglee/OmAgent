# ltm_base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class LTMBase(ABC):
    @abstractmethod
    def handler_register(self, name: str, handler):
        pass

    @abstractmethod
    def add(
        self,
        data: List[Union[Any, Dict]],
    ):
        pass

    @abstractmethod
    def match(
        self,
        query_data: str,
        bot_id: str,
        size: int = 1,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_data(self, ids: List[str]):
        pass

    @abstractmethod
    def init_knowledge(self) -> int:
        pass
