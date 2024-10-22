# stm_base.py

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict


class STMBase(ABC):
    @abstractmethod
    def set(self, key: str, value: Any, expire: Optional[int] = None):
        """
        Set a key-value pair in STM with an optional expiration time.

        Args:
            key (str): The key to store the value under.
            value (Any): The value to store.
            expire (Optional[int]): Expiration time in seconds.
        """
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from STM by its key.

        Args:
            key (str): The key of the value to retrieve.

        Returns:
            Optional[Any]: The value if found, else None.
        """
        pass

    @abstractmethod
    def delete(self, key: str):
        """
        Delete a key-value pair from STM.

        Args:
            key (str): The key to delete.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear all key-value pairs from STM.
        """
        pass

    @abstractmethod
    def keys(self, pattern: str = "*") -> list:
        """
        Retrieve a list of keys matching a pattern.

        Args:
            pattern (str): The pattern to match keys.

        Returns:
            list: A list of matching keys.
        """
        pass
