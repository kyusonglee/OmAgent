# in_memory_stm.py

from typing import Any, Optional, Dict
from pydantic import BaseModel
from threading import Lock
from .base import STMBase
from ...utils.error import VQLError
import time


class InMemorySTM(STMBase):
    def __init__(self):
        self._storage: Dict[str, Any] = {}
        self._expiry_times: Dict[str, float] = {}
        self._lock = Lock()

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True

    def set(self, key: str, value: Any, expire: Optional[int] = None):
        with self._lock:
            self._storage[key] = value
            if expire is not None:
                self._expiry_times[key] = time.time() + expire
            elif key in self._expiry_times:
                del self._expiry_times[key]

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._expiry_times:
                if time.time() > self._expiry_times[key]:
                    # Key has expired
                    del self._storage[key]
                    del self._expiry_times[key]
                    return None
            return self._storage.get(key)

    def delete(self, key: str):
        with self._lock:
            self._storage.pop(key, None)
            self._expiry_times.pop(key, None)

    def clear(self):
        with self._lock:
            self._storage.clear()
            self._expiry_times.clear()

    def keys(self, pattern: str = "*") -> list:
        with self._lock:
            # Simple pattern matching using fnmatch
            import fnmatch
            return [key for key in self._storage.keys() if fnmatch.fnmatch(key, pattern)]
