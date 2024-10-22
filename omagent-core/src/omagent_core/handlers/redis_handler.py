# redis_handler.py

from typing import Any, Optional, Dict
from pydantic import BaseModel
import redis

from ..utils.error import VQLError


class RedisHandler(BaseModel):
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    redis_client: Optional[redis.Redis] = None

    class Config:
        extra = 'allow'
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True  # Decode bytes to strings
            )
            # Test the connection
            self.redis_client.ping()
        except redis.RedisError as e:
            raise VQLError(500, detail=f'Redis connection error: {str(e)}')

    def set(self, key: str, value: Any, expire: Optional[int] = None):
        try:
            self.redis_client.set(name=key, value=value, ex=expire)
        except redis.RedisError as e:
            raise VQLError(500, detail=f'Redis set error: {str(e)}')

    def get(self, key: str) -> Optional[Any]:
        try:
            return self.redis_client.get(name=key)
        except redis.RedisError as e:
            raise VQLError(500, detail=f'Redis get error: {str(e)}')

    def delete(self, key: str):
        try:
            self.redis_client.delete(key)
        except redis.RedisError as e:
            raise VQLError(500, detail=f'Redis delete error: {str(e)}')

    def clear(self):
        try:
            self.redis_client.flushdb()
        except redis.RedisError as e:
            raise VQLError(500, detail=f'Redis clear error: {str(e)}')

    def keys(self, pattern: str = "*") -> list:
        try:
            return self.redis_client.keys(pattern)
        except redis.RedisError as e:
            raise VQLError(500, detail=f'Redis keys error: {str(e)}')
