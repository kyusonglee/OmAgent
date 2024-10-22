# sql_data_handler.py

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy_utils import create_database, database_exists
from sqlmodel import Session, SQLModel, create_engine, delete, select
from .handler_base import DataHandler
from ..utils.error import VQLError
from ..utils.logger import logging


class SQLDataHandler(DataHandler, BaseModel):
    db: str
    user: str
    passwd: str
    host: str = "localhost"
    port: str = 3306
    table: Any
    DELETED: int = 1
    NO_DELETED: int = 0
    engine: Any = None

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.alchemy_uri = f"mysql+pymysql://{self.user}:{self.passwd}@{self.host}:{self.port}/{self.db}"
        self.engine = create_engine(
            self.alchemy_uri,
            pool_pre_ping=True,
            pool_size=50,
            max_overflow=4,
            pool_recycle=7200,
            pool_timeout=600,
            echo=True,
            pool_use_lifo=True,
        )
        if not database_exists(self.engine.url):
            create_database(self.engine.url)
        else:
            self.engine.connect()
        SQLModel.metadata.create_all(bind=self.engine)

    def execute_sql(self, sql_query: str):
        matches = []
        with self.engine.connect() as connection:
            result = connection.execute(text(sql_query))
            for row in result:
                matches.append(row._asdict())
        return matches

    def simple_get(self, bot_id: str, number: int = None) -> List[Any]:
        with Session(self.engine) as session:
            statement = (
                select(self.table)
                .where(self.table.bot_id == bot_id)
                .where(self.table.deleted == self.NO_DELETED)
                .order_by(self.table.id.desc())
            )
            if number:
                statement = statement.limit(number)
            query_results = session.exec(statement).all()
        return query_results or []

    def simple_add(self, data: List[Union[Any, Dict]]) -> List[str]:
        num = 0
        inserted = []
        with Session(self.engine) as session:
            for item in data:
                if isinstance(item, dict):
                    item = self.table(**item)
                session.add(item)
                num += 1
                inserted.append(item)
            session.commit()
            logging.debug(f"{num} data is added to table [{self.table.__tablename__}]")
            ids = [item.id for item in inserted]
        return ids

    def simple_update(self, id: int, key: str, value: Any):
        with Session(self.engine) as session:
            statement = select(self.table).where(self.table.id == id)
            query_result = session.exec(statement).one_or_none()
            if not query_result:
                raise VQLError(
                    500,
                    detail=f"Trying to update non-existent data [{id}] in table [{self.table.__tablename__}]",
                )
            elif query_result.deleted == self.DELETED:
                raise VQLError(
                    500,
                    detail=f"Trying to update deleted data [{id}] in table [{self.table.__tablename__}]",
                )
            setattr(query_result, key, value)
            session.add(query_result)
            session.commit()
            logging.debug(
                f"Key [{key}] in id [{id}] is updated to [{value}] in table [{self.table.__tablename__}]"
            )

    def simple_delete(self, id: int):
        self.simple_update(id=id, key="deleted", value=self.DELETED)
        logging.debug(f"Id [{id}] is deleted in table [{self.table.__tablename__}]")

    def init_table(self):
        with Session(self.engine) as session:
            result = session.exec(delete(self.table))
            session.commit()
            del_row_num = result.rowcount
        return del_row_num

    def get_by_id(self, bot_id: str, id: int) -> Optional[Any]:
        with Session(self.engine) as session:
            statement = (
                select(self.table)
                .where(self.table.bot_id == bot_id)
                .where(self.table.deleted == self.NO_DELETED)
                .where(self.table.id == id)
            )
            query_res = session.exec(statement).one_or_none()
        return query_res

    def text_search(self, query: str, bot_id: str, limit: int) -> List[Any]:
        with Session(self.engine) as session:
            # Implement full-text search or simple LIKE query
            # Assuming there is a 'content' field in the table to search
            statement = (
                select(self.table)
                .where(self.table.bot_id == bot_id)
                .where(self.table.deleted == self.NO_DELETED)
                .where(self.table.content.ilike(f"%{query}%"))
                .limit(limit)
            )
            query_results = session.exec(statement).all()
        return query_results or []
