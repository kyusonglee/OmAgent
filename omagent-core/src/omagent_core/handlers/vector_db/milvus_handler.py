# milvus_handler.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from pymilvus import (
    Collection,
    DataType,
    MilvusClient,
    CollectionSchema,
    FieldSchema,
)
from ..handler_base import VectorDBHandler
from ...utils.error import VQLError


class MilvusHandler(VectorDBHandler, BaseModel):
    host_url: str = "tcp://localhost:19530"
    user: str = ""
    password: str = ""
    db_name: str = "default"
    primary_field: str = "pk"
    vector_field: str = "vector"

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.milvus_client = MilvusClient(
            uri=self.host_url,
            user=self.user,
            password=self.password,
            db_name=self.db_name,
        )

    def create_index(self, index_id: str, mapping: Dict[str, Any]):
        if self.is_collection_in(index_id):
            self.drop_collection(index_id)
    
        fields = []
        for field_name, field_info in mapping.items():
            if field_info['type'] == 'vector':
                field = FieldSchema(
                    name=field_name,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=field_info['dim'],
                )
                self.vector_field = field_name
            elif field_info['type'] == 'keyword':
                field = FieldSchema(
                    name=field_name,
                    dtype=DataType.VARCHAR,
                    max_length=255,
                )
            else:
                continue
            fields.append(field)

        pk_field = FieldSchema(
            name=self.primary_field,
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=100,
        )
        fields.append(pk_field)

        schema = CollectionSchema(
            fields=fields,
            description=f"Collection for index {index_id}",
        )
        res = self.make_collection(index_id, schema)
        print (res)

    def delete_index(self, index_id: str):
        self.drop_collection(index_id)

    def add_vectors(self, index_id: str, vectors: List[Dict[str, Any]]) -> List[str]:
        if self.is_collection_in(index_id):
            res = self.milvus_client.insert(
                collection_name=index_id,
                data=vectors,
            )
            print (res)
            return res["ids"]
        else:
            raise VQLError(500, detail=f"{index_id} collection does not exist")

    def search_vectors(
        self,
        index_id: str,
        vector: List[float],
        top_k: int,
        threshold: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_collection_in(index_id):
            raise VQLError(500, detail=f"{index_id} collection does not exist")

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }

        filter_expr = ""
        if filters:
            filter_expr = " and ".join(f"{k} == '{v}'" for k, v in filters.items())

        hits = self.milvus_client.search(
            collection_name=index_id,
            data=[vector],
            anns_field=self.vector_field,
            param=search_params,
            limit=top_k,
            expr=filter_expr,
        )

        results = []
        for hit in hits[0]:
            results.append({
                '_source': {
                    '_uid': hit.id,
                },
                'score': hit.distance,
            })
        return results

    def delete_vectors(self, index_id: str, ids: List[str]):
        if self.is_collection_in(index_id):
            delete_expr = f"{self.primary_field} in {ids}]"
            print (delete_expr)
            res = self.milvus_client.delete(
                collection_name=index_id,
                filter=delete_expr,
            )
            print (res)
        else:
            raise VQLError(500, detail=f"{index_id} collection does not exist")

    # Additional methods
    def is_collection_in(self, collection_name):
        return self.milvus_client.has_collection(collection_name)

    def make_collection(self, collection_name, schema):
        self.milvus_client.create_collection(
            collection_name,
            schema=schema,
        )

    def drop_collection(self, collection_name):
        self.milvus_client.drop_collection(collection_name)

    def get_vectors_by_id(self, collection_name, ids, output_fields=['pk', 'vector']):
        return self.milvus_client.query(
            collection_name=collection_name,
            filter=f"pk in {ids}",
            output_fields=output_fields
        )
    
    def get_all_vectors(self, collection_name):
        return self.milvus_client.query(
            collection_name=collection_name,
            filter="",  # Empty expression to match all records
            limit=10,
            output_fields=['pk', 'bot_id', 'content']  # Add any other fields you want to retrieve
        )