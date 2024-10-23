# vector_ltm.py

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from .base import LTMBase
from ...handlers.handler_base import VectorDBHandler
from ...models.encoders.base import EncoderBase  # Assuming this exists
from ...utils.error import VQLError
from ...utils.logger import logging


class VectorLTM(LTMBase):
    def __init__(self, index_id: str):
        self.encoders: Dict[str, EncoderBase] = {}
        self.vector_db_handler: Optional[VectorDBHandler] = None
        self.dim: Optional[int] = None
        self.index_id: str = index_id

    def handler_register(self, name: str, handler):
        setattr(self, name, handler)
        if isinstance(handler, VectorDBHandler):
            self.vector_db_handler = handler

    def encoder_register(self, modality: str, encoder: EncoderBase):
        self.encoders[modality] = encoder
        if self.dim is None:
            self.dim = encoder.dim
        elif self.dim != encoder.dim:
            raise VQLError(500, detail="All encoders must have the same dimension")
 
    def add(
        self,
        data: List[Dict[str, Any]],
        encode_data: List[Any],
        modality: str,
        src_type: Optional[str] = None,
    ):
        """
        Add data to the knowledge base with vector representation for searching.

        Args:
            data (List[Dict[str, Any]]): Data to add to the knowledge base. Each item should be a dictionary containing metadata.
            encode_data (List[Any]): Data used for encoding (e.g., file paths, raw data).
            modality (str): The modality of the data (e.g., 'image', 'audio').
            src_type (Optional[str]): The source type of the encode_data if needed (e.g., 'file', 'base64').
        """
        encoder = self.encoders.get(modality)
        if not encoder:
            raise VQLError(500, detail=f'Missing encoder for modality "{modality}"')
        if not self.vector_db_handler:
            raise VQLError(500, detail='VectorDBHandler must be registered before adding data')

        # Encode the data using the specified encoder
        vectors = encoder.infer(encode_data, src_type=src_type)
        if len(vectors) != len(data):
            raise VQLError(500, detail='Mismatch between data and encoded vectors')

        # Prepare records for the vector database
        vector_records = [
            {**item, "vector": vector} for vector, item in zip(vectors, data)
        ]

        # Add vectors and metadata to the vector database        
        print (self.index_id)
        res = self.vector_db_handler.add_vectors(self.index_id, vector_records)
        print ("Add Done")
        return res
        

    def get_all_data(self, index_id):
        return self.vector_db_handler.get_all_vectors(self.index_id)

    def match(
        self,
        query_data: Any,
        modality: str,
        filters: Dict[str, Any] = {},
        threshold: float = 0.0,
        size: int = 1,
        src_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Match query data against the knowledge base.

        Args:
            query_data (Any): The query data to match (e.g., file path, raw data).
            modality (str): The modality of the query data.
            filters (Dict[str, Any]): Filters to apply during the search (e.g., {'bot_id': 'bot123'}).
            threshold (float): The similarity threshold.
            size (int): The number of results to return.
            src_type (Optional[str]): The source type of the query data if needed.

        Returns:
            List[Dict[str, Any]]: A list of matched data with their confidence scores and metadata.
        """
        encoder = self.encoders.get(modality)
        if not encoder:
            raise VQLError(500, detail=f'Missing encoder for modality "{modality}"')
        if not self.vector_db_handler:
            raise VQLError(500, detail='VectorDBHandler must be registered before matching data')

        # Encode the query data
        vector = encoder.infer([query_data], src_type=src_type)[0]

        # Perform vector search in the vector database
        match_results = self.vector_db_handler.search_vectors(
            index_id=self.index_id,
            vector=vector,
            top_k=size,
            threshold=threshold,
            filters=filters,
        )

        return match_results

    def delete_data(self, ids: List[str]):
        """
        Delete data from the knowledge base.

        Args:
            ids (List[str]): A list of IDs corresponding to the data to delete.
        """
        if not self.vector_db_handler:
            raise VQLError(500, detail='VectorDBHandler must be registered before deleting data')
        self.vector_db_handler.delete_vectors(self.index_id, ids)

    def init_knowledge(self, mapping=None) -> None:
        """
        Initialize the knowledge base by deleting existing data and recreating the index.
        """
        if not self.vector_db_handler:
            raise VQLError(500, detail='VectorDBHandler must be registered before initializing knowledge')

        # Delete existing index
        self.vector_db_handler.delete_index(self.index_id)

        # Create a new index with the required mapping
        if not mapping:
            mapping = {
                'vector': {'type': 'vector', 'metric_type': 'L2', 'dim': self.dim},
                # Include other fields as needed
            }
        print ("create_index",self.index_id,mapping)
        res = self.vector_db_handler.create_index(self.index_id, mapping)
        print (res)
