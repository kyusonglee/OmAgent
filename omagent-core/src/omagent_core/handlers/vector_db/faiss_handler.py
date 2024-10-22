# faiss_handler.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import faiss
import numpy as np
from handler_base import VectorDBHandler
from utils.error import VQLError


class FaissHandler(VectorDBHandler, BaseModel):
    index: Optional[faiss.IndexIDMap] = None
    dim: int
    id_map: Dict[int, str] = {}  # Mapping from FAISS internal IDs to external IDs

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def create_index(self, index_id: str, mapping: Dict[str, Any]):
        self.dim = mapping['vector']['dim']
        # Create a flat L2 index with ID mapping
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
        self.id_map = {}
        print(f"FAISS index created with dimension {self.dim}")

    def delete_index(self, index_id: str):
        self.index = None
        self.id_map = {}
        print(f"FAISS index {index_id} deleted")

    def add_vectors(self, index_id: str, vectors: List[Dict[str, Any]]) -> List[str]:
        if self.index is None:
            raise VQLError(500, detail="Index has not been created")

        vector_list = [np.array(v['vector'], dtype='float32') for v in vectors]
        ids = [int(v['_uid']) for v in vectors]  # Ensure '_uid' can be converted to int
        self.index.add_with_ids(np.vstack(vector_list), np.array(ids))
        # Update the id_map
        for v in vectors:
            self.id_map[int(v['_uid'])] = v['_uid']
        print(f"Added {len(vectors)} vectors to FAISS index")
        return [v['_uid'] for v in vectors]

    def search_vectors(
        self,
        index_id: str,
        vector: List[float],
        top_k: int,
        threshold: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.index is None:
            raise VQLError(500, detail="Index has not been created")

        query_vector = np.array([vector], dtype='float32')
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            # Convert L2 distance to similarity score (e.g., using negative distance)
            score = -dist
            if score < threshold:
                continue
            external_id = self.id_map.get(idx)
            if external_id is None:
                continue
            results.append({
                '_source': {
                    '_uid': external_id,
                },
                'score': score,
            })
        print(f"Found {len(results)} results in FAISS search")
        return results

    def delete_vectors(self, index_id: str, ids: List[str]):
        if self.index is None:
            raise VQLError(500, detail="Index has not been created")
        # FAISS does not support deleting vectors directly
        # This is a placeholder to comply with the interface
        raise NotImplementedError("FAISS does not support vector deletion directly")

