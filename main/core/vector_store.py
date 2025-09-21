# core/vector_store.py - Vector store functionality for document embeddings
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Represents a vector search result"""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class VectorStore:
    """Base class for vector storage and retrieval"""

    def __init__(self):
        self.dimension = None
        self.vectors = {}
        self.metadata = {}

    def add_vectors(self, vectors: List[np.ndarray], texts: List[str],
                    metadata: List[Dict[str, Any]], ids: List[str] = None) -> bool:
        """Add vectors to the store"""
        try:
            if not ids:
                ids = [f"vec_{i}" for i in range(len(vectors))]

            for i, (vector, text, meta) in enumerate(zip(vectors, texts, metadata)):
                vector_id = ids[i]
                self.vectors[vector_id] = vector
                self.metadata[vector_id] = {
                    "text": text,
                    **meta
                }

            if self.dimension is None and vectors:
                self.dimension = len(vectors[0])

            return True

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False

    def search(self, query_vector: np.ndarray, top_k: int = 10,
               score_threshold: float = 0.0) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        try:
            if not self.vectors:
                return []

            results = []

            for vector_id, stored_vector in self.vectors.items():
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vector, stored_vector)

                if similarity >= score_threshold:
                    meta = self.metadata.get(vector_id, {})
                    result = VectorSearchResult(
                        id=vector_id,
                        score=similarity,
                        text=meta.get("text", ""),
                        metadata=meta
                    )
                    results.append(result)

            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.score, reverse=True)

            return results[:top_k]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def delete(self, vector_id: str) -> bool:
        """Delete a vector by ID"""
        try:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
                del self.metadata[vector_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all vectors"""
        try:
            self.vectors.clear()
            self.metadata.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to clear vectors: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            "total_vectors": len(self.vectors),
            "dimension": self.dimension,
            "memory_usage_mb": self._estimate_memory_usage()
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)

            # Calculate dot product (cosine similarity for normalized vectors)
            similarity = np.dot(vec1_norm, vec2_norm)

            return float(similarity)

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if not self.vectors:
            return 0.0

        # Rough estimation: each float64 = 8 bytes
        total_elements = sum(len(v) for v in self.vectors.values())
        bytes_used = total_elements * 8  # float64

        return bytes_used / (1024 * 1024)  # Convert to MB


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store implementation"""

    def __init__(self):
        super().__init__()
        logger.info("Initialized in-memory vector store")

    def persist(self, filepath: str) -> bool:
        """Save vectors to file (placeholder for future implementation)"""
        logger.warning("Persistence not implemented for in-memory store")
        return False

    def load(self, filepath: str) -> bool:
        """Load vectors from file (placeholder for future implementation)"""
        logger.warning("Loading not implemented for in-memory store")
        return False
