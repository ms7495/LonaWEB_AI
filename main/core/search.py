# core/search.py - Search functionality for document retrieval
import logging
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result"""
    text: str
    score: float
    metadata: Dict
    chunk_id: str


class SearchEngine:
    """Handles document search and retrieval"""

    def __init__(self, vector_store=None, embeddings=None):
        self.vector_store = vector_store
        self.embeddings = embeddings

    def semantic_search(self, query: str, top_k: int = 10, score_threshold: float = 0.0) -> List[SearchResult]:
        """Perform semantic search using embeddings"""
        if not self.vector_store or not self.embeddings:
            logger.warning("Vector store or embeddings not initialized")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embeddings.encode_single(query)

            # Search in vector store
            results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold
            )

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_result = SearchResult(
                    text=result.get('text', ''),
                    score=result.get('score', 0.0),
                    metadata=result.get('metadata', {}),
                    chunk_id=result.get('id', '')
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def keyword_search(self, query: str, documents: List[Dict], top_k: int = 10) -> List[SearchResult]:
        """Perform keyword-based search"""
        try:
            query_terms = query.lower().split()
            scored_docs = []

            for doc in documents:
                text = doc.get('text', '').lower()
                score = 0

                # Simple TF-based scoring
                for term in query_terms:
                    score += text.count(term)

                if score > 0:
                    scored_docs.append((doc, score))

            # Sort by score and take top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            results = []
            for doc, score in scored_docs[:top_k]:
                search_result = SearchResult(
                    text=doc.get('text', ''),
                    score=float(score),
                    metadata=doc.get('metadata', {}),
                    chunk_id=doc.get('id', '')
                )
                results.append(search_result)

            return results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 10, semantic_weight: float = 0.7) -> List[SearchResult]:
        """Combine semantic and keyword search"""
        try:
            # Get results from both methods
            semantic_results = self.semantic_search(query, top_k * 2)
            # Note: keyword_search needs document list - this is a simplified version

            # For now, just return semantic results
            # In a full implementation, you'd combine and re-rank the results
            return semantic_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def filter_results_by_source(self, results: List[SearchResult], source_filter: str) -> List[SearchResult]:
        """Filter search results by document source"""
        filtered_results = []

        for result in results:
            filename = result.metadata.get('filename', '').lower()
            if source_filter.lower() in filename:
                filtered_results.append(result)

        return filtered_results

    def rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Re-rank search results based on additional criteria"""

        # Simple re-ranking based on text length and score
        def rerank_score(result: SearchResult) -> float:
            base_score = result.score
            text_length_penalty = len(result.text) / 1000  # Prefer moderate length texts
            return base_score - text_length_penalty * 0.1

        # Sort by new score
        reranked = sorted(results, key=rerank_score, reverse=True)

        # Update scores
        for i, result in enumerate(reranked):
            result.score = rerank_score(result)

        return reranked
