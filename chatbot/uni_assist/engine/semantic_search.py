"""
Semantic Search Module
=======================
Implements embedding generation and similarity-based document retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from openai import OpenAI

from .segmentation import TextSegment


@dataclass(frozen=True)
class SearchResult:
    """Container for a retrieved segment with its relevance score."""
    segment: TextSegment
    relevance_score: float


def _apply_l2_normalization(vector: np.ndarray) -> np.ndarray:
    """Normalize vector for cosine similarity using L2 norm."""
    faiss.normalize_L2(vector)
    return vector


def generate_query_embedding(api_client: OpenAI, *, model_name: str, query_text: str) -> np.ndarray:
    """
    Create a vector embedding for the given query using OpenAI's API.
    
    Parameters:
        api_client: Configured OpenAI client instance
        model_name: Name of the embedding model to use
        query_text: The search query to embed
        
    Returns:
        Normalized embedding as a numpy array
        
    Raises:
        ValueError: If the query is empty
    """
    if not query_text or not query_text.strip():
        raise ValueError("Search query must not be empty")
    
    cleaned_query = query_text.strip()
    # Truncate if exceeds model's context limit
    if len(cleaned_query) > 8000:
        cleaned_query = cleaned_query[:8000]
    
    api_response = api_client.embeddings.create(model=model_name, input=[cleaned_query])
    embedding_vector = np.array([api_response.data[0].embedding], dtype=np.float32)
    return _apply_l2_normalization(embedding_vector)


def find_relevant_segments(
    *,
    vector_index: faiss.Index,
    all_segments: list[TextSegment],
    query_embedding: np.ndarray,
    num_results: int,
) -> list[SearchResult]:
    """
    Search for the most relevant segments using vector similarity.
    
    Applies multiple filtering strategies:
    - Minimum relevance threshold to exclude weak matches
    - Source diversity to avoid over-representation from single documents
    """
    # Over-fetch candidates for better final selection
    candidate_count = min(num_results * 3, len(all_segments))
    similarity_scores, matched_indices = vector_index.search(query_embedding, candidate_count)
    
    candidate_results: list[SearchResult] = []
    for score, idx in zip(similarity_scores[0].tolist(), matched_indices[0].tolist()):
        if idx < 0 or idx >= len(all_segments):
            continue
        candidate_results.append(SearchResult(segment=all_segments[idx], relevance_score=float(score)))
    
    # Apply relevance threshold (cosine similarity minimum)
    RELEVANCE_THRESHOLD = 0.3
    candidate_results = [r for r in candidate_results if r.relevance_score >= RELEVANCE_THRESHOLD]
    
    # Ensure diversity across source documents
    source_frequency: dict[str, int] = {}
    MAX_FROM_SINGLE_SOURCE = 3
    diversified_results: list[SearchResult] = []
    
    for result in candidate_results:
        source_id = result.segment.doc_identifier
        current_count = source_frequency.get(source_id, 0)
        if current_count < MAX_FROM_SINGLE_SOURCE:
            diversified_results.append(result)
            source_frequency[source_id] = current_count + 1
        
        if len(diversified_results) >= num_results:
            break
    
    return diversified_results
