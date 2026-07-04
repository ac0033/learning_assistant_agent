"""Result reranking for improved retrieval precision.

Uses LLM-based reranking to select the most relevant chunks
from the candidate set produced by hybrid retrieval.
"""

import logging
from typing import List

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


class Reranker:
    """Rerank retrieved chunks using LLM relevance scoring.

    Strategy: LLM evaluates each candidate chunk against the query,
    keeping only the most relevant ones with scores.
    """

    def rerank(
        self,
        query: str,
        candidates: List[NodeWithScore],
        top_k: int = 5,
    ) -> List[NodeWithScore]:
        """Rerank candidates, keeping the top-k most relevant.

        For MVP: simple heuristic reranking based on score + content signals.
        Production path: Cohere Rerank API or cross-encoder model.

        Args:
            query: The original user query.
            candidates: List of candidate nodes with preliminary scores.
            top_k: Number of final results to return.

        Returns:
            Reranked list, limited to top_k.
        """
        if len(candidates) <= top_k:
            return candidates

        # Heuristic reranking boosts:
        # 1. Query terms appearing in content (exact match bonus)
        # 2. Title/section headers match
        # 3. LaTeX formula match (for math queries)

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for item in candidates:
            content = item.node.get_content().lower()
            boost = 0.0

            # Exact term match boost
            matching_terms = sum(1 for term in query_terms if term in content)
            boost += matching_terms * 0.05

            # Math/LaTeX match boost
            if "$" in query and "$" in content:
                boost += 0.1

            # Source title match (e.g., "Lecture 1: Introduction")
            source = item.node.metadata.get("source", "").lower()
            if any(term in source for term in query_terms):
                boost += 0.1

            item.score = (item.score or 0.0) + boost

        # Sort by boosted score, return top_k
        candidates.sort(key=lambda x: x.score or 0.0, reverse=True)
        return candidates[:top_k]
