"""Document Retrieval Node.

Retrieves relevant chunks from the vector store using hybrid search
(dense + BM25), then reranks for precision.
"""

import logging
from typing import Any

from ..state import TeachingState, RetrievedContext
from ...indexing.hybrid_retriever import HybridRetriever, set_retriever
from ...indexing.vector_store import VectorStoreManager
from ...indexing.embeddings import get_embed_model

logger = logging.getLogger(__name__)

# Module-level retriever singleton (rebuilt when documents change)
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Get or create the hybrid retriever singleton.

    Uses set_retriever() (public API) so that refresh_retriever()
    (called after file upload) can find and rebuild BM25.
    """
    global _retriever
    if _retriever is None:
        vsm = VectorStoreManager()
        embed_model = get_embed_model()
        _retriever = HybridRetriever(
            vector_store_manager=vsm,
            embed_model=embed_model,
        )
        set_retriever(_retriever)
    return _retriever


async def document_retrieval_node(state: TeachingState) -> dict[str, Any]:
    """Search for relevant document chunks.

    Uses the target_topic from query understanding to retrieve
    the most relevant sections from uploaded course materials.

    Returns updated state with retrieved_chunks.
    """
    target_topic = state.get("target_topic", "")
    user_query = state.get("user_query", "")

    # Build effective retrieval query
    retrieval_query = f"{target_topic} {user_query}"

    logger.info("[DocumentRetrieval] Retrieving for: %s", retrieval_query[:120])

    retriever = _get_retriever()

    # If no documents indexed, return empty
    if not retriever._bm25.is_initialized():
        logger.warning("[DocumentRetrieval] No documents indexed — returning empty")
        return {
            "retrieved_chunks": [],
            "retrieval_query": retrieval_query,
            "current_node": "document_retrieval",
        }

    # Execute hybrid retrieval
    try:
        results = retriever.retrieve(retrieval_query)
    except Exception as e:
        logger.error("[DocumentRetrieval] Retrieval failed: %s", e)
        return {
            "retrieved_chunks": [],
            "retrieval_query": retrieval_query,
            "current_node": "document_retrieval",
        }

    # Convert to RetrievedContext format
    chunks: list[RetrievedContext] = []
    for r in results:
        meta = r.node.metadata
        chunks.append(RetrievedContext(
            content=r.node.get_content(),
            source=meta.get("source", "unknown"),
            page=meta.get("page", -1),
            relevance_score=round(r.score or 0.0, 4),
        ))

    logger.info(
        "[DocumentRetrieval] Found %d relevant chunks (top score=%.3f)",
        len(chunks), chunks[0]["relevance_score"] if chunks else 0.0
    )

    return {
        "retrieved_chunks": chunks,
        "retrieval_query": retrieval_query,
        "current_node": "document_retrieval",
    }
