"""Document Retrieval Node.

Retrieves relevant chunks from the vector store using hybrid search
(dense + BM25), then reranks for precision.
"""

import logging
import re
from typing import Any, Optional

from ..state import TeachingState, RetrievedContext
from ...indexing.hybrid_retriever import HybridRetriever, set_retriever
from ...indexing.vector_store import VectorStoreManager
from ...indexing.embeddings import get_embed_model

logger = logging.getLogger(__name__)

# Default number of chunks shown to the teaching LLM (defined in settings, but
# read once at import time since changing it mid-session is uncommon).
try:
    from config.settings import settings as _settings
    settings_retrieval_top_k = _settings.retrieval_top_k
except Exception:  # pragma: no cover - defensive import
    settings_retrieval_top_k = 5

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


def _detect_referenced_source(user_query: str, known_sources: list[str]) -> Optional[str]:
    """Return the source filename the student explicitly named, if any.

    Case-insensitive substring match against the list of indexed files.
    Returns the canonical (indexed) filename so the Chroma ``where`` filter
    matches exactly, or None if no source is referenced.
    """
    q = user_query.lower()
    for src in known_sources:
        if src and src.lower() in q:
            return src
    return None


async def document_retrieval_node(state: TeachingState) -> dict[str, Any]:
    """Search for relevant document chunks.

    Uses the target_topic from query understanding to retrieve
    the most relevant sections from uploaded course materials.

    Per-conversation isolation: retrieval is restricted to the files the
    student actually uploaded in the *current* conversation (looked up via
    ``history_manager.get_thread_files``). A fresh conversation with no
    uploads yields an empty result so the agent falls back to direct LLM
    response — chunks from other conversations are never mixed in. This is
    the core fix for cross-conversation contamination.

    If the student explicitly names a file (e.g. "讲讲 xxx.pdf 第2部分")
    that is among the current conversation's files, retrieval is further
    restricted to that single source.

    Returns updated state with retrieved_chunks.
    """
    target_topic = state.get("target_topic", "")
    user_query = state.get("user_query", "")
    thread_id = state.get("thread_id", "default")

    # Build effective retrieval query
    retrieval_query = f"{target_topic} {user_query}"

    logger.info("[DocumentRetrieval] Retrieving for: %s (thread=%s)",
                retrieval_query[:120], thread_id)

    # --- Per-conversation file isolation --------------------------------
    # Only the files the student uploaded in THIS conversation may be
    # retrieved. Files uploaded in other conversations live in the same
    # global ChromaDB collection but are excluded here by a source whitelist.
    from src.ui.history_manager import history_manager
    allowed_sources: list[str] = history_manager.get_thread_files(thread_id)
    allowed_set: set[str] = set(allowed_sources)

    retriever = _get_retriever()

    # If no documents indexed at all, return empty
    if not retriever._bm25.is_initialized():
        logger.warning("[DocumentRetrieval] No documents indexed — returning empty")
        return {
            "retrieved_chunks": [],
            "retrieval_query": retrieval_query,
            "current_node": "document_retrieval",
        }

    # If this conversation has no uploaded files, strict isolation: return
    # empty so the explanation node falls back to DIRECT_RESPONSE (pure LLM
    # answer). Historical files from other threads must NOT leak in.
    if not allowed_set:
        logger.info("[DocumentRetrieval] Thread '%s' has no uploaded files — "
                    "strict isolation, returning empty (LLM will self-answer)",
                    thread_id)
        return {
            "retrieved_chunks": [],
            "retrieval_query": retrieval_query,
            "current_node": "document_retrieval",
            "teaching_phase": "retrieved",
        }

    # Detect if the student pointed at a specific file *within this
    # conversation's files*. A named file that wasn't uploaded here is not
    # honored (strict isolation per product decision).
    source_filter = _detect_referenced_source(user_query, allowed_sources)
    if source_filter:
        logger.info("[DocumentRetrieval] Query references source '%s' — "
                    "restricting retrieval", source_filter)
    else:
        # Student may have named a file that exists globally but not in this
        # thread; log it for debugging but do not relax isolation.
        vsm = VectorStoreManager()
        global_hit = _detect_referenced_source(user_query, vsm.list_sources())
        if global_hit and global_hit not in allowed_set:
            logger.info("[DocumentRetrieval] Query names '%s' which is not in "
                        "this thread's files — isolation maintained", global_hit)

    # When section diversity is needed, pull a wider candidate pool from the
    # retriever so that lower-scored chunks from other sections are available
    # for the per-section cap to draw on (the default top-k=5 would otherwise
    # be dominated by the highest-scoring section and never include the others).
    final_k_override = None
    if source_filter:
        final_k_override = max(settings_retrieval_top_k * 4, 20)
    elif len(allowed_sources) > 1:
        final_k_override = max(settings_retrieval_top_k * 2, 10)

    # Execute hybrid retrieval with the per-conversation source whitelist.
    # source_whitelist enforces isolation; source_filter (if set and in the
    # whitelist) further narrows to a single file inside the retriever.
    try:
        results = retriever.retrieve(
            retrieval_query,
            source_filter=source_filter,
            final_k=final_k_override,
            source_whitelist=allowed_set,
        )
    except Exception as e:
        logger.error("[DocumentRetrieval] Retrieval failed: %s", e)
        return {
            "retrieved_chunks": [],
            "retrieval_query": retrieval_query,
            "current_node": "document_retrieval",
        }

    # Source diversity (Fix3): cap per-source hits so other uploaded files
    # (within this conversation) also contribute when no explicit filter.
    if not source_filter and len(allowed_sources) > 1:
        results = _enforce_source_diversity(results, max_per_source=3)

    # Section diversity (Fix2b): when retrieval is restricted to a single
    # file, ensure chunks from multiple top-level sections are represented.
    if source_filter:
        results = _enforce_section_diversity(results, max_per_top_section=2)

    # Trim the (possibly widened) pool back to the configured final-k.
    if len(results) > settings_retrieval_top_k:
        results = results[:settings_retrieval_top_k]

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

    # Fix3: tell the LLM which OTHER uploaded materials (in this conversation)
    # exist so it can cross-reference them when explaining. Only inject when
    # this conversation has multiple files and not all were returned.
    other_sources = [s for s in allowed_sources if s != source_filter] if len(allowed_sources) > 1 else []
    if chunks and other_sources:
        other_text = "[Other available course materials you may cross-reference: " + ", ".join(other_sources) + "]"
        chunks.insert(0, RetrievedContext(
            content=other_text,
            source="system",
            page=-1,
            relevance_score=0.0,
        ))

    logger.info(
        "[DocumentRetrieval] Found %d relevant chunks (top score=%.3f) "
        "from thread files=%s",
        len(chunks), chunks[0]["relevance_score"] if chunks else 0.0,
        allowed_sources,
    )

    return {
        "retrieved_chunks": chunks,
        "retrieval_query": retrieval_query,
        "current_node": "document_retrieval",
    }


def _enforce_source_diversity(results: list, max_per_source: int = 3) -> list:
    """Cap the number of chunks returned from any single source.

    Keeps results in score order but never returns more than
    ``max_per_source`` chunks from the same source. Remaining slots are
    filled from lower-scored chunks of OTHER sources (a source that already
    hit its cap is never re-added). This guarantees that when the student
    has uploaded multiple files, the explanation can cross-reference them
    instead of only leaning on one file's top hits.
    """
    if not results:
        return results
    from collections import defaultdict
    per_source: dict[str, int] = defaultdict(int)
    kept = []
    deferred = []
    for item in results:
        src = item.node.metadata.get("source", "unknown")
        if per_source[src] < max_per_source:
            kept.append(item)
            per_source[src] += 1
        else:
            deferred.append(item)
    # Fill remaining slots from deferred — but only from sources still under cap.
    target_size = len(results)
    progress = True
    while deferred and len(kept) < target_size and progress:
        progress = False
        still_deferred = []
        for item in deferred:
            src = item.node.metadata.get("source", "unknown")
            if per_source[src] < max_per_source:
                kept.append(item)
                per_source[src] += 1
                progress = True
            else:
                still_deferred.append(item)
        deferred = still_deferred
    # If after capping we still can't fill all slots (every source is at cap),
    # that's fine — return what we have. Capping wins over total count.
    return kept


def _section_top_number(section_heading: str) -> str:
    """Extract the top-level section number from a ``section_heading`` tag.

    E.g. "3.2 Retraining Word Vectors" → "3", "2 Evaluation of Word Vectors"
    → "2". Returns "" when no leading number is found.
    """
    if not section_heading:
        return ""
    m = re.match(r"\s*(\d+)", section_heading)
    return m.group(1) if m else ""


def _enforce_section_diversity(results: list, max_per_top_section: int = 2) -> list:
    """Cap the number of chunks returned from any single top-level section.

    Used when retrieval is restricted to a single file (the student named a
    PDF) so that the LLM gets chunks spanning multiple parts of that file —
    e.g. parts 2 AND 3 instead of five copies from part 3 only. Mirrors the
    source-diversity algorithm but groups by top-level section number read
    from chunk ``metadata['section_heading']``.
    """
    if not results:
        return results
    from collections import defaultdict
    per_section: dict[str, int] = defaultdict(int)
    kept = []
    deferred = []
    for item in results:
        sec_tag = item.node.metadata.get("section_heading", "")
        sec = _section_top_number(sec_tag) or "_unknown_"
        if per_section[sec] < max_per_top_section:
            kept.append(item)
            per_section[sec] += 1
        else:
            deferred.append(item)
    target_size = len(results)
    progress = True
    while deferred and len(kept) < target_size and progress:
        progress = False
        still_deferred = []
        for item in deferred:
            sec_tag = item.node.metadata.get("section_heading", "")
            sec = _section_top_number(sec_tag) or "_unknown_"
            if per_section[sec] < max_per_top_section:
                kept.append(item)
                per_section[sec] += 1
                progress = True
            else:
                still_deferred.append(item)
        deferred = still_deferred
    return kept
