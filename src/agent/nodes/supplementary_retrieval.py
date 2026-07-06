"""Supplementary Retrieval Node (extension feature).

After the main four-part teaching is complete, this node optionally retrieves
from files the student uploaded in *other* conversations and appends a short
supplementary note to the response.

Design constraints (per product decision):
- The main four-part teaching (core / examples / math / summary) is NEVER
  affected — this node only adds an extra ``supplementary_text`` field.
- Default ON (settings.supplementary_from_history), user-toggleable via the
  ``/supplement`` command in the UI.
- Only retrieves from sources NOT in the current conversation's file list,
  so it never duplicates what the main teaching already used.
- Fail-soft: any retrieval/LLM error yields an empty supplementary_text and
  the main response is returned unchanged.
"""

import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from ..state import TeachingState
from ...indexing.hybrid_retriever import HybridRetriever
from ...indexing.vector_store import VectorStoreManager
from ...utils import retry_async, create_llm, extract_text, render_prompt
from config.prompts import TEACHING_SYSTEM_PROMPT, SUPPLEMENTARY_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)


def _get_retriever() -> HybridRetriever:
    """Reuse the same retriever singleton as the main retrieval node."""
    from .document_retrieval import _get_retriever as _main_get
    return _main_get()


async def supplementary_retrieval_node(state: TeachingState) -> dict[str, Any]:
    """Append a supplementary section drawn from other conversations' files.

    Returns state delta with ``supplementary_text`` (possibly empty).
    """
    enabled = state.get("supplementary_enabled", True)
    if not enabled:
        logger.info("[Supplementary] Disabled for this conversation — skipping")
        return {"supplementary_text": "", "current_node": "supplementary_retrieval"}

    thread_id = state.get("thread_id", "default")
    user_query = state.get("user_query", "")
    target_topic = state.get("target_topic", "")
    section_summary = state.get("section_summary", "")
    retrieval_query = state.get("retrieval_query", f"{target_topic} {user_query}")

    # Current conversation's files — these are EXCLUDED from supplementary
    # retrieval so we only surface materials from OTHER conversations.
    from src.ui.history_manager import history_manager
    current_files: set[str] = set(history_manager.get_thread_files(thread_id))

    try:
        retriever = _get_retriever()
    except Exception as e:
        logger.warning("[Supplementary] Retriever unavailable: %s — skipping", e)
        return {"supplementary_text": "", "current_node": "supplementary_retrieval"}

    if not retriever._bm25.is_initialized() or retriever._bm25.node_count == 0:
        logger.info("[Supplementary] No documents indexed — skipping")
        return {"supplementary_text": "", "current_node": "supplementary_retrieval"}

    # Global retrieval (source_whitelist=None = no isolation) then filter OUT
    # the current conversation's files so only cross-conversation material
    # remains as supplementary.
    try:
        global_results = retriever.retrieve(
            retrieval_query,
            source_whitelist=None,
            final_k=settings.supplementary_top_k * 3,
        )
    except Exception as e:
        logger.warning("[Supplementary] Global retrieval failed: %s — skipping", e)
        return {"supplementary_text": "", "current_node": "supplementary_retrieval"}

    supplementary_chunks = []
    for r in global_results:
        src = r.node.metadata.get("source", "")
        if src and src not in current_files:
            supplementary_chunks.append(r)
        if len(supplementary_chunks) >= settings.supplementary_top_k:
            break

    if not supplementary_chunks:
        logger.info("[Supplementary] No cross-conversation chunks available — skipping")
        return {"supplementary_text": "", "current_node": "supplementary_retrieval"}

    # Build context text for the LLM, each excerpt labelled with its source.
    excerpts = []
    for r in supplementary_chunks:
        src = r.node.metadata.get("source", "unknown")
        content = r.node.get_content().strip()
        excerpts.append(f"[Source: {src}]\n{content}")
    supplementary_context = "\n\n".join(excerpts)

    logger.info(
        "[Supplementary] %d cross-conversation chunks from sources=%s",
        len(supplementary_chunks),
        sorted({r.node.metadata.get("source", "") for r in supplementary_chunks}),
    )

    # Generate the supplementary note via LLM.
    try:
        llm = create_llm(temperature=0.4, max_tokens=800)
        prompt = render_prompt(
            SUPPLEMENTARY_PROMPT,
            user_query=user_query,
            section_summary=section_summary,
            supplementary_context=supplementary_context,
        )
        response = await retry_async(lambda: llm.ainvoke([
            SystemMessage(content=TEACHING_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]))
        supplementary_text = extract_text(response).strip()
    except Exception as e:
        logger.warning("[Supplementary] LLM generation failed: %s — skipping", e)
        return {"supplementary_text": "", "current_node": "supplementary_retrieval"}

    logger.info("[Supplementary] Generated %d chars of supplementary text",
                len(supplementary_text))

    return {
        "supplementary_text": supplementary_text,
        "current_node": "supplementary_retrieval",
    }
