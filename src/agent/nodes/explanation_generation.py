"""Explanation Generation Node.

Generates Part ① of the four-part teaching output: Core Process.
This is the primary teaching content — a clear, logical walkthrough of the concept.
"""

import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from ..state import TeachingState
from ...utils import retry_async, create_llm, extract_text
from config.prompts import (
    TEACHING_SYSTEM_PROMPT,
    EXPLANATION_GENERATION_PROMPT,
    DIRECT_RESPONSE_PROMPT,
)

logger = logging.getLogger(__name__)


async def explanation_generation_node(state: TeachingState) -> dict[str, Any]:
    """Generate the core process explanation (Part ①).

    Uses retrieved chunks as primary context for the explanation.
    If no chunks were retrieved, falls back to a direct response.
    """
    target_topic = state.get("target_topic", "")
    user_query = state.get("user_query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    logger.info(
        "[ExplanationGen] Generating for topic='%s' with %d chunks",
        target_topic, len(retrieved_chunks)
    )

    llm = create_llm(temperature=0.5, max_tokens=3000)

    if retrieved_chunks:
        # With retrieved context: use structured prompt
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['source']}, Page: {c['page']}, Score: {c['relevance_score']}]\n{c['content']}"
            for c in retrieved_chunks
        )

        prompt = EXPLANATION_GENERATION_PROMPT.format(
            retrieved_context=context_text,
            target_topic=target_topic,
            user_query=user_query,
        )
    else:
        # No retrieval: direct response
        prompt = DIRECT_RESPONSE_PROMPT.format(
            user_query=user_query,
        )

    response = await retry_async(lambda: llm.ainvoke([
        SystemMessage(content=TEACHING_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]))

    explanation = extract_text(response)

    logger.info(
        "[ExplanationGen] Generated %d chars of core explanation",
        len(explanation)
    )

    return {
        "core_explanation": explanation,
        "teaching_phase": "core",
        "current_node": "explanation_generation",
    }
