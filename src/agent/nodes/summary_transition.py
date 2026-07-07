"""Summary & Transition Node.

Generates Part ④ of the four-part teaching output: Summary.
Distills key takeaways and bridges to related topics.
"""

import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from ..state import TeachingState
from ...utils import retry_async, create_llm, extract_text, is_truncated, render_prompt
from config.prompts import TEACHING_SYSTEM_PROMPT, SUMMARY_PROMPT

logger = logging.getLogger(__name__)


async def summary_transition_node(state: TeachingState) -> dict[str, Any]:
    """Generate summary and transition (Part ④).

    Distills key takeaways, identifies the ONE most important idea,
    bridges to related topics, and ends with a check-in question.
    """
    user_query = state.get("user_query", "")
    core_explanation = state.get("core_explanation", "")
    examples = state.get("examples", "")
    math_notation = state.get("math_notation", "")

    logger.info("[Summary] Generating summary for: %s", user_query[:80])

    # 4000-token budget (was 2000): the summary compresses all three prior
    # parts, so it needs more room to avoid a mid-sentence truncation.
    llm = create_llm(temperature=0.5, max_tokens=4000)

    prompt = render_prompt(
        SUMMARY_PROMPT,
        core_explanation=core_explanation,
        examples=examples,
        math_notation=math_notation,
        user_query=user_query,
    )

    response = await retry_async(lambda: llm.ainvoke([
        SystemMessage(content=TEACHING_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]))

    summary = extract_text(response)

    if is_truncated(response):
        logger.warning(
            "[Summary] Output truncated at max_tokens=4000 (%d chars) — "
            "Part ④ may be incomplete; consider raising the budget.",
            len(summary),
        )

    logger.info("[Summary] Generated %d chars of summary", len(summary))

    return {
        "section_summary": summary,
        "teaching_phase": "summary",
        "current_node": "summary_transition",
    }
