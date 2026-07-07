"""Math & Notation Node.

Generates Part ③ of the four-part teaching output: Math & Notation.
Provides rigorous formula derivation with symbol-by-symbol explanation.
"""

import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from ..state import TeachingState
from ...utils import retry_async, create_llm, extract_text, is_truncated, render_prompt
from config.prompts import TEACHING_SYSTEM_PROMPT, MATH_NOTATION_PROMPT

logger = logging.getLogger(__name__)


async def math_notation_node(state: TeachingState) -> dict[str, Any]:
    """Generate math notation explanation (Part ③).

    Derives formulas step-by-step and explains every symbol.
    Connects mathematical formalism to the intuition from Parts ① and ②.
    """
    user_query = state.get("user_query", "")
    core_explanation = state.get("core_explanation", "")
    examples = state.get("examples", "")

    logger.info("[MathNotation] Generating math notation for: %s", user_query[:80])

    # 6000-token budget: Part ③ routinely contains many numbered subsections
    # (e.g. "5. 你可能会遇到的符号变体"); the previous 3000 cap truncated
    # mid-subsection, making the output jump to ④ Summary prematurely.
    llm = create_llm(temperature=0.3, max_tokens=6000)

    prompt = render_prompt(
        MATH_NOTATION_PROMPT,
        core_explanation=core_explanation,
        examples=examples,
        user_query=user_query,
    )

    response = await retry_async(lambda: llm.ainvoke([
        SystemMessage(content=TEACHING_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]))

    math_notation = extract_text(response)

    if is_truncated(response):
        logger.warning(
            "[MathNotation] Output truncated at max_tokens=6000 (%d chars) — "
            "Part ③ may be incomplete; consider raising the budget.",
            len(math_notation),
        )

    logger.info("[MathNotation] Generated %d chars of math notation", len(math_notation))

    return {
        "math_notation": math_notation,
        "teaching_phase": "math",
        "current_node": "math_notation",
    }
