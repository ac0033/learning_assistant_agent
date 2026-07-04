"""Example Generation Node.

Generates Part ② of the four-part teaching output: Interspersed Examples.
Provides concrete, detailed examples to illuminate abstract concepts.
"""

import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from ..state import TeachingState
from ...utils import retry_async, create_llm, extract_text, render_prompt
from config.prompts import TEACHING_SYSTEM_PROMPT, EXAMPLE_GENERATION_PROMPT

logger = logging.getLogger(__name__)


async def example_generation_node(state: TeachingState) -> dict[str, Any]:
    """Generate detailed examples (Part ②).

    Builds on the core explanation to provide concrete illustrations
    of the concept in action.
    """
    user_query = state.get("user_query", "")
    core_explanation = state.get("core_explanation", "")

    logger.info("[ExampleGen] Generating examples for query: %s", user_query[:80])

    llm = create_llm(temperature=0.5, max_tokens=3000)

    prompt = render_prompt(
        EXAMPLE_GENERATION_PROMPT,
        core_explanation=core_explanation,
        user_query=user_query,
    )

    response = await retry_async(lambda: llm.ainvoke([
        SystemMessage(content=TEACHING_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]))

    examples = extract_text(response)

    logger.info("[ExampleGen] Generated %d chars of examples", len(examples))

    return {
        "examples": examples,
        "teaching_phase": "example",
        "current_node": "example_generation",
    }
