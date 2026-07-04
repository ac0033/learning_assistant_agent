"""Query Understanding Node.

Analyzes the student's query to determine:
- Intent: are they learning something new, clarifying, reviewing, or just navigating?
- Target topic: what concept are they asking about?
- Difficulty: beginner, intermediate, or advanced?
- Whether examples and math notation are needed.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from ..state import TeachingState
from ...utils import retry_async, create_llm, extract_text
from config.prompts import QUERY_UNDERSTANDING_PROMPT

logger = logging.getLogger(__name__)


async def query_understanding_node(state: TeachingState) -> dict[str, Any]:
    """Analyze the student's query and extract structured information.

    Returns updated state with intent, target_topic, difficulty,
    needs_example, needs_math, and current_node set.
    """
    user_query = state.get("user_query", "")
    logger.info("[QueryUnderstanding] Analyzing: %s", user_query[:100])

    # Use LLM to extract structured information
    llm = create_llm(temperature=0.0, max_tokens=500)

    prompt = QUERY_UNDERSTANDING_PROMPT.format(user_query=user_query)
    # Must use HumanMessage: DeepSeek's Anthropic-compatible API rejects
    # messages arrays that contain only system messages with no user message.
    response = await retry_async(lambda: llm.ainvoke([HumanMessage(content=prompt)]))

    try:
        # Parse JSON response
        content = extract_text(response)
        # Strip markdown code block markers if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
    except (json.JSONDecodeError, IndexError) as e:
        logger.warning("Failed to parse query understanding JSON: %s", e)
        # Fallback: assume it's a learning query
        result = {
            "intent": "learn_new",
            "target_topic": user_query,
            "difficulty": "intermediate",
            "needs_example": True,
            "needs_math": True,
        }

    intent = result.get("intent", "learn_new")
    logger.info(
        "[QueryUnderstanding] intent=%s, topic=%s, difficulty=%s",
        intent, result.get("target_topic"), result.get("difficulty")
    )

    return {
        "intent": intent,
        "target_topic": result.get("target_topic", user_query),
        "difficulty": result.get("difficulty", "intermediate"),
        "needs_example": result.get("needs_example", True),
        "needs_math": result.get("needs_math", True),
        "current_node": "query_understanding",
    }
