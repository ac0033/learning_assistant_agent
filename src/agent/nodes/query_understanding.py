"""Query Understanding Node.

Analyzes the student's query to determine:
- Intent: are they learning something new, clarifying, reviewing, or just navigating?
- Target topic: what concept are they asking about?
- Difficulty: beginner, intermediate, or advanced?
- Whether examples and math notation are needed.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage

from ..state import TeachingState
from ...utils import retry_async, create_llm, extract_text, render_prompt
from config.prompts import QUERY_UNDERSTANDING_PROMPT

logger = logging.getLogger(__name__)

_TEACHING_INTENT_RE = re.compile(
    r"(?:讲解|讲一下|讲讲|讲下|开始讲|开始讲解|开始吧|开始|学习|学一下|学学|"
    r"介绍|介绍一下|解释|解释一下|说明|说明一下|教|教教|带我|帮我学|"
    r"explain|teach|introduce|walk\s+through|walk\s+me\s+through|cover|go\s+over|"
    r"learn|tell\s+me\s+about|describe|elaborate)",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^(?:你好|您好|hi|hello|hey|哈喽|嗨|你是谁|怎么用|如何使用|谢谢|thanks|"
    r"ok|okay|好的|收到|明白)[\s!！.。?？]*$",
    re.IGNORECASE,
)


async def query_understanding_node(state: TeachingState) -> dict[str, Any]:
    """Analyze the student's query and extract structured information.

    Returns updated state with intent, target_topic, difficulty,
    needs_example, needs_math, and current_node set.
    """
    user_query = state.get("user_query", "")
    logger.info("[QueryUnderstanding] Analyzing: %s", user_query[:100])

    # Use LLM to extract structured information
    llm = create_llm(temperature=0.0, max_tokens=500)

    prompt = render_prompt(QUERY_UNDERSTANDING_PROMPT, user_query=user_query)
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

    # --- Code-level guard: fix LLM misclassification of vague teaching requests ---
    # LLM may label "开始讲解" etc. as "navigate" because the topic is unstated,
    # which sends the pipeline straight to END with an empty greeting.
    # Override: if intent is navigate but the query contains teaching keywords
    # AND is not a pure greeting, reclassify as learn_new.
    if intent == "navigate":
        is_greeting = bool(_GREETING_RE.match(user_query.strip()))
        has_teaching_keyword = bool(_TEACHING_INTENT_RE.search(user_query))
        if has_teaching_keyword and not is_greeting:
            logger.info(
                "[QueryUnderstanding] Override navigate→learn_new (teaching keywords detected, not a pure greeting)"
            )
            intent = "learn_new"
            if not result.get("target_topic") or result.get("target_topic") == "general":
                result["target_topic"] = "overview of uploaded course materials"

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
