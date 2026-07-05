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

# Math-signal keywords (en + zh). When the spoken topic matches one of these,
# the LLM's `needs_math=False` is overridden to True so the ③ Math & Notation
# node is never skipped for topics that obviously contain formulas.
_MATH_SIGNAL_RE = re.compile(
    r"(?:softmax|sigmoid|cross[\s-]*entropy|loss|gradient|derivative|backprop(?:agation)?|"
    r"matrix|matrices|vector|向量|矩阵|概率|probability|distribution|分布|"
    r"激活|activation|relu|tanh|"
    r"rnn|lstm|gru|transformer|attention|q\s*k\s*v|"
    r"normaliz(?:e|ation)|regulariz(?:e|ation)|dropout|"
    r"最小化|minimi[sz]e|最大|maximi[sz]e|优化|optimi[sz]|objective|"
    r"线性|linear|非线性|nonlinear|"
    r"likelihood|似然|bayes|贝叶斯|posterior|先验|prior|"
    r"epoch|lr|learning\s*rate|update|更新规则|"
    r"correlation|correlat|co-?occurrence|co\s*occurrence|"
    r"fitness|误差|偏差|variance|偏差|偏差方差|"
    r"singular|decompos|分解|eigen|特征值|特征向量|特征向量|"
    r"公式|formula|equation|等式|推导|derivation|"
    r"\\sum|\\frac|\\int|\\beta|\\alpha|\\theta|\\lambda|\\sigma\-?|\$\$)",
    re.IGNORECASE,
)
_MATH_HINT_RE = re.compile(r"(?:第\s*[2-9]\s*部分|part\s*[2-9]|math|notation|符号|推导|公式)", re.IGNORECASE)


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

    # --- Code-level guard: never skip the ③ Math & Notation node for mathy topics ---
    # LLM may set needs_math=False for "softmax / loss / gradient / 正则化" because
    # it classified the *task* as conceptual rather than formulaic. But the
    # student still wants the derivation. Force needs_math=True when:
    #   - the query or the (LLM-extracted) target_topic mentions a math signal word.
    # We never downgrade a True to False — only the other way around.
    needs_math = bool(result.get("needs_math", True))
    target_topic = result.get("target_topic", "") or user_query
    if not needs_math:
        text_for_check = f"{user_query} {target_topic} {result.get('intent','')}"
        looks_mathy = bool(_MATH_SIGNAL_RE.search(text_for_check)) or bool(_MATH_HINT_RE.search(text_for_check))
        if looks_mathy:
            needs_math = True
            logger.info(
                "[QueryUnderstanding] Override needs_math False→True (math signal detected in query/topic)"
            )

    logger.info(
        "[QueryUnderstanding] intent=%s, topic=%s, difficulty=%s, needs_math=%s",
        intent, target_topic, result.get("difficulty"), needs_math
    )

    return {
        "intent": intent,
        "target_topic": target_topic,
        "difficulty": result.get("difficulty", "intermediate"),
        "needs_example": result.get("needs_example", True),
        "needs_math": needs_math,
        "current_node": "query_understanding",
    }
