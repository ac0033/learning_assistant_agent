"""Shared utilities: retry logic, LLM factory, response parsing."""

import logging
import asyncio
import re
from typing import Any, Callable, Union

logger = logging.getLogger(__name__)


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_prompt(template: str, **kwargs: Any) -> str:
    """Safely substitute {placeholder} tokens in a prompt template.

    Unlike str.format(), injected values are NOT re-parsed for braces,
    so LaTeX subscripts like ``X_{ij}`` in retrieved chunks or LLM output
    do not raise KeyError.
    """

    def _replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name in kwargs:
            return str(kwargs[name])
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)


def extract_text(response: Any) -> str:
    """Extract plain text from an LLM response object.

    Handles both:
    - Standard Anthropic responses (response.content is str)
    - DeepSeek v4 responses (response.content is list of content blocks,
      e.g., [{'type': 'thinking', ...}, {'type': 'text', 'text': '...'}])

    Args:
        response: An LLM response object with a .content attribute.

    Returns:
        Plain text string.
    """
    content = response.content if hasattr(response, 'content') else str(response)

    # Standard string response
    if isinstance(content, str):
        return content

    # DeepSeek v4: list of content blocks (thinking + text)
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text' and block.get('text'):
                    text_parts.append(block['text'])
                elif block.get('type') == 'text_delta' and block.get('text'):
                    text_parts.append(block['text'])
        if text_parts:
            return ''.join(text_parts)
        # Fallback: if no text blocks, return the raw list as string
        return str(content)

    return str(content)


def is_truncated(response: Any) -> bool:
    """Return True if the LLM stopped due to a token-length cap (truncated).

    Checks the common stop-reason keys across Anthropic / OpenAI-compatible
    backends exposed via ``response_metadata``. A truncated response means the
    node's ``max_tokens`` budget was hit mid-generation — the caller should
    raise the budget or the content will appear cut off.
    """
    meta = getattr(response, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return False
    # Anthropic uses 'stop_reason'; OpenAI-compatible uses 'finish_reason'.
    reason = meta.get("stop_reason") or meta.get("finish_reason") or ""
    reason = str(reason).lower()
    return reason in ("max_tokens", "length", "max-tokens")


async def retry_async(
    coro_factory: Callable[[], Any],
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
) -> Any:
    """Async retry helper for one-off LLM calls.

    Args:
        coro_factory: Async callable that returns a coroutine.
        max_attempts: Maximum number of attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.

    Returns:
        The result of the successful coroutine call.

    Raises:
        The last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt == max_attempts:
                logger.error("LLM call failed after %d attempts: %s", attempt, e)
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "LLM call attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt, max_attempts, e, delay
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLM factory — single place to configure ChatAnthropic
# ---------------------------------------------------------------------------

def create_llm(temperature: float = 0.5, max_tokens: int = 3000) -> Any:
    """Create a ChatAnthropic instance with centralized configuration.

    All LLM nodes should use this factory so that base_url, model,
    and API key are configured in one place.

    Args:
        temperature: LLM temperature (0.0 = deterministic, 0.5 = creative).
        max_tokens: Maximum output tokens.

    Returns:
        Configured ChatAnthropic instance.
    """
    from langchain_anthropic import ChatAnthropic
    from config.settings import settings

    kwargs: dict[str, Any] = {
        "model": settings.anthropic_model,
        "api_key": settings.anthropic_api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Only set base_url if configured (empty string → use default endpoint)
    if settings.anthropic_base_url:
        kwargs["base_url"] = settings.anthropic_base_url

    return ChatAnthropic(**kwargs)
