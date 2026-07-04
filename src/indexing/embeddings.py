"""Embedding model factory.

Supports SiliconFlow (BGE-M3, free tier), Voyage AI, and OpenAI embeddings.
All providers use API-based inference — no local GPU required.
Uses LlamaIndex's embedding abstractions for seamless integration.
"""

import logging
from typing import Optional

from llama_index.core.embeddings import BaseEmbedding

from config.settings import settings

logger = logging.getLogger(__name__)

_embed_model: Optional[BaseEmbedding] = None


def get_embed_model() -> BaseEmbedding:
    """Get or create the global embedding model singleton.

    Safe without a lock: asyncio is single-threaded and this function
    has no await points, so it runs atomically from check to assignment.

    Returns:
        LlamaIndex BaseEmbedding instance (VoyageAI or OpenAI).
    """
    global _embed_model

    if _embed_model is not None:
        return _embed_model

    provider = settings.embedding_provider
    model_name = settings.embedding_model

    if provider == "siliconflow":
        if not settings.siliconflow_api_key:
            raise ValueError(
                "SILICONFLOW_API_KEY is required for SiliconFlow embeddings. "
                "Get a free key at https://siliconflow.cn, or switch EMBEDDING_PROVIDER."
            )
        # SiliconFlow is OpenAI-compatible — reuse OpenAIEmbedding with custom base URL
        from llama_index.embeddings.openai import OpenAIEmbedding
        _embed_model = OpenAIEmbedding(
            model_name=model_name,
            api_key=settings.siliconflow_api_key,
            api_base="https://api.siliconflow.cn/v1",
        )
        logger.info("Using SiliconFlow embeddings: %s", model_name)
    elif provider == "voyage":
        if not settings.voyage_api_key:
            raise ValueError(
                "VOYAGE_API_KEY is required for Voyage embeddings. "
                "Set it in .env or switch EMBEDDING_PROVIDER."
            )
        try:
            from llama_index.embeddings.voyageai import VoyageEmbedding
            _embed_model = VoyageEmbedding(
                model_name=model_name,
                voyage_api_key=settings.voyage_api_key,
            )
            logger.info("Using Voyage AI embeddings: %s", model_name)
        except ImportError:
            logger.warning(
                "llama-index-embeddings-voyageai not installed. "
                "Install with: uv add llama-index-embeddings-voyageai"
            )
            raise
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI embeddings. "
                "Set it in .env or switch EMBEDDING_PROVIDER to 'voyage'."
            )
        from llama_index.embeddings.openai import OpenAIEmbedding
        _embed_model = OpenAIEmbedding(
            model_name=model_name,
            api_key=settings.openai_api_key,
        )
        logger.info("Using OpenAI embeddings: %s", model_name)
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider}. "
            "Supported: 'siliconflow', 'openai', 'voyage'."
        )

    return _embed_model
