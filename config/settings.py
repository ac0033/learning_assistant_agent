"""Centralized configuration via Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Anthropic ---
    anthropic_api_key: str = Field(
        default="",
        alias="ANTHROPIC_API_KEY",
        description="Anthropic API key. Required at runtime, but not at import time."
    )
    anthropic_base_url: str = Field(
        default="",
        alias="ANTHROPIC_BASE_URL",
        description="Custom base URL for Anthropic-compatible API (e.g., DeepSeek, OpenRouter). Leave empty for default."
    )
    # --- Model ---
    anthropic_model: str = Field(
        default="deepseek-v4-flash",
        alias="MODEL_ID",
        description="LLM model ID. Set via MODEL_ID env var (e.g., deepseek-v4-flash, claude-sonnet-4-6)."
    )

    # --- Embedding ---
    embedding_provider: str = Field(
        default="siliconflow",
        alias="EMBEDDING_PROVIDER",
        description="Embedding provider: 'siliconflow' (recommended, free tier), 'openai', or 'voyage'."
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        alias="EMBEDDING_MODEL",
        description="Embedding model name. SiliconFlow: BAAI/bge-m3. OpenAI: text-embedding-3-small. Voyage: voyage-3."
    )
    siliconflow_api_key: str = Field(default="", alias="SILICONFLOW_API_KEY")
    voyage_api_key: str = Field(default="", alias="VOYAGE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # --- Chainlit ---
    chainlit_port: int = Field(default=8000, alias="CHAINLIT_PORT")
    chainlit_auth_enabled: bool = Field(default=False, alias="CHAINLIT_AUTH_ENABLED")

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=5, description="Final chunks after reranking")
    retrieval_candidate_k: int = Field(default=20, description="Candidates before reranking")
    chunk_size: int = Field(default=1024, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=128, description="Chunk overlap in tokens")

    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    documents_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "documents")
    chroma_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "chroma_db")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def embedding_api_key(self) -> str:
        """Get the active embedding API key based on provider."""
        if self.embedding_provider == "siliconflow":
            return self.siliconflow_api_key
        elif self.embedding_provider == "voyage":
            return self.voyage_api_key
        else:
            return self.openai_api_key

    def validate_runtime(self) -> None:
        """Validate that all required API keys are set. Call before using LLM/embeddings."""
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if self.embedding_provider == "siliconflow" and not self.siliconflow_api_key:
            missing.append("SILICONFLOW_API_KEY (or switch EMBEDDING_PROVIDER)")
        if self.embedding_provider == "voyage" and not self.voyage_api_key:
            missing.append("VOYAGE_API_KEY (or switch EMBEDDING_PROVIDER)")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY (or switch EMBEDDING_PROVIDER)")
        if missing:
            raise ValueError(
                f"Missing required API keys: {', '.join(missing)}. "
                f"Set them in .env file or environment variables."
            )


# Singleton
settings = Settings()
