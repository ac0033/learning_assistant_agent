"""Chainlit entry point for the AI Teaching Assistant Agent.

Start with:
    chainlit run src/main.py

Or programmatically:
    python src/main.py
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import UI handlers — registers @cl.on_chat_start, @cl.on_message, @cl.on_chat_resume
from src.ui.chat_handler import on_chat_start, on_message
from src.ui.file_handler import on_chat_resume

from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Startup Validation ---
try:
    settings.validate_runtime()
    logger.info("API keys validated successfully")
    logger.info("LLM model: %s", settings.anthropic_model)
    logger.info("Embedding: %s / %s", settings.embedding_provider, settings.embedding_model)
except ValueError as e:
    logger.warning("API keys not configured: %s", e)
    logger.warning("Set them in .env file — the app will start but API calls will fail")

logger.info("Project root: %s", PROJECT_ROOT)
logger.info("Documents dir: %s", settings.documents_dir)
logger.info("ChromaDB dir: %s", settings.chroma_dir)

# Expose handlers for Chainlit to discover
__all__ = ["on_chat_start", "on_message", "on_chat_resume"]


def main():
    """Run Chainlit programmatically (alternative to `chainlit run`)."""
    import subprocess
    logger.info("Starting AI Teaching Assistant Agent...")

    subprocess.run(
        ["chainlit", "run", str(Path(__file__).resolve()), "--port", str(settings.chainlit_port)],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
