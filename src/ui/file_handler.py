"""Chainlit file upload handler.

Provides reusable async functions for processing uploaded PDF files
through the ingestion pipeline: Save → Parse → Chunk → Embed → Index.

Chainlit v2.11.1 does not have @cl.on_upload or @cl.on_file_upload.
File uploads are handled in two ways:
  1. cl.AskFileMessage — prompted dialog in @cl.on_chat_start (in chat_handler.py)
  2. message.elements — files attached to user messages in @cl.on_message (in chat_handler.py)
"""

import logging
from pathlib import Path
import asyncio
import shutil
from typing import List, Tuple

import chainlit as cl

from ..ingestion.pipeline import IngestionPipeline
from config.settings import settings

logger = logging.getLogger(__name__)

# Singleton pipeline (lazy init)
_pipeline: IngestionPipeline | None = None


def _get_pipeline() -> IngestionPipeline:
    """Get or create the ingestion pipeline singleton.

    Safe without a lock: asyncio is single-threaded and this function
    has no await points, so it runs atomically from check to assignment.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


async def process_uploaded_files(
    files: list,
    status_msg: cl.Message,
) -> Tuple[int, int]:
    """Process uploaded PDF files through the ingestion pipeline.

    This is a reusable pure function — call it from @cl.on_chat_start
    or @cl.on_message when file elements are detected.

    Args:
        files: A list of uploaded file objects (from cl.AskFileMessage or
               message.elements). Each must have .path and .name attributes.
        status_msg: A cl.Message to update with progress information.

    Returns:
        (success_count, total_chunks): Number of files processed and total chunks created.
    """
    # Filter PDF files
    pdf_files = [f for f in files if f.name.lower().endswith(".pdf")]
    non_pdf = [f.name for f in files if not f.name.lower().endswith(".pdf")]

    if non_pdf:
        logger.warning("Skipping non-PDF files: %s", non_pdf)

    if not pdf_files:
        status_msg.content = "⚠️ 未检测到 PDF 文件，请上传课程讲义（.pdf 格式）。"
        await status_msg.update()
        return 0, 0

    status_msg.content = f"⚙️ 正在处理 {len(pdf_files)} 个 PDF 文件，请稍候..."
    await status_msg.update()

    pipeline = _get_pipeline()
    total_chunks = 0
    success_count = 0

    for file in pdf_files:
        try:
            file_path = Path(file.path)

            # Ensure persistent storage directory exists
            doc_dir = settings.documents_dir
            doc_dir.mkdir(parents=True, exist_ok=True)
            dest_path = doc_dir / file.name

            # Copy file to local document repository for persistence
            if dest_path != file_path.resolve():
                shutil.copy2(str(file_path), str(dest_path))

            # Run CPU/IO-intensive ingestion in thread pool
            chunks = await asyncio.to_thread(pipeline.ingest_file, dest_path)
            total_chunks += chunks
            success_count += 1

            logger.info("Ingested: %s (%d chunks)", file.name, chunks)

        except Exception as e:
            logger.error("Failed to process %s: %s", file.name, e)
            await cl.Message(
                content=f"❌ 处理 {file.name} 时出错: {str(e)}"
            ).send()
            continue

    # Refresh hybrid retriever's BM25 index
    try:
        from ..indexing.hybrid_retriever import refresh_retriever
        refresh_retriever()
    except Exception as e:
        logger.warning("Failed to refresh retriever: %s. Will rebuild on demand.", e)

    return success_count, total_chunks


@cl.on_chat_resume
async def on_chat_resume():
    """Restore session when a user reconnects to an existing chat."""
    thread_id = cl.user_session.get("thread_id", "unknown")
    logger.info("Chat resumed: thread=%s", thread_id)
