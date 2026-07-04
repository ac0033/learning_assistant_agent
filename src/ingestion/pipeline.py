"""End-to-end document ingestion pipeline.

Orchestrates: PDF load → chunk → embed → store.
"""

import logging
from pathlib import Path
from typing import List, Optional

from .loader import PDFLoader, ParsedDocument
from .chunker import SemanticChunker
from ..indexing.embeddings import get_embed_model
from ..indexing.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrate the full document ingestion workflow.

    Usage:
        pipeline = IngestionPipeline()
        pipeline.ingest_file(Path("lecture1.pdf"))
        pipeline.ingest_directory(Path("data/documents/stanford_cs229/"))
    """

    def __init__(self):
        self._loader = PDFLoader()
        self._embed_model = get_embed_model()
        self._chunker = SemanticChunker(embed_model=self._embed_model)
        self._vector_store = VectorStoreManager()

    def ingest_file(self, file_path: Path) -> int:
        """Ingest a single PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Number of chunks indexed.

        Raises:
            RuntimeError: Wraps any step failure with context for debugging.
        """
        logger.info("Starting ingestion: %s", file_path.name)

        try:
            # Step 1: Load PDF
            doc = self._loader.load(file_path)

            # Step 2: Chunk
            chunk_result = self._chunker.chunk(doc.text, source_path=file_path.name)

            # Step 3: Generate embeddings for every chunk
            if len(chunk_result.nodes) > 1:
                texts = [node.get_content() for node in chunk_result.nodes]
                embeddings = self._embed_model.get_text_embedding_batch(texts)
                for node, embedding in zip(chunk_result.nodes, embeddings):
                    node.embedding = embedding
            elif len(chunk_result.nodes) == 1:
                chunk_result.nodes[0].embedding = self._embed_model.get_text_embedding(
                    chunk_result.nodes[0].get_content()
                )
            else:
                logger.warning("No chunks produced for %s", file_path.name)
                return 0

            # Step 4: Store in vector DB
            self._vector_store.add_nodes(chunk_result.nodes)

        except Exception as e:
            raise RuntimeError(
                f"Ingestion failed for {file_path.name} at step: "
                f"See cause above. File: {file_path}"
            ) from e

        logger.info(
            "Ingested %s: %d pages → %d chunks → indexed",
            file_path.name, doc.metadata.get("num_pages", -1), len(chunk_result)
        )

        return len(chunk_result)

    def ingest_directory(self, dir_path: Path, recursive: bool = True) -> int:
        """Ingest all PDFs in a directory.

        Args:
            dir_path: Directory containing PDF files.
            recursive: Whether to search subdirectories.

        Returns:
            Total number of chunks indexed.
        """
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(dir_path.glob(pattern))

        if not pdf_files:
            logger.warning("No PDF files found in %s", dir_path)
            return 0

        total_chunks = 0
        for pdf_path in pdf_files:
            try:
                chunks = self.ingest_file(pdf_path)
                total_chunks += chunks
            except Exception as e:
                logger.error("Failed to ingest %s: %s", pdf_path.name, e)
                continue

        logger.info("Ingested %d PDFs → %d total chunks", len(pdf_files), total_chunks)
        return total_chunks

    def get_indexed_documents(self) -> List[str]:
        """Return list of unique source filenames in the vector store."""
        return self._vector_store.list_sources()
