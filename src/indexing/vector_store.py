"""ChromaDB vector store wrapper for document storage and retrieval."""

import logging
import threading
from typing import List, Optional

from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store for document embeddings.

    Wraps LlamaIndex's ChromaVectorStore with collection management,
    persistent storage, and convenience methods for retrieval.
    """

    COLLECTION_NAME = "course_materials"

    def __init__(self):
        self._lock = threading.Lock()
        self._chroma_dir = str(settings.chroma_dir)
        self._chroma_dir_path = settings.chroma_dir

        try:
            self._chroma_dir_path.mkdir(parents=True, exist_ok=True)

            # Initialize persistent ChromaDB client
            self._client = chromadb.PersistentClient(
                path=self._chroma_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

            # LlamaIndex wrapper
            self._vector_store = ChromaVectorStore(
                chroma_collection=self._collection,
            )

            doc_count = self._collection.count()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB at {self._chroma_dir}: {e}\n"
                f"Check disk space, permissions, and that no other process is locking the database."
            ) from e

        logger.info(
            "VectorStore ready: %s (collection=%s, docs=%d)",
            self._chroma_dir,
            self.COLLECTION_NAME,
            doc_count,
        )

    @property
    def store(self) -> ChromaVectorStore:
        """Get the underlying LlamaIndex ChromaVectorStore."""
        return self._vector_store

    @property
    def collection(self):
        """Get the raw ChromaDB collection."""
        return self._collection

    def add_nodes(self, nodes: List[BaseNode]) -> None:
        """Add document nodes to the vector store.

        Each node is embedded and indexed with its metadata.
        Thread-safe: serializes writes via internal lock.
        """
        if not nodes:
            return

        with self._lock:
            self._vector_store.add(nodes)
        logger.info("Added %d nodes to vector store", len(nodes))

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> VectorStoreQueryResult:
        """Query the vector store by embedding.

        Args:
            query_embedding: Dense embedding vector.
            top_k: Number of results to return.
            where: Optional metadata filter dict.

        Returns:
            VectorStoreQueryResult with nodes, similarities, and ids.
        """
        q = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            filters=where,
        )
        with self._lock:
            return self._vector_store.query(q)

    def delete_by_source(self, source_filename: str) -> int:
        """Delete all chunks from a specific source document.

        Returns:
            Number of chunks deleted.
        """
        # Get IDs of chunks from this source
        result = self._collection.get(
            where={"source": source_filename},
        )
        ids_to_delete = result.get("ids", [])

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info("Deleted %d chunks from source: %s", len(ids_to_delete), source_filename)

        return len(ids_to_delete)

    def list_sources(self) -> List[str]:
        """List all unique source documents in the store."""
        result = self._collection.get()
        metadata_list = result.get("metadatas", [])
        sources = set()
        for meta in metadata_list:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def count(self) -> int:
        """Total number of chunks in the store."""
        return self._collection.count()
