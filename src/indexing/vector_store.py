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

        Each node is embedded and indexed with its metadata. Re-upload of the
        same content is suppressed: nodes whose text already exists in the
        store (same source + identical content) are skipped, which is the
        safety net that prevents re-ingestion from polluting the index even
        when a content-hash dedup at the upload layer missed a case.

        Thread-safe: serializes writes via internal lock.
        """
        if not nodes:
            return

        # --- De-dup guard: skip nodes whose (source, content) already exists ---
        try:
            by_source: dict[str, set[str]] = {}
            for node in nodes:
                src = node.metadata.get("source", "")
                by_source.setdefault(src, set()).add(node.get_content())
            existing_contents_per_source: dict[str, set[str]] = {}
            for src, contents in by_source.items():
                if not src:
                    continue
                res = self._collection.get(where={"source": src})
                existing_contents_per_source[src] = set(res.get("documents", []) or [])
            deduped = []
            for node in nodes:
                src = node.metadata.get("source", "")
                existing = existing_contents_per_source.get(src, set())
                if node.get_content() in existing:
                    logger.info("Skip duplicate chunk (source=%s, %d chars)",
                                src, len(node.get_content()))
                    continue
                deduped.append(node)
            skipped = len(nodes) - len(deduped)
            if skipped:
                logger.info("De-dup suppressed %d duplicate chunks before add", skipped)
        except Exception as e:
            logger.warning("De-dup pre-check failed (%s) — adding all nodes", e)
            deduped = list(nodes)

        if not deduped:
            logger.info("add_nodes: all %d nodes were duplicates — nothing to add", len(nodes))
            return

        with self._lock:
            self._vector_store.add(deduped)
        logger.info("Added %d nodes to vector store (skipped %d duplicates)",
                    len(deduped), len(nodes) - len(deduped))

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
            where: Optional metadata filter dict like ``{"source": "x.pdf"}``.
                Converted to a LlamaIndex ``MetadataFilters`` (AND of equality
                conditions) before being passed to the vector store.

        Returns:
            VectorStoreQueryResult with nodes, similarities, and ids.
        """
        filters_obj = None
        if where:
            from llama_index.core.vector_stores.types import (
                MetadataFilter, MetadataFilters, FilterOperator,
            )
            filters_obj = MetadataFilters(
                filters=[
                    MetadataFilter(key=k, value=v, operator=FilterOperator.EQ)
                    for k, v in where.items()
                ]
            )
        q = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            filters=filters_obj,
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

    def dedup_by_content(self, dry_run: bool = False) -> int:
        """Remove chunks with duplicate text, keeping one copy of each.

        Use this once to clean legacy pollution from repeated ingestion runs
        that happened before the upload-layer dedup was added. Returns the
        number of chunks removed.

        Args:
            dry_run: If True, only report the count without deleting.
        """
        result = self._collection.get()
        ids: list[str] = result.get("ids", []) or []
        docs: list[str] = result.get("documents", []) or []
        metas: list[dict] = result.get("metadatas", []) or [{} for _ in ids]

        seen: dict[tuple[str, str], str] = {}  # (source, content) -> kept id
        to_delete: list[str] = []
        for id_, doc, meta in zip(ids, docs, metas):
            src = (meta or {}).get("source", "")
            key = (src, doc)
            if key in seen:
                to_delete.append(id_)
            else:
                seen[key] = id_

        if dry_run:
            logger.info("dedup dry-run: %d of %d chunks are duplicates", len(to_delete), len(ids))
            return len(to_delete)

        if to_delete:
            # Delete in batches to avoid payload limits.
            BATCH = 500
            for i in range(0, len(to_delete), BATCH):
                self._collection.delete(ids=to_delete[i:i + BATCH])
            logger.info("Dedup: removed %d duplicate chunks (kept %d unique)",
                        len(to_delete), len(ids) - len(to_delete))
        return len(to_delete)
