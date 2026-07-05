"""Hybrid Retriever: Dense (vector) + Sparse (BM25) with RRF fusion.

This is the core retrieval engine. It combines:
- Dense retrieval: semantic similarity via embeddings
- Sparse retrieval: exact term matching via BM25 (critical for LaTeX symbols)
- RRF fusion: combines rankings from both retrievers
"""

import logging
from typing import List, Optional, Tuple

from llama_index.core.schema import NodeWithScore, QueryBundle

from .vector_store import VectorStoreManager
from .reranker import Reranker
from config.settings import settings

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Simple in-memory BM25 retriever using rank-bm25 or custom implementation."""

    def __init__(self):
        self._corpus: List[str] = []
        self._nodes: List = []
        self._tokenized_corpus: List[List[str]] = []
        self._initialized = False

    def index(self, nodes: List) -> None:
        """Build BM25 index from nodes."""
        self._nodes = nodes
        self._corpus = [node.get_content() for node in nodes]
        self._tokenized_corpus = [self._tokenize(text) for text in self._corpus]
        self._initialized = True
        logger.info("BM25 index built with %d documents", len(nodes))

    def is_initialized(self) -> bool:
        """Check if the BM25 index has been built."""
        return self._initialized and len(self._nodes) > 0

    @property
    def node_count(self) -> int:
        """Number of indexed nodes."""
        return len(self._nodes)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        import re
        # Keep LaTeX commands and math symbols as tokens
        tokens = re.findall(r'\\[a-zA-Z]+|[$][^$]+[$]|\w+|[^\s\w]', text.lower())
        return [t for t in tokens if t.strip()]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search BM25 index and return (doc_index, score) pairs."""
        if not self._initialized:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        avgdl = sum(len(doc) for doc in self._tokenized_corpus) / max(len(self._tokenized_corpus), 1)
        N = len(self._tokenized_corpus)
        k1, b = 1.5, 0.75

        # Compute IDF
        df = {}
        for token in set(query_tokens):
            df[token] = sum(1 for doc in self._tokenized_corpus if token in doc)

        for idx, doc_tokens in enumerate(self._tokenized_corpus):
            score = 0.0
            doc_len = len(doc_tokens)
            for token in query_tokens:
                if token not in df or df[token] == 0:
                    continue
                tf = doc_tokens.count(token)
                idf = ((N - df[token] + 0.5) / (df[token] + 0.5)) + 1
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
                score += idf * (numerator / max(denominator, 0.001))
            scores.append((idx, score))

        # Sort by score descending, return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    """Hybrid retrieval combining dense (vector) and sparse (BM25) search.

    Fusion strategy: Reciprocal Rank Fusion (RRF) with k=60.

    Pipeline:
        1. Dense search → top-K_candidate from vector store
        2. Sparse search → top-K_candidate from BM25
        3. RRF fusion → merged ranking
        4. Reranker → top-K final results
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        embed_model=None,
    ):
        self._vsm = vector_store_manager
        self._embed_model = embed_model
        self._bm25 = BM25Retriever()
        self._reranker = Reranker()
        self._candidate_k = settings.retrieval_candidate_k
        self._final_k = settings.retrieval_top_k

        # One-time cleanup of legacy duplicate chunks (pre-dedup-feature
        # re-ingestion leaves 8-9x copies of the same content). Runs once
        # at retriever construction; subsequent uploads are guarded by
        # VectorStoreManager.add_nodes' own dedup check, so this is a no-op
        # after the first clean run.
        try:
            removed = self._vsm.dedup_by_content()
            if removed:
                logger.info("One-time dedup removed %d legacy duplicate chunks", removed)
        except Exception as e:
            logger.warning("Startup dedup_by_content failed (continuing): %s", e)

        # Build BM25 index from existing store (post-cleanup, so it's clean)
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from all nodes in the vector store."""
        collection = self._vsm.collection
        result = collection.get()
        if result.get("documents"):
            # Reconstruct nodes for BM25
            from llama_index.core.schema import TextNode
            nodes = []
            for i, doc in enumerate(result["documents"]):
                meta = result["metadatas"][i] if result.get("metadatas") else {}
                nodes.append(TextNode(text=doc, metadata=meta))
            self._bm25.index(nodes)

    def retrieve(self, query: str, source_filter: Optional[str] = None,
                final_k: Optional[int] = None) -> List[NodeWithScore]:
        """Execute hybrid retrieval for a query.

        Args:
            query: The search query string.
            source_filter: If given, restrict retrieval to chunks whose
                ``metadata['source']`` equals this filename. Used when the
                student explicitly names a PDF in their query.
            final_k: Override the default final-k (reranker output size).
                Pass a larger value when the caller wants a wider candidate
                pool to apply its own diversity post-selection (e.g. ensuring
                chunks from multiple sections are represented).

        Returns:
            List of NodeWithScore, sorted by relevance (highest first).
        """
        if self._bm25._initialized is False or len(self._bm25._nodes) == 0:
            logger.warning("BM25 index is empty—returning empty results")
            return []

        dense_where = {"source": source_filter} if source_filter else None
        out_k = final_k if final_k is not None else self._final_k

        # Step 1: Dense retrieval (with optional metadata filter)
        dense_results = self._dense_search(query, top_k=self._candidate_k, where=dense_where)

        # Step 2: Sparse retrieval. BM25 keeps an in-memory node list, so we
        # post-filter its results to the requested source (cheaper than
        # rebuilding the index per source).
        sparse_results = self._sparse_search(query, top_k=self._candidate_k, source_filter=source_filter)

        # Step 3: RRF fusion
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results, k=60)

        # Step 4: Rerank
        final = self._reranker.rerank(query, fused, top_k=out_k)

        logger.info(
            "Hybrid retrieval (source=%s, final_k=%d): dense=%d, sparse=%d, fused=%d, final=%d",
            source_filter or "ALL", out_k, len(dense_results), len(sparse_results), len(fused), len(final)
        )

        return final

    def _dense_search(self, query: str, top_k: int, where: Optional[dict] = None) -> List[NodeWithScore]:
        """Vector similarity search. Errors are isolated — returns [] on failure."""
        if self._embed_model is None:
            return []

        try:
            query_embedding = self._embed_model.get_query_embedding(query)
            result = self._vsm.query(query_embedding, top_k=top_k, where=where)
        except Exception as e:
            logger.warning("Dense search failed: %s. Returning empty.", e)
            return []

        nodes = []
        for node, similarity in zip(result.nodes or [], result.similarities or []):
            if node is not None:
                nodes.append(NodeWithScore(node=node, score=similarity or 0.0))
        return nodes

    def _sparse_search(self, query: str, top_k: int, source_filter: Optional[str] = None) -> List[NodeWithScore]:
        """BM25 exact/partial match search. Errors are isolated — returns [] on failure."""
        try:
            results = self._bm25.search(query, top_k=top_k)
        except Exception as e:
            logger.warning("Sparse search failed: %s. Returning empty.", e)
            return []

        nodes = []
        for idx, score in results:
            if score > 0:
                node = self._bm25._nodes[idx]
                if source_filter and node.metadata.get("source") != source_filter:
                    continue
                nodes.append(NodeWithScore(node=node, score=score))
        return nodes

    def _reciprocal_rank_fusion(
        self,
        dense: List[NodeWithScore],
        sparse: List[NodeWithScore],
        k: int = 60,
    ) -> List[NodeWithScore]:
        """Merge dense and sparse rankings using RRF.

        RRF score = sum(1 / (k + rank_i)) for each retriever i.
        """
        scores = {}

        # Dense rankings (rank 0 = best)
        for rank, item in enumerate(dense):
            node_id = item.node.node_id
            rrf_score = 1.0 / (k + rank)
            scores[node_id] = scores.get(node_id, 0.0) + rrf_score

        # Sparse rankings
        for rank, item in enumerate(sparse):
            node_id = item.node.node_id
            rrf_score = 1.0 / (k + rank)
            scores[node_id] = scores.get(node_id, 0.0) + rrf_score

        # Build merged result
        # Collect unique nodes
        node_map = {}
        for item in dense + sparse:
            node_map[item.node.node_id] = item.node

        merged = []
        for node_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if node_id in node_map:
                merged.append(NodeWithScore(node=node_map[node_id], score=rrf_score))

        return merged

    def refresh_bm25(self):
        """Rebuild BM25 index (call after adding new documents)."""
        self._build_bm25_index()


# Module-level retriever singleton (managed by document_retrieval.py)
_retriever: HybridRetriever | None = None


def set_retriever(retriever: HybridRetriever) -> None:
    """Public API: set the module-level retriever singleton.

    Called by document_retrieval.py when the retriever is first created,
    so that refresh_retriever() can find and refresh it.
    """
    global _retriever
    _retriever = retriever
    logging.getLogger(__name__).debug("Module-level retriever set")


def refresh_retriever() -> None:
    """Public API: refresh the global retriever's BM25 index.

    Call after ingesting new documents to rebuild the sparse index.
    If retriever hasn't been initialized yet, this is a no-op.
    """
    global _retriever
    if _retriever is not None:
        _retriever.refresh_bm25()
        logging.getLogger(__name__).info("Retriever BM25 index refreshed")
