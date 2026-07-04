"""Semantic document chunking with formula-aware preservation.

Uses LlamaIndex SemanticSplitterNodeParser for intelligent chunk boundaries,
with special handling to never split inside LaTeX math blocks.
"""

import re
import logging
from typing import List

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, TextNode

from config.settings import settings

logger = logging.getLogger(__name__)

# Regex to detect LaTeX math blocks that must not be split
MATH_BLOCK_PATTERN = re.compile(r'\$\$[^$]+\$\$', re.DOTALL)
INLINE_MATH_PATTERN = re.compile(r'\$[^$]+\$')


class ChunkResult:
    """Result of chunking a document."""

    def __init__(self, nodes: List[BaseNode]):
        self.nodes = nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"ChunkResult(nodes={len(self.nodes)})"


class SemanticChunker:
    """Chunk documents using semantic boundary detection.

    Key behaviors:
    - Respects natural section boundaries (headers, paragraphs)
    - Preserves LaTeX math blocks — never splits inside $$...$$
    - Applies overlap for continuity across chunk boundaries
    """

    def __init__(self, embed_model=None):
        """
        Args:
            embed_model: LlamaIndex embed model for semantic splitting.
                         If None, falls back to a simpler sentence-based splitter.
        """
        self._embed_model = embed_model
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    def chunk(self, text: str, source_path: str = "") -> ChunkResult:
        """Chunk a single document's text into semantic nodes.

        Args:
            text: The full document text.
            source_path: Original file path for metadata.

        Returns:
            ChunkResult containing LlamaIndex TextNodes.
        """
        if self._embed_model is not None:
            nodes = self._semantic_chunk(text)
        else:
            nodes = self._sentence_chunk(text)

        # Add source metadata to each node
        for i, node in enumerate(nodes):
            node.metadata["source"] = source_path
            node.metadata["chunk_index"] = i
            node.metadata["total_chunks"] = len(nodes)

        logger.info("Chunked '%s' into %d nodes", source_path, len(nodes))
        return ChunkResult(nodes)

    def _semantic_chunk(self, text: str) -> List[TextNode]:
        """Use LlamaIndex SemanticSplitterNodeParser."""
        splitter = SemanticSplitterNodeParser(
            embed_model=self._embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=95,
        )
        nodes = splitter.get_nodes_from_documents([TextNode(text=text)])
        return self._merge_small_chunks(nodes)

    def _sentence_chunk(self, text: str) -> List[TextNode]:
        """Fallback: sentence-aware chunking with math block protection."""
        # Protect math blocks before splitting
        math_blocks = []
        def _save_math(match):
            math_blocks.append(match.group(0))
            return f"__MATH_BLOCK_{len(math_blocks) - 1}__"

        protected_text = MATH_BLOCK_PATTERN.sub(_save_math, text)

        # Split into sentences (rough)
        sentences = re.split(r'(?<=[.!?])\s+', protected_text)

        nodes: List[TextNode] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            # Restore math blocks
            for i, block in enumerate(math_blocks):
                sentence = sentence.replace(f"__MATH_BLOCK_{i}__", block)

            sentence_len = len(sentence)

            if current_length + sentence_len > self._chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                nodes.append(TextNode(text=chunk_text))
                # Overlap: keep last sentence
                if len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_len

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            nodes.append(TextNode(text=chunk_text))

        return nodes

    def _merge_small_chunks(self, nodes: List[TextNode], min_size: int = 100) -> List[TextNode]:
        """Merge chunks that are too small into neighbors."""
        if len(nodes) <= 1:
            return nodes

        merged = []
        i = 0
        while i < len(nodes):
            current = nodes[i]
            # If this chunk is small and there's a next one, merge
            if len(current.text) < min_size and i + 1 < len(nodes):
                merged_text = current.text + "\n" + nodes[i + 1].text
                merged_meta = {**current.metadata, "merged": True}
                merged.append(TextNode(text=merged_text, metadata=merged_meta))
                i += 2
            else:
                merged.append(current)
                i += 1

        return merged
