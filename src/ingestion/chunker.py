"""Semantic document chunking with formula-aware preservation.

Uses LlamaIndex SemanticSplitterNodeParser for intelligent chunk boundaries,
with special handling to never split inside LaTeX math blocks. Each chunk is
also annotated with the section heading it belongs to (when detectable), so
the retrieval pipeline and the teaching LLM know which part of the document a
chunk comes from.
"""

import re
import logging
from typing import List, Optional

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, TextNode

from config.settings import settings

logger = logging.getLogger(__name__)

# Regex to detect LaTeX math blocks that must not be split
MATH_BLOCK_PATTERN = re.compile(r'\$\$[^$]+\$\$', re.DOTALL)
INLINE_MATH_PATTERN = re.compile(r'\$[^$]+\$')

# Heading detection — captures the section number and title.
# Order matters: try the strictest patterns first.
_HEADING_PATTERNS = [
    # Markdown headings: "## 2 Evaluation of Word Vector", "# 3.1 Training"
    re.compile(r'^\s{0,3}#{1,6}\s+(\d+(?:\.\d+)*)\s+(.+?)\s*$', re.MULTILINE),
    # Numbered headings at line start: "2 Evaluation of Word Vector", "3.1 Foo"
    # Require the title to start with a capital letter or CJK to reduce false positives.
    re.compile(r'^\s{0,3}(\d+(?:\.\d+)*)\.?\s+([A-Z\u4e00-\u9fff].+?)\s*$', re.MULTILINE),
    # PDF extraction often splits the number and title across two lines:
    #   "2\nEvaluation of Word Vectors\n..."
    # Require the title line to start with a capital letter, be 3-80 chars,
    # and NOT end with a sentence terminator (filters body text false positives).
    re.compile(
        r'(?:^|\n)\s{0,3}(\d+(?:\.\d+)*)\s*\n\s{0,3}([A-Z][^\n.]{2,80}?)\s*(?=\n)',
        re.MULTILINE,
    ),
]

# Page markers the loader inserts, e.g. "--- Page 5 ---"
_PAGE_MARKER_RE = re.compile(r'^\s*-{0,3}\s*Page\s+(\d+)\s*-{0,3}\s*$', re.MULTILINE)


def _is_plausible_section_title(num: str, title: str) -> bool:
    """Heuristic filter to reject false-positive headings.

    Real section titles in academic notes (e.g. "2 Evaluation of Word
    Vectors", "3 Training for Extrinsic Tasks", "1.2 Co-occurrence Matrix")
    are short and Title-Cased. Numbered-list items ("1 Gather fixed size
    context windows of all occurrences of the word") and cover-page lines
    ("1 Course Instructors: Christopher", "2 Authors: Rohit Mundra, ...")
    trail off in lowercase or punctuation and are rejected here.
    """
    title = title.strip().rstrip(".,;:!?)")
    if not (4 <= len(title) <= 80):
        return False
    words = [w for w in title.split() if w]
    if len(words) < 2:
        # Single-word titles like "Summary" are rare in these notes; reject.
        return False
    last_alpha = re.sub(r"[^A-Za-z]", "", words[-1])
    if not last_alpha:
        return False
    # Title-Case heuristic: last word starts with uppercase (Vectors, Matrix,
    # Tasks, Analogies, Classifiers) — accept. Lowercase trailing word
    # (e.g. "word", "the") → likely a list-item, reject.
    if last_alpha[0].isupper():
        return True
    return False


def _find_all_headings(text: str) -> list[str]:
    """Return all section headings found in ``text`` (in document order)."""
    found: list[str] = []
    for pat in _HEADING_PATTERNS:
        for m in pat.finditer(text):
            num, title = m.group(1), m.group(2).strip()
            if _is_plausible_section_title(num, title):
                found.append(f"{num} {title}")
    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for h in found:
        if h not in seen:
            seen.add(h)
            ordered.append(h)
    return ordered


def _annotate_with_section(nodes: List[TextNode]) -> List[TextNode]:
    """Annotate each node with the section heading it belongs to.

    Strategy: walk nodes in document order. For each chunk, look at all
    headings it contains:
      - Tag the chunk with the FIRST heading it contains (the chunk
        primarily started inside that section).
      - Propagate the LAST heading as the running ``current_section`` so
        the next chunk (which may not contain any heading) inherits the
        correct section — this handles the common case where a section
        heading appears mid-chunk and the following chunks continue it.
    The section is recorded in ``metadata["section_heading"]`` and
    prepended to the chunk text, so both the dense embedder and BM25 see
    it, and the teaching LLM can reliably read which section it is reading.
    """
    current_section: Optional[str] = None
    for node in nodes:
        text = node.get_content()
        headings_here = _find_all_headings(text)
        if headings_here:
            own = headings_here[0]            # chunk's primary section
            current_section = headings_here[-1]  # propagate last → next chunk
        else:
            own = current_section             # inherit from previous chunk
        if own:
            node.metadata["section_heading"] = own
            marker = f"[Section: {own}]"
            if not text.lstrip().startswith("[Section:"):
                node.text = f"{marker}\n{text}"
        else:
            node.metadata.setdefault("section_heading", "")
    return nodes


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
    - Annotates each chunk with the section heading it belongs to (Fix2b)
      so retrieval can match "section 2"/"2 Evaluation" and the teaching
      LLM can reliably tell which part of the document it is explaining.
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
            ChunkResult containing LlamaIndex TextNodes, each annotated with
            its source, chunk_index, total_chunks, and section_heading.
        """
        if self._embed_model is not None:
            nodes = self._semantic_chunk(text)
        else:
            nodes = self._sentence_chunk(text)

        # Annotate each chunk with the section it belongs to (Fix2b)
        nodes = _annotate_with_section(nodes)

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
