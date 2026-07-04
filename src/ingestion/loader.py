"""PDF document loader with multi-strategy parsing.

Strategy chain: PyMuPDF (primary, reliable) → fallback notes.
MinerU is noted as an optional enhancement for LaTeX-heavy PDFs on Linux/macOS.
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ParsedDocument:
    """Result of parsing a single PDF document."""

    def __init__(
        self,
        text: str,
        source_path: Path,
        pages: List[str],
        metadata: Optional[dict] = None,
    ):
        self.text = text
        self.source_path = source_path
        self.pages = pages  # page-by-page text
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"ParsedDocument(source={self.source_path.name}, pages={len(self.pages)})"


class PDFLoader:
    """Load and parse PDF documents using the best available strategy.

    Strategy chain:
    1. PyMuPDF (fitz) — fast, reliable, good text extraction on all platforms
    2. (Optional) MinerU — best LaTeX extraction, may not be available on Windows
    """

    def __init__(self, prefer_mineru: bool = False):
        self._prefer_mineru = prefer_mineru
        self._mineru_available = self._check_mineru()

    @staticmethod
    def _check_mineru() -> bool:
        """Check if MinerU (magic_pdf) is importable."""
        try:
            import magic_pdf  # noqa: F401
            return True
        except ImportError:
            logger.info("MinerU not available, will use PyMuPDF for PDF parsing.")
            return False

    def load(self, file_path: Path) -> ParsedDocument:
        """Load and parse a single PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument with extracted text and metadata.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf file, got: {file_path.suffix}")

        # Try MinerU first if preferred and available
        if self._prefer_mineru and self._mineru_available:
            try:
                return self._load_with_mineru(file_path)
            except Exception as e:
                logger.warning(f"MinerU failed ({e}), falling back to PyMuPDF")
                return self._load_with_pymupdf(file_path)

        # Default: PyMuPDF (reliable on all platforms)
        return self._load_with_pymupdf(file_path)

    def _load_with_pymupdf(self, file_path: Path) -> ParsedDocument:
        """Extract text using PyMuPDF (fitz)."""
        import fitz  # pymupdf

        doc = fitz.open(str(file_path))
        pages: List[str] = []
        full_text_parts: List[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")  # plain text, preserves layout order
            pages.append(page_text)
            full_text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

        doc.close()

        full_text = "\n\n".join(full_text_parts)

        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "num_pages": len(pages),
            "parser": "pymupdf",
        }

        logger.info(
            "Loaded %s: %d pages, %d chars (pymupdf)",
            file_path.name, len(pages), len(full_text)
        )

        return ParsedDocument(
            text=full_text,
            source_path=file_path,
            pages=pages,
            metadata=metadata,
        )

    def _load_with_mineru(self, file_path: Path) -> ParsedDocument:
        """Extract text using MinerU (magic_pdf) for best LaTeX preservation.

        This is the optimal path for math-heavy PDFs. MinerU outputs Markdown
        with LaTeX equations preserved.
        """
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

        # MinerU writes intermediate files to a working directory
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Prepare I/O handlers for MinerU
            local_image_dir = tmp_path / "images"
            local_md_dir = tmp_path / "md"
            local_image_dir.mkdir(exist_ok=True)
            local_md_dir.mkdir(exist_ok=True)

            image_writer = DiskReaderWriter(str(local_image_dir))
            md_writer = DiskReaderWriter(str(local_md_dir))

            # Read PDF bytes
            pdf_bytes = file_path.read_bytes()

            # Run MinerU pipeline
            jso_useful_key = {"_pdf_type": "", "model_list": []}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
            pipe.pipe_classify()
            pipe.pipe_parse()
            pipe.pipe_mk_uni_markdown(md_writer)

            content_list = pipe.pipe_mk_uni_markdown(md_writer)

            # Extract markdown text
            md_content = pipe.doc_md
            if not md_content and hasattr(pipe, '_md_content'):
                md_content = pipe._md_content

            # Build page-by-page content
            pages: List[str] = []
            if isinstance(content_list, list):
                pages = [str(item) for item in content_list]

            full_text = md_content if md_content else "\n".join(pages)

            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "num_pages": len(pages) if pages else -1,
                "parser": "mineru",
            }

            logger.info(
                "Loaded %s: %d pages, %d chars (MinerU)",
                file_path.name, len(pages), len(full_text)
            )

            return ParsedDocument(
                text=full_text,
                source_path=file_path,
                pages=pages,
                metadata=metadata,
            )
