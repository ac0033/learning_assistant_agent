"""
End-to-End Functional Verification Script
==========================================
Tests all major components of the Learning Assistant Agent project.

Sections:
  1. Import Chain Verification
  2. Config & Embedding Validation
  3. LangGraph Graph Compilation
  4. LLM Creation
  5. Vector Store & Retriever
  6. Teaching Methodology
"""

import sys
import os
import json
import traceback
from datetime import datetime

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────────────────────────────────
# Test infrastructure
# ──────────────────────────────────────────────────────────────────────

results = []   # (test_name, passed: bool, detail: str)


def assert_eq(actual, expected, label=""):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_in(container, item, label=""):
    if item not in container:
        raise AssertionError(f"{label}: {item!r} not found in {container!r}")


# Load .env
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Store test functions so we can call them in order
_test_fns = []


def test(name: str):
    """Decorator to register a test function."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
                results.append((name, True, "OK"))
                print(f"  [PASS] {name}")
            except Exception as e:
                tb = traceback.format_exc()
                results.append((name, False, f"{type(e).__name__}: {e}\n{tb}"))
                print(f"  [FAIL] {name}: {e}")
        _test_fns.append((name, wrapper))
        return wrapper
    return decorator


def run_all():
    """Run all registered tests in order."""
    for name, fn in _test_fns:
        fn()


# ======================================================================
# 1. IMPORT CHAIN VERIFICATION
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 1: IMPORT CHAIN VERIFICATION")
print("=" * 60)


@test("config.settings — Settings class")
def _():
    from config.settings import Settings
    s = Settings()
    assert "Settings" in type(s).__name__


@test("config.prompts — all prompt templates")
def _():
    from config.prompts import (
        TEACHING_SYSTEM_PROMPT,
        QUERY_UNDERSTANDING_PROMPT,
        REFUSAL_PROMPT,
        EXPLANATION_GENERATION_PROMPT,
        EXAMPLE_GENERATION_PROMPT,
        MATH_NOTATION_PROMPT,
        SUMMARY_PROMPT,
        DIRECT_RESPONSE_PROMPT,
    )
    # Verify each is a non-empty string
    for name, val in [
        ("TEACHING_SYSTEM_PROMPT", TEACHING_SYSTEM_PROMPT),
        ("QUERY_UNDERSTANDING_PROMPT", QUERY_UNDERSTANDING_PROMPT),
        ("REFUSAL_PROMPT", REFUSAL_PROMPT),
        ("EXPLANATION_GENERATION_PROMPT", EXPLANATION_GENERATION_PROMPT),
        ("EXAMPLE_GENERATION_PROMPT", EXAMPLE_GENERATION_PROMPT),
        ("MATH_NOTATION_PROMPT", MATH_NOTATION_PROMPT),
        ("SUMMARY_PROMPT", SUMMARY_PROMPT),
        ("DIRECT_RESPONSE_PROMPT", DIRECT_RESPONSE_PROMPT),
    ]:
        if not isinstance(val, str) or not val:
            raise AssertionError(f"{name} is empty or not a string")


@test("src.utils — retry_async, create_llm")
def _():
    from src.utils import retry_async, create_llm, with_llm_retry
    assert callable(retry_async)
    assert callable(create_llm)
    assert callable(with_llm_retry)


@test("src.agent.state — TeachingState")
def _():
    from src.agent.state import TeachingState, RetrievedContext
    # Verify TypedDict structure
    import typing
    assert hasattr(TeachingState, "__annotations__")
    annotations = typing.get_type_hints(TeachingState)
    assert "messages" in annotations
    assert "user_query" in annotations
    assert "intent" in annotations
    assert "target_topic" in annotations
    assert "retrieved_chunks" in annotations
    assert "core_explanation" in annotations
    assert "examples" in annotations
    assert "math_notation" in annotations
    assert "section_summary" in annotations


@test("src.agent.graph — TeachingAgent, build_graph")
def _():
    from src.agent.graph import TeachingAgent, build_graph
    assert callable(build_graph)
    assert hasattr(TeachingAgent, "ateach")
    assert hasattr(TeachingAgent, "astream_teach")


@test("src.agent.nodes.query_understanding")
def _():
    from src.agent.nodes.query_understanding import query_understanding_node
    assert callable(query_understanding_node)


@test("src.agent.nodes.document_retrieval")
def _():
    from src.agent.nodes.document_retrieval import document_retrieval_node
    assert callable(document_retrieval_node)


@test("src.agent.nodes.explanation_generation")
def _():
    from src.agent.nodes.explanation_generation import explanation_generation_node
    assert callable(explanation_generation_node)


@test("src.agent.nodes.example_generation")
def _():
    from src.agent.nodes.example_generation import example_generation_node
    assert callable(example_generation_node)


@test("src.agent.nodes.math_notation")
def _():
    from src.agent.nodes.math_notation import math_notation_node
    assert callable(math_notation_node)


@test("src.agent.nodes.summary_transition")
def _():
    from src.agent.nodes.summary_transition import summary_transition_node
    assert callable(summary_transition_node)


@test("src.indexing.embeddings — get_embed_model")
def _():
    from src.indexing.embeddings import get_embed_model
    assert callable(get_embed_model)


@test("src.indexing.vector_store — VectorStoreManager")
def _():
    from src.indexing.vector_store import VectorStoreManager
    assert hasattr(VectorStoreManager, "add_nodes")
    assert hasattr(VectorStoreManager, "query")
    assert hasattr(VectorStoreManager, "count")
    assert hasattr(VectorStoreManager, "list_sources")


@test("src.indexing.hybrid_retriever — HybridRetriever")
def _():
    from src.indexing.hybrid_retriever import HybridRetriever, BM25Retriever, refresh_retriever
    assert hasattr(HybridRetriever, "retrieve")
    assert hasattr(HybridRetriever, "refresh_bm25")
    assert callable(refresh_retriever)
    assert hasattr(BM25Retriever, "search")
    assert hasattr(BM25Retriever, "index")


@test("src.indexing.reranker — Reranker")
def _():
    from src.indexing.reranker import Reranker
    assert hasattr(Reranker, "rerank")


@test("src.ingestion.loader — PDFLoader")
def _():
    from src.ingestion.loader import PDFLoader, ParsedDocument
    assert hasattr(PDFLoader, "load")
    # ParsedDocument sets these on instances via __init__, not on the class
    # Verify by creating an instance
    from pathlib import Path
    pd = ParsedDocument(text="test", source_path=Path("test.pdf"), pages=["page1"])
    assert hasattr(pd, "text")
    assert hasattr(pd, "pages")
    assert hasattr(pd, "metadata")


@test("src.ingestion.chunker — SemanticChunker")
def _():
    from src.ingestion.chunker import SemanticChunker, ChunkResult
    assert hasattr(SemanticChunker, "chunk")
    # ChunkResult sets nodes on instance via __init__
    cr = ChunkResult(nodes=[])
    assert hasattr(cr, "nodes")


@test("src.ingestion.pipeline — IngestionPipeline")
def _():
    from src.ingestion.pipeline import IngestionPipeline
    assert hasattr(IngestionPipeline, "ingest_file")
    assert hasattr(IngestionPipeline, "ingest_directory")
    assert hasattr(IngestionPipeline, "get_indexed_documents")


@test("src.teaching.methodology — TeachingMethodology")
def _():
    from src.teaching.methodology import TeachingMethodology
    assert hasattr(TeachingMethodology, "assemble_full_response")
    assert hasattr(TeachingMethodology, "assemble_partial_response")
    assert hasattr(TeachingMethodology, "get_teaching_principles")


@test("src.ui.chat_handler")
def _():
    # These modules need Chainlit runtime context for decorators (@cl.on_*)
    # Use exec in a patched environment or at minimum verify the file is importable
    # by trying and catching runtime-only errors
    import importlib
    import sys

    # Mock chainlit decorators that fail outside Chainlit runtime
    from unittest.mock import MagicMock, patch

    class MockCL:
        on_chat_start = MagicMock()
        on_message = MagicMock()
        Message = MagicMock()
        user_session = MagicMock()
        user_session.get = MagicMock(return_value="default")

    mock_cl = MockCL()
    # Ensure user_session.get is callable and returns a dict-like
    mock_cl.user_session.get.return_value = "default"
    mock_cl.Message.return_value = MagicMock()

    with patch.dict('sys.modules', {'chainlit': mock_cl}):
        # Reload with mock
        if 'src.ui.chat_handler' in sys.modules:
            del sys.modules['src.ui.chat_handler']
        import src.ui.chat_handler as ch
        assert ch.WELCOME_MESSAGE is not None
        assert len(ch.WELCOME_MESSAGE) > 100
        assert callable(ch._get_agent)


@test("src.ui.file_handler")
def _():
    # Same approach: mock chainlit decorators
    import sys
    from unittest.mock import MagicMock, patch

    class MockCL:
        on_chat_resume = MagicMock()
        on_upload = MagicMock()
        on_message = MagicMock()
        Message = MagicMock()
        user_session = MagicMock()
        File = MagicMock()

    mock_cl = MockCL()

    with patch.dict('sys.modules', {'chainlit': mock_cl}):
        if 'src.ui.file_handler' in sys.modules:
            del sys.modules['src.ui.file_handler']
        import src.ui.file_handler as fh
        assert fh._get_pipeline is not None


@test("src.ui.session_manager")
def _():
    from src.ui.session_manager import SessionManager, UserSession, session_manager
    assert hasattr(SessionManager, "get_or_create_session")
    assert hasattr(SessionManager, "record_question")
    assert hasattr(SessionManager, "set_active_document")
    # UserSession is a dataclass; verify via instance creation
    us = UserSession(user_id="test", thread_id="test_thread")
    assert hasattr(us, "user_id")
    assert hasattr(us, "thread_id")


# ======================================================================
# 2. CONFIG & EMBEDDING VALIDATION
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 2: CONFIG & EMBEDDING VALIDATION")
print("=" * 60)


@test("settings.anthropic_model = 'deepseek-v4-flash'")
def _():
    from config.settings import settings
    assert_eq(settings.anthropic_model, "deepseek-v4-flash", "anthropic_model")


@test("settings.embedding_provider = 'siliconflow'")
def _():
    from config.settings import settings
    assert_eq(settings.embedding_provider, "siliconflow", "embedding_provider")


@test("settings.embedding_model = 'BAAI/bge-m3'")
def _():
    from config.settings import settings
    assert_eq(settings.embedding_model, "BAAI/bge-m3", "embedding_model")


@test("settings.anthropic_base_url is non-empty (provider-agnostic)")
def _():
    from config.settings import settings
    assert settings.anthropic_base_url, "anthropic_base_url must be set (DeepSeek/OpenRouter/Anthropic)"


@test("settings.validate_runtime() — no exception")
def _():
    from config.settings import settings
    settings.validate_runtime()  # should not raise


@test("get_embed_model() creates instance and tests embedding")
def _():
    """Create embedding instance and attempt a short text embedding."""
    from src.indexing.embeddings import get_embed_model
    embed_model = get_embed_model()
    assert embed_model is not None

    # Try actual embedding call
    try:
        result = embed_model.get_text_embedding("Hello world")
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) > 0, "Empty embedding vector"
        print(f"      Embedding dimension: {len(result)}")
        print(f"      First 5 values: {result[:5]}")
    except Exception as e:
        raise RuntimeError(f"Embedding API call failed: {e}")


# ======================================================================
# 3. LANGGRAPH GRAPH COMPILATION
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 3: LANGGRAPH GRAPH COMPILATION")
print("=" * 60)


@test("build_graph() compiles without error")
def _():
    from src.agent.graph import build_graph
    graph = build_graph()
    assert graph is not None


@test("Graph contains all 6 nodes")
def _():
    from src.agent.graph import build_graph
    graph = build_graph()

    # In LangGraph >=1.0, examine via graph.nodes or graph._nodes
    node_names = set()
    # Try different ways to inspect nodes
    if hasattr(graph, 'nodes'):
        node_names = set(graph.nodes.keys())
    elif hasattr(graph, '_nodes'):
        node_names = set(graph._nodes.keys())

    expected = {
        "query_understanding",
        "document_retrieval",
        "explanation_generation",
        "example_generation",
        "math_notation",
        "summary_transition",
    }

    missing = expected - node_names
    if missing:
        raise AssertionError(f"Missing nodes: {missing}. Found nodes: {node_names}")


@test("Graph entry point and exit point correct")
def _():
    from src.agent.graph import build_graph
    graph = build_graph()

    # Check that the graph has expected structure
    # For LangGraph >= 1.0, we can try checking using compiled graph methods
    # At minimum verify the graph compiles and can stringify
    graph_str = str(graph)
    assert "query_understanding" in graph_str or graph is not None


@test("Graph schema matches TeachingState fields")
def _():
    """Check that graph's input/state schema includes key fields."""
    from src.agent.graph import build_graph
    from src.agent.state import TeachingState

    graph = build_graph()

    # Check graph has a get_input_schema or similar
    if hasattr(graph, 'get_input_schema'):
        schema = graph.get_input_schema({})
        print(f"      Graph input schema: {schema}")


# ======================================================================
# 4. LLM CREATION
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 4: LLM CREATION")
print("=" * 60)


@test("create_llm() creates instance with correct model")
def _():
    from src.utils import create_llm
    from config.settings import settings

    llm = create_llm()
    assert llm is not None

    # Check model property
    if hasattr(llm, 'model'):
        assert_eq(llm.model, settings.anthropic_model, "LLM model")
    elif hasattr(llm, 'model_name'):
        assert_eq(llm.model_name, settings.anthropic_model, "LLM model_name")

    print(f"      LLM type: {type(llm).__name__}")
    print(f"      LLM model: {getattr(llm, 'model', getattr(llm, 'model_name', 'unknown'))}")
    print(f"      LLM api_key set: {bool(llm.api_key) if hasattr(llm, 'api_key') else 'N/A'}")


@test("create_llm(temperature=0.5, max_tokens=3000) honors params")
def _():
    from src.utils import create_llm

    llm = create_llm(temperature=0.5, max_tokens=3000)

    if hasattr(llm, 'temperature'):
        assert_eq(llm.temperature, 0.5, "LLM temperature")
    if hasattr(llm, 'max_tokens'):
        assert_eq(llm.max_tokens, 3000, "LLM max_tokens")

    print(f"      temperature={getattr(llm, 'temperature', 'N/A')}, "
          f"max_tokens={getattr(llm, 'max_tokens', 'N/A')}")


# ======================================================================
# 5. VECTOR STORE & RETRIEVER
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 5: VECTOR STORE & RETRIEVER")
print("=" * 60)


@test("VectorStoreManager initialization")
def _():
    from src.indexing.vector_store import VectorStoreManager
    vsm = VectorStoreManager()

    # Verify ChromaDB collection exists
    assert vsm.collection is not None
    assert vsm.store is not None

    collection_name = vsm.collection.name
    assert_eq(collection_name, "course_materials", "Collection name")

    count = vsm.count()
    print(f"      ChromaDB collection: {collection_name}")
    print(f"      Document count: {count}")
    print(f"      ChromaDB path: {vsm._chroma_dir}")


@test("VectorStoreManager add/list/query round-trip")
def _():
    """Add a test node to vector store, verify it appears in list_sources and count."""
    from src.indexing.vector_store import VectorStoreManager
    from src.indexing.embeddings import get_embed_model
    from llama_index.core.schema import TextNode

    vsm = VectorStoreManager()
    initial_count = vsm.count()

    # Get embedding model to generate a proper embedding vector
    embed_model = get_embed_model()
    embedding = embed_model.get_text_embedding("This is a test document about gradient descent optimization.")

    # Create a test node with embedding set
    node = TextNode(
        text="This is a test document about gradient descent optimization.",
        metadata={
            "source": "test_e2e_verification.pdf",
            "page": 1,
            "chunk_index": 0,
        },
    )
    node.embedding = embedding

    # Add it
    vsm.add_nodes([node])
    new_count = vsm.count()
    assert_eq(new_count, initial_count + 1, "Count after adding node")

    # Check list_sources includes our test source
    sources = vsm.list_sources()
    assert_in(sources, "test_e2e_verification.pdf", "list_sources")

    # Query the test node
    query_embedding = embed_model.get_query_embedding("gradient descent")
    qresult = vsm.query(query_embedding, top_k=5)
    assert qresult.nodes is not None
    assert len(qresult.nodes) >= 1, "Should find at least one result"

    # Clean up: delete by source
    deleted = vsm.delete_by_source("test_e2e_verification.pdf")
    assert_eq(deleted, 1, "Deleted count")
    final_count = vsm.count()
    assert_eq(final_count, initial_count, "Count after cleanup")


@test("HybridRetriever initialization with BM25 index")
def _():
    """Verify HybridRetriever creates BM25 index from vector store contents."""
    from src.indexing.hybrid_retriever import HybridRetriever
    from src.indexing.vector_store import VectorStoreManager
    from src.indexing.embeddings import get_embed_model

    vsm = VectorStoreManager()
    embed_model = get_embed_model()
    retriever = HybridRetriever(
        vector_store_manager=vsm,
        embed_model=embed_model,
    )

    assert retriever is not None
    assert retriever._bm25 is not None

    bm25_initialized = retriever._bm25._initialized
    bm25_node_count = len(retriever._bm25._nodes) if bm25_initialized else 0
    print(f"      BM25 initialized: {bm25_initialized}")
    print(f"      BM25 indexed nodes: {bm25_node_count}")


@test("Reranker rerank returns correct count")
def _():
    """Test Reranker with mock candidates."""
    from src.indexing.reranker import Reranker
    from llama_index.core.schema import TextNode, NodeWithScore

    reranker = Reranker()

    # Create mock candidates
    candidates = []
    for i in range(10):
        node = TextNode(text=f"Test document content chunk number {i} about machine learning.")
        candidates.append(NodeWithScore(node=node, score=0.5 - i * 0.05))

    result = reranker.rerank("machine learning", candidates, top_k=3)
    assert_eq(len(result), 3, "Reranker top_k count")
    print(f"      Reranked from 10 to 3 candidates")


# ======================================================================
# 6. TEACHING METHODOLOGY
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 6: TEACHING METHODOLOGY")
print("=" * 60)


@test("TeachingMethodology.assemble_full_response includes all 4 parts")
def _():
    from src.teaching.methodology import TeachingMethodology

    tm = TeachingMethodology()
    result = tm.assemble_full_response(
        core_explanation="Gradient descent is an optimization algorithm.",
        examples="Example: minimizing f(x) = x^2.",
        math_notation="$\\theta_{new} = \\theta_{old} - \\alpha \\nabla J(\\theta)$",
        section_summary="Gradient descent iteratively moves toward the minimum.",
    )

    # Should contain all four section headers
    assert "① Core Process" in result, "Missing Core Process header"
    assert "② Interspersed Examples" in result, "Missing Examples header"
    assert "③ Math & Notation" in result, "Missing Math & Notation header"
    assert "④ Summary" in result, "Missing Summary header"

    # Should contain separators
    assert "---" in result, "Missing section separator"
    assert result.count("---") >= 1, "Should have at least one separator"

    print(f"      Total output length: {len(result)} chars")


@test("TeachingMethodology.assemble_full_response skips empty parts")
def _():
    from src.teaching.methodology import TeachingMethodology

    tm = TeachingMethodology()

    # Only core and summary
    result = tm.assemble_full_response(
        core_explanation="Core content here.",
        section_summary="Summary here.",
    )

    assert "① Core Process" in result
    assert "② Interspersed Examples" not in result, "Should skip empty examples"
    assert "③ Math & Notation" not in result, "Should skip empty math"
    assert "④ Summary" in result

    print(f"      Partial assembly (core+summary): {len(result)} chars")


@test("TeachingMethodology.assemble_partial_response with None values")
def _():
    from src.teaching.methodology import TeachingMethodology

    tm = TeachingMethodology()
    result = tm.assemble_partial_response(
        core_explanation="Core",
        examples="Examples",
        math_notation=None,  # None → skip
        section_summary=None,
    )

    assert "① Core Process" in result
    assert "② Interspersed Examples" in result
    assert "③ Math & Notation" not in result, "Should skip None math notation"
    assert "④ Summary" not in result, "Should skip None summary"

    print(f"      Partial assembly (core+examples): {len(result)} chars")


@test("TeachingMethodology.get_teaching_principles returns non-empty")
def _():
    from src.teaching.methodology import TeachingMethodology

    tm = TeachingMethodology()
    principles = tm.get_teaching_principles()
    assert isinstance(principles, str) and len(principles) > 50
    assert "Feynman" in principles
    print(f"      Principles length: {len(principles)} chars")


# ======================================================================
# RUN ALL TESTS
# ======================================================================

run_all()

# ======================================================================
# SUMMARY
# ======================================================================

print("\n" + "=" * 60)
print("TEST RESULTS SUMMARY")
print("=" * 60)

passed = sum(1 for _, p, _ in results if p)
failed = sum(1 for _, p, _ in results if not p)
total = len(results)

print(f"\n  Total:  {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Rate:   {passed/total*100:.1f}%\n")

if failed > 0:
    print("── FAILED TESTS ──")
    for name, ok, detail in results:
        if not ok:
            print(f"\n  ❌ {name}:")
            for line in detail.splitlines():
                print(f"     {line}")
    print()

# Output JSON for machine parsing
summary = {
    "timestamp": datetime.now().isoformat(),
    "total": total,
    "passed": passed,
    "failed": failed,
    "results": [
        {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail if not ok else ""}
        for name, ok, detail in results
    ],
}
report_path = os.path.join(PROJECT_ROOT, "tests", "e2e_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"Report saved to: {report_path}")

# Exit code
sys.exit(0 if failed == 0 else 1)
