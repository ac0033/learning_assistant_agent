"""End-to-end verification pipeline (CLAUDE.md Testing Checklist step 4).

Covers the five-stage journey a student takes so the agent flow can't
silently break between commits:

    S1  Open a new conversation      — history record created
    S2  Upload a PDF + re-upload     — chunks indexed, dedup, registered
    S3  Render a four-part teaching  — duplicate section headers stripped
    S4  Route after examples         — Math step forced for mathy topics
                                       even when the LLM sets needs_math=False
    S5  Retrieval with named file    — source filter + cross-source diversity

Designed to run **without** any LLM / embedding API call (it stubs the
ingestion + LLM-shape inputs where needed) so it stays cheap (seconds),
deterministic, and runnable in CI. Any FAIL exits non-zero.

Usage:
    uv run python tests/e2e_verify.py

Exit code 0 = all five stages PASS, 1 = at least one FAIL.
"""
import os
import re
import sys
import shutil
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

PASS = 0
FAIL = 0
FAILURES: list[str] = []


def check(stage: str, name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{stage} / {name}")
    print(f"[{tag}] {stage} {name}  {detail}")


# ---------------------------------------------------------------------------
# S1 — Open a new conversation: history record is created and listed
# ---------------------------------------------------------------------------
def stage_open_conversation() -> None:
    from src.ui.history_manager import HistoryManager
    # Use a sandbox under data/_e2e so we don't touch real user data
    tmp = settings.data_dir / "_e2e"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    import src.ui.history_manager as hm
    hm._history_dir = lambda: tmp  # type: ignore[attr-defined]
    hm._conv_path = lambda: tmp / "conversations.json"  # type: ignore[attr-defined]
    hm._registry_path = lambda: tmp / "file_registry.json"  # type: ignore[attr-defined]

    mgr = HistoryManager()
    conv = mgr.create_conversation("e2e_thread_1", user_id="e2e_user")
    check("S1", "create", conv["thread_id"] == "e2e_thread_1",
          f"thread={conv['thread_id']}")
    listed = mgr.list_conversations()
    check("S1", "list", any(c["thread_id"] == "e2e_thread_1" for c in listed),
          f"listed {len(listed)} conversations")

    # /history usable before any upload (this was the AskFileMessage-block bug)
    check("S1", "pre-upload-ok", len(listed) >= 1,
          "history view returns data even with zero uploads")


# ---------------------------------------------------------------------------
# S2 — Upload a PDF; re-upload of the same file is deduped at the registry level
# ---------------------------------------------------------------------------
def stage_upload_dedup() -> None:
    from src.ui.history_manager import compute_file_hash, history_manager
    tmp = settings.data_dir / "_e2e"
    fake_pdf = tmp / "lecture_e2e.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 e2e fake content for upload dedup test " * 50)

    h = compute_file_hash(fake_pdf)
    check("S2", "sha256", len(h) == 64, f"hash prefix {h[:12]}…")

    history_manager.register_file("e2e_thread_1", fake_pdf.name, h, chunks=8)
    history_manager.register_file_global(h, fake_pdf.name, chunks=8, thread_id="e2e_thread_1")
    conv = history_manager.get_conversation("e2e_thread_1")
    files = conv.get("files", [])
    check("S2", "register", len(files) == 1, f"files={files}")

    # Second registration of the same hash must NOT create a second entry
    history_manager.register_file("e2e_thread_1", "lecture_e2e_renamed.pdf", h, chunks=8)
    conv = history_manager.get_conversation("e2e_thread_1")
    check("S2", "dedup", len(conv.get("files", [])) == 1,
          f"after re-upload files={conv['files']}")

    # Registry protects uploads across threads
    known = history_manager.is_file_known(h)
    check("S2", "registry-hit", known is not None and known["name"] == fake_pdf.name,
          f"registry says: {known}")


# ---------------------------------------------------------------------------
# S3 — Render a four-part teaching: duplicate section headers stripped
# ---------------------------------------------------------------------------
def stage_render_no_dup_header() -> None:
    from src.teaching.methodology import TeachingMethodology

    # Mirror the user-reported shape exactly: LLM emits a preamble line AND a
    # bare header line mid-output.
    core = "Nucleus concept: a co-occurrence matrix $X_{ij}$ records word pairs."
    examples = (
        "好的，现在我们来写 **② Interspersed Examples（示例展开）**。我会从最简单的例子开始。\n"
        "② Interspersed Examples（示例展开）\n"
        "Example 1：情感分类中的固定 vs. 重训词向量\n"
        "Finally we touch on cross-entropy. 接下来可进入 ③ Math & Notation（数学与符号）。"
    )
    math_t = (
        "$\\beta$ derivation; gradient $\\nabla_\\theta J(\\theta)$. The ③ Math & Notation\n"
        "section should not have its name duplicated."
    )
    summary = (
        "Part ④: **Summary（总结提炼）**\n"
        "④ Summary（总结提炼）\n"
        "Key takeaway: ①②③④ markers stay exactly once."
    )

    full = TeachingMethodology.assemble_full_response(
        core_explanation=core,
        examples=examples,
        math_notation=math_t,
        section_summary=summary,
    )

    # Each canonical header should appear exactly once
    for canon in ["## ① Core Process", "## ② Interspersed Examples",
                  "## ③ Math & Notation", "## ④ Summary"]:
        check("S3", f"canon-{canon}", full.count(canon) == 1,
              f"appears {full.count(canon)}x")

    # LLM-echoed bare-header variants must NOT survive (the bug from this round)
    check("S3", "no-bare-example",
          "② Interspersed Examples（示例展开）\n" not in full,
          "bare ② header line stripped")
    check("S3", "no-bare-summary",
          "④ Summary（总结提炼）\n" not in full,
          "bare ④ header line stripped")

    # LLM preamble voice line should be kept (it carries transitional context)
    check("S3", "keep-preamble",
          "好的，现在我们来写" in full,
          "preamble line preserved (not clobbered as header)")

    # Example/Step body content must be kept (header-dedup must not eat it)
    check("S3", "keep-example-body",
          "Example 1：情感分类" in full,
          "Example 1 body line preserved")

    # All-in-one: at most one occurrence of each circled-marker English title
    # (the bare header would otherwise add a second 'Interspersed Examples' word)
    times = len(re.findall(r"Interspersed Examples", full))
    check("S3", "title-once", times <= 2,
          f"'Interspersed Examples' appears {times}x (1 canon + ≤1 preamble mention)")


# ---------------------------------------------------------------------------
# S4 — Route after examples: Math step forced for mathy topics even when
#       the LLM sets needs_math=False (the reported bug)
# ---------------------------------------------------------------------------
def stage_route_math_forced() -> None:
    from src.agent.graph import _route_after_explanation, _route_after_example

    # Case A: learn_new + mathy topic, but LLM gave needs_math=False
    state_mathy = {
        "intent": "learn_new", "needs_math": False,
        "needs_example": True, "target_topic": "softmax loss gradient",
    }
    # First hop: explanation → example
    route_ex = _route_after_explanation(state_mathy)
    check("S4", "ex→example", route_ex == "example_generation",
          f"got {route_ex}")
    # Second hop: example → math (must be math, not summary, for learn_new+mathy)
    state_after_example = {**state_mathy, "needs_example": True}
    route = _route_after_example(state_after_example)
    check("S4", "force-math", route == "math_notation",
          f"got {route} (expected math_notation so ③ is never skipped)")

    # But query_understanding's own guard sets needs_math=True for mathy topics,
    # which is the real path. Simulate that too.
    state_overridden = {
        "intent": "learn_new", "needs_math": True,
        "needs_example": False,  # router's example-or-math branch
        "target_topic": "soft loss",
    }
    route = _route_after_explanation(state_overridden)
    check("S4", "math-no-example", route == "math_notation",
          f"when needs_example=False, got {route}")

    # Case B: refuse / navigate / review + needs_math=False may skip the math node
    state_non_math = {"intent": "navigate", "needs_math": False, "needs_example": False}
    route_ex = _route_after_explanation(state_non_math)
    check("S4", "skip-non-math", route_ex == "summary_transition",
          f"navigate+no-math got {route_ex}")
    state_after_ex_non_math = {**state_non_math}
    route = _route_after_example(state_after_ex_non_math)
    check("S4", "skip-after-example", route == "summary_transition",
          f"navigate+no-math after examples got {route}")

    # Case C: query_understanding's math-signal override test
    from src.agent.nodes.query_understanding import _MATH_SIGNAL_RE, _MATH_HINT_RE
    hits = ["softmax", "gradient", "损失向量矩阵", "co-occurrence", "backprop"]
    for kw in hits:
        check("S4", f"mathsig-{kw}",
              bool(_MATH_SIGNAL_RE.search(kw)) or bool(_MATH_HINT_RE.search(kw)),
              f"signal regex matches '{kw}'")
    non_hits = ["吃午饭", "holiday plan", "篮球队长"]
    for kw in non_hits:
        check("S4", f"mathsig-no-{kw[:8]}",
              not (_MATH_SIGNAL_RE.search(kw) or _MATH_HINT_RE.search(kw)),
              f"signal regex correctly skips '{kw}'")


# ---------------------------------------------------------------------------
# S5 — Retrieval: detect when the student named a file, filter by that
#       source, and apply source + section diversity so cross-file / cross-
#       section knowledge surfaces.
# ---------------------------------------------------------------------------
def stage_retrieval_filter_diversity() -> None:
    from src.indexing.vector_store import VectorStoreManager
    from src.indexing.hybrid_retriever import HybridRetriever
    from src.indexing.embeddings import get_embed_model
    from src.agent.nodes.document_retrieval import (
        _detect_referenced_source, _enforce_source_diversity,
        _enforce_section_diversity,
    )

    vsm = VectorStoreManager()
    sources = vsm.list_sources()
    if not sources:
        check("S5", "has-index", False, "no documents indexed — ingest a PDF first")
        return
    check("S5", "has-index", True, f"{len(sources)} sources: {sources}")

    # Build retriever (runs startup dedup automatically)
    embed = get_embed_model()
    retr = HybridRetriever(vsm, embed)

    # Fix2c: file-name detection from the user query
    chosen = sources[0]
    fake_query = f"下面和我讲讲 {chosen} 第2部分和第3部分"
    detected = _detect_referenced_source(fake_query, sources)
    check("S5", "file-detect", detected == chosen, f"detected {detected!r}")

    # Fix2c: retrieval restricted to that source returns only chunks from it
    if detected:
        results = retr.retrieve("Evaluation Training word vector", source_filter=detected)
        srcs_in = {r.node.metadata.get("source", "") for r in results}
        check("S5", "filter-only", srcs_in <= {detected},
              f"sources in filtered results: {srcs_in}")
    else:
        check("S5", "filter-only", False, "skipped (no detected source)")

    # Fix3: source diversity — cap per-source when not filtered
    pool = retr.retrieve("word vector evaluation training")
    if len(sources) > 1 and len(pool) >= 4:
        capped = _enforce_source_diversity(pool, max_per_source=2)
        from collections import Counter
        counts = Counter(r.node.metadata.get("source", "") for r in capped)
        check("S5", "source-diversity",
              max(counts.values()) <= 2,
              f"per-source counts after cap: {dict(counts)}")
    else:
        check("S5", "source-diversity", True,
              "single-source install — diversity cap trivially holds")

    # Fix2b: section diversity caps per top-level section when filtered
    if detected:
        from collections import Counter
        big_pool = retr.retrieve("Evaluation Training word vector",
                                 source_filter=detected, final_k=20)
        capped = _enforce_section_diversity(big_pool, max_per_top_section=2)
        sections = []
        for r in capped:
            sec_tag = r.node.metadata.get("section_heading", "")
            m = re.match(r"\s*(\d+)", sec_tag)
            sections.append(m.group(1) if m else "_")
        sc = Counter(sections)
        check("S5", "section-diversity",
              max(sc.values()) <= 2 or len(big_pool) < 4,
              f"per-section counts: {dict(sc)}")
    else:
        check("S5", "section-diversity", True, "skipped (no detected source)")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("  e2e_verify — five-stage end-to-end verification pipeline")
    print("=" * 70)
    try:
        stage_open_conversation()
    except Exception as e:
        global FAIL
        FAIL += 1
        FAILURES.append(f"S1 uncaught: {e}")
        print(f"[FAIL] S1 uncaught exception: {e}")
    try:
        stage_upload_dedup()
    except Exception as e:
        FAIL += 1
        FAILURES.append(f"S2 uncaught: {e}")
        print(f"[FAIL] S2 uncaught exception: {e}")
    try:
        stage_render_no_dup_header()
    except Exception as e:
        FAIL += 1
        FAILURES.append(f"S3 uncaught: {e}")
        print(f"[FAIL] S3 uncaught exception: {e}")
    try:
        stage_route_math_forced()
    except Exception as e:
        FAIL += 1
        FAILURES.append(f"S4 uncaught: {e}")
        print(f"[FAIL] S4 uncaught exception: {e}")
    try:
        stage_retrieval_filter_diversity()
    except Exception as e:
        FAIL += 1
        FAILURES.append(f"S5 uncaught: {e}")
        print(f"[FAIL] S5 uncaught exception: {e}")
    finally:
        # clean sandbox
        tmp = settings.data_dir / "_e2e"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 70)
    print(f"  RESULT: {PASS} passed, {FAIL} failed")
    if FAIL:
        print("  Failures:")
        for f in FAILURES:
            print(f"    - {f}")
    else:
        print("  All five stages PASSED — safe to commit.")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())