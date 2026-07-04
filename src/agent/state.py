"""LangGraph TeachingState definition.

The state flows through the teaching graph:
  query_understanding → document_retrieval → explanation_generation
  → example_generation → math_notation → summary_transition → END
"""

from typing import TypedDict, List, Optional, Literal, Annotated
from langgraph.graph.message import add_messages


class RetrievedContext(TypedDict):
    """A single retrieved document chunk with metadata."""
    content: str
    source: str        # e.g., "cs229_lecture1.pdf"
    page: int
    relevance_score: float


class TeachingState(TypedDict, total=False):
    """Complete state for the teaching agent graph.

    Fields marked with Optional[...] may be absent in intermediate states.
    """

    # --- Conversation History ---
    messages: Annotated[list, add_messages]

    # --- User Query Analysis ---
    user_query: str
    intent: str                         # "learn_new" | "clarify" | "review" | "navigate" | "refuse"
    target_topic: str                   # e.g., "gradient descent", "Fourier transform"
    difficulty: str                     # "beginner" | "intermediate" | "advanced"

    # --- Retrieval Results ---
    retrieved_chunks: List[RetrievedContext]
    retrieval_query: str                # The rewritten query used for retrieval

    # --- Teaching Outputs (four-part structure) ---
    core_explanation: str               # Part ①: Core Process
    examples: str                       # Part ②: Interspersed Examples
    math_notation: str                  # Part ③: Math & Notation
    section_summary: str                # Part ④: Summary

    # --- Flow Control ---
    current_node: str                   # Which node is currently executing
    teaching_phase: str                 # "core" | "example" | "math" | "summary"
    needs_example: bool                 # Whether to generate detailed examples
    needs_math: bool                    # Whether to do math notation derivation

    # --- Session Tracking ---
    iteration_count: int
    user_feedback: str                  # Optional feedback from user for iterative improvement
