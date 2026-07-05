"""LangGraph Teaching Agent — graph construction and compilation.

Builds the teaching workflow graph:
  query_understanding → document_retrieval → explanation_generation
  → example_generation → math_notation → summary_transition → END

Conditional routing:
  - Greetings/navigation queries skip retrieval and go directly to END
  - Empty retrieval results skip explanation generation
  - Topics without math skip math_notation node
"""

import logging
from typing import Literal, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import TeachingState
from .nodes import (
    query_understanding_node,
    document_retrieval_node,
    explanation_generation_node,
    example_generation_node,
    math_notation_node,
    summary_transition_node,
)

logger = logging.getLogger(__name__)


def _route_after_query_understanding(state: TeachingState) -> Literal["document_retrieval", "__end__"]:
    """Route based on intent.

    - "navigate" (greetings, non-learning queries) → END
    - "refuse" (lab/homework/project requests) → END
    - All other intents → proceed to retrieval
    """
    intent = state.get("intent", "learn_new")
    if intent in ("navigate", "refuse"):
        logger.info("[Router] Intent '%s' → END (no retrieval needed)", intent)
        return "__end__"
    logger.info("[Router] Learning intent '%s' → document_retrieval", intent)
    return "document_retrieval"


def _route_after_retrieval(state: TeachingState) -> Literal["explanation_generation"]:
    """Route after retrieval — always proceed to explanation generation.

    When chunks are found: explanation uses retrieved context.
    When no chunks: explanation falls back to DIRECT_RESPONSE_PROMPT
    (LLM answers from general knowledge while maintaining TA persona).
    """
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        logger.info("[Router] No chunks retrieved — will use direct LLM response")
    else:
        logger.info("[Router] %d chunks retrieved → explanation with context", len(chunks))
    return "explanation_generation"


def _route_after_explanation(state: TeachingState) -> Literal["example_generation", "math_notation", "summary_transition"]:
    """Route based on what the topic needs.

    - needs_example → example_generation
    - needs_math (but no example needed) → math_notation
    - neither → straight to summary

    Defense: for ``learn_new`` intent we always advance to either example or
    math, never jump straight to summary — that is the four-part-flow contract.
    A spurious ``needs_example=False`` AND ``needs_math=False`` from the LLM
    is overridden to ``needs_math=True`` here so learning requests never skip
    all middle stages.
    """
    needs_example = state.get("needs_example", True)
    needs_math = state.get("needs_math", True)
    intent = state.get("intent", "learn_new")

    if intent == "learn_new" and not needs_example and not needs_math:
        logger.info("[Router] learn_new with no example/math flags → force math_notation")
        needs_math = True

    if needs_example:
        logger.info("[Router] → example_generation")
        return "example_generation"
    elif needs_math:
        logger.info("[Router] → math_notation (skip examples)")
        return "math_notation"
    else:
        logger.info("[Router] → summary_transition (intent=%s, skip examples & math)", intent)
        return "summary_transition"


def _route_after_example(state: TeachingState) -> Literal["math_notation", "summary_transition"]:
    """After examples, go to math notation unless the query is clearly non-mathy.

    Defaults to ``math_notation``: the four-part teaching flow expects a math
    step, and `query_understanding_node` already override-sets ``needs_math``
    True when math signal words appear. We only skip the math node when the
    intent is clearly non-learning (refuse / navigate / review of pure recall)
    AND ``needs_math`` is explicitly False — this catches "Summarize what we
    covered" / "Hello" while never skipping a learn_new request even if the
    LLM happened to leave needs_math=False (the upstream guard already flips
    it, but this defense-in-depth keeps a redundant safety net).
    """
    needs_math = state.get("needs_math", True)
    intent = state.get("intent", "learn_new")
    if not needs_math and intent in ("refuse", "navigate", "review"):
        logger.info("[Router] → summary_transition (intent=%s, needs_math=False)", intent)
        return "summary_transition"
    logger.info("[Router] → math_notation (intent=%s, needs_math=%s)", intent, needs_math)
    return "math_notation"


def build_graph() -> StateGraph:
    """Build and compile the teaching agent LangGraph.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    logger.info("Building teaching agent graph...")

    # Create the graph with TeachingState
    graph_builder = StateGraph(TeachingState)

    # --- Add nodes ---
    graph_builder.add_node("query_understanding", query_understanding_node)
    graph_builder.add_node("document_retrieval", document_retrieval_node)
    graph_builder.add_node("explanation_generation", explanation_generation_node)
    graph_builder.add_node("example_generation", example_generation_node)
    graph_builder.add_node("math_notation", math_notation_node)
    graph_builder.add_node("summary_transition", summary_transition_node)

    # --- Add edges ---
    # Entry point
    graph_builder.add_edge(START, "query_understanding")

    # Conditional: query understanding → retrieval or END
    graph_builder.add_conditional_edges(
        "query_understanding",
        _route_after_query_understanding,
        {
            "document_retrieval": "document_retrieval",
            "__end__": END,
        },
    )

    # Conditional: retrieval → explanation (always — direct response fallback if no chunks)
    graph_builder.add_conditional_edges(
        "document_retrieval",
        _route_after_retrieval,
        {
            "explanation_generation": "explanation_generation",
        },
    )

    # Conditional: explanation → example, math, or summary
    graph_builder.add_conditional_edges(
        "explanation_generation",
        _route_after_explanation,
        {
            "example_generation": "example_generation",
            "math_notation": "math_notation",
            "summary_transition": "summary_transition",
        },
    )

    # Conditional: example → math or summary
    graph_builder.add_conditional_edges(
        "example_generation",
        _route_after_example,
        {
            "math_notation": "math_notation",
            "summary_transition": "summary_transition",
        },
    )

    # Math → summary (always)
    graph_builder.add_edge("math_notation", "summary_transition")

    # Summary → END (always)
    graph_builder.add_edge("summary_transition", END)

    # Compile with memory for conversation persistence
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    logger.info("Teaching agent graph compiled successfully")
    return graph


class TeachingAgent:
    """Teaching agent wrapper around the compiled LangGraph.

    Usage:
        agent = TeachingAgent()
        result = await agent.ateach("What is gradient descent?")
        # Process streaming events from result
    """

    def __init__(self):
        self._graph = build_graph()

    @property
    def graph(self):
        return self._graph

    async def ateach(self, user_query: str, thread_id: str = "default") -> dict[str, Any]:
        """Run the teaching pipeline for a student query.

        Args:
            user_query: The student's question or topic request.
            thread_id: Conversation thread ID for session persistence.

        Returns:
            Final TeachingState with all four parts populated.
        """
        initial_state: TeachingState = {
            "user_query": user_query,
            "iteration_count": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}

        result = await self._graph.ainvoke(initial_state, config)
        return result

    async def astream_teach(self, user_query: str, thread_id: str = "default"):
        """Stream the teaching pipeline, yielding events as nodes execute.

        Yields events of type:
            - "on_chain_start" / "on_chain_end": node boundaries
            - "on_chat_model_stream": LLM token streaming (for UI)
        """
        initial_state: TeachingState = {
            "user_query": user_query,
            "iteration_count": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}

        async for event in self._graph.astream_events(initial_state, config, version="v2"):
            yield event
