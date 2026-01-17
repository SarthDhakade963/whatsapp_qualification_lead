from langgraph.graph import StateGraph, END
from typing import Literal, Dict, Any
from graph.state import ConversationWorkflowState

# Entry
from graph.nodes.entry.inbound_message import inbound_message

# Pipeline
from graph.nodes.pipeline.normalize_and_split import normalize_and_split
from graph.nodes.pipeline.classify_each_question import classify_each_question
from graph.nodes.pipeline.partition_questions import partition_questions
from graph.nodes.pipeline.merge_outputs import merge_outputs

# Non-skippable
from graph.nodes.non_skippable.normalize_and_structure import normalize_and_structure
from graph.nodes.non_skippable.resolve_trip_context import resolve_trip_context
from graph.nodes.non_skippable.answer_planner import answer_planner
from graph.nodes.non_skippable.merge_handler_outputs import merge_handler_outputs
from graph.nodes.non_skippable.compose_answer import compose_answer
from graph.nodes.non_skippable.handlers.logistics import logistics_handler
from graph.nodes.non_skippable.handlers.pricing import pricing_handler
from graph.nodes.non_skippable.handlers.itinerary import itinerary_handler

# Skippable
from graph.nodes.skippable.malformed import malformed
from graph.nodes.skippable.forbidden import forbidden
from graph.nodes.skippable.hostile import hostile

# Post-processing
from graph.nodes.post_processing.update_interaction_state import update_interaction_state
from graph.nodes.post_processing.post_answer_action import post_answer_action


def noop_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """No-op node for routing."""
    return state if isinstance(state, dict) else {}


def build_graph() -> StateGraph:
    """Build the LangGraph workflow with conditional handler routing and LangSmith tracing."""
    
    # Configure LangSmith tracing
    import os
    from app.settings import Settings
    settings = Settings()
    
    if settings.langsmith_tracing:
        langsmith_key = settings.effective_langsmith_api_key()
        if langsmith_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key
            os.environ["LANGCHAIN_PROJECT"] = settings.effective_langsmith_project()
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        elif os.getenv("LANGCHAIN_API_KEY"):
            # Use existing environment variable if set
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = settings.effective_langsmith_project()
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
    workflow = StateGraph(ConversationWorkflowState)
    
    # Entry
    workflow.add_node("inbound_message", inbound_message)
    
    # Pipeline
    workflow.add_node("normalize_and_split", normalize_and_split)
    workflow.add_node("classify_each_question", classify_each_question)
    workflow.add_node("partition_questions", partition_questions)
    
    # Non-skippable branch
    workflow.add_node("normalize_and_structure", normalize_and_structure)
    workflow.add_node("resolve_trip_context", resolve_trip_context)
    workflow.add_node("answer_planner", answer_planner)
    workflow.add_node("handlers_start", noop_node)  # PARALLEL: Fan-out point for handlers
    workflow.add_node("logistics_handler", logistics_handler)
    workflow.add_node("pricing_handler", pricing_handler)
    workflow.add_node("itinerary_handler", itinerary_handler)
    workflow.add_node("merge_handler_outputs", merge_handler_outputs)
    workflow.add_node("compose_answer", compose_answer)
    
    # Skippable branch
    workflow.add_node("skippable_start", noop_node)
    workflow.add_node("malformed", malformed)
    workflow.add_node("forbidden", forbidden)
    workflow.add_node("hostile", hostile)
    
    # Convergence
    workflow.add_node("converge", noop_node)
    
    # Post-processing
    workflow.add_node("merge_outputs", merge_outputs)
    workflow.add_node("update_interaction_state", update_interaction_state)
    workflow.add_node("post_answer_action", post_answer_action)
    
    # Define edges
    workflow.set_entry_point("normalize_and_split")
    
    # Pipeline flow (skip inbound_message since input is already set)
    workflow.add_edge("normalize_and_split", "classify_each_question")
    workflow.add_edge("classify_each_question", "partition_questions")
    
    # After partition, route to non-skippable if needed, otherwise to skippable
    def route_after_partition(state: Dict[str, Any]) -> str:
        questions = state.get("questions", {})
        partitioned = questions.get("partitioned", {}) if isinstance(questions, dict) else getattr(questions, "partitioned", {})
        if isinstance(partitioned, dict):
            non_skippable = partitioned.get("non_skippable", [])
            return "normalize_and_structure" if non_skippable else "skippable_start"
        else:
            non_skippable = getattr(partitioned, "non_skippable", [])
            return "normalize_and_structure" if non_skippable else "skippable_start"
    
    workflow.add_conditional_edges(
        "partition_questions",
        route_after_partition,
        {
            "normalize_and_structure": "normalize_and_structure",
            "skippable_start": "skippable_start"
        }
    )
    
    # Non-skippable branch (sequential until handlers)
    workflow.add_edge("normalize_and_structure", "resolve_trip_context")
    workflow.add_edge("resolve_trip_context", "answer_planner")
    
    # After answer_planner, route to handlers_start if handlers needed, otherwise to compose
    def route_to_handlers(state: Dict[str, Any]) -> str:
        """Route to handlers_start if handlers are needed, or compose_answer if none needed."""
        answerable_processing = state.get("answerable_processing")
        if not answerable_processing:
            return "compose_answer"
        
        answer_plan = answerable_processing.get("answer_plan", {})
        answer_blocks = answer_plan.get("answer_blocks", [])
        
        if not answer_blocks:
            return "compose_answer"
        
        # Determine if any handlers are needed
        handlers_needed = set()
        for block in answer_blocks:
            handler = block.get("handler", "")
            if handler in ["logistics_handler", "pricing_handler", "itinerary_handler"]:
                handlers_needed.add(handler)
        
        if not handlers_needed:
            return "compose_answer"
        
        # PARALLEL: Route to handlers_start to fan-out to all handlers
        return "handlers_start"
    
    workflow.add_conditional_edges(
        "answer_planner",
        route_to_handlers,
        {
            "handlers_start": "handlers_start",
            "compose_answer": "compose_answer"
        }
    )
    
    # APPROACH 1: PARALLEL EXECUTION WITH EARLY EXIT
    # =============================================
    # Fan-out: All handlers execute in parallel from handlers_start
    # - Even handlers without work run (for true parallelism)
    # - Each handler checks for blocks and exits early if none exist (microseconds overhead)
    # - This approach prioritizes simplicity and true parallelism over avoiding function calls
    # - Overhead is negligible (~0.0003s) compared to LLM latency (~1000ms)
    workflow.add_edge("handlers_start", "logistics_handler")
    workflow.add_edge("handlers_start", "pricing_handler")
    workflow.add_edge("handlers_start", "itinerary_handler")
    
    # Fan-in (Barrier): Each handler routes directly to merge_handler_outputs
    # - merge_handler_outputs waits for ALL handlers to complete (barrier pattern)
    # - Handlers with work process and update state
    # - Handlers without work return {} immediately (early exit)
    # - LangGraph automatically merges state updates from parallel handlers
    workflow.add_edge("logistics_handler", "merge_handler_outputs")
    workflow.add_edge("pricing_handler", "merge_handler_outputs")
    workflow.add_edge("itinerary_handler", "merge_handler_outputs")
    
    # merge_handler_outputs is a pass-through (handlers update state directly)
    # Routes to compose_answer after all handlers complete
    workflow.add_edge("merge_handler_outputs", "compose_answer")
    
    # After compose_answer, go to skippable branch
    workflow.add_edge("compose_answer", "skippable_start")
    
    # Skippable branch - route to appropriate nodes
    def route_skippable(state: Dict[str, Any]) -> str:
        questions = state.get("questions", {})
        partitioned = questions.get("partitioned", {}) if isinstance(questions, dict) else getattr(questions, "partitioned", {})
        skippable = partitioned.get("skippable", {}) if isinstance(partitioned, dict) else getattr(partitioned, "skippable", {})
        if isinstance(skippable, dict):
            if skippable.get("malformed"):
                return "malformed"
            elif skippable.get("forbidden"):
                return "forbidden"
            elif skippable.get("hostile"):
                return "hostile"
        else:
            if getattr(skippable, "malformed", None):
                return "malformed"
            elif getattr(skippable, "forbidden", None):
                return "forbidden"
            elif getattr(skippable, "hostile", None):
                return "hostile"
        return "converge"
    
    workflow.add_conditional_edges(
        "skippable_start",
        route_skippable,
        {
            "malformed": "malformed",
            "forbidden": "forbidden",
            "hostile": "hostile",
            "converge": "converge"
        }
    )
    
    # All skippable nodes converge
    workflow.add_edge("malformed", "converge")
    workflow.add_edge("forbidden", "converge")
    workflow.add_edge("hostile", "converge")
    
    # Convergence point
    workflow.add_edge("converge", "merge_outputs")
    
    # Post-processing
    workflow.add_edge("merge_outputs", "update_interaction_state")
    workflow.add_edge("update_interaction_state", "post_answer_action")
    workflow.add_edge("post_answer_action", END)
    
    return workflow.compile()
