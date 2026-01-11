from typing import TypedDict, Dict, Any
from graph.state import HandlerOutput


def merge_handler_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pass-through node after conditional handler execution.
    Handlers now update answerable_processing.handler_outputs directly,
    so this node just ensures state consistency before compose_answer.
    """
    # Handlers already updated answerable_processing.handler_outputs directly
    # This is now a pass-through node for compatibility with the graph structure
    return {}
