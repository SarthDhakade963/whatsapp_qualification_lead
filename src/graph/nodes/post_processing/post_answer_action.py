from typing import TypedDict, Dict, Any
from graph.state import NextAction
from utils.state_adapter import get_state_value


def post_answer_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine next workflow action.
    Only modifies: next_action
    """
    # Default to END if we have a merged output
    merged_output = get_state_value(state, "merged_output")
    if merged_output:
        workflow = "END"
    else:
        workflow = "FOLLOW_UP"
    
    next_action = {
        "workflow": workflow
    }
    
    return {"next_action": next_action}
