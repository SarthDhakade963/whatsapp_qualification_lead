from typing import TypedDict, Dict, Any
from graph.state import SkippableActions
from domain.policies import REFUND_POLICY
from utils.state_adapter import get_state_value, to_dict


def forbidden(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle forbidden questions (e.g., refunds, guarantees).
    Only modifies: skippable_actions.boundaries
    """
    questions = get_state_value(state, "questions", {})
    questions_dict = to_dict(questions)
    partitioned = questions_dict.get("partitioned", {})
    
    if not partitioned or not partitioned.get("skippable", {}).get("forbidden"):
        return {}
    
    # Get or create skippable_actions
    skippable_actions = get_state_value(state, "skippable_actions")
    if skippable_actions:
        skippable_dict = to_dict(skippable_actions).copy()
    else:
        skippable_dict = {
            "clarifications": [],
            "boundaries": [],
            "tone_safe_messages": []
        }
    
    # Add boundary message
    skippable_dict["boundaries"].append(
        REFUND_POLICY["boundary_message"]
    )
    
    return {"skippable_actions": skippable_dict}
