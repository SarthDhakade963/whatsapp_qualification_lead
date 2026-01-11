from typing import TypedDict, Dict, Any
from graph.state import SkippableActions
from utils.state_adapter import get_state_value, to_dict


def malformed(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle malformed questions.
    Only modifies: skippable_actions.clarifications
    """
    questions = get_state_value(state, "questions", {})
    questions_dict = to_dict(questions)
    partitioned = questions_dict.get("partitioned", {})
    
    if not partitioned or not partitioned.get("skippable", {}).get("malformed"):
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
    
    # Add clarification request
    skippable_dict["clarifications"].append(
        "Could you please rephrase your question? I want to make sure I understand correctly."
    )
    
    return {"skippable_actions": skippable_dict}
