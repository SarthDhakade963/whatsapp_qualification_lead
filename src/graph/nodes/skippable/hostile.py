from typing import TypedDict, Dict, Any
from graph.state import SkippableActions
from utils.state_adapter import get_state_value, to_dict


def hostile(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle hostile questions.
    Only modifies: skippable_actions.tone_safe_messages
    """
    questions = get_state_value(state, "questions", {})
    questions_dict = to_dict(questions)
    partitioned = questions_dict.get("partitioned", {})
    
    if not partitioned or not partitioned.get("skippable", {}).get("hostile"):
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
    
    # Add tone-safe message
    skippable_dict["tone_safe_messages"].append(
        "I'm here to help. Let's focus on how I can assist you with your travel plans."
    )
    
    return {"skippable_actions": skippable_dict}
