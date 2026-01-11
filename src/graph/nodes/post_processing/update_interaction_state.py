from typing import TypedDict, Dict, Any
from graph.state import InteractionState
from utils.state_adapter import get_state_value, to_dict


def update_interaction_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update interaction state based on processing results.
    Only modifies: interaction_state
    """
    # Check if call escalation was already set in merge_outputs
    existing_interaction_state = state.get("interaction_state")
    if existing_interaction_state and existing_interaction_state.get("escalation_flag"):
        return {"interaction_state": existing_interaction_state}
    
    # Determine decision stage
    merged_output = get_state_value(state, "merged_output")
    if merged_output:
        decision_stage = "ANSWERED"
    else:
        decision_stage = "EVALUATING"
    
    # Check for escalation (e.g., if skippable actions have boundaries)
    escalation_flag = False
    skippable_actions = get_state_value(state, "skippable_actions")
    if skippable_actions:
        skippable_dict = to_dict(skippable_actions)
        boundaries = skippable_dict.get("boundaries", [])
        if boundaries:
            escalation_flag = True
    
    interaction_state = {
        "decision_stage": decision_stage,
        "escalation_flag": escalation_flag
    }
    
    return {"interaction_state": interaction_state}
