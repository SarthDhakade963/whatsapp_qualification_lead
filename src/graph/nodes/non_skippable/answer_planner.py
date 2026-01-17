from typing import TypedDict, Dict, Any, List
from graph.state import AnswerPlan
from utils.state_adapter import get_state_value, to_dict
from utils.ids import generate_block_id


def answer_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create answer plan grouping questions by handler using deterministic logic.
    Groups questions by category and maps to handlers.
    Only modifies: answerable_processing.answer_plan
    
    Note: Uses deterministic grouping logic (no LLM call) for optimal latency.
    This matches the fallback logic previously used when LLM planning failed.
    """
    answerable_processing = get_state_value(state, "answerable_processing")
    if not answerable_processing:
        return {}
    
    answerable_dict = to_dict(answerable_processing)
    structured_questions = answerable_dict.get("structured_questions", [])
    
    if not structured_questions:
        return {}
    
    # DETERMINISTIC: Group questions by category
    category_groups: Dict[str, List[str]] = {}
    
    for q in structured_questions:
        cat = q.get("category", "LOGISTICS") if isinstance(q, dict) else getattr(q, "category", "LOGISTICS")
        q_id = q.get("id", "") if isinstance(q, dict) else getattr(q, "id", "")
        if cat not in category_groups:
            category_groups[cat] = []
        if q_id:
            category_groups[cat].append(q_id)
    
    # DETERMINISTIC: Map category to handler and create blocks
    blocks = []
    handler_map = {
        "LOGISTICS": "logistics_handler",
        "COST": "pricing_handler",
        "ITINERARY": "itinerary_handler",
        "POLICY": "pricing_handler"
    }
    
    for category, question_ids in category_groups.items():
        if not question_ids:
            continue
        
        blocks.append({
            "block_id": generate_block_id(),
            "question_ids": question_ids,
            "handler": handler_map.get(category, "logistics_handler"),
            "answer_style": "HIGH_LEVEL" if len(question_ids) == 1 else "DETAILED"
        })
    
    # Keep as dict for compatibility
    answer_plan = {
        "answer_blocks": blocks
    }
    
    # Update answerable_processing
    answerable_dict = answerable_dict.copy()
    answerable_dict["answer_plan"] = answer_plan
    
    return {"answerable_processing": answerable_dict}
