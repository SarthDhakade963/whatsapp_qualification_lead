from typing import TypedDict, Dict, Any
from graph.state import AnswerPlan
from llm.client import LLMClient
from utils.state_adapter import get_state_value, to_dict


def answer_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create answer plan grouping questions by handler.
    Only modifies: answerable_processing.answer_plan
    """
    answerable_processing = get_state_value(state, "answerable_processing")
    if not answerable_processing:
        return {}
    
    answerable_dict = to_dict(answerable_processing)
    structured_questions = answerable_dict.get("structured_questions", [])
    
    if not structured_questions:
        return {}
    
    llm = LLMClient()
    
    # Prepare structured questions for LLM (already in dict format)
    structured_qs = [
        {
            "id": q.get("id") if isinstance(q, dict) else getattr(q, "id", ""),
            "category": q.get("category") if isinstance(q, dict) else getattr(q, "category", ""),
            "text": q.get("text") if isinstance(q, dict) else getattr(q, "text", "")
        }
        for q in structured_questions
    ]
    
    trip_context = answerable_dict.get("trip_context", {})
    trip_ctx = {
        "trip_id": trip_context.get("trip_id") if isinstance(trip_context, dict) else getattr(trip_context, "trip_id", "spiti_7d"),
        "confidence": trip_context.get("confidence") if isinstance(trip_context, dict) else getattr(trip_context, "confidence", "MEDIUM")
    }
    
    # Get plan from LLM
    plan_dict = llm.plan_answer(structured_qs, trip_ctx)
    
    # Keep as dict for compatibility
    answer_plan = {
        "answer_blocks": plan_dict.get("answer_blocks", [])
    }
    
    # Update answerable_processing
    answerable_dict = answerable_dict.copy()
    answerable_dict["answer_plan"] = answer_plan
    
    return {"answerable_processing": answerable_dict}
