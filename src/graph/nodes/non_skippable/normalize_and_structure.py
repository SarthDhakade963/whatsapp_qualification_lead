from typing import TypedDict, Dict, Any
from graph.state import AnswerableProcessing, StructuredQuestion, TripContext
from llm.client import LLMClient
from utils.text import normalize_text
from utils.state_adapter import get_state_value, to_dict


def normalize_and_structure(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize text and structure answerable questions.
    Only modifies: answerable_processing.structured_questions, answerable_processing.normalized_text
    """
    questions = get_state_value(state, "questions", {})
    questions_dict = to_dict(questions)
    partitioned = questions_dict.get("partitioned", {})
    
    if not partitioned or not partitioned.get("non_skippable"):
        return {}
    
    llm = LLMClient()
    
    # Get answerable question IDs
    answerable_ids = partitioned.get("non_skippable", [])
    
    # Get corresponding atomic questions
    atomic_questions = questions_dict.get("atomic", [])
    atomic_map = {q.get("id") if isinstance(q, dict) else getattr(q, "id", ""): q for q in atomic_questions}
    
    # Normalize the full input text
    input_obj = get_state_value(state, "input", {})
    raw_text = input_obj.get("raw_text") if isinstance(input_obj, dict) else getattr(input_obj, "raw_text", "")
    normalized_text = normalize_text(raw_text)
    
    # Structure each answerable question
    structured = []
    for q_id in answerable_ids:
        atomic_q = atomic_map.get(q_id)
        if atomic_q:
            q_text = atomic_q.get("text") if isinstance(atomic_q, dict) else getattr(atomic_q, "text", "")
            category = llm.categorize_question(q_text)
            structured.append({
                "id": q_id,
                "category": category,
                "text": q_text
            })
    
    # Initialize answerable_processing if needed (as dict)
    answerable_processing = get_state_value(state, "answerable_processing")
    if not answerable_processing:
        answerable_processing = {
            "normalized_text": normalized_text,
            "trip_context": {
                "trip_id": "",
                "confidence": ""
            },
            "answer_plan": {
                "answer_blocks": []
            },
            "structured_questions": structured,
            "handler_outputs": []
        }
    else:
        answerable_processing = to_dict(answerable_processing).copy()
        answerable_processing["structured_questions"] = structured
        answerable_processing["normalized_text"] = normalized_text
    
    return {"answerable_processing": answerable_processing}
