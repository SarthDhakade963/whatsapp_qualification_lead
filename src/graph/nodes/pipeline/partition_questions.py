from typing import TypedDict, Dict, Any
from graph.state import ConversationWorkflowState, Questions, PartitionedQuestions
from utils.state_adapter import get_state_value, to_dict


def partition_questions(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Partition classified questions into non_skippable and skippable buckets.
    Only modifies: questions.partitioned
    """
    non_skippable = []
    skippable = {
        "malformed": [],
        "forbidden": [],
        "hostile": []
    }
    
    questions = get_state_value(state, "questions", {})
    questions_dict = to_dict(questions)
    classified_questions = questions_dict.get("classified", [])
    
    for classified_q in classified_questions:
        # Handle both dict and Pydantic
        q_class = classified_q.get("class") or classified_q.get("class_") if isinstance(classified_q, dict) else getattr(classified_q, "class_", None) or getattr(classified_q, "class", None)
        q_id = classified_q.get("id") if isinstance(classified_q, dict) else getattr(classified_q, "id", "")
        
        if q_class == "ANSWERABLE":
            non_skippable.append(q_id)
        elif q_class == "MALFORMED":
            skippable["malformed"].append(q_id)
        elif q_class == "FORBIDDEN":
            skippable["forbidden"].append(q_id)
        elif q_class == "HOSTILE":
            skippable["hostile"].append(q_id)
    
    questions_dict["partitioned"] = {
        "non_skippable": non_skippable,
        "skippable": skippable
    }
    
    return {"questions": questions_dict}

