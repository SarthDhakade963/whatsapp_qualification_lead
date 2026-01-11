from typing import TypedDict, Dict, Any
from graph.state import ConversationWorkflowState, Questions, AtomicQuestion
from utils.text import normalize_text, split_into_questions
from utils.ids import generate_question_id
from utils.state_adapter import get_state_value, to_dict


def normalize_and_split(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize input text and split into atomic questions.
    Only modifies: questions.atomic
    """
    # Handle both dict and Pydantic
    input_obj = get_state_value(state, "input", {})
    raw_text = input_obj.get("raw_text") if isinstance(input_obj, dict) else getattr(input_obj, "raw_text", "")
    
    normalized = normalize_text(raw_text)
    
    # Split into atomic questions
    question_texts = split_into_questions(normalized)
    
    # Create atomic questions (as dicts)
    atomic_questions = [
        {
            "id": generate_question_id(),
            "text": q_text
        }
        for q_text in question_texts
    ]
    
    # Update state
    questions = get_state_value(state, "questions", {})
    questions_dict = to_dict(questions)
    questions_dict["atomic"] = atomic_questions
    
    return {"questions": questions_dict}

