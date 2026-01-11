from typing import TypedDict, Dict, Any
from graph.state import ConversationWorkflowState, Questions, ClassifiedQuestion
from llm.client import LLMClient
from utils.state_adapter import get_state_value, to_dict


def classify_each_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify each atomic question using LLM.
    Only modifies: questions.classified
    """
    llm = LLMClient()
    
    questions = get_state_value(state, "questions", {})
    questions_dict = to_dict(questions)
    atomic_questions = questions_dict.get("atomic", [])
    
    classified = []
    for atomic_q in atomic_questions:
        question_id = atomic_q.get("id") if isinstance(atomic_q, dict) else getattr(atomic_q, "id", "")
        question_text = atomic_q.get("text") if isinstance(atomic_q, dict) else getattr(atomic_q, "text", "")
        classification = llm.classify_question(question_text)
        classified.append({
            "id": question_id,
            "class": classification  # Use "class" not "class_" for alias compatibility
        })
    
    questions_dict["classified"] = classified
    
    return {"questions": questions_dict}

