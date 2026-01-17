from typing import TypedDict, Dict, Any
from graph.state import ConversationWorkflowState, Questions, ClassifiedQuestion
from llm.client import LLMClient
from utils.state_adapter import get_state_value, to_dict


def classify_each_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify each atomic question using LLM batch processing.
    Only modifies: questions.classified
    """
    llm = LLMClient()
    
    questions = get_state_value(state, "questions", {})
    questions_dict = to_dict(questions)
    atomic_questions = questions_dict.get("atomic", [])
    
    if not atomic_questions:
        questions_dict["classified"] = []
        return {"questions": questions_dict}
    
    # Extract question texts and IDs for batch processing
    question_texts = []
    question_ids = []
    for atomic_q in atomic_questions:
        question_id = atomic_q.get("id") if isinstance(atomic_q, dict) else getattr(atomic_q, "id", "")
        question_text = atomic_q.get("text") if isinstance(atomic_q, dict) else getattr(atomic_q, "text", "")
        if question_text:
            question_texts.append(question_text)
            question_ids.append(question_id)
    
    # BATCH: Classify all questions in a single LLM call
    if question_texts:
        batch_results = llm.classify_questions_batch(question_texts)
    else:
        batch_results = {}
    
    # Map results back to question IDs in order
    classified = []
    for i, question_id in enumerate(question_ids):
        question_text = question_texts[i] if i < len(question_texts) else ""
        classification = batch_results.get(question_text, "ANSWERABLE")  # Fallback to ANSWERABLE if not found
        classified.append({
            "id": question_id,
            "class": classification  # Use "class" not "class_" for alias compatibility
        })
    
    questions_dict["classified"] = classified
    
    return {"questions": questions_dict}
