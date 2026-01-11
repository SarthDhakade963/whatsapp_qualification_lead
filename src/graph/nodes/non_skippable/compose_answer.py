from typing import TypedDict, Dict, Any
from graph.state import ConversationWorkflowState
from llm.client import LLMClient


def compose_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compose final answer from handler outputs.
    Only modifies: answerable_processing.answer_text
    """
    answerable_processing = state.get("answerable_processing")
    if not answerable_processing:
        return {}
    
    handler_outputs_list = answerable_processing.get("handler_outputs", [])
    if not handler_outputs_list:
        return {}
    
    llm = LLMClient()
    
    # Handler outputs are already in dict format from merge_handler_outputs
    handler_outputs = handler_outputs_list
    
    # Compose answer
    normalized_text = answerable_processing.get("normalized_text", "")
    answer_text = llm.compose_answer(
        handler_outputs,
        normalized_text
    )
    
    # Update answerable_processing with composed answer
    if isinstance(answerable_processing, dict):
        answerable_processing = answerable_processing.copy()
        answerable_processing["answer_text"] = answer_text
    else:
        # Pydantic model - convert to dict
        answerable_processing_dict = answerable_processing.dict() if hasattr(answerable_processing, "dict") else dict(answerable_processing)
        answerable_processing_dict["answer_text"] = answer_text
        answerable_processing = answerable_processing_dict
    
    return {"answerable_processing": answerable_processing}
