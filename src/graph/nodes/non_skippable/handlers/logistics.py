from typing import TypedDict, Dict, Any
from graph.state import HandlerOutput
from domain.trips.loader import get_trip_data
from llm.client import LLMClient
from utils.behaviors import check_empathetic_response, check_seat_availability


def logistics_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle logistics questions using LLM-based fact extraction.
    Returns facts only.
    Updates answerable_processing.handler_outputs directly (for conditional/sequential execution)
    """
    answerable_processing = state.get("answerable_processing")
    if not answerable_processing or not answerable_processing.get("answer_plan"):
        return {}
    
    # Find logistics blocks
    answer_plan = answerable_processing.get("answer_plan", {})
    logistics_blocks = [
        block for block in answer_plan.get("answer_blocks", [])
        if block.get("handler") == "logistics_handler"
    ]
    
    if not logistics_blocks:
        return {}
    
    # Get trip data based on trip_context
    trip_context = answerable_processing.get("trip_context", {})
    trip_id = trip_context.get("trip_id", "") if isinstance(trip_context, dict) else getattr(trip_context, "trip_id", "")
    
    if not trip_id:
        trip_id = ""
    
    trip_data = get_trip_data(trip_id)
    if not trip_data or not isinstance(trip_data, dict):
        trip_data = {}
    
    # Initialize LLM client for fact extraction
    llm = LLMClient()
    
    # Process each logistics block
    new_handler_outputs = []
    structured_questions = answerable_processing.get("structured_questions", [])
    
    for block in logistics_blocks:
        # Get questions for this block
        question_ids = block.get("question_ids", [])
        questions = [
            q for q in structured_questions
            if isinstance(q, dict) and q.get("id") in question_ids
        ]
        
        # Extract facts using LLM
        facts = []
        for q in questions:
            question_text = q.get("text", "") if isinstance(q, dict) else getattr(q, "text", "")
            if question_text:
                # Check for seat availability first
                seat_availability_response = check_seat_availability(trip_data, question_text)
                if seat_availability_response:
                    facts.append(seat_availability_response)
                # Check for empathetic responses
                elif check_empathetic_response(question_text):
                    empathetic_response = check_empathetic_response(question_text)
                    facts.append(empathetic_response)
                else:
                    extracted_facts = llm.extract_facts(question_text, trip_data)
                    facts.extend(extracted_facts)
        
        # If no facts found, provide fallback message
        if not facts:
            if trip_data:
                facts = ["I'd be happy to share logistics details. Would you like to know about pickup points, meeting locations, or transportation arrangements?"]
            else:
                facts = ["I'd be happy to share that information. Could you clarify which trip you're asking about (e.g., Kashmir, Andaman)?"]
        
        # Create handler output (as dict)
        output = {
            "block_id": block.get("block_id"),
            "facts": facts,
            "requires_confirmation": False
        }
        
        new_handler_outputs.append(output)
    
    answerable_processing = answerable_processing.copy()
    existing_outputs = answerable_processing.get("handler_outputs", [])
    if not existing_outputs:
        existing_outputs = []
    existing_outputs.extend(new_handler_outputs)
    answerable_processing["handler_outputs"] = existing_outputs
    
    return {"answerable_processing": answerable_processing}
